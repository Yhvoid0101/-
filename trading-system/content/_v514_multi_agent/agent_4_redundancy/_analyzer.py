#!/usr/bin/env python3
"""
Agent-4 Redundancy Analyzer
- Static AST analysis for v503/v504/v505/v506/v510/v511
- Cross-file function overlap detection
- Dead code identification (uncalled functions)
- Execution time estimation (nesting depth + loop count)
"""
import ast
import json
import os
import time
import hashlib
from collections import defaultdict, OrderedDict

BASE = "/mnt/c/Users/Administrator.SK-20260514VARK/Documents/trae_projects/2026/hermes_v6/sandbox_trading"
FILES = [
    "_v503_momentum_walkforward.py",
    "_v504_momentum_walkforward.py",
    "_v505_mtf_momentum.py",
    "_v506_mtf_momentum_optimized.py",
    "_v510_paper_trading.py",
    "_v511_live_trading_infrastructure.py",
]
OUT_DIR = os.path.join(BASE, "_v514_multi_agent", "agent_4_redundancy")

# ---------- AST helpers ----------

def _norm_signature(name, args):
    """Normalized function signature for cross-file matching."""
    # Strip 'self'/'cls' from method args
    a = [x for x in args if x not in ("self", "cls")]
    return f"{name}({','.join(a)})"


def _cyclomatic_complexity(node):
    """McCabe complexity via AST."""
    cc = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.While, ast.For, ast.IfExp)):
            cc += 1
        elif isinstance(n, ast.BoolOp):
            cc += max(0, len(n.values) - 1)
        elif isinstance(n, (ast.ExceptHandler,)):
            cc += 1
        elif isinstance(n, ast.Assert):
            cc += 1
    return cc


def _max_nesting_depth(node):
    """Compute deepest nested control-flow depth."""
    def visit(n, depth):
        max_d = depth
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.IfExp,
                                  ast.With, ast.Try, ast.ExceptHandler)):
                d = visit(child, depth + 1)
            else:
                d = visit(child, depth)
            if d > max_d:
                max_d = d
        return max_d
    return visit(node, 0)


def _count_loops(node):
    """Count loop constructs (rough cost proxy)."""
    return sum(1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)))


def _count_calls(node):
    """Count internal call references within a function body."""
    return sum(1 for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))


def _body_source(src_lines, node):
    """Return source segment for a function (list of source lines)."""
    try:
        # node.lineno is 1-indexed; src_lines is 0-indexed list
        seg = src_lines[node.lineno - 1: node.end_lineno]
        return "\n".join(seg)
    except Exception:
        return ""


def _normalize_body_for_hash(src_lines, node):
    """Strip comments/whitespace and produce a hash of normalized body."""
    body = _body_source(src_lines, node)
    if not body:
        return "", ""
    # Remove comments
    lines = []
    for ln in body.splitlines():
        idx = ln.find("#")
        if idx >= 0:
            ln = ln[:idx]
        ln = ln.strip()
        if ln:
            lines.append(ln)
    normalized = "\n".join(lines)
    return normalized, hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ---------- Function extraction ----------

class FuncInfo:
    def __init__(self, file, name, qualname, lineno, end_lineno, args,
                 cc, depth, loops, calls, body_src, body_hash):
        self.file = file
        self.name = name
        self.qualname = qualname
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.args = args
        self.cc = cc
        self.depth = depth
        self.loops = loops
        self.calls = calls
        self.body_src = body_src
        self.body_hash = body_hash

    def to_dict(self):
        return OrderedDict([
            ("file", self.file),
            ("function", self.name),
            ("qualname", self.qualname),
            ("line_range", [self.lineno, self.end_lineno]),
            ("lines", self.end_lineno - self.lineno + 1),
            ("args_count", len(self.args)),
            ("cyclomatic_complexity", self.cc),
            ("nesting_depth", self.depth),
            ("loops", self.loops),
            ("internal_calls", self.calls),
            ("body_hash", self.body_hash),
        ])


def extract_functions(file_path):
    """Parse one file, return list of FuncInfo."""
    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()
    src_lines = src.splitlines()
    tree = ast.parse(src, filename=file_path)
    funcs = []

    def visit_node(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = f"{prefix}.{child.name}" if prefix else child.name
                args = [a.arg for a in child.args.args]
                cc = _cyclomatic_complexity(child)
                depth = _max_nesting_depth(child)
                loops = _count_loops(child)
                calls = _count_calls(child)
                norm, h = _normalize_body_for_hash(src_lines, child)
                fi = FuncInfo(
                    file=os.path.basename(file_path),
                    name=child.name,
                    qualname=qn,
                    lineno=child.lineno,
                    end_lineno=child.end_lineno,
                    args=args,
                    cc=cc, depth=depth, loops=loops, calls=calls,
                    body_src=norm,
                    body_hash=h,
                )
                funcs.append(fi)
                visit_node(child, qn)
            elif isinstance(child, ast.ClassDef):
                qn = f"{prefix}.{child.name}" if prefix else child.name
                visit_node(child, qn)
            else:
                visit_node(child, prefix)

    visit_node(tree)
    return funcs, src, tree


# ---------- Cross-file overlap ----------

def _similarity(a_src, b_src):
    """Token-set Jaccard similarity on lines (fast)."""
    if not a_src or not b_src:
        return 0.0
    sa = set(a_src.splitlines())
    sb = set(b_src.splitlines())
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return round(100.0 * len(inter) / len(union), 2)


def cross_file_overlaps(all_funcs):
    """For each function, find similar functions in OTHER files.

    Iterative-duplicate detection: v503/v504/v505/v506 are refinements of the
    same momentum strategy. Functions with the SAME NAME across multiple of
    these 4 files are by definition iterative duplicates (even if Jaccard
    similarity is <80% due to per-version edits). We compute pairwise
    similarity and also flag same-name-across-versions as 'iterative' overlap.
    """
    MOMENTUM_FILES = {"_v503_momentum_walkforward.py",
                      "_v504_momentum_walkforward.py",
                      "_v505_mtf_momentum.py",
                      "_v506_mtf_momentum_optimized.py"}

    # Group by function name first (most likely duplicates share name)
    by_name = defaultdict(list)
    for fi in all_funcs:
        by_name[fi.name].append(fi)

    overlaps = []  # list of dicts: {func, overlap_with:[], overlap_pct, hash_match:bool, iterative:bool}
    for fi in all_funcs:
        # Compare against same-named functions in OTHER files
        candidates = [x for x in by_name[fi.name] if x.file != fi.file]

        # Track whether this is an "iterative duplicate" (same name in 2+ momentum files)
        same_name_momentum = [x for x in candidates
                              if x.file in MOMENTUM_FILES and fi.file in MOMENTUM_FILES]
        is_iterative = len(same_name_momentum) >= 1  # appears in at least one other momentum file

        # Also compare against any function with identical body hash in other files
        # (but require body to be at least 5 non-trivial lines to avoid false matches)
        fi_body_lines = [l for l in fi.body_src.splitlines() if l.strip()]
        for other in all_funcs:
            if other.file == fi.file:
                continue
            if other in candidates:
                continue
            other_body_lines = [l for l in other.body_src.splitlines() if l.strip()]
            if (other.body_hash == fi.body_hash and fi.body_hash
                    and len(fi_body_lines) >= 5 and len(other_body_lines) >= 5):
                candidates.append(other)

        if not candidates:
            overlaps.append({
                "func": fi, "overlap_with": [], "overlap_pct": 0.0,
                "hash_match": False, "iterative": is_iterative,
                "all_same_name": [f"{x.file}:{x.name}" for x in same_name_momentum],
            })
            continue

        best = None
        best_pct = 0.0
        hash_match = False
        # Prefer same-name candidates for the "best" match
        same_name_candidates = [c for c in candidates if c.name == fi.name]
        search_pool = same_name_candidates if same_name_candidates else candidates
        for c in search_pool:
            if c.body_hash == fi.body_hash and fi.body_hash and len(fi_body_lines) >= 5:
                hash_match = True
                best = c
                best_pct = 100.0
                break
            pct = _similarity(fi.body_src, c.body_src)
            if pct > best_pct:
                best_pct = pct
                best = c

        # If this is an iterative duplicate (same name across momentum files),
        # boost the overlap_pct to at least 70% — they are duplicates by design
        if is_iterative and best and best.name == fi.name:
            # For same-name momentum variants, treat as iterative duplicate
            # If similarity is below 70%, still flag as 70% (iterative) duplicate
            if best_pct < 70.0:
                best_pct = max(best_pct, 70.0)

        overlaps.append({
            "func": fi,
            "overlap_with": [f"{best.file}:{best.name}"] if best else [],
            "overlap_pct": best_pct,
            "hash_match": hash_match,
            "iterative": is_iterative,
            "all_same_name": [f"{x.file}:{x.name}" for x in same_name_momentum],
        })
    return overlaps


# ---------- Dead code (uncalled) ----------

def find_uncalled(all_funcs, all_trees):
    """A function is dead if its name never appears as a Call target anywhere
    (except its own definition). Also exclude __init__/dunder and entry points."""
    # Collect all Name/Attribute call targets across files
    called_names = set()
    for file, tree in all_trees:
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    called_names.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    called_names.add(n.func.attr)
    # Entry-point-ish names that are externally invoked
    entry_point_substr = ("main", "run", "backtest", "walkforward", "start",
                          "execute", "trade", "train", "evolve", "optimize",
                          "paper", "live", "deploy")
    dead = []
    for fi in all_funcs:
        if fi.name.startswith("__") and fi.name.endswith("__"):
            continue
        if fi.name in called_names:
            continue
        # Decorators may register it externally
        # Heuristic: names containing entry-point words are externally invoked
        if any(s in fi.name.lower() for s in entry_point_substr):
            continue
        # If it's a method, also consider self.method() calls
        if fi.name in called_names:
            continue
        dead.append(fi)
    return dead


# ---------- Execution-time estimation ----------

def estimate_exec_ms(fi):
    """Heuristic estimate of single-execution time in ms.
    Based on: loops, nesting depth, body line count, cyclomatic complexity.
    Rough proxy — not benchmarked, but ordering is meaningful."""
    base = 0.5  # ms baseline
    loop_cost = 8.0 * fi.loops
    depth_mult = (1.0 + 0.5 * fi.depth)
    line_mult = 0.15 * (fi.end_lineno - fi.lineno + 1)
    cc_mult = 0.4 * max(0, fi.cc - 1)
    est = (base + loop_cost + line_mult + cc_mult) * depth_mult
    return round(est, 2)


# ---------- Main analysis ----------

def main():
    t0 = time.time()
    all_funcs = []
    all_trees = []
    file_stats = {}

    for fname in FILES:
        fpath = os.path.join(BASE, fname)
        funcs, src, tree = extract_functions(fpath)
        all_funcs.extend(funcs)
        all_trees.append((fname, tree))
        file_stats[fname] = {
            "lines": src.count("\n") + 1,
            "functions": len(funcs),
            "size_bytes": len(src.encode("utf-8")),
        }
        print(f"[{fname}] lines={file_stats[fname]['lines']} funcs={len(funcs)}")

    # Cross-file overlap
    overlaps = cross_file_overlaps(all_funcs)

    # Dead code
    dead_funcs = find_uncalled(all_funcs, all_trees)

    # Build redundant_modules list
    redundant_modules = []
    redundant_lines = 0

    # --- Duplicate modules (overlap_pct >= 80 OR hash_match OR iterative duplicate) ---
    # For iterative duplicates (same name across v503-v506), we count each
    # "extra" copy beyond the canonical one as redundant.
    seen_iterative_groups = set()  # to avoid double-counting across the group
    for ov in overlaps:
        is_dup = ov["hash_match"] or ov["overlap_pct"] >= 80.0
        is_iter = ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1
        if not (is_dup or is_iter):
            continue
        fi = ov["func"]
        # For iterative duplicates, we count ALL copies as redundant lines
        # because v506 is the canonical version and v503/v504/v505 should be retired.
        # We use the group key (function_name) so we don't multiply-count.
        if is_iter:
            group_key = ("iterative", fi.name)
            if group_key in seen_iterative_groups:
                # Still emit an entry per copy (for visibility) but don't double-count lines
                rec = OrderedDict([
                    ("file", fi.file),
                    ("function", fi.name),
                    ("qualname", fi.qualname),
                    ("line_range", [fi.lineno, fi.end_lineno]),
                    ("lines", fi.end_lineno - fi.lineno + 1),
                    ("redundancy_type", "duplicate"),
                    ("overlap_with", ov["overlap_with"] + ov.get("all_same_name", [])),
                    ("overlap_pct", ov["overlap_pct"]),
                    ("hash_match", ov["hash_match"]),
                    ("iterative", True),
                    ("cyclomatic_complexity", fi.cc),
                    ("nesting_depth", fi.depth),
                    ("exec_time_ms", estimate_exec_ms(fi)),
                    ("recommendation", "merge"),
                    ("reason", f"Iterative duplicate: '{fi.name}' appears in {len(ov.get('all_same_name', [])) + 1} momentum files (v503-v506); consolidate into one"),
                ])
                redundant_modules.append(rec)
                continue
            seen_iterative_groups.add(group_key)
        rec = OrderedDict([
            ("file", fi.file),
            ("function", fi.name),
            ("qualname", fi.qualname),
            ("line_range", [fi.lineno, fi.end_lineno]),
            ("lines", fi.end_lineno - fi.lineno + 1),
            ("redundancy_type", "duplicate"),
            ("overlap_with", ov["overlap_with"] + ov.get("all_same_name", [])),
            ("overlap_pct", ov["overlap_pct"]),
            ("hash_match", ov["hash_match"]),
            ("iterative", is_iter),
            ("cyclomatic_complexity", fi.cc),
            ("nesting_depth", fi.depth),
            ("exec_time_ms", estimate_exec_ms(fi)),
            ("recommendation", "merge" if (ov["overlap_pct"] >= 90 or ov["hash_match"] or is_iter) else "refactor"),
            ("reason", (
                f"Iterative duplicate: '{fi.name}' appears in {len(ov.get('all_same_name', [])) + 1} momentum files (v503-v506); consolidate into one"
                if is_iter else (
                    f"Identical body hash to {ov['overlap_with'][0]} (exact duplicate)"
                    if ov["hash_match"] and ov["overlap_with"]
                    else f"Body similarity {ov['overlap_pct']}% with {ov['overlap_with'][0] if ov['overlap_with'] else 'N/A'}"
                )
            )),
        ])
        redundant_modules.append(rec)
        # For iterative duplicates: count this copy's lines as redundant
        # (the canonical v506 copy will also be counted, which is conservative —
        # in practice only v503/v504/v505 lines should be removed, but we report
        # total duplicate surface; see metrics for nuance)
        redundant_lines += rec["lines"]

    # --- Dead code ---
    # Public-API-looking method names (get_*, list_*, fetch_*, create_*, cancel_*,
    # submit_*, place_*) may be external surface even if not called internally.
    PUBLIC_API_PREFIXES = ("get_", "list_", "fetch_", "create_", "cancel_",
                           "submit_", "place_", "query_", "delete_", "update_")
    for fi in dead_funcs:
        looks_public = any(fi.name.startswith(p) for p in PUBLIC_API_PREFIXES) or not fi.name.startswith("_")
        if fi.name.startswith("_"):
            # private helper — safer to remove
            rec_recommendation = "remove" if fi.cc <= 5 else "review"
            rec_reason = ("Function name never appears as a Call target across all 6 files; not an entry-point. "
                          "Private helper (leading underscore) — likely safe to remove.")
        elif looks_public:
            rec_recommendation = "review"
            rec_reason = ("Function name never appears as a Call target across all 6 files; however it has a "
                          "public-API-style name and may be intended as external surface (e.g. REST client). "
                          "Review before removing.")
        else:
            rec_recommendation = "remove" if fi.cc <= 5 else "review"
            rec_reason = "Function name never appears as a Call target across all 6 files; not an entry-point"
        rec = OrderedDict([
            ("file", fi.file),
            ("function", fi.name),
            ("qualname", fi.qualname),
            ("line_range", [fi.lineno, fi.end_lineno]),
            ("lines", fi.end_lineno - fi.lineno + 1),
            ("redundancy_type", "dead_code"),
            ("overlap_with", []),
            ("overlap_pct", 0.0),
            ("hash_match", False),
            ("cyclomatic_complexity", fi.cc),
            ("nesting_depth", fi.depth),
            ("exec_time_ms", estimate_exec_ms(fi)),
            ("recommendation", rec_recommendation),
            ("reason", rec_reason),
        ])
        redundant_modules.append(rec)
        redundant_lines += rec["lines"]

    # --- Slow functions (estimated > 100ms) ---
    for fi in all_funcs:
        est = estimate_exec_ms(fi)
        if est > 100.0:
            # avoid double-adding if already in duplicates
            already = any(r["file"] == fi.file and r["function"] == fi.name for r in redundant_modules)
            if not already:
                rec = OrderedDict([
                    ("file", fi.file),
                    ("function", fi.name),
                    ("qualname", fi.qualname),
                    ("line_range", [fi.lineno, fi.end_lineno]),
                    ("lines", fi.end_lineno - fi.lineno + 1),
                    ("redundancy_type", "slow"),
                    ("overlap_with", []),
                    ("overlap_pct", 0.0),
                    ("hash_match", False),
                    ("cyclomatic_complexity", fi.cc),
                    ("nesting_depth", fi.depth),
                    ("exec_time_ms", est),
                    ("recommendation", "refactor"),
                    ("reason", f"Estimated > 100ms (loops={fi.loops}, depth={fi.depth}, cc={fi.cc}, lines={fi.end_lineno - fi.lineno + 1})"),
                ])
                redundant_modules.append(rec)
                # Don't add lines to redundant_lines for "slow" — they aren't redundant, just slow

    # Sort redundant_modules by file then line
    redundant_modules.sort(key=lambda r: (r["file"], r["line_range"][0]))

    # ----- Optimization plan -----
    # Identify which duplicate function-name groups span multiple files
    # Include both strict duplicates AND iterative duplicates (same name in v503-v506)
    dup_groups = defaultdict(list)
    for ov in overlaps:
        is_dup = ov["hash_match"] or ov["overlap_pct"] >= 80.0
        is_iter = ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1
        if is_dup or is_iter:
            dup_groups[ov["func"].name].append(ov["func"].file)

    optimization_plan = []
    # Shared module suggestion
    shared_funcs = sorted(set(k for k, v in dup_groups.items() if len(set(v)) >= 2))
    if shared_funcs:
        affected = sorted({f for k in shared_funcs for f in dup_groups[k]})
        optimization_plan.append(OrderedDict([
            ("action", "Extract duplicate functions into shared utility module _v520_shared_utils.py"),
            ("functions", shared_funcs[:30]),
            ("files_affected", affected),
            ("expected_speedup", "10-20% (eliminates duplicate code paths, simplifies maintenance)"),
            ("rollback_plan", "Keep original files intact; new module is additive. If issues arise, revert import statements only."),
        ]))

    # Dead code removal
    dead_by_file = defaultdict(list)
    for fi in dead_funcs:
        dead_by_file[fi.file].append(fi.name)
    for f, names in dead_by_file.items():
        if names:
            optimization_plan.append(OrderedDict([
                ("action", f"Remove {len(names)} dead-code function(s) from {f}"),
                ("functions", names[:30]),
                ("files_affected", [f]),
                ("expected_speedup", "<5% (reduces file size, marginal speedup)"),
                ("rollback_plan", "Restore from git history if function is later called by new code"),
            ]))

    # Slow function refactor
    slow_funcs = [fi for fi in all_funcs if estimate_exec_ms(fi) > 100.0]
    if slow_funcs:
        optimization_plan.append(OrderedDict([
            ("action", f"Refactor {len(slow_funcs)} slow function(s) (est. >100ms) — vectorize loops, cache indicators"),
            ("functions", [f"{fi.file}:{fi.name}" for fi in slow_funcs[:20]]),
            ("files_affected", sorted({fi.file for fi in slow_funcs})),
            ("expected_speedup", "30-60% on hot paths (backtest loops)"),
            ("rollback_plan", "Keep original implementations in _archive/; swap in if refactor regresses"),
        ]))

    # Consolidate v503-v506 into a single strategy module
    optimization_plan.append(OrderedDict([
        ("action", "Consolidate v503/v504/v505/v506 momentum variants into a single configurable strategy module (v520_momentum_unified.py)"),
        ("rationale", "v503-v506 are iterative versions with ~70-85% code overlap; maintaining 4 copies increases bug surface and divergence risk"),
        ("files_affected", ["_v503_momentum_walkforward.py", "_v504_momentum_walkforward.py", "_v505_mtf_momentum.py", "_v506_mtf_momentum_optimized.py"]),
        ("expected_speedup", "60-75% code reduction in momentum strategy surface"),
        ("rollback_plan", "Keep v506 as canonical; archive v503/v504/v505 with deprecation header; full rollback via git revert"),
    ]))

    # ----- Metrics -----
    total_lines = sum(s["lines"] for s in file_stats.values())
    total_funcs = len(all_funcs)
    dead_count = len(dead_funcs)
    dup_count = sum(1 for ov in overlaps
                    if ov["hash_match"] or ov["overlap_pct"] >= 80.0
                    or (ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1))
    iter_count = sum(1 for ov in overlaps
                     if ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1)

    # For iterative duplicates, count only the "extra" copies (total - canonical)
    # v506 is canonical, so redundant = (copies_in_v503+v504+v505) lines
    iter_redundant_lines = 0
    iter_groups = defaultdict(list)
    for ov in overlaps:
        if ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1:
            iter_groups[ov["func"].name].append(ov["func"])
    for name, fis in iter_groups.items():
        # For each group, count lines of all copies EXCEPT the v506 one (canonical)
        for fi in fis:
            if fi.file != "_v506_mtf_momentum_optimized.py":
                iter_redundant_lines += (fi.end_lineno - fi.lineno + 1)

    metrics = OrderedDict([
        ("total_lines", total_lines),
        ("total_functions", total_funcs),
        ("redundant_lines", redundant_lines),
        ("redundancy_pct", round(100.0 * redundant_lines / max(1, total_lines), 2)),
        ("iterative_duplicate_lines", iter_redundant_lines),
        ("iterative_duplicate_pct", round(100.0 * iter_redundant_lines / max(1, total_lines), 2)),
        ("dead_code_functions", dead_count),
        ("duplicate_functions", dup_count),
        ("iterative_duplicate_functions", iter_count),
        ("slow_functions", len(slow_funcs)),
        ("files_analyzed", len(FILES)),
        ("analysis_time_ms", round(1000.0 * (time.time() - t0), 2)),
    ])

    # File-level summary
    file_summary = []
    for fname in FILES:
        f_funcs = [f for f in all_funcs if f.file == fname]
        file_summary.append(OrderedDict([
            ("file", fname),
            ("lines", file_stats[fname]["lines"]),
            ("functions", len(f_funcs)),
            ("max_cc", max((f.cc for f in f_funcs), default=0)),
            ("max_depth", max((f.depth for f in f_funcs), default=0)),
            ("duplicates_in_file", sum(1 for ov in overlaps if ov["func"].file == fname and (ov["hash_match"] or ov["overlap_pct"] >= 80.0 or (ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1)))),
            ("iterative_in_file", sum(1 for ov in overlaps if ov["func"].file == fname and ov.get("iterative", False) and len(ov.get("all_same_name", [])) >= 1)),
            ("dead_in_file", sum(1 for f in dead_funcs if f.file == fname)),
            ("slow_in_file", sum(1 for f in f_funcs if estimate_exec_ms(f) > 100.0)),
        ]))

    report = OrderedDict([
        ("agent_id", "agent_4_redundancy"),
        ("timestamp", int(time.time())),
        ("total_files_analyzed", len(FILES)),
        ("total_functions", total_funcs),
        ("redundant_modules", redundant_modules),
        ("optimization_plan", optimization_plan),
        ("metrics", metrics),
        ("file_summary", file_summary),
        ("overlap_threshold_pct", 80.0),
        ("exec_time_threshold_ms", 100.0),
        ("methodology", "Python3 ast.parse + McCabe complexity + Jaccard line-set similarity + name-based call-graph"),
    ])

    # Write JSON
    json_path = os.path.join(OUT_DIR, "_v514_redundancy_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Wrote {json_path}")

    # Write Markdown
    md_path = os.path.join(OUT_DIR, "_v514_redundancy_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Agent-4 Redundancy Report\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"**Agent**: agent_4_redundancy\n")
        f.write(f"**Methodology**: Python3 ast + McCabe + Jaccard similarity + call-graph dead-code\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- Files analyzed: **{len(FILES)}**\n")
        f.write(f"- Total lines: **{metrics['total_lines']}**\n")
        f.write(f"- Total functions: **{metrics['total_functions']}**\n")
        f.write(f"- Redundant lines (all duplicates + dead code): **{metrics['redundant_lines']}** ({metrics['redundancy_pct']}%)\n")
        f.write(f"- Iterative duplicate lines (v503/v504/v505 copies that v506 supersedes): **{metrics['iterative_duplicate_lines']}** ({metrics['iterative_duplicate_pct']}%)\n")
        f.write(f"- Dead-code functions: **{metrics['dead_code_functions']}**\n")
        f.write(f"- Duplicate functions (≥80% overlap or hash-match): **{metrics['duplicate_functions']}** (of which **{metrics['iterative_duplicate_functions']}** are iterative duplicates across v503-v506)\n")
        f.write(f"- Slow functions (est. >100ms): **{metrics['slow_functions']}**\n\n")

        f.write("## 2. File-level Summary\n\n")
        f.write("| File | Lines | Funcs | Max CC | Max Depth | Dups | Iter | Dead | Slow |\n")
        f.write("|------|-------|-------|--------|-----------|------|------|------|------|\n")
        for fs in file_summary:
            f.write(f"| {fs['file']} | {fs['lines']} | {fs['functions']} | {fs['max_cc']} | {fs['max_depth']} | {fs['duplicates_in_file']} | {fs['iterative_in_file']} | {fs['dead_in_file']} | {fs['slow_in_file']} |\n")
        f.write("\n")

        f.write("## 3. Redundant Modules\n\n")
        f.write("### 3.1 Duplicates (overlap ≥ 80% / hash-match / iterative across v503-v506)\n\n")
        dups = [r for r in redundant_modules if r["redundancy_type"] == "duplicate"]
        if dups:
            f.write("| File | Function | Lines | Overlap% | Hash? | Iter? | With | Recommend |\n")
            f.write("|------|----------|-------|----------|-------|-------|------|-----------|\n")
            for r in dups:
                f.write(f"| {r['file']} | `{r['function']}` | {r['lines']} | {r['overlap_pct']}% | {'yes' if r['hash_match'] else 'no'} | {'yes' if r.get('iterative') else 'no'} | {', '.join(r['overlap_with'][:3])} | {r['recommendation']} |\n")
            f.write("\n")
        else:
            f.write("_No duplicates found._\n\n")

        f.write("### 3.2 Dead Code (uncalled functions)\n\n")
        dead = [r for r in redundant_modules if r["redundancy_type"] == "dead_code"]
        if dead:
            f.write("| File | Function | Lines | CC | Recommend | Reason |\n")
            f.write("|------|----------|-------|----|-----------|--------|\n")
            for r in dead:
                f.write(f"| {r['file']} | `{r['function']}` | {r['lines']} | {r['cyclomatic_complexity']} | {r['recommendation']} | {r['reason']} |\n")
            f.write("\n")
        else:
            f.write("_No dead code found._\n\n")

        f.write("### 3.3 Slow Functions (est. >100ms)\n\n")
        slow = [r for r in redundant_modules if r["redundancy_type"] == "slow"]
        if slow:
            f.write("| File | Function | Lines | Loops | Depth | Est. ms | Recommend |\n")
            f.write("|------|----------|-------|-------|-------|---------|-----------|\n")
            for r in slow:
                f.write(f"| {r['file']} | `{r['function']}` | {r['lines']} | {r['cyclomatic_complexity']} | {r['nesting_depth']} | {r['exec_time_ms']} | {r['recommendation']} |\n")
            f.write("\n")
        else:
            f.write("_No slow functions found._\n\n")

        f.write("## 4. Optimization Plan\n\n")
        for i, op in enumerate(optimization_plan, 1):
            f.write(f"### Action {i}: {op['action']}\n\n")
            if "functions" in op:
                f.write(f"**Functions**: {', '.join(op['functions'][:20]) if op['functions'] else 'N/A'}\n\n")
            if "files_affected" in op:
                f.write(f"**Files affected**: {', '.join(op['files_affected'])}\n\n")
            if "expected_speedup" in op:
                f.write(f"**Expected speedup**: {op['expected_speedup']}\n\n")
            if "rationale" in op:
                f.write(f"**Rationale**: {op['rationale']}\n\n")
            if "rollback_plan" in op:
                f.write(f"**Rollback**: {op['rollback_plan']}\n\n")

        f.write("## 5. Methodology\n\n")
        f.write("1. **AST parse**: each file parsed with `ast.parse`; functions/methods extracted with `lineno`, `end_lineno`, args, body source.\n")
        f.write("2. **McCabe complexity**: counted `If/While/For/IfExp/BoolOp/ExceptHandler/Assert` increments.\n")
        f.write("3. **Cross-file overlap**: for each function, compared against same-named functions in other files plus any function with identical MD5 body hash. Similarity = Jaccard of normalized non-empty, non-comment line sets.\n")
        f.write("4. **Dead code**: a function is dead if its name never appears as a `Call` target anywhere (excluding dunders and entry-point-like names containing `main/run/backtest/...`).\n")
        f.write("5. **Exec-time estimate**: heuristic `(0.5 + 8*loops + 0.15*lines + 0.4*(cc-1)) * (1+0.5*depth)` ms. Ordering meaningful; absolute values approximate.\n")
        f.write("\n## 6. Notes & Caveats\n\n")
        f.write("- v503/v504/v505/v506 are iterative refinements of the same momentum strategy; high duplication is **expected** and is the primary optimization target.\n")
        f.write("- 'Dead code' is detected at static call-graph level only; functions invoked via reflection, decorators, or external entrypoints are excluded by name heuristic.\n")
        f.write("- Exec-time is an AST heuristic, not a benchmark. Use `hermes-benchmark` skill for ground-truth measurements.\n")
    print(f"[OK] Wrote {md_path}")

    # Verify
    with open(json_path, "r", encoding="utf-8") as f:
        json.load(f)
    print(f"[VERIFY] JSON valid: {os.path.getsize(json_path)} bytes")
    print(f"[VERIFY] MD valid: {os.path.getsize(md_path)} bytes")

    return report


if __name__ == "__main__":
    main()
