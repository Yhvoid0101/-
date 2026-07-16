"""Inject MacroEventGate into evolution_loop.py (3 injection points).

Layer 6: Risk Control Layer - Macro Event Gate
- CPI/FOMC events: 6h before → reduce 50%, 1h before to 1h after → block, 1h-2h after → reduce 50%
- Injected at 3 points: import / __init__ / apply_to_decision

This script reads evolution_loop.py, injects MacroEventGate code at marked positions,
and writes the modified file back. Idempotent: skips if already injected.
"""

import shutil
from pathlib import Path

SRC = Path("/home/lmy/hermes_v6/sandbox_trading/evolution_loop.py")
BAK = SRC.with_suffix(".py.bak_pre_macro_gate")


def inject(content: str) -> str:
    """Inject MacroEventGate at 3 marked positions."""

    # ----- Position 1: Import block (after OOD import) -----
    import_marker = "    OODLevel = None  # type: ignore\n"
    import_addition = """    OODLevel = None  # type: ignore
# Layer 6: Macro Event Gate — 宏观事件门控 (Aegis-X 8层架构补全)
try:
    from .macro_event_gate import MacroEventGate, GateAction
    _MACRO_GATE_AVAILABLE = True
except ImportError:
    _MACRO_GATE_AVAILABLE = False
    MacroEventGate = None  # type: ignore
    GateAction = None  # type: ignore
"""
    if "_MACRO_GATE_AVAILABLE" in content:
        print("[SKIP] Import already injected")
    else:
        if import_marker not in content:
            raise RuntimeError(f"Import marker not found: {import_marker!r}")
        content = content.replace(import_marker, import_addition, 1)
        print("[OK] Import injected")

    # ----- Position 2: __init__ (after OOD detector init) -----
    init_marker = '            logger.warning("Layer5 OODDetector not available")\n'
    init_addition = '''            logger.warning("Layer5 OODDetector not available")
        # Layer 6: Macro Event Gate — 宏观事件门控 (Aegis-X 8层架构补全)
        # CPI/FOMC事件前后自动缩仓/禁止开仓, 防止数据发布打穿止损
        if _MACRO_GATE_AVAILABLE:
            self.macro_event_gate = MacroEventGate(enabled=True)
            _n_events = self.macro_event_gate.load_events()
            logger.info(
                "Layer6 MacroEventGate integrated: events=%d pre=6h block=2h post=2h",
                _n_events,
            )
        else:
            self.macro_event_gate = None
            logger.warning("Layer6 MacroEventGate not available")
'''
    if "self.macro_event_gate" in content:
        print("[SKIP] __init__ already injected")
    else:
        if init_marker not in content:
            raise RuntimeError(f"Init marker not found: {init_marker!r}")
        content = content.replace(init_marker, init_addition, 1)
        print("[OK] __init__ injected")

    # ----- Position 3: apply_to_decision (after OOD check, before v503 min qty) -----
    # Find the v503 marker - that's where we insert before
    v503_marker = "                        # v503 Fix 9: 最小仓位下限"
    macro_gate_block = '''                        # Layer 6: Macro Event Gate — 宏观事件门控 (Aegis-X 8层架构补全)
                        # CPI/FOMC事件前后自动缩仓/禁止开仓
                        if decision and self.macro_event_gate is not None:
                            _macro_orig_action = decision.get("action", "neutral")
                            _macro_orig_qty = decision.get("quantity", 0)
                            _macro_ts = getattr(market, 'timestamp', 0)
                            # 如果market.timestamp是datetime格式, 转为Unix时间戳
                            if _macro_ts and not isinstance(_macro_ts, (int, float)):
                                try:
                                    import datetime as _dt_mod
                                    if isinstance(_macro_ts, _dt_mod.datetime):
                                        if _macro_ts.tzinfo is None:
                                            _macro_ts = _macro_ts.replace(tzinfo=_dt_mod.timezone.utc)
                                        _macro_ts = _macro_ts.timestamp()
                                    else:
                                        _macro_ts = 0
                                except Exception:
                                    _macro_ts = 0
                            decision = self.macro_event_gate.apply_to_decision(
                                decision, _macro_ts, agent_id=aid, symbol=symbol,
                            )
                            if decision is None:
                                if not hasattr(self, '_macro_block_count'):
                                    self._macro_block_count = 0
                                self._macro_block_count += 1
                                if self._macro_block_count <= 5 or self._macro_block_count % 100 == 0:
                                    logger.info(
                                        "Layer6 MacroGate BLOCK #%d: agent=%s symbol=%s action=%s",
                                        self._macro_block_count, aid[:12], symbol, _macro_orig_action,
                                    )
                                continue
                            elif decision.get("macro_gate_action") == "reduce":
                                if not hasattr(self, '_macro_reduce_count'):
                                    self._macro_reduce_count = 0
                                self._macro_reduce_count += 1
                                if self._macro_reduce_count <= 5 or self._macro_reduce_count % 100 == 0:
                                    logger.info(
                                        "Layer6 MacroGate REDUCE #%d: agent=%s symbol=%s action=%s qty=%.6f→%.6f",
                                        self._macro_reduce_count, aid[:12], symbol, _macro_orig_action,
                                        _macro_orig_qty, decision.get("quantity", 0),
                                    )

                        # v503 Fix 9: 最小仓位下限'''
    if "Layer6 MacroGate BLOCK" in content:
        print("[SKIP] apply_to_decision already injected")
    else:
        if v503_marker not in content:
            raise RuntimeError(f"v503 marker not found: {v503_marker!r}")
        content = content.replace(v503_marker, macro_gate_block, 1)
        print("[OK] apply_to_decision injected")

    return content


def main():
    if not SRC.exists():
        raise RuntimeError(f"Source not found: {SRC}")

    # Backup
    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"[OK] Backup created: {BAK}")
    else:
        print(f"[SKIP] Backup already exists: {BAK}")

    # Read
    content = SRC.read_text(encoding="utf-8")
    original_len = len(content)

    # Inject
    content = inject(content)

    # Write
    SRC.write_text(content, encoding="utf-8")
    new_len = len(content)
    print(f"\n[OK] Done. Size: {original_len} → {new_len} (+{new_len - original_len})")

    # Verify
    verify = SRC.read_text(encoding="utf-8")
    checks = [
        ("_MACRO_GATE_AVAILABLE", "import"),
        ("self.macro_event_gate = MacroEventGate", "__init__"),
        ("Layer6 MacroGate BLOCK", "apply_to_decision"),
        ("Layer6 MacroGate REDUCE", "apply_to_decision"),
    ]
    print("\n[VERIFY]")
    all_ok = True
    for marker, label in checks:
        ok = marker in verify
        print(f"  {'OK' if ok else 'FAIL'} {label}: {marker!r}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n[PASS] All 4 verification points passed")
    else:
        print("\n[FAIL] Some verification points failed!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
