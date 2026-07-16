# Agent-4 Redundancy Report

**Generated**: 2026-06-28 23:32:49
**Agent**: agent_4_redundancy
**Methodology**: Python3 ast + McCabe + Jaccard similarity + call-graph dead-code

## 1. Executive Summary

- Files analyzed: **6**
- Total lines: **4828**
- Total functions: **93**
- Redundant lines (all duplicates + dead code): **833** (17.25%)
- Iterative duplicate lines (v503/v504/v505 copies that v506 supersedes): **2121** (43.93%)
- Dead-code functions: **6**
- Duplicate functions (≥80% overlap or hash-match): **48** (of which **46** are iterative duplicates across v503-v506)
- Slow functions (est. >100ms): **19**

## 2. File-level Summary

| File | Lines | Funcs | Max CC | Max Depth | Dups | Iter | Dead | Slow |
|------|-------|-------|--------|-----------|------|------|------|------|
| _v503_momentum_walkforward.py | 885 | 11 | 30 | 7 | 11 | 11 | 0 | 4 |
| _v504_momentum_walkforward.py | 817 | 11 | 36 | 7 | 11 | 11 | 0 | 4 |
| _v505_mtf_momentum.py | 823 | 12 | 39 | 7 | 12 | 12 | 0 | 4 |
| _v506_mtf_momentum_optimized.py | 798 | 12 | 42 | 7 | 12 | 12 | 0 | 4 |
| _v510_paper_trading.py | 546 | 14 | 16 | 6 | 2 | 0 | 0 | 1 |
| _v511_live_trading_infrastructure.py | 959 | 33 | 18 | 7 | 0 | 0 | 6 | 2 |

## 3. Redundant Modules

### 3.1 Duplicates (overlap ≥ 80% / hash-match / iterative across v503-v506)

| File | Function | Lines | Overlap% | Hash? | Iter? | With | Recommend |
|------|----------|-------|----------|-------|-------|------|-----------|
| _v503_momentum_walkforward.py | `load_klines` | 31 | 100.0% | yes | yes | _v504_momentum_walkforward.py:load_klines, _v504_momentum_walkforward.py:load_klines, _v505_mtf_momentum.py:load_klines | merge |
| _v503_momentum_walkforward.py | `calc_indicators` | 48 | 97.5% | no | yes | _v504_momentum_walkforward.py:calc_indicators, _v504_momentum_walkforward.py:calc_indicators, _v505_mtf_momentum.py:calc_indicators | merge |
| _v503_momentum_walkforward.py | `momentum_signal` | 33 | 82.61% | no | yes | _v504_momentum_walkforward.py:momentum_signal, _v504_momentum_walkforward.py:momentum_signal | merge |
| _v503_momentum_walkforward.py | `backtest_window` | 147 | 85.33% | no | yes | _v504_momentum_walkforward.py:backtest_window, _v504_momentum_walkforward.py:backtest_window, _v505_mtf_momentum.py:backtest_window | merge |
| _v503_momentum_walkforward.py | `_calc_metrics` | 82 | 77.46% | no | yes | _v504_momentum_walkforward.py:_calc_metrics, _v504_momentum_walkforward.py:_calc_metrics, _v505_mtf_momentum.py:_calc_metrics | merge |
| _v503_momentum_walkforward.py | `walk_forward_evaluate` | 153 | 81.36% | no | yes | _v504_momentum_walkforward.py:walk_forward_evaluate, _v504_momentum_walkforward.py:walk_forward_evaluate, _v505_mtf_momentum.py:walk_forward_evaluate | merge |
| _v503_momentum_walkforward.py | `random_individual` | 2 | 100.0% | no | yes | _v504_momentum_walkforward.py:random_individual, _v504_momentum_walkforward.py:random_individual, _v505_mtf_momentum.py:random_individual | merge |
| _v503_momentum_walkforward.py | `tournament_select` | 3 | 100.0% | no | yes | _v504_momentum_walkforward.py:tournament_select, _v504_momentum_walkforward.py:tournament_select, _v505_mtf_momentum.py:tournament_select | merge |
| _v503_momentum_walkforward.py | `uniform_crossover` | 12 | 100.0% | yes | yes | _v504_momentum_walkforward.py:uniform_crossover, _v504_momentum_walkforward.py:uniform_crossover, _v505_mtf_momentum.py:uniform_crossover | merge |
| _v503_momentum_walkforward.py | `gaussian_mutation` | 10 | 100.0% | yes | yes | _v504_momentum_walkforward.py:gaussian_mutation, _v504_momentum_walkforward.py:gaussian_mutation, _v505_mtf_momentum.py:gaussian_mutation | merge |
| _v503_momentum_walkforward.py | `main` | 208 | 70.0% | no | yes | _v504_momentum_walkforward.py:main, _v504_momentum_walkforward.py:main, _v505_mtf_momentum.py:main | merge |
| _v504_momentum_walkforward.py | `load_klines` | 31 | 100.0% | yes | yes | _v503_momentum_walkforward.py:load_klines, _v503_momentum_walkforward.py:load_klines, _v505_mtf_momentum.py:load_klines | merge |
| _v504_momentum_walkforward.py | `calc_indicators` | 42 | 97.5% | no | yes | _v503_momentum_walkforward.py:calc_indicators, _v503_momentum_walkforward.py:calc_indicators, _v505_mtf_momentum.py:calc_indicators | merge |
| _v504_momentum_walkforward.py | `momentum_signal` | 25 | 82.61% | no | yes | _v503_momentum_walkforward.py:momentum_signal, _v503_momentum_walkforward.py:momentum_signal | merge |
| _v504_momentum_walkforward.py | `backtest_window` | 131 | 85.33% | no | yes | _v503_momentum_walkforward.py:backtest_window, _v503_momentum_walkforward.py:backtest_window, _v505_mtf_momentum.py:backtest_window | merge |
| _v504_momentum_walkforward.py | `_calc_metrics` | 77 | 100.0% | yes | yes | _v505_mtf_momentum.py:_calc_metrics, _v503_momentum_walkforward.py:_calc_metrics, _v505_mtf_momentum.py:_calc_metrics | merge |
| _v504_momentum_walkforward.py | `walk_forward_evaluate` | 158 | 81.36% | no | yes | _v503_momentum_walkforward.py:walk_forward_evaluate, _v503_momentum_walkforward.py:walk_forward_evaluate, _v505_mtf_momentum.py:walk_forward_evaluate | merge |
| _v504_momentum_walkforward.py | `random_individual` | 2 | 100.0% | no | yes | _v503_momentum_walkforward.py:random_individual, _v503_momentum_walkforward.py:random_individual, _v505_mtf_momentum.py:random_individual | merge |
| _v504_momentum_walkforward.py | `tournament_select` | 3 | 100.0% | no | yes | _v503_momentum_walkforward.py:tournament_select, _v503_momentum_walkforward.py:tournament_select, _v505_mtf_momentum.py:tournament_select | merge |
| _v504_momentum_walkforward.py | `uniform_crossover` | 12 | 100.0% | yes | yes | _v503_momentum_walkforward.py:uniform_crossover, _v503_momentum_walkforward.py:uniform_crossover, _v505_mtf_momentum.py:uniform_crossover | merge |
| _v504_momentum_walkforward.py | `gaussian_mutation` | 10 | 100.0% | yes | yes | _v503_momentum_walkforward.py:gaussian_mutation, _v503_momentum_walkforward.py:gaussian_mutation, _v505_mtf_momentum.py:gaussian_mutation | merge |
| _v504_momentum_walkforward.py | `main` | 193 | 70.0% | no | yes | _v503_momentum_walkforward.py:main, _v503_momentum_walkforward.py:main, _v505_mtf_momentum.py:main | merge |
| _v505_mtf_momentum.py | `load_klines` | 30 | 100.0% | yes | yes | _v506_mtf_momentum_optimized.py:load_klines, _v503_momentum_walkforward.py:load_klines, _v504_momentum_walkforward.py:load_klines | merge |
| _v505_mtf_momentum.py | `calc_indicators` | 44 | 70.0% | no | yes | _v506_mtf_momentum_optimized.py:calc_indicators, _v503_momentum_walkforward.py:calc_indicators, _v504_momentum_walkforward.py:calc_indicators | merge |
| _v505_mtf_momentum.py | `calc_ma` | 8 | 87.5% | no | yes | _v506_mtf_momentum_optimized.py:calc_ma, _v506_mtf_momentum_optimized.py:calc_ma | merge |
| _v505_mtf_momentum.py | `mtf_signal` | 35 | 84.0% | no | yes | _v506_mtf_momentum_optimized.py:mtf_signal, _v506_mtf_momentum_optimized.py:mtf_signal | merge |
| _v505_mtf_momentum.py | `backtest_window` | 145 | 97.33% | no | yes | _v506_mtf_momentum_optimized.py:backtest_window, _v503_momentum_walkforward.py:backtest_window, _v504_momentum_walkforward.py:backtest_window | merge |
| _v505_mtf_momentum.py | `_calc_metrics` | 74 | 100.0% | yes | yes | _v504_momentum_walkforward.py:_calc_metrics, _v503_momentum_walkforward.py:_calc_metrics, _v504_momentum_walkforward.py:_calc_metrics | merge |
| _v505_mtf_momentum.py | `walk_forward_evaluate` | 162 | 89.39% | no | yes | _v506_mtf_momentum_optimized.py:walk_forward_evaluate, _v503_momentum_walkforward.py:walk_forward_evaluate, _v504_momentum_walkforward.py:walk_forward_evaluate | merge |
| _v505_mtf_momentum.py | `random_individual` | 2 | 100.0% | no | yes | _v503_momentum_walkforward.py:random_individual, _v503_momentum_walkforward.py:random_individual, _v504_momentum_walkforward.py:random_individual | merge |
| _v505_mtf_momentum.py | `tournament_select` | 3 | 100.0% | no | yes | _v503_momentum_walkforward.py:tournament_select, _v503_momentum_walkforward.py:tournament_select, _v504_momentum_walkforward.py:tournament_select | merge |
| _v505_mtf_momentum.py | `uniform_crossover` | 12 | 100.0% | yes | yes | _v503_momentum_walkforward.py:uniform_crossover, _v503_momentum_walkforward.py:uniform_crossover, _v504_momentum_walkforward.py:uniform_crossover | merge |
| _v505_mtf_momentum.py | `gaussian_mutation` | 10 | 100.0% | yes | yes | _v503_momentum_walkforward.py:gaussian_mutation, _v503_momentum_walkforward.py:gaussian_mutation, _v504_momentum_walkforward.py:gaussian_mutation | merge |
| _v505_mtf_momentum.py | `main` | 183 | 70.0% | no | yes | _v506_mtf_momentum_optimized.py:main, _v503_momentum_walkforward.py:main, _v504_momentum_walkforward.py:main | merge |
| _v506_mtf_momentum_optimized.py | `load_klines` | 30 | 100.0% | yes | yes | _v505_mtf_momentum.py:load_klines, _v503_momentum_walkforward.py:load_klines, _v504_momentum_walkforward.py:load_klines | merge |
| _v506_mtf_momentum_optimized.py | `calc_indicators` | 19 | 100.0% | yes | yes | _v510_paper_trading.py:calc_indicators, _v503_momentum_walkforward.py:calc_indicators, _v504_momentum_walkforward.py:calc_indicators | merge |
| _v506_mtf_momentum_optimized.py | `calc_ma` | 7 | 100.0% | yes | yes | _v510_paper_trading.py:calc_ma, _v505_mtf_momentum.py:calc_ma | merge |
| _v506_mtf_momentum_optimized.py | `mtf_signal` | 29 | 84.0% | no | yes | _v505_mtf_momentum.py:mtf_signal, _v505_mtf_momentum.py:mtf_signal | merge |
| _v506_mtf_momentum_optimized.py | `backtest_window` | 143 | 97.33% | no | yes | _v505_mtf_momentum.py:backtest_window, _v503_momentum_walkforward.py:backtest_window, _v504_momentum_walkforward.py:backtest_window | merge |
| _v506_mtf_momentum_optimized.py | `_calc_metrics` | 74 | 100.0% | yes | yes | _v504_momentum_walkforward.py:_calc_metrics, _v503_momentum_walkforward.py:_calc_metrics, _v504_momentum_walkforward.py:_calc_metrics | merge |
| _v506_mtf_momentum_optimized.py | `walk_forward_evaluate` | 182 | 89.39% | no | yes | _v505_mtf_momentum.py:walk_forward_evaluate, _v503_momentum_walkforward.py:walk_forward_evaluate, _v504_momentum_walkforward.py:walk_forward_evaluate | merge |
| _v506_mtf_momentum_optimized.py | `random_individual` | 2 | 100.0% | no | yes | _v503_momentum_walkforward.py:random_individual, _v503_momentum_walkforward.py:random_individual, _v504_momentum_walkforward.py:random_individual | merge |
| _v506_mtf_momentum_optimized.py | `tournament_select` | 3 | 100.0% | no | yes | _v503_momentum_walkforward.py:tournament_select, _v503_momentum_walkforward.py:tournament_select, _v504_momentum_walkforward.py:tournament_select | merge |
| _v506_mtf_momentum_optimized.py | `uniform_crossover` | 12 | 100.0% | yes | yes | _v503_momentum_walkforward.py:uniform_crossover, _v503_momentum_walkforward.py:uniform_crossover, _v504_momentum_walkforward.py:uniform_crossover | merge |
| _v506_mtf_momentum_optimized.py | `gaussian_mutation` | 10 | 100.0% | yes | yes | _v503_momentum_walkforward.py:gaussian_mutation, _v503_momentum_walkforward.py:gaussian_mutation, _v504_momentum_walkforward.py:gaussian_mutation | merge |
| _v506_mtf_momentum_optimized.py | `main` | 180 | 70.0% | no | yes | _v505_mtf_momentum.py:main, _v503_momentum_walkforward.py:main, _v504_momentum_walkforward.py:main | merge |
| _v510_paper_trading.py | `calc_indicators` | 19 | 100.0% | yes | no | _v506_mtf_momentum_optimized.py:calc_indicators | merge |
| _v510_paper_trading.py | `calc_ma` | 7 | 100.0% | yes | no | _v506_mtf_momentum_optimized.py:calc_ma | merge |

### 3.2 Dead Code (uncalled functions)

| File | Function | Lines | CC | Recommend | Reason |
|------|----------|-------|----|-----------|--------|
| _v511_live_trading_infrastructure.py | `get_account` | 8 | 5 | review | Function name never appears as a Call target across all 6 files; however it has a public-API-style name and may be intended as external surface (e.g. REST client). Review before removing. |
| _v511_live_trading_infrastructure.py | `get_order` | 5 | 1 | review | Function name never appears as a Call target across all 6 files; however it has a public-API-style name and may be intended as external surface (e.g. REST client). Review before removing. |
| _v511_live_trading_infrastructure.py | `list_open_orders` | 6 | 2 | review | Function name never appears as a Call target across all 6 files; however it has a public-API-style name and may be intended as external surface (e.g. REST client). Review before removing. |
| _v511_live_trading_infrastructure.py | `list_tickers` | 6 | 2 | review | Function name never appears as a Call target across all 6 files; however it has a public-API-style name and may be intended as external surface (e.g. REST client). Review before removing. |
| _v511_live_trading_infrastructure.py | `list_candlesticks` | 5 | 1 | review | Function name never appears as a Call target across all 6 files; however it has a public-API-style name and may be intended as external surface (e.g. REST client). Review before removing. |
| _v511_live_trading_infrastructure.py | `_cancel_order` | 5 | 4 | remove | Function name never appears as a Call target across all 6 files; not an entry-point. Private helper (leading underscore) — likely safe to remove. |

### 3.3 Slow Functions (est. >100ms)

| File | Function | Lines | Loops | Depth | Est. ms | Recommend |
|------|----------|-------|-------|-------|---------|-----------|
| _v510_paper_trading.py | `main` | 120 | 10 | 6 | 184.4 | refactor |
| _v511_live_trading_infrastructure.py | `monitor_positions` | 44 | 15 | 7 | 129.15 | refactor |
| _v511_live_trading_infrastructure.py | `main` | 151 | 18 | 5 | 188.83 | refactor |

## 4. Optimization Plan

### Action 1: Extract duplicate functions into shared utility module _v520_shared_utils.py

**Functions**: _calc_metrics, backtest_window, calc_indicators, calc_ma, gaussian_mutation, load_klines, main, momentum_signal, mtf_signal, random_individual, tournament_select, uniform_crossover, walk_forward_evaluate

**Files affected**: _v503_momentum_walkforward.py, _v504_momentum_walkforward.py, _v505_mtf_momentum.py, _v506_mtf_momentum_optimized.py, _v510_paper_trading.py

**Expected speedup**: 10-20% (eliminates duplicate code paths, simplifies maintenance)

**Rollback**: Keep original files intact; new module is additive. If issues arise, revert import statements only.

### Action 2: Remove 6 dead-code function(s) from _v511_live_trading_infrastructure.py

**Functions**: get_account, get_order, list_open_orders, list_tickers, list_candlesticks, _cancel_order

**Files affected**: _v511_live_trading_infrastructure.py

**Expected speedup**: <5% (reduces file size, marginal speedup)

**Rollback**: Restore from git history if function is later called by new code

### Action 3: Refactor 19 slow function(s) (est. >100ms) — vectorize loops, cache indicators

**Functions**: _v503_momentum_walkforward.py:load_klines, _v503_momentum_walkforward.py:backtest_window, _v503_momentum_walkforward.py:walk_forward_evaluate, _v503_momentum_walkforward.py:main, _v504_momentum_walkforward.py:load_klines, _v504_momentum_walkforward.py:backtest_window, _v504_momentum_walkforward.py:walk_forward_evaluate, _v504_momentum_walkforward.py:main, _v505_mtf_momentum.py:load_klines, _v505_mtf_momentum.py:backtest_window, _v505_mtf_momentum.py:walk_forward_evaluate, _v505_mtf_momentum.py:main, _v506_mtf_momentum_optimized.py:load_klines, _v506_mtf_momentum_optimized.py:backtest_window, _v506_mtf_momentum_optimized.py:walk_forward_evaluate, _v506_mtf_momentum_optimized.py:main, _v510_paper_trading.py:main, _v511_live_trading_infrastructure.py:monitor_positions, _v511_live_trading_infrastructure.py:main

**Files affected**: _v503_momentum_walkforward.py, _v504_momentum_walkforward.py, _v505_mtf_momentum.py, _v506_mtf_momentum_optimized.py, _v510_paper_trading.py, _v511_live_trading_infrastructure.py

**Expected speedup**: 30-60% on hot paths (backtest loops)

**Rollback**: Keep original implementations in _archive/; swap in if refactor regresses

### Action 4: Consolidate v503/v504/v505/v506 momentum variants into a single configurable strategy module (v520_momentum_unified.py)

**Files affected**: _v503_momentum_walkforward.py, _v504_momentum_walkforward.py, _v505_mtf_momentum.py, _v506_mtf_momentum_optimized.py

**Expected speedup**: 60-75% code reduction in momentum strategy surface

**Rationale**: v503-v506 are iterative versions with ~70-85% code overlap; maintaining 4 copies increases bug surface and divergence risk

**Rollback**: Keep v506 as canonical; archive v503/v504/v505 with deprecation header; full rollback via git revert

## 5. Methodology

1. **AST parse**: each file parsed with `ast.parse`; functions/methods extracted with `lineno`, `end_lineno`, args, body source.
2. **McCabe complexity**: counted `If/While/For/IfExp/BoolOp/ExceptHandler/Assert` increments.
3. **Cross-file overlap**: for each function, compared against same-named functions in other files plus any function with identical MD5 body hash. Similarity = Jaccard of normalized non-empty, non-comment line sets.
4. **Dead code**: a function is dead if its name never appears as a `Call` target anywhere (excluding dunders and entry-point-like names containing `main/run/backtest/...`).
5. **Exec-time estimate**: heuristic `(0.5 + 8*loops + 0.15*lines + 0.4*(cc-1)) * (1+0.5*depth)` ms. Ordering meaningful; absolute values approximate.

## 6. Notes & Caveats

- v503/v504/v505/v506 are iterative refinements of the same momentum strategy; high duplication is **expected** and is the primary optimization target.
- 'Dead code' is detected at static call-graph level only; functions invoked via reflection, decorators, or external entrypoints are excluded by name heuristic.
- Exec-time is an AST heuristic, not a benchmark. Use `hermes-benchmark` skill for ground-truth measurements.
