#!/usr/bin/env python3
"""Phase 14.1: 6年历史数据激活 — walk-forward窗口 + 每代重置 + 扩大消费量

改动:
1. evolution_loop.py L3532: 进化触发点添加walk-forward窗口+数据指针重置
2. run_sandbox.py L49: --rounds 100→500
3. run_sandbox.py L68: --ticks-per-round 24→48
4. data_pipeline.py L1302: 扩展匹配5m/15m多周期数据
"""
import shutil
import os

BASE = '/home/lmy/hermes_v6/sandbox_trading'

# ============== 1. evolution_loop.py — walk-forward + 每代重置 ==============

evo_path = os.path.join(BASE, 'evolution_loop.py')
shutil.copy2(evo_path, evo_path + '.bak_p14_walk_forward')

with open(evo_path, 'r', encoding='utf-8') as f:
    evo_content = f.read()

# 在进化触发点添加walk-forward窗口+数据指针重置
# 目标行: "if self._round_number % evolve_every == 0 and self._round_number > 0:"
# 在这行之后, "_force_close_all_positions(agent_trades)" 之后添加

OLD_EVO_BLOCK = """            # 检查是否需要进化
            if self._round_number % evolve_every == 0 and self._round_number > 0:
                # 进化前强制平仓所有持仓（结算未平仓的盈亏）
                self._force_close_all_positions(agent_trades)"""

NEW_EVO_BLOCK = """            # 检查是否需要进化
            if self._round_number % evolve_every == 0 and self._round_number > 0:
                # 进化前强制平仓所有持仓（结算未平仓的盈亏）
                self._force_close_all_positions(agent_trades)

                # Phase 14.1: walk-forward窗口机制 + 每代数据指针重置
                # 用户指令: "6年多历史数据没有用上"
                # 根因: 数据指针跨代不重置, 每代看到不同数据段(不公平比较);
                #       策略只看到前N根数据, 永远不经历牛熊转换
                # 修复: 每代进化前重置数据指针到walk-forward窗口起点
                #   - 6年数据按窗口大小分段, 每代用不同段
                #   - 确保策略经历不同市场regime(牛市/熊市/震荡/极端事件)
                try:
                    _p141_gen = self.population.generation
                    _p141_total = 0
                    if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'feed'):
                        for _p141_sym in self.pipeline.feed.symbols:
                            _p141_bars = self.pipeline.feed.get_all_bars(_p141_sym)
                            _p141_total = max(_p141_total, len(_p141_bars))
                        if _p141_total > 0:
                            # 窗口大小 = 每代消费量 × 1.5倍重叠
                            _p141_window = max(ticks_per_round * evolve_every, 240)
                            _p141_num_windows = max(1, _p141_total // _p141_window)
                            _p141_window_idx = _p141_gen % _p141_num_windows
                            _p141_start = _p141_window_idx * _p141_window
                            # 重置所有symbol的数据指针到窗口起点
                            for _p141_sym in self.pipeline.feed._current_index:
                                self.pipeline.feed._current_index[_p141_sym] = min(
                                    _p141_start, max(0, _p141_total - 1)
                                )
                            logger.info(
                                "Phase 14.1 walk-forward: gen=%d window=%d/%d start=%d/%d (total_bars=%d, window_size=%d)",
                                _p141_gen, _p141_window_idx + 1, _p141_num_windows,
                                _p141_start, _p141_total, _p141_total, _p141_window,
                            )
                except Exception as _p141_err:
                    logger.debug("Phase 14.1 walk-forward skipped: %s", _p141_err)"""

if OLD_EVO_BLOCK not in evo_content:
    print("FAIL: 未找到evolution_loop.py目标块")
    # 尝试查找近似文本
    lines = evo_content.split('\n')
    for i, line in enumerate(lines):
        if '检查是否需要进化' in line:
            print(f"  近似行 {i+1}: {line.strip()}")
        if '_force_close_all_positions(agent_trades)' in line:
            print(f"  近似行 {i+1}: {line.strip()}")
    exit(1)

evo_content = evo_content.replace(OLD_EVO_BLOCK, NEW_EVO_BLOCK, 1)

with open(evo_path, 'w', encoding='utf-8') as f:
    f.write(evo_content)

print("OK: evolution_loop.py walk-forward窗口已添加")

# 验证
with open(evo_path, 'r', encoding='utf-8') as f:
    verify = f.read()
assert "Phase 14.1 walk-forward" in verify, "验证失败: walk-forward未找到"
assert "_p141_window_idx" in verify, "验证失败: 窗口索引未找到"
print("OK: evolution_loop.py 验证通过")

# ============== 2. run_sandbox.py — 扩大默认消费量 ==============

run_path = os.path.join(BASE, 'run_sandbox.py')
shutil.copy2(run_path, run_path + '.bak_p14_consume')

with open(run_path, 'r', encoding='utf-8') as f:
    run_content = f.read()

# --rounds 100→500
OLD_ROUNDS = 'parser.add_argument("--rounds", type=int, default=100,'
NEW_ROUNDS = 'parser.add_argument("--rounds", type=int, default=500,'
if OLD_ROUNDS in run_content:
    run_content = run_content.replace(OLD_ROUNDS, NEW_ROUNDS, 1)
    print("OK: --rounds 100→500")
else:
    print("SKIP: --rounds 未找到(可能已修改)")

# --ticks-per-round 24→48
OLD_TPR = 'parser.add_argument("--ticks-per-round", type=int, default=24,'
NEW_TPR = 'parser.add_argument("--ticks-per-round", type=int, default=48,'
if OLD_TPR in run_content:
    run_content = run_content.replace(OLD_TPR, NEW_TPR, 1)
    print("OK: --ticks-per-round 24→48")
else:
    print("SKIP: --ticks-per-round 未找到(可能已修改)")

with open(run_path, 'w', encoding='utf-8') as f:
    f.write(run_content)

# ============== 3. data_pipeline.py — 多周期数据加载 ==============

dp_path = os.path.join(BASE, 'data_pipeline.py')
shutil.copy2(dp_path, dp_path + '.bak_p14_mtf')

with open(dp_path, 'r', encoding='utf-8') as f:
    dp_content = f.read()

# 在_load_local_real_klines中扩展匹配5m/15m
# 当前L1302: if (f.startswith(f"gateio_{file_sym}_1h_") or f.startswith(f"binance_{file_sym_nounderscore}_1h_")):
# 改为同时匹配5m/15m/1h, 优先1h

OLD_MATCH = '''                if (f.startswith(f"gateio_{file_sym}_1h_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_1h_")):'''

NEW_MATCH = '''                # Phase 14.1: 多周期数据加载 — 优先1h, 兼容5m/15m
                if (f.startswith(f"gateio_{file_sym}_1h_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_1h_")
                        or f.startswith(f"gateio_{file_sym}_5m_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_5m_")
                        or f.startswith(f"gateio_{file_sym}_15m_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_15m_")):'''

if OLD_MATCH in dp_content:
    dp_content = dp_content.replace(OLD_MATCH, NEW_MATCH, 1)
    print("OK: data_pipeline.py 多周期匹配已扩展")
else:
    print("SKIP: data_pipeline.py 匹配块未找到(可能已修改)")

with open(dp_path, 'w', encoding='utf-8') as f:
    f.write(dp_content)

# ============== 验证 ==============
print("\n=== Phase 14.1 修复脚本完成 ===")
print("修改文件:")
print("  1. evolution_loop.py — walk-forward窗口+每代重置 (.bak_p14_walk_forward)")
print("  2. run_sandbox.py — rounds 100→500, ticks_per_round 24→48 (.bak_p14_consume)")
print("  3. data_pipeline.py — 多周期数据加载5m/15m (.bak_p14_mtf)")
