#!/usr/bin/env python3
"""Phase 13.4: Activate R17-5 LivePaperComparator sandbox comparison."""
import sys, shutil

FILE = "/home/lmy/hermes_v6/sandbox_trading/evolution_loop.py"

OLD_BLOCK = """        if self.live_paper_comparator is not None:
            try:
                # 框架就绪状态报告(实际对比由实盘运行时触发)
                r17_live_paper_compare_result = {
                    "framework_ready": True,
                    "sharpe_diff_warn_threshold": 0.3,
                    "sharpe_diff_block_threshold": 0.8,
                    "slippage_diff_block_bps": 10.0,
                    "matched_pairs": 0,  # 实盘启动后由KPIMonitor填充
                    "note": "框架就绪,待实盘数据启动对比分析",
                }
            except Exception as _r17_5_err:
"""

NEW_BLOCK = """        if self.live_paper_comparator is not None:
            try:
                # Phase 13.4: 激活沙盘模式真实对比 (paper=理想执行 vs live=含成本执行)
                # 用户原话: "防模拟牛逼实盘亏钱保障机制"
                from .live_paper_comparator import TradeRecord as _LPCTradeRecord
                _r17_paper_records = []
                _r17_live_records = []
                _r17_paper_pnls = []
                _r17_live_pnls = []
                _r17_pair_count = 0

                for _aid, _trades_list in (all_trades or {}).items():
                    for _t in (_trades_list or []):
                        if isinstance(_t, dict):
                            _status = _t.get("status", "")
                            _pnl = float(_t.get("pnl", 0.0))
                            _entry_price = float(_t.get("entry_price", 0.0))
                            _symbol = _t.get("symbol", "unknown")
                            _side = _t.get("side", _t.get("direction", "long"))
                            _entry_ts = float(_t.get("entry_timestamp", _t.get("entry_time", 0.0)))
                            _qty = float(_t.get("quantity", _t.get("size", 0.0)))
                            _fee = float(_t.get("fee_total", 0.0))
                            _slip_bps = float(_t.get("slippage_total", 0.0))
                        else:
                            _status = _t.status
                            _pnl = float(_t.pnl)
                            _entry_price = float(getattr(_t, "entry_price", 0.0))
                            _symbol = getattr(_t, "symbol", "unknown")
                            _side = _t.side.value if hasattr(_t.side, "value") else str(_t.side)
                            _entry_ts = float(getattr(_t, "entry_time", 0.0))
                            _qty = float(getattr(_t, "quantity", getattr(_t, "size", 0.0)))
                            _fee = float(getattr(_t, "fee_total", 0.0))
                            _slip_bps = float(getattr(_t, "slippage_total", 0.0))

                        if _status not in ("closed", "liquidated"):
                            continue
                        if _entry_price <= 0 or _qty <= 0:
                            continue

                        # paper端: 理想执行 (无滑点无手续费)
                        _paper_pnl = _pnl + _fee + (_slip_bps / 10000.0) * _entry_price * _qty
                        _r17_paper_records.append(_LPCTradeRecord(
                            trade_id="paper_%s_%d" % (_aid, int(_entry_ts)),
                            timestamp=_entry_ts,
                            symbol=_symbol,
                            side=_side,
                            price=_entry_price,
                            quantity=_qty,
                            fee_usd=0.0,
                            slippage_bps=0.0,
                            pnl_usd=_paper_pnl,
                            source="paper",
                        ))
                        # live端: 含真实成本执行 (滑点+手续费)
                        _r17_live_records.append(_LPCTradeRecord(
                            trade_id="live_%s_%d" % (_aid, int(_entry_ts)),
                            timestamp=_entry_ts,
                            symbol=_symbol,
                            side=_side,
                            price=_entry_price * (1.0 + _slip_bps / 10000.0),
                            quantity=_qty,
                            fee_usd=_fee,
                            slippage_bps=_slip_bps,
                            pnl_usd=_pnl,
                            source="live",
                        ))
                        _r17_paper_pnls.append(_paper_pnl)
                        _r17_live_pnls.append(_pnl)
                        _r17_pair_count += 1

                if _r17_pair_count >= 5:
                    _r17_pairs = self.live_paper_comparator.match_trades(
                        _r17_paper_records, _r17_live_records, max_time_diff_s=60.0
                    )
                    _r17_report = self.live_paper_comparator.compare(
                        pairs=_r17_pairs,
                        paper_pnls=_r17_paper_pnls,
                        live_pnls=_r17_live_pnls,
                        periods_per_year=365,
                    )
                    _r17_slip_diff = (
                        _r17_report.discrepancy.avg_slippage_diff_bps
                        if _r17_report.discrepancy else 0.0
                    )
                    _r17_sharpe_diff = (
                        abs(_r17_report.kpi_diff.sharpe_diff)
                        if _r17_report.kpi_diff else 0.0
                    )
                    _r17_verdict = _r17_report.final_verdict
                    _r17_verdict_str = _r17_verdict.value if hasattr(_r17_verdict, "value") else str(_r17_verdict)

                    r17_live_paper_compare_result = {
                        "framework_ready": True,
                        "activated": True,
                        "matched_pairs": len(_r17_pairs),
                        "total_trades": _r17_pair_count,
                        "slippage_diff_bps": _r17_slip_diff,
                        "sharpe_diff": _r17_sharpe_diff,
                        "final_verdict": _r17_verdict_str,
                        "reasons": _r17_report.reasons[:3] if _r17_report.reasons else [],
                        "paper_total_pnl": sum(_r17_paper_pnls),
                        "live_total_pnl": sum(_r17_live_pnls),
                        "cost_drag_usd": sum(_r17_paper_pnls) - sum(_r17_live_pnls),
                        "note": "Phase 13.4 沙盘对比: paper=理想执行 vs live=含成本执行",
                    }
                    if _r17_verdict_str == "BLOCKED":
                        can_deploy = False
                        logger.warning(
                            "R17-5 LivePaperComparator BLOCK: slip_diff=%.2fbps sharpe_diff=%.3f reasons=%s",
                            _r17_slip_diff, _r17_sharpe_diff, _r17_report.reasons[:3],
                        )
                    else:
                        logger.info(
                            "R17-5 LivePaperComparator %s: pairs=%d slip_diff=%.2fbps sharpe_diff=%.3f cost_drag=$%.2f",
                            _r17_verdict_str, len(_r17_pairs), _r17_slip_diff,
                            _r17_sharpe_diff, sum(_r17_paper_pnls) - sum(_r17_live_pnls),
                        )
                else:
                    r17_live_paper_compare_result = {
                        "framework_ready": True,
                        "activated": False,
                        "matched_pairs": 0,
                        "note": "交易数不足(<5) 当前=%d" % _r17_pair_count,
                    }
            except Exception as _r17_5_err:
"""

def main():
    with open(FILE, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(OLD_BLOCK)
    if count == 0:
        print("FAIL: 未找到目标块")
        idx = 0
        positions = []
        while True:
            idx = content.find("R17-5", idx)
            if idx == -1:
                break
            line_start = content.rfind("\n", 0, idx) + 1
            line_end = content.find("\n", idx)
            line = content[line_start:line_end].strip()
            positions.append("  pos=%d: %s" % (idx, line[:80]))
            idx += 1
        print("发现 %d 处 R17-5:" % len(positions))
        for p in positions:
            print(p)
        sys.exit(1)

    if count > 1:
        print("FAIL: 找到 %d 处匹配, 期望1处" % count)
        sys.exit(1)

    new_content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)
    shutil.copy2(FILE, FILE + ".bak_p13_r17_5")
    print("OK: 备份已创建 .bak_p13_r17_5")

    with open(FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("OK: R17-5 LivePaperComparator沙盘对比已激活")

    with open(FILE, "r", encoding="utf-8") as f:
        verify = f.read()
    if "Phase 13.4: 激活沙盘模式真实对比" in verify:
        print("OK: 新代码已写入")
    if "框架就绪,待实盘数据启动对比分析" not in verify:
        print("OK: 旧占位符已移除")
    else:
        print("WARN: 旧占位符文本仍存在")

if __name__ == "__main__":
    main()
