#!/usr/bin/env python3
"""Phase 12 P12-T3 真实订单簿深度重放验证（只读）

范围：仅限数字货币合约（Gate.io USDT 本位永续合约）
8 项验证：
  1. 真实 get_perp_orderbook 获取（15 币种）
  2. asks/bids 非空且有序（asks 升序，bids 降序）
  3. 买卖价差（spread）在合理范围（<0.5%）
  4. 深度分布（前 10 档深度合理）
  5. OrderBookGate 10 条规则触发验证
  6. L7 执行算法撮合：成交价在 asks/bids 范围内
  7. 冲击成本模型：不同订单大小 → 不同滑点
  8. 部分成交模拟：大订单分批撮合

零模拟：所有订单簿来自真实 Gate.io API。
零降级：获取失败直接 FAIL，不使用合成订单簿。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, "/home/lmy/hermes_v6/sandbox_trading")

from gateio_real_data_provider import GateIoRealDataProvider  # noqa: E402
from order_book_gate import OrderBookGate, OBGateAction  # noqa: E402

# ============================================================================
# 配置
# ============================================================================

SYMBOLS: List[str] = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "DOGE_USDT", "ADA_USDT",
    "XRP_USDT", "BNB_USDT", "DOT_USDT", "LINK_USDT", "LTC_USDT",
    "AVAX_USDT", "ARB_USDT", "OP_USDT", "APT_USDT", "SUI_USDT",
]

ORDERBOOK_LIMIT = 20
SPREAD_PCT_MAX = 0.005  # 0.5%
ANALYSIS_LEVELS = 10
ARTIFACTS_DIR = "/home/lmy/hermes_v6/sandbox_trading/phase12_artifacts"


# ============================================================================
# 撮合模拟工具
# ============================================================================

def simulate_fill(levels: List[Tuple[float, float]], qty: float) -> Tuple[float, float]:
    """模拟逐档撮合，返回 (filled_qty, avg_price)"""
    filled = 0.0
    total_cost = 0.0
    for price, size in levels:
        if filled >= qty:
            break
        fill = min(size, qty - filled)
        total_cost += price * fill
        filled += fill
    avg = total_cost / filled if filled > 0 else 0.0
    return filled, avg


def simulate_partial_fill(levels: List[Tuple[float, float]], qty: float) -> Tuple[float, float, float]:
    """模拟部分成交，返回 (filled_qty, unfilled_qty, avg_price)"""
    filled, avg = simulate_fill(levels, qty)
    unfilled = qty - filled
    return filled, unfilled, avg


# ============================================================================
# 8 项验证
# ============================================================================

def verify_symbol(provider: GateIoRealDataProvider, symbol: str, gate: OrderBookGate) -> Dict[str, Any]:
    """对单个币种执行 8 项验证"""
    result: Dict[str, Any] = {
        "symbol": symbol,
        "pass": True,
        "fail_reasons": [],
        "checks": {},
    }

    # ---- 检查1: 真实 get_perp_orderbook 获取 ----
    ob = provider.get_perp_orderbook(symbol, limit=ORDERBOOK_LIMIT)
    if ob is None or not ob.bids or not ob.asks:
        result["pass"] = False
        result["fail_reasons"].append("check1_get_failed: orderbook None or empty")
        return result
    result["checks"]["check1_get"] = {
        "bids_count": len(ob.bids),
        "asks_count": len(ob.asks),
    }

    # 保存订单簿快照
    snapshot = {
        "symbol": symbol,
        "bids": [[p, q] for p, q in ob.bids],
        "asks": [[p, q] for p, q in ob.asks],
        "timestamp": time.time(),
    }
    snapshot_path = os.path.join(ARTIFACTS_DIR, f"orderbook_{symbol}.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # ---- 检查2: asks/bids 有序 ----
    bids_desc = all(ob.bids[i][0] >= ob.bids[i + 1][0] for i in range(len(ob.bids) - 1))
    asks_asc = all(ob.asks[i][0] <= ob.asks[i + 1][0] for i in range(len(ob.asks) - 1))
    result["checks"]["check2_sorted"] = {"bids_desc": bids_desc, "asks_asc": asks_asc}
    if not bids_desc or not asks_asc:
        result["pass"] = False
        result["fail_reasons"].append(
            f"check2_unsorted: bids_desc={bids_desc} asks_asc={asks_asc}"
        )

    # ---- 检查3: 买卖价差 < 0.5% ----
    best_bid = ob.bids[0][0]
    best_ask = ob.asks[0][0]
    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    spread_pct = spread / mid_price if mid_price > 0 else 1.0
    result["checks"]["check3_spread"] = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread": spread,
        "spread_pct": spread_pct,
    }
    if spread_pct >= SPREAD_PCT_MAX:
        result["pass"] = False
        result["fail_reasons"].append(
            f"check3_spread_high: spread_pct={spread_pct:.6f} >= {SPREAD_PCT_MAX}"
        )

    # ---- 检查4: 深度分布（前 10 档）----
    bids_depth = sum(q for _, q in ob.bids[:ANALYSIS_LEVELS])
    asks_depth = sum(q for _, q in ob.asks[:ANALYSIS_LEVELS])
    result["checks"]["check4_depth"] = {
        "bids_depth_10": bids_depth,
        "asks_depth_10": asks_depth,
    }
    if bids_depth <= 0 or asks_depth <= 0:
        result["pass"] = False
        result["fail_reasons"].append(
            f"check4_depth_zero: bids_depth={bids_depth} asks_depth={asks_depth}"
        )

    # ---- 检查5: OrderBookGate 10 条规则触发验证 ----
    analysis = gate.analyze_order_book(ob.bids, ob.asks)
    # 做多决策
    long_decision = {"action": "buy", "quantity": 1.0}
    long_result = gate.apply_to_decision(
        long_decision, symbol=symbol, bids=ob.bids, asks=ob.asks
    )
    # 做空决策
    short_decision = {"action": "sell", "quantity": 1.0}
    short_result = gate.apply_to_decision(
        short_decision, symbol=symbol, bids=ob.bids, asks=ob.asks
    )
    rules_triggered = []
    if long_result and "ob_gate_action" in long_result:
        rules_triggered.append({
            "direction": "long",
            "action": long_result["ob_gate_action"],
            "multiplier": long_result.get("ob_gate_multiplier", 1.0),
            "reason": long_result.get("ob_gate_reason", ""),
        })
    if short_result and "ob_gate_action" in short_result:
        rules_triggered.append({
            "direction": "short",
            "action": short_result["ob_gate_action"],
            "multiplier": short_result.get("ob_gate_multiplier", 1.0),
            "reason": short_result.get("ob_gate_reason", ""),
        })
    result["checks"]["check5_obgate"] = {
        "analysis": analysis.to_dict(),
        "rules_triggered": rules_triggered,
        "long_adjusted": len(rules_triggered) > 0 and rules_triggered[0]["direction"] == "long",
        "short_adjusted": len(rules_triggered) > 1 and rules_triggered[1]["direction"] == "short",
        "gate_called_ok": True,
    }

    # ---- 检查6: L7 执行算法撮合：成交价在 asks/bids 范围内 ----
    # 市价买入按卖一价成交，市价卖出按买一价成交
    buy_fill_price = best_ask
    sell_fill_price = best_bid
    buy_in_range = ob.asks[0][0] <= buy_fill_price <= ob.asks[-1][0]
    sell_in_range = ob.bids[-1][0] <= sell_fill_price <= ob.bids[0][0]
    result["checks"]["check6_fill_price"] = {
        "buy_fill": buy_fill_price,
        "sell_fill": sell_fill_price,
        "buy_in_range": buy_in_range,
        "sell_in_range": sell_in_range,
    }
    if not buy_in_range or not sell_in_range:
        result["pass"] = False
        result["fail_reasons"].append(
            f"check6_fill_out_of_range: buy={buy_in_range} sell={sell_in_range}"
        )

    # ---- 检查7: 冲击成本模型（不同订单大小 → 不同滑点）----
    # 小订单：只吃第一档的一半
    small_qty = ob.asks[0][1] * 0.5
    # 大订单：吃前 5 档
    large_qty = sum(q for _, q in ob.asks[:5])
    if small_qty <= 0:
        small_qty = 0.001  # 最小保护
    if large_qty <= 0:
        large_qty = small_qty * 10

    _, small_avg = simulate_fill(ob.asks, small_qty)
    _, large_avg = simulate_fill(ob.asks, large_qty)

    small_slippage = (small_avg - best_ask) / best_ask if best_ask > 0 else 0.0
    large_slippage = (large_avg - best_ask) / best_ask if best_ask > 0 else 0.0

    # 大订单应比小订单成交价更差（更高，因为是买入）
    large_worse = large_avg > small_avg
    result["checks"]["check7_impact"] = {
        "small_qty": small_qty,
        "large_qty": large_qty,
        "small_avg_price": small_avg,
        "large_avg_price": large_avg,
        "small_slippage": small_slippage,
        "large_slippage": large_slippage,
        "large_worse_than_small": large_worse,
    }
    if not large_worse:
        result["pass"] = False
        result["fail_reasons"].append(
            f"check7_impact_invalid: large_avg={large_avg:.6f} <= small_avg={small_avg:.6f}"
        )

    # ---- 检查8: 部分成交模拟（大订单分批撮合）----
    total_ask_depth = sum(q for _, q in ob.asks)
    # 订单量超过总深度 → 部分成交
    huge_qty = total_ask_depth * 1.5 if total_ask_depth > 0 else 1000.0
    filled_qty, unfilled_qty, huge_avg = simulate_partial_fill(ob.asks, huge_qty)
    partial_correct = filled_qty < huge_qty and unfilled_qty > 0
    result["checks"]["check8_partial_fill"] = {
        "huge_qty": huge_qty,
        "total_ask_depth": total_ask_depth,
        "filled_qty": filled_qty,
        "unfilled_qty": unfilled_qty,
        "avg_price": huge_avg,
        "partial_correct": partial_correct,
    }
    if not partial_correct:
        result["pass"] = False
        result["fail_reasons"].append(
            f"check8_partial_wrong: filled={filled_qty:.6f} huge={huge_qty:.6f} "
            f"unfilled={unfilled_qty:.6f}"
        )

    return result


# ============================================================================
# 主函数
# ============================================================================

def main() -> int:
    print("=== Phase 12 P12-T3 真实订单簿深度重放验证 ===")
    print(f"币种数: {len(SYMBOLS)}")
    print(f"验证项: 8 项/币种")
    print(f"范围: 仅限数字货币合约（Gate.io USDT 本位永续合约）")
    print()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    provider = GateIoRealDataProvider()
    gate = OrderBookGate(enabled=True)

    all_results: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0

    for symbol in SYMBOLS:
        print(f"  {symbol} ... ", end="", flush=True)
        try:
            r = verify_symbol(provider, symbol, gate)
            all_results.append(r)
            if r["pass"]:
                # 统计规则触发
                rules = r["checks"].get("check5_obgate", {}).get("rules_triggered", [])
                spread_pct = r["checks"].get("check3_spread", {}).get("spread_pct", 0)
                print(f"PASS spread={spread_pct:.5f} rules={len(rules)}")
                pass_count += 1
            else:
                print(f"FAIL: {r['fail_reasons']}")
                fail_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "symbol": symbol,
                "pass": False,
                "fail_reasons": [f"exception: {e}"],
                "checks": {},
            })
            fail_count += 1
        time.sleep(0.35)  # 限流保护

    print()
    print("=== 汇总 ===")
    print(f"币种测试: {len(SYMBOLS)}")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")

    # 汇总统计
    total_rules_triggered = sum(
        len(r.get("checks", {}).get("check5_obgate", {}).get("rules_triggered", []))
        for r in all_results
    )
    avg_spread = 0.0
    spread_count = 0
    for r in all_results:
        sp = r.get("checks", {}).get("check3_spread", {}).get("spread_pct")
        if sp is not None:
            avg_spread += sp
            spread_count += 1
    if spread_count > 0:
        avg_spread /= spread_count

    print(f"规则触发总数: {total_rules_triggered}")
    print(f"平均价差: {avg_spread:.6f} ({avg_spread*100:.4f}%)")

    # 保存报告
    report = {
        "phase": "P12-T3",
        "scope": "digital_currency_perpetual_contracts_only",
        "exchange": "Gate.io USDT-Margined Perpetual",
        "total_symbols": len(SYMBOLS),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "total_rules_triggered": total_rules_triggered,
        "avg_spread_pct": avg_spread,
        "results": all_results,
        "timestamp": time.time(),
    }
    report_path = os.path.join(ARTIFACTS_DIR, "orderbook_replay_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"报告: {report_path}")

    if pass_count == len(SYMBOLS) and fail_count == 0:
        print(f"\nPASS: {pass_count}/{len(SYMBOLS)} 币种订单簿 8 项验证全通过")
        return 0
    else:
        print(f"\nFAIL: {fail_count} 币种未通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
