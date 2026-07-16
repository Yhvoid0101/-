# -*- coding: utf-8 -*-
"""Maker 成本模型 (Maker Cost Model) — 6大核心短板 #3

提取自 _v560_maker_integrated.py，去除脚本式副作用（sys.exit劫持/sys.path.insert/
重量级v82回测/模块级文件写入），保留纯数据+纯函数，遵循 sim_live_gap_model.py 的
架构范本（dataclass + 类封装 + __main__ 守卫）。

用户核心诉求: "模拟/实盘差异全成本量化模型+资金容量压力测试，根治模拟好看实盘亏"

设计原则:
  - spread 从成本变收益: Maker 挂单赚取半 spread，而非支付 spread
  - 滑点/流动性/延迟打折: Maker 挂单不影响市场 (30%/30%/50%)
  - fee 折扣: Maker 比 Taker 便宜 0.2bps
  - adverse selection: Maker 被成交时价格已朝不利方向移动 (0.5bps)
  - per-symbol 精确成本: 11 个主流交易对各自点差/流动性参数

核心公式 (v73p/v86, ERR-102验证):
  spread_revenue = position × spread_bps/10000 × 0.5 × fill_rate × 2
  slippage = taker_slippage × 0.3  (Maker挂单不影响市场)
  liquidity = taker_liquidity × 0.3  (Maker提供流动性)
  latency = taker_latency × 0.5  (Maker挂单延迟影响更小)
  fee = (taker_fee - 0.2bps) × 2  (Maker fee折扣)
  adverse = 0.5bps × fill_rate × 2  (Adverse selection)
  total = slippage + liquidity + latency + fee + adverse - spread_revenue

v73p Maker模型已验证 (ERR-102):
  gap 35.23%→6.19% (改善29.04pp)
  核心突破: spread_revenue = position × spread_bps/10000 × 0.5 × fill_rate × 2

来源: _v560_maker_integrated.py (654行) → 提取约 180行纯逻辑
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


# ============================================================
# Maker 成本模型参数 (遵循 sim_live_gap_model.py 范本)
# ============================================================

@dataclass
class MakerCostParams:
    """Maker 成本模型参数

    所有参数来自 v73p/v86 实测验证 (ERR-102)。
    使用方式:
        params = MakerCostParams()
        cost = compute_maker_cost_usd(symbol, order_size, vol_ratio, atr_pct, params)
    """
    maker_fee_discount_bps: float = 0.2     # Maker比Taker便宜0.2bps
    adverse_selection_bps: float = 0.5      # Adverse selection成本
    fill_rate_base: float = 0.85            # 基础成交率85%
    latency_ms_maker: float = 1050.0        # 国内延迟(150+900ms, ERR-094)
    fee_taker_roundtrip_bps: float = 7.0    # Taker往返手续费
    k_slippage: float = 0.1                 # Almgren-Chriss滑点系数


# ============================================================
# 常量: per-symbol 点差与日均成交量
# ============================================================

# per-symbol one-way 点差 (bps), 来源: v559/v479b 实测
SPREAD_ONEWAY_BPS: Dict[str, float] = {
    "BTC_USDT": 1.5, "ETH_USDT": 2.5, "SOL_USDT": 4.0, "BNB_USDT": 5.0,
    "XRP_USDT": 7.0, "ADA_USDT": 8.0, "LTC_USDT": 9.0, "AVAX_USDT": 10.0,
    "DOGE_USDT": 12.0, "LINK_USDT": 12.0, "DOT_USDT": 15.0,
}

# per-symbol 日均成交量 (USD), 来源: v559/v479b 实测
ADV_USD: Dict[str, float] = {
    "BTC_USDT": 5_000_000_000, "ETH_USDT": 3_000_000_000, "SOL_USDT": 1_500_000_000,
    "BNB_USDT": 500_000_000, "XRP_USDT": 800_000_000, "ADA_USDT": 400_000_000,
    "LTC_USDT": 300_000_000, "AVAX_USDT": 250_000_000, "DOGE_USDT": 600_000_000,
    "LINK_USDT": 200_000_000, "DOT_USDT": 150_000_000,
}


# ============================================================
# 纯函数: 符号标准化 + 点差查询
# ============================================================

def _normalize_symbol(symbol: str) -> str:
    """将 symbol 统一为 "BTC_USDT" 格式 (SPREAD_ONEWAY_BPS / ADV_USD 的 key 格式)

    修复原 v560 bug:
      - get_spread_oneway_bps: symbol.replace("_USDT","USDT") 把 "BTC_USDT" → "BTCUSDT" 查不到
      - compute_*_cost: symbol.replace("USDT","_USDT") 把 "BTC_USDT" → "BTC__USDT" 双下划线

    支持输入格式:
      - "BTC_USDT" (带下划线, 直接返回)
      - "BTCUSDT"  (无下划线, 转为 "BTC_USDT")
      - "BTC-USDT" (连字符, 转为 "BTC_USDT")

    Args:
        symbol: 任意格式的交易对名称

    Returns:
        标准化的 "XXX_USDT" 格式
    """
    if not symbol:
        return symbol
    # 统一连字符为下划线
    s = symbol.replace("-USDT", "_USDT").replace("-usdt", "_USDT")
    # 如果是 "BTCUSDT" 格式 (无下划线但以 USDT 结尾)
    if "_USDT" not in s and s.endswith("USDT"):
        s = s[:-4] + "_USDT"
    return s


def get_spread_oneway_bps(symbol: str) -> float:
    """获取 symbol 的 one-way 点差 (bps)

    支持 "BTC_USDT" / "BTCUSDT" / "BTC-USDT" 三种格式。
    未知 symbol 返回默认值 10.0 bps。

    Args:
        symbol: 交易对名称, e.g. "BTC_USDT" 或 "BTCUSDT"

    Returns:
        one-way 点差 (bps)
    """
    sym = _normalize_symbol(symbol)
    return SPREAD_ONEWAY_BPS.get(sym, 10.0)


# ============================================================
# 纯函数: Taker 成本计算 (对比基线)
# ============================================================

def compute_taker_cost_bps(symbol: str, order_size_usd: float,
                            volume_ratio: float, atr_pct: float,
                            params: MakerCostParams = None) -> Dict[str, float]:
    """v559 Taker 4因素成本 (round-trip, bps) — 用于对比

    4因素: fee + spread + slippage + latency
    使用 Almgren-Chriss 平方根模型计算滑点。

    Args:
        symbol: 交易对名称
        order_size_usd: 订单大小 (USD)
        volume_ratio: 当前成交量 / 历史平均成交量 (0.5-1.5)
        atr_pct: ATR百分比 (相对价格), e.g. 0.01 = 1%
        params: MakerCostParams (使用 fee_taker_roundtrip_bps / k_slippage)

    Returns:
        {fee, spread, slippage, latency, total} 全部单位 bps
    """
    if params is None:
        params = MakerCostParams()

    # 1. 手续费 (round-trip)
    fee = params.fee_taker_roundtrip_bps

    # 2. 点差 (round-trip = 2 × one-way)
    spread = 2.0 * get_spread_oneway_bps(symbol)

    # 3. 滑点 (Almgren-Chriss 平方根模型)
    sym_key = _normalize_symbol(symbol)
    adv = ADV_USD.get(sym_key, 300_000_000)
    rate = order_size_usd / adv if adv > 0 else 0
    slip_one = min(params.k_slippage * math.sqrt(max(rate, 0)) * 10000, 50.0) if rate > 0 else 0
    # 流动性乘数: 低成交量时滑点放大
    liq_mult = (
        2.0 if volume_ratio < 0.5 else
        (1.5 if volume_ratio < 0.8 else
         (1.0 if volume_ratio < 1.2 else 0.8))
    )
    slippage = 2.0 * slip_one * liq_mult

    # 4. 延迟成本 (国内 1056ms 物理极限)
    lat_one = min(atr_pct * math.sqrt(max(500 / 3600, 0.001)) * 100, 10.0) if atr_pct > 0 else 0.5
    latency = 2.0 * lat_one

    return {
        "fee": fee,
        "spread": spread,
        "slippage": slippage,
        "latency": latency,
        "total": fee + spread + slippage + latency,
    }


# ============================================================
# 纯函数: Maker 成本计算 (核心)
# ============================================================

def compute_maker_cost_usd(symbol: str, order_size_usd: float,
                            volume_ratio: float, atr_pct: float,
                            params: MakerCostParams = None) -> Dict[str, float]:
    """v560 Maker 成本模型 (round-trip, USD) — spread 从成本变收益

    核心公式 (v73p/v86, ERR-102验证):
      spread_revenue = position × spread_bps/10000 × 0.5 × fill_rate × 2
      slippage = taker_slippage × 0.3  (Maker挂单不影响市场)
      liquidity = taker_liquidity × 0.3  (Maker提供流动性)
      latency = taker_latency × 0.5  (Maker挂单延迟影响更小)
      fee = (taker_fee - 0.2bps) × 2  (Maker fee折扣)
      adverse = 0.5bps × fill_rate × 2  (Adverse selection)
      total = slippage + liquidity + latency + fee + adverse - spread_revenue

    Args:
        symbol: 交易对名称, e.g. "BTC_USDT"
        order_size_usd: 订单大小 (USD)
        volume_ratio: 当前成交量 / 历史平均成交量 (0.5-1.5)
        atr_pct: ATR百分比 (相对价格), e.g. 0.01 = 1%
        params: MakerCostParams

    Returns:
        {spread_revenue, slippage_cost, liquidity_cost, latency_cost,
         fee_cost, adverse_cost, total_cost, fill_rate} 全部单位 USD
        注意: total_cost 可能为负值 (净收益)
    """
    if params is None:
        params = MakerCostParams()

    fill_rate = params.fill_rate_base
    spread_bps = get_spread_oneway_bps(symbol)

    # 1. Spread 收益 (Maker 赚取半 spread, 双向)
    spread_revenue = order_size_usd * spread_bps / 10000.0 * 0.5 * fill_rate * 2

    # 2. 滑点成本 (Taker 的 30%)
    sym_key = _normalize_symbol(symbol)
    adv = ADV_USD.get(sym_key, 300_000_000)
    rate = order_size_usd / adv if adv > 0 else 0
    taker_slip_one = min(params.k_slippage * math.sqrt(max(rate, 0)) * 10000, 50.0) if rate > 0 else 0
    liq_mult = (
        2.0 if volume_ratio < 0.5 else
        (1.5 if volume_ratio < 0.8 else
         (1.0 if volume_ratio < 1.2 else 0.8))
    )
    taker_slippage_bps = 2.0 * taker_slip_one * liq_mult
    slippage_cost = order_size_usd * taker_slippage_bps / 10000.0 * 0.3  # 30%折扣

    # 3. 流动性成本 (Taker 的 30%, 简化: 用 spread 作为流动性代理)
    liquidity_bps = spread_bps * 0.5  # 流动性成本约为半 spread
    liquidity_cost = order_size_usd * liquidity_bps / 10000.0 * 2 * 0.3  # 30%折扣

    # 4. 延迟成本 (Taker 的 50%)
    lat_one = (
        min(atr_pct * math.sqrt(max(params.latency_ms_maker / 3600000, 0.001)) * 10000, 10.0)
        if atr_pct > 0 else 0.5
    )
    taker_latency_bps = 2.0 * lat_one
    latency_cost = order_size_usd * taker_latency_bps / 10000.0 * 0.5  # 50%折扣

    # 5. Fee (Maker 折扣 0.2bps, round-trip)
    maker_fee_bps = params.fee_taker_roundtrip_bps - params.maker_fee_discount_bps  # 7.0 - 0.2 = 6.8bps
    fee_cost = order_size_usd * maker_fee_bps / 10000.0

    # 6. Adverse selection (Maker 被成交时价格已朝不利方向移动)
    adverse_cost = order_size_usd * params.adverse_selection_bps / 10000.0 * 2 * fill_rate

    # 总成本 (可能为负 = 净收益)
    total_cost = (
        slippage_cost + liquidity_cost + latency_cost
        + fee_cost + adverse_cost - spread_revenue
    )

    return {
        "spread_revenue": spread_revenue,
        "slippage_cost": slippage_cost,
        "liquidity_cost": liquidity_cost,
        "latency_cost": latency_cost,
        "fee_cost": fee_cost,
        "adverse_cost": adverse_cost,
        "total_cost": total_cost,  # 可能是负值(净收益)
        "fill_rate": fill_rate,
    }


# ============================================================
# __main__ 守卫自检 (遵循 sim_live_gap_model.py 范本)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("maker_cost_model.py — L6 功能自检")
    print("=" * 70)

    n_pass = 0
    n_total = 0

    # 测试1: MakerCostParams 默认值
    n_total += 1
    params = MakerCostParams()
    assert params.maker_fee_discount_bps == 0.2, "maker_fee_discount_bps默认值错误"
    assert params.adverse_selection_bps == 0.5, "adverse_selection_bps默认值错误"
    assert params.fill_rate_base == 0.85, "fill_rate_base默认值错误"
    assert params.latency_ms_maker == 1050.0, "latency_ms_maker默认值错误"
    assert params.fee_taker_roundtrip_bps == 7.0, "fee_taker_roundtrip_bps默认值错误"
    assert params.k_slippage == 0.1, "k_slippage默认值错误"
    print("✅ 测试1: MakerCostParams 默认值正确")
    n_pass += 1

    # 测试2: get_spread_oneway_bps 符号格式兼容
    n_total += 1
    assert get_spread_oneway_bps("BTC_USDT") == 1.5, "BTC_USDT点差应为1.5"
    assert get_spread_oneway_bps("BTCUSDT") == 1.5, "BTCUSDT点差应为1.5"
    assert get_spread_oneway_bps("ETH_USDT") == 2.5, "ETH_USDT点差应为2.5"
    assert get_spread_oneway_bps("DOT_USDT") == 15.0, "DOT_USDT点差应为15.0"
    assert get_spread_oneway_bps("UNKNOWN") == 10.0, "未知symbol应返回默认10.0"
    print("✅ 测试2: get_spread_oneway_bps 符号格式兼容正确")
    n_pass += 1

    # 测试3: SPREAD_ONEWAY_BPS 覆盖11个symbol
    n_total += 1
    assert len(SPREAD_ONEWAY_BPS) == 11, f"SPREAD_ONEWAY_BPS应有11个symbol, 实际{len(SPREAD_ONEWAY_BPS)}"
    expected_symbols = {"BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT",
                        "ADA_USDT", "LTC_USDT", "AVAX_USDT", "DOGE_USDT", "LINK_USDT", "DOT_USDT"}
    assert set(SPREAD_ONEWAY_BPS.keys()) == expected_symbols, "SPREAD_ONEWAY_BPS symbol集合错误"
    print("✅ 测试3: SPREAD_ONEWAY_BPS 覆盖11个symbol正确")
    n_pass += 1

    # 测试4: ADV_USD 覆盖11个symbol
    n_total += 1
    assert len(ADV_USD) == 11, f"ADV_USD应有11个symbol, 实际{len(ADV_USD)}"
    assert ADV_USD["BTC_USDT"] == 5_000_000_000, "BTC_USDT ADV应为5B"
    assert ADV_USD["DOT_USDT"] == 150_000_000, "DOT_USDT ADV应为150M"
    print("✅ 测试4: ADV_USD 覆盖11个symbol正确")
    n_pass += 1

    # 测试5: compute_taker_cost_bps BTC_USDT 基本计算
    n_total += 1
    taker_btc = compute_taker_cost_bps("BTC_USDT", order_size_usd=1000.0,
                                        volume_ratio=1.0, atr_pct=0.01)
    assert taker_btc["fee"] == 7.0, "Taker fee应为7.0bps"
    assert taker_btc["spread"] == 3.0, "BTC round-trip spread应为3.0bps (2×1.5)"
    assert taker_btc["total"] > 0, "Taker total应为正值"
    assert "slippage" in taker_btc and "latency" in taker_btc, "Taker成本应包含slippage和latency"
    print(f"✅ 测试5: compute_taker_cost_bps BTC_USDT total={taker_btc['total']:.2f}bps")
    n_pass += 1

    # 测试6: compute_maker_cost_usd BTC_USDT 核心计算 (spread_revenue > 0)
    n_total += 1
    maker_btc = compute_maker_cost_usd("BTC_USDT", order_size_usd=1000.0,
                                        volume_ratio=1.0, atr_pct=0.01)
    assert maker_btc["spread_revenue"] > 0, "Maker spread_revenue应>0 (spread从成本变收益)"
    assert maker_btc["fee_cost"] > 0, "Maker fee_cost应>0"
    assert maker_btc["adverse_cost"] > 0, "Maker adverse_cost应>0"
    assert maker_btc["fill_rate"] == 0.85, "fill_rate应为0.85"
    # 验证 total_cost 可能为负 (Maker净收益)
    print(f"✅ 测试6: BTC_USDT Maker spread_revenue=${maker_btc['spread_revenue']:.4f}, "
          f"total_cost=${maker_btc['total_cost']:.4f}")
    n_pass += 1

    # 测试7: Maker 成本 < Taker 成本 (核心价值: gap 35%→6%)
    n_total += 1
    taker_total_usd = taker_btc["total"] / 10000 * 1000.0  # bps → USD
    maker_total_usd = maker_btc["total_cost"]
    assert maker_total_usd < taker_total_usd, \
        f"Maker成本(${maker_total_usd:.4f})应<Taker成本(${taker_total_usd:.4f})"
    savings_pct = (taker_total_usd - maker_total_usd) / taker_total_usd * 100
    print(f"✅ 测试7: Maker成本节省 {savings_pct:.1f}% (Taker ${taker_total_usd:.4f} → "
          f"Maker ${maker_total_usd:.4f})")
    n_pass += 1

    # 测试8: 11个symbol全覆盖测试 (无异常)
    n_total += 1
    test_symbols = list(SPREAD_ONEWAY_BPS.keys())
    for sym in test_symbols:
        t = compute_taker_cost_bps(sym, 1000.0, 1.0, 0.01)
        m = compute_maker_cost_usd(sym, 1000.0, 1.0, 0.01)
        assert t["total"] > 0, f"{sym} Taker total应>0"
        assert m["spread_revenue"] > 0, f"{sym} Maker spread_revenue应>0"
    print(f"✅ 测试8: 11个symbol全覆盖测试通过 (无异常)")
    n_pass += 1

    # 测试9: 自定义 params 参数注入
    n_total += 1
    custom_params = MakerCostParams(
        maker_fee_discount_bps=0.5,
        fill_rate_base=0.90,
        latency_ms_maker=500.0,
    )
    maker_custom = compute_maker_cost_usd("BTC_USDT", 1000.0, 1.0, 0.01, custom_params)
    assert maker_custom["fill_rate"] == 0.90, "自定义fill_rate未生效"
    # 更高 fill_rate → 更高 spread_revenue
    assert maker_custom["spread_revenue"] > maker_btc["spread_revenue"], \
        "fill_rate=0.90的spread_revenue应>fill_rate=0.85"
    print(f"✅ 测试9: 自定义params注入正确 (fill_rate=0.90, "
          f"spread_revenue=${maker_custom['spread_revenue']:.4f})")
    n_pass += 1

    # 测试10: 边界条件 (order_size=0 / volume_ratio=0)
    n_total += 1
    edge1 = compute_maker_cost_usd("BTC_USDT", 0.0, 1.0, 0.01)
    assert edge1["spread_revenue"] == 0.0, "order_size=0时spread_revenue应为0"
    assert edge1["total_cost"] == 0.0, "order_size=0时total_cost应为0"
    edge2 = compute_taker_cost_bps("BTC_USDT", 1000.0, 0.0, 0.0)
    assert edge2["total"] > 0, "volume_ratio=0/atr_pct=0时Taker total仍应>0 (fee+spread)"
    print(f"✅ 测试10: 边界条件处理正确 (order_size=0 → total_cost=0)")
    n_pass += 1

    print("\n" + "=" * 70)
    print(f"L6 自检结果: {n_pass}/{n_total} PASS")
    if n_pass == n_total:
        print("🎉 全部通过! maker_cost_model.py 功能可用")
    else:
        print(f"❌ {n_total - n_pass} 项失败")
    print("=" * 70)
