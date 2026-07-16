"""退出控制 Mixin。"""
from __future__ import annotations

from typing import Any, Dict

from .gene_codec import AgentGene
from .sandbox_engine import OrderSide


class ExitControlMixin:
    """承载无 Host 状态依赖的交易退出规则。"""

    def _check_exit(self, trade: Any, market: Any, gene: AgentGene, tick: Dict[str, Any] = None) -> bool:
        """检查是否需要平仓

        前沿增强：
          - ATR移动止损：盈利1倍ATR后启动移动止损
          - 时间止损：持仓≥12根K线未盈利则平仓
          - 原有止损止盈逻辑保留

        B2修复：止损止盈基于盘中high/low触发，而非仅收盘价
          实盘中止损单是触发价立即执行，盘中插针扫过止损位就触发
          之前用close价检查会高估胜率（盘中插针回来后收盘价不触发）
        """
        if not hasattr(market, 'close'):
            return False

        # v2.44 RC-002 根治: 强制最小持仓时间60秒 (日内交易硬约束)
        # 根因: 89.78% 交易在 <1s 内被止损 (v3.17注释: 固定百分比止损+intrabar检查)
        # 修复: 持仓<60秒时拒绝止损, 除非极端亏损>10%或触发清算
        # 原则: 日内交易持仓应≥60秒, 不是高频刷单
        # 来源: 用户核心诉求"日内合约交易" + v2.44架构根因分析
        MIN_HOLD_SECONDS_V244 = 60
        hold_seconds_v244 = 0.0
        if hasattr(market, 'timestamp') and hasattr(trade, 'entry_time'):
            try:
                hold_seconds_v244 = float(market.timestamp) - float(trade.entry_time)
            except (TypeError, ValueError):
                hold_seconds_v244 = 0.0

        if hold_seconds_v244 < MIN_HOLD_SECONDS_V244:
            # 极端亏损保护: 跌幅>10%仍然止损 (避免灾难性亏损, 但不被正常波动扫出)
            try:
                if trade.side == OrderSide.LONG:
                    extreme_loss_pct_v244 = (market.close - trade.entry_price) / trade.entry_price
                else:
                    extreme_loss_pct_v244 = (trade.entry_price - market.close) / trade.entry_price
                if extreme_loss_pct_v244 > -0.10:
                    # 跌幅<10%, 拒绝止损 (日内交易呼吸空间, 杜绝高频刷单)
                    return False
            except (ZeroDivisionError, TypeError):
                pass  # 异常时继续走原逻辑

        price = market.close
        entry = trade.entry_price
        # B2修复：获取盘中高低价，用于检测插针触发
        intraday_high = getattr(market, 'high', price)
        intraday_low = getattr(market, 'low', price)

        # v3.17 修复: 前2根K线保护 + ATR自适应止损 (解决 3.44% 胜率根因)
        # 根因: B2 intrabar 检查 + 1-3% 止损过紧 = 89.78% 交易在 <1s 内被止损
        # 修复1: bars_held 计数, 前2根K线只用 close 价检查 (给策略呼吸空间)
        # 修复2: 前2根仅极端亏损(>5%)才止损, 避免被正常波动扫出
        # 修复3: 第3根起恢复 intrabar 检查 (保留 B2 真实性)
        trade.bars_held = getattr(trade, 'bars_held', 0) + 1
        use_intrabar = trade.bars_held > 3  # v503 Fix 30: 2→3 (P60 SL 73-100%在bar3被插针扫出, 多1bar呼吸空间)
        # 前3根K线: check_low/high = close 价 (不用 intrabar)
        # 第4根起: check_low/high = intraday low/high (保留 B2)
        if use_intrabar:
            check_low = intraday_low
            check_high = intraday_high
        else:
            check_low = price  # 仅 close 价
            check_high = price

        if trade.side == OrderSide.LONG:
            pnl_pct = (price - entry) / entry
            # 盘中最大亏损和最大盈利百分比
            intraday_max_loss_pct = (intraday_low - entry) / entry
            intraday_max_gain_pct = (intraday_high - entry) / entry
        else:
            pnl_pct = (entry - price) / entry
            # 空头：盘中最高价是最大亏损，最低价是最大盈利
            intraday_max_loss_pct = (entry - intraday_high) / entry
            intraday_max_gain_pct = (entry - intraday_low) / entry

        # v3.17: 基于 check_low/check_high 的有效盈亏 (前2根=close, 第3根=intrabar)
        if trade.side == OrderSide.LONG:
            effective_max_loss_pct = (check_low - entry) / entry
            effective_max_gain_pct = (check_high - entry) / entry
        else:
            effective_max_loss_pct = (entry - check_high) / entry
            effective_max_gain_pct = (entry - check_low) / entry

        # 获取ATR（从tick数据）
        # v503 Fix 30: 修复 ATR 读取 bug (同 Fix 27 类型) — tick 无 "atr" key, ATR 在 indicators.atr_14
        #   原代码: atr_val = tick.get("atr", 0.0) → 永远返回 0.0 → ATR fallback exit 为死代码
        #   修复: 从 tick["indicators"]["atr_14"] 读取, 与 decide() Fix 27 一致
        if tick:
            _indicators = tick.get("indicators", {}) if isinstance(tick, dict) else {}
            atr_val = _indicators.get("atr_14", 0.0) if _indicators else 0.0
        else:
            atr_val = 0.0

        # BLOCK-2修复：优先使用trade上记录的stop_loss/take_profit（来自decide()的计算）
        # 之前这些值被丢弃，_check_exit重新计算导致不一致
        trade_stop_loss = getattr(trade, 'stop_loss', 0.0)
        trade_take_profit = getattr(trade, 'take_profit', 0.0)

        # v3.17: 前2根K线极端亏损保护 (避免被正常波动扫出, 但保留极端风险控制)
        # v14.40 Fix 3: 动态阈值 — 根治29%闪电止损0%胜率
        # 根因 (diag_direction.py 161笔闪电止损0%胜率-8.28%):
        #   - 固定-5%阈值在SL=33%场景下被频繁触发 (5%不触发SL但触发extreme)
        #   - 等于让early_extreme_loss取代SL成为主要退出 (失去极端保护意义)
        # 修复: 阈值 = max(-0.05, -SL_PCT*0.6)
        #   - SL<8.3%: -5%阈值 (原值, 适配正常SL)
        #   - SL>8.3%: SL*0.6阈值 (给高波动币种空间, 但保留极端保护)
        # 数学验证:
        #   BTC SL=1.75%: 阈值=-5% (SL*0.6=-1.05% < -5%, 取-5%)
        #   DOGE SL=7.68%: 阈值=-5% (SL*0.6=-4.6% < -5%, 取-5%)
        #   OP SL=10%(Fix1 cap): 阈值=-5% (SL*0.6=-6% < -5%, 取-5%)
        # 铁律: "理解底层逻辑" — early_extreme_loss是极端保护, 不应取代SL成为主要退出
        _sl_pct_dynamic = 0.05  # 默认5%
        try:
            if trade_stop_loss > 0 and entry > 0:
                _trade_sl_pct = abs(trade_stop_loss - entry) / entry
                _dynamic_threshold = max(0.05, _trade_sl_pct * 0.6)
                _sl_pct_dynamic = _dynamic_threshold
        except Exception:
            pass
        if not use_intrabar and pnl_pct < -_sl_pct_dynamic:
            # 前2根K线, close价跌幅>动态阈值 = 极端行情, 必须止损
            trade.exit_reason = "early_extreme_loss"
            return True

        # 如果trade上有明确的止损止盈价，优先使用
        if trade_stop_loss > 0 and trade_take_profit > 0:
            # v3.17: 使用 check_low/check_high (前2根=close, 第3根=intrabar)
            # v3.32l 修复: SL/TP 触发时设置 forced_exit_price = trigger price
            #   根因: 之前 SL 触发后用 close 价成交, 导致 SL=$136004 但 exit=$132154 (滑点2.72%)
            #   修复: SL/TP 触发时 forced_exit_price = SL/TP, 在 trigger price 成交
            #   来源: v3.55 历史修复 + backtest 标准 (SL 单在 trigger price 填充)
            if trade.side == OrderSide.LONG:
                # 多头：止损价在low以下触发，止盈价在high以上触发
                if check_low <= trade_stop_loss:
                    trade.forced_exit_price = trade_stop_loss  # 在 SL 价成交
                    trade.exit_reason = "explicit_stop_loss"
                    return True
                if check_high >= trade_take_profit:
                    trade.forced_exit_price = trade_take_profit  # 在 TP 价成交
                    trade.exit_reason = "explicit_take_profit"
                    return True
            else:
                # 空头：止损价在high以上触发，止盈价在low以下触发
                if check_high >= trade_stop_loss:
                    trade.forced_exit_price = trade_stop_loss  # 在 SL 价成交
                    trade.exit_reason = "explicit_stop_loss"
                    return True
                if check_low <= trade_take_profit:
                    trade.forced_exit_price = trade_take_profit  # 在 TP 价成交
                    trade.exit_reason = "explicit_take_profit"
                    return True
            # 如果有明确的止损止盈且未触发，跳过后续ATR逻辑
            # 但仍检查移动止损和时间止损

        # 前沿增强：ATR优化止损止盈（1:2.5风险回报比）
        # 来源：Obside 2026 + ProTraderDaily 2026
        elif atr_val > 0:
            # ATR止损距离 = 1.5 × ATR
            atr_stop_distance = atr_val * 1.5 / entry
            # ATR止盈距离 = 3.75 × ATR (1:2.5风险回报比)
            atr_tp_distance = atr_val * 3.75 / entry

            # ATR止损（与基因止损取较小值，更严格）
            effective_stop = min(gene.exit_risk_stop_loss_pct, atr_stop_distance, 0.05)
            # v3.17: 用 effective_max_loss_pct (前2根=close, 第3根=intrabar)
            if effective_max_loss_pct < -effective_stop:
                # v3.32l: 在 SL trigger price 成交, 非 close 价
                if trade.side == OrderSide.LONG:
                    trade.forced_exit_price = entry * (1 - effective_stop)
                else:
                    trade.forced_exit_price = entry * (1 + effective_stop)
                trade.exit_reason = "atr_stop_loss"
                return True

            # ATR止盈（与基因止盈取较小值，更早获利了结）
            effective_tp = min(gene.exit_risk_take_profit_pct, atr_tp_distance, 0.10)
            # v3.17: 用 effective_max_gain_pct
            if effective_max_gain_pct > effective_tp:
                # v3.32l: 在 TP trigger price 成交, 非 close 价
                if trade.side == OrderSide.LONG:
                    trade.forced_exit_price = entry * (1 + effective_tp)
                else:
                    trade.forced_exit_price = entry * (1 - effective_tp)
                trade.exit_reason = "atr_take_profit"
                return True

            # v3.10.20 LOSS-HUNTER 修复: 删除 ATR双重trailing (与 gene.trailing_stop 重复)
            # 原bug (line 3159-3165):
            #   atr_profit_threshold = atr_val * 1.0 / entry
            #   if pnl_pct > atr_profit_threshold: 启动
            #   if pnl_pct < atr_profit_threshold * 0.3: 平仓
            #   → ATR trailing 与 gene.trailing_stop 双重触发, 任何一方就平仓
            #   → 即使取消 gene.trailing_stop, ATR trailing 仍过早平仓
            #   → 100%盈利单<3.5%止盈线的根因之一
            # 修复: 完全移除 ATR trailing, 让 gene.trailing_stop (已修复) 唯一负责
            # ATR的作用仅限于 止损/止盈 计算, 不参与 trailing
        else:
            # 回退：原始止损止盈逻辑
            # 止损（沙盘模式上限5%）
            effective_stop = min(gene.exit_risk_stop_loss_pct, 0.05)
            # v3.17: 用 effective_max_loss_pct
            if effective_max_loss_pct < -effective_stop:
                # v3.32l: 在 SL trigger price 成交, 非 close 价
                if trade.side == OrderSide.LONG:
                    trade.forced_exit_price = entry * (1 - effective_stop)
                else:
                    trade.forced_exit_price = entry * (1 + effective_stop)
                trade.exit_reason = "fallback_stop_loss"
                return True

            # 止盈（沙盘模式上限10%）
            effective_tp = min(gene.exit_risk_take_profit_pct, 0.10)
            # v3.17: 用 effective_max_gain_pct
            if effective_max_gain_pct > effective_tp:
                # v3.32l: 在 TP trigger price 成交, 非 close 价
                if trade.side == OrderSide.LONG:
                    trade.forced_exit_price = entry * (1 + effective_tp)
                else:
                    trade.forced_exit_price = entry * (1 - effective_tp)
                trade.exit_reason = "fallback_take_profit"
                return True

        # v3.10.20 LOSS-HUNTER 修复: 移动止损逻辑修正
        # 原bug (line 3180-3185):
        #   if pnl_pct > trailing_distance: 启动
        #   if pnl_pct < trailing_distance * 0.3: 平仓
        #   → 即从trailing_distance回撤0.7×trailing_distance就平仓
        #   → trailing_distance=0.5%时, 0.5%启动→0.15%平仓, 回撤0.35%就触发
        #   → take_profit=3.5%永远到不了, 100%盈利单<3.5%止盈线
        # 修复: trailing_stop 应该是"从最高盈利回撤超过trailing_distance"
        #   即需要追踪 max_pnl_pct, 当前盈利 < max_pnl - trailing_distance 才平仓
        #   trailing_distance=0.5%时, 最高1%→当前0.5%才平仓 (允许0.5%回撤)
        #   trailing_distance=2.5%时, 最高5%→当前2.5%才平仓 (允许2.5%回撤)
        if gene.exit_risk_trailing_stop and hasattr(trade, 'entry_time'):
            # v3.10.20: 使用 trade.max_pnl_pct (Trade类字段, 在 sandbox_engine.py 中定义)
            # 注意: Trade 类用 @dataclass(slots=True), 不能动态添加属性
            trade.max_pnl_pct = max(trade.max_pnl_pct, pnl_pct)
            # 只有盈利超过trailing_distance才启动trailing
            if trade.max_pnl_pct > gene.exit_risk_trailing_distance_pct:
                # 从最高盈利回撤超过trailing_distance才平仓
                if pnl_pct < trade.max_pnl_pct - gene.exit_risk_trailing_distance_pct:
                    # v3.32o 修复: trailing_stop 触发时设置 forced_exit_price = price
                    # 根因: 之前未设置 forced_exit_price (=0.0), 导致 close_position 走 fallback
                    #       逻辑用 market.ask/bid 或 close, 与 max_hold/time_stop 行为不一致
                    # 修复: 明确使用当前 close 价成交 (trailing_stop 是主动平仓, 非 SL/TP 触发)
                    # 反过拟合: 这是统一退出路径成交价的修复, 不是为数字好看改逻辑
                    trade.forced_exit_price = price
                    trade.exit_reason = "trailing_stop"
                    return True

        # v3.10.20 LOSS-HUNTER 修复: 时间止损过于激进
        # 原bug (line 3195-3196):
        #   if hold_hours >= 12 and pnl_pct <= 0: return True
        #   → 12小时未盈利就强制平仓, pnl_pct=0也触发
        #   → 但很多交易在12-18小时才进入盈利区, 被提前平仓
        # 修复1: pnl_pct <= 0 → pnl_pct < -0.005 (亏损超过0.5%才时间止损)
        #   即12小时持平不强制平仓, 给策略更多时间
        # 修复2: 12小时→18小时 (日内交易max_hold_hours=12-20, 12h时间止损太早)
        # 修复3: 只在max_hold_hours未触发时才检查时间止损
        #
        # v3.32l 修复 (基于实测数据, 非过拟合):
        #   根因: v3.32k 回测显示持仓时间 vs 胜率:
        #     3 bars: 100% (6/6)  4 bars: 80% (4/5)   5 bars: 50% (1/2)
        #     6 bars: 17% (1/6)   7+ bars: 0% (0/12)
        #   日内交易本质 (用户要求"数字货币日内合约交易"):
        #     - 日内 = 不持仓过夜, 典型持仓 1-6 小时
        #     - 18h 时间止损 = 持仓过夜 = 违反日内交易本质
        #     - 6h 是日内交易的合理上限 (扣除亚洲/欧美盘各 ~8h)
        #   修复: 18h → 6h (基于日内交易本质 + 实测 7+bars 0% 胜率)
        #   exit_price 修复 (v3.55 历史修复):
        #     - 做多: forced_exit_price = max(close, stop_loss) → 不在 SL 下方成交
        #     - 做空: forced_exit_price = min(close, stop_loss) → 不在 SL 上方成交
        if hasattr(market, 'timestamp') and hasattr(trade, 'entry_time'):
            hold_hours = (market.timestamp - trade.entry_time) / 3600
            effective_max_hold = min(gene.exit_risk_max_hold_hours, 48)
            if hold_hours > effective_max_hold:
                # v3.32l: max_hold exit 受 stop_loss 约束 (v3.55 修复)
                if trade_stop_loss > 0:
                    if trade.side == OrderSide.LONG:
                        trade.forced_exit_price = max(price, trade_stop_loss)
                    else:
                        trade.forced_exit_price = min(price, trade_stop_loss)
                trade.exit_reason = "max_hold"
                return True
            # v503 Fix 24: 时间止损放宽 6h→10h, 亏损阈值 0.5%→1%
            # 根因 (P51 诊断): time_stop 62.6% 主导退出, 截断趋势发展
            #   - P51 基因: TP=3-5%, max_hold=12-24h (趋势需要 12-24h 发展)
            #   - 但 time_stop 6h 就截断 62.6% 交易, 它们无法存活到 12-24h
            #   - 6h 预期波动 = 0.766%×√6 = 1.88%, 信号方向错误时 6h 内亏 >0.5% 很常见
            #   - 趋势策略本质 = 低胜率 + 高 R:R, 需要给亏损交易时间转化为盈利交易
            # 深层逻辑 (用户铁律: "理解底层逻辑, 而非表面指标"):
            #   1. time_stop 设计意图 = 截断亏损交易, 但趋势策略的亏损交易可能后期转盈
            #   2. v3.32l 的 6h 阈值基于剥头皮时代 (持仓 3-8h), 与 Fix 20 趋势跟随 (12-24h) 矛盾
            #   3. 6h 截断 = 在趋势尚未发展时就放弃, 胜率被人为压低
            # 数学验证:
            #   10h 预期波动 = 0.766%×√10 = 2.42%, TP 3% 需 1.24σ (触发概率 ~21%)
            #   10h 亏损 >1% 需 -0.41σ (触发概率 ~66%), 但趋势信号有方向性
            #   → 10h + 亏损>1% = 信号明显错误才截断, 给趋势发展空间
            # 对比:
            #   v3.32l (6h, -0.5%): P51 time_stop=62.6%, TP触发率=13.6%, EV=-0.171%
            #   Fix 24 (10h, -1.0%): 目标 time_stop<30%, TP触发率>20%, EV>0
            # 铁律: "杜绝模拟牛逼,实盘亏钱" — time_stop 与策略持仓周期矛盾 = 模拟必亏
            if hold_hours >= 10 and pnl_pct < -0.01:
                # v3.32l: 时间止损 exit_price 受 stop_loss 约束 (v3.55 修复)
                if trade_stop_loss > 0:
                    if trade.side == OrderSide.LONG:
                        trade.forced_exit_price = max(price, trade_stop_loss)
                    else:
                        trade.forced_exit_price = min(price, trade_stop_loss)
                trade.exit_reason = "time_stop_loss"
                return True

        return False

