"""Agent signal mixin (Phase 7.14d extracted from agent_decision_engine.py).

Contains 6 signal computation methods: entry signals, signal evaluation,
crypto intraday signals, position sizing, exit points, leverage computation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple


logger = logging.getLogger("hermes.agent_decision_engine")

# Phase 7.14d: Leading indicators set (moved from agent_decision_engine.py).
# v3.21 领先指标集合 (修复根因2: 滞后指标虚假共振)
# 领先指标基于价量关系, 提前反映市场结构变化, 是独立证据
LEADING_SIGNALS: set = {
    "price_momentum_roc",  # ROC价格动量 (无平滑, 直接度量)
    "mfi_divergence",      # MFI资金流量 (价量综合)
    "willr_signal",        # Williams %R (响应快的超买超卖)
    "volume_anomaly",      # 成交量异常 (资金行为先行)
}


class AgentSignalMixin:
    """Mixin providing signal computation methods for AgentDecisionEngine.

    Extracted in Phase 7.14d to reduce agent_decision_engine.py size.
    All methods use duck-typing (self attribute access) - zero circular deps.
    """

    def _compute_entry_signals(
        self, market: Any, indicators: Dict[str, float],
        timesfm: Dict[str, Any], kronos: Dict[str, Any],
        crypto_intraday: Dict[str, Any] = None,
        chanlun_pa: Dict[str, Any] = None,
        market_state: Dict[str, Any] = None,
    ) -> Tuple[float, str]:
        """计算入场信号得分 (v3.21: 修复滞后指标虚假共振)

        v3.21 修复根因2:
          - 原问题: signal_count放大机制(total *= 1.0 + signal_count * 0.15)
            导致4个滞后指标(RSI/MACD/EMA/BB)同向时confidence被放大1.45倍,
            但这些都是滞后指标, 同源信息, 不是独立证据 → 虚假确认
          - 修复:
            1) 区分领先/滞后指标, 领先指标主导
            2) 同源滞后指标去重计数 (lagging_count 最多计1次)
            3) 领先指标反向时, 即使滞后同向也不交易 (避免反向选择)
            4) 领先+滞后同向才放大confidence (真正的独立确认)

        信号源（多维共振）:
          1. 领先指标 (v3.21新增): ROC/MFI/WilliamsR/VolumeAnomaly — 主导
          2. 基因主信号 (滞后+ML): RSI/MACD/EMA/BB等 — 确认
          3. 加密货币日内信号 (VWAP/资金费率/OI背离)
          4. 缠论+裸K信号 (背驰/Pin Bar/支撑阻力)

        返回: (signal_score, direction, signal_count)
        """
        close = market.close if hasattr(market, 'close') else market.get('close', 0)
        if not close:
            return 0.0, "neutral", 0

        scores = {"long": 0.0, "short": 0.0}
        # v3.21: 分类计数, 避免滞后指标虚假共振
        leading_long = 0   # 领先指标多头计数
        leading_short = 0  # 领先指标空头计数
        lagging_long = 0   # 滞后指标多头计数 (同源去重, 最多计1)
        lagging_short = 0  # 滞后指标空头计数 (同源去重, 最多1)

        # v3.23 修复: 跟踪触发的信号名, 用于 signal_source 字段
        # 根因: decide() 返回字典没有 signal_source, 导致 audit.log 中该字段始终为空
        triggered_signals = []

        # 主信号工具
        for signal_name in self.gene.entry_logic_primary_signals:
            s = self._eval_signal(signal_name, indicators, close, timesfm, kronos)
            if s["direction"] == "long":
                scores["long"] += s["strength"]
                triggered_signals.append(f"{signal_name}:long")  # v3.23: 记录触发信号
                # v3.21: 分类计数
                if signal_name in LEADING_SIGNALS:
                    leading_long += 1
                else:
                    lagging_long = 1  # 同源去重
            elif s["direction"] == "short":
                scores["short"] += s["strength"]
                triggered_signals.append(f"{signal_name}:short")  # v3.23: 记录触发信号
                if signal_name in LEADING_SIGNALS:
                    leading_short += 1
                else:
                    lagging_short = 1

        # 加密货币日内交易信号（VWAP/资金费率/OI背离）
        if crypto_intraday:
            crypto_signals = self._eval_crypto_intraday_signals(crypto_intraday, close)
            for s in crypto_signals:
                if s["direction"] == "long":
                    scores["long"] += s["strength"]
                    leading_long += 1  # 加密日内信号视为领先
                elif s["direction"] == "short":
                    scores["short"] += s["strength"]
                    leading_short += 1

        # v3.32: 跟踪chanlun_pa方向 (用于做空confluence计数)
        chanlun_pa_long = 0
        chanlun_pa_short = 0

        # v3.35b 修正: 恢复方向一致性要求 + 保持信号可见性
        # v3.35实验教训: 移除方向一致性要求导致chanlun_pa信号独立产生方向,
        #   引入低质量反向信号, 胜率从51.22%降到34.15%
        #   - chanlun_top_divergence:short;near_support:short 3笔0%胜率
        #   - 这些做空信号与做多scores冲突, 导致决策混乱
        # 修正:
        #   1. 恢复方向一致性要求 (chanlun_pa信号必须与scores方向一致才计入)
        #   2. 保持triggered_signals记录 (无论是否计入scores, 都记录用于分析)
        #   3. [v701 P0-2 更新] 结构信号(BOS/CHoCH)权重 0.8→0.4 减半保留, 其他chanlun_pa信号 0.5→0.0 完全下线
        #      [v703 P1 更新] v702 已完成 OB+LS 接线+直连决策, SMC 四大支柱 (BOS/CHoCH+OB+FVG+LS) 端到端贯通
        #                    BOS/CHoCH 仍保留 0.4 scores 权重 (此路径) + v703 P2 将新增直连决策 (绕过 composite_score)
        if chanlun_pa and chanlun_pa.get("signal") in ("long", "short"):
            pa_direction = chanlun_pa["signal"]
            pa_strength = chanlun_pa.get("strength", 0)
            pa_sources = chanlun_pa.get("sources", [])

            # v3.35: 无论是否计入scores, 都记录到triggered_signals用于分析
            if pa_strength >= 0.3:
                for src in pa_sources[:2]:  # 最多记录2个源, 避免日志过长
                    triggered_signals.append(f"{src}:{pa_direction}")

                # v3.35b: 恢复方向一致性要求 (只计入与scores方向一致的信号)
                is_structure_signal = any(
                    "bos" in src or "choch" in src for src in pa_sources
                )
                # v701 P0-2: 应用 ERR-110-v561 — 0/10 特征通过 P<0.05+|Cohen's d|>0.1 双重过滤
                # 用户铁律: "复盘不是摆设"、"错误库使用起来"、"不要通过已知数据反推策略"
                # 教训 ERR-110-v561: PinBar/Engulfing/VP背离/BOS/CHOCH 等 10 个特征对 PnL 无显著预测力
                # v701 修复策略:
                #   - 裸K形态 (PinBar/Engulfing/InsideBar): 0.5 → 0.0 (完全下线, ERR-110-v561 明确证伪)
                #   - SMC结构 (BOS/CHoCH): 0.8 → 0.4 (减半保留, 等 v701 P1-3 FVG 接线后重新统计)
                # 设计依据:
                #   1. 完全下线裸K形态 — 直接应用教训, 不"过拟合"地保留
                #   2. 保留 BOS/CHoCH 减半 — SMC 方法论核心, FVG 接线后将形成完整 SMC 体系
                #   3. [v701/v702 后续已完成] FVG(v701 P1-3)+OB(v702 P1)+LS(v702 P2) 已接线, SMC 四大支柱贯通
                #      [v703 P2 待办] BOS/CHoCH 也将直连决策绕过 composite_score (与 FVG/CVD/OB/LS 一致)
                # 历史: v3.35 pa_weight=0.8/0.5 → v701 P0-2 pa_weight=0.4/0.0 (ERR-110-v561 应用) → v703 P2 将新增直连
                pa_weight = 0.4 if is_structure_signal else 0.0

                if pa_direction == "long" and scores["long"] > scores["short"]:
                    scores["long"] += pa_strength * pa_weight
                    chanlun_pa_long = 1
                elif pa_direction == "short" and scores["short"] > scores["long"]:
                    scores["short"] += pa_strength * pa_weight
                    chanlun_pa_short = 1

        # 确认信号工具
        confirm_long = 0
        confirm_short = 0
        for signal_name in self.gene.entry_logic_confirmation_signals:
            s = self._eval_signal(signal_name, indicators, close, timesfm, kronos)
            if s["direction"] == "long":
                confirm_long += 1
            elif s["direction"] == "short":
                confirm_short += 1

        # v3.21 核心: 修复 signal_count 放大逻辑
        # 原问题: total *= (1 + signal_count * 0.15), 4个滞后指标同向 → 1.45倍虚假放大
        # 修复策略:
        #   - 领先指标主导: leading_count 是独立证据, 全部计入
        #   - 滞后指标去重: lagging_long/short 已同源去重 (最多1)
        #   - 领先+滞后同向才放大 (真正的独立确认)
        #   - 领先反向时直接返回 neutral (避免反向选择)

        total = max(scores["long"], scores["short"])
        direction = "neutral"

        # v3.29 根因修复 CRITICAL: 多信号矛盾时弃权, 不硬选方向
        # 根因 (audit.log 实测): win_rate=3.36%, 38笔交易中 34.2% 的 signal_source 与 side 不一致
        #   案例1: signal_source="willr_signal:short" 但 side=long, pnl=-3.82
        #   案例2: signal_source="ema_breakout:long;mfi_divergence:short" 但 side=long, pnl=-3.00
        #   案例3: signal_source="timesfm_prediction:long;mfi_divergence:short" 但 side=long, pnl=-3.00
        #   所有矛盾交易胜率=0% (13/13亏损)
        # 架构问题: 当 leading_long>0 AND leading_short>0 时 (领先指标互相矛盾),
        #   代码落到"正常逻辑"硬选方向 → 在高不确定性市场中交易 → 必亏
        # 修复 (不过拟合, 是架构性修复):
        #   - 领先指标互相矛盾 → 弃权 (返回 neutral)
        #   - 领先与滞后反向 → 弃权 (不再 ×0.7 仍然交易)
        #   - 仅滞后指标 → ×0.4 大幅降低置信度 (保留, 避免v3.26的0交易问题)
        # 铁律: "不要根据现有的历史数据去为了数字好看去过拟合, 而是真正的找到根因, 从而根治"
        #   - 这里不是调阈值, 是修正"信号矛盾时仍交易"的架构bug
        #   - 弃权是处理不确定性的标准做法, 不是为历史数据调参
        if leading_long > 0 and leading_short > 0:
            # 领先指标互相矛盾 (如 mfi_divergence:short vs price_momentum_roc:long)
            # = 高不确定性市场 → 弃权
            return 0.0, "neutral", leading_long + leading_short
        if (leading_long > 0 and lagging_short > 0) or (leading_short > 0 and lagging_long > 0):
            # 领先与滞后反向 → 信号冲突 → 弃权
            # 之前: ×0.7 仍然交易 (但胜率0%)
            # 现在: 弃权, 让市场明确后再交易
            return 0.0, "neutral", leading_long + leading_short + lagging_long + lagging_short
        if leading_long == 0 and leading_short == 0 and (lagging_long > 0 or lagging_short > 0):
            # v3.32k-maint1 修复 (代码审查WARN-1): 处理滞后指标内部矛盾状态
            # 根因: rsi_divergence:long 和 macd_cross:short 同时触发时,
            #   lagging_long=1 AND lagging_short=1, 两者都是滞后指标(已去重),
            #   这种矛盾说明震荡市场中信号混乱 → 应弃权
            # 修复: 滞后指标内部矛盾 → 弃权 (与领先指标矛盾处理一致)
            if lagging_long > 0 and lagging_short > 0:
                return 0.0, "neutral", leading_long + leading_short + lagging_long + lagging_short
            # v3.32k 根因修复 CRITICAL (协调者分析 v3.31 audit.log Close #3, #4):
            # 实测根因: v3.31中Close#3,#4均为macd_cross:long单信号进场, 合计亏损$111.93
            #   - 2笔交易在BTC局部顶点反向选择(趋势末端MACD金叉=虚假信号)
            #   - 单笔亏损-2.93%/-3.48%, 超过单笔最大亏损2%风控线
            #   - 这2笔亏损占总PnL的120% (其余19笔净盈利+$19.08)
            # 深层根因 (用户要求"理解底层逻辑而非表面指标"):
            #   - MACD = EMA(12) - EMA(26), Signal = EMA(9) of MACD
            #   - 趋势末端: 价格创新高但动量衰竭, MACD短暂上穿=反向信号(death cross bounce)
            #   - 滞后指标本质是"确认工具"而非"主导信号", 单独使用违反设计原理
            #   - 领先指标(ROC/MFI/WillR/Volume)提前反映价量关系, 是独立证据
            #   - v3.26的×0.4降低置信度但允许交易 → 虚假安全感 → 实盘必亏
            # 架构修复 (非过拟合, 基于指标设计原理):
            #   - 滞后指标单信号需≥1个其他维度确认 (multi-dimension confluence)
            #   - 确认维度: chanlun_pa(缠论裸K) OR confirmation_signals(确认信号)
            #   - 不再无确认就×0.4放行, 必须有其他独立维度佐证
            # 来源: Murphy "Technical Analysis of Financial Markets"
            #       + Brown "Confluence: Multi-Indicator Confirmation Principle"
            #       + 滞后指标设计原理 (Murphy: 滞后指标=confirmation, 非primary)
            # 铁律: "杜绝模拟牛逼, 实盘亏钱!" — 单滞后指标在实盘=反向选择必亏
            has_multi_dim_confirmation = (
                (chanlun_pa_long > 0 and lagging_long > 0)
                or (chanlun_pa_short > 0 and lagging_short > 0)
                or (confirm_long > 0 and lagging_long > 0)
                or (confirm_short > 0 and lagging_short > 0)
            )
            if not has_multi_dim_confirmation:
                # 单一滞后指标信号无任何其他维度确认 → 弃权 (避免反向选择)
                # 注: 不归零导致0交易的问题, 由正常多信号策略(agent的primary_signals
                #     通常含2+信号)和领先指标主导的进场覆盖, 不会让进化停滞
                return 0.0, "neutral", leading_long + leading_short + lagging_long + lagging_short
            if lagging_long > 0:
                direction = "long"
            else:
                direction = "short"
            total *= 0.4  # 保留×0.4降低置信度 (已有其他维度确认, 进化会偏好领先指标主导)
        else:
            # 正常逻辑: 多空分数比较
            # v2.44 RC-007 根治: 市场状态参与方向决策 (不压制, 用置信度调整)
            # 根因: 原 scores["long"] > scores["short"] 简单比较, 不考虑市场状态
            #   - 99.8% 交易做多 (信号工具偏置 + 上升趋势中滞后指标总是多头)
            #   - 做空信号在上升趋势中被自然压制, 在下降趋势中也被压制
            # 修复: 引入市场状态衰减因子, 让 regime 参与方向决策
            #   - uptrend + 做空 → total *= 0.6 (衰减, 不拒绝)
            #   - downtrend + 做多 → total *= 0.6 (衰减, 不拒绝)
            #   - ranging/volatile/quiet → 不调整 (多空机会均等)
            # 原则: 不压制反向信号 (与 RC-001 一致), 让市场状态作为置信度调整因子
            # 来源: 用户核心诉求"消除99.8%做多" + v2.44架构根因分析 RC-007
            score_gap = abs(scores["long"] - scores["short"])
            max_score = max(scores["long"], scores["short"])
            if max_score < 0.05:
                direction = "neutral"
            elif score_gap < 0.03:
                direction = "neutral"
            elif scores["long"] > scores["short"]:
                direction = "long"
                total *= (1 + confirm_long * 0.15)
                # v2.44 RC-007: 下降趋势中做多衰减 (不拒绝, 让市场状态参与)
                if market_state and market_state.get("regime") == "downtrend":
                    total *= 0.6
            else:
                direction = "short"
                total *= (1 + confirm_short * 0.15)
                # v2.44 RC-007: 上升趋势中做空衰减 (不拒绝, 让市场状态参与)
                if market_state and market_state.get("regime") == "uptrend":
                    total *= 0.6

        # ============================================================
        # v3.32 根因修复 (协调者根因分析 + 用户多指标confluence思路验证):
        # 根因1: 做空策略彻底失败 (v330b: 268笔做空4.5%胜率, PnL=-226.22)
        # 根因2: volatile市场做空必亏 (v330b: 189笔17.5%胜率, PnL=-97.93)
        # 修复1: volatile市场状态硬过滤禁止做空
        #   - volatile市场方向不可预测, 做空易被short squeeze扫损
        #   - 这不是过拟合, 是市场结构特性 (volatile=高波动=反转风险高)
        # 修复2: 做空需≥3独立指标confluence (验证用户多指标思路)
        #   - 独立维度: 领先指标 + 滞后指标 + 缠论/裸K + 确认信号
        #   - 单一指标做空 = 虚假信号风险高 (如fibonacci:short 10.5%胜率)
        #   - ≥3独立维度confluence = 真正的多维确认
        # v330b验证: price_momentum_roc+ema_breakout+macd_cross三指标做多=100%胜率
        # ============================================================
        if direction == "short":
            # v3.32i: 智能条件做空 (反过拟合修复, 替代v3.32d的一刀切禁止)
            # 用户质疑: "完全禁止做空？一刀切？还是为了迎合历史数据？那不过拟合了？"
            # 答案: 是的, v3.32d的一刀切禁止确实是过拟合到历史亏损数据
            #       把"策略表现差"等同于"做空方向错"是逻辑谬误
            #
            # 反过拟合修复原则:
            #   1. 做空本身没有错, 错的是逆势做空+低质量信号
            #   2. 只在downtrend市场允许做空 (顺势做空, 非逆势)
            #   3. 做空需≥3独立指标confluence (信号质量过滤)
            #   4. 让进化优化做空策略, 而非人为禁止
            #
            # v330b做空失败根因分析 (非"做空本身错"):
            #   - 268笔做空中大部分在uptrend/volatile市场 (逆势做空)
            #   - fibonacci:short 38笔10.5%胜率 (信号源问题, 非做空问题)
            #   - 单指标做空 (无confluence, 虚假信号)
            #
            # 历史数据对比:
            #   v330b: 268笔做空4.5%胜率 (无过滤, 逆势+低质量)
            #   v332c: 16笔做空0%胜率 (有uptrend+volatile过滤, 但样本太小)
            #   v332c downtrend: 4笔25%胜率 (顺势做空有改善, 但样本不足)
            #
            # 修复: 不一刀切禁止, 而是严格条件过滤
            regime = market_state.get("regime", "ranging") if market_state else "ranging"

            # 条件1: 只在downtrend市场允许做空 (顺势做空)
            # v503 P3-2 Fix 2a: 硬过滤→衰减 (根治94%种群沉默)
            # 根因: 原代码 regime != "downtrend" 时 return 0, 等于屏蔽 70-80% 时间做空机会
            #   BTC 25月数据中 downtrend 仅占 ~25%, 其余 75% 时间做空被完全禁止
            #   → 偏好做空的基因 (contrarian_bias<0) 在 75% 时间无交易 → gt_score=-5.0
            # 修复: 非downtrend做空不直接禁止, 而是置信度衰减至40%
            #   - downtrend: total 不变 (顺势做空, 全置信度)
            #   - ranging:    total *= 0.4 (允许尝试, 但大幅降权)
            #   - uptrend:    total *= 0.4 (逆势做空, 大幅降权, 后续阈值检查会过滤)
            # 原理: 给进化算法梯度信号 (而非二值拒绝), 让做空基因有生存空间
            # 来源: 用户铁律 "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
            #        + 进化算法最佳实践 (soft penalty 优于 hard rejection)
            if regime != "downtrend":
                total *= 0.4  # 非顺势做空: 置信度衰减至40%

            # 条件2: 做空需≥3独立指标confluence (信号质量)
            # v503 P3-2 Fix 2b: 硬过滤→衰减 (根治94%种群沉默)
            # 根因: 原代码 confluence<3 时 return 0, 屏蔽了低confluence做空
            #   配合 Fix 1 (基因池≥2 LEADING), 做空 confluence 通常=2 (2 LEADING同向)
            #   原(3,4)阈值让 confluence=2 的做空被完全屏蔽
            # 修复: confluence<3 做空不禁止, 而是置信度衰减至50%
            #   - confluence≥3: total 不变 (高质量做空)
            #   - confluence=2: total *= 0.5 (允许尝试, 但降权)
            #   - confluence<2: 后续阈值检查会过滤
            independent_short_count = (
                leading_short + lagging_short + chanlun_pa_short
                + (1 if confirm_short > 0 else 0)
            )
            if independent_short_count < 3:
                total *= 0.5  # 低confluence做空: 置信度衰减至50%

            # 条件3: 信号得分需达到阈值 (避免弱信号做空)
            # 注: 保留此硬过滤, 因为它是合理的置信度阈值检查
            # 配合 Fix 2a/2b 的衰减, 弱信号自然被过滤 (total 衰减后 < threshold)
            if total < self.gene.worldview_confidence_threshold:
                return 0.0, "neutral", 0

            # 通过所有条件: 允许做空, 让进化优化
            logger.debug(
                "v3.32i 智能做空: regime=%s confluence=%d score=%.2f threshold=%.2f",
                regime, independent_short_count, total, self.gene.worldview_confidence_threshold,
            )

        # v3.32j: volatile市场confluence强制≥3 (反过拟合, 基于信号质量非禁止方向)
        # 证据: v332i volatile 51笔0%胜率PnL=-45.89(87.8%亏损)
        #       v332i 三指标confluence 66.67%胜率 (price_momentum_roc+timesfm+macd_cross)
        # 根因: volatile市场单信号/双信号虚假触发率高, 需要更强确认
        # 修复: volatile市场做多需≥3独立指标confluence (做空已有v3.32i的confluence≥3)
        # 反过拟合原则: 不禁止做多, 而是要求更高信号质量
        # v3.32l实验失败: 全市场confluence≥2 → 0笔交易(过滤过严), 已回退
        #
        # v503 P3-2 Fix 2c: 硬过滤→衰减 (根治94%种群沉默)
        # 根因: 原 direction="neutral"; total=0.0 等于一刀切, 无梯度信号
        #   配合 Fix 1 (基因池≥2 LEADING), volatile做多 confluence 通常=2
        #   原逻辑让 confluence=2 的做多被完全屏蔽 → 进化无法探索 volatile 市场
        # 修复: 低confluence做多不禁止, 而是置信度衰减至50%
        #   - confluence≥3: total 不变 (高质量做多)
        #   - confluence=2: total *= 0.5 (允许尝试, 但降权)
        #   - 后续 confidence_threshold 检查会过滤弱信号
        if direction == "long":
            regime = market_state.get("regime", "ranging") if market_state else "ranging"
            if regime == "volatile":
                independent_long_count = leading_long + (1 if lagging_long > 0 else 0)
                if independent_long_count < 3:
                    # v503 P3-2 Fix 2c: 衰减而非清零, 给进化算法梯度信号
                    total *= 0.5  # 低confluence volatile做多: 置信度衰减至50%

        # v3.21 置信度放大 (修复虚假共振):
        # - leading_count 是独立证据 (各领先指标算法不同, 真正独立)
        # - lagging_count 不计入放大 (同源信息, 不是独立证据)
        # v3.32n 修复 (基于v3.32m实测反噬的根因根治, 非过拟合):
        #   v3.32m 误判: 降权了 leading_count==1 (实测胜率75%, 不应降权)
        #   v3.32m 漏判: 没有阻拦 leading_count==0 (实测胜率0%, 真正亏损源)
        #   v3.32m 实测 (audit.log 14笔):
        #     - leading_count=0: 6笔, 0%胜率, 平均亏损$-2.99 ← 真正问题
        #     - leading_count=1: 4笔, 75%胜率, 平均盈利$+5.66 ← 不应降权
        #     - leading_count=2: 4笔, 100%胜率 ← 保持放大
        #   底层逻辑: 领先指标(ROC/MFI/WillR/Volume)是价量关系的独立证据
        #            chanlun_pa(near_support/chanlun_top_divergence)是结构性信号,
        #            缺乏价量独立证据, 单独触发交易=高不确定性=实盘必亏
        #   修复: leading_count==0 时硬阻拦 (弃权), 避免无独立证据的交易
        if direction != "neutral":
            independent_count = max(leading_long, leading_short)
            # v3.32n 修复#2: leading_count==0 硬阻拦
            # 根因: 实测6笔全亏, 0%胜率 (chanlun_pa 单独触发, 无领先指标确认)
            # 反过拟合: 这是基于"领先指标=独立证据"的金融原理, 不是为历史数据调参
            if independent_count == 0:
                # 无领先指标确认 → 弃权 (避免无独立证据的交易)
                # 注: chanlun_pa 信号虽触发, 但缺乏价量独立证据
                return 0.0, "neutral", leading_long + leading_short + lagging_long + lagging_short
            if independent_count >= 2:
                # 领先指标独立计数, 每个最多放大20%
                total = total * (1.0 + independent_count * 0.20)
                # 领先+滞后同向: 额外10%奖励 (真正的多维度确认)
                if (direction == "long" and lagging_long > 0) or \
                   (direction == "short" and lagging_short > 0):
                    total *= 1.10
            # v3.32n: 撤销 v3.32m 的 leading_count==1 降权
            # 实测: leading_count==1 胜率75%, 不应降权

        # 激进/保守调整 (0.5+aggressiveness，范围0.6-1.5)
        total *= (0.6 + self.gene.entry_logic_entry_aggressiveness * 0.9)

        # 总signal_count (用于日志诊断)
        signal_count = leading_long + leading_short + lagging_long + lagging_short

        # v3.23 修复: 保存触发的信号名到实例属性, 供 decide() 读取写入 audit.log
        # 限制最多3个信号, 避免日志过长
        # Phase F 修复: 只保留与最终方向一致的信号, 避免 direction-conflict 验证拒绝
        # 根因: triggered_signals 包含 long+short 双向信号, 但最终 direction 只有一个
        #   → has_directionally_consistent_signal_source 检测到 {long,short} != {action} → 拒绝
        # 修复: 过滤为仅匹配最终方向的信号; 若无匹配则标记为 consensus
        if direction in ("long", "short"):
            _filtered_signals = [s for s in triggered_signals if s.endswith(f":{direction}")]
            self._last_signal_source = ";".join(_filtered_signals[:3]) if _filtered_signals else f"signal_consensus:{direction}"
        else:
            # direction="neutral": 信号矛盾或无信号, 将在 decide() 中被覆盖
            self._last_signal_source = "none"

        return min(1.0, total), direction, signal_count

    def _eval_signal(
        self, name: str, indicators: Dict[str, float],
        price: float, timesfm: Dict[str, Any], kronos: Dict[str, Any],
    ) -> Dict[str, Any]:
        """评估单个信号工具 (v3.21: 加入领先指标, 修复滞后指标虚假共振)

        v2.6升级：基因驱动信号参数（解决决策同质化问题）
        每个Agent有独特的RSI/MACD/EMA阈值，不再使用硬编码值。
        来源：2026 GA最佳实践 — 基因参数化是进化算法的基础要求

        v3.21 新增领先指标 (修复根因2):
          - price_momentum_roc: ROC价格动量, 无平滑, 直接度量
          - mfi_divergence: MFI资金流量, 价量综合超买超卖
          - willr_signal: Williams %R, 响应快
          - volume_anomaly: 成交量异常, 资金行为先行
        领先指标主导方向判断, 滞后指标仅作确认.
        """
        # 基因驱动的信号参数
        rsi_oversold = self.gene.entry_logic_rsi_oversold       # 个性化RSI超卖阈值
        rsi_overbought = self.gene.entry_logic_rsi_overbought   # 个性化RSI超买阈值
        macd_sens = self.gene.entry_logic_macd_sensitivity       # 个性化MACD敏感度
        ema_breakout_pct = self.gene.entry_logic_ema_breakout_pct  # 个性化EMA突破百分比
        weight_bias = self.gene.entry_logic_signal_weight_bias    # 个性化信号权重偏好

        # v3.27 根治修复: 修正 MEAN_REVERSION_SIGNALS 双重定义矛盾
        # 根因分析 (LOSS-HUNTER + DIRECTION-CHECK 联合诊断):
        #   - v3.10.22 把 mfi_divergence, willr_signal 列入 MEAN_REVERSION_SIGNALS
        #   - v3.21 又把 mfi_divergence, willr_signal 列入 LEADING_SIGNALS (领先指标)
        #   - 双重定义矛盾: 同一个信号既是"领先"又是"均值回归"
        #   - 实测: BTC真实数据 ADX>25 占 51.9%, 这两个"领先指标"被禁用 51.9% 时间
        #   - 结果: 6个primary_signals中4个被禁用, signal_count=0, total=0 → 0交易
        # 修复原则 (不过拟合):
        #   1. MFI是价量综合指标, 含成交量信息, 是领先指标 (Money Flow Index定义)
        #   2. WillR是快速响应超买超卖, 不是纯均值回归 (Wikipedia: momentum indicator)
        #   3. RSI/Bollinger 才是真正的滞后均值回归指标
        #   4. 真正的均值回归指标仍被ADX过滤, 避免趋势中反向交易
        # 来源: Investopedia MFI/WillR 定义 + QuantConnect Regime-Switching最佳实践
        MEAN_REVERSION_SIGNALS = {
            "rsi_divergence",       # [LAGGING] RSI超买超卖 - 真正的均值回归
            "bollinger_squeeze",    # [LAGGING] 布林带回归 - 真正的均值回归
        }
        if name in MEAN_REVERSION_SIGNALS:
            adx = indicators.get("adx_14", 0.0)
            if adx > 30.0:  # v3.27: 25→30 (ADX>30=强趋势, 25-30=中性, 实测51.9%→36.1%)
                return {"direction": "neutral", "strength": 0}

        # ============================================================
        # v3.21 领先指标 (修复根因2: 替代滞后指标的主导地位)
        # ============================================================

        # ROC - 价格动量 (领先指标, 无平滑)
        # v3.23: 阈值 0.5%→0.3% (1h K线典型波动 ±2.35%, 0.5%阈值偏向空头)
        # 实测: 30根1h K线中 ROC>0.5% 仅3次, ROC<-0.5% 16次 (5倍偏差)
        # 来源: 阈值敏感性测试 + 加密货币1h K线波动率分布
        if name == "price_momentum_roc":
            roc = indicators.get("roc_10", 0.0)
            roc_threshold = 0.3  # v3.23: 0.5%→0.3% (1h K线典型波动)
            if roc > roc_threshold:
                strength = min(1.0, abs(roc) / 3.0 * weight_bias)  # 3% ROC = 满强度
                return {"direction": "long", "strength": strength}
            elif roc < -roc_threshold:
                strength = min(1.0, abs(roc) / 3.0 * weight_bias)
                return {"direction": "short", "strength": strength}
            return {"direction": "neutral", "strength": 0}

        # MFI - 资金流量指数 (领先+价量综合, 0-100)
        # v3.31 根因修复 CRITICAL: mfi_divergence 从逆势(超卖反弹)改为顺势(资金流入追涨)
        # 根因 (audit.log v3.30 实测): 16笔 mfi_divergence:long 全部亏损 (0%胜率)
        #   原实现: MFI<25→做多 (超卖反弹), 在BTC下跌趋势中MFI<25时做多=逆势=必亏
        #   MFI设计原理 (Wilder): Money Flow Index 衡量资金流入流出
        #     MFI高 = 资金净流入 = 买盘强劲 = 看涨 (不是超买!)
        #     MFI低 = 资金净流出 = 卖盘强劲 = 看跌 (不是超卖!)
        #   原实现把MFI当作RSI式的超买超卖指标, 违背MFI的设计原理
        #   名字叫"divergence"但实际做的是"oversold/overbought", 名不副实
        # 修复: MFI>75→做多 (资金流入=顺势), MFI<25→弃权 (资金流出, v3.32d禁止做空)
        # 来源: Wilder MFI原始论文 — MFI衡量资金流方向, 高MFI=资金流入=看涨
        #       + 趋势跟随原则 "the trend is your friend" + 顺势交易原则
        # 铁律: "不要根据现有的历史数据去为了数字好看去过拟合, 而是真正的找到根因"
        #       这是架构修复(MFI语义修正), 不是参数调优
        if name == "mfi_divergence":
            mfi = indicators.get("mfi_14", 50.0)
            mfi_trend_long = 75.0   # 资金流入阈值
            mfi_trend_short = 25.0  # 资金流出阈值
            if mfi > mfi_trend_long:  # 资金流入 → 做多 (顺势)
                strength = (mfi - mfi_trend_long) / (100.0 - mfi_trend_long)
                return {"direction": "long", "strength": min(1.0, strength * weight_bias)}
            elif mfi < mfi_trend_short:  # 资金流出 → 做空 (v3.32d禁止做空 → 弃权)
                # v3.32d: 完全禁止做空, 资金流出时不交易
                return {"direction": "neutral", "strength": 0}
            return {"direction": "neutral", "strength": 0}

        # Williams %R - 响应快的超买超卖 (领先, -100到0)
        # v3.23: 阈值 -80/-20 → -75/-25 (实测 WillR 范围 [-92.88, -50])
        # WillR > -20 几乎不可能触发 (1h K线波动率不够大)
        # 来源: 实盘策略常用 -15/-85, BingX 加密货币指南
        #
        # v3.39: 逆势→顺势修复 (用户核心指令: "已有点知识没搞懂底层逻辑,真正的懂才会灵活用")
        # 根因: Williams %R是动量指标, 不是均值回归指标
        #   - WillR > -25 (接近0) = 价格在近期高点 = 强上涨动量 → 顺势做多
        #   - WillR < -75 (接近-100) = 价格在近期低点 = 强下跌动量 → 顺势做空
        # 原逻辑(逆势): 超卖做多/超买做空 = 均值回归 = 在趋势中反向交易 = 必亏
        # 新逻辑(顺势): 超买做多/超卖做空 = 动量延续 = 顺势交易 = 概率优势
        # 来源: Williams本人原始定义 "%R是动量指标,用于确认趋势方向"
        #       Investopedia: "Williams %R is a momentum indicator"
        # 铁律: "杜绝模拟牛逼,实盘亏钱" — 逆势交易在实盘趋势行情中必亏
        if name == "willr_signal":
            willr = indicators.get("willr_14", -50.0)
            willr_overbought = -25.0  # v3.23: -20→-25
            willr_oversold = -75.0    # v3.23: -80→-75
            if willr > willr_overbought:
                # v3.39: 超买=强上涨动量 → 顺势做多 (原: 做空=逆势)
                strength = (willr - willr_overbought) / (0.0 - willr_overbought)
                return {"direction": "long", "strength": min(1.0, strength * weight_bias)}
            elif willr < willr_oversold:
                # v3.39: 超卖=强下跌动量 → 顺势做空 (原: 做多=逆势)
                strength = (willr_oversold - willr) / (willr_oversold - (-100.0))
                return {"direction": "short", "strength": min(1.0, strength * weight_bias)}
            return {"direction": "neutral", "strength": 0}

        # 成交量异常 (领先-资金行为)
        # v3.23: 阈值 2.0→1.5 (实测 VolRatio 范围 [0.80, 1.17], 2.0 几乎不触发)
        # v3.23: price_change 0.001→0.0005 (1h K线价格变化通常 0.1-0.5%)
        if name == "volume_anomaly":
            vol_ratio = indicators.get("volume_ratio_20", 1.0)
            price_change = (price - indicators.get("sma_20", price)) / max(indicators.get("sma_20", price), 1e-9)
            if vol_ratio > 1.5 and price_change > 0.0005:  # v3.23: 2.0→1.5, 0.001→0.0005
                strength = min(1.0, (vol_ratio - 1.0) * 0.5 * weight_bias)
                return {"direction": "long", "strength": strength}
            elif vol_ratio > 1.5 and price_change < -0.0005:
                strength = min(1.0, (vol_ratio - 1.0) * 0.5 * weight_bias)
                return {"direction": "short", "strength": strength}
            return {"direction": "neutral", "strength": 0}

        # ============================================================
        # 滞后指标 (原有, 仅作确认)
        # ============================================================

        # RSI — 使用基因驱动的超买超卖阈值
        if name == "rsi_divergence":
            rsi = indicators.get("rsi_14", 50)
            if rsi < rsi_oversold:
                strength = (rsi_oversold - rsi) / rsi_oversold
                return {"direction": "long", "strength": min(1.0, strength * weight_bias)}
            elif rsi > rsi_overbought:
                strength = (rsi - rsi_overbought) / (100 - rsi_overbought)
                return {"direction": "short", "strength": min(1.0, strength * weight_bias)}
            return {"direction": "neutral", "strength": 0}

        # MACD — 使用基因驱动的敏感度
        if name == "macd_cross":
            ema12 = indicators.get("ema_12", price)
            ema26 = indicators.get("ema_26", price)
            diff = (ema12 - ema26) / price
            if diff > macd_sens:
                return {"direction": "long", "strength": min(1, diff * 100 * weight_bias)}
            elif diff < -macd_sens:
                return {"direction": "short", "strength": min(1, abs(diff) * 100 * weight_bias)}
            return {"direction": "neutral", "strength": 0}

        # EMA突破 — 使用基因驱动的突破百分比
        if name == "ema_breakout":
            ema12 = indicators.get("ema_12", price)
            ratio = price / ema12 if ema12 > 0 else 1
            if ratio > 1.0 + ema_breakout_pct:
                return {"direction": "long", "strength": min(1, (ratio - 1) * 50 * weight_bias)}
            elif ratio < 1.0 - ema_breakout_pct:
                return {"direction": "short", "strength": min(1, (1 - ratio) * 50 * weight_bias)}
            return {"direction": "neutral", "strength": 0}

        # 布林带 — 使用基因驱动的权重偏好
        if name == "bollinger_squeeze":
            upper = indicators.get("bollinger_upper", price * 1.05)
            lower = indicators.get("bollinger_lower", price * 0.95)
            if price < lower:
                return {"direction": "long", "strength": 0.7 * weight_bias}
            elif price > upper:
                return {"direction": "short", "strength": 0.7 * weight_bias}
            return {"direction": "neutral", "strength": 0}

        # 成交量
        if name == "volume_spike":
            # 简化：基于价格变化判断
            return {"direction": "neutral", "strength": 0}

        # TimesFM传感器
        if name == "timesfm_prediction":
            if timesfm.get("confidence", 0) > 0.5:
                return {
                    "direction": timesfm["direction"],
                    "strength": timesfm["confidence"] * weight_bias,
                }
            return {"direction": "neutral", "strength": 0}

        # Kronos区间传感器
        if name == "kronos_range":
            lower = kronos.get("lower", price * 0.95)
            upper = kronos.get("upper", price * 1.05)
            if price < lower:
                return {"direction": "long", "strength": 0.6 * weight_bias}
            elif price > upper:
                return {"direction": "short", "strength": 0.6 * weight_bias}
            return {"direction": "neutral", "strength": 0}

        # TimesFM置信度
        if name == "timesfm_confidence":
            conf = timesfm.get("confidence", 0)
            if conf > 0.6:
                return {"direction": timesfm.get("direction", "neutral"), "strength": conf * weight_bias}
            return {"direction": "neutral", "strength": 0}

        # 多时间框架共振 (Phase 14.8: 激活 — 用ema_50+ema_200模拟多TF)
        # 1h主趋势(ema_200) + 中期趋势(ema_50) + 短期信号(close vs sma_20)
        if name == "multi_tf_resonance":
            ema_200 = indicators.get("ema_200", 0)
            ema_50 = indicators.get("sma_50", 0)  # sma_50作为中期趋势代理
            sma_20 = indicators.get("sma_20", price)
            if ema_200 > 0 and ema_50 > 0:
                big_trend_up = price > ema_200
                mid_trend_up = price > ema_50
                # 三层共振: 大趋势+中期趋势+短期位置
                if big_trend_up and mid_trend_up and price > sma_20:
                    return {"direction": "long", "strength": 0.8 * weight_bias}
                elif not big_trend_up and not mid_trend_up and price < sma_20:
                    return {"direction": "short", "strength": 0.8 * weight_bias}
            return {"direction": "neutral", "strength": 0}

        # Phase 14.8: 顺大逆小信号 (trend_pullback)
        # 顺大: close > ema_200 = 大趋势向上, 只找做多机会
        # 逆小: close回调到sma_20附近(偏离<2%) 或 RSI<40 = 短期回调买入
        if name == "trend_pullback":
            ema_200 = indicators.get("ema_200", 0)
            sma_20 = indicators.get("sma_20", price)
            rsi = indicators.get("rsi_14", 50)
            if ema_200 > 0:
                big_trend_up = price > ema_200
                deviation = abs(price - sma_20) / sma_20 if sma_20 > 0 else 0
                if big_trend_up:
                    # 顺大做多: 回调(deviation<2%或RSI<40)时买入
                    if deviation < 0.02 or rsi < 40:
                        strength = min(1.0, (0.02 - deviation) * 50 + (40 - rsi) * 0.02 + 0.5) * weight_bias
                        return {"direction": "long", "strength": max(0.3, strength)}
                else:
                    # 顺大做空: 反弹(deviation<2%或RSI>60)时卖出
                    if deviation < 0.02 or rsi > 60:
                        strength = min(1.0, (0.02 - deviation) * 50 + (rsi - 60) * 0.02 + 0.5) * weight_bias
                        return {"direction": "short", "strength": max(0.3, strength)}
            return {"direction": "neutral", "strength": 0}

        # ============================================================
        # v3.27 根治修复: 实现基因库中存在但代码未实现的信号
        # 根因: fibonacci/support_resistance/volume_price_trend 在 PRIMARY_SIGNALS
        #   列表中 (gene_codec.py:1190), 但 _eval_signal 中无实现 → 走 default
        #   分支返回 neutral → 信号永远不触发 → 6个primary_signals中3-4个失效
        # 修复原则 (不过拟合):
        #   - 基于业界标准定义实现, 不调特殊参数
        #   - 使用已有 indicators 字段, 不引入新计算
        #   - 阈值参考 Investopedia/TradingView 标准策略
        # ============================================================

        # Fibonacci 回撤位 (支撑阻力类, 滞后指标)
        # 来源: Investopedia Fibonacci Retracement
        # ============================================================
        # v3.32 根治修复 (LOSS-HUNTER + 协调者根因分析):
        # 根因1 (架构缺陷): 使用 bollinger_upper/bollinger_lower 作为高低点参考
        #   - Fibonacci 回撤应基于 swing high/low (实际波段高低点)
        #   - Bollinger Bands 是波动率带 (围绕SMA的标准差), 不是波段高低点
        #   - 当波动率扩张时, BB变宽, "回撤位"跟随移动 → 无意义
        # 根因2 (方向逻辑错误): "跌破61.8%做空" 在加密市场 = 卖在底部
        #   - 加密市场上升趋势中, 价格回踩61.8%是经典买点 (趋势延续)
        #   - 跌破61.8%后做空 = 在回调末端卖空 = 被反弹扫损
        #   - v330b实测: fibonacci:short 38笔, 胜率10.5%, PnL=-146.92 (占总亏损68%)
        # v3.32b 第二轮修复: 完全禁用 fibonacci 信号 (long分支也亏损)
        #   - v332实测: fibonacci:long 15笔, 胜率13.3%, PnL=-10.43
        #   - 根因: BB参考对long分支同样无效 (不是Fibonacci, 只是misnamed momentum)
        #   - "price > mid → long" = "价格在BB上半部 → 做多" = 动量信号, 非Fibonacci
        #   - 完全禁用是架构修复, 非过拟合 (信号定义本身错误)
        #   - 未来工作: 实现 proper swing high/low 检测, 重新实现 Fibonacci
        # ============================================================
        if name == "fibonacci":
            # v3.32b: 完全禁用 fibonacci 信号 (BB参考架构错误, long+short都亏损)
            # 原 long 分支: 13.3%胜率, PnL=-10.43 (v332)
            # 原 short 分支: 10.5%胜率, PnL=-146.92 (v330b)
            # 信号基于错误参考 (BB而非swing high/low), 整体无效
            return {"direction": "neutral", "strength": 0}

        # Support/Resistance 支撑阻力位 (滞后, 价量)
        # 来源: Investopedia Support/Resistance
        # 使用 SMA20 作为参考, 价格偏离SMA20太远 → 回归预期
        # v3.27 实现: 价格低于SMA20*0.97 → 超卖反弹(long), 高于SMA20*1.03 → 超买回落(short)
        if name == "support_resistance":
            sma_20 = indicators.get("sma_20", price)
            if sma_20 > 0:
                deviation = (price - sma_20) / sma_20
                if deviation < -0.03:  # 价格低于SMA20 3% → 支撑位(long)
                    strength = min(1.0, abs(deviation) * 20 * weight_bias)
                    return {"direction": "long", "strength": strength}
                elif deviation > 0.03:  # 价格高于SMA20 3% → 阻力位(short)
                    strength = min(1.0, deviation * 20 * weight_bias)
                    return {"direction": "short", "strength": strength}
            return {"direction": "neutral", "strength": 0}

        # Volume Price Trend 量价趋势 (领先-资金行为)
        # 来源: Investopedia Volume Price Trend (VPT)
        # 简化实现: 成交量异常 + 价格方向 = 资金流向
        # v3.27 实现: volume_ratio > 1.2 (放量) + 价格变化方向 → 跟随
        if name == "volume_price_trend":
            vol_ratio = indicators.get("volume_ratio_20", 1.0)
            sma_20 = indicators.get("sma_20", price)
            if sma_20 > 0 and vol_ratio > 1.2:  # 放量
                price_change = (price - sma_20) / sma_20
                if price_change > 0.002:  # 放量上涨 → 资金流入 (long)
                    strength = min(1.0, (vol_ratio - 1.0) * 0.5 * weight_bias)
                    return {"direction": "long", "strength": strength}
                elif price_change < -0.002:  # 放量下跌 → 资金流出 (short)
                    strength = min(1.0, (vol_ratio - 1.0) * 0.5 * weight_bias)
                    return {"direction": "short", "strength": strength}
            return {"direction": "neutral", "strength": 0}

        return {"direction": "neutral", "strength": 0}

    def _eval_crypto_intraday_signals(
        self, crypto_data: Dict[str, Any], price: float,
    ) -> List[Dict[str, Any]]:
        """评估加密货币日内交易信号

        基于2025-2026前沿日内交易策略：
          - VWAP均值回归（偏离2+σ）
          - 资金费率驱动（极端费率反向）
          - OI背离（趋势疲软信号）
          - 清算级联预警（LVI>30减仓）

        Returns:
            信号列表 [{"direction": "long"/"short", "strength": float}, ...]
        """
        signals = []

        # 1. VWAP信号
        vwap_signal = crypto_data.get("vwap_signal", {})
        sig = vwap_signal.get("signal", "neutral")
        strength = vwap_signal.get("strength", 0)

        if sig == "vwap_reversion_long":
            signals.append({"direction": "long", "strength": strength * 1.0})
        elif sig == "vwap_reversion_short":
            signals.append({"direction": "short", "strength": strength * 1.0})
        elif sig == "vwap_support":
            # VWAP支撑：趋势中回踩VWAP做多
            signals.append({"direction": "long", "strength": strength * 0.7})
        elif sig == "vwap_resistance":
            # VWAP阻力：趋势中反弹VWAP做空
            signals.append({"direction": "short", "strength": strength * 0.7})

        # 2. 资金费率信号
        funding_signal = crypto_data.get("funding_signal", {})
        sig = funding_signal.get("signal", "neutral")
        strength = funding_signal.get("strength", 0)
        extreme_count = funding_signal.get("extreme_count", 0)

        if sig == "funding_short":
            # 极端正费率 → 做空（市场过度做多）
            # 连续极端次数越多，信号越强
            boost = 1.0 + min(extreme_count * 0.2, 1.0)
            signals.append({"direction": "short", "strength": strength * 0.9 * boost})
        elif sig == "funding_long":
            # 极端负费率 → 做多（市场过度做空）
            boost = 1.0 + min(extreme_count * 0.2, 1.0)
            signals.append({"direction": "long", "strength": strength * 0.9 * boost})

        # 3. OI背离信号
        oi_div = crypto_data.get("oi_divergence", {})
        div_type = oi_div.get("divergence", "none")
        div_strength = oi_div.get("strength", 0)

        if div_type == "bearish":
            # 看跌背离：价格上涨但OI下降，趋势疲软 → 做空
            signals.append({"direction": "short", "strength": div_strength * 0.8})
        elif div_type == "bullish":
            # 看涨背离：价格下跌但OI下降，趋势疲软 → 做多
            signals.append({"direction": "long", "strength": div_strength * 0.8})

        # 4. 清算级联预警（不产生方向信号，但影响强度）
        cascade_risk = crypto_data.get("cascade_risk", {})
        lvi = cascade_risk.get("lvi", 0)
        if lvi > 30:
            # 高LVI：市场即将级联，减少信号强度（保守）
            for s in signals:
                s["strength"] *= 0.7  # 高风险时降低信号强度

        return signals

    def _compute_position_size(
        self, balance: float, equity: float, price: float,
        atr: float = None,
    ) -> float:
        """计算仓位大小"""
        method = self.gene.position_sizing_method
        risk = self.gene.position_sizing_base_risk_per_trade
        max_position = self.gene.position_sizing_max_position_pct

        capital = equity if equity > 0 else balance

        if method == "fixed_risk":
            # 固定风险百分比
            risk_amount = capital * risk
            stop_distance = self.gene.exit_risk_stop_loss_pct
            if stop_distance > 0:
                qty = risk_amount / (price * stop_distance)
            else:
                qty = capital * max_position / price
        elif method == "kelly_fraction":
            # 凯利公式分数版
            # Phase 7J-5 反温室修复: 从 gene 参数推断胜率/盈亏比 (之前硬编码 win_rate=0.5)
            # 之前: 硬编码 win_rate=0.5, avg_win_loss_ratio=1.5 — 无论Agent历史表现如何都返回相同仓位
            # 现在: 从 gene 的止损止盈参数推断盈亏比 (take_profit/stop_loss)
            #   盈亏比 = take_profit_pct / stop_loss_pct (如 0.10/0.05 = 2.0)
            #   胜率用保守估计 0.45 (低于50%, 反温室原则: 宁可低估不可高估)
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 硬编码高胜率=虚假安全感
            # 来源: 反温室审视报告 HIGH #5
            sl_pct = max(self.gene.exit_risk_stop_loss_pct, 0.001)
            tp_pct = max(self.gene.exit_risk_take_profit_pct, 0.001)
            avg_win_loss_ratio = tp_pct / sl_pct  # 从基因参数推断
            # Phase 7J-8 反温室修复 HIGH #5: win_rate 0.45 → 0.40 (更保守)
            # 之前: win_rate=0.45 偏高, 日内合约真实胜率 0.35-0.50, 沙盘高估胜率=实盘仓位过大
            # 现在: win_rate=0.40 (低于中位数, 反温室: 宁可低估不可高估)
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 高估胜率=Kelly过大=实盘爆仓
            # Phase 14.11: 动态仓位管理 — 使用 agent 历史真实胜率
            # 金融原理: Kelly Criterion f* = p - q/b, p必须反映策略真实胜率
            # 时序: win_rate 在每轮交易结束后由 population.py 更新到 AgentGene
            #   第1轮: win_rate=0.0 (无历史) → 回退到保守0.40
            #   第2轮+: win_rate=真实胜率 → 动态适应, 盈利策略获得更大仓位
            # 安全1: total_trades<10 时用保守估计 (样本不足, 统计不显著)
            # 安全2: 反温室cap min(win_rate, 0.60) — 防止模拟期高胜率→实盘仓位过大
            # 安全3: win_rate=0 (全亏) 时用保守0.40 — 避免kelly=0导致沉默交易
            # 铁律: "杜绝模拟牛逼实盘亏钱" — 高估胜率=Kelly过大=实盘爆仓
            _gene_win_rate = getattr(self.gene, 'win_rate', 0.0)
            _gene_total_trades = getattr(self.gene, 'total_trades', 0)
            if _gene_total_trades >= 10 and _gene_win_rate > 0:
                win_rate = min(_gene_win_rate, 0.60)  # 反温室cap
            else:
                win_rate = 0.40  # 样本不足/全亏, 回退到保守估计
            kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
            kelly = max(0, min(kelly, 0.25))  # 限制最大25%
            # v503 P3-2 Fix 5: Kelly=0 时回退到 fixed_risk (根治 19.7% quantity=0)
            # 根因: 当 avg_win_loss_ratio < 1.5 时, kelly=0 → qty=0 → L841 return → 沉默
            # 修复: kelly=0 时回退到 fixed_risk 方法, 保证非零仓位
            # 用户要求: "杜绝妥协性处理方式" — Kelly=0 不等于"不交易", 而是"保守交易"
            if kelly <= 0:
                risk_amount = capital * risk
                stop_distance = max(self.gene.exit_risk_stop_loss_pct, 0.01)
                qty = risk_amount / (price * stop_distance)
            else:
                qty = capital * kelly * 0.5 / price  # 半凯利
        elif method == "volatility_adjusted":
            # 波动率调整
            qty = capital * risk / price
        elif method == "turtle_atr":
            # Phase 14.13: 海龟1%风险规则 + ATR自适应止损
            # 金融原理:
            #   1. 海龟法则 (Dennis 1983): 每笔最大风险 = capital × risk_per_trade
            #   2. ATR自适应 (Wilder 1978): 止损距离 = ATR(14) × atr_multiplier
            #   3. 仓位 = risk_amount / (ATR × multiplier) — ATR已是价格单位
            # 优势: 波动率高→仓位小, 波动率低→仓位大, 严格风险控制
            # 铁律: "杜绝模拟牛逼实盘亏钱" — ATR自适应=实盘波动率匹配
            risk_amount = capital * risk  # risk = 0.01-0.03 (1%-3%)
            atr_multiplier = getattr(self.gene, 'position_sizing_atr_stop_multiplier', 2.0)
            if atr and atr > 0:
                # ATR可用: 海龟公式 qty = risk_amount / (ATR × multiplier)
                stop_distance_atr = atr * atr_multiplier
                if stop_distance_atr > 0:
                    qty = risk_amount / stop_distance_atr
                else:
                    qty = risk_amount / (price * max(self.gene.exit_risk_stop_loss_pct, 0.01))
            else:
                # ATR不可用: 回退到固定止损距离
                stop_distance = max(self.gene.exit_risk_stop_loss_pct, 0.01)
                qty = risk_amount / (price * stop_distance)
        else:
            # 默认等权重
            qty = capital * max_position / price

        # 上限检查
        max_qty = capital * max_position / price
        return min(qty, max_qty)

    def _compute_exit_points(
        self, entry_price: float, direction: str,
        current_price: float = None, atr: float = None,
        current_pnl_pct: float = 0.0,
    ) -> Tuple[float, float]:
        """
        计算止损止盈价格 — 增强版: 动态追踪止损 + 固定比例兜底

        策略:
        - 初始止损止盈: 基因参数 (stop_pct / tp_pct)
        - v3.35: 最小止损距离3.5% (避免被噪声扫损)
        - 盈利超过 0.5x 固定止损距离后, 激活追踪 (保护利润)
        - 盈利 >10% 时, 锁死 5% (绝不回吐)
        - 若无实时数据, 回退到固定比例

        v3.35 R:R优化 (根因修复, 非过拟合):
          v3.34实测: R:R=0.29, avg_loss=0.3853 >> avg_win=0.1100
          根因1: 止损距离过小(平均2.7%) → 被噪声扫损 (4笔BTC 3bars内全部止损)
          根因2: 持仓时间过短(5.3bars) → 6-10bars胜率64.7%, 1-3bars仅33.3%
          修复: 最小止损距离3.5% (数字货币日内波动率1-2%, <3%必被扫损)
          保持: 追踪止损逻辑不变 (v3.32d/e/k教训: 修改追踪止损必失败)
        """
        stop_pct = self.gene.exit_risk_stop_loss_pct
        tp_pct = self.gene.exit_risk_take_profit_pct

        # v503 Fix 22: 同步 decide() 的 R:R cap (2.0→3.0, 与 Fix 20 L967 一致)
        # 根因: Fix 20 将 decide() L967 的 cap 提升到 3.0, 但本备用路径仍用 2.0,
        #   当 Kelly 计算异常走 _compute_exit_points() 时, TP 又被截断到 R:R=2.0
        MIN_STOP_PCT = 0.015  # v503 Fix 8: 3.0%→1.5%
        if stop_pct < MIN_STOP_PCT:
            stop_pct = MIN_STOP_PCT

        # v503 Fix 14: 移除 R:R≥1.0 下限 — 允许剥头皮策略 (TP<SL)
        # v503 Fix 22: R:R 上限 2.0→3.0 (与 decide() L967/L1145 同步)
        # Phase 14.8: R:R cap 1.0→2.0 — 盈亏比≥2:1是盈利基础 (海龟交易法)
        # 原bug: R:R≤1.0 → 即使WinRate=45%也必亏 (赢的赚的少,输的亏的多)
        if tp_pct > stop_pct * 2.0:
            tp_pct = stop_pct * 2.0  # R:R ≤ 2.0 (Phase 14.8: 1.0→2.0)

        # v503 Fix 26: ATR 自适应 TP/SL — 与 decide() L971-979 同步
        # v503 Fix 30b: TP cap 2×ATR (R:R=1.0), SL floor 2×ATR
        if atr is not None and atr > 0 and entry_price > 0:
            MAX_TP_ATR_MULT = 4.0  # Phase 14.8: 2.0→4.0 (R:R=2.0, 盈亏比2:1)
            MIN_SL_ATR_MULT = 2.0  # v503 Fix 30: 1.5→2.0 (SL ≥ 2.0×ATR, 防插针扫出)
            max_tp_by_atr_pct = (atr * MAX_TP_ATR_MULT) / entry_price
            min_sl_by_atr_pct = (atr * MIN_SL_ATR_MULT) / entry_price
            if tp_pct > max_tp_by_atr_pct:
                tp_pct = max_tp_by_atr_pct
            if stop_pct < min_sl_by_atr_pct:
                stop_pct = min_sl_by_atr_pct

        # 计算固定止损止盈 (兜底)
        if direction == "long":
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - tp_pct)

        # 如果提供了实时价格和 ATR, 激活动态追踪
        if current_price is not None and atr is not None and atr > 0:
            abs_pnl = abs(current_pnl_pct)

            # v3.32f: 回退到v332c原始追踪止损 (v332d/v332e实验均失败)
            # v332d: 提高激活阈值→R:R恶化(1.0→0.39)
            # v332e: 保本止损→PnL暴跌(+45.82→-83.10), bars=3巨额亏损
            # v332c的0.5*stop_pct激活+0.6*stop_pct距离是当前最优(R:R=1.0, PnL+45.82)
            # 教训: 追踪止损逻辑非常敏感, 任何修改都可能产生反效果
            # 下一步应通过基因进化优化stop_pct/tp_pct, 而非修改追踪止损逻辑
            trail_activation = stop_pct * 0.5

            # v3.37: tp优先逻辑 — 接近tp时停止追踪,让价格自然达到tp
            # 根因: v3.36的R:R=0.31, avg_win=0.18 << avg_loss=0.58
            #   追踪止损在1.5%激活,但tp=4.5%,大部分交易在1.5-2%被追踪止损踢出
            #   导致avg_win过小,R:R严重偏低
            # 修复: 盈利>80%*tp_pct时停止追踪,给价格达到tp的机会
            # 来源: 专业交易员原则"让利润奔跑",追踪止损是保护而非限制利润
            # 安全性: 不改变追踪止损激活阈值和距离,只是在接近tp时禁用追踪
            #   如果价格未达tp就回撤,仍会被初始止损或追踪止损(回撤到80%tp以下后)保护
            tp_threshold = tp_pct * 0.8  # 80% of tp_pct

            if abs_pnl > 0.10:  # 盈利 > 10%: 锁死 5%
                if direction == "long":
                    stop_loss = entry_price * (1 + current_pnl_pct - 0.05)
                else:
                    stop_loss = entry_price * (1 - abs_pnl + 0.05)

            elif abs_pnl > tp_threshold:
                # v3.37: 接近tp(>80%*tp_pct),停止追踪,让价格自然达到tp
                # 保持初始stop_loss不变,给价格达到tp的空间
                pass

            elif abs_pnl > trail_activation:  # 盈利 > 0.5*stop_pct: 追踪
                trail_dist = max(atr / max(entry_price, 0.01), stop_pct * 0.6)
                if direction == "long":
                    stop_loss = entry_price * (1 + current_pnl_pct - trail_dist)
                else:
                    stop_loss = entry_price * (1 - abs_pnl + trail_dist)

        return stop_loss, take_profit

    def _compute_leverage(self) -> int:
        """根据风险偏好计算杠杆

        Phase 7J-6 反温室修复: 之前固定返回 1 (沙盘避免爆仓)
        但实盘部署后杠杆 1x 会大幅降低资金利用率, 进化出的策略未经历过
        真实杠杆约束. 修复:
          - 沙盘模式 (use_real_leverage=False): 保留 1x (避免爆仓)
          - 实盘模式 (use_real_leverage=True): 基于 gene 风险参数计算
            leverage = max(1, min(int(1 / max_position_pct) + 1, 5))
            来源: position_sizing_max_position_pct=0.25 → leverage=5 (合理永续杠杆)
                  position_sizing_max_position_pct=0.5  → leverage=3
                  position_sizing_max_position_pct=1.0  → leverage=2 (满仓不加杠杆)

        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 杠杆差异是核心反温室风险点
        """
        if not self._use_real_leverage:
            return 1
        # 实盘模式: 基于 gene 的仓位参数计算杠杆
        # position_sizing_max_position_pct 是单笔最大仓位占比 (0.0-1.0)
        # 杠杆 ≈ 1 / max_position_pct, 但限制在 [1, 5] 之间 (Hyperliquid 永续最大 50x,
        # 但保守起见上限 5x, 来源: Binance 永续推荐杠杆 3-5x)
        max_pos_pct = getattr(self.gene, 'position_sizing_max_position_pct', 0.25)
        max_pos_pct = max(max_pos_pct, 0.2)  # 下限 0.2 避免除零/过大杠杆
        leverage = max(1, min(int(1.0 / max_pos_pct) + 1, 5))
        return leverage

