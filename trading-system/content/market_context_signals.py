# -*- coding: utf-8 -*-
"""Phase 14.16r: 市场上下文信号生成器
基于新数据源(资金费率/OI/情绪/相关性)生成交易信号
"""
import logging
logger = logging.getLogger(__name__)

def compute_funding_rate_signal(funding_rate, weight=0.0):
    """资金费率信号: funding_rate>0.1% 做空信号, <-0.05% 做多信号"""
    if abs(weight) < 0.01:
        return 0.0
    if funding_rate > 0.001:  # 0.1%
        return -weight * min(1.0, funding_rate / 0.003)  # 做空
    elif funding_rate < -0.0005:  # -0.05%
        return weight * min(1.0, abs(funding_rate) / 0.002)  # 做多
    return 0.0

def compute_oi_signal(oi_change_pct, price_change_pct, weight=0.0):
    """OI信号: OI增价格不增=变盘预警"""
    if abs(weight) < 0.01:
        return 0.0
    divergence = oi_change_pct - price_change_pct
    if divergence > 0.1:  # OI增10%但价格不增
        return -weight * min(1.0, divergence / 0.3)  # 做空预警
    return 0.0

def compute_sentiment_signal(fear_greed_value, weight=0.0):
    """情绪信号: >80减仓, <20加仓"""
    if abs(weight) < 0.01:
        return 0.0
    if fear_greed_value > 80:
        return -weight * (fear_greed_value - 80) / 20  # 减仓
    elif fear_greed_value < 20:
        return weight * (20 - fear_greed_value) / 20  # 加仓
    return 0.0

def compute_correlation_penalty(current_positions, correlation_matrix, penalty_strength=0.3):
    """相关性惩罚: 持仓相关性>0.8 返回惩罚分数"""
    if penalty_strength < 0.01 or not current_positions or not correlation_matrix:
        return 0.0
    max_corr = 0.0
    for i, s1 in enumerate(current_positions):
        for s2 in current_positions[i+1:]:
            key = f"{s1}-{s2}" if f"{s1}-{s2}" in correlation_matrix else f"{s2}-{s1}"
            corr = correlation_matrix.get(key, 0.0)
            max_corr = max(max_corr, corr)
    if max_corr > 0.8:
        return penalty_strength * (max_corr - 0.8) / 0.2  # 0-1的惩罚
    return 0.0

def generate_market_context_signals(market_context, gene):
    """统一生成所有市场上下文信号"""
    signals = {}
    signals['funding_rate'] = compute_funding_rate_signal(
        market_context.get('funding_rate', 0.0),
        getattr(gene, 'funding_rate_weight', 0.0)
    )
    signals['oi'] = compute_oi_signal(
        market_context.get('oi_change_pct', 0.0),
        market_context.get('price_change_pct', 0.0),
        getattr(gene, 'oi_weight', 0.0)
    )
    signals['sentiment'] = compute_sentiment_signal(
        market_context.get('fear_greed', 50),
        getattr(gene, 'sentiment_weight', 0.0)
    )
    return signals
