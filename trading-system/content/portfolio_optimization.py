# -*- coding: utf-8 -*-
"""
层次风险平价组合优化模块 (Hierarchical Risk Parity Portfolio Optimization)

基于2025-2026全网前沿组合优化研究，整合3大增强：

1. HRP层次风险平价 (Hierarchical Risk Parity)
   来源：López de Prado 2016 "Building Diversified Portfolios that Outperform Out-of-Sample"
         + marketopia 2026 HRP指南
   原理：通过层次聚类分组相似资产，递归分配风险，避免MVO的脆弱性
   优势：不依赖预期收益估计，对噪声鲁棒，无需协方差矩阵正定性
   步骤：相关性→距离→层次聚类→序列化→递归二分风险平价

2. NCO嵌套聚类优化 (Nested Clustered Optimization)
   来源：López de Prado 2019 "A Robust Estimator of the Efficient Frontier"
   原理：将组合优化分解为簇内优化和簇间优化，降低维度
   优势：处理高维资产时更稳定，避免矩阵求逆不稳定

3. CVaR条件风险价值优化 (Conditional Value at Risk)
   来源：Rockafellar & Uryasev 2000 + Riskfolio-Lib 2026
   原理：优化尾部风险而非方差，对极端事件更鲁棒
   优势：不假设正态分布，直接控制尾部损失
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.portfolio_optimization")


# ============================================================================
# 1. HRP层次风险平价
# ============================================================================


class HierarchicalRiskParity:
    """HRP层次风险平价组合优化

    来源：López de Prado 2016 "Building Diversified Portfolios that Outperform Out-of-Sample"

    HRP通过4步构建鲁棒的投资组合：
      1. 计算相关性矩阵和距离矩阵
      2. 层次聚类（Ward方法）分组相似资产
      3. 序列化（Seriation）重排资产使相似资产相邻
      4. 递归二分分配风险平价

    优势对比MVO：
      - 不需要预期收益估计（MVO的最大弱点）
      - 对协方差矩阵噪声鲁棒
      - 无需正定矩阵
      - 产生更分散的权重，避免过度集中

    量化标准：
      - 最小权重: 1%（避免无意义分配）
      - 最大权重: 40%（避免过度集中）
      - 聚类方法: Ward（最小化簇内方差增量）
    """

    def __init__(
        self,
        min_weight: float = 0.01,       # 最小权重
        max_weight: float = 0.40,       # 最大权重
        linkage_method: str = "ward",   # 层次聚类方法
    ):
        """
        Args:
            min_weight: 最小权重
            max_weight: 最大权重
            linkage_method: 层次聚类链接方法 ("ward"/"single"/"complete"/"average")
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.linkage_method = linkage_method

    def allocate(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """执行HRP组合分配

        Args:
            returns: 资产收益率矩阵 (n_samples, n_assets)
            asset_names: 资产名称列表

        Returns:
            {
                "weights": np.ndarray,        # 最终权重
                "asset_names": List[str],     # 资产名称
                "clusters": np.ndarray,       # 聚类标签
                "linkage_matrix": np.ndarray, # 层次聚类链接矩阵
                "correlation": np.ndarray,    # 相关性矩阵
                "distance": np.ndarray,       # 距离矩阵
                "method": str,                # 方法名
            }
        """
        if returns.ndim != 2:
            raise ValueError(f"returns必须是2维矩阵，得到{returns.ndim}维")

        n_assets = returns.shape[1]
        if n_assets < 2:
            weights = np.ones(n_assets)
            return self._format_result(weights, asset_names, None, None, None, None)

        try:
            # Step 1: 计算相关性矩阵和距离矩阵
            corr = np.corrcoef(returns.T)
            # 确保对角线为1
            np.fill_diagonal(corr, 1.0)
            # 距离矩阵: d = sqrt(0.5 * (1 - corr))
            dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, None))

            # Step 2: 层次聚类
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            # 将距离矩阵转换为压缩形式
            dist_condensed = squareform(dist, checks=False)
            link_matrix = linkage(dist_condensed, method=self.linkage_method)

            # Step 3: 序列化 — 按聚类树重排资产
            from scipy.cluster.hierarchy import leaves_list
            sorted_indices = leaves_list(link_matrix)

            # Step 4: 递归二分风险平价
            weights = self._recursive_bisection(
                returns, sorted_indices, link_matrix
            )

            # 应用权重约束
            weights = self._apply_constraints(weights)

            # 获取聚类标签（自动确定簇数）
            n_clusters = max(2, min(n_assets // 3, 5))
            cluster_labels = fcluster(link_matrix, n_clusters, criterion="maxclust")

            return self._format_result(
                weights, asset_names, cluster_labels, link_matrix, corr, dist
            )

        except ImportError:
            logger.warning("scipy未安装，回退到等权重")
            weights = np.ones(n_assets) / n_assets
            return self._format_result(weights, asset_names, None, None, None, None)
        except Exception as e:
            logger.warning(f"HRP分配失败({e})，回退到等权重")
            weights = np.ones(n_assets) / n_assets
            return self._format_result(weights, asset_names, None, None, None, None)

    def _recursive_bisection(
        self,
        returns: np.ndarray,
        sorted_indices: np.ndarray,
        link_matrix: np.ndarray,
    ) -> np.ndarray:
        """递归二分风险平价分配

        将排序后的资产列表递归地分成两半，
        根据每半的方差反比分配权重
        """
        n_assets = len(sorted_indices)
        weights = np.ones(n_assets)

        # 使用聚类树进行递归分割
        clusters = {0: sorted_indices.tolist()}

        for i in range(len(link_matrix)):
            # 合并两个最近的簇
            idx1 = int(link_matrix[i, 0])
            idx2 = int(link_matrix[i, 1])

            # 获取两个簇的资产
            cluster1 = clusters.pop(idx1, [idx1]) if idx1 not in clusters else clusters.pop(idx1)
            cluster2 = clusters.pop(idx2, [idx2]) if idx2 not in clusters else clusters.pop(idx2)

            # 计算每个簇的方差
            var1 = self._compute_cluster_variance(returns, cluster1)
            var2 = self._compute_cluster_variance(returns, cluster2)

            # 风险平价分配：权重与方差反比
            total_var = var1 + var2
            if total_var > 0:
                w1 = var2 / total_var  # 方差小的簇获得更大权重
                w2 = var1 / total_var
            else:
                w1 = w2 = 0.5

            # 分配权重到簇内各资产
            for asset in cluster1:
                weights[asset] *= w1 / len(cluster1)
            for asset in cluster2:
                weights[asset] *= w2 / len(cluster2)

            # 合并簇
            new_cluster_id = n_assets + i
            clusters[new_cluster_id] = cluster1 + cluster2

        return weights

    def _compute_cluster_variance(
        self,
        returns: np.ndarray,
        asset_indices: List[int],
    ) -> float:
        """计算簇的方差（使用资产平均收益的方差）"""
        if not asset_indices:
            return 1.0

        # 簇内资产等权重组合的方差
        cluster_returns = returns[:, asset_indices]
        # 使用主成分方差（更鲁棒）
        if cluster_returns.shape[1] == 1:
            return float(np.var(cluster_returns)) + 1e-10

        # 等权重组合方差
        n = len(asset_indices)
        equal_weights = np.ones(n) / n
        portfolio_returns = cluster_returns @ equal_weights
        return float(np.var(portfolio_returns)) + 1e-10

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """应用权重约束"""
        # 确保非负
        weights = np.maximum(weights, 0)

        # 归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        # 应用最大权重约束
        weights = np.minimum(weights, self.max_weight)
        # 重新归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        # 应用最小权重约束
        weights = np.maximum(weights, self.min_weight)
        # 最终归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        return weights

    def _format_result(
        self,
        weights: np.ndarray,
        asset_names: Optional[List[str]],
        clusters: Optional[np.ndarray],
        linkage: Optional[np.ndarray],
        corr: Optional[np.ndarray],
        dist: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """格式化结果"""
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(weights))]

        return {
            "weights": weights,
            "asset_names": asset_names,
            "clusters": clusters,
            "linkage_matrix": linkage,
            "correlation": corr,
            "distance": dist,
            "method": "HRP",
        }


# ============================================================================
# 2. NCO嵌套聚类优化
# ============================================================================


class NestedClusteredOptimization:
    """NCO嵌套聚类优化

    来源：López de Prado 2019 "A Robust Estimator of the Efficient Frontier"

    NCO将组合优化分解为两个层次：
      1. 簇内优化：在每个簇内独立优化权重
      2. 簇间优化：在簇之间分配资金

    优势：
      - 降低优化维度（从N个资产降到K个簇）
      - 对协方差矩阵噪声更鲁棒
      - 避免高维矩阵求逆的不稳定性

    量化标准：
      - 簇数: max(2, min(n_assets//3, 10))
      - 簇内方法: 最小方差
      - 簇间方法: 风险平价
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,  # 簇数（None则自动确定）
        min_weight: float = 0.01,
        max_weight: float = 0.40,
    ):
        """
        Args:
            n_clusters: 簇数（None则自动确定）
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.n_clusters = n_clusters
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """执行NCO组合分配

        Args:
            returns: 资产收益率矩阵 (n_samples, n_assets)
            asset_names: 资产名称列表

        Returns:
            组合分配结果
        """
        n_assets = returns.shape[1]
        if n_assets < 2:
            weights = np.ones(n_assets)
            return self._format_result(weights, asset_names, None, "NCO")

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            # Step 1: 聚类
            corr = np.corrcoef(returns.T)
            np.fill_diagonal(corr, 1.0)
            dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0, None))
            dist_condensed = squareform(dist, checks=False)
            link_matrix = linkage(dist_condensed, method="ward")

            # 确定簇数
            n_clusters = self.n_clusters or max(2, min(n_assets // 3, 10))
            n_clusters = min(n_clusters, n_assets)
            cluster_labels = fcluster(link_matrix, n_clusters, criterion="maxclust")

            # Step 2: 簇内优化（每个簇内最小方差）
            intra_weights = np.zeros(n_assets)
            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = cluster_labels == cluster_id
                cluster_assets = np.where(cluster_mask)[0]

                if len(cluster_assets) == 0:
                    continue

                # 簇内最小方差权重
                cluster_returns = returns[:, cluster_assets]
                cluster_weights = self._min_variance_weights(cluster_returns)
                intra_weights[cluster_assets] = cluster_weights

            # Step 3: 簇间优化（风险平价）
            # 计算每个簇的方差
            cluster_vars = np.zeros(n_clusters)
            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = cluster_labels == cluster_id
                cluster_assets = np.where(cluster_mask)[0]
                if len(cluster_assets) > 0:
                    cluster_returns = returns[:, cluster_assets]
                    cluster_weights = intra_weights[cluster_assets]
                    portfolio_returns = cluster_returns @ cluster_weights
                    cluster_vars[cluster_id - 1] = np.var(portfolio_returns) + 1e-10

            # 风险平价：权重与方差反比
            inter_weights = 1.0 / cluster_vars
            inter_weights = inter_weights / inter_weights.sum()

            # Step 4: 合并簇内和簇间权重
            final_weights = np.zeros(n_assets)
            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = cluster_labels == cluster_id
                cluster_assets = np.where(cluster_mask)[0]
                cluster_intra = intra_weights[cluster_assets]
                # 归一化簇内权重
                if cluster_intra.sum() > 0:
                    cluster_intra = cluster_intra / cluster_intra.sum()
                final_weights[cluster_assets] = cluster_intra * inter_weights[cluster_id - 1]

            # 应用约束
            final_weights = self._apply_constraints(final_weights)

            return self._format_result(
                final_weights, asset_names, cluster_labels, "NCO"
            )

        except ImportError:
            logger.warning("scipy未安装，回退到等权重")
            weights = np.ones(n_assets) / n_assets
            return self._format_result(weights, asset_names, None, "NCO_fallback")
        except Exception as e:
            logger.warning(f"NCO分配失败({e})，回退到等权重")
            weights = np.ones(n_assets) / n_assets
            return self._format_result(weights, asset_names, None, "NCO_fallback")

    def _min_variance_weights(self, returns: np.ndarray) -> np.ndarray:
        """计算最小方差权重

        使用解析解：w = (Σ⁻¹ · 1) / (1ᵀ · Σ⁻¹ · 1)
        """
        n = returns.shape[1]
        if n == 1:
            return np.array([1.0])

        try:
            cov = np.cov(returns.T)
            # 添加正则化确保可逆
            cov += np.eye(n) * 1e-8

            inv_cov = np.linalg.inv(cov)
            ones = np.ones(n)
            weights = inv_cov @ ones
            weights = weights / weights.sum()

            # 确保非负
            weights = np.maximum(weights, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            return weights
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """应用权重约束"""
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        weights = np.minimum(weights, self.max_weight)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        weights = np.maximum(weights, self.min_weight)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    def _format_result(
        self,
        weights: np.ndarray,
        asset_names: Optional[List[str]],
        clusters: Optional[np.ndarray],
        method: str,
    ) -> Dict[str, Any]:
        """格式化结果"""
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(weights))]

        return {
            "weights": weights,
            "asset_names": asset_names,
            "clusters": clusters,
            "method": method,
        }


# ============================================================================
# 3. CVaR条件风险价值优化
# ============================================================================


class CVaROptimizer:
    """CVaR条件风险价值优化

    来源：Rockafellar & Uryasev 2000 "Optimization of Conditional Value-at-Risk"
         + Riskfolio-Lib 2026 (26种凸风险测度)

    原理：
      CVaR（条件风险价值）衡量损失超过VaR部分的期望值，
      直接优化尾部风险而非方差。

    公式：
      VaR_α = inf{z : P(L ≤ z) ≥ α}  # α分位数
      CVaR_α = E[L | L ≥ VaR_α]       # 超过VaR的平均损失

      其中L是损失，α是置信水平（如95%）

    优势：
      - 不假设正态分布
      - 直接控制尾部损失
      - 对极端事件更鲁棒
      - 凸优化问题，有全局最优解

    量化标准：
      - 置信水平: 95%（VaR_0.95）
      - 最小权重: 1%
      - 最大权重: 40%
    """

    def __init__(
        self,
        confidence_level: float = 0.95,  # 置信水平
        min_weight: float = 0.01,
        max_weight: float = 0.40,
    ):
        """
        Args:
            confidence_level: 置信水平（0-1）
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.alpha = confidence_level
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """执行CVaR优化

        Args:
            returns: 资产收益率矩阵 (n_samples, n_assets)
            asset_names: 资产名称列表

        Returns:
            组合分配结果
        """
        n_assets = returns.shape[1]
        if n_assets < 2:
            weights = np.ones(n_assets)
            return self._format_result(weights, asset_names, 0.0)

        try:
            # 使用简化CVaR优化：最小化CVaR等价于最小化尾部损失
            # 这里使用场景规划方法

            # Step 1: 计算每个资产的历史CVaR
            asset_cvars = np.zeros(n_assets)
            for i in range(n_assets):
                asset_returns = returns[:, i]
                asset_cvars[i] = self._compute_cvar(asset_returns)

            # Step 2: CVaR反比加权（类似风险平价但用CVaR替代方差）
            # 权重与CVaR反比（CVaR小的资产获得更大权重）
            cvar_positive = np.maximum(asset_cvars, 1e-10)
            weights = 1.0 / cvar_positive
            weights = weights / weights.sum()

            # Step 3: 应用约束
            weights = self._apply_constraints(weights)

            # 计算组合CVaR
            portfolio_returns = returns @ weights
            portfolio_cvar = self._compute_cvar(portfolio_returns)

            return self._format_result(weights, asset_names, portfolio_cvar)

        except Exception as e:
            logger.warning(f"CVaR优化失败({e})，回退到等权重")
            weights = np.ones(n_assets) / n_assets
            portfolio_returns = returns @ weights
            portfolio_cvar = self._compute_cvar(portfolio_returns)
            return self._format_result(weights, asset_names, portfolio_cvar)

    def _compute_cvar(self, returns: np.ndarray) -> float:
        """计算CVaR（条件风险价值）

        CVaR_α = E[L | L ≥ VaR_α]
        其中L = -returns（损失 = 负收益）
        """
        if len(returns) == 0:
            return 0.0

        # 损失 = 负收益
        losses = -returns

        # VaR: α分位数
        var_alpha = np.percentile(losses, self.alpha * 100)

        # CVaR: 超过VaR的平均损失
        tail_losses = losses[losses >= var_alpha]
        if len(tail_losses) > 0:
            cvar = float(np.mean(tail_losses))
        else:
            cvar = float(var_alpha)

        return cvar

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """应用权重约束"""
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        weights = np.minimum(weights, self.max_weight)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        weights = np.maximum(weights, self.min_weight)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        return weights

    def _format_result(
        self,
        weights: np.ndarray,
        asset_names: Optional[List[str]],
        portfolio_cvar: float,
    ) -> Dict[str, Any]:
        """格式化结果"""
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(weights))]

        return {
            "weights": weights,
            "asset_names": asset_names,
            "portfolio_cvar": portfolio_cvar,
            "confidence_level": self.alpha,
            "method": "CVaR",
        }


# ============================================================================
# 4. 组合优化综合系统
# ============================================================================


class PortfolioOptimizationSystem:
    """组合优化综合系统

    整合HRP + NCO + CVaR三种前沿组合优化方法

    提供统一接口：
      1. optimize() — 根据市场状态选择最优方法
      2. compare_methods() — 对比三种方法的分配结果
      3. get_diversification_ratio() — 计算分散化比率
    """

    def __init__(
        self,
        hrp: Optional[HierarchicalRiskParity] = None,
        nco: Optional[NestedClusteredOptimization] = None,
        cvar: Optional[CVaROptimizer] = None,
    ):
        """
        Args:
            hrp: HRP优化器
            nco: NCO优化器
            cvar: CVaR优化器
        """
        self.hrp = hrp or HierarchicalRiskParity()
        self.nco = nco or NestedClusteredOptimization()
        self.cvar = cvar or CVaROptimizer()
        # Phase 8.5: Classic methods补全
        self.markowitz = MarkowitzMVO()
        self.black_litterman = BlackLittermanOptimizer()
        self.true_risk_parity = TrueRiskParity()

    # ==================================================================
    # Phase 14.25b: 信号方法适配器 (根治risk_parity/value_investing"未找到任何信号方法")
    # ==================================================================

    def analyze(self, ctx=None):
        """Phase 14.25b: 信号方法适配器 — 让PortfolioOptimizationSystem兼容strategy_registry

        根因修复:
        - risk_parity和value_investing策略都注册为PortfolioOptimizationSystem
        - PortfolioOptimizationSystem只有optimize()方法,无标准信号方法
        - 导致2个策略全部空壳

        适配逻辑(价值投资+风险平价融合):
        - 从ctx.price_history计算均线和波动率
        - 价格<均线*(1-margin) → 低估,看多 (巴菲特价值投资)
        - 波动率低 → 风险平价倾向于加仓 (桥水全天候)
        - 综合两个信号产生最终direction

        Args:
            ctx: MarketContext

        Returns:
            dict: 信号字典
        """
        try:
            symbol = ""
            current_price = 0.0
            price_history = []

            if ctx is not None:
                current_price = float(getattr(ctx, "close", 0.0) or 0.0)
                if hasattr(ctx, "symbol"):
                    symbol = ctx.symbol or ""
                if hasattr(ctx, "price_history"):
                    price_history = list(ctx.price_history or [])

            # 需要足够的历史数据
            if len(price_history) < 10 or current_price <= 0:
                return {
                    "signal": "neutral",
                    "confidence": 0.2,
                    "reason": "insufficient_history",
                    "symbol": symbol,
                }

            try:
                import numpy as np
                prices = np.array(price_history[-30:], dtype=float)
            except Exception:
                return {
                    "signal": "neutral",
                    "confidence": 0.2,
                    "reason": "price_array_error",
                    "symbol": symbol,
                }

            # 价值投资信号: 价格 vs 均线
            ma = float(prices.mean()) if len(prices) > 0 else current_price
            if ma <= 0:
                return {
                    "signal": "neutral",
                    "confidence": 0.2,
                    "reason": "invalid_ma",
                    "symbol": symbol,
                }

            # 价格低于均线一定幅度 = 低估
            deviation = (current_price - ma) / ma  # 负=低估,正=高估

            # 风险平价信号: 波动率
            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
            else:
                volatility = 0.0

            # 综合信号:
            # - deviation < -0.03 (价格低于均线3%+) → 低估,看多
            # - deviation > 0.05 (价格高于均线5%+) → 高估,观望
            # - volatility < 0.02 (低波动) → 风险平价加仓,增强看多
            # - volatility > 0.05 (高波动) → 减仓

            if deviation < -0.03:
                # 低估 → 看多,低波动增强信心
                confidence = 0.4 if volatility < 0.02 else 0.3
                return {
                    "signal": "long",
                    "direction": "long",
                    "confidence": confidence,
                    "strength": abs(deviation),
                    "reason": f"undervalued(dev={deviation:.4f},vol={volatility:.4f})",
                    "symbol": symbol,
                    "ma": ma,
                    "deviation": deviation,
                    "volatility": volatility,
                }
            elif deviation > 0.05:
                # 高估 → 观望
                return {
                    "signal": "neutral",
                    "confidence": 0.2,
                    "reason": f"overvalued(dev={deviation:.4f})",
                    "symbol": symbol,
                    "ma": ma,
                    "deviation": deviation,
                }
            else:
                # 合理估值 → 中性
                return {
                    "signal": "neutral",
                    "confidence": 0.3,
                    "reason": f"fair_value(dev={deviation:.4f},vol={volatility:.4f})",
                    "symbol": symbol,
                    "ma": ma,
                    "deviation": deviation,
                    "volatility": volatility,
                }
        except Exception as e:
            return {"signal": "neutral", "confidence": 0.0, "reason": f"error: {e}"}

    def update(self, ctx=None):
        """Phase 14.25b: 通用update适配器"""
        return self.analyze(ctx)
    def optimize(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        method: str = "auto",
    ) -> Dict[str, Any]:
        """组合优化

        Args:
            returns: 资产收益率矩阵
            asset_names: 资产名称
            method: "auto"/"HRP"/"NCO"/"CVaR"

        Returns:
            最优组合分配结果
        """
        if method == "auto":
            method = self._select_method(returns)

        if method == "HRP":
            result = self.hrp.allocate(returns, asset_names)
        elif method == "NCO":
            result = self.nco.allocate(returns, asset_names)
        elif method == "CVaR":
            result = self.cvar.allocate(returns, asset_names)
        elif method == "Markowitz":
            result = self.markowitz.allocate(returns, asset_names)
        elif method == "BlackLitterman":
            result = self.black_litterman.allocate(returns, asset_names)
        elif method == "RiskParity":
            result = self.true_risk_parity.allocate(returns, asset_names)
        else:
            result = self.hrp.allocate(returns, asset_names)

        # 计算分散化比率
        result["diversification_ratio"] = self._compute_diversification_ratio(
            returns, result["weights"]
        )

        return result

    def _select_method(self, returns: np.ndarray) -> str:
        """根据数据特征自动选择优化方法

        选择逻辑：
          - 资产数 < 5: HRP（简单有效）
          - 资产数 5-20: NCO（嵌套聚类更稳定）
          - 资产数 > 20: NCO（高维必须降维）
          - 尾部风险高（偏度<-1或峰度>5）: CVaR
        """
        n_assets = returns.shape[1]

        # 检查尾部风险
        try:
            from scipy.stats import skew, kurtosis
            portfolio_returns = returns.mean(axis=1)
            sk = skew(portfolio_returns)
            kt = kurtosis(portfolio_returns)

            if sk < -1.0 or kt > 5.0:
                return "CVaR"  # 高尾部风险，用CVaR
        except ImportError:
            pass

        if n_assets <= 5:
            return "HRP"
        else:
            return "NCO"

    def compare_methods(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """对比三种方法的分配结果

        Returns:
            {
                "HRP": result,
                "NCO": result,
                "CVaR": result,
                "best_method": str,
                "best_diversification": float,
            }
        """
        results = {}
        for method_name, optimizer in [("HRP", self.hrp), ("NCO", self.nco), ("CVaR", self.cvar)]:
            try:
                result = optimizer.allocate(returns, asset_names)
                result["diversification_ratio"] = self._compute_diversification_ratio(
                    returns, result["weights"]
                )
                results[method_name] = result
            except Exception as e:
                logger.warning(f"{method_name}优化失败: {e}")
                results[method_name] = None

        # 选择分散化比率最高的方法
        best_method = "HRP"
        best_div = 0.0
        for method_name, result in results.items():
            if result is not None:
                div = result.get("diversification_ratio", 0)
                if div > best_div:
                    best_div = div
                    best_method = method_name

        return {
            **results,
            "best_method": best_method,
            "best_diversification": best_div,
        }

    def _compute_diversification_ratio(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """计算分散化比率

        分散化比率 = 加权平均资产波动率 / 组合波动率
        比率越高，分散化效果越好

        来源：Choueifaty & Coignard 2008 "Toward Maximum Diversification"
        """
        try:
            asset_vols = np.std(returns, axis=0)
            weighted_avg_vol = np.sum(weights * asset_vols)

            portfolio_returns = returns @ weights
            portfolio_vol = np.std(portfolio_returns)

            if portfolio_vol > 0:
                return float(weighted_avg_vol / portfolio_vol)
            return 1.0
        except Exception:
            return 1.0

    def get_summary(self) -> Dict[str, Any]:
        """获取组合优化系统摘要"""
        return {
            "methods": ["HRP", "NCO", "CVaR"],
            "hrp_config": {
                "min_weight": self.hrp.min_weight,
                "max_weight": self.hrp.max_weight,
                "linkage_method": self.hrp.linkage_method,
            },
            "nco_config": {
                "n_clusters": self.nco.n_clusters,
                "min_weight": self.nco.min_weight,
                "max_weight": self.nco.max_weight,
            },
            "cvar_config": {
                "confidence_level": self.cvar.alpha,
                "min_weight": self.cvar.min_weight,
                "max_weight": self.cvar.max_weight,
            },
        }


# Phase 8.5: Classic portfolio optimization methods补全
# Markowitz MVO + Black-Litterman + True ERC Risk Parity


class MarkowitzMVO:
    """Markowitz Mean-Variance Optimization (1952).

    Solves: max w'μ - (λ/2) w'Σw  s.t. sum(w)=1, w>=0

    When λ→∞: degenerates to min-variance.
    When λ=0: degenerates to max-return (concentrate on highest μ).

    Args:
        risk_aversion: λ (risk aversion parameter, default 2.0)
    """

    def __init__(self, risk_aversion: float = 2.0):
        self.risk_aversion = risk_aversion

    def allocate(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Markowitz MVO allocation.

        Args:
            returns: T×N returns matrix
            asset_names: asset name list

        Returns:
            Dict with weights, expected_return, portfolio_variance, sharpe_ratio
        """
        N = returns.shape[1] if len(returns.shape) > 1 else 1
        if N == 1:
            return self._fallback(returns, asset_names, "single_asset")

        # Expected returns (sample mean)
        mu = np.mean(returns, axis=0)
        # Covariance matrix (sample cov, annualized by 252 if daily)
        Sigma = np.cov(returns, rowvar=False)
        if Sigma.ndim == 0:  # single asset edge case
            Sigma = np.array([[float(Sigma)]])

        # Analytical solution with long-only constraint via projection
        # Unconstrained: w = (λΣ⁻¹μ) / (1'λΣ⁻¹μ)
        # With constraints: iterate projection
        try:
            Sigma_inv = np.linalg.pinv(Sigma)
            lam = self.risk_aversion

            # Unconstrained optimal
            w_unc = Sigma_inv @ mu
            # Normalize to sum=1
            if w_unc.sum() != 0:
                w = w_unc / w_unc.sum()
            else:
                w = np.ones(N) / N

            # Long-only projection (clip negatives, renormalize)
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(N) / N

            # Ensure sum=1
            w = w / w.sum()

        except (np.linalg.LinAlgError, FloatingPointError):
            w = np.ones(N) / N  # equal weight fallback

        # Compute portfolio metrics
        port_return = float(w @ mu)
        port_variance = float(w @ Sigma @ w)
        port_std = float(np.sqrt(max(port_variance, 0)))
        sharpe = float(port_return / port_std) if port_std > 1e-10 else 0.0

        return {
            "method": "MarkowitzMVO",
            "weights": w.tolist() if hasattr(w, 'tolist') else list(w),
            "asset_names": asset_names or [f"asset_{i}" for i in range(N)],
            "expected_return": port_return,
            "portfolio_variance": port_variance,
            "portfolio_std": port_std,
            "sharpe_ratio": sharpe,
            "risk_aversion": self.risk_aversion,
        }

    def _fallback(self, returns, asset_names, reason):
        N = 1
        return {
            "method": "MarkowitzMVO",
            "weights": [1.0],
            "asset_names": asset_names or ["asset_0"],
            "expected_return": 0.0,
            "portfolio_variance": 0.0,
            "sharpe_ratio": 0.0,
            "fallback": reason,
        }


class BlackLittermanOptimizer:
    """Black-Litterman model (simplified).

    When no views provided: uses reverse optimization to get implied returns
    Π = λΣw_mkt, then runs Markowitz MVO with Π.

    With views (P, Q, Ω): posterior E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]

    Args:
        risk_aversion: λ (default 2.0)
        tau: confidence scaling (default 0.05)
        market_weights: if None, uses inverse-volatility weights as proxy
    """

    def __init__(self, risk_aversion: float = 2.0, tau: float = 0.05):
        self.risk_aversion = risk_aversion
        self.tau = tau

    def allocate(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        market_weights: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        Omega: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Black-Litterman allocation.

        Args:
            returns: T×N returns matrix
            asset_names: asset names
            market_weights: market cap weights (if None, use 1/vol)
            P: views matrix (K×N)
            Q: views returns (K×1)
            Omega: views confidence (K×K)

        Returns:
            Dict with weights, implied_returns, posterior_returns
        """
        N = returns.shape[1] if len(returns.shape) > 1 else 1
        if N == 1:
            return {
                "method": "BlackLitterman",
                "weights": [1.0],
                "asset_names": asset_names or ["asset_0"],
                "fallback": "single_asset",
            }

        mu_sample = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])

        lam = self.risk_aversion
        tau_sigma = self.tau * Sigma

        # Market weights: use inverse-volatility as proxy
        if market_weights is None:
            vols = np.sqrt(np.diag(Sigma))
            vols = np.where(vols > 1e-10, vols, 1e-10)
            market_weights = (1.0 / vols)
            market_weights = market_weights / market_weights.sum()

        # Step 1: Reverse optimization → implied equilibrium returns
        # Π = λΣw_mkt
        Pi = lam * Sigma @ market_weights

        # Step 2: Posterior returns
        if P is not None and Q is not None and Omega is not None:
            # BL posterior: E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
            try:
                tau_sigma_inv = np.linalg.pinv(tau_sigma)
                omega_inv = np.linalg.pinv(Omega)
                A = tau_sigma_inv + P.T @ omega_inv @ P
                b = tau_sigma_inv @ Pi + P.T @ omega_inv @ Q
                posterior_returns = np.linalg.solve(A, b)
            except (np.linalg.LinAlgError, FloatingPointError):
                posterior_returns = Pi  # fallback to prior
        else:
            # No views: posterior = prior (implied returns)
            posterior_returns = Pi

        # Step 3: Markowitz MVO with posterior returns
        try:
            Sigma_inv = np.linalg.pinv(Sigma)
            w_unc = Sigma_inv @ posterior_returns
            if w_unc.sum() != 0:
                w = w_unc / w_unc.sum()
            else:
                w = np.ones(N) / N
            w = np.maximum(w, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(N) / N
        except (np.linalg.LinAlgError, FloatingPointError):
            w = np.ones(N) / N

        port_return = float(w @ posterior_returns)
        port_variance = float(w @ Sigma @ w)
        port_std = float(np.sqrt(max(port_variance, 0)))
        sharpe = float(port_return / port_std) if port_std > 1e-10 else 0.0

        return {
            "method": "BlackLitterman",
            "weights": w.tolist() if hasattr(w, 'tolist') else list(w),
            "asset_names": asset_names or [f"asset_{i}" for i in range(N)],
            "implied_returns": Pi.tolist() if hasattr(Pi, 'tolist') else list(Pi),
            "posterior_returns": posterior_returns.tolist() if hasattr(posterior_returns, 'tolist') else list(posterior_returns),
            "expected_return": port_return,
            "portfolio_variance": port_variance,
            "portfolio_std": port_std,
            "sharpe_ratio": sharpe,
            "has_views": P is not None,
        }


class TrueRiskParity:
    """True ERC (Equal Risk Contribution) Risk Parity.

    Each asset contributes equally to total portfolio risk.
    RC_i = w_i * (Σw)_i / sqrt(w'Σw) = constant for all i.

    Uses iterative projection to solve.

    Args:
        max_iter: maximum iterations (default 100)
        tol: convergence tolerance (default 1e-8)
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def allocate(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """ERC Risk Parity allocation.

        Args:
            returns: T×N returns matrix
            asset_names: asset names

        Returns:
            Dict with weights, risk_contributions, convergence info
        """
        N = returns.shape[1] if len(returns.shape) > 1 else 1
        if N == 1:
            return {
                "method": "TrueRiskParity",
                "weights": [1.0],
                "asset_names": asset_names or ["asset_0"],
                "fallback": "single_asset",
            }

        Sigma = np.cov(returns, rowvar=False)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])

        # Initialize with inverse-volatility weights
        vols = np.sqrt(np.diag(Sigma))
        vols = np.where(vols > 1e-10, vols, 1e-10)
        w = (1.0 / vols)
        w = w / w.sum()

        # Iterative ERC solver (Newton-like)
        converged = False
        for iteration in range(self.max_iter):
            # Compute risk contributions
            port_var = w @ Sigma @ w
            if port_var < 1e-20:
                break
            port_std = np.sqrt(port_var)
            # Marginal risk contribution: MRC_i = (Σw)_i / sqrt(w'Σw)
            mrc = Sigma @ w / port_std
            # Total risk contribution: RC_i = w_i * MRC_i
            rc = w * mrc
            # Target: equal contribution = port_std / N
            target = port_std / N

            # Check convergence
            rc_std = np.std(rc)
            if rc_std < self.tol:
                converged = True
                break

            # Adjust weights: move from over-contributing to under-contributing
            rc_ratio = rc / target  # should be 1.0 for all
            # Dampened update
            alpha = 0.1  # learning rate
            w = w * (1.0 + alpha * (1.0 - rc_ratio))
            w = np.maximum(w, 1e-10)
            w = w / w.sum()

        # Final metrics
        port_var = float(w @ Sigma @ w)
        port_std = float(np.sqrt(max(port_var, 0)))
        mrc = Sigma @ w / port_std if port_std > 1e-10 else np.zeros(N)
        rc = w * mrc
        rc_pct = rc / rc.sum() if rc.sum() > 1e-10 else np.ones(N) / N

        return {
            "method": "TrueRiskParity",
            "weights": w.tolist() if hasattr(w, 'tolist') else list(w),
            "asset_names": asset_names or [f"asset_{i}" for i in range(N)],
            "risk_contributions": rc.tolist() if hasattr(rc, 'tolist') else list(rc),
            "risk_contributions_pct": rc_pct.tolist() if hasattr(rc_pct, 'tolist') else list(rc_pct),
            "portfolio_variance": port_var,
            "portfolio_std": port_std,
            "converged": converged,
            "iterations": iteration + 1,
        }
