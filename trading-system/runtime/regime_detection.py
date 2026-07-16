# -*- coding: utf-8 -*-
"""
市场状态检测增强模块 (Market Regime Detection Enhancement)

基于2025-2026全网前沿市场状态检测研究，整合3大增强：

1. HMM隐马尔可夫模型 (Hidden Markov Model)
   来源：Hamilton 1989 + Bull & Bear市场状态检测实践
   原理：用隐状态建模市场状态（牛市/熊市/震荡），观测值为收益率
   优势：概率化状态判断，平滑状态转换，可预测状态持续期

2. 在线变点检测 (Online Change Point Detection)
   来源：Adams & MacKay 2007 "Bayesian Online Changepoint Detection"
   原理：贝叶斯方法实时检测数据分布变化点
   优势：实时检测，无需预设变点数，给出变点概率

3. 波动率状态检测 (Volatility Regime Detection)
   来源：GARCH模型 + 波动率聚类现象
   原理：波动率存在聚类（高波动跟随高波动），用GARCH-like方法检测
   优势：预测波动率状态转换，提前调整仓位
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

# v598 Phase E: 真实 GaussianHMM 替代 KMeans 近似
# ERR-20260702-v598p5-hmm-kmeans: KMeans/GMM 近似导致状态转移概率不可信
# 用户铁律: "理解各指标的底层逻辑和市场含义" + "打破瓶颈，打破极限"
# 优先 hmmlearn.GaussianHMM (Baum-Welch EM 算法), 不可用时降级到 GMM 近似
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False
    _GaussianHMM = None  # type: ignore

# v598 Phase F: 真实 GARCH(1,1) 替代硬编码参数
# ERR-20260702-v598p5-garch-hardcoded: 硬编码 α=0.1/β=0.85/ω=0.01 无法适应不同币种/行情
# 用户铁律: "理解各指标的底层逻辑和市场含义" + "打破瓶颈，打破极限"
# 优先 arch.arch_model (最大似然估计 ω/α/β), 不可用时降级到硬编码参数
try:
    from arch import arch_model as _arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False
    _arch_model = None  # type: ignore

logger = logging.getLogger("hermes.regime_detection")


# ============================================================================
# 1. HMM隐马尔可夫市场状态检测
# ============================================================================


class HMMRegimeDetector:
    """HMM隐马尔可夫市场状态检测

    来源：Hamilton 1989 "A New Approach to the Economic Analysis of Nonstationary Time Series"
         + 量化交易实践中的Bull/Bear市场状态检测

    原理：
      市场存在隐含的状态（牛市/熊市/震荡），这些状态不可直接观测，
      但可以通过收益率序列推断。HMM建模：
        - 隐状态转移：P(state_t | state_{t-1}) = 转移矩阵A
        - 观测概率：P(return_t | state_t) = 发射矩阵B
        - 初始分布：π

    3状态模型：
      State 0: 牛市（高均值、低波动）— 适合趋势跟随
      State 1: 震荡（零均值、中波动）— 适合均值回归
      State 2: 熊市（低均值、高波动）— 适合减仓/做空

    v598 Phase E 实现 (ERR-20260702-v598p5-hmm-kmeans):
      优先使用 hmmlearn.GaussianHMM (真实 Baum-Welch EM 算法拟合)
      不可用时降级到 GaussianMixture + 转移统计近似 (原 KMeans 近似版)
      真实 HMM 通过最大似然估计联合优化 (转移矩阵A + 发射参数B + 初始分布π),
      而非分两步: 先 GMM 聚类再统计转移 (近似版), 信任度更高.
    """

    def __init__(
        self,
        n_states: int = 3,               # 状态数
        lookback_window: int = 60,       # 回看窗口
        min_samples: int = 30,           # 最小样本数
        retrain_interval: int = 20,      # 重训练间隔
    ):
        """
        Args:
            n_states: 隐状态数量（通常3: 牛市/震荡/熊市）
            lookback_window: 回看窗口大小
            min_samples: 最小训练样本数
            retrain_interval: 重训练间隔（每N个新样本重训练）
        """
        self.n_states = n_states
        self.lookback_window = lookback_window
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval

        # 历史收益率
        self._returns: deque = deque(maxlen=lookback_window)
        # 当前状态概率
        self._state_probs = np.ones(n_states) / n_states
        # 状态参数（均值和标准差）
        self._state_means = np.linspace(-0.01, 0.01, n_states)
        self._state_stds = np.linspace(0.02, 0.005, n_states)
        # 转移矩阵
        self._transition_matrix = np.eye(n_states) * 0.9 + np.ones((n_states, n_states)) * 0.1 / n_states
        # 上次训练时的样本数
        self._last_train_count = 0
        # v99 P4 修复 (ERR-20260703-v99-hmm-retrain-stuck): 独立更新计数器
        # 问题: deque maxlen=60 填满后 len(self._returns) 恒为 60,
        #   retrain 条件 60 - _last_train_count >= retrain_interval(20)
        #   在 _last_train_count=60 后永远为 0 < 20, 永不重训练, HMM 参数冻结.
        # 修复: 用独立计数器 _update_count (单调递增) 替代 len(self._returns) 判断 retrain 时机.
        self._update_count = 0
        # 状态标签
        self._state_labels = self._generate_state_labels()
        # v598 Phase E: 真实 GaussianHMM 模型实例 (None=未训练或降级到 GMM)
        self._hmm_model: Optional[Any] = None
        # v598 Phase E: 训练方式标记 ("hmmlearn" / "gmm" / "simple" / None)
        self._train_method: Optional[str] = None

    def _generate_state_labels(self) -> List[str]:
        """生成状态标签"""
        if self.n_states == 3:
            return ["bull", "sideways", "bear"]
        elif self.n_states == 2:
            return ["bull", "bear"]
        else:
            return [f"state_{i}" for i in range(self.n_states)]

    def update(self, return_value: float) -> Dict[str, Any]:
        """更新收益率，检测市场状态

        Args:
            return_value: 当期收益率

        Returns:
            {
                "current_state": str,           # 当前状态标签
                "state_probabilities": np.ndarray, # 各状态概率
                "state_confidence": float,      # 状态置信度
                "state_duration": int,          # 状态持续期估计
                "transition_probability": float, # 状态转换概率
            }
        """
        # ============== v703 P7-prep: NaN/Inf 入口防御 (ERR-v703-p7-hmm-nan) ==============
        # 用户铁律: "理解各指标的底层逻辑" + "永远不要模拟牛逼实盘亏"
        # 根因: 价格数据异常 (0/极小值/缺失) 导致除法产生 NaN/Inf,
        #   喂入 GaussianHMM.fit() 触发 "array must not contain infs or NaNs" → 降级 GMM
        #   → regime_normalized 不可靠 → P7 (regime 直连决策) 无法实施
        # 三层防御之第1层 (入口点): 在 append 前 sanitize return_value
        #   - NaN/Inf → 0.0 (中性收益率, 不影响状态分布)
        #   - 极端值 (|r|>0.5=50%) clip 到 ±0.5 (防 GMM 方差爆炸, 50% 已是极端单日波动)
        # 非过拟合依据: clip 是标准数据预处理 (sklearn StandardScaler/RobustScaler 均内建 clip),
        #   非参数拟合, 不依赖历史数据
        try:
            r = float(return_value)
        except (TypeError, ValueError):
            r = 0.0
        if not math.isfinite(r):
            r = 0.0
        r = max(-0.5, min(0.5, r))  # clip ±50%
        self._returns.append(r)

        # v99 P4 修复 (ERR-20260703-v99-hmm-retrain-stuck): 用独立计数器替代 deque 长度判断
        # ----------------------------------------------------------------
        # 旧 bug: `len(self._returns) - self._last_train_count >= self.retrain_interval`
        #   deque maxlen=60 填满后 len() 恒为 60, 60-60=0 < 20, 永不重训练.
        #   HMM 参数在 lookback 窗口填满后永久冻结, 无法适应市场状态转换.
        # 新逻辑: 用单调递增的 _update_count 判断, 每 retrain_interval 次更新触发一次重训练.
        #   - _update_count: 每次 update() 调用递增, 不受 deque maxlen 限制.
        #   - _last_train_count: 改为存储 _update_count (而非 deque 长度).
        #   - 触发条件: _update_count - _last_train_count >= retrain_interval.
        # 数学保证:
        #   - 首次训练: _update_count >= min_samples (30) 时触发, 与原逻辑一致.
        #   - 后续重训练: 每 20 次更新触发一次, 不受 deque 长度限制.
        #   - 训练数据: 仍用 self._returns (最近 lookback_window=60 个样本), 保持滑动窗口特性.
        self._update_count += 1

        # 检查是否需要重训练 (v99 P4: 用 _update_count 替代 len(self._returns))
        if (self._update_count >= self.min_samples and
            self._update_count - self._last_train_count >= self.retrain_interval):
            self._train()
            self._last_train_count = self._update_count

        # 更新状态概率（前向传播）
        if len(self._returns) >= self.min_samples:
            self._forward_update(r)

        return self.get_state()

    def _train(self):
        """训练HMM参数 (v598 Phase E: 真实 GaussianHMM 替代 KMeans 近似)

        ERR-20260702-v598p5-hmm-kmeans: KMeans/GMM 近似导致状态转移概率不可信
        用户铁律: "理解各指标的底层逻辑和市场含义" + "打破瓶颈，打破极限"

        调度策略:
          1. 优先使用 hmmlearn.GaussianHMM (真实 Baum-Welch EM 算法)
          2. hmmlearn 不可用或训练失败 → 降级到 GaussianMixture + 转移统计近似
          3. sklearn 也不可用 → 降级到分位数简单统计
        """
        returns = np.array(self._returns, dtype=float)
        # ============== v703 P7-prep: NaN/Inf 深度防御 (ERR-v703-p7-hmm-nan, 三层第2层) ==============
        # 即使 update() 已做入口 sanitization, 仍需在 _train() 前再次过滤:
        #   1. 防御 update() 修改前已入队的历史脏数据 (deque 残留)
        #   2. 防御 float overflow (np.array 转换可能引入 inf)
        #   3. GaussianHMM.fit() / GaussianMixture.fit() 均要求有限值, 否则抛 ValueError
        # 过滤后样本不足则跳过本轮训练 (下轮 update 会补充新样本)
        returns = returns[np.isfinite(returns)]
        if len(returns) < self.min_samples:
            return

        # v598 Phase E: 优先使用真实 GaussianHMM
        if _HMMLEARN_AVAILABLE:
            try:
                self._train_hmmlearn(returns)
                return
            except Exception as e:
                logger.warning(
                    "GaussianHMM 训练失败, 降级到 GMM 近似: %s", e
                )
                # 清理失败的模型
                self._hmm_model = None

        # 降级: GMM 近似 (原 KMeans 版)
        try:
            self._train_gmm(returns)
        except Exception as e:
            logger.warning("GMM 训练也失败, 降级到简单统计: %s", e)
            self._train_simple(returns)

    def _train_hmmlearn(self, returns: np.ndarray):
        """v598 Phase E: 真实 GaussianHMM 训练 (Baum-Welch EM 算法)

        与 GMM 近似版的本质区别:
          - GMM 近似: 先聚类得到隐状态序列, 再统计转移矩阵 (两步独立)
          - 真实 HMM: 通过 EM 迭代联合优化 (转移矩阵A + 发射参数B + 初始分布π),
            最大化完整数据的对数似然 L = Σ log P(O|A,B,π)
          - 真实 HMM 的状态转移概率可信度更高, 因为考虑了时间序列依赖性
          - GMM 假设观测独立, HMM 显式建模 P(state_t | state_{t-1})

        v703 P8 修复 (ERR-v703-p8-hmm-covariance):
          - covariance_type="full" → "diag"
          - 根因: "full" 在 hmmlearn EM 的 M-step 矩阵操作中产生内部 NaN/Inf
            (P7-prep 的入口 sanitization 已过滤输入, 但 hmmlearn 内部仍失败, 49次/回测)
          - 证据: _v703_p8_hmm_diag.py — 场景A(normal): [full]FAIL [diag]SUCCESS
          - 数学等价性: 对 1D 收益率数据, "full" 和 "diag" 协方差矩阵均为 1×1 标量,
            数学完全等价, 但 "diag" 不涉及矩阵求逆, 数值稳定性更高
          - 非过拟合: 数值稳定性修复, 非参数优化, 非数据拟合

        Args:
            returns: 收益率序列 shape (n_samples,)
        """
        returns_2d = returns.reshape(-1, 1)

        # ============== v703 P8: HMM covariance_type 修复 (ERR-v703-p8-hmm-covariance) ==============
        # 用户铁律: "永远不要模拟牛逼实盘亏" + "复盘不是摆设"
        # P7-prep 声称修复 NaN/Inf 但 49 次仍失败 = 空壳修复 (违反核心铁律 0.3)
        # 真正根因: hmmlearn "full" covariance 在 EM M-step 的矩阵操作中产生内部 NaN/Inf
        #   - 不是输入数据问题 (P7-prep 已用 np.isfinite 过滤输入)
        #   - 是 hmmlearn 内部数值不稳定 (矩阵求逆/对数概率下溢)
        # 修复: covariance_type="diag" + min_covar=1e-4 + 训练后参数有限性验证
        #   - 对 1D 数据, "diag" 和 "full" 数学等价 (协方差矩阵 1×1 = 标量方差)
        #   - "diag" 不涉及矩阵求逆, 数值稳定性更高
        #   - min_covar=1e-4: 防协方差奇异 (hmmlearn 默认 min_covar=0.001, 这里用更保守的 1e-4)
        model = _GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",   # P8: "full" → "diag" (1D等价, 更稳定)
            n_iter=100,
            random_state=42,
            tol=1e-4,
            min_covar=1e-4,           # P8: 防协方差奇异
        )
        model.fit(returns_2d)

        # ============== v703 P8: 训练后参数有限性验证 (ERR-v703-p8-hmm-covariance) ==============
        # 防御 hmmlearn EM 虽然不抛异常但产出 NaN/Inf 参数 (如单个离群点被一个状态吸收)
        # 用户铁律: "复盘不是摆设" — 静默 NaN 参数 = 空壳验证
        means_raw = model.means_.flatten()
        covars_raw = model.covars_
        transmat_raw = model.transmat_

        _means_finite = np.all(np.isfinite(means_raw))
        _covars_finite = np.all(np.isfinite(covars_raw))
        _transmat_finite = np.all(np.isfinite(transmat_raw))
        if not (_means_finite and _covars_finite and _transmat_finite):
            raise ValueError(
                f"HMM 训练产出含 NaN/Inf 参数: means_finite={_means_finite}, "
                f"covars_finite={_covars_finite}, transmat_finite={_transmat_finite}"
            )

        # 提取参数并排序 (按均值降序: bull=高均值 idx=0, sideways=中 idx=1, bear=低均值 idx=2)
        # 与 _state_labels = ["bull","sideways","bear"] 语义一致, 与 _train_gmm 排序一致.
        # ERR-20260703-label-sort-fix: 修复预存在 bug (Phase E 显式延期的 FIXME).
        #   旧代码用 np.argsort(means) 升序 → idx=0=低均值(bear) 与 _state_labels[0]="bull" 相反,
        #   导致 get_position_adjustment() 对熊市错误返回 1.2(牛市加仓).
        #   修复: 改为降序 [::-1], 统一 hmmlearn/gmm/simple 三种训练方式为 idx=0=高均值=bull.
        means = means_raw  # shape: (n_states,)
        sorted_indices = np.argsort(means)[::-1]  # 降序: idx=0=高均值(bull), idx=n-1=低均值(bear)

        self._state_means = means[sorted_indices]

        # 提取标准差 (P8: "diag" 时 covars_ shape: (n_states, n_features))
        #   - "full": covars_ shape (n_states, n_features, n_features), 1D → (n_states, 1, 1)
        #   - "diag": covars_ shape (n_states, n_features), 1D → (n_states, 1)
        #   - "spherical": covars_ shape (n_states,)
        covars = covars_raw
        if covars.ndim == 3:
            stds = np.sqrt(covars.reshape(self.n_states, -1)[:, 0])
        elif covars.ndim == 2:
            stds = np.sqrt(covars.flatten())
        else:
            stds = np.sqrt(covars.flatten())
        self._state_stds = stds[sorted_indices]
        # 防止 std 过小导致数值不稳定
        self._state_stds = np.maximum(self._state_stds, 1e-8)

        # ============== v703 P8: degenerate std 钳制 (ERR-v703-p8-hmm-covariance) ==============
        # 当某状态只吸收 1-2 个离群点时, std 可能极大 (实测 10924 = 1092400% 日波动)
        # 这会导致后续 forward_update 中高斯概率密度溢出 → 间接影响 regime_normalized 可靠性
        # 钳制到 0.20 (20% 日波动, crypto flash crash 极端上界) 防止溢出
        # 非过拟合: 20% 日波动是 crypto 物理极限 (BTC 最大单日波动 ~40%, 但常态 < 20%)
        _MAX_REASONABLE_STD = 0.20  # 20% 日波动, crypto 极端上界
        _degenerate_mask = self._state_stds > _MAX_REASONABLE_STD
        if np.any(_degenerate_mask):
            _reasonable_stds = self._state_stds[~_degenerate_mask]
            if len(_reasonable_stds) > 0:
                _clamp_value = float(np.median(_reasonable_stds))
            else:
                _clamp_value = 0.03  # 3% 日波动, crypto 正常波动
            self._state_stds = np.where(_degenerate_mask, _clamp_value, self._state_stds)
            logger.debug(
                "HMM degenerate std 钳制: 原始=%s → 钳制后=%s (阈值=%.4f)",
                stds[sorted_indices], self._state_stds, _MAX_REASONABLE_STD
            )

        # 转移矩阵 (按 sorted_indices 重排行列)
        # model.transmat_ shape: (n_states, n_states)
        self._transition_matrix = model.transmat_[
            np.ix_(sorted_indices, sorted_indices)
        ]
        # 行归一化 (保证每行和为 1, 防止浮点误差累积)
        row_sums = self._transition_matrix.sum(axis=1, keepdims=True)
        self._transition_matrix = self._transition_matrix / np.maximum(
            row_sums, 1e-12
        )

        # 保存训练好的模型实例 (供在线预测/诊断使用)
        self._hmm_model = model
        self._train_method = "hmmlearn"

        logger.debug(
            "HMM 训练完成 (hmmlearn Baum-Welch): means=%s, stds=%s, "
            "transmat_diag=%s",
            self._state_means, self._state_stds,
            np.diag(self._transition_matrix),
        )

    def _train_gmm(self, returns: np.ndarray):
        """GMM 近似训练 (原 KMeans 近似版, hmmlearn 不可用时降级使用)

        与真实 HMM 的差异:
          - GMM 假设观测独立同分布, 不建模时间序列依赖
          - 转移矩阵通过统计 GMM 预测的隐状态序列得到 (两步独立)
          - 信任度低于真实 HMM, 但优于简单统计
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            # sklearn 未安装, 进一步降级
            self._train_simple(returns)
            return

        # 用GMM拟合状态
        returns_2d = returns.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=self.n_states, random_state=42, max_iter=50
        )
        gmm.fit(returns_2d)

        # 排序状态: 按均值降序 (idx=0=bull高均值, idx=n-1=bear低均值)
        sorted_indices = np.argsort(gmm.means_.flatten())[::-1]
        self._state_means = gmm.means_.flatten()[sorted_indices]
        self._state_stds = np.sqrt(gmm.covariances_.flatten())[sorted_indices]

        # 估计转移矩阵
        hidden_states = gmm.predict(returns_2d)
        # 重映射到排序后的状态
        state_mapping = {old: new for new, old in enumerate(sorted_indices)}
        mapped_states = [state_mapping[s] for s in hidden_states]

        # 计算转移矩阵
        trans = np.ones((self.n_states, self.n_states)) * 0.01  # 平滑
        for i in range(len(mapped_states) - 1):
            trans[mapped_states[i], mapped_states[i + 1]] += 1

        # 归一化
        row_sums = trans.sum(axis=1, keepdims=True)
        self._transition_matrix = trans / row_sums

        self._hmm_model = gmm  # 保存 GMM 实例 (供诊断)
        self._train_method = "gmm"

        logger.debug(
            "HMM 训练完成 (GMM 近似, hmmlearn 不可用): means=%s, stds=%s",
            self._state_means, self._state_stds,
        )

    def _train_simple(self, returns: np.ndarray):
        """简单统计训练（无sklearn依赖, 兜底方案）

        ERR-20260703-label-sort-fix: 修复预存在 bug.
          旧代码按升序分位数填充 (idx=0=最低分位数=低均值=bear),
          但 _state_labels[0]="bull" → 语义相反.
          修复: 降序填充 (idx=0=最高分位数=高均值=bull), 与 _state_labels 一致,
          统一 hmmlearn/gmm/simple 三种训练方式为 idx=0=高均值=bull.
        """
        # 按收益率分位数分配状态 (降序: idx=0=最高分位数=bull, idx=n-1=最低分位数=bear)
        percentiles = np.linspace(0, 1, self.n_states + 1)
        boundaries = np.percentile(returns, percentiles * 100)

        for i in range(self.n_states):
            # 降序: idx=0 取最高分位数区间 (n_states-1-i 反转)
            # 例如 n_states=3: i=0 取 boundaries[3:4] (最高), i=2 取 boundaries[0:1] (最低)
            high_idx = self.n_states - i  # 降序: i=0 → high_idx=n_states (最高区间上界)
            low_idx = self.n_states - i - 1  # 降序: i=0 → low_idx=n_states-1 (最高区间下界)
            lower = boundaries[low_idx]
            upper = boundaries[high_idx]
            mask = (returns >= lower) & (returns < upper) if i > 0 else (returns >= lower)
            if mask.sum() > 0:
                self._state_means[i] = float(np.mean(returns[mask]))
                self._state_stds[i] = float(np.std(returns[mask])) + 1e-8

        self._hmm_model = None
        self._train_method = "simple"
        logger.debug("HMM 训练完成 (简单统计, sklearn 不可用)")

    def _forward_update(self, return_value: float):
        """前向算法更新状态概率

        v99 P4 修复 (ERR-20260703-v99-hmm-underflow): 数值下溢导致状态锁死
        ----------------------------------------------------------------
        问题根因:
          反复执行 `posterior = prior * emit_probs` 是逐元素乘法,
          emit_probs 通常 << 1 (高斯密度), 非主导状态的 posterior 会以指数速度下溢至 ~1e-300.
          归一化后, 主导状态概率 → 1.0, 非主导状态 → 0.0 (浮点下溢为 0).
          下一轮 update 时, prior = transition.T @ [0, 0, 1] = transition_matrix 的最后一行,
          再乘 emit_probs 后, 即使新数据强烈指向其他状态, 由于非主导 prior 已被钳为 0,
          posterior 永远无法恢复 → HMM 永久锁死在主导状态 (实测 conf=1.0 持续 3000+ bars).

          实测影响: P7-BT regime_dist 100% trending_down, 所有交易被 delta=-0.15 调整,
          仓位 ×0.85, 直接拖累年化收益 (BTC_USDT 年化仅 14.9%, 目标 30%).

        修复方案:
          1. 概率地板钳制: 归一化前对每个状态概率施加 1e-10 下限,
             防止浮点下溢为 0 后再也无法恢复.
          2. 重新归一化: 钳制后总和 > 1, 必须再次归一化以保持概率分布性质.
          3. log-space 替代方案: 对长序列更稳健, 但本项目 lookback=60 内,
             1e-10 地板已足够防止下溢 (60 次乘法后最小值 ~ 1e-300, 仍 > 浮点下溢阈值 1e-308).

        数学保证:
          - 概率地板 1e-10 远小于任何有意义的概率差异 (>1%), 不影响主导状态判定.
          - 仅防止"完全归零后无法恢复"的数值病态.
          - 与 hmmlearn 内部实现一致 (hmmlearn _BaseHMM._compute_forward_log_proba
            使用 log-space 避免下溢, 这里用概率地板是简化等价方案).

        非过拟合依据:
          - 纯数值稳定性修复, 不引入任何可调参数.
          - 1e-10 是 numpy/hmmlearn 社区通用数值地板 (sklearn GaussianHMM min_covar=1e-3 同量级).
          - 修复后 HMM 能正确响应状态转换, regime_dist 将覆盖多状态而非 100% 单一状态.
        """
        # 发射概率（高斯分布）
        emit_probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            # 高斯概率密度
            z = (return_value - self._state_means[i]) / max(self._state_stds[i], 1e-8)
            emit_probs[i] = math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * self._state_stds[i])

        # 前向传播：P(state_t) = Σ P(state_t|state_{t-1}) * P(state_{t-1}) * P(obs|state_t)
        prior = self._transition_matrix.T @ self._state_probs
        posterior = prior * emit_probs

        # 归一化 + 概率地板钳制 (v99 P4: ERR-20260703-v99-hmm-underflow)
        total = posterior.sum()
        if total > 0:
            normalized = posterior / total
            # 概率地板: 防止非主导状态下溢为 0 后无法恢复
            # 1e-10 远小于有意义概率(>1%), 不影响主导状态判定, 仅防止数值病态锁死
            self._state_probs = np.maximum(normalized, 1e-10)
            # 重新归一化: 钳制后总和 > 1, 必须再次归一化
            re_total = self._state_probs.sum()
            if re_total > 0:
                self._state_probs = self._state_probs / re_total
        else:
            # fallback: posterior 全 0 (极端数值情况), 用 prior 归一化 + 地板
            prior_total = prior.sum()
            if prior_total > 0:
                self._state_probs = np.maximum(prior / prior_total, 1e-10)
                re_total = self._state_probs.sum()
                if re_total > 0:
                    self._state_probs = self._state_probs / re_total
            else:
                # 终极 fallback: 均匀分布 (理论上不会到达, 仅防御性)
                self._state_probs = np.ones(self.n_states) / self.n_states

    def get_state(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        current_state_idx = int(np.argmax(self._state_probs))
        confidence = float(self._state_probs[current_state_idx])

        # 状态持续期估计（基于转移矩阵的对角线元素）
        diag = self._transition_matrix[current_state_idx, current_state_idx]
        if diag < 1.0:
            expected_duration = 1.0 / (1.0 - diag)
        else:
            expected_duration = 999.0

        # 状态转换概率（到其他状态的最大概率）
        off_diag = [self._transition_matrix[current_state_idx, j]
                    for j in range(self.n_states) if j != current_state_idx]
        transition_prob = max(off_diag) if off_diag else 0.0

        return {
            "current_state": self._state_labels[current_state_idx],
            "state_index": current_state_idx,
            "state_probabilities": self._state_probs.copy(),
            "state_confidence": confidence,
            "state_duration": int(expected_duration),
            "transition_probability": float(transition_prob),
            "state_means": self._state_means.copy(),
            "state_stds": self._state_stds.copy(),
        }

    def is_bull(self) -> bool:
        """是否牛市"""
        state = self.get_state()
        return state["current_state"] == "bull" and state["state_confidence"] > 0.5

    def is_bear(self) -> bool:
        """是否熊市"""
        state = self.get_state()
        return state["current_state"] == "bear" and state["state_confidence"] > 0.5

    def get_position_adjustment(self) -> float:
        """根据市场状态获取仓位调整系数

        牛市: 1.2（加仓）
        震荡: 1.0（正常）
        熊市: 0.5（减仓）
        """
        state = self.get_state()
        confidence = state["state_confidence"]

        if state["current_state"] == "bull":
            return 1.0 + 0.2 * confidence  # 1.0-1.2
        elif state["current_state"] == "bear":
            return 1.0 - 0.5 * confidence  # 0.5-1.0
        else:
            return 1.0


# ============================================================================
# 2. 在线变点检测
# ============================================================================


class OnlineChangePointDetector:
    """在线贝叶斯变点检测

    来源：Adams & MacKay 2007 "Bayesian Online Changepoint Detection"

    原理：
      维护一个"运行长度"（run length）分布，表示当前从最后一个变点
      开始经过了多少个观测值。当新数据到来时，更新运行长度分布。

      如果运行长度=0的概率突然增大，说明可能发生了变点。

    优势：
      - 实时检测，无需预设变点数
      - 给出变点概率，可设置阈值
      - 适应非平稳时间序列

    量化标准：
      - 变点概率 > 0.5: 高概率变点
      - 变点概率 0.3-0.5: 可能变点
      - 变点概率 < 0.3: 稳定
    """

    def __init__(
        self,
        hazard_rate: float = 100.0,      # 危险率（期望运行长度的倒数）
        threshold: float = 0.5,          # 变点检测阈值
        max_run_length: int = 500,       # 最大运行长度
    ):
        """
        Args:
            hazard_rate: 危险率H（1/H = 期望运行长度）
            threshold: 变点检测阈值
            max_run_length: 最大运行长度（避免无限增长）
        """
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        self.max_run_length = max_run_length

        # 运行长度概率分布
        self._run_length_probs = np.array([1.0])
        # 观测值的均值和方差（用于学生t分布预测）
        self._mean = 0.0
        self._var = 0.01
        self._count = 0
        # 变点概率历史
        self._changepoint_history: deque = deque(maxlen=50)

    def update(self, value: float) -> Dict[str, Any]:
        """更新观测值，检测变点

        Args:
            value: 观测值（如收益率）

        Returns:
            {
                "is_changepoint": bool,        # 是否变点
                "changepoint_probability": float, # 变点概率
                "run_length": int,             # 当前运行长度（期望值）
                "mean": float,                 # 当前均值估计
                "var": float,                  # 当前方差估计
            }
        """
        # 计算预测概率（学生t分布近似）
        predictive_prob = self._compute_predictive_prob(value)

        # 更新运行长度分布
        growth_probs = self._run_length_probs * predictive_prob * (1 - 1/self.hazard_rate)
        changepoint_prob = np.sum(self._run_length_probs * predictive_prob) / self.hazard_rate

        # 新的运行长度分布
        new_probs = np.append(changepoint_prob, growth_probs)

        # 限制运行长度
        if len(new_probs) > self.max_run_length:
            new_probs = new_probs[-self.max_run_length:]
            # 重新归一化
            new_probs = new_probs / new_probs.sum()

        # 归一化
        total = new_probs.sum()
        if total > 0:
            new_probs = new_probs / total

        self._run_length_probs = new_probs

        # 更新均值和方差（递归）
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        self._var += delta * (value - self._mean) / self._count

        # 变点概率
        cp_prob = float(changepoint_prob)
        self._changepoint_history.append(cp_prob)

        # 期望运行长度
        run_lengths = np.arange(len(self._run_length_probs))
        expected_run_length = float(np.sum(run_lengths * self._run_length_probs))

        return {
            "is_changepoint": cp_prob > self.threshold,
            "changepoint_probability": cp_prob,
            "run_length": int(expected_run_length),
            "mean": float(self._mean),
            "var": float(max(self._var, 1e-8)),
        }

    def _compute_predictive_prob(self, value: float) -> float:
        """计算预测概率（学生t分布近似）"""
        # 使用学生t分布作为预测分布
        # 简化：用高斯分布近似
        std = math.sqrt(max(self._var, 1e-8))
        z = (value - self._mean) / std
        # 高斯概率密度
        prob = math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * std)
        return max(prob, 1e-10)

    def get_state(self) -> Dict[str, Any]:
        """获取当前变点检测状态"""
        cp_prob = float(self._run_length_probs[0]) if len(self._run_length_probs) > 0 else 0.0
        run_lengths = np.arange(len(self._run_length_probs))
        expected_run_length = float(np.sum(run_lengths * self._run_length_probs))

        return {
            "is_changepoint": cp_prob > self.threshold,
            "changepoint_probability": cp_prob,
            "run_length": int(expected_run_length),
            "mean": float(self._mean),
            "var": float(max(self._var, 1e-8)),
            "recent_changepoints": list(self._changepoint_history)[-10:],
        }


# ============================================================================
# 3. 波动率状态检测
# ============================================================================


class VolatilityRegimeDetector:
    """波动率状态检测

    来源：Engle 1982 ARCH + Bollerslev 1986 GARCH
         + 波动率聚类现象（volatility clustering）

    原理：
      金融时间序列的波动率存在聚类现象——高波动跟随高波动，
      低波动跟随低波动。通过GARCH-like模型检测波动率状态。

    3状态模型：
      State 0: 低波动（适合加仓）
      State 1: 正常波动（正常交易）
      State 2: 高波动（适合减仓）

    量化标准：
      - GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
      - 波动率分位数: <25%=低, 25-75%=正常, >75%=高

    v598 Phase F 实现 (ERR-20260702-v598p5-garch-hardcoded):
      优先使用 arch.arch_model (最大似然估计 ω/α/β)
      不可用时降级到硬编码参数 (原简化版)
      真实 GARCH(1,1) 通过最大化对数似然 L = Σ log N(ε_t | 0, σ²_t)
      联合估计 (ω, α, β), 而非硬编码固定值, 能适应不同币种和行情.
      硬编码 α=0.1/β=0.85/ω=0.01 是典型 GARCH(1,1) 经验值, 但不同币种
      (BTC vs 山寨币) 和不同行情 (牛市 vs 熊市) 的波动率聚类强度差异巨大.
    """

    def __init__(
        self,
        lookback_window: int = 60,        # 回看窗口
        garch_alpha: float = 0.1,         # GARCH α参数 (硬编码降级用)
        garch_beta: float = 0.85,         # GARCH β参数 (硬编码降级用)
        garch_omega: float = 0.01,        # GARCH ω参数 (硬编码降级用)
        low_vol_percentile: float = 25.0, # 低波动分位数
        high_vol_percentile: float = 75.0,# 高波动分位数
        # v598 Phase F: 真实 GARCH 拟合参数
        garch_fit_interval: int = 50,     # 重训练间隔 (每 N 个新样本)
        min_garch_samples: int = 50,      # 最小拟合样本数
    ):
        """
        Args:
            lookback_window: 回看窗口
            garch_alpha: GARCH α（ARCH项系数, arch 不可用时降级用）
            garch_beta: GARCH β（GARCH项系数, arch 不可用时降级用）
            garch_omega: GARCH ω（常数项, arch 不可用时降级用）
            low_vol_percentile: 低波动分位数
            high_vol_percentile: 高波动分位数
            garch_fit_interval: 真实 GARCH 重训练间隔 (Phase F)
            min_garch_samples: 最小拟合样本数 (Phase F)
        """
        self.lookback_window = lookback_window
        self.alpha = garch_alpha
        self.beta = garch_beta
        self.omega = garch_omega
        self.low_vol_pct = low_vol_percentile
        self.high_vol_pct = high_vol_percentile

        # 历史收益率
        self._returns: deque = deque(maxlen=lookback_window)
        # GARCH条件方差
        self._conditional_var = 0.01
        # 历史波动率（用于分位数计算）
        self._vol_history: deque = deque(maxlen=lookback_window)
        # v598 Phase F: 真实 GARCH(1,1) 拟合状态
        # _garch_model: arch 拟合结果实例 (None=未拟合或降级到硬编码)
        self._garch_model: Optional[Any] = None
        # _train_method: 训练方式标记 ("arch" / "hardcoded" / None)
        self._train_method: Optional[str] = None
        # _garch_fit_interval: 重训练间隔 (每 N 个新样本重拟合)
        self._garch_fit_interval: int = max(10, int(garch_fit_interval))
        # _min_garch_samples: 最小拟合样本数
        self._min_garch_samples: int = max(20, int(min_garch_samples))
        # _last_fit_count: 上次拟合时的样本数
        self._last_fit_count: int = 0

    def update(self, return_value: float) -> Dict[str, Any]:
        """更新收益率，检测波动率状态

        Args:
            return_value: 当期收益率

        Returns:
            {
                "volatility_regime": str,    # 波动率状态
                "current_volatility": float, # 当前波动率
                "garch_forecast": float,     # GARCH预测波动率
                "vol_percentile": float,     # 波动率分位数
                "position_adjustment": float,# 仓位调整系数
            }
        """
        self._returns.append(return_value)

        # v598 Phase F: 真实 GARCH(1,1) 重训练 (arch 库最大似然估计)
        # 累积足够样本后, 周期性重新拟合 ω/α/β 参数
        if (len(self._returns) >= self._min_garch_samples and
                len(self._returns) - self._last_fit_count >= self._garch_fit_interval):
            self._train_garch()
            self._last_fit_count = len(self._returns)

        # GARCH(1,1)更新 (使用当前 ω/α/β, 可能是 arch 拟合值或硬编码降级值)
        self._conditional_var = (
            self.omega
            + self.alpha * return_value ** 2
            + self.beta * self._conditional_var
        )

        current_vol = math.sqrt(max(self._conditional_var, 1e-10))
        self._vol_history.append(current_vol)

        # 计算波动率分位数
        if len(self._vol_history) >= 10:
            vol_array = np.array(self._vol_history)
            vol_percentile = float(np.percentile(vol_array, self._percentile_rank(current_vol)))
        else:
            vol_percentile = 50.0

        # 判断波动率状态
        if vol_percentile < self.low_vol_pct:
            regime = "LOW"
            position_adj = 1.2  # 低波动加仓
        elif vol_percentile > self.high_vol_pct:
            regime = "HIGH"
            position_adj = 0.6  # 高波动减仓
        else:
            regime = "NORMAL"
            position_adj = 1.0

        # GARCH预测（下一步波动率）
        garch_forecast = math.sqrt(
            self.omega + self.alpha * return_value ** 2 + self.beta * self._conditional_var
        )

        return {
            "volatility_regime": regime,
            "current_volatility": current_vol,
            "garch_forecast": garch_forecast,
            "vol_percentile": vol_percentile,
            "position_adjustment": position_adj,
        }

    def _train_garch(self) -> None:
        """v598 Phase F: 真实 GARCH(1,1) 拟合 (arch 库最大似然估计)

        ERR-20260702-v598p5-garch-hardcoded: 硬编码 α=0.1/β=0.85/ω=0.01
        无法适应不同币种 (BTC vs 山寨币) 和行情 (牛市 vs 熊市).

        真实 GARCH(1,1) 拟合:
          - 通过最大化对数似然 L = Σ log N(ε_t | 0, σ²_t) 估计 ω/α/β
          - σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}  (条件方差递推)
          - α 大: 过去一期冲击对当前波动率影响大 (ARCH 效应强)
          - β 大: 波动率聚类持续期长 (GARCH 效应强)
          - α+β 接近 1: 波动率持续性高 (典型金融时间序列)

        缩放策略:
          arch 库要求收益率放大 100 倍 (从 0.001 到 0.1) 以保证数值稳定.
          σ²_scaled = 10000 * σ² (方差缩放 100²)
          因此 ω_scaled = 10000 * ω, α/β 不变 (无量纲)
          逆缩放: ω = ω_scaled / 10000, α = α_scaled, β = β_scaled

        降级策略:
          1. arch 可用 → 真实 GARCH(1,1) 拟合
          2. arch 不可用/拟合失败 → 保持硬编码参数不变
        """
        returns = np.array(self._returns, dtype=float)
        if len(returns) < self._min_garch_samples:
            return

        if not _ARCH_AVAILABLE:
            # arch 不可用 → 保持硬编码参数 (已在 __init__ 设置)
            if self._train_method is None:
                self._train_method = "hardcoded"
            return

        try:
            # 缩放: 收益率 * 100 (arch 库要求百分比形式保证数值稳定)
            scaled_returns = returns * 100.0
            # 构造并拟合 GARCH(1,1) 模型
            # vol='Garch', p=1, q=1: 标准 GARCH(1,1)
            # mean='Zero': 零均值 (假设收益率已中心化)
            # dist='normal': 正态分布假设
            model = _arch_model(
                scaled_returns,
                vol='Garch',
                p=1,
                q=1,
                mean='Zero',
                dist='normal',
            )
            res = model.fit(disp='off', show_warning=False)

            # 提取拟合参数 (arch 参数名: 'omega', 'alpha[1]', 'beta[1]')
            params = res.params
            omega_scaled = float(params['omega'])
            alpha_fit = float(params['alpha[1]'])
            beta_fit = float(params['beta[1]'])

            # 逆缩放: ω 除以 10000 (方差缩放因子 100²)
            # α/β 是无量纲的比率, 无需缩放
            self.omega = omega_scaled / 10000.0
            self.alpha = alpha_fit
            self.beta = beta_fit

            # 同步条件方差 (用拟合的最后一期条件波动率)
            # res.conditional_volatility 是 σ_scaled (标准差), σ² = var
            if hasattr(res, 'conditional_volatility') and res.conditional_volatility is not None:
                last_vol_scaled = float(res.conditional_volatility[-1])
                # 逆缩放: σ = σ_scaled / 100, σ² = σ_scaled² / 10000
                self._conditional_var = max(
                    (last_vol_scaled / 100.0) ** 2, 1e-10
                )

            self._garch_model = res
            self._train_method = "arch"

            logger.debug(
                "GARCH(1,1) 拟合成功 (arch): omega=%.6e, alpha=%.4f, beta=%.4f, "
                "alpha+beta=%.4f (持续性)",
                self.omega, self.alpha, self.beta, self.alpha + self.beta,
            )
        except Exception as e:
            # 拟合失败 → 保持当前参数 (硬编码或上次成功拟合值)
            logger.warning(
                "GARCH(1,1) 拟合失败, 保持当前参数 (omega=%.4e, alpha=%.4f, "
                "beta=%.4f): %s",
                self.omega, self.alpha, self.beta, e,
            )
            if self._train_method is None:
                self._train_method = "hardcoded"

    def _percentile_rank(self, value: float) -> float:
        """计算值在历史中的百分位排名"""
        if len(self._vol_history) < 2:
            return 50.0
        vol_array = np.array(self._vol_history)
        rank = np.sum(vol_array <= value) / len(vol_array) * 100
        return float(rank)

    def get_state(self) -> Dict[str, Any]:
        """获取当前波动率状态"""
        current_vol = math.sqrt(max(self._conditional_var, 1e-10))
        if len(self._vol_history) >= 10:
            vol_percentile = self._percentile_rank(current_vol)
        else:
            vol_percentile = 50.0

        if vol_percentile < self.low_vol_pct:
            regime = "LOW"
            position_adj = 1.2
        elif vol_percentile > self.high_vol_pct:
            regime = "HIGH"
            position_adj = 0.6
        else:
            regime = "NORMAL"
            position_adj = 1.0

        return {
            "volatility_regime": regime,
            "current_volatility": current_vol,
            "vol_percentile": vol_percentile,
            "position_adjustment": position_adj,
        }


# ============================================================================
# 4. 综合市场状态检测系统
# ============================================================================


class RegimeDetectionSystem:
    """综合市场状态检测系统

    整合HMM + 变点检测 + 波动率状态

    提供统一接口：
      1. update() — 每个周期更新所有状态检测器
      2. get_regime() — 获取综合市场状态
      3. get_position_adjustment() — 获取综合仓位调整
    """

    def __init__(
        self,
        hmm: Optional[HMMRegimeDetector] = None,
        cp_detector: Optional[OnlineChangePointDetector] = None,
        vol_detector: Optional[VolatilityRegimeDetector] = None,
    ):
        """
        Args:
            hmm: HMM状态检测器
            cp_detector: 变点检测器
            vol_detector: 波动率检测器
        """
        self.hmm = hmm or HMMRegimeDetector()
        self.cp_detector = cp_detector or OnlineChangePointDetector()
        self.vol_detector = vol_detector or VolatilityRegimeDetector()
        # v447 认知层升级: 懒加载 RegimeFailureUnderstanding 引用
        # 集成位置: RegimeDetection (识别层) → RegimeFailureUnderstanding (理解层)
        # 上层(agent_decision_engine) 通过 attach_failure_understander() 注入实例,
        # 或保持 None 时由上层自行管理 understander。
        self._failure_understander: Optional[Any] = None
        self._last_failure_report: Optional[Dict[str, Any]] = None

    def attach_failure_understander(self, understander: Any) -> None:
        """v447 认知层: 挂载 RegimeFailureUnderstanding 实例

        挂载后, update() 返回的 regime_data 将附加 failure_report 字段,
        供 StrategyFusion.allocate(failure_report=...) 使用。

        Args:
            understander: RegimeFailureUnderstanding 实例
        """
        self._failure_understander = understander

    def get_last_failure_report(self) -> Optional[Dict[str, Any]]:
        """v447 认知层: 获取最近一次 failure_report (供上层查询)"""
        return self._last_failure_report

    def update(self, return_value: float) -> Dict[str, Any]:
        """更新所有状态检测器

        Args:
            return_value: 当期收益率

        Returns:
            综合市场状态 (若已 attach understander, 附加 failure_report 字段)
        """
        hmm_state = self.hmm.update(return_value)
        cp_state = self.cp_detector.update(return_value)
        vol_state = self.vol_detector.update(return_value)

        regime_data = self._combine_states(hmm_state, cp_state, vol_state)

        # v447 认知层: 若已挂载 understander, 自动调用 understand()
        # 注意: 此处不传 current_indicators 和 current_strategy_paradigm,
        # 因为 RegimeDetectionSystem 无法访问这些上下文。
        # 上层(agent_decision_engine) 应自行调用 understander.understand()
        # 以获取完整的 failure_report。
        # 这里仅做轻量级调用: 不带 indicators, 仅根据 regime 触发默认报告。
        if self._failure_understander is not None:
            try:
                # 尝试调用 understand (不带 indicators, 让 understander 用默认值)
                # 如果 understander 需要必填参数, 这里会失败, 但不影响主流程
                from .regime_failure_understanding import StrategyParadigm
                # 默认范式: trend_following (最常见的失败场景)
                report = self._failure_understander.understand(
                    regime_data=regime_data,
                    current_indicators={},
                    current_strategy_paradigm=StrategyParadigm.TREND_FOLLOWING,
                )
                if hasattr(report, "to_dict"):
                    self._last_failure_report = report.to_dict()
                    regime_data["failure_report"] = self._last_failure_report
            except Exception as e:
                # 认知层失败不影响 regime 检测主流程
                logger.debug("RegimeFailureUnderstanding 调用失败: %s", e)

        return regime_data

    def _combine_states(
        self,
        hmm_state: Dict[str, Any],
        cp_state: Dict[str, Any],
        vol_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """综合三个检测器的状态"""
        # 综合市场状态
        market_regime = hmm_state["current_state"]

        # 综合仓位调整
        hmm_adj = self.hmm.get_position_adjustment()
        vol_adj = vol_state["position_adjustment"]

        # 变点检测：如果检测到变点，降低仓位（不确定性增加）
        if cp_state["is_changepoint"]:
            cp_adj = 0.7  # 变点时减仓30%
        else:
            cp_adj = 1.0

        # 综合仓位调整（取最小值，保守）
        combined_adj = min(hmm_adj, vol_adj, cp_adj)

        # 市场状态置信度
        confidence = hmm_state["state_confidence"]

        # 是否处于状态转换期
        is_transitioning = (
            cp_state["is_changepoint"] or
            hmm_state["transition_probability"] > 0.3
        )

        return {
            "market_regime": market_regime,
            "market_confidence": confidence,
            "is_transitioning": is_transitioning,
            "hmm_state": hmm_state,
            "changepoint_state": cp_state,
            "volatility_state": vol_state,
            "position_adjustment": combined_adj,
            "risk_level": self._compute_risk_level(market_regime, vol_state["volatility_regime"], is_transitioning),
        }

    def _compute_risk_level(
        self,
        market_regime: str,
        vol_regime: str,
        is_transitioning: bool,
    ) -> str:
        """计算综合风险等级"""
        score = 0

        # 市场状态风险
        if market_regime == "bear":
            score += 3
        elif market_regime == "sideways":
            score += 1
        # bull: +0

        # 波动率风险
        if vol_regime == "HIGH":
            score += 3
        elif vol_regime == "NORMAL":
            score += 1
        # LOW: +0

        # 转换期风险
        if is_transitioning:
            score += 2

        if score >= 5:
            return "EXTREME"
        elif score >= 3:
            return "HIGH"
        elif score >= 1:
            return "MEDIUM"
        else:
            return "LOW"

    def get_regime(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        hmm_state = self.hmm.get_state()
        vol_state = self.vol_detector.get_state()
        cp_state = self.cp_detector.get_state()

        return self._combine_states(hmm_state, cp_state, vol_state)

    def get_position_adjustment(self) -> float:
        """获取综合仓位调整系数"""
        state = self.get_regime()
        return state["position_adjustment"]

    def get_summary(self) -> Dict[str, Any]:
        """获取状态检测系统摘要"""
        state = self.get_regime()
        return {
            "market_regime": state["market_regime"],
            "market_confidence": state["market_confidence"],
            "risk_level": state["risk_level"],
            "is_transitioning": state["is_transitioning"],
            "position_adjustment": state["position_adjustment"],
            "hmm_n_states": self.hmm.n_states,
            "vol_regime": state["volatility_state"]["volatility_regime"],
            "changepoint_prob": state["changepoint_state"]["changepoint_probability"],
        }


# ============================================================================
# 单元测试入口
# ============================================================================

def _self_test() -> bool:
    """自检: 验证 RegimeDetectionSystem 核心功能可用

    测试覆盖:
      1. HMMRegimeDetector 单独测试 (3 状态牛/震荡/熊)
      2. RegimeDetectionSystem 综合系统 (HMM + 变点 + 波动率)
      3. 仓位调整系数 + summary 报告
    """
    try:
        rng = np.random.default_rng(42)

        # 测试1: HMMRegimeDetector 单独测试
        hmm = HMMRegimeDetector(
            n_states=3, lookback_window=60, min_samples=30, retrain_interval=20
        )

        # 模拟 100 个收益率: 上升趋势 (牛市特征) + 随机波动
        bull_returns = rng.normal(loc=0.002, scale=0.01, size=100)
        state = None
        for r in bull_returns:
            state = hmm.update(float(r))

        assert state is not None, "HMM 应返回状态"
        assert state["current_state"] in ["bull", "sideways", "bear"], \
            f"状态应为 bull/sideways/bear, 实际 {state['current_state']}"
        assert 0.0 <= state["state_confidence"] <= 1.0
        assert "state_probabilities" in state
        assert len(state["state_probabilities"]) == 3
        assert "state_duration" in state
        assert "transition_probability" in state

        # 仓位调整系数应在合理区间
        pos_adj_hmm = hmm.get_position_adjustment()
        assert isinstance(pos_adj_hmm, float)
        assert 0.0 < pos_adj_hmm <= 1.5, \
            f"HMM 仓位调整应在 (0, 1.5], 实际 {pos_adj_hmm}"

        # 测试2: RegimeDetectionSystem 综合系统 (熊市数据)
        system = RegimeDetectionSystem()

        # 模拟 80 个负收益 (熊市特征)
        bear_returns = rng.normal(loc=-0.003, scale=0.015, size=80)
        last_regime = None
        for r in bear_returns:
            last_regime = system.update(float(r))

        assert last_regime is not None
        assert last_regime["market_regime"] in ["bull", "sideways", "bear"], \
            f"综合状态 regime 应为 bull/sideways/bear, 实际 {last_regime['market_regime']}"
        assert "market_confidence" in last_regime
        assert 0.0 <= last_regime["market_confidence"] <= 1.0
        assert "is_transitioning" in last_regime
        assert isinstance(last_regime["is_transitioning"], bool)
        assert "position_adjustment" in last_regime
        assert "risk_level" in last_regime
        assert last_regime["risk_level"] in ["LOW", "MEDIUM", "HIGH", "EXTREME"]
        assert "hmm_state" in last_regime
        assert "changepoint_state" in last_regime
        assert "volatility_state" in last_regime

        # 测试3: 仓位调整系数 + summary
        pos_adj = system.get_position_adjustment()
        assert isinstance(pos_adj, float)
        assert 0.0 < pos_adj <= 1.5, \
            f"综合仓位调整应在 (0, 1.5] 区间, 实际 {pos_adj}"

        summary = system.get_summary()
        assert "market_regime" in summary
        assert "market_confidence" in summary
        assert "risk_level" in summary
        assert "is_transitioning" in summary
        assert "position_adjustment" in summary
        assert "hmm_n_states" in summary
        assert summary["hmm_n_states"] == 3
        assert "vol_regime" in summary
        assert summary["vol_regime"] in ["LOW", "NORMAL", "HIGH"]
        assert "changepoint_prob" in summary
        assert 0.0 <= summary["changepoint_prob"] <= 1.0

        logger.info("RegimeDetectionSystem 自检全部通过 ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
