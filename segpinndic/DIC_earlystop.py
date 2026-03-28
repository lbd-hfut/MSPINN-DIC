from dataclasses import dataclass, field
from collections import deque
import numpy as np

@dataclass
class PartitionStat:
    # 最近窗口的分区loss历史
    loss_hist: deque
    # 最近一次评估时刻（epoch）
    last_eval_step: int = -1
    # 分区成为active的起始epoch（用于warmup）
    active_since: int = -1
    # 当前是否已冻结（冗余标记，和active数组一致性检查用）
    is_fixed: bool = False
    # 连续“全局一致”计数（用于global_patience）
    # 这个建议放manager全局，也可以放这里做调试
    # global_good_count: int = 0

@dataclass
class EarlyStopState:
    m: int
    trend_patience: int
    # 每个分区一个统计对象
    parts: list[PartitionStat] = field(default_factory=list)
    # 全局判据连续命中次数
    global_good_count: int = 0
    # 是否触发整体验收停止
    should_stop: bool = False
    # 调试信息
    last_metrics: dict = field(default_factory=dict)
    
@dataclass
class EarlyStopConfig:
    enabled: bool = True
    eval_interval: int = 20
    global_patience: int = 3

    # 全局一致判据
    cv_thr: float = 0.08
    gap_thr: float = 0.15
    ratio_thr: float = 1.3

    # 局部冻结判据
    low_loss_quantile: float = 0.30
    trend_patience: int = 30
    min_drop: float = 0.015
    slope_thr: float = 1e-3      # 归一化后阈值
    osc_thr: float = 2.5
    warmup_epochs: int = 100
    freeze_ratio_max: float = 0.2
    min_active_keep: int = 4     # 最少保留active分区数
    
class EarlyStopManager:
    def __init__(self, cfg: EarlyStopConfig, m: int): ...
    
    def on_active_updated(self, step: int, active: np.ndarray):
        """当scheduler给出新的active数组时调用，更新active_since/is_fixed状态。"""

    def on_eval(
        self,
        step: int,
        active: np.ndarray,                 # shape=(m,), 0/1/2
        part_losses: np.ndarray             # shape=(m,), 非active可为nan
    ) -> tuple[np.ndarray, bool, dict]:
        """
        输入当前分区loss，输出:
          - new_active: 可能把部分1改成2
          - should_stop: 是否全局早停
          - info: 指标字典（日志用）
        """

    def need_eval(self, step: int) -> bool:
        """step % eval_interval == 0"""