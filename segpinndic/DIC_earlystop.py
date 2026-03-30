from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any, List, Tuple
import numpy as np

@dataclass
class EarlyStopConfig:
    enabled: bool = True
    eval_interval: int = 10
    global_patience: int = 3

    # global consistency thresholds
    cv_thr: float = 0.08
    gap_thr: float = 0.15
    ratio_thr: float = 1.30

    # local stagnation thresholds
    patience: int = 20        # history length for stagnation check
    delta: float = 1e-1       # max allowed change to consider converged

    warmup_epochs = 300
    eps: float = 1e-12

@dataclass
class PartitionStat:
    loss_hist: deque
    last_eval_step: int = -1

@dataclass
class EarlyStopState:
    m: int
    parts: List[PartitionStat]
    total_loss: deque
    global_good_count: int = 0
    should_stop: bool = False
    stop_reason: str = ""
    last_metrics: Dict[str, Any] = field(default_factory=dict)

class EarlyStopManager:
    def __init__(self, cfg: EarlyStopConfig, m: int):
        self.cfg = cfg
        self.m = int(m)
        self.state = EarlyStopState(
            m=self.m,
            parts=[PartitionStat(loss_hist=deque(maxlen=cfg.patience)) for _ in range(self.m)],
            total_loss=deque(maxlen=cfg.patience)
        )

    def need_eval(self, step: int) -> bool:
        if not self.cfg.enabled or step <= 0:
            return False
        return (step % self.cfg.eval_interval) == 0

    def on_eval(
        self,
        step: int,
        active: np.ndarray,      # (m,) in {0,1}
        part_losses: np.ndarray  # (m,), inactive can be nan
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:

        cfg = self.cfg
        st = self.state
        active = np.asarray(active, dtype=int)
        losses = np.asarray(part_losses, dtype=float)

        assert active.shape == (self.m,)
        assert losses.shape == (self.m,)
        assert np.isin(active, [0,1]).all()

        info: Dict[str, Any] = {
            "step": int(step),
            "enabled": bool(cfg.enabled),
            "global": {},
            "local": {},
            "should_stop": False,
            "stop_reason": "",
        }

        if not cfg.enabled:
            st.last_metrics = info
            return active, False, info

        # ---------------------------
        # 1) collect active losses + history update
        # ---------------------------
        active_mask = (active == 1)
        n_active = int(active_mask.sum())

        for i in np.where(active_mask)[0]:
            li = losses[i]
            if np.isfinite(li):
                st.parts[i].loss_hist.append(float(li))
                st.parts[i].last_eval_step = int(step)

        active_losses = losses[active_mask]
        active_losses = active_losses[np.isfinite(active_losses)]
        st.total_loss.append(float(np.mean(active_losses)))

        info["n_active"] = n_active
        info["n_active_valid_loss"] = int(active_losses.size)

        if active_losses.size == 0:
            info["global"]["skipped"] = "no_valid_active_losses"
            st.last_metrics = info
            return active, False, info

        # ---------------------------
        # 2) global consistency check
        # ---------------------------
        mean_l = float(np.mean(active_losses))
        std_l = float(np.std(active_losses))
        cv = std_l / (mean_l + cfg.eps)

        q10 = float(np.quantile(active_losses, 0.10))
        q50 = float(np.quantile(active_losses, 0.50))
        q90 = float(np.quantile(active_losses, 0.90))
        gap = (q90 - q10) / (q50 + cfg.eps)

        lmin = float(np.min(active_losses))
        lmax = float(np.max(active_losses))
        ratio = lmax / (lmin + cfg.eps)

        global_ok = (cv < cfg.cv_thr) and (gap < cfg.gap_thr) and (ratio < cfg.ratio_thr)

        if global_ok:
            st.global_good_count += 1
        else:
            st.global_good_count = 0

        info["global"] = {
            "cv": cv, "gap": gap, "ratio": ratio,
            "thr_cv": cfg.cv_thr, "thr_gap": cfg.gap_thr, "thr_ratio": cfg.ratio_thr,
            "global_ok": bool(global_ok),
            "good_count": int(st.global_good_count),
            "patience": int(cfg.global_patience),
        }

        # ---------------------------
        # 3) local stagnation check
        # ---------------------------
        stagnated_parts = []
        for i, p in enumerate(st.parts):
            if not active[i]:
                continue
            hist = np.asarray(p.loss_hist, dtype=float)
            if hist.size < cfg.patience:
                continue
            if np.max(hist) - np.min(hist) <= cfg.delta:
                stagnated_parts.append(i)
            if np.max(st.total_loss) - np.min(st.total_loss) <= cfg.delta:
                stagnated_parts = [1] * n_active

        info["local"] = {
            "stagnated_parts": stagnated_parts,
            "patience": cfg.patience,
            "delta": cfg.delta
        }

        # ---------------------------
        # 4) early stop decision
        # ---------------------------
        should_stop = (st.global_good_count >= cfg.global_patience) or \
                      (len(stagnated_parts) == n_active)

        if should_stop:
            st.should_stop = True
            st.stop_reason = "global+local_converged"
            info["should_stop"] = True
            info["stop_reason"] = st.stop_reason

        st.last_metrics = info
        return active, should_stop, info