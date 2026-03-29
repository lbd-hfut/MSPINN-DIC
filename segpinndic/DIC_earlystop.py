from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any, Tuple, List
import numpy as np


# ---------------------------
# Config / State
# ---------------------------

@dataclass
class EarlyStopConfig:
    enabled: bool = True
    eval_interval: int = 20
    global_patience: int = 3

    # global consistency thresholds
    cv_thr: float = 0.08
    gap_thr: float = 0.15
    ratio_thr: float = 1.30

    # local freeze thresholds
    low_loss_quantile: float = 0.30
    trend_patience: int = 30
    min_drop: float = 0.015          # net drop ratio
    slope_thr: float = 1e-3          # normalized slope threshold
    osc_thr: float = 2.5             # std / abs(drop_abs)
    warmup_epochs: int = 500

    freeze_ratio_max: float = 0.2    # each eval max freeze portion of active
    min_active_keep: int = 4         # keep at least this many active partitions

    eps: float = 1e-12


@dataclass
class PartitionStat:
    loss_hist: deque
    active_since: int = -1
    is_fixed: bool = False
    last_eval_step: int = -1


@dataclass
class EarlyStopState:
    m: int
    parts: List[PartitionStat]
    global_good_count: int = 0
    should_stop: bool = False
    stop_reason: str = ""
    last_metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Manager
# ---------------------------

class EarlyStopManager:
    def __init__(self, cfg: EarlyStopConfig, m: int):
        self.cfg = cfg
        self.m = int(m)
        self.state = EarlyStopState(
            m=self.m,
            parts=[PartitionStat(loss_hist=deque(maxlen=cfg.trend_patience)) for _ in range(self.m)]
        )

    def on_active_updated(self, step: int, active: np.ndarray):
        active = np.asarray(active, dtype=int)
        assert active.shape == (self.m,)
        assert np.isin(active, [0, 1, 2]).all()

        for i in range(self.m):
            if active[i] == 1 and self.state.parts[i].active_since < 0:
                self.state.parts[i].active_since = int(step)
            if active[i] == 2:
                self.state.parts[i].is_fixed = True
            elif active[i] == 1:
                self.state.parts[i].is_fixed = False

    def need_eval(self, step: int) -> bool:
        """
        建议在“完成一次训练更新后”的 step 调用。
        例如你用 i+1 传入，这样第 eval_interval, 2*eval_interval ... 次触发。
        """
        if not self.cfg.enabled:
            return False
        if step <= 0:
            return False
        return (step % self.cfg.eval_interval) == 0

    def on_eval(
        self,
        step: int,
        active: np.ndarray,      # (m,) in {0,1,2}
        part_losses: np.ndarray  # (m,), inactive can be nan
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        cfg = self.cfg
        st = self.state

        active = np.asarray(active, dtype=int).copy()
        losses = np.asarray(part_losses, dtype=float).copy()

        assert active.shape == (self.m,)
        assert losses.shape == (self.m,)
        assert np.isin(active, [0, 1, 2]).all()

        info: Dict[str, Any] = {
            "step": int(step),
            "enabled": bool(cfg.enabled),
            "global": {},
            "local": {},
            "frozen_ids": [],
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
        fixed_mask = (active == 2)
        n_active = int(active_mask.sum())
        n_fixed = int(fixed_mask.sum())

        # append history for active partitions with finite loss
        for i in np.where(active_mask)[0]:
            li = losses[i]
            if np.isfinite(li):
                st.parts[i].loss_hist.append(float(li))
                st.parts[i].last_eval_step = int(step)

        active_losses = losses[active_mask]
        active_losses = active_losses[np.isfinite(active_losses)]

        info["n_active"] = n_active
        info["n_fixed"] = n_fixed
        info["n_active_valid_loss"] = int(active_losses.size)

        # no valid losses -> skip
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

        should_stop = st.global_good_count >= cfg.global_patience
        if should_stop:
            st.should_stop = True
            st.stop_reason = "global_converged"

        info["global"] = {
            "cv": cv,
            "gap": gap,
            "ratio": ratio,
            "thr_cv": cfg.cv_thr,
            "thr_gap": cfg.gap_thr,
            "thr_ratio": cfg.ratio_thr,
            "global_ok": bool(global_ok),
            "good_count": int(st.global_good_count),
            "patience": int(cfg.global_patience),
        }

        if should_stop:
            info["should_stop"] = True
            info["stop_reason"] = st.stop_reason
            st.last_metrics = info
            return active, True, info

        # ---------------------------
        # 3) local freeze if global not converged
        # ---------------------------
        # candidates among active, low-loss quantile, warmup met, enough history
        # build active-index + corresponding finite loss
        act_ids = np.where(active_mask)[0]
        act_losses = losses[act_ids]
        finite_idx = np.isfinite(act_losses)
        act_ids = act_ids[finite_idx]
        act_losses = act_losses[finite_idx]

        if act_ids.size == 0:
            info["local"]["skipped"] = "no_finite_active_losses"
            st.last_metrics = info
            return active, False, info

        q_low = float(np.quantile(act_losses, cfg.low_loss_quantile))
        candidate_ids = []

        for pid, li in zip(act_ids, act_losses):
            p = st.parts[int(pid)]
            age_ok = (p.active_since >= 0) and ((step - p.active_since) >= cfg.warmup_epochs)
            hist_ok = (len(p.loss_hist) >= cfg.trend_patience)
            low_ok = (li <= q_low)
            not_fixed = (active[pid] == 1)

            if low_ok and age_ok and hist_ok and not_fixed:
                candidate_ids.append(int(pid))

        # freeze budget
        n_active_now = int((active == 1).sum())
        freeze_budget = int(np.ceil(cfg.freeze_ratio_max * max(n_active_now, 1)))
        freeze_budget = max(0, freeze_budget)
        max_can_freeze_by_keep = max(0, n_active_now - cfg.min_active_keep)
        freeze_budget = min(freeze_budget, max_can_freeze_by_keep)

        freeze_scores = []   # (pid, stagnation_votes, details)

        for pid in candidate_ids:
            hist = np.asarray(st.parts[pid].loss_hist, dtype=float)
            # use last trend_patience points
            h = hist[-cfg.trend_patience:]
            x = np.arange(h.size, dtype=float)

            h0 = float(h[0])
            h1 = float(h[-1])
            drop_ratio = (h0 - h1) / (h0 + cfg.eps)   # larger => better decrease
            drop_abs = abs(h0 - h1)

            # normalized slope
            slope = float(np.polyfit(x, h, 1)[0])     # raw slope
            norm_slope = slope / (float(np.mean(h)) + cfg.eps)

            osc = float(np.std(h) / (drop_abs + cfg.eps))

            c1 = (drop_ratio < cfg.min_drop)
            c2 = (norm_slope > -cfg.slope_thr)
            c3 = (osc > cfg.osc_thr)

            votes = int(c1) + int(c2) + int(c3)
            stagnated = (votes >= 2)  # 2/3 voting

            details = {
                "drop_ratio": drop_ratio,
                "norm_slope": norm_slope,
                "osc": osc,
                "c_drop_small": bool(c1),
                "c_slope_weak": bool(c2),
                "c_osc_high": bool(c3),
                "votes": votes,
                "stagnated": bool(stagnated),
            }

            if stagnated:
                # 排序优先冻结“更停滞”的分区：votes高优先，drop更小优先
                score = (votes, -drop_ratio, osc)
                freeze_scores.append((pid, score, details))

        # apply freezing with budget
        freeze_scores.sort(key=lambda t: t[1], reverse=True)

        frozen_ids = []
        trend_debug = {}
        for pid, _, details in freeze_scores:
            if len(frozen_ids) >= freeze_budget:
                break
            if active[pid] != 1:
                continue
            active[pid] = 2
            st.parts[pid].is_fixed = True
            frozen_ids.append(int(pid))
            trend_debug[int(pid)] = details

        info["local"] = {
            "q_low": q_low,
            "low_loss_quantile": cfg.low_loss_quantile,
            "candidate_ids": candidate_ids,
            "freeze_budget": freeze_budget,
            "frozen_count": len(frozen_ids),
            "trend_debug": trend_debug,   # 可以按需关闭，避免日志过大
        }
        info["frozen_ids"] = frozen_ids
        info["should_stop"] = False
        info["stop_reason"] = ""

        st.last_metrics = info
        return active, False, info