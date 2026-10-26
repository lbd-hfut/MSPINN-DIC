# MSPINN-DIC server handoff (2026-10-26)

## Repository and runtime

- Server: `ssh a306@192.168.31.29` (existing SSH key).
- Checkout: `/home/a306/01project/MSPINN-DIC`.
- Python: `/home/a306/miniconda3/envs/jax_env/bin/python`.
- Verified JAX 0.6.2 reports `CudaDevice(id=0)`; GPU is RTX 4070 SUPER, approximately 12 GB.
- Continue development on `main`. Preserve `POUassemble` and annotated tag `POUassemble-20260831`. Preserve other remote branches, including `earlystop`.
- Fetch every remote branch with `git fetch origin '+refs/heads/*:refs/remotes/origin/*' --tags`; inspect status before switching or fast-forwarding. Never reset or clean local work to update.
- Production/training code was not modified during the overlap sensitivity experiment. Its implementation is an independent script, `scripts/analyze_overlap_sensitivity.py`.

## Research objective and current conclusion

The original objective is to understand and reduce subdomain-interface strain artifacts while retaining useful photometric fit. Scale separation was an attempted mechanism, not the final scientific objective. No ground truth is available for bbf/1; never infer displacement/strain accuracy from loss or roughness.

Read these documents before proposing the next experiment:

1. `MSPINN_C1_ASSEMBLY_IMPLEMENTATION.md`
2. `MSPINN_STRAIN_INTERFACE_COMPATIBILITY_PHASE.md`
3. `MSPINN_BACKBONE_P2_PHASE.md`
4. `MSPINN_P2_BSPLINE_SCALE_PHASE.md`
5. `MSPINN_PROJECTED_RESIDUAL_PHASE.md`
6. `MSPINN_OVERLAP_SENSITIVITY_PHASE.md`

Preserve the distinction between engineering qualification and scientific support. Projected residual is `MSPINN_PROJECTED_RESIDUAL_QUALIFIED`, but its scientific state is `PROJECTED_SCALE_SEPARATION_NOT_SUPPORTED`. Previous B-spline 8x8 + P2 has R_disp about 0.912 and residual low-frequency fraction about 0.998. Finite polynomial orthogonalization did not solve scale separation.

The latest offline experiment uses cosine 500/3000 checkpoints, central overlap x approximately [290.15,373.85], and 300 nonzero counterfactual perturbations. No model/optimizer parameter updates were made. Baselines reproduce full-ROI losses approximately 43.89876175 / 27.86007500. At the predeclared 1% global/local mean-loss and 2% pointwise absolute-loss-change limits, maximum accepted strain changes were only 1.915% / 1.039%; no point reached the 20% screen. This does not exclude untested directions or combinations.

Matched relative/common perturbation loss curvatures were of the same order. Actual mismatch contraction incurred material photometric costs and strong corrections could worsen whole-overlap roughness. In the 3000 checkpoint, local/window strain terms have normalized inner product -0.93282; the negative cross term cancels 92.71% of their sum of squares. Do not interpret window strain RMS alone as removable error. Taper-induced effects belong partly to the chosen intervention, not necessarily the original artifact.

## Non-Git artifacts

`case/**/results/` and PNG masks are ignored by Git. Git pull alone cannot restore the experiment. The migration copies a separate archive into the checkout, preserving any conflicting preexisting server files as timestamped backups. Its manifest records file sizes and SHA-256 hashes.

The archive contains:

- `case/bbf/1/03.png` (mask; the two BMP images are tracked by Git).
- The complete latest `case/bbf/1/results/overlap_sensitivity_phase/central/` output directory.
- All existing result JSON/CSV analysis records from earlier stages.
- Cosine `model_00000500.jax` and `model_00003000.jax` from `strain_interface_phase/fair_sweep/models/AdaptiveFCN_All_4x1_DIC_ZNSSD_cosine_pou/`.

Other historical checkpoints, MAT/NPZ fields and plots are not claimed to have migrated. If an extension needs them, inventory the server and transfer the missing originals instead of retraining to recreate them.

## Baseline and scan commands (offline evaluation only)

```bash
cd /home/a306/01project/MSPINN-DIC
PY=/home/a306/miniconda3/envs/jax_env/bin/python
"$PY" -c 'import jax; print(jax.__version__); print(jax.devices())'
"$PY" -m scripts.analyze_overlap_sensitivity --baseline-only
"$PY" -m scripts.analyze_overlap_sensitivity --reuse-baseline
```

Do not rerun these blindly over the migrated CPU outputs. First copy/preserve the `central/` outputs or add a separate output destination. CPU/GPU float32 reductions can differ slightly; compare the actual numeric residual and validation tolerances before deciding whether an assertion means a scientific difference. No CUDA/JAX upgrade or environment mutation is needed for the verified environment.

## Copyable instruction for the server Codex

```text
你现在接手 /home/a306/01project/MSPINN-DIC。先读取 SERVER_HANDOFF.md，再完整阅读其中六份阶段文档和机器可读结果。

先核验 main、HEAD、工作区、所有远端分支和 POUassemble-20260831 标签；保留服务器本地改动，禁止 reset --hard 或 clean。使用 /home/a306/miniconda3/envs/jax_env/bin/python，先确认 jax.devices() 是 CUDA GPU。核验迁移 manifest 中的文件 SHA-256，确认 03.png、cosine 500/3000 checkpoint、历史 analysis JSON/CSV 和最新 central 离线结果齐全。不要把 Git pull 当作实验数据迁移。

本轮先完成接收检查，报告代码提交、GPU、资料完整性和数值基线的可复现条件。先不要启动训练或长时间扫描，不要自动升级 CUDA/JAX，也不要提交或推送代码。

研究主线是界面应变问题。最新结论是：单方向离线扫描未找到预设的低光度代价大应变变化方向，3000-step local/window 项存在强抵消。不要继续以 R_disp 或低频占比单独定义成功，不得宣称 accuracy。

接收检查后，提出下一项有限范围离线验证：在固定 checkpoint 上建立六个常数/仿射相对方向的联合光度曲率与应变响应，检查组合方向是否弱受约束，并检查 local/window 抵消的空间分布。区分单方向扫描的阴性结果与联合空间的可辨识性；先给具体方案，等待用户开始执行的指令。
```
