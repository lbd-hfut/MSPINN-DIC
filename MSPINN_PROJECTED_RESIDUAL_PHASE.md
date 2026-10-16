# MSPINN Projected Residual Phase

本阶段研究固定在 `case/bbf/1`、`[4,1]` 子域、B-spline `8×8` 全局场和 500-step 公平预算上。目标不是增加损失项，而是把局部网络的表示空间硬限制为低阶多项式空间的正交补，检验这种 function-space 约束能否建立真正的 global/local scale separation。当前案例没有位移或应变 ground truth；本文只讨论表示分解、光度目标、界面相容性和数值性质，不宣称 strain accuracy。

# 1. Motivation

前序 B-spline global branch 并未自动获得低频场的可辨识性。局部 PINN 与 B-spline 都能表示平滑、低频的位移模式，因此优化器仍可把主体形变放进 local branch。前序 8×8 B-spline + P2 assembly 的 `R_disp≈0.912` 已说明：只限制或平滑 global branch 不足以阻止 local branch 重新吸收全局低频成分。

本阶段直接改变局部函数空间：先在每个子域上计算局部 raw output 的低阶多项式投影，再只把正交余量送入 cosine POU assembly。

# 2. Function-space Overlap

设全局空间为 `U_global`，局部网络空间为 `U_local`。当前不可辨识性的来源是两者存在较大的功能重叠：

`U_global ∩ U_local ≠ {0}`。

特别是常数、仿射和部分二次趋势同时可由 B-spline 与局部 MLP 表示。仅改变 B-spline 网格或 assembly 连续性不会消除该交集。硬投影构造受限空间

`U_local,⊥ = (I - P_k) U_local`

并以 `U_global + U_local,⊥` 作为实际训练表示。

# 3. Local Modal Decomposition

诊断对象是已有 cosine full-local 第 500 步 checkpoint（loss `43.9031`）。每个子域使用全部有效 ROI 点、均匀权重和局部归一化坐标 `(ξ,η)∈[-1,1]^2`。

- P0：基底 `[1]`。各局部场的低阶位移 RMS 比例为 `0.417–0.849`，但常数项应变为零。三个界面的 P0 位移 mismatch fraction 分别为 `13.417、0.593、14.306`；大于 1 的值来自 low/high 分量抵消，说明 P0 不是稳定的 mismatch 解释。
- P1：基底 `[1, ξ, η]`。各局部场的低阶位移比例为 `0.9987–0.9994`，低阶应变比例为 `0.168–0.357`。三个界面的低阶 strain mismatch fraction 为 `0.249、0.071、0.204`，而正交 strain mismatch fraction 为 `0.962、1.001、0.932`。
- P2：基底 `[1, ξ, η, ξ², ξη, η²]`。各局部场的低阶位移比例为 `0.99963–0.99977`，低阶应变比例为 `0.624–0.811`。三个界面的低阶 strain mismatch fraction 为 `0.652、0.791、0.579`，正交部分仍为 `0.838、0.533、0.832`。

结论是：低阶模式几乎支配 local displacement 本身，但并不单独支配 interface strain mismatch。P2 比 P1 解释更多 strain mismatch，不过低阶与正交 mismatch 存在明显抵消，不能把分数当作正交能量份额直接相加。

# 4. Projection Mathematics

对归一化采样点 `z_i=(ξ_i,η_i)`、基底矩阵 `Φ` 和归一化均匀权重矩阵 `W`，离散加权 L2 投影系数为

`c = (ΦᵀWΦ + λI)⁻¹ ΦᵀW δu_raw`，

低阶分量与正交分量为

`δu_low(z)=φ(z)c`，

`δu_⊥(z)=δu_raw(z)-δu_low(z)`。

P1 与 P2 对 `u、v` 两个分量使用同一个 coefficient operator。`λ=1e-8` 仅用于线性代数稳定，不进入 photometric loss，也不是正则化 penalty。未正则化 Gram matrix 负责 rank/condition 检查；若请求空间秩不足，operator 会确定性降阶，而不是静默使用病态逆矩阵。

# 5. Orthogonal Local Residual

最终前向表示为

`u = u_B + A_cos[(I-P_k)δu_1_raw,…,(I-P_k)δu_4_raw]`，

`v = v_B + A_cos[(I-P_k)δv_1_raw,…,(I-P_k)δv_4_raw]`。

投影采样固定为每个子域 `16×16=256` 点，不依赖当前训练 batch。query 点上的低阶值由同一系数和 query 的归一化基底直接计算，因此空间 autodiff 看到完整的 `x→ξ→φ(ξ,η)c` 路径。投影逻辑封装在 assembly 内，trainer 只负责阶段调度和指标记录；这使投影成为一个深的、可复用的 representation module，而不是散落在训练器中的特殊分支。

# 6. Differentiability

coefficient solve、低阶重建和 subtraction 全部保留在 JAX 计算图内。测试覆盖：

- 参数到 projection coefficient 的 gradient；
- 参数到 projected output 的 gradient；
- spatial JVP 到最终位移梯度/应变；
- reverse-mode VJP 回到 local parameters；
- Stage 2 中 global 与 local 两条梯度路径同时非零。

最终 strain 始终由 final assembled displacement 的 autodiff 得到，不由独立 strain 网络或离线差分代替。

# 7. Mathematical Qualification

投影测试结果全部通过：

- identity decomposition：`raw = low + orthogonal`；
- P1 exact-mode removal：常数与仿射模式在数值容差内清零；
- P2 exact-mode removal：常数、仿射与二次模式在数值容差内清零；
- high-mode preservation：非多项式高模态保留；
- strain test：投影后空间导数与解析导数一致；
- conditioning/fallback：秩检查、condition number 和确定性降阶路径通过；
- formal checkpoints 的最大 moment residual：P1 `3.48e-7`，P2 `8.72e-7`，均小于 qualification tolerance `1e-6`。

正式 `16×16` 网格的 Gram condition number 为 P1 `2.647`、P2 `15.263`；两者均满秩且没有 fallback。

# 8. Cosine 500-step Modal Diagnosis

cosine checkpoint 的逐界面结论如下：

| Interface x | Full disp mismatch | P1 low / orth disp | P2 low / orth disp | Full strain mismatch | P1 low / orth strain | P2 low / orth strain |
|---:|---:|---:|---:|---:|---:|---:|
| 192.5 | 0.2270 | 0.2729 / 0.1664 | 0.2474 / 0.0948 | 0.003688 | 0.000919 / 0.003548 | 0.002404 / 0.003092 |
| 332.0 | 1.9669 | 2.2049 / 0.3603 | 2.0292 / 0.1238 | 0.003320 | 0.000237 / 0.003323 | 0.002626 / 0.001768 |
| 471.5 | 0.2123 | 0.2714 / 0.1643 | 0.2328 / 0.1074 | 0.003919 | 0.000800 / 0.003654 | 0.002269 / 0.003259 |

P1/P2 确实能抽走 local field 中的大部分多项式位移，但 cosine 模型的 strain mismatch 并非主要来自 P1；P2 也没有让所有界面的正交 mismatch 变小到可忽略。

# 9. P1 Projected Residual Training

固定配置：Adam 500、Stage 1=200、Stage 2=300、seed=0、LBFGS=0、B-spline 8×8、cosine assembly。Stage 1 loss 为 `1335.1021`，最终 loss 为 `50.7658`，Stage 2 降低 `1284.3363`。

P1 的 global displacement RMS 为 `0.2496`；raw residual RMS 为 `9.9038`，投影后 residual RMS 为 `2.2386`，对应 operator 内 reduction `77.4%`。但是相对于 final total，`R_disp=0.9221`，所以 projected residual 仍然是主体而非 small correction。`R_strain=0.9964`。

raw/projected local strain mismatch 为 `0.12190/0.11268`，compatibility gain 仅 `7.56%`。interface roughness 为 `6.8878e-3`，RMS jump 为 `1.8928e-2`。

# 10. P2 Projected Residual Training

P2 使用完全相同预算与训练设置。Stage 1 loss 同为 `1335.1021`，最终 loss 为 `173.9568`，Stage 2 降低 `1161.1453`。

global displacement RMS 为 `0.2694`；raw residual RMS 为 `10.5673`，投影后为 `2.7664`，operator 内 reduction `73.8%`。然而 `R_disp=0.9698`、`R_strain=0.9978`，scale separation 比 P1 更差。raw/projected local strain mismatch 为 `0.21985/0.18880`，compatibility gain `14.12%`。interface roughness 为 `6.0649e-3`，RMS jump 为 `1.4448e-2`。

# 11. Scale Separation Metrics

同规格无投影 baseline 为 `R_disp=0.9365`、`R_strain=0.8805`。P1 只把 `R_disp` 降到 `0.9221`，仍高于前序判据参考值 `0.912`；P2 则升到 `0.9698`。两种投影的 `R_strain` 都接近 1。

这一区分很关键：projection operator 对“同一 raw output 中的指定多项式部分”去除是强且正确的，但训练会在允许的正交空间中重新形成平滑、非多项式、低频模式。因此 operator 内 RMS reduction 并没有转化为 global branch 对 total field 的主导权。

# 12. Interface Metrics

| Method | Interface roughness | RMS jump | Max jump | Raw local strain mismatch | Projected local strain mismatch | Compatibility gain |
|---|---:|---:|---:|---:|---:|---:|
| 8×8 cosine none | 1.0790e-3 | 7.6153e-4 | 2.4729e-3 | 5.0193e-3 | 5.0193e-3 | 0% |
| P1 projected | 6.8878e-3 | 1.8928e-2 | 1.7396e-1 | 1.2190e-1 | 1.1268e-1 | 7.56% |
| P2 projected | 6.0649e-3 | 1.4448e-2 | 4.2170e-2 | 2.1985e-1 | 1.8880e-1 | 14.12% |

投影方案没有改善界面 artifact。P1 roughness 是 none baseline 的约 `6.38×`，P2 约 `5.62×`；P1/P2 的 RMS jump 分别约为 baseline 的 `24.85×/18.97×`。

# 13. Spectral Diagnostics

projected residual 的低频功率比例为：none `0.99843`、P1 `0.99885`、P2 `0.98477`。P1 没有降低低频比例；P2 虽略降，但仍由低频主导且伴随不可接受的 loss 恶化。P1 投影后 high-frequency fraction 从 raw 的 `0.00230` 降到 `0.000534`，说明该有限阶多项式投影没有把局部支路转化成高频 correction，反而留下了更纯的平滑非多项式成分。

# 14. Trade-off

P1 的 photometric loss 相对 none baseline 从 `44.5797` 增至 `50.7658`（约 `+13.9%`），尚能完成有效 Stage-2 correction，但没有建立 scale separation，界面指标显著恶化。P2 loss 增至 `173.9568`（约 baseline 的 `3.90×`），同时 `R_disp`、`R_strain` 和界面指标都不支持更好分解。

因此 P1 是两种投影中较好的工程 trade-off，但不是科学成功；P2 既非必要，也没有优于 P1。

# 15. Failure/Success Analysis

数学与工程实现成功：projection identity、moment orthogonality、可微路径、真实训练 forward 和 checkpoint 均通过。科学结果符合 failure pattern A：去除 P1/P2 后，local residual 仍能用更广泛的低频非多项式模式主导 total field。

P2 还部分呈现 failure pattern B：约束更强但 photometric loss 大幅恶化。由于 main 方法没有先建立 scale separation，failure pattern C 的前提不成立，因此没有运行 optional `P2 projected residual + P2 assembly`。这不是缺失实验，而是遵守预先规定的条件式停止规则。

# 16. Scientific Conclusion

P0/P1/P2 low-order modes 对 local displacement 很重要，但不是当前 interface strain mismatch 和 global/local non-identifiability 的充分根因。P1 投影有效移除了仿射 moments，却未显著降低 `R_disp` 或 residual low-frequency fraction；P2 进一步去除二次 moments 仍未成功，并显著牺牲光度拟合。

因此当前证据不支持“有限阶 polynomial orthogonalization 足以建立 MSPINN scale separation”。global B-spline 仍未成为主体 low-frequency field，projected local residual 也没有成为 small correction。下一阶段若继续该方向，应研究更宽的、与 B-spline global space 直接匹配的离散投影，或 non-polynomial/spectral subspace projection，而不是仅提升 polynomial order。

工程资格：`MSPINN_PROJECTED_RESIDUAL_QUALIFIED`。

科学状态：`PROJECTED_SCALE_SEPARATION_NOT_SUPPORTED`。

# 17. Limitations

- 结论来自单一 `bbf/1`、单一 seed、固定 500-step 公平预算；它足以否定本阶段强成功判据，但不是跨案例统计结论。
- 没有 ground truth，因此不能从 photometric loss 或平滑度推出 displacement/strain accuracy。
- 频谱指标来自 ROI 中三条有效横向 trace，主要用于方法内相对比较。
- 固定 `16×16` 投影网格近似连续子域 L2 内积；虽然 moments 与 conditioning 已验证，仍可能遗漏更细尺度的采样效应。
- polynomial orthogonality 不等于与有限 B-spline global space 精确正交；后者可能是下一阶段更合适的函数空间定义。
- 本阶段没有运行 optional P2 assembly，因为 P1/P2 cosine main 均未满足“先建立合理 scale separation”的触发条件。

正式机器可读结果位于 `case/bbf/1/results/projected_residual_phase/analysis/projected_residual_metrics.json` 和 `formal_500_metrics.csv`；图件 `01–03` 位于 `modal_diagnostics/`，图件 `04–09` 位于 `analysis/`。
