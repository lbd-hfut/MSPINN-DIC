# MSPINN P2 / B-spline Scale Phase

本阶段固定研究对象为 `case/bbf/1`、`[4,1]` 子域、Adam 500、seed-training 0、L-BFGS 0。所有应变均由最终组装位移做空间自动微分得到。该算例没有位移或应变真值，因此本文只比较光度拟合、界面兼容性、场分解和计算成本，不宣称应变准确度。

# 1. Motivation

前一阶段的 global neural MLP 在 500 step 时 loss 为 43.61，但 global displacement RMS 仅 0.641，而 residual RMS 为 2.298；它没有形成“全局趋势 + 小局部修正”的可辨识分工。继续调 MLP 容量或 schedule 不能回答表示空间是否正确，因此本阶段停止调参，改为检验两个结构性假设：完整局部场的 P2 polynomial-reproducing assembly，以及显式低频 cubic B-spline global field。

# 2. Existing Evidence

500-step 固定预算基线为 cosine 43.9031、quintic 52.1661、Hermite 56.8397。它们的 interface strain roughness 分别为 `1.0893e-3`、`1.8871e-4`、`7.1621e-5`。已有 longer-run 只用于过拟合诊断：局部网络继续降低光度 loss 时，raw-local mismatch 和组装应变伪影会放大，故本阶段不把预算增加到 5000。

基线说明界面问题不是单纯位移跳跃：cosine 在 500 step 的 local gradient mismatch 为 0.02134，而组装权重的空间导数会把局部位移差放大到应变。Hermite 明显降低界面粗糙度，但用更高 photometric cost 换取兼容性。

# 3. P2 Full-field Formulation

设四个子域中心为 `c_i`，归一化横坐标为 `z_i=(c_i-c_bar)/s_c`。对每个查询点 `(x,y)`，先在相同 `y`、各自中心 `c_i` 处求完整局部位移 `d_i(c_i,y)`，再以基函数

`b(z) = [1, z, z^2]`

做 query-dependent 加权最小二乘：

`a(x,y) = (B^T W(x) B + lambda I)^(-1) B^T W(x) d(y)`，

`d_P2(x,y) = b(z(x)) a(x,y)`。

权重为中心距离的正函数，`lambda=1e-8` 仅用于 3×3 重建矩阵数值稳定，不是训练 loss。中心、归一化尺度和设计矩阵是静态预计算；query-dependent 权重、正规方程求解和最终求值保留在 JAX 图中。因此 `grad`、`jvp`、`vjp` 以及最终空间应变都穿过 P2 solve。full-field 直接重建完整 local fields；residual 版本使用 zero-offset local outputs 并加到 global field。

# 4. P2 Conditioning

四锚点设计矩阵 rank 为 3。整个 ROI 查询点上的条件数为：minimum 8.1495、median 10.9620、maximum 15.9311；最小奇异值比为 0.06277。该几何在 `lambda=1e-8` 下没有秩亏或病态证据。conditioning 由纯 NumPy 诊断独立计算，并写入 `analysis/p2_bspline_scale_metrics.json`。

# 5. P0/P1/P2 Qualification

单元测试对 P0、P1、P2 的位移值、一阶梯度和小应变逐项做精确重现检查，并验证 `grad/jvp/vjp` 能到达相邻局部网络参数。prototype 的 P1 exact displacement/strain RMS 为 `8.89e-17 / 6.19e-18`；P2 exact 为 `2.37e-16 / 3.66e-17`，处于浮点误差量级。seed、Adam 和 L-BFGS smoke path 使用同一个 assembled forward。

# 6. Perturbed Patch Tests

perturbed tests 覆盖 constant offset mismatch、linear slope mismatch 和 small-amplitude high-frequency mismatch，并同时报告 displacement、strain、interface/interior 和 raw-local mismatch。以 P2 full-field 为例，代表性 trend error 为：constant `2.78e-4`、linear `7.38e-4`、high-frequency `4.12e-4`。误差随扰动幅值线性增大，并对更短波长更敏感；这符合局部差异经空间微分放大的预期，但不等价于真实图像精度。

图：`analysis/08_perturbed_p2_tests.png`。

# 7. P2 Full-field bbf/1 Result

P2 full-field 在 500 step 得到 loss 320.4862、global roughness `1.4215e-4`、interface roughness `7.9586e-5`、RMS jump `9.2802e-5`、max jump `2.2514e-4`、interface/interior ratio 0.8490。raw-local gradient/strain mismatch 为 `0.03291 / 0.01985`，13,840 参数，运行 389.6 s。

它的界面粗糙度接近 Hermite，但 loss 是 Hermite 的 5.64 倍；相对 quintic 虽把 roughness 降至 42%，loss 升至 6.14 倍；相对 cosine roughness 仅为 7.3%，loss 升至 7.30 倍。因此 full-field P2 的实际 verdict 为 **NO**：数学重现与界面兼容性成立，但对 bbf/1 的完整局部场施加了不可接受的低阶横向表示限制。

图：`analysis/01_p2_fullfield_vs_baselines.png`。

# 8. Coarse B-spline Global Field

global field 是二维 tensor-product、degree-3、open-uniform clamped cubic B-spline。物理坐标先按 ROI bounding box 映射到 `[0,1]^2`；端点通过重复 knot 保持定义，基函数用 Cox–de Boor 递推求值。控制点是 `trainable.global_field.controls`，shape 为 `(nx, ny, 2)`；4×4、6×6、8×8 分别增加 32、72、128 个参数。该模块只暴露初始化和场求值，组装模块负责把 global 与 local residual 合成，trainer 只负责 gate/freeze schedule。

# 9. B-spline Residual Representation

最终表示为

`d(x,y) = d_B(x,y) + gate * A(delta d_1,...,delta d_4)`。

`d_B` 是显式 coarse cubic B-spline；局部 residual 的末层零初始化。`A` 分别取 cosine POU 或共享的 P2 reconstruction。没有新增 strain、interface、residual 或 smoothing penalty。组合测试验证 zero residual、zero global、控制点梯度、局部网络梯度，以及 cosine/P2 两条路径均在 JAX 中可微。

# 10. Two-stage Training

Stage 1 为 step 0–199，gate 精确为 0，只训练 B-spline global field；step 200 时 gate 直接变为 1。Stage 2 为 step 200–499，joint 版本同时更新 global 和 local residual，frozen 版本只更新 local residual。快照固定为 0/200/500。

6×6 P2、6×6 cosine 和 6×6 frozen 的 Stage-1 loss 都精确为 1349.1237，证明它们经过相同粗场路径。最初的 frozen 实现暴露了 Adam momentum drift；修复后 trainer 在每个 Stage-2 update 后精确保留 global subtree，step 200→500 control drift RMS 为严格 0，并有专门回归测试。

# 11. 4×4 / 6×6 / 8×8 Scale Study

| Grid + P2 | Loss | Interface roughness | R_disp | R_strain | Raw-local strain mismatch | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| 4×4 | 320.6550 | 7.7420e-5 | 0.9662 | 0.9183 | 0.01937 | 460.2 |
| 6×6 | 264.0465 | 7.4056e-5 | 0.9354 | 0.7714 | 0.01762 | 398.3 |
| 8×8 | 227.6147 | 7.9408e-5 | 0.9124 | 0.7380 | 0.01586 | 395.6 |

4×4 的 Stage-1 loss 1365.1941 且最终 loss 最高，属于相对 underfit。增大到 8×8 改善 loss 和 residual ratio，没有观察到数值发散或明显 roughness 恶化，因此不能仅凭本实验称其“过度灵活”；但它仍没有形成小 residual。若只在 P2 grid 中选择，8×8 是 loss/ratio 最好的诊断点，不是可升级的 production 方法。

图：`analysis/02_bspline_grid_fields.png`、`analysis/05_residual_ratio_vs_grid.png`。

# 12. Cosine-vs-P2 Residual Assembly

相同 6×6 B-spline、初始化和 Stage 1 下，cosine residual 的 loss/roughness/R_disp/R_strain 为 `44.0972 / 1.0857e-3 / 0.9428 / 0.9454`；P2 residual 为 `264.0465 / 7.4056e-5 / 0.9354 / 0.7714`。P2 显著改善界面 roughness 和 strain ratio，但 photometric loss 增加约 6 倍。结论是 **PARTIAL in interface metrics, NO as an overall residual assembly improvement**。

# 13. Freeze-vs-Joint Stage 2

6×6 P2 joint 的 loss/R_disp/R_strain 为 `264.0465 / 0.9354 / 0.7714`；严格 frozen 为 `322.3062 / 0.9485 / 0.7793`。冻结后 control drift 为 0，但 scale separation 没有改善，拟合反而更差。因此冻结不是当前必要机制，也不能解决表示不可辨识性。

# 14. Interface Analysis

所有方法使用相同 ROI、三条实际界面 `x=192.5, 332, 471.5` 和统一应变定义。P2 full-field 与 B-spline+P2 的 assembled traces 连续且 interface roughness 约 `7–8e-5`，但 raw-local left/right values 和 gradients 仍明显不同；P2 是用低阶重建吸收这些不一致，而不是让局部网络本身一致。6×6 cosine 保留低 loss，但 roughness 与 classical cosine 几乎相同。

图：`analysis/04_interface_diagnostics.png`。

# 15. Scale Separation Metrics

**Does explicit B-spline function space solve identifiability? NO.**

4×4/6×6/8×8 P2 的 R_disp 为 0.966/0.935/0.912，R_strain 为 0.918/0.771/0.738，远未达到“小局部修正”。对应 residual/global displacement RMS 比为 14.17、10.41、7.85。8×8 频谱中 global 低频能量占比 0.995，但 residual 自身低频占比也为 0.998；residual/global high-band absolute power ratio 为 18.58。显式 B-spline 确实限制了 global 的频带，却没有阻止 local residual 重新承担低频大振幅趋势。

图：`analysis/03_bspline6_decomposition.png`、`analysis/09_frequency_decomposition.png`。

# 16. Runtime

500-step wall time 为：cosine 1588.3 s、quintic 1601.3 s、Hermite 2390.9 s（历史正式基线）；本阶段 P2 full-field 389.6 s，B-spline 4×4/6×6/8×8 P2 为 460.2/398.3/395.6 s，6×6 cosine 为 162.2 s，6×6 frozen 为 387.9 s。不同阶段运行时受编译缓存和机器状态影响，因此只作为同机记录，不将 wall time 解释为纯算法复杂度。

# 17. Trade-off

loss–roughness Pareto 图显示两条清晰路线：cosine/6×6 cosine 位于低 loss、高 roughness 端；Hermite 与 P2 系列位于低 roughness 端。Hermite 在同等 roughness 量级下 loss 显著优于 P2 full-field 和 B-spline+P2，所以 H1 不成立。B-spline+cosine 几乎复现 classical cosine，说明 coarse field 没有改变主导 residual 的解。

图：`analysis/06_loss_vs_interface_tradeoff.png`、`analysis/07_runtime_vs_interface_tradeoff.png`。

# 18. Scientific Conclusion

1. P2 full-field 不比 Hermite 更有实际价值：roughness 相近但 loss 高 5.64 倍。
2. P2 full-field 不比 quintic 整体更优：界面更平滑，但 photometric cost 高 6.14 倍。
3. 是，P2 的低 interface roughness 伴随不可接受的 photometric cost。
4. coarse B-spline 在函数空间上承担低频场，但在拟合幅值上没有成为主导 global trend。
5. 否，local residual 不是 small correction；R_disp 仍为 0.91–0.97。
6. P2 grid 中 8×8 的 loss 和 ratio 最好，但只能作为诊断配置。
7. 4×4 相对 underfit。
8. 没有证据表明 8×8 已过度 flexible；问题是 residual 仍主导，而非控制点过多。
9. P2 residual 只在界面指标上优于 cosine，在整体 trade-off 上不优。
10. Stage-2 freeze 不必要，且恶化拟合与分离指标。
11. 不升级任何新方法为 production。保留 cosine 作为 photometric reference、Hermite 作为 interface-compatible reference。
12. 最值得继续研究的是“显式 coarse field + 结构上受限的 local correction”，但约束应来自表示/参数化的频带或正交分解，而不是新增应变/interface penalty；在提出可辨识结构前不再扩充正式预算。

工程实现资格通过，但两个科学假设的结论为：**H1 NOT SUPPORTED；H2 的显式低频函数空间动机成立，但当前 additive local residual 仍不可辨识，因此 SCALE SEPARATION HYPOTHESIS NOT SUPPORTED。**

# 19. Limitations

- bbf/1 没有 ground truth，不能由 roughness 或 photometric loss 推断应变准确度。
- 只有一个 ROI、一个图像对、一个 `[4,1]` 几何和一个随机种子；结论是当前算例的机制证据，不是普适性能排名。
- 固定 500 step 用于公平比较，不保证每种表示都达到各自最优点；本阶段按约束不延长预算。
- P2 只沿横向子域链重建，结论不自动推广到二维不规则邻接。
- 频谱按三条 ROI 横向 trace 统计；它能诊断尺度占用，但不是完整二维谱分解。
- wall time 跨历史批次受 JAX 编译和缓存状态影响。
- 未新增任何 strain/interface/residual/smoothing loss，因此观察到的 trade-off 是表示与 assembly 本身的结果。

可审计数据位于 `case/bbf/1/results/p2_bspline_scale_phase/analysis/`：正式表 `formal_500_metrics.csv`、完整指标 `p2_bspline_scale_metrics.json`、每方法 MAT/NPZ fields 和 01–09 图件。工程资格：`MSPINN_P2_BSPLINE_SCALE_QUALIFIED`；科学结论：`SCALE SEPARATION HYPOTHESIS NOT SUPPORTED`。
