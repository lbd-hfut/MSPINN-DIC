# MSPINN-DIC C1 Assembly Implementation and Qualification

## 1. Objective

This work isolates the effect of the global multi-subdomain assembly space on
the displacement and strain fields. The target is to reduce strain artifacts
near overlap/interface regions without adding strain-continuity, gradient, or
smoothness penalties to the DIC loss. Four assemblies are compared under the
same local networks, images, ROI, optimizer, initialization seed, subdomain
geometry, and training budget:

1. `cosine_pou`
2. `quintic_pou`
3. `hermite_1d_c1`
4. `bicubic_hermite_c1`

The distinction is essential: a smooth scalar window is not a C1-conforming
field. The two Hermite methods explicitly use derivatives of local neural
fields in the assembled displacement evaluated by the photometric loss.

## 2. Original Architecture

`segpinndic.DIC_analysis.main()` reads the images and mask, constructs one
`Constants` object per ROI, and selects `FBPINNTrainer` when the product of
`n_subdomains` exceeds one. The problem samples every ROI pixel. The original
`FBPINN_model()` evaluated all point/model pairs, multiplied each local output
by a squared raised-cosine window, accumulated with `segment_sum`, and divided
by the accumulated weight:

$$
\mathbf u(\mathbf x)=
\frac{\sum_i w_i(\mathbf x)\mathbf u_i(\mathbf x)}
     {\sum_i w_i(\mathbf x)}.
$$

The ZNSSD loss warps the deformed image with this assembled displacement.
Prediction differentiates the same assembled field by chained JAX JVPs, then
computes

$$
\varepsilon_{xx}=u_x,\qquad
\varepsilon_{xy}=\tfrac12(u_y+v_x),\qquad
\varepsilon_{yy}=v_y.
$$

Before this change, seed fitting bypassed the global assembly and supervised
individual local outputs. That inconsistency has been removed.

## 3. New Assembly Architecture

`segpinndic/DIC_assemblies.py` is a deep module with one external seam:

- `Assembly.init_params()` stores static assembly options.
- `Assembly.build_plan()` prepares point ownership and geometry outside the
  jitted optimization step.
- `Assembly.assemble()` evaluates the differentiable global displacement.

`Constants -> assembly registry -> get_inputs/build_plan -> FBPINN_model`
provides selection without a method cascade inside the trainer. The same
`FBPINN_model` is reused by seed fitting, Adam, L-BFGS, and prediction. Geometry
and ownership are computed once per point batch; the jitted path contains only
JAX operations. This adapter-plus-prebuilt-plan design also preserves the exact
legacy cosine calculation.

## 4. Cosine POU

The cosine implementation is retained through
`RectangularDecompositionND.window_fn`, including the repository's original
squared raised-cosine window. Its normalized POU formula and `segment_sum`
ordering are unchanged. If `assembly_method` is absent, `cosine_pou` is chosen
and the historical output run name is preserved. A numerical regression test
compares the adapter with a direct implementation of the old formula.

## 5. Quintic Smoothstep POU

The compact C2 window uses

$$
S(t)=6t^5-15t^4+10t^3,
$$

with $S(0)=0$, $S(1)=1$, and zero first and second derivatives at both ends.
For each axis it constructs a complete rise/core/fall profile:

$$
w_d(x_d)=\min\left[
S\!\left(\frac{x_d-x_{\min}}{w_{\min}}\right),
S\!\left(\frac{x_{\max}-x_d}{w_{\max}}\right)
\right],
$$

clamped to zero outside support. The two-dimensional weight is the tensor
product. The final displacement remains a normalized POU, with a small
denominator guard configured by `assembly_eps`.

## 6. 1D Hermite C1

For an overlap $x\in[x_L,x_R]$, $t=(x-x_L)/L$ and $L=x_R-x_L$. For each
component $q\in\{u,v\}$, the transition is

$$
q_H=h_{00}q_L+h_{10}Lq_{x,L}+h_{01}q_R+h_{11}Lq_{x,R},
$$

where

$$
h_{00}=2t^3-3t^2+1,\quad h_{10}=t^3-2t^2+t,
$$
$$
h_{01}=-2t^3+3t^2,\quad h_{11}=t^3-t^2.
$$

`q_L,qx_L` are evaluated from the left network at $(x_L,y)$ and
`q_R,qx_R` from the right network at $(x_R,y)`. JAX JVPs compute the endpoint
derivatives without stopping gradients to network parameters. This guarantees
matching endpoint values and physical-coordinate x derivatives. A point in an
overlap is controlled only by this Hermite transition; a point in a core is
controlled directly by its nearest local network. There is no POU division and
no simultaneous POU/Hermite double assembly.

## 7. Bicubic Hermite C1

This is a real two-dimensional bicubic implementation, not an alias of the 1D
method. A conforming rectangular cell uses 16 corner DOFs for each displacement
component:

$$q,\quad q_x,\quad q_y,\quad q_{xy}.$$

It interpolates in x and then y, algebraically equivalent to the standard
Hermite matrix form. Physical dimensions are preserved by the factors
$L_x$, $L_y$, and $L_xL_y$ for the derivative DOFs. Corner values and
directional derivatives are obtained from local neural fields with nested JVPs.

For the qualified `[nx,1]` topology, a compatible C1 x-strip determines corner
DOFs; core corners use one local network and transition corners use the 1D
Hermite strip. Shared corners are deterministically deduplicated, and each
query is interpolated in a 4-pixel conforming cell. Sharing identical corner
DOFs gives compatible bicubic traces across cell edges. No corner is silently
assigned to the first overlapping model.

This implementation exposes true 2D bicubic semantics, but arbitrary
`[nx,ny]`, `ny>1` four-patch junction topology is not claimed or enabled.

## 8. Training Integration

All four methods participate in the field seen by optimization:

- Seed fitting calls `FBPINN_model()` and supervises its assembled `u,v`.
- Photometric Adam calls `FBPINN_loss -> FBPINN_forward -> FBPINN_model`.
- L-BFGS calls the same `FBPINN_loss`; the bbf/1 baseline has zero L-BFGS
  steps, so it was correctly not enabled for these experiments.
- Prediction calls the same model, and strain is autodiff of the final assembled
  displacement rather than a blend of local strains.

The seed path is regression-tested with nonzero assembled loss and nonzero,
finite local-parameter gradients for all methods. No new strain or interface
loss term was introduced.

## 9. Patch Tests

P1 is affine displacement and P2 is a curved quadratic displacement, both on a
synthetic `[4,1]` decomposition. Every local model represents the same analytic
field. Errors below are maximum displacement / maximum strain errors:

| Method | P1 displacement | P1 strain | P2 displacement | P2 strain |
|---|---:|---:|---:|---:|
| Cosine POU | 2.22e-16 | 3.02e-16 | 3.33e-16 | 2.60e-16 |
| Quintic POU | 2.22e-16 | 1.18e-16 | 2.22e-16 | 1.01e-16 |
| Hermite 1D C1 | 2.22e-16 | 5.20e-17 | 2.22e-16 | 4.51e-17 |
| Bicubic Hermite C1 | 3.33e-16 | 9.02e-17 | 2.22e-16 | 1.01e-16 |

RMS and interface-band errors are recorded in
`case/bbf/1/results/assembly_comparison/patch_test_metrics.json`; all are at
floating-point precision. Primitive tests additionally verify quintic endpoint
derivatives, exact cubic Hermite reproduction, endpoint C1 conditions, bicubic
polynomial/gradient reproduction, and grad/JVP/VJP propagation.

## 10. bbf/1 [4,1] Experiment

`case/bbf/1` contains a 664x741 reference/deformed image pair and one mask. The
ROI has 62,332 pixels and bounding box `(x=53..611, y=315..426)`. Four x
subdomains have centers `(122.75, 262.25, 401.75, 541.25)`, width 223.2 pixels,
and adjacent overlap width 83.7 pixels. Internal geometric interfaces are
`x=(192.5, 332.0, 471.5)`.

The four files in `config/assembly/` differ only in `assembly_method`. All use
AdaptiveFCN `[2,32,32,32,32,2]`, ZNSSD kernel 7, seed 0, 500 Adam steps,
learning rate 0.001, and the same seed calculation. Historical bbf/1 settings
have `seed_train_epochs=0` and `lbfgs_epochs=0`. Each method starts from the
same deterministic local-network initialization and writes to a method-suffixed
run directory.

Reproduction commands, using the project environment, are:

```powershell
& 'C:\Users\LBD\.conda\envs\jax_env\python.exe' scripts/run_assembly_experiments.py
& 'C:\Users\LBD\.conda\envs\jax_env\python.exe' -m unittest discover -s tests -v
```

## 11. Results

There is no displacement or strain ground truth in bbf/1. The following are
roughness/consistency diagnostics, not accuracy errors.

| Method | Final DIC loss | Global disp. roughness | Global strain roughness | Interface strain roughness | Max strain jump | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cosine POU | 43.9031 | 0.025718 | 5.785e-4 | 1.089e-3 | 2.164e-3 | 177.20 |
| Quintic POU | 52.1661 | 0.025423 | 2.912e-4 | 1.887e-4 | 1.058e-3 | 162.11 |
| Hermite 1D C1 | 56.8397 | 0.025226 | 2.600e-4 | 7.162e-5 | 2.768e-4 | 312.07 |
| Bicubic Hermite C1 | 56.8394 | 0.025212 | 2.597e-4 | 7.160e-5 | 2.767e-4 | 67.98 |

Relative to cosine, interface strain roughness falls about 82.7% with quintic
and 93.4% with either Hermite method. However, final photometric loss rises
18.8% for quintic and 29.5% for Hermite. Therefore the continuity improvement
does not establish better physical accuracy and comes with a convergence
tradeoff under this fixed budget.

## 12. Interface Diagnostics

The interface band is distance `<=5 px` from the three internal interfaces,
near-interface is `(5,15] px`, and interior is `>15 px`. For half-pixel
interfaces, traces use the adjacent integer columns at $x_\Gamma\pm0.5$; for
the integer interface at 332 they use columns 331 and 333. The reported
$J_u$, $J_{\nabla u}$, and $J_\varepsilon$ are RMS left/right trace
differences over valid ROI rows. These finite-offset indicators include true
field variation over one or two pixels and therefore are diagnostics rather
than exact mathematical jumps.

| Method | RMS displacement trace | RMS gradient trace | RMS strain trace | Interface/interior artifact ratio |
|---|---:|---:|---:|---:|
| Cosine POU | 0.036190 | 3.695e-3 | 6.720e-4 | 3.201 |
| Quintic POU | 0.027264 | 1.994e-3 | 3.264e-4 | 0.528 |
| Hermite 1D C1 | 0.026240 | 9.857e-4 | 6.609e-5 | 0.213 |
| Bicubic Hermite C1 | 0.026165 | 9.857e-4 | 6.610e-5 | 0.213 |

The common-color-limit strain comparison shows that the strongest cosine
stripe aligns with the subdomain geometry; the quintic transition suppresses
much of it, and the Hermite fields are smoother still. Exact values and all
three interface traces are in `assembly_metrics.json`.

## 13. Runtime

| Method | Compile (s) | Train + output (s) | Prediction (s) |
|---|---:|---:|---:|
| Cosine POU | 1.07 | 177.20 | 0.63 |
| Quintic POU | 0.73 | 162.11 | 0.66 |
| Hermite 1D C1 | 1.37 | 312.07 | 1.54 |
| Bicubic Hermite C1 | 2.67 | 67.98 | 1.67 |

The first naive bicubic prototype redundantly evaluated four corners per pixel
and built full Hessians; compilation estimated 206.9 billion FLOPs per step and
memory exceeded 35 GB, so that incomplete run was terminated. Deduplicating
shared corners, using a 4-pixel mesh, and directional JVPs reduced the measured
step graph to 2.51 billion FLOPs. The unexpectedly low optimized wall time is a
property of this mesh/deduplication implementation and case; it must not be
generalized to arbitrary 2D layouts. Peak memory was not instrumented for the
final four runs.

## 14. Limitations

- Qualification is for a rectangular x-chain `[nx,1]`, exercised at `[4,1]`.
  General `ny>1` four-patch junction ownership and global C1 compatibility are
  deliberately rejected by configuration validation.
- bbf/1 has no ground truth. The experiment can support claims about loss,
  roughness, finite-offset continuity, and runtime, but not displacement or
  strain accuracy.
- The fixed 500-step budget favors no post-comparison retuning; Hermite's higher
  final loss may change with another optimization schedule, but that was not
  tested because assembly was the intended independent variable.
- Real-case seed pretraining and L-BFGS were zero-step in the historical case.
  Their shared assembly wiring is tested, but nonzero full bbf/1 phases were not
  added because doing so would violate the fairness requirement.
- The bicubic cell size is currently fixed by config and has not been subjected
  to a resolution sensitivity study.

## 15. Conclusion

All four assemblies are differentiable, enter the training forward, pass the
mathematical and regression suites, and complete the same real `[4,1]` case.
The compact quintic POU reduces the cosine interface artifact while retaining a
standard normalized POU. True derivative-aware Hermite construction produces
the strongest interface continuity indicators, at the cost of a higher fixed-
budget photometric loss; 1D Hermite also incurs substantial runtime. Bicubic
Hermite matches the x-chain continuity behavior while preserving a genuine 16
DOF 2D formulation. The evidence supports qualification for `[nx,1]`, not a
claim that arbitrary 2D global C1 topology or strain accuracy has been solved.
