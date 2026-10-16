# 1. Motivation

This phase fixes the formal comparison budget at Adam 500, with
`seed_train_epochs=0` and `lbfgs_epochs=0`, for `case/bbf/1` on the fixed
`[4,1]` decomposition.  The goal is not a new window family: it is to test
whether a low-frequency global field plus polynomial-compatible local
residuals can reduce derivative incompatibility without adding a strain,
gradient, interface, residual, or smoothing loss.

# 2. Local Overfitting Evidence

The original raw cosine trajectory is retained only as
`OVERFITTING_DIAGNOSTIC_ONLY`.  Its loss falls from 43.90 (500) to 27.86
(3000), while raw-local strain mismatch rises from `3.65e-3` to `5.26e-2`
and interface roughness from `1.09e-3` to `1.47e-2`.  The diagnostic
indicator `O_local=E_local_strain(t)/E_local_strain(500)` reaches 14.41 at
3000.  This is evidence of derivative incompatibility after displacement
optimization; it is not a ranking of methods at different budgets.

# 3. Failed Simultaneous Backbone

The old simultaneous global-plus-cosine-residual reference at 500 steps has
loss 43.61, global displacement RMS 0.641, residual RMS 2.298, and
residual/total displacement RMS ratio 0.943.  Thus a global branch exists,
but the residual branch still carries nearly all of the representation.  The
older long-run diagnosis (ratio about 0.967) remains consistent with this
identifiability failure.

# 4. Backbone-first Formulation

The v2 assemblies implement

`u(x,y) = u_G(x,y) + gate(t) A(delta_u_1,...,delta_u_4)`.

The trainer owns only the scalar gate.  The assembly owns cosine or P2
mathematics, zero output initialization, global coordinate scaling, and the
global/residual decomposition.  The global AdaptiveFCN is shallower and
narrower (`[12,12,12]`) than each local field (`[32,32,32,32]`) and receives
normalized coordinates multiplied by 0.5.

# 5. Residual Scale Control

The single gate is structural, not a loss penalty:

| Stage | Steps | Gate | Active representation |
|---|---:|---|---|
| 1 | 0--150 | 0 | global only; local output has no forward/gradient contribution |
| 2 | 150--300 | 0 -> 0.25 | global plus weak residual |
| 3 | 300--500 | 0.25 -> 1 | progressive full residual release |

The local final affine output map is initialized to zero.  Checkpoints at
0/150/300/500 store global, residual and total `u,v,exx,exy,eyy` as MAT/NPZ
under `case/bbf/1/results/backbone_p2_phase/analysis/`.

# 6. Global/Local Decomposition

At 500, progressive cosine has global/residual displacement RMS
4.509/3.580 (ratio 1.028) and progressive P2 has 2.825/2.750 (ratio 1.147).
The P2 residual strain RMS is 0.00856 versus global 0.00904; cosine residual
strain RMS is 0.10465 versus global 0.00791.  The schedule enforces a clean
global-only Stage 1, but neither final run establishes a small-residual
separation.  Direct release is simpler and also remains residual-dominated
(ratio 0.932).

# 7. P2 Reproducing Residual Assembly

`BackboneFirstP2ResidualAssembly` evaluates the four local residual fields at
the x-chain centers and solves a JAX differentiable, weighted 3-by-3 least
squares problem in basis `[1, x, x^2]` at each query point.  It is a complete
chain operation: all local fields contribute to each reconstruction query.
The global backbone remains separate; P2 is applied only to local residuals.

# 8. Synthetic Qualification

P0, P1 and P2 value and first-derivative tests pass at numerical tolerance.
The independent prototype gives P1 strain RMS `6.19e-18` and P2 strain RMS
`3.66e-17` in their exact cases.  Perturbed constant, linear and high-frequency
tests are saved in `backbone_p2_phase/patch_tests/`; increasing perturbation
amplitude and spatial frequency increases reconstructed derivative error as
expected.  `jax.grad` reaches the local parameters, and `jax.jvp` is used in
the derivative qualification and field extraction.

# 9. bbf/1 500-step Results

| Method | Loss | Interface roughness | RMS strain jump | Raw-local strain mismatch |
|---|---:|---:|---:|---:|
| Cosine POU | 43.90 | 1.09e-3 | 6.72e-4 | 3.65e-3 |
| Quintic POU | 52.17 | 1.89e-4 | 3.26e-4 | 4.22e-3 |
| Hermite 1D C1 | 56.84 | 7.16e-5 | 6.61e-5 | 4.12e-3 |
| Backbone-first cosine | 177.50 | 3.10e-2 | 1.99e-3 | 1.29e-1 |
| Backbone-first P2 | 128.62 | 8.32e-5 | 1.34e-4 | 1.82e-2 |

The backbone variants each have 14,252 trainable parameters (13,840 local +
412 global); local-only references have 13,840.  The complete machine-readable
table is `analysis/formal_500_metrics.csv`.

# 10. Interface Diagnostics

The P2 run produces low assembled interface roughness (`8.32e-5`), close to
the Hermite compatibility reference (`7.16e-5`) and much lower than cosine.
Its raw-local strain mismatch remains `1.82e-2`, higher than classical POU
baselines.  Therefore the low assembled interface-band indicator must be read
as reconstruction compatibility, not proof that each local residual learned a
compatible derivative.  The three y-traces are in
`analysis/backbone_progressive_p2_interface_y_traces.png`.

# 11. Trade-off

At 500 steps, P2 is substantially more expensive (405 s) than progressive
cosine (213 s) and local-only cosine (160 s).  It improves the assembled
interface roughness relative to progressive cosine by roughly 372x, but has a
worse photometric fit than local-only cosine and has not made residuals small.
Hermite remains the compatibility upper reference: it obtains the smoothest
interface behavior at the cost of the highest loss among the main local-only
methods.

# 12. Limitations

`bbf/1` has no displacement or strain ground truth; no statement here is a
strain-accuracy claim.  P2 exact reproduction verifies the reconstruction
mathematics, not real-image accuracy.  The 500-step P2 run is stable and fully
differentiable, but its residual/total ratio exceeds one, so the intended
global-trend-plus-small-correction representation is not identified by this
schedule alone.

# 13. Scientific Conclusion

Adam 500 is an appropriate fixed formal budget for this case because longer
local photometric optimization demonstrably worsens local derivative
compatibility.  The backbone-first implementation and P2 training path are
operationally qualified: Stage 1 is truly global-only, gates evolve as
specified, no forbidden loss was added, and final strains are autodiff of the
final displacement.  Scientifically, the central scale-separation hypothesis
is only partially supported.  P2 strongly suppresses assembled interface
roughness, but the residual remains comparable to or larger than the global
component.  Future work should investigate compatibility-aware early-stopping
diagnostics and representation-level identifiability, without treating lower
roughness as ground-truth strain accuracy.
