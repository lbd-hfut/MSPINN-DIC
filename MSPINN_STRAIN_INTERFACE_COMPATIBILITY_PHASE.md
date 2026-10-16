# MSPINN strain-interface compatibility phase

## 1. Background

This phase asks why the `[4,1]` MSPINN-DIC solution for `case/bbf/1` can have
small displacement jumps yet retain strain bands near the three internal
interfaces.  The study deliberately separates three questions: whether the
independently trained local neural fields already disagree in their
derivatives, whether normalized partition-of-unity (POU) assembly suppresses
or amplifies that disagreement, and whether changing the representation is
more effective than changing only the window.  No strain, gradient, interface,
or smoothing loss is introduced.  Every reported trained strain is obtained
by automatic differentiation of the final assembled displacement.

The real ROI is `x=53..611`, `y=315..426`.  Four x-directed subdomains have
centres `122.75, 262.25, 401.75, 541.25`, support width `223.2`, adjacent
overlap width `83.7`, and nominal internal interfaces
`x=192.5, 332.0, 471.5`.  Unless stated otherwise, the seed is 42.

## 2. Mathematical Decomposition

For local physical displacements \(u_i(x)\), windows \(w_i(x)\),
\(A=\sum_iw_i u_i\), and \(W=\sum_iw_i\), normalized POU gives

\[
u_{POU}=\frac{A}{W},\qquad
\nabla u_{POU}=
\frac{\sum_i w_i\nabla u_i}{W}
+\frac{\sum_i u_i\otimes\nabla w_i}{W}
-\frac{A\otimes\nabla W}{W^2}.
\]

The first term transports local-network gradients; the second and third are
the normalized-window contribution.  The diagnostic evaluates every term at
identical physical points throughout each adjacent overlap.  Its maximum
numerical reconstruction error is `2.03e-7` in float32, so the classification
does not rely on a finite-offset trace approximation.

## 3. Local Network Diagnostics

At the cosine 500-step checkpoint, raw adjacent-local mismatches are:

| Interface | RMS displacement | RMS gradient | RMS strain | POU window-gradient strain RMS | Window/local ratio |
|---|---:|---:|---:|---:|---:|
| 192.5 | 0.227016 | 0.004405 | 0.003688 | 0.001584 | 0.429 |
| 332.0 | 1.966876 | 0.036421 | 0.003320 | 0.023782 | 7.164 |
| 471.5 | 0.212277 | 0.004538 | 0.003919 | 0.001291 | 0.330 |

The pooled cosine local-strain mismatch is `0.003651`; the pooled POU
window-gradient contribution is `0.013781`.  Thus the local networks are not
strain-compatible even before assembly.  The much larger middle-interface
displacement and gradient mismatch also makes that overlap qualitatively
different from the outer two.

## 4. Artifact Classification

The source is classified as **BOTH**, with spatially heterogeneous dominance.
At the outer interfaces, window/local ratios below 0.5 mean raw local
derivative inconsistency is the larger source.  At the middle interface, the
ratio `7.164` means normalized window differentiation strongly amplifies a
large local displacement mismatch.  A single global `LOCAL` or `ASSEMBLY`
label would therefore hide the mechanism that matters most.

Quintic lowers the pooled window-gradient contribution from `0.013781` to
`0.012210`, but its pooled local-strain mismatch increases from `0.003651` to
`0.004216`.  Its improvement is window-shape redistribution, not evidence that
the local PINNs became derivative-compatible.  Hermite makes the endpoint
value and x-slope compatible by construction, but the trained raw local-strain
mismatches remain `0.003450`, `0.004295`, and `0.004531`; it redistributes the
transition over a broad strip and can replace a sharp jump with a smooth hump.

## 5. C1 Minimal Correction

The minimal method retains the trained cosine POU field \(u_P\).  At interface
\(x_k\), for every sampled y, define compatible trace targets

\[
\bar u_k=\tfrac12(u_k+u_{k+1}),\qquad
\bar u_{x,k}=\tfrac12(u_{x,k}+u_{x,k+1}),
\]

and correction degrees of freedom
\(a_k=\bar u_k-u_P(x_k,y)\),
\(d_k=\bar u_{x,k}-\partial_xu_P(x_k,y)\).  The correction is the unique
piecewise cubic Hermite function that has value/slope `(a_k,d_k)` at the
interface and zero value/slope at both compact-support edges.  The default
support spans half of the adjacent overlap.  Therefore only two vector trace
DOFs are imposed per `(interface,y)`; the overlap field is corrected rather
than replaced.  When adjacent local fields already match, both correction DOFs
vanish exactly.

The real training route is fully differentiable: local parameters affect both
the cosine base and the value/x-derivative trace targets through JAX JVPs.  The
support, zero-edge C1 conditions, exact zero-mismatch identity, interface
value/slope targets, and optimizer gradients are covered by tests.

## 6. Global Backbone + Local Residual

The alternative representation is

\[
u(x,y)=u_G(x,y)+\mathcal A(\delta u_1,\ldots,\delta u_4).
\]

`u_G` is one global `AdaptiveFCN [2,16,16,16,16,2]` normalized over the full
domain; it has 964 trainable parameters.  The four original local networks
remain unchanged in topology and initialization (13,840 parameters) but are
interpreted as zero-offset residuals and assembled with cosine POU.  Total
count is 14,804.  The local initialization SHA-256 is identical across all
five methods, while the global parameters use a separate deterministic key.
The generic assembly `init_extra_params` hook keeps this representation behind
the same `init_params/build_plan/assemble` seam as existing methods.

## 7. Polynomial Reproduction

The P1/P2 study is a deterministic weighted least-squares assembly-space
prototype, not a registered training method.  Coordinates are centred and
scaled before solving the minimum-norm polynomial reconstruction.  P1 exactly
reproduces a linear vector field (`Disp_RMS=8.89e-17`,
`Strain_RMS=6.19e-18`); P2 exactly reproduces a quadratic vector field
(`2.37e-16`, `3.66e-17`).

Under sinusoidal local perturbations at wavelength 11, P1 strain error scales
linearly with amplitude: `5.33e-6`, `2.66e-5`, `5.33e-5` for amplitudes
`0.001`, `0.005`, `0.01`.  P2 gives `1.75e-5`, `8.74e-5`, `1.75e-4` for the
same samples.  This particular sparse sampling geometry does not make P2
uniformly better for unresolved oscillations; reproduction order and
high-frequency robustness are different properties.

## 8. Perturbed Patch Tests

Synthetic patches isolate assembly effects while preserving the real
four-subdomain geometry.  Constant offsets have zero local strain mismatch but
nonzero assembled strain, proving that displacement disagreement alone can
enter strain through window gradients.  At offset amplitude `0.005`, strain
RMS is `8.01e-4` (cosine), `6.01e-4` (quintic), `5.51e-4` (Hermite), and
`6.82e-4` (minimal correction).  All values scale linearly with amplitude.

Linear mismatch creates both local-strain and assembly terms.  For slope
amplitude `5e-4`, local-strain mismatch is `4.56e-4`, while total strain RMS is
`1.70e-3`, `1.29e-3`, `1.18e-3`, and `1.45e-3` in the same method order.

High-frequency perturbations expose frequency dependence.  At amplitude
`0.01`, wavelength 10, local-strain mismatch is `0.003867`; strain RMS is
`0.002210` (cosine), `0.002019` (quintic), `0.001833` (Hermite), and
`0.002071` (minimal correction).  At wavelength 5 the local mismatch rises to
`0.007586`, and all methods retain substantial strain content.  The global
prototype matches cosine in these patch tests because the perturbation is
placed entirely in its residual component; a global architecture helps only
if training actually moves low-frequency content into the backbone.

## 9. Real bbf/1 Results

The formal comparison stops at **3000 Adam steps** (user-directed early
cutoff after the trajectories had reached a clear plateau).  All five methods
use the same ROI, four local subnetworks, initialization seed, optimizer, and
checkpoint budgets.  Cosine and quintic already had longer runs available, but
only their 3000-step checkpoints are used here.  Hermite's normal completion
record was interrupted after its 4000-step checkpoint; its 3000-step loss and
wall-time history are reconstructed directly from the intact TensorBoard event
stream and are marked accordingly in the JSON output.

| Method | ZNSSD | interface roughness | RMS strain jump | max strain jump | interface/interior | local strain mismatch | time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosine POU | 27.8601 | 1.474e-2 | 1.123e-2 | 3.957e-2 | 41.49 | 5.262e-2 | 950.2 |
| Quintic POU | 31.5283 | 1.136e-2 | 1.346e-2 | 4.871e-2 | 9.24 | 9.750e-2 | 963.2 |
| Hermite 1D C1 | 47.6063 | 2.023e-4 | 1.516e-4 | 1.467e-3 | 0.081 | 9.577e-3 | 1862.3* |
| C1 minimal correction | 31.3435 | 8.706e-3 | 1.248e-2 | 6.226e-2 | 9.25 | 7.056e-2 | 949.5 |
| Global + local residual | 26.7021 | 2.641e-2 | 1.100e-2 | 5.873e-2 | 71.61 | 6.753e-2 | 1008.8 |

`*` Hermite time comes from `loss/train` TensorBoard event wall times because
the interruption prevented the normal terminal summary write.

The Local-to-Assembled Amplification Ratio is
\(A_\epsilon=J_{\epsilon,\mathrm{RMS}}/M_{\epsilon,\mathrm{local,RMS}}\),
where the numerator is the final autodiff strain RMS-jump indicator and the
denominator is pooled adjacent-local raw strain mismatch at identical physical
points.  Values above one would indicate amplification by this indicator;
values below one indicate suppression.  At 3000 steps the values are `0.213`,
`0.138`, `0.0158`, `0.177`, and `0.163` in table order.  This does not absolve
an assembly with a large *absolute* artifact: cosine and global residual still
have large middle-interface bands because their raw local mismatch is larger.

The minimal correction is compact: it is nonzero on 22.0% of ROI samples,
with correction displacement RMS `0.02081`, vector maximum `0.45941`, and
strain RMS `0.002167`.  It improves roughness relative to cosine, but its
middle-interface maximum jump is larger, so it is not a universal replacement.

For global + residual, global displacement RMS is `0.5483` (`u=0.4350`,
`v=0.6420`) while residual RMS is `2.3604` (`u=1.4643`, `v=2.9998`), with
residual vector maximum `5.9998`.  The residual/total displacement RMS ratio
is `0.967`, and residual/global strain RMS ratio is `8.21`.  Thus this training
did not move most low-frequency displacement or strain into the backbone; it
achieves the best photometric fit by relying overwhelmingly on local residuals.

## 10. Convergence

The results below are recomputed full-ROI ZNSSD / interface roughness, not the
periodic training-batch log:

| steps | Cosine | Quintic | Hermite | C1 minimal | Global + residual |
|---|---|---|---|---|---|
| 500 | 43.8988 / 1.089e-3 | 52.1666 / 1.887e-4 | 56.8289 / 7.162e-5 | 52.1487 / 2.291e-3 | 43.8437 / 1.088e-3 |
| 1000 | 42.3429 / 1.179e-3 | 48.1578 / 3.095e-4 | 53.8558 / 7.970e-5 | 50.7739 / 2.380e-3 | 42.0962 / 1.216e-3 |
| 2000 | 30.9404 / 8.399e-3 | 33.1406 / 5.597e-3 | 48.5592 / 1.503e-4 | 40.0988 / 2.640e-3 | 33.0321 / 5.486e-3 |
| 3000 | 27.8601 / 1.474e-2 | 31.5283 / 1.136e-2 | 47.6063 / 2.023e-4 | 31.3435 / 8.706e-3 | 26.7021 / 2.641e-2 |

Each cell is `ZNSSD / interface roughness`.  Better photometric fit does not
track better interface compatibility: in particular, cosine and global
residual improve loss while roughness grows sharply after 1000 steps.

## 11. Trade-off

The non-dominated 3000-step trade-off is not a single winner.  Global residual
has the best photometric result (`26.7021`) but the worst interface roughness;
Hermite has the strongest interface compatibility but the worst photometric
loss and roughly double runtime.  C1 minimal correction is slightly better
than quintic in both loss and roughness at the final budget, but has the largest
maximum jump.  Cosine is a strong fit/roughness reference point and is less
rough than global residual for a modest loss cost.  The 500/1000/2000/3000
connected trajectories and distinct markers are plotted in the photometric and
runtime trade-off figures.

## 12. Limitations

- No ground-truth strain is available for `bbf/1`; metrics describe
  compatibility, roughness, displacement photometric fit, and decomposition,
  not strain accuracy.
- The Hermite and minimal-correction implementations target an x-directed
  `[nx,1]` chain.  They do not establish arbitrary 2-D C1 compatibility at
  corners, junctions, or irregular topology.
- Roughness is measured with finite differences only after autodiff strain has
  been evaluated.  It is a diagnostic, never a training objective.
- The P1/P2 reconstruction is a qualified prototype and has not been coupled
  to neural-field training, conditioning control, or arbitrary support
  selection.
- One seed, one ROI, and one image pair are insufficient for claims of general
  statistical superiority.

## 13. Scientific Conclusion

The evidence rejects the simple hypothesis that a higher-order POU alone will
remove strain-interface artifacts.  The raw local neural fields already have
derivative incompatibility, concentrated at the central overlap; normalized
windows can additionally turn displacement mismatch into strain.  Hermite
enforces compatibility at its transition endpoints and strongly lowers local
mismatch and absolute interface artifacts, but the smooth transition remains a
representation bias rather than evidence of ground-truth strain accuracy.

C1 minimal correction is a useful, compact, fully trainable control: it proves
that compatibility correction can be localized without replacing an entire
overlap.  On this case it is not superior to full Hermite on all interface
metrics, especially the maximum jump.  The global-backbone architecture is
also fully trainable, but at 3000 steps its residual dominates both displacement
and strain; it is therefore an architectural prototype rather than evidence
that the intended global/local separation has been learned.

The next experiment should constrain or parameterize the residual scale and
test whether a backbone-first schedule can genuinely absorb low-frequency
motion before local residual optimization.  That addresses the diagnosed cause
without introducing forbidden strain or interface losses.
