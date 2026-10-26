"""Read-only checkpoint counterfactuals; never updates model/optimizer parameters.

Run with the project Python: python -m scripts.analyze_overlap_sensitivity.
The loss-map adapter uses the installed loss function's AST, changing only its
return value, so neighborhood normalization remains exactly the production one.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import textwrap
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_strain_interface_sweep import (
    _checkpoint, _roi, _assembled_batched, _local_batched, _model_fns,
    _map_fields, _field_metrics,
)
from segpinndic.DIC_assemblies import CosinePOUAssembly
from segpinndic.DIC_field_diagnostics import (
    adjacent_overlap_points, _take_model_params, strain_from_gradient,
    decompose_normalized_pou_gradient, _local_field,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_problem import DIC_ZNSSD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "case/bbf/1/results/overlap_sensitivity_phase/central"
AMPLITUDES = (-.1, -.05, -.025, -.01, -.005, .005, .01, .025, .05, .1)
CONTRACTIONS = (-.5, -.25, -.1, -.05, -.025, -.01, -.005,
                .005, .01, .025, .05, .1, .25, .5, 1.)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def loss_map_adapter():
    tree = ast.parse(textwrap.dedent(inspect.getsource(DIC_ZNSSD.loss_fn)))
    fn = tree.body[0]
    fn.decorator_list = []
    assert isinstance(fn.body[-1], ast.Return)
    assert isinstance(fn.body[-1].value, ast.Name) and fn.body[-1].value.id == "znssd"
    fn.body[-1].value = ast.Name(id="sampled_znssd", ctx=ast.Load())
    scope = dict(DIC_ZNSSD.loss_fn.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<offline_loss_map>", "exec"), scope)
    return scope[fn.name]


def envelope(point, left, right):
    t = (point[0]-left)/(right-left)
    z = jnp.clip(jnp.minimum(t, 1-t)*4, 0., 1.)
    return z**3*(10-15*z+6*z*z)


def prepare(step):
    params, reported, path = _checkpoint("cosine_pou", step)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, params)
    mask, points = _roi(params)
    overlap = adjacent_overlap_points(params, points)[1]
    selected = (points[:, 0] >= overlap["x_left"]) & (points[:, 0] <= overlap["x_right"])
    op = points[selected]
    fields = {k: np.asarray(v) for k, v in _assembled_batched(params, points, CosinePOUAssembly).items()}
    left = _local_batched(params, op, np.ones(len(op), dtype=np.int32))
    right = _local_batched(params, op, np.full(len(op), 2, dtype=np.int32))
    lp, _ = _take_model_params(params, jnp.asarray(1))
    rp, _ = _take_model_params(params, jnp.asarray(2))
    fns = _model_fns(CosinePOUAssembly)

    def alpha(p):
        wl = CosinePOUAssembly.weight(lp, p, fns[:4])[0]
        wr = CosinePOUAssembly.weight(rp, p, fns[:4])[0]
        return wl/(wl+wr)

    alpha_v, alpha_g = jax.jit(jax.vmap(lambda p: (alpha(p), jax.grad(alpha)(p))))(jnp.asarray(op))
    hfn = lambda p: envelope(p, overlap["x_left"], overlap["x_right"])
    h, hg = jax.jit(jax.vmap(lambda p: (hfn(p), jax.grad(hfn)(p))))(jnp.asarray(op))
    delta = np.asarray(left["value"])-np.asarray(right["value"])
    deltag = np.asarray(left["gradient"])-np.asarray(right["gradient"])
    av, ag = np.asarray(alpha_v), np.asarray(alpha_g)
    direct_u = av[:, None]*left["value"]+(1-av[:, None])*right["value"]
    direct_g = (av[:, None, None]*left["gradient"]+(1-av[:, None, None])*right["gradient"]
                +delta[:, :, None]*ag[:, None, :])
    reconstruction_u = float(np.max(np.abs(direct_u-fields["value"][selected])))
    reconstruction_g = float(np.max(np.abs(direct_g-fields["gradient"][selected])))
    assert reconstruction_u < 2e-5 and reconstruction_g < 2e-6
    # Original decomposition includes any gradient cancellation in the base field.
    decomposed = decompose_normalized_pou_gradient(
        params, op, RectangularDecompositionND, CosinePOUAssembly, fns)
    lm = jax.jit(lambda uv: loss_map_adapter()(params, jnp.asarray(points), uv))
    baseline_loss_map = np.asarray(lm(jnp.asarray(fields["value"])))
    original_loss = float(jax.jit(lambda uv: DIC_ZNSSD.loss_fn(params, jnp.asarray(points), uv))(
        jnp.asarray(fields["value"])))
    assert abs(float(baseline_loss_map.mean())-original_loss) < 2e-4
    expected = json.loads((ROOT / "case/bbf/1/results/strain_interface_phase/analysis/strain_interface_metrics.json").read_text())["methods"]["cosine_pou"]["budgets"][str(step)]
    metrics = _field_metrics(_map_fields(mask, fields), mask, __import__(
        "scripts.analyze_assembly_results", fromlist=["geometry"]).geometry(mask))
    assert abs(original_loss-expected["dic_loss_recomputed"]) < 2e-3
    for key in ("interface_strain_roughness", "rms_strain_jump"):
        assert np.isclose(metrics[key], expected[key], rtol=2e-4, atol=1e-8), (key, metrics[key], expected[key])
    # Check spatial differentiation of the compact envelope (also at joins).
    check_points = jnp.array([[overlap["x_left"]+t*(overlap["x_right"]-overlap["x_left"]), 365.]
                             for t in (0., .125, .25, .5, .75, .875, 1.)])
    derivatives = jax.vmap(lambda p: (hfn(p), jax.grad(hfn)(p), jax.hessian(hfn)(p)))(check_points)
    assert np.max(np.abs(np.asarray(derivatives[0])[[0, -1]])) < 1e-6
    assert np.max(np.abs(np.asarray(derivatives[1])[[0, 2, 3, 4, 6]])) < 1e-6
    cache = dict(points=points, mask=mask, selected=selected, op=op,
                 value=fields["value"], gradient=fields["gradient"], strain=fields["strain"],
                 delta=delta, deltag=deltag, alpha=av, alphag=ag, h=np.asarray(h), hg=np.asarray(hg),
                 loss_map=baseline_loss_map, local_gradient=np.asarray(decomposed["local_gradient"]),
                 window_gradient=np.asarray(decomposed["window_gradient"]))
    np.savez_compressed(OUT / f"baseline_{step}.npz", **cache)
    record = dict(step=step, checkpoint=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  checkpoint_loss=reported, recomputed_loss=original_loss, expected_loss=expected["dic_loss_recomputed"],
                  loss_map_mean=float(baseline_loss_map.mean()), reconstruction_displacement_max=reconstruction_u,
                  reconstruction_gradient_max=reconstruction_g, point_count=len(points), overlap_point_count=len(op),
                  overlap=[overlap["x_left"], overlap["x_right"]], baseline_metrics=metrics,
                  jax_version=jax.__version__, device=str(jax.devices()), dtype=str(fields["value"].dtype))
    dump(OUT / f"baseline_{step}.json", record)
    print(f"BASELINE {step}: loss={original_loss:.8f}, roughness={metrics['interface_strain_roughness']:.8g}, "
          f"jump={metrics['rms_strain_jump']:.8g}, reconstruction={reconstruction_g:.3g}", flush=True)
    return params, cache


def region_metrics(c, value, gradient, lossmap, dg_local, dg_window):
    points, mask, selected = c["points"], c["mask"], c["selected"]
    x = points[:, 0]
    left, right = float(c["op"][:, 0].min()), float(c["op"][:, 0].max())
    affected = (x >= np.floor(left-3)) & (x <= np.ceil(right+3))
    strains = np.asarray(strain_from_gradient(jnp.asarray(gradient)))
    dl = np.asarray(lossmap, dtype=np.float64)-c["loss_map"]
    de = strains-c["strain"]
    du = value-c["value"]
    result = dict(loss=float(np.mean(lossmap)), loss_change=float(np.mean(dl)),
                  loss_relative=float(np.mean(dl)/np.mean(c["loss_map"])),
                  displacement_delta_rms=rms(du[selected]), displacement_delta_max=float(np.max(np.abs(du))),
                  strain_delta_rms=rms(de[selected]), strain_delta_relative=rms(de[selected])/rms(c["strain"][selected]),
                  strain_delta_p95=float(np.percentile(np.abs(de[selected]), 95)),
                  strain_delta_max=float(np.max(np.abs(de[selected]))))
    for name, region in {"affected": affected, "overlap": selected, "center": selected & (c["h_full"] == 1),
                         "taper": selected & (c["h_full"] < 1), "outside": ~affected}.items():
        base = float(np.mean(c["loss_map"][region]))
        result[f"{name}_loss_change"] = float(np.mean(dl[region]))
        result[f"{name}_loss_relative"] = float(np.mean(dl[region])/base)
        result[f"{name}_loss_absolute_relative"] = float(np.mean(np.abs(dl[region]))/base)
        result[f"{name}_loss_positive"] = float(np.mean(np.maximum(dl[region], 0)))
        result[f"{name}_loss_negative"] = float(np.mean(np.minimum(dl[region], 0)))
        result[f"{name}_strain_rms"] = rms(strains[region])
        result[f"{name}_strain_delta_rms"] = rms(de[region])
    smap = np.zeros(mask.shape+(3,), dtype=np.float64)
    smap[mask] = strains
    # Same adjacency-based roughness definition as the historical report.
    vx = mask[:, 1:] & mask[:, :-1]
    vy = mask[1:] & mask[:-1]
    xm = np.arange(mask.shape[1]-1)+.5
    ym_x = np.arange(mask.shape[1])
    dx2 = np.mean(np.diff(smap, axis=1)**2, axis=-1)
    dy2 = np.mean(np.diff(smap, axis=0)**2, axis=-1)
    for name, low, high in (("overlap", left, right), ("center", 311.075, 352.925),
                            ("interface", 327., 337.)):
        xx = vx & (xm[None, :] >= low) & (xm[None, :] <= high)
        yy = vy & (ym_x[None, :] >= low) & (ym_x[None, :] <= high)
        result[f"{name}_roughness_x"] = float(np.sqrt(np.mean(dx2[xx])))
        result[f"{name}_roughness_xy"] = float(np.sqrt(np.mean(np.r_[dx2[xx], dy2[yy]])))
    taper_x = vx & (xm[None, :] >= left) & (xm[None, :] <= right) & ((xm[None, :] < 311.075) | (xm[None, :] > 352.925))
    result["taper_roughness_x"] = float(np.sqrt(np.mean(dx2[taper_x])))
    # Reuse exact historical trace validity rules.
    fields = {"u": np.zeros(mask.shape), "v": np.zeros(mask.shape)}
    fields["u"][mask], fields["v"][mask] = value[:, 0], value[:, 1]
    fields.update({name: smap[..., i] for i, name in enumerate(("exx", "exy", "eyy"))})
    from scripts.analyze_assembly_results import interface_traces
    trace = interface_traces(fields, mask, np.array([332.]))[0]
    result["rms_jump"], result["max_jump"] = trace["J_strain_rms"], trace["max_abs_strain_jump"]
    local = np.asarray(strain_from_gradient(c["local_gradient"]+dg_local))
    window = np.asarray(strain_from_gradient(c["window_gradient"]+dg_window))
    result.update(local_term_strain_rms=rms(local), window_term_strain_rms=rms(window),
                  local_window_cross_mean=float(np.mean(local*window)),
                  decomposition_max=float(np.max(np.abs(local+window-strains[selected]))))
    for i, name in enumerate(("exx", "exy", "eyy")):
        result[f"{name}_delta_rms"] = rms(de[selected, i])
        result[f"{name}_rms"] = rms(strains[selected, i])
    return result, strains


def scan(step, params, c):
    selected, op = c["selected"], c["op"]
    n = len(op)
    c["h_full"] = np.zeros(len(c["points"]))
    c["h_full"][selected] = c["h"]
    h, hg = c["h"][:, None], c["hg"][:, None, :]
    av, ag = c["alpha"][:, None], c["alphag"][:, None, :]
    center = (op.max(axis=0)+op.min(axis=0))/2
    scale = (op.max(axis=0)-op.min(axis=0))/2
    z = (op-center)/scale
    design = np.c_[np.ones(n), z]
    coef = np.linalg.lstsq(design, c["delta"], rcond=None)[0]
    affine = design@coef
    affine_g = np.broadcast_to((coef[1:]/scale[:, None]).T, (n, 2, 2)).copy()
    # Independent direct autodiff check on actual neural fields. The scan uses
    # cached first derivatives and the exact product rule for efficiency.
    lp, _ = _take_model_params(params, jnp.asarray(1))
    rp, _ = _take_model_params(params, jnp.asarray(2))
    fns = _model_fns(CosinePOUAssembly)
    edges = json.loads((OUT / f"baseline_{step}.json").read_text())["overlap"]
    probe = np.linspace(0, n-1, 17).astype(int)

    def direct_counterfactual(p):
        l = _local_field(lp, p, fns[:4]); r = _local_field(rp, p, fns[:4])
        wl = CosinePOUAssembly.weight(lp, p, fns[:4])[0]
        wr = CosinePOUAssembly.weight(rp, p, fns[:4])[0]
        a = wl/(wl+wr)
        hh = envelope(p, *edges)
        return a*l+(1-a)*r - .1*(a-.5)*hh*(l-r)

    pvalue, pgradient = jax.jit(jax.vmap(lambda p: (direct_counterfactual(p),
                                 jax.jacfwd(direct_counterfactual)(p))))(jnp.asarray(op[probe]))
    dq = -.1*h*c["delta"]
    dqg = -.1*(h[:, :, None]*c["deltag"]+c["delta"][:, :, None]*hg)
    cached_value = c["value"][selected]+(av-.5)*dq
    cached_gradient = c["gradient"][selected]+(av-.5)[:, :, None]*dqg+dq[:, :, None]*ag
    checks = dict(probe_count=len(probe), displacement_max=float(np.max(np.abs(pvalue-cached_value[probe]))),
                  gradient_max=float(np.max(np.abs(pgradient-cached_gradient[probe]))))
    assert checks["displacement_max"] < 3e-5 and checks["gradient_max"] < 3e-6, checks
    dump(OUT / f"direct_autodiff_check_{step}.json", checks)
    modes = []
    for component in range(2):
        for axis in (-1, 0, 1):
            b = np.zeros((n, 2)); bg = np.zeros((n, 2, 2))
            b[:, component] = 1 if axis == -1 else z[:, axis]
            if axis != -1:
                bg[:, component, axis] = 1/scale[axis]
            norm = rms(h*b)
            b, bg = b/norm, bg/norm
            name = f"{'uv'[component]}_{('constant','xi','eta')[axis+1]}"
            modes.append((name, b, bg))
    loss_adapter = loss_map_adapter()
    lm = jax.jit(lambda uv: loss_adapter(params, jnp.asarray(c["points"]), uv))
    rows = []
    maps = {"baseline": (c["value"], c["strain"], c["loss_map"])}
    traces = {}

    def evaluate(family, mode, amplitude, b, bg, common=False):
        q = amplitude*h*b
        qg = amplitude*(h[:, :, None]*bg+b[:, :, None]*hg)
        if common:
            du, dgl, dgw = q, qg, np.zeros_like(qg)
            dlocal, dlocalg = c["delta"], c["deltag"]
        else:
            du = (av-.5)*q
            dgl = (av-.5)[:, :, None]*qg
            dgw = q[:, :, None]*ag
            dlocal, dlocalg = c["delta"]+q, c["deltag"]+qg
        value = c["value"].copy(); value[selected] += du
        gradient = c["gradient"].copy(); gradient[selected] += dgl+dgw
        lossmap = np.asarray(lm(jnp.asarray(value)))
        metrics, strains = region_metrics(c, value, gradient, lossmap, dgl, dgw)
        row = dict(step=step, family=family, mode=mode, amplitude=float(amplitude),
                   raw_mismatch_rms=rms(dlocal), raw_gradient_mismatch_rms=rms(dlocalg),
                   raw_strain_mismatch_rms=rms(strain_from_gradient(dlocalg)), **metrics)
        assert np.isfinite(list(metrics.values())).all()
        assert metrics["decomposition_max"] < 3e-6
        assert np.max(np.abs(value[~selected]-c["value"][~selected])) == 0
        rows.append(row)
        key = f"{family}_{mode}_{amplitude:g}"
        if family == "baseline" or (family == "contraction" and amplitude in (.1, .5, 1.)):
            maps[key] = (value, strains, lossmap)
        # Full-y column RMS and means retain overlap-wide shape, not three selected traces.
        traces[key] = np.array([[xx, *np.mean(strains[(c['points'][:,0] == xx)], axis=0),
                                rms(strains[(c['points'][:,0] == xx)]),
                                np.mean(lossmap[c['points'][:,0] == xx])]
                               for xx in np.unique(op[:, 0])])
        return row

    evaluate("baseline", "none", 0., np.zeros((n, 2)), np.zeros((n, 2, 2)))
    for name, b, bg in modes:
        unit_relative_rms = rms((av-.5)*h*b)
        unit_common_rms = rms(h*b)
        for a in AMPLITUDES:
            evaluate("relative", name, a, b, bg)
            evaluate("common_matched", name, a*unit_relative_rms/unit_common_rms, b, bg, True)
        print(f"SCAN {step} {name}: {len(rows)} rows", flush=True)
    for name, b, bg in (("actual", c["delta"], c["deltag"]), ("affine", affine, affine_g)):
        for s in CONTRACTIONS:
            evaluate("contraction", name, s, -b, -bg)
        print(f"SCAN {step} contraction {name}: {len(rows)} rows", flush=True)
    with (OUT / f"scan_{step}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for key, (value, strain, lossmap) in maps.items():
        np.savez_compressed(OUT / f"{step}_{key}.npz", points=c["points"], value=value, strain=strain, loss_map=lossmap)
    np.savez_compressed(OUT / f"traces_{step}.npz", **traces)
    dump(OUT / f"affine_fit_{step}.json", dict(center=center.tolist(), scale=scale.tolist(), coefficients=coef.tolist(),
         full_mismatch_rms=rms(c["delta"]), affine_mismatch_rms=rms(affine), remainder_rms=rms(c["delta"]-affine)))
    make_figures(step, c, rows, maps, traces)
    return rows


def make_figures(step, c, rows, maps, traces):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for mode in sorted({r["mode"] for r in rows if r["family"] == "relative"}):
        rs = [r for r in rows if r["family"] == "relative" and r["mode"] == mode]
        axes[0].plot([r["amplitude"] for r in rs], [100*r["affected_loss_relative"] for r in rs], "o-", label=mode)
        axes[1].plot([100*r["affected_loss_absolute_relative"] for r in rs], [r["strain_delta_relative"] for r in rs], "o-", label=mode)
    for mode in ("actual", "affine"):
        rs = [rows[0]]+[r for r in rows if r["family"] == "contraction" and r["mode"] == mode]
        rs.sort(key=lambda r: r["amplitude"])
        axes[2].plot([r["amplitude"] for r in rs], [r["interface_roughness_x"] for r in rs], "o-", label=mode)
    axes[0].set(xlabel="Relative-output amplitude (px RMS)", ylabel="Affected-region loss change (%)")
    axes[1].set(xlabel="Mean absolute pointwise loss change (%)", ylabel="Strain change / baseline RMS")
    axes[2].set(xlabel="Mismatch contraction s", ylabel="Central interface roughness")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.savefig(OUT / f"response_{step}.png", dpi=160); plt.close(fig)
    selected_keys = ["baseline", "contraction_actual_0.1", "contraction_actual_0.5", "contraction_actual_1"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for key in selected_keys:
        t = traces["baseline_none_0" if key == "baseline" else key]
        for i, ax in enumerate(axes):
            ax.plot(t[:, 0], t[:, i+1], label=key)
    for i, ax in enumerate(axes):
        ax.set(ylabel=("Mean exx", "Mean exy", "Mean eyy")[i], xlabel="x (px)"); ax.axvline(332, color="gray", ls="--"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.savefig(OUT / f"profiles_{step}.png", dpi=160); plt.close(fig)
    # Fixed scales across all counterfactuals, no per-panel autoscaling.
    strain_limits = np.percentile(np.abs(c["strain"][c["selected"]]), 99.5, axis=0)
    loss_limit = max(np.percentile(np.abs(maps[k][2]-c["loss_map"]), 99.5) for k in selected_keys)
    fig, axes = plt.subplots(4, 4, figsize=(16, 9), constrained_layout=True)
    ys, xs = c["points"][:, 1].astype(int), c["points"][:, 0].astype(int)
    for col, key in enumerate(selected_keys):
        _, strain, lossmap = maps[key]
        for row in range(4):
            values = strain[:, row] if row < 3 else lossmap-c["loss_map"]
            m = np.full(c["mask"].shape, np.nan); m[ys, xs] = values
            limit = strain_limits[row] if row < 3 else max(loss_limit, 1e-8)
            im = axes[row, col].imshow(m, cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, col].set_xlim(285, 380); axes[row, col].set_ylim(430, 310)
            axes[row, col].set_title(f"{key}: {('exx','exy','eyy','loss change')[row]}", fontsize=8)
            axes[row, col].axvline(332, color="gray", ls="--", lw=.6)
            if col == 3 and row < 3:
                fig.colorbar(im, ax=list(axes[row, :]), shrink=.7, label="strain")
        fig.colorbar(im, ax=axes[3, col], shrink=.7)
    fig.savefig(OUT / f"maps_{step}.png", dpi=160); plt.close(fig)


def summarize():
    report = {"scope": "cosine 500/3000, central overlap only", "steps": {}}
    for step in (500, 3000):
        rows = [{k: v if k in ("family", "mode") else float(v) for k, v in row.items()}
                for row in csv.DictReader((OUT / f"scan_{step}.csv").open(encoding="utf-8"))]
        base = rows[0]
        thresholds = []
        for tolerance in (.005, .01, .02):
            accepted = [r for r in rows if r["family"] != "baseline"
                        and abs(r["loss_relative"]) <= tolerance
                        and abs(r["affected_loss_relative"]) <= tolerance
                        and r["affected_loss_absolute_relative"] <= 2*tolerance]
            best = max(accepted, key=lambda r: r["strain_delta_relative"])
            thresholds.append(dict(tolerance=tolerance, accepted=len(accepted),
                strain_change_20pct_count=sum(r["strain_delta_relative"] >= .2 for r in accepted),
                maximum_accepted_strain_change=best["strain_delta_relative"], best=best))
        directional = []
        for mode in sorted({r["mode"] for r in rows if r["family"] == "relative"}):
            item = {"mode": mode}
            for family in ("relative", "common_matched"):
                rs = sorted([r for r in rows if r["family"] == family and r["mode"] == mode], key=lambda r: r["amplitude"])
                positive, negative = rs[6], rs[3]  # +/-0.01 relative amplitude; paired matched controls.
                amplitude = positive["displacement_delta_rms"]
                assert np.isclose(amplitude, negative["displacement_delta_rms"], rtol=2e-4)
                item[family] = dict(assembled_displacement_rms=amplitude,
                    slope=(positive["loss"]-negative["loss"])/(2*amplitude),
                    curvature=(positive["loss"]+negative["loss"]-2*base["loss"])/amplitude**2)
            item["relative_common_curvature_ratio"] = item["relative"]["curvature"]/item["common_matched"]["curvature"]
            directional.append(item)
        # The energy identity uses component-wise strain RMS throughout.
        el, ew = base["local_term_strain_rms"]**2, base["window_term_strain_rms"]**2
        cross = base["local_window_cross_mean"]
        report["steps"][str(step)] = dict(row_count=len(rows), baseline=base, thresholds=thresholds,
             directional=directional, local_window_normalized_inner_product=cross/np.sqrt(el*ew),
             cancellation_fraction=-2*cross/(el+ew),
             energy_identity_error=abs(el+ew+2*cross-base["overlap_strain_rms"]**2),
             max_decomposition_error=max(r["decomposition_max"] for r in rows),
             max_outside_halo_loss_change=max(r["outside_loss_absolute_relative"] for r in rows),
             contraction_rows=[r for r in rows if r["family"] == "contraction"])
    dump(OUT / "summary.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--reuse-baseline", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    dump(OUT / "protocol.json", dict(checkpoints=[500, 3000], interface=332, amplitudes=AMPLITUDES,
        contractions=CONTRACTIONS, envelope="C2 quintic rise/fall, middle half plateau", seed="checkpoint-owned",
        contraction_refinement="Added +/-0.005,0.01,0.025,0.05 after initial scan: s=0.1 already changed strain by 16/36%; fill small-amplitude response gap, same criteria",
        field_rms="sqrt(mean of squared scalar components)", loss_map="original loss AST with only return changed",
        no_parameter_updates=True, no_training=True, perturbation_scope="physical local output, not guaranteed parameter-realizable"))
    for step in (500, 3000):
        if args.reuse_baseline:
            params, _, _ = _checkpoint("cosine_pou", step)
            params = jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, params)
            c = dict(np.load(OUT / f"baseline_{step}.npz"))
        else:
            params, c = prepare(step)
        if not args.baseline_only:
            scan(step, params, c)
        jax.clear_caches()
    if not args.baseline_only:
        summarize()
    print(f"DONE in {time.time()-start:.1f}s; {OUT}", flush=True)


if __name__ == "__main__":
    main()
