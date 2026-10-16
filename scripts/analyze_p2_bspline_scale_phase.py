"""Analyze the formal 500-step full-P2 and coarse B-spline scale study."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from scripts.analyze_backbone_p2_phase import (
    _component_fields, _component_metrics,
)
from scripts.analyze_strain_interface_sweep import (
    ROOT, _assembled_batched, _field_metrics, _local_batched,
    _local_interface_metrics, _map_fields, _roi, geometry, rms,
)
from segpinndic.DIC_assemblies import (
    BSplineCosineResidualAssembly, BSplineP2ResidualAssembly,
    CosinePOUAssembly, Hermite1DC1Assembly, P2FullFieldAssembly,
    QuinticPOUAssembly, p2_chain_conditioning,
)


PHASE = ROOT / "case" / "bbf" / "1" / "results" / "p2_bspline_scale_phase"
OUT = PHASE / "analysis"
REFERENCE = PHASE / "reference"
OLD_PHASE = ROOT / "case" / "bbf" / "1" / "results" / "strain_interface_phase"
OLD_METRICS = OLD_PHASE / "analysis" / "strain_interface_metrics.json"

CLASSICAL = {
    "cosine_pou": CosinePOUAssembly,
    "quintic_pou": QuinticPOUAssembly,
    "hermite_1d_c1": Hermite1DC1Assembly,
}
NEW = {
    "p2_fullfield": (P2FullFieldAssembly, "p2_fullfield",
                     "AdaptiveFCN_All_4x1_DIC_ZNSSD_p2_fullfield", False),
    "bspline4_p2": (BSplineP2ResidualAssembly, "bspline4_p2",
                    "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_p2_residual", True),
    "bspline6_p2": (BSplineP2ResidualAssembly, "bspline6_p2",
                    "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_p2_residual", True),
    "bspline8_p2": (BSplineP2ResidualAssembly, "bspline8_p2",
                    "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_p2_residual", True),
    "bspline6_cosine": (BSplineCosineResidualAssembly, "bspline6_cosine",
                        "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_cosine_residual", True),
    "bspline6_p2_frozen": (BSplineP2ResidualAssembly, "bspline6_p2_frozen",
                           "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_p2_residual", True),
}
LABELS = {
    "cosine_pou": "Cosine", "quintic_pou": "Quintic",
    "hermite_1d_c1": "Hermite", "p2_fullfield": "P2 full-field",
    "bspline4_p2": "B-spline 4x4 + P2",
    "bspline6_p2": "B-spline 6x6 + P2",
    "bspline8_p2": "B-spline 8x8 + P2",
    "bspline6_cosine": "B-spline 6x6 + cosine",
    "bspline6_p2_frozen": "B-spline 6x6 + P2 frozen",
}


def _load_checkpoint(folder, run, step=500):
    path = PHASE / folder / "models" / run / f"model_{step:08d}.jax"
    with path.open("rb") as stream:
        saved_step, params, _state, _active, loss = pickle.load(stream)
    if int(saved_step) != step:
        raise ValueError(f"{path} contains step {saved_step}, expected {step}")
    return params, float(np.asarray(loss)), path


def _runtime(folder, run):
    path = PHASE / folder / "summaries" / run / "run_metrics_roi0_pair1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _conditioning(params, points):
    xmins, xmaxs = [np.asarray(value)[:, 0] for value in
                    params["static"]["decomposition"]["subdomain"]["params"][:2]]
    centres = 0.5*(xmins+xmaxs)
    regularization = float(np.asarray(
        params["static"]["assembly"]["p2_regularization"]))
    return p2_chain_conditioning(centres, points[:, 0], regularization)


def _spectral_separation_metrics(mask, assembled, components):
    """Summarize low/high x-frequency content on three valid ROI traces."""
    maps = {
        "global": _map_fields(mask, components["global"]),
        "residual": _map_fields(mask, components["residual"]),
        "total": _map_fields(mask, assembled),
    }
    candidate_rows = np.where(np.sum(mask, axis=1) > 100)[0]
    rows = candidate_rows[np.linspace(
        0, len(candidate_rows)-1, 3).astype(int)]
    raw = {key: {"low": [], "high": [], "all": []} for key in maps}
    for row in rows:
        valid = mask[row]
        for key, values in maps.items():
            signal = values["u"][row, valid] + 1j*values["v"][row, valid]
            signal = signal-signal.mean()
            frequency = np.fft.rfftfreq(len(signal))
            power = (np.abs(np.fft.rfft(np.real(signal)))**2
                     + np.abs(np.fft.rfft(np.imag(signal)))**2)
            raw[key]["low"].append(float(power[(frequency > 0)
                                                 & (frequency <= .05)].sum()))
            raw[key]["high"].append(float(power[frequency >= .1].sum()))
            raw[key]["all"].append(float(power[frequency > 0].sum()))
    result = {}
    for key, bands in raw.items():
        low, high, total = (float(np.mean(bands[name]))
                            for name in ("low", "high", "all"))
        result[key] = {
            "low_band_power_fraction": low/total,
            "high_band_power_fraction": high/total,
            "high_to_low_power_ratio": high/low,
        }
    residual_high = float(np.mean(raw["residual"]["high"]))
    global_high = float(np.mean(raw["global"]["high"]))
    result["residual_to_global_high_band_power_ratio"] = (
        residual_high/global_high)
    return result


def _evaluate_new(method, spec):
    assembly, folder, run, residual_method = spec
    params, checkpoint_loss, checkpoint = _load_checkpoint(folder, run)
    runtime = _runtime(folder, run)
    mask, points = _roi(params)
    assembled = _assembled_batched(params, points, assembly)
    mapped = _map_fields(mask, assembled)
    local = _local_interface_metrics(
        params, points, zero_offset=residual_method)
    item = {
        "adam_budget": 500,
        "checkpoint_reported_loss": checkpoint_loss,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "displacement_rms": rms(assembled["value"]),
        "local_displacement_mismatch_rms": rms([
            value["local_displacement_mismatch_rms"] for value in local]),
        "local_gradient_mismatch_rms": rms([
            value["local_gradient_mismatch_rms"] for value in local]),
        "local_strain_mismatch_rms": rms([
            value["local_strain_mismatch_rms"] for value in local]),
        "local_interfaces": local,
        "training": runtime,
        **_field_metrics(mapped, mask, geometry(mask)),
    }
    item["interface_interior_roughness_ratio"] = (
        item["interface_strain_roughness"]/item["interior_strain_roughness"])
    if assembly in (P2FullFieldAssembly, BSplineP2ResidualAssembly):
        item["p2_conditioning"] = _conditioning(params, points)
    components = None
    if residual_method:
        components = _component_fields(params, points, assembly)
        item["decomposition"] = _component_metrics(assembled, components)
        item["spectral_separation"] = _spectral_separation_metrics(
            mask, assembled, components)
        stage_params, stage_loss, stage_path = _load_checkpoint(folder, run, 200)
        stage_controls = np.asarray(
            stage_params["trainable"]["global_field"]["controls"])
        final_controls = np.asarray(
            params["trainable"]["global_field"]["controls"])
        item["stage1"] = {
            "loss": stage_loss,
            "checkpoint": str(stage_path.relative_to(ROOT)),
            "control_rms": rms(stage_controls),
            "final_control_rms": rms(final_controls),
            "control_drift_rms": rms(final_controls-stage_controls),
        }
    return item, mapped, components, params, assembled


def _evaluate_classical(old):
    results, fields = {}, {}
    for method, assembly in CLASSICAL.items():
        run = f"AdaptiveFCN_All_4x1_DIC_ZNSSD_{method}"
        path = OLD_PHASE / "fair_sweep" / "models" / run / "model_00000500.jax"
        with path.open("rb") as stream:
            _step, params, _state, _active, _loss = pickle.load(stream)
        mask, points = _roi(params)
        assembled = _assembled_batched(params, points, assembly)
        fields[method] = _map_fields(mask, assembled)
        item = old["methods"][method]["budgets"]["500"].copy()
        item["training"] = old["methods"][method]["training"]
        item["displacement_rms"] = rms(assembled["value"])
        item["local_gradient_mismatch_rms"] = rms([
            value["local_gradient_mismatch_rms"]
            for value in item["local_interfaces"]])
        item["interface_interior_roughness_ratio"] = (
            item["interface_strain_roughness"]/
            item["interior_strain_roughness"])
        results[method] = item
    return results, fields


def _save_fields(method, mapped):
    np.savez_compressed(OUT / f"{method}_fields.npz", **mapped)
    savemat(OUT / f"{method}_fields.mat", mapped)


def _shared_field_figure(methods, fields, mask, filename):
    quantities = ("u", "v", "exx", "exy", "eyy")
    limits = {}
    for name in quantities:
        values = np.concatenate([fields[method][name][mask] for method in methods])
        if name in ("u", "v"):
            limits[name] = tuple(np.percentile(values, (1, 99)))
        else:
            limit = np.percentile(np.abs(values), 99)
            limits[name] = (-limit, limit)
    fig, axes = plt.subplots(len(methods), len(quantities),
                             figsize=(18, 2.6*len(methods)), dpi=170,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, method in enumerate(methods):
        for col, name in enumerate(quantities):
            cmap = "viridis" if name in ("u", "v") else "coolwarm"
            image = axes[row, col].imshow(
                np.where(mask, fields[method][name], np.nan), cmap=cmap,
                vmin=limits[name][0], vmax=limits[name][1])
            axes[row, col].set_title(f"{LABELS[method]} — {name}", fontsize=8)
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=.03, pad=.01)
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def _decomposition_figure(method, mapped, components, mask, filename):
    component_maps = {
        "global": _map_fields(mask, components["global"]),
        "residual": _map_fields(mask, components["residual"]),
        "total": mapped,
    }
    quantities = ("u", "v", "exx", "exy", "eyy")
    limits = {}
    for name in quantities:
        values = np.concatenate([
            component_maps[key][name][mask] for key in component_maps])
        limit = np.percentile(np.abs(values), 99)
        limits[name] = (-limit, limit)
    fig, axes = plt.subplots(3, 5, figsize=(18, 9), dpi=170,
                             constrained_layout=True)
    for row, key in enumerate(("global", "residual", "total")):
        for col, name in enumerate(quantities):
            image = axes[row, col].imshow(
                np.where(mask, component_maps[key][name], np.nan),
                cmap="coolwarm", vmin=limits[name][0], vmax=limits[name][1])
            axes[row, col].set_title(f"{key} — {name}", fontsize=9)
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=.03, pad=.01)
    fig.suptitle(LABELS[method])
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def _interface_figure(mask, methods, params_by_method):
    quantities = ("u", "v", "exx", "exy", "eyy")
    interfaces = geometry(mask)["interfaces"]
    fig, axes = plt.subplots(len(methods)*len(interfaces), 5,
                             figsize=(18, 3*len(methods)*len(interfaces)),
                             dpi=150, constrained_layout=True)
    for method_index, method in enumerate(methods):
        assembly = NEW[method][0]
        params = params_by_method[method]
        residual_method = NEW[method][3]
        for interface_index, interface in enumerate(interfaces):
            row = method_index*len(interfaces)+interface_index
            x_index = int(round(interface))
            ys = np.where(mask[:, x_index])[0]
            points = np.stack([
                np.full(len(ys), interface, dtype=np.float32), ys.astype(np.float32)],
                axis=1)
            left_ids = np.full(len(ys), interface_index, dtype=np.int32)
            right_ids = np.full(len(ys), interface_index+1, dtype=np.int32)
            local_fns = (
                __import__("segpinndic.DIC_decompositions", fromlist=["RectangularDecompositionND"]).RectangularDecompositionND.norm_fn,
                __import__("segpinndic.DIC_networks", fromlist=["AdaptiveFCN"]).AdaptiveFCN.network_fn,
                __import__("segpinndic.DIC_decompositions", fromlist=["RectangularDecompositionND"]).RectangularDecompositionND.unnorm_fn,
                __import__("segpinndic.DIC_decompositions", fromlist=["RectangularDecompositionND"]).RectangularDecompositionND.window_fn,
            )
            left = _local_batched(params, points, left_ids,
                                  zero_offset=residual_method)
            right = _local_batched(params, points, right_ids,
                                   zero_offset=residual_method)
            total = _assembled_batched(params, points, assembly)
            traces = {"local left": left, "local right": right, "total": total}
            if residual_method:
                traces.update(_component_fields(params, points, assembly))
            for col, name in enumerate(quantities):
                ax = axes[row, col]
                for label, data in traces.items():
                    if name == "u": values = data["value"][:, 0]
                    elif name == "v": values = data["value"][:, 1]
                    else: values = data["strain"][:, ("exx", "exy", "eyy").index(name)]
                    ax.plot(ys, values, label=label, linewidth=.8)
                ax.set_title(f"{LABELS[method]} x={interface:g} {name}", fontsize=7)
                ax.grid(alpha=.2)
                if row == 0 and col == 0:
                    ax.legend(fontsize=6)
    fig.savefig(OUT / "04_interface_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def _tradeoff_figures(results):
    methods = list(results)
    for x_key, x_label, filename in (
            ("loss", "Final photometric loss", "06_loss_vs_interface_tradeoff.png"),
            ("runtime", "Training wall time [s]", "07_runtime_vs_interface_tradeoff.png")):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=170)
        for method in methods:
            item = results[method]
            x = (item.get("checkpoint_reported_loss", item.get("dic_loss_recomputed"))
                 if x_key == "loss" else item["training"]["total_seconds"])
            y = item["interface_strain_roughness"]
            ax.scatter(x, y, s=45)
            ax.annotate(LABELS[method], (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        ax.set_xlabel(x_label); ax.set_ylabel("Interface strain roughness")
        ax.set_yscale("log"); ax.grid(alpha=.25)
        fig.savefig(OUT / filename, bbox_inches="tight")
        plt.close(fig)


def _ratio_figure(results):
    methods = ("bspline4_p2", "bspline6_p2", "bspline8_p2")
    grids = np.array([4, 6, 8])
    disp = [results[m]["decomposition"]["residual_to_total_displacement_rms_ratio"]
            for m in methods]
    strain = [results[m]["decomposition"]["residual_to_total_strain_rms_ratio"]
              for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    ax.plot(grids, disp, "o-", label="R_disp")
    ax.plot(grids, strain, "s-", label="R_strain")
    ax.axhline(1., color="gray", linestyle="--", linewidth=.8)
    ax.set_xlabel("B-spline control grid"); ax.set_ylabel("Residual / total RMS")
    ax.set_xticks(grids); ax.grid(alpha=.25); ax.legend()
    fig.savefig(OUT / "05_residual_ratio_vs_grid.png", bbox_inches="tight")
    plt.close(fig)


def _spectrum_figure(mask, mapped, components):
    maps = {"global": _map_fields(mask, components["global"]),
            "residual": _map_fields(mask, components["residual"]),
            "total": mapped}
    candidate_rows = np.where(np.sum(mask, axis=1) > 100)[0]
    rows = candidate_rows[np.linspace(0, len(candidate_rows)-1, 3).astype(int)]
    spectra = {key: [] for key in maps}
    for row in rows:
        valid = mask[row]
        for key, values in maps.items():
            signal = values["u"][row, valid] + 1j*values["v"][row, valid]
            spectra[key].append(np.abs(np.fft.rfft(np.real(signal-signal.mean())))**2
                                + np.abs(np.fft.rfft(np.imag(signal-signal.mean())))**2)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    for key, values in spectra.items():
        minimum = min(len(value) for value in values)
        power = np.mean([value[:minimum] for value in values], axis=0)
        frequency = np.fft.rfftfreq(2*(minimum-1))[:minimum]
        ax.semilogy(frequency[1:], power[1:]+1e-20, label=key)
    ax.set_xlabel("x-direction spatial frequency [cycles/pixel]")
    ax.set_ylabel("Mean trace power"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(OUT / "09_frequency_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def _write_csv(results):
    rows = []
    for method, item in results.items():
        decomposition = item.get("decomposition", {})
        rows.append({
            "method": method,
            "loss": item.get("checkpoint_reported_loss", item.get("dic_loss_recomputed")),
            "interface_roughness": item["interface_strain_roughness"],
            "global_roughness": item["global_strain_roughness"],
            "rms_jump": item["rms_strain_jump"],
            "max_jump": item["max_strain_jump"],
            "interface_interior": item["interface_interior_roughness_ratio"],
            "local_gradient_mismatch": item["local_gradient_mismatch_rms"],
            "local_strain_mismatch": item["local_strain_mismatch_rms"],
            "R_disp": decomposition.get("residual_to_total_displacement_rms_ratio"),
            "R_strain": decomposition.get("residual_to_total_strain_rms_ratio"),
            "params": item["training"]["total_trainable_parameter_count"],
            "runtime": item["training"]["total_seconds"],
            "stage1_loss": item.get("stage1", {}).get("loss"),
            "control_drift_rms": item.get("stage1", {}).get(
                "control_drift_rms"),
        })
    with (OUT / "formal_500_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    old = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    results, fields = _evaluate_classical(old)
    (REFERENCE / "classical_500_reference.json").write_text(
        json.dumps({
            "source": str(OLD_METRICS.relative_to(ROOT)),
            "methods": results,
        }, indent=2, allow_nan=True), encoding="utf-8")
    components, params_by_method = {}, {}
    common_mask = None
    for method, spec in NEW.items():
        print(f"Evaluating {method}", flush=True)
        item, mapped, method_components, params, _assembled = _evaluate_new(method, spec)
        results[method] = item; fields[method] = mapped
        components[method] = method_components; params_by_method[method] = params
        common_mask = np.asarray(params["static"]["problem"]["mask"], dtype=bool)
        _save_fields(method, mapped)

    _shared_field_figure(
        ("cosine_pou", "quintic_pou", "hermite_1d_c1", "p2_fullfield"),
        fields, common_mask, "01_p2_fullfield_vs_baselines.png")
    _shared_field_figure(
        ("bspline4_p2", "bspline6_p2", "bspline8_p2"), fields,
        common_mask, "02_bspline_grid_fields.png")
    _decomposition_figure(
        "bspline6_p2", fields["bspline6_p2"], components["bspline6_p2"],
        common_mask, "03_bspline6_decomposition.png")
    _interface_figure(
        common_mask, ("p2_fullfield", "bspline8_p2"), params_by_method)
    _ratio_figure(results)
    _tradeoff_figures(results)
    _spectrum_figure(
        common_mask, fields["bspline8_p2"], components["bspline8_p2"])
    patch_figure = PHASE / "patch_tests" / "perturbed_patch_summary.png"
    if patch_figure.exists():
        shutil.copy2(patch_figure, OUT / "08_perturbed_p2_tests.png")
    _write_csv(results)
    payload = {
        "formal_budget": {"adam_steps": 500, "seed_train_steps": 0,
                          "lbfgs_steps": 0},
        "ground_truth_available": False,
        "strain_definition": "Autodiff of final assembled displacement.",
        "methods": results,
    }
    (OUT / "p2_bspline_scale_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({
        method: {
            "loss": item.get("checkpoint_reported_loss", item.get("dic_loss_recomputed")),
            "roughness": item["interface_strain_roughness"],
            "R_disp": item.get("decomposition", {}).get(
                "residual_to_total_displacement_rms_ratio"),
        } for method, item in results.items()}, indent=2))


if __name__ == "__main__":
    main()
