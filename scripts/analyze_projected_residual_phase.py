"""Analyze the formal 8x8 none/P1/P2 projected-residual experiments."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from scripts.analyze_backbone_p2_phase import _component_fields, _component_metrics
from scripts.analyze_strain_interface_sweep import (
    ROOT, _assembled_batched, _checkpoint as old_checkpoint, _field_metrics,
    _local_batched, _local_interface_metrics, _map_fields, _roi,
    geometry, rms,
)
from scripts.diagnose_projected_residual_modal import (
    basis, basis_gradient, normalized, strain_from_gradient,
)
from segpinndic.DIC_assemblies import (
    BSplineCosineResidualAssembly,
    BSplineProjectedCosineResidualAssembly,
    CosinePOUAssembly, Hermite1DC1Assembly,
)
from segpinndic.DIC_field_diagnostics import adjacent_overlap_points
from segpinndic.DIC_projections import build_projection_plan


PHASE = ROOT/"case"/"bbf"/"1"/"results"/"projected_residual_phase"
OUT = PHASE/"analysis"
OLD_METRICS = (ROOT/"case"/"bbf"/"1"/"results"/"strain_interface_phase"
               /"analysis"/"strain_interface_metrics.json")
RUN_NONE = "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_cosine_residual"
RUN_PROJECTED = (
    "AdaptiveFCN_All_4x1_DIC_ZNSSD_bspline_projected_cosine_residual")
SPECS = {
    "bspline8_cosine_none": (
        BSplineCosineResidualAssembly,
        PHASE/"reference"/"bspline8_cosine_none", RUN_NONE, None),
    "bspline8_cosine_p1": (
        BSplineProjectedCosineResidualAssembly,
        PHASE/"p1_projection", RUN_PROJECTED, 1),
    "bspline8_cosine_p2": (
        BSplineProjectedCosineResidualAssembly,
        PHASE/"p2_projection", RUN_PROJECTED, 2),
}
LABELS = {
    "cosine_pou": "Cosine full local",
    "hermite_1d_c1": "Hermite",
    "bspline8_cosine_none": "B-spline 8x8 + cosine",
    "bspline8_cosine_p1": "B-spline 8x8 + P1-perp cosine",
    "bspline8_cosine_p2": "B-spline 8x8 + P2-perp cosine",
}


def checkpoint(folder, run, step=500):
    path = folder/"models"/run/f"model_{step:08d}.jax"
    with path.open("rb") as stream:
        saved_step, params, _state, _active, loss = pickle.load(stream)
    if int(saved_step) != step:
        raise ValueError(f"{path} contains {saved_step}, expected {step}")
    return params, float(np.asarray(loss)), path


def runtime(folder, run):
    path = folder/"summaries"/run/"run_metrics_roi0_pair1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_projection(params, order):
    axis = np.linspace(-1, 1, 16, dtype=np.float32)
    norm_points = np.stack(
        np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    plan = build_projection_plan(norm_points, order, regularization=1e-8)
    xmins, xmaxs = [np.asarray(value) for value in
                    params["static"]["decomposition"]["subdomain"]["params"][:2]]
    coefficients, diagnostics = [], []
    for model in range(len(xmins)):
        center, scale = .5*(xmins[model]+xmaxs[model]), .5*(xmaxs[model]-xmins[model])
        physical = center+scale*norm_points
        raw = _local_batched(
            params, physical, np.full(len(physical), model, np.int32),
            zero_offset=True)["value"]
        coefficient = np.asarray(plan["coefficient_operator"])@raw
        projected = raw-np.asarray(plan["sample_basis"])@coefficient
        moment = np.asarray(plan["sample_basis"]).T@(
            np.asarray(plan["weights"])[:, None]*projected)
        coefficients.append(coefficient)
        diagnostics.append({
            "model": model,
            "raw_sample_rms": rms(raw),
            "projected_sample_rms": rms(projected),
            "projection_reduction": 1-rms(projected)/rms(raw),
            "max_abs_moment": float(np.max(np.abs(moment))),
            "rms_moment": rms(moment),
        })
    return np.asarray(coefficients), {
        "requested_order": order,
        "effective_order": plan["effective_order"],
        "rank": plan["rank"],
        "condition_number": plan["condition_number"],
        "fallback": plan["fallback"],
        "regularization": plan["regularization"],
        "point_count_per_subdomain": len(norm_points),
        "models": diagnostics,
    }


def projected_local_fields(params, points, model_ids, order, coefficients):
    raw = _local_batched(params, points, model_ids, zero_offset=True)
    xmins, xmaxs = [np.asarray(value) for value in
                    params["static"]["decomposition"]["subdomain"]["params"][:2]]
    low_value = np.zeros_like(raw["value"])
    low_gradient = np.zeros_like(raw["gradient"])
    for model in np.unique(model_ids):
        selected = model_ids == model
        norm_points, scale = normalized(
            points[selected], xmins[model], xmaxs[model])
        low_value[selected] = basis(norm_points, order)@coefficients[model]
        low_gradient[selected] = np.einsum(
            "nbx,bd->ndx", basis_gradient(norm_points, order, scale),
            coefficients[model])
    projected_gradient = raw["gradient"]-low_gradient
    return {
        "value": raw["value"]-low_value,
        "gradient": projected_gradient,
        "strain": strain_from_gradient(projected_gradient),
    }


def projected_interface_metrics(params, roi_points, order, coefficients):
    records = []
    for overlap in adjacent_overlap_points(params, roi_points):
        points = overlap["points"].astype(np.float32)
        left = projected_local_fields(
            params, points, np.full(len(points), overlap["left_model"], np.int32),
            order, coefficients)
        right = projected_local_fields(
            params, points, np.full(len(points), overlap["right_model"], np.int32),
            order, coefficients)
        records.append({
            "interface_x": overlap["x_interface"],
            "point_count": len(points),
            "local_displacement_mismatch_rms": rms(left["value"]-right["value"]),
            "local_gradient_mismatch_rms": rms(left["gradient"]-right["gradient"]),
            "local_strain_mismatch_rms": rms(left["strain"]-right["strain"]),
        })
    return records


def spectral_metrics(mask, fields):
    mapped = _map_fields(mask, fields)
    candidate = np.where(np.sum(mask, axis=1) > 100)[0]
    rows = candidate[np.linspace(0, len(candidate)-1, 3).astype(int)]
    low, high, total = [], [], []
    for row in rows:
        valid = mask[row]
        signal = mapped["u"][row, valid]+1j*mapped["v"][row, valid]
        signal = signal-signal.mean()
        frequency = np.fft.rfftfreq(len(signal))
        power = (np.abs(np.fft.rfft(signal.real))**2
                 + np.abs(np.fft.rfft(signal.imag))**2)
        low.append(power[(frequency > 0)&(frequency <= .05)].sum())
        high.append(power[frequency >= .1].sum())
        total.append(power[frequency > 0].sum())
    low, high, total = map(lambda values: float(np.mean(values)),
                           (low, high, total))
    return {
        "low_frequency_power_fraction": low/total,
        "high_frequency_power_fraction": high/total,
        "high_to_low_power_ratio": high/low,
    }


def evaluate_method(method, spec):
    assembly, folder, run, order = spec
    params, loss, path = checkpoint(folder, run)
    mask, points = _roi(params)
    total = _assembled_batched(params, points, assembly)
    mapped = _map_fields(mask, total)
    components = _component_fields(params, points, assembly)
    raw_components = _component_fields(
        params, points, BSplineCosineResidualAssembly)
    raw_local = _local_interface_metrics(params, points, zero_offset=True)
    raw_local_strain = rms([
        item["local_strain_mismatch_rms"] for item in raw_local])
    if order is None:
        projected_local, projection = raw_local, None
    else:
        coefficients, projection = fixed_projection(params, order)
        projected_local = projected_interface_metrics(
            params, points, order, coefficients)
    projected_local_strain = rms([
        item["local_strain_mismatch_rms"] for item in projected_local])
    stage_params, stage_loss, stage_path = checkpoint(folder, run, 200)
    del stage_params
    item = {
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_reported_loss": loss,
        "stage1_loss": stage_loss,
        "stage2_loss_reduction": stage_loss-loss,
        "stage1_checkpoint": str(stage_path.relative_to(ROOT)),
        "training": runtime(folder, run),
        "raw_local_interfaces": raw_local,
        "projected_local_interfaces": projected_local,
        "raw_local_strain_mismatch_rms": raw_local_strain,
        "projected_local_strain_mismatch_rms": projected_local_strain,
        "compatibility_gain_projection": (
            1-projected_local_strain/raw_local_strain),
        "decomposition": _component_metrics(total, components),
        "raw_residual_displacement_rms": rms(raw_components["residual"]["value"]),
        "raw_residual_strain_rms": rms(raw_components["residual"]["strain"]),
        "projected_residual_displacement_rms": rms(components["residual"]["value"]),
        "projected_residual_strain_rms": rms(components["residual"]["strain"]),
        "projection_reduction_disp": (
            1-rms(components["residual"]["value"])
            / rms(raw_components["residual"]["value"])),
        "projection_reduction_strain": (
            1-rms(components["residual"]["strain"])
            / rms(raw_components["residual"]["strain"])),
        "spectral": {
            "global": spectral_metrics(mask, components["global"]),
            "raw_residual": spectral_metrics(mask, raw_components["residual"]),
            "projected_residual": spectral_metrics(mask, components["residual"]),
            "total": spectral_metrics(mask, total),
        },
        "projection": projection,
        **_field_metrics(mapped, mask, geometry(mask)),
    }
    item["formal_runtime_seconds"] = item["training"]["total_seconds"]
    return item, mapped, components, raw_components, mask


def classical_references():
    old = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    results, fields, mask = {}, {}, None
    for method, assembly in (("cosine_pou", CosinePOUAssembly),
                             ("hermite_1d_c1", Hermite1DC1Assembly)):
        params, loss, path = old_checkpoint(method, 500)
        mask, points = _roi(params)
        total = _assembled_batched(params, points, assembly)
        fields[method] = _map_fields(mask, total)
        item = old["methods"][method]["budgets"]["500"].copy()
        item["checkpoint_reported_loss"] = loss
        item["checkpoint"] = str(path.relative_to(ROOT))
        item["training"] = old["methods"][method]["training"]
        item["formal_runtime_seconds"] = item["adam_elapsed_seconds"]
        results[method] = item
    return results, fields, mask


def shared_figure(methods, fields, mask):
    names = ("u", "v", "exx", "exy", "eyy")
    limits = {}
    for name in names:
        values = np.concatenate([fields[m][name][mask] for m in methods])
        if name in ("u", "v"):
            limits[name] = tuple(np.percentile(values, (1, 99)))
        else:
            limit = np.percentile(np.abs(values), 99)
            limits[name] = (-limit, limit)
    fig, axes = plt.subplots(len(methods), 5, figsize=(18, 2.7*len(methods)),
                             dpi=160, constrained_layout=True)
    for row, method in enumerate(methods):
        for col, name in enumerate(names):
            image = axes[row, col].imshow(
                np.where(mask, fields[method][name], np.nan),
                cmap="viridis" if name in ("u", "v") else "coolwarm",
                vmin=limits[name][0], vmax=limits[name][1])
            axes[row, col].set_title(f"{LABELS[method]} — {name}", fontsize=8)
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=.03, pad=.01)
    fig.savefig(OUT/"06_projection_interface_comparison.png", bbox_inches="tight")
    plt.close(fig)


def decomposition_figure(method, total, components, raw_components, mask, filename):
    maps = {
        "global": _map_fields(mask, components["global"]),
        "raw residual": _map_fields(mask, raw_components["residual"]),
        "projected residual": _map_fields(mask, components["residual"]),
        "total": total,
    }
    names = ("u", "v", "exx", "exy", "eyy")
    fig, axes = plt.subplots(4, 5, figsize=(18, 11), dpi=160,
                             constrained_layout=True)
    for col, name in enumerate(names):
        values = np.concatenate([value[name][mask] for value in maps.values()])
        limit = np.percentile(np.abs(values), 99)
        for row, (label, value) in enumerate(maps.items()):
            image = axes[row, col].imshow(
                np.where(mask, value[name], np.nan), cmap="coolwarm",
                vmin=-limit, vmax=limit)
            axes[row, col].set_title(f"{label} — {name}", fontsize=8)
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=.03, pad=.01)
    fig.suptitle(LABELS[method])
    fig.savefig(OUT/filename, bbox_inches="tight")
    plt.close(fig)


def summary_figures(results):
    methods = list(results)
    annotation_offsets = {
        "cosine_pou": (4, 7),
        "hermite_1d_c1": (4, -2),
        "bspline8_cosine_none": (4, -10),
        "bspline8_cosine_p1": (4, 4),
        "bspline8_cosine_p2": (4, 4),
    }
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for method in methods:
        item = results[method]
        ax.scatter(item["checkpoint_reported_loss"],
                   item["interface_strain_roughness"], s=45)
        ax.annotate(LABELS[method],
                    (item["checkpoint_reported_loss"],
                     item["interface_strain_roughness"]), fontsize=7,
                    xytext=annotation_offsets[method], textcoords="offset points")
    ax.set_xlabel("Final photometric loss")
    ax.set_ylabel("Interface strain roughness")
    ax.set_yscale("log"); ax.grid(alpha=.25)
    fig.savefig(OUT/"07_projection_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    projected = [m for m in methods if "decomposition" in results[m]]
    x = np.arange(len(projected)); width = .36
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar(x-width/2, [results[m]["decomposition"]
           ["residual_to_total_displacement_rms_ratio"] for m in projected],
           width, label="R_disp")
    ax.bar(x+width/2, [results[m]["decomposition"]
           ["residual_to_total_strain_rms_ratio"] for m in projected],
           width, label="R_strain")
    ax.set_xticks(x, [LABELS[m] for m in projected], rotation=15, ha="right")
    ax.axhline(.912, color="black", linestyle="--", linewidth=.8,
               label="prior 8x8 P2 R_disp")
    ax.grid(axis="y", alpha=.2); ax.legend()
    fig.savefig(OUT/"08_residual_ratio_comparison.png", bbox_inches="tight")
    plt.close(fig)


def spectrum_figure(results):
    methods = ("bspline8_cosine_none", "bspline8_cosine_p1",
               "bspline8_cosine_p2")
    labels = ("raw residual", "projected residual")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160,
                             constrained_layout=True)
    for ax, key in zip(axes, ("low_frequency_power_fraction",
                              "high_frequency_power_fraction")):
        x = np.arange(len(methods)); width=.36
        ax.bar(x-width/2, [results[m]["spectral"]["raw_residual"][key]
                           for m in methods], width, label=labels[0])
        ax.bar(x+width/2, [results[m]["spectral"]["projected_residual"][key]
                           for m in methods], width, label=labels[1])
        ax.set_xticks(x, ["none", "P1", "P2"])
        ax.set_title(key); ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.savefig(OUT/"09_projection_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def write_csv(results):
    rows=[]
    for method,item in results.items():
        dec=item.get("decomposition",{})
        rows.append({
            "method":method,"loss":item["checkpoint_reported_loss"],
            "interface_roughness":item["interface_strain_roughness"],
            "rms_jump":item["rms_strain_jump"],
            "raw_local_strain_mismatch":item.get("raw_local_strain_mismatch_rms"),
            "projected_local_strain_mismatch":item.get("projected_local_strain_mismatch_rms"),
            "R_disp":dec.get("residual_to_total_displacement_rms_ratio"),
            "R_strain":dec.get("residual_to_total_strain_rms_ratio"),
            "residual_low_frequency_fraction":item.get("spectral",{}).get(
                "projected_residual",{}).get("low_frequency_power_fraction"),
            "params":item["training"]["total_trainable_parameter_count"],
            "runtime":item["formal_runtime_seconds"],
        })
    with (OUT/"formal_500_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results, fields, mask = classical_references()
    components, raw_components = {}, {}
    for method,spec in SPECS.items():
        print(f"Evaluating {method}",flush=True)
        item,mapped,comp,raw,mask=evaluate_method(method,spec)
        results[method]=item;fields[method]=mapped
        components[method]=comp;raw_components[method]=raw
        np.savez_compressed(OUT/f"{method}_fields.npz",**mapped)
        savemat(OUT/f"{method}_fields.mat",mapped)
    shared_figure(tuple(results),fields,mask)
    decomposition_figure("bspline8_cosine_p1",fields["bspline8_cosine_p1"],
                         components["bspline8_cosine_p1"],
                         raw_components["bspline8_cosine_p1"],mask,
                         "04_p1_projected_decomposition.png")
    decomposition_figure("bspline8_cosine_p2",fields["bspline8_cosine_p2"],
                         components["bspline8_cosine_p2"],
                         raw_components["bspline8_cosine_p2"],mask,
                         "05_p2_projected_decomposition.png")
    summary_figures(results);spectrum_figure(results);write_csv(results)
    payload={
        "formal_budget":{"adam_steps":500,"seed_train_steps":0,"lbfgs_steps":0},
        "ground_truth_available":False,
        "strain_definition":"Autodiff of final assembled displacement.",
        "methods":results,
    }
    (OUT/"projected_residual_metrics.json").write_text(
        json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps({m:{
        "loss":v["checkpoint_reported_loss"],
        "roughness":v["interface_strain_roughness"],
        "R_disp":v.get("decomposition",{}).get(
            "residual_to_total_displacement_rms_ratio"),
        "R_strain":v.get("decomposition",{}).get(
            "residual_to_total_strain_rms_ratio"),
    } for m,v in results.items()},indent=2))


if __name__=="__main__":
    main()
