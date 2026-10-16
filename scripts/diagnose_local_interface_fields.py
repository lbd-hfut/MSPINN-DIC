"""Phase A: diagnose local-network versus assembly strain artifacts on bbf/1."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from segpinndic import DIC_assemblies, DIC_networks
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_field_diagnostics import (
    adjacent_overlap_points,
    decompose_normalized_pou_gradient,
    evaluate_assembled_fields,
    evaluate_local_fields,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "case" / "bbf" / "1" / "results"
OUT = RESULTS / "strain_interface_phase" / "diagnostics"

RUNS = {
    "cosine_pou": "AdaptiveFCN_All_4x1_DIC_ZNSSD_cosine_pou",
    "quintic_pou": "AdaptiveFCN_All_4x1_DIC_ZNSSD_quintic_pou",
    "hermite_1d_c1": "AdaptiveFCN_All_4x1_DIC_ZNSSD_hermite_1d_c1",
    "bicubic_hermite_c1": "AdaptiveFCN_All_4x1_DIC_ZNSSD_bicubic_hermite_c1",
}
ASSEMBLIES = {
    "cosine_pou": DIC_assemblies.CosinePOUAssembly,
    "quintic_pou": DIC_assemblies.QuinticPOUAssembly,
    "hermite_1d_c1": DIC_assemblies.Hermite1DC1Assembly,
    "bicubic_hermite_c1": DIC_assemblies.BicubicHermiteC1Assembly,
}
LABELS = {
    "cosine_pou": "Cosine POU",
    "quintic_pou": "Quintic POU",
    "hermite_1d_c1": "Hermite 1D C1",
    "bicubic_hermite_c1": "Bicubic Hermite C1",
}
COMPONENTS = ("exx", "exy", "eyy")


def rms(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def maximum(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.max(np.abs(values))) if values.size else float("nan")


def load_checkpoint(method):
    run = RUNS[method]
    path = RESULTS / "models" / run / "model_00000500.jax"
    with path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    return checkpoint[1], path


def model_fns(assembly):
    local = (
        RectangularDecompositionND.norm_fn,
        DIC_networks.AdaptiveFCN.network_fn,
        RectangularDecompositionND.unnorm_fn,
        RectangularDecompositionND.window_fn,
    )
    return local, local + (assembly,)


def roi_points(params):
    mask = np.asarray(params["static"]["problem"]["mask"], dtype=bool)
    ys, xs = np.where(mask)
    return mask, np.stack([xs, ys], axis=1).astype(np.float32)


def to_map(values, points, shape):
    values = np.asarray(values)
    result = np.full(shape + values.shape[1:], np.nan, dtype=np.float64)
    xs = points[:, 0].astype(int)
    ys = points[:, 1].astype(int)
    result[ys, xs] = values
    return result


def plot_map(axis, image, title, limits=None, cmap="turbo"):
    kwargs = {}
    if limits is not None:
        kwargs.update(vmin=limits[0], vmax=limits[1])
    shown = axis.imshow(image, origin="upper", cmap=cmap, **kwargs)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return shown


def interface_field_figure(method, index, points, left, right, assembled, shape):
    fields = [
        ("u", left["value"][:, 0], right["value"][:, 0], assembled["value"][:, 0]),
        ("v", left["value"][:, 1], right["value"][:, 1], assembled["value"][:, 1]),
        ("exx", left["strain"][:, 0], right["strain"][:, 0], assembled["strain"][:, 0]),
        ("exy", left["strain"][:, 1], right["strain"][:, 1], assembled["strain"][:, 1]),
        ("eyy", left["strain"][:, 2], right["strain"][:, 2], assembled["strain"][:, 2]),
    ]
    figure, axes = plt.subplots(5, 3, figsize=(12, 12), constrained_layout=True)
    for row, (name, left_value, right_value, assembled_value) in enumerate(fields):
        values = np.concatenate([left_value, right_value, assembled_value])
        limits = (float(np.nanpercentile(values, 1)), float(np.nanpercentile(values, 99)))
        if limits[0] == limits[1]:
            limits = None
        images = [
            to_map(left_value, points, shape),
            to_map(right_value, points, shape),
            to_map(assembled_value, points, shape),
        ]
        for column, (label, image) in enumerate(zip(("left local", "right local", "assembled"), images)):
            shown = plot_map(axes[row, column], image, f"{name}: {label}", limits)
        figure.colorbar(shown, ax=axes[row, :], shrink=0.72)
    figure.suptitle(f"{LABELS[method]} — interface {index+1}: local vs assembled")
    figure.savefig(
        OUT / f"{method}_interface{index+1}_local_vs_assembled.png",
        dpi=160, facecolor="white")
    plt.close(figure)


def interface_mismatch_figure(method, index, points, left, right, shape):
    differences = [
        ("Delta u", left["value"][:, 0]-right["value"][:, 0]),
        ("Delta v", left["value"][:, 1]-right["value"][:, 1]),
        ("Delta exx", left["strain"][:, 0]-right["strain"][:, 0]),
        ("Delta exy", left["strain"][:, 1]-right["strain"][:, 1]),
        ("Delta eyy", left["strain"][:, 2]-right["strain"][:, 2]),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(18, 3.2), constrained_layout=True)
    for axis, (name, values) in zip(axes, differences):
        limit = float(np.nanpercentile(np.abs(values), 99))
        shown = plot_map(
            axis, to_map(values, points, shape), name,
            (-limit, limit) if limit else None, cmap="coolwarm")
        figure.colorbar(shown, ax=axis, shrink=0.7)
    figure.suptitle(f"{LABELS[method]} — interface {index+1}: local mismatch")
    figure.savefig(
        OUT / f"{method}_interface{index+1}_local_mismatch.png",
        dpi=160, facecolor="white")
    plt.close(figure)


def trace_figure(method, index, x_interface, mask, params, local_fns, full_fns, assembly):
    floor_x = int(np.floor(x_interface))
    ceil_x = int(np.ceil(x_interface))
    valid = mask[:, floor_x] & mask[:, ceil_x]
    y = np.where(valid)[0].astype(np.float32)
    points = np.stack([np.full_like(y, x_interface), y], axis=1)
    left_ids = np.full(points.shape[0], index, dtype=np.int32)
    right_ids = np.full(points.shape[0], index+1, dtype=np.int32)
    left = evaluate_local_fields(params, points, left_ids, local_fns)
    right = evaluate_local_fields(params, points, right_ids, local_fns)
    assembled = evaluate_assembled_fields(
        params, points, RectangularDecompositionND, assembly, full_fns)
    left = {key: np.asarray(value) for key, value in left.items()}
    right = {key: np.asarray(value) for key, value in right.items()}
    assembled = {key: np.asarray(value) for key, value in assembled.items()}

    traces = [
        ("u", left["value"][:, 0], right["value"][:, 0], assembled["value"][:, 0]),
        ("du/dx", left["gradient"][:, 0, 0], right["gradient"][:, 0, 0], assembled["gradient"][:, 0, 0]),
        ("exx", left["strain"][:, 0], right["strain"][:, 0], assembled["strain"][:, 0]),
        ("exy", left["strain"][:, 1], right["strain"][:, 1], assembled["strain"][:, 1]),
        ("eyy", left["strain"][:, 2], right["strain"][:, 2], assembled["strain"][:, 2]),
    ]
    figure, axes = plt.subplots(5, 2, figsize=(12, 13), sharex=True, constrained_layout=True)
    for row, (name, left_value, right_value, assembled_value) in enumerate(traces):
        axes[row, 0].plot(y, left_value, label="left local", lw=1.2)
        axes[row, 0].plot(y, right_value, label="right local", lw=1.2)
        axes[row, 0].plot(y, assembled_value, label="assembled", lw=1.2, ls="--")
        axes[row, 0].set_ylabel(name)
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].plot(y, left_value-right_value, color="tab:red", lw=1.2)
        axes[row, 1].set_ylabel(f"Delta {name}")
        axes[row, 1].grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("y [px]")
    axes[-1, 1].set_xlabel("y [px]")
    figure.suptitle(
        f"{LABELS[method]} — local traces at interface {index+1}, x={x_interface:g}")
    figure.savefig(
        OUT / f"{method}_interface{index+1}_local_traces.png",
        dpi=160, facecolor="white")
    plt.close(figure)


def diagnose_method(method):
    params, checkpoint_path = load_checkpoint(method)
    assembly = ASSEMBLIES[method]
    local_fns, full_fns = model_fns(assembly)
    mask, points_roi = roi_points(params)
    overlaps = adjacent_overlap_points(params, points_roi)
    result = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "interfaces": [],
    }
    archive = {}

    for index, overlap in enumerate(overlaps):
        points = overlap["points"].astype(np.float32)
        left_ids = np.full(points.shape[0], overlap["left_model"], dtype=np.int32)
        right_ids = np.full(points.shape[0], overlap["right_model"], dtype=np.int32)
        left = evaluate_local_fields(params, points, left_ids, local_fns)
        right = evaluate_local_fields(params, points, right_ids, local_fns)
        assembled = evaluate_assembled_fields(
            params, points, RectangularDecompositionND, assembly, full_fns)
        left = {key: np.asarray(value) for key, value in left.items()}
        right = {key: np.asarray(value) for key, value in right.items()}
        assembled = {key: np.asarray(value) for key, value in assembled.items()}
        displacement_mismatch = left["value"]-right["value"]
        gradient_mismatch = left["gradient"]-right["gradient"]
        strain_mismatch = left["strain"]-right["strain"]
        local_average_strain = 0.5*(left["strain"]+right["strain"])
        assembly_distortion = assembled["strain"]-local_average_strain

        interface_metrics = {
            key: overlap[key] for key in (
                "left_model", "right_model", "x_left", "x_right", "x_interface")
        }
        interface_metrics.update({
            "point_count": int(points.shape[0]),
            "rms_displacement_mismatch": rms(displacement_mismatch),
            "max_displacement_mismatch": maximum(displacement_mismatch),
            "rms_gradient_mismatch": rms(gradient_mismatch),
            "max_gradient_mismatch": maximum(gradient_mismatch),
            "rms_local_strain_mismatch": rms(strain_mismatch),
            "max_local_strain_mismatch": maximum(strain_mismatch),
            "rms_assembled_minus_local_average_strain": rms(assembly_distortion),
            "max_assembled_minus_local_average_strain": maximum(assembly_distortion),
            "strain_mismatch_by_component": {
                name: {"rms": rms(strain_mismatch[:, component]),
                       "max": maximum(strain_mismatch[:, component])}
                for component, name in enumerate(COMPONENTS)
            },
        })

        if method in ("cosine_pou", "quintic_pou"):
            decomposition = decompose_normalized_pou_gradient(
                params, points, RectangularDecompositionND,
                assembly, full_fns)
            decomposition = {
                key: np.asarray(value) for key, value in decomposition.items()}
            identity_error = (
                decomposition["total_gradient"]-assembled["gradient"])
            window_rms = rms(decomposition["window_strain"])
            local_mismatch_rms = interface_metrics["rms_local_strain_mismatch"]
            interface_metrics.update({
                "pou_gradient_identity_max_error": maximum(identity_error),
                "rms_pou_local_gradient_strain": rms(decomposition["local_strain"]),
                "rms_pou_window_gradient_strain": window_rms,
                "rms_pou_total_strain": rms(decomposition["total_strain"]),
                "assembly_to_local_mismatch_ratio": (
                    window_rms/local_mismatch_rms if local_mismatch_rms else float("nan")),
            })
            for key in ("total_strain", "local_strain", "window_strain"):
                archive[f"interface{index+1}_{key}"] = decomposition[key]

        result["interfaces"].append(interface_metrics)
        for prefix, fields in (("left", left), ("right", right), ("assembled", assembled)):
            for key, value in fields.items():
                archive[f"interface{index+1}_{prefix}_{key}"] = value
        archive[f"interface{index+1}_points"] = points

        interface_field_figure(
            method, index, points, left, right, assembled, mask.shape)
        interface_mismatch_figure(method, index, points, left, right, mask.shape)
        trace_figure(
            method, index, overlap["x_interface"], mask, params,
            local_fns, full_fns, assembly)

    np.savez_compressed(OUT / f"{method}_local_interface_fields.npz", **archive)
    return result, params, mask, points_roi, full_fns


def summary_figure(metrics):
    methods = list(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(3)
    width = 0.18
    for offset, method in enumerate(methods):
        records = metrics[method]["interfaces"]
        axes[0].bar(x+(offset-1.5)*width,
                    [record["rms_local_strain_mismatch"] for record in records],
                    width, label=LABELS[method])
        axes[1].bar(x+(offset-1.5)*width,
                    [record["rms_assembled_minus_local_average_strain"] for record in records],
                    width, label=LABELS[method])
        values = [record.get("rms_pou_window_gradient_strain", np.nan)
                  for record in records]
        axes[2].bar(x+(offset-1.5)*width, values, width, label=LABELS[method])
    titles = (
        "Local-to-local strain mismatch RMS",
        "Assembled minus local-average strain RMS",
        "POU window-gradient strain RMS",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xticks(x, ("Interface 1", "Interface 2", "Interface 3"))
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.savefig(OUT / "local_network_strain_diagnostics.png", dpi=180, facecolor="white")
    plt.close(figure)


def mismatch_summary_map(params, mask, points_roi, full_fns):
    overlaps = adjacent_overlap_points(params, points_roi)
    local_fns = full_fns[:4]
    figure, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row, overlap in enumerate(overlaps):
        points = overlap["points"].astype(np.float32)
        left = evaluate_local_fields(
            params, points, np.full(len(points), row), local_fns)
        right = evaluate_local_fields(
            params, points, np.full(len(points), row+1), local_fns)
        difference = np.asarray(left["strain"]-right["strain"])
        for column, name in enumerate(COMPONENTS):
            values = difference[:, column]
            limit = float(np.nanpercentile(np.abs(values), 99))
            shown = plot_map(
                axes[row, column], to_map(values, points, mask.shape),
                f"Interface {row+1}: Delta {name}",
                (-limit, limit) if limit else None, "coolwarm")
            figure.colorbar(shown, ax=axes[row, column], shrink=0.7)
    figure.suptitle("Cosine-trained local-network strain mismatch maps")
    figure.savefig(OUT / "local_strain_mismatch_maps.png", dpi=180, facecolor="white")
    plt.close(figure)


def pou_decomposition_figure(cache):
    figure, axes = plt.subplots(6, 3, figsize=(12, 16), constrained_layout=True)
    row = 0
    for method in ("cosine_pou", "quintic_pou"):
        params, mask, points, full_fns = cache[method]
        assembly = ASSEMBLIES[method]
        decomposition = decompose_normalized_pou_gradient(
            params, points, RectangularDecompositionND, assembly, full_fns)
        total = np.asarray(decomposition["total_strain"])
        local = np.asarray(decomposition["local_strain"])
        window = np.asarray(decomposition["window_strain"])
        np.savez_compressed(
            OUT / f"{method}_pou_gradient_decomposition.npz",
            points=points, total_strain=total,
            local_strain=local, window_strain=window)
        for component, name in enumerate(COMPONENTS):
            arrays = [total[:, component], local[:, component], window[:, component]]
            limit = float(np.nanpercentile(np.abs(np.concatenate(arrays)), 99))
            for column, (label, values) in enumerate(zip(("total", "local", "window"), arrays)):
                shown = plot_map(
                    axes[row, column], to_map(values, points, mask.shape),
                    f"{LABELS[method]} {name}: {label}",
                    (-limit, limit) if limit else None, "coolwarm")
            figure.colorbar(shown, ax=axes[row, :], shrink=0.65)
            row += 1
    figure.suptitle("Normalized POU strain decomposition: total = local + window")
    figure.savefig(OUT / "pou_gradient_decomposition.png", dpi=180, facecolor="white")
    plt.close(figure)


def classify(metrics):
    classification = {}
    for method in ("cosine_pou", "quintic_pou"):
        records = metrics[method]["interfaces"]
        local = float(np.sqrt(np.mean([
            record["rms_local_strain_mismatch"]**2 for record in records])))
        window = float(np.sqrt(np.mean([
            record["rms_pou_window_gradient_strain"]**2 for record in records])))
        ratio = window/local if local else float("nan")
        classification[method] = {
            "pooled_local_strain_mismatch_rms": local,
            "pooled_window_gradient_strain_rms": window,
            "window_to_local_ratio": ratio,
        }
    interface_ratios = [
        record["assembly_to_local_mismatch_ratio"]
        for record in metrics["cosine_pou"]["interfaces"]
    ]
    if max(interface_ratios) > 2.0 and min(interface_ratios) < 0.5:
        dominant = "BOTH"
    elif min(interface_ratios) > 2.0:
        dominant = "ASSEMBLY"
    elif max(interface_ratios) < 0.5:
        dominant = "LOCAL"
    else:
        dominant = "BOTH"
    classification["dominant_artifact_source"] = dominant
    classification["rule"] = (
        "Classify spatially: ASSEMBLY only if every cosine interface ratio > 2; "
        "LOCAL only if every ratio < 0.5; otherwise BOTH.")
    return classification


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    cache = {}
    for method in RUNS:
        print(f"Diagnosing {method}...", flush=True)
        result, params, mask, points, full_fns = diagnose_method(method)
        metrics[method] = result
        cache[method] = (params, mask, points, full_fns)

    classification = classify(metrics)
    payload = {
        "metric_scope": (
            "Raw adjacent local neural fields evaluated at identical physical coordinates "
            "throughout each common overlap. No finite-offset trace is used as local mismatch."),
        "amplification_definition": (
            "assembly_to_local_mismatch_ratio = RMS normalized-POU window-gradient strain "
            "/ RMS adjacent local-network strain mismatch, in the same overlap."),
        "methods": metrics,
        "classification": classification,
    }
    (OUT / "local_interface_diagnostics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    summary_figure(metrics)
    mismatch_summary_map(*cache["cosine_pou"])
    pou_decomposition_figure(cache)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
