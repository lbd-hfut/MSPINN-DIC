"""Quantitative and visual comparison of the four bbf/1 assembly runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "case" / "bbf" / "1"
RESULTS = CASE / "results"
OUT = RESULTS / "assembly_comparison"

RUNS = {
    "cosine_pou": "AdaptiveFCN_All_4x1_DIC_ZNSSD_cosine_pou",
    "quintic_pou": "AdaptiveFCN_All_4x1_DIC_ZNSSD_quintic_pou",
    "hermite_1d_c1": "AdaptiveFCN_All_4x1_DIC_ZNSSD_hermite_1d_c1",
    "bicubic_hermite_c1": "AdaptiveFCN_All_4x1_DIC_ZNSSD_bicubic_hermite_c1",
}
LABELS = {
    "cosine_pou": "Cosine POU",
    "quintic_pou": "Quintic POU",
    "hermite_1d_c1": "Hermite 1D C1",
    "bicubic_hermite_c1": "Bicubic Hermite C1",
}
FIELDS = ("u", "v", "exx", "exy", "eyy")
STRAINS = ("exx", "exy", "eyy")


def rms(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def load_runs():
    runs = {}
    for method, run in RUNS.items():
        mat_path = RESULTS / "mats" / run / "segpinndic_001.mat"
        metrics_path = RESULTS / "summaries" / run / "run_metrics_roi0_pair1.json"
        data = loadmat(mat_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs[method] = {
            "fields": {name: np.asarray(data[name], dtype=np.float64) for name in FIELDS},
            "runtime": metrics,
            "mat_path": str(mat_path.relative_to(ROOT)),
            "metrics_path": str(metrics_path.relative_to(ROOT)),
        }
    return runs


def geometry(mask):
    ys, xs = np.where(mask)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    nodes_x = np.linspace(xmin, xmax, 5)
    nodes_y = np.linspace(ymin, ymax, 2)
    centers_x = 0.5*(nodes_x[:-1]+nodes_x[1:])
    centers_y = 0.5*(nodes_y[:-1]+nodes_y[1:])
    widths_x = np.full(4, 1.6*np.diff(nodes_x).min())
    widths_y = np.full(1, 1.6*np.diff(nodes_y).min())
    support_left = centers_x-0.5*widths_x
    support_right = centers_x+0.5*widths_x
    transition_left = support_left[1:]
    transition_right = support_right[:-1]
    return {
        "roi_bbox": [xmin, xmax, ymin, ymax],
        "nodes_x": nodes_x,
        "centers_x": centers_x,
        "centers_y": centers_y,
        "widths_x": widths_x,
        "widths_y": widths_y,
        "interfaces": nodes_x[1:-1],
        "support_left": support_left,
        "support_right": support_right,
        "transition_left": transition_left,
        "transition_right": transition_right,
    }


def finite_difference_roughness(components, mask, interfaces):
    stacked = np.stack(components, axis=-1)
    valid_x = mask[:, 1:] & mask[:, :-1]
    valid_y = mask[1:, :] & mask[:-1, :]
    dx = np.sqrt(np.mean(np.diff(stacked, axis=1)**2, axis=-1))
    dy = np.sqrt(np.mean(np.diff(stacked, axis=0)**2, axis=-1))
    x_mid = np.arange(mask.shape[1]-1, dtype=np.float64)+0.5
    distance_x = np.min(np.abs(x_mid[:, None]-interfaces[None, :]), axis=1)

    result = {
        "global": rms(np.concatenate([dx[valid_x], dy[valid_y]])),
        "interface_band": rms(dx[valid_x & (distance_x[None, :] <= 5.0)]),
        "near_interface": rms(dx[valid_x & (distance_x[None, :] > 5.0) &
                                      (distance_x[None, :] <= 15.0)]),
        "interior": rms(dx[valid_x & (distance_x[None, :] > 15.0)]),
    }
    result["periodic_artifact_ratio"] = (
        result["interface_band"]/result["interior"]
        if result["interior"] > 0 else float("nan"))
    return result


def band_field_rms(components, mask, interfaces):
    stacked = np.stack(components, axis=-1)
    magnitude = np.sqrt(np.mean(stacked**2, axis=-1))
    x = np.arange(mask.shape[1], dtype=np.float64)
    distance = np.min(np.abs(x[:, None]-interfaces[None, :]), axis=1)
    return {
        "global": rms(magnitude[mask]),
        "interface_band": rms(magnitude[mask & (distance[None, :] <= 5.0)]),
        "near_interface": rms(magnitude[mask & (distance[None, :] > 5.0) &
                                                (distance[None, :] <= 15.0)]),
        "interior": rms(magnitude[mask & (distance[None, :] > 15.0)]),
    }


def interface_traces(fields, mask, interfaces):
    u, v = fields["u"], fields["v"]
    exx, exy, eyy = fields["exx"], fields["exy"], fields["eyy"]
    records = []
    for interface in interfaces:
        if float(interface).is_integer():
            left = int(interface)-1
            right = int(interface)+1
        else:
            left = int(np.floor(interface))
            right = int(np.ceil(interface))
        delta = 0.5*(right-left)
        rows = np.arange(1, mask.shape[0]-1)
        valid = (mask[rows, left-1] & mask[rows, left] &
                 mask[rows, right] & mask[rows, right+1] &
                 mask[rows-1, left] & mask[rows+1, left] &
                 mask[rows-1, right] & mask[rows+1, right])
        rows = rows[valid]

        displacement_jump = np.stack([
            u[rows, right]-u[rows, left],
            v[rows, right]-v[rows, left],
        ], axis=1)
        grad_left = np.stack([
            u[rows, left]-u[rows, left-1],
            0.5*(u[rows+1, left]-u[rows-1, left]),
            v[rows, left]-v[rows, left-1],
            0.5*(v[rows+1, left]-v[rows-1, left]),
        ], axis=1)
        grad_right = np.stack([
            u[rows, right+1]-u[rows, right],
            0.5*(u[rows+1, right]-u[rows-1, right]),
            v[rows, right+1]-v[rows, right],
            0.5*(v[rows+1, right]-v[rows-1, right]),
        ], axis=1)
        strain_jump = np.stack([
            exx[rows, right]-exx[rows, left],
            exy[rows, right]-exy[rows, left],
            eyy[rows, right]-eyy[rows, left],
        ], axis=1)
        records.append({
            "interface_x": float(interface),
            "trace_definition": (
                f"left x={left}, right x={right}, delta={delta:g} px"),
            "valid_rows": int(rows.size),
            "J_u_rms": rms(displacement_jump),
            "J_grad_u_rms": rms(grad_right-grad_left),
            "J_strain_rms": rms(strain_jump),
            "max_abs_strain_jump": float(np.max(np.abs(strain_jump))),
        })
    return records


def compute_metrics(runs, mask, geom):
    summary = {}
    interfaces = geom["interfaces"]
    for method, run in runs.items():
        fields = run["fields"]
        displacement = [fields["u"], fields["v"]]
        strains = [fields[name] for name in STRAINS]
        strain_roughness = finite_difference_roughness(strains, mask, interfaces)
        traces = interface_traces(fields, mask, interfaces)
        summary[method] = {
            "final_dic_loss": run["runtime"]["final_loss"],
            "compile_seconds": run["runtime"]["compile_seconds"],
            "total_seconds": run["runtime"]["total_seconds"],
            "prediction_seconds": run["runtime"]["prediction_seconds"],
            "global_displacement_roughness": finite_difference_roughness(
                displacement, mask, interfaces)["global"],
            "global_strain_roughness": strain_roughness["global"],
            "interface_strain_roughness": strain_roughness["interface_band"],
            "near_interface_strain_roughness": strain_roughness["near_interface"],
            "interior_strain_roughness": strain_roughness["interior"],
            "subdomain_periodic_artifact_ratio": strain_roughness[
                "periodic_artifact_ratio"],
            "strain_field_rms_by_band": band_field_rms(strains, mask, interfaces),
            "interface_traces": traces,
            "max_strain_jump": max(item["max_abs_strain_jump"] for item in traces),
            "rms_strain_jump": rms([item["J_strain_rms"] for item in traces]),
            "rms_displacement_jump": rms([item["J_u_rms"] for item in traces]),
            "rms_gradient_jump": rms([item["J_grad_u_rms"] for item in traces]),
        }
    return summary


def masked(field, mask):
    return np.where(mask, field, np.nan)


def common_limits(runs, field, mask):
    values = np.concatenate([run["fields"][field][mask] for run in runs.values()])
    return float(np.nanpercentile(values, 0.5)), float(np.nanpercentile(values, 99.5))


def comparison_figure(runs, mask, fields, filename, geom, diagnostics=False):
    limits = {field: common_limits(runs, field, mask) for field in fields}
    fig, axes = plt.subplots(len(RUNS), len(fields), figsize=(5*len(fields), 3.2*len(RUNS)),
                             constrained_layout=True, dpi=180)
    axes = np.atleast_2d(axes)
    for row, method in enumerate(RUNS):
        for col, field in enumerate(fields):
            ax = axes[row, col]
            image = ax.imshow(masked(runs[method]["fields"][field], mask), cmap="turbo",
                              vmin=limits[field][0], vmax=limits[field][1])
            if diagnostics:
                for x in geom["interfaces"]:
                    ax.axvline(x, color="white", linestyle="--", linewidth=0.8)
                for x in np.concatenate([geom["transition_left"], geom["transition_right"]]):
                    ax.axvline(x, color="black", linestyle=":", linewidth=0.55)
            ax.set_title(f"{LABELS[method]} — {field}")
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def geometry_figure(mask, geom):
    fig, ax = plt.subplots(figsize=(13, 4), dpi=180)
    ymin, ymax = geom["roi_bbox"][2:]
    for i, (left, right, center) in enumerate(zip(
            geom["support_left"], geom["support_right"], geom["centers_x"])):
        ax.add_patch(plt.Rectangle((left, ymin), right-left, ymax-ymin,
                                   fill=False, linewidth=1.8, label=f"support {i+1}"))
        ax.plot(center, 0.5*(ymin+ymax), "o", markersize=6)
    for i, (left, right) in enumerate(zip(
            geom["transition_left"], geom["transition_right"])):
        ax.axvspan(left, right, alpha=0.16, color=f"C{i}")
    for x in geom["interfaces"]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.0)
    ax.contour(mask.astype(float), levels=[0.5], colors="black", linewidths=0.7)
    ax.set_xlim(geom["support_left"].min()-5, geom["support_right"].max()+5)
    ax.set_ylim(ymin-8, ymax+8)
    ax.set_aspect("equal")
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    ax.set_title("bbf/1 [4,1] subdomain supports, centers, overlaps, and interfaces")
    fig.savefig(OUT / "subdomain_geometry.png", bbox_inches="tight")
    plt.close(fig)


def loss_runtime_figure(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180, constrained_layout=True)
    for method, run in runs.items():
        history = np.asarray(run["runtime"]["adam_loss_history"], dtype=float)
        axes[0].plot(history[:, 0], history[:, 1], label=LABELS[method])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel("ZNSSD loss")
    axes[0].set_title("Photometric convergence")
    axes[0].legend()

    methods = list(RUNS)
    total = [runs[m]["runtime"]["total_seconds"] for m in methods]
    prediction = [runs[m]["runtime"]["prediction_seconds"] for m in methods]
    x = np.arange(len(methods))
    axes[1].bar(x-0.18, total, width=0.36, label="trainer total")
    axes[1].bar(x+0.18, prediction, width=0.36, label="prediction")
    axes[1].set_xticks(x, [LABELS[m] for m in methods], rotation=20, ha="right")
    axes[1].set_ylabel("seconds")
    axes[1].set_title("Wall-clock cost")
    axes[1].legend()
    fig.savefig(OUT / "loss_runtime_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(Image.open(CASE / "03.png").convert("L")) > 0
    runs = load_runs()
    geom = geometry(mask)
    metrics = compute_metrics(runs, mask, geom)
    payload = {
        "ground_truth_available": False,
        "metric_note": (
            "bbf/1 has no displacement/strain ground truth. Reported field metrics are "
            "finite-difference consistency/roughness diagnostics, not accuracy errors."),
        "interface_band_definition": "distance to x=[192.5,332,471.5] <= 5 px",
        "near_interface_definition": "5 < distance <= 15 px",
        "interior_definition": "distance > 15 px",
        "geometry": {key: np.asarray(value).tolist() for key, value in geom.items()},
        "methods": metrics,
    }
    (OUT / "assembly_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    geometry_figure(mask, geom)
    comparison_figure(runs, mask, ("u", "v"), "displacement_comparison.png", geom)
    comparison_figure(runs, mask, STRAINS, "strain_comparison.png", geom)
    comparison_figure(runs, mask, STRAINS, "strain_interface_diagnostics.png",
                      geom, diagnostics=True)
    loss_runtime_figure(runs)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
