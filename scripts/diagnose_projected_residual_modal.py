"""Diagnose P0/P1/P2 content of the cosine-500 raw local fields.

This script is read-only with respect to training.  Polynomial strain is
obtained by differentiating the fitted displacement polynomial analytically;
strain itself is never projected.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_strain_interface_sweep import (
    ROOT, _checkpoint, _local_batched, _roi, rms,
)
from segpinndic.DIC_field_diagnostics import adjacent_overlap_points


OUT = (ROOT / "case" / "bbf" / "1" / "results"
       / "projected_residual_phase" / "modal_diagnostics")
ORDERS = (0, 1, 2)


def basis(points, order):
    xi, eta = points[:, 0], points[:, 1]
    columns = [np.ones_like(xi)]
    if order >= 1:
        columns.extend([xi, eta])
    if order >= 2:
        columns.extend([xi*xi, xi*eta, eta*eta])
    return np.stack(columns, axis=1)


def basis_gradient(points, order, physical_scale):
    xi, eta = points[:, 0], points[:, 1]
    n = len(points)
    columns = [np.zeros((n, 2))]
    if order >= 1:
        columns.extend([
            np.stack([np.ones(n)/physical_scale[0], np.zeros(n)], axis=1),
            np.stack([np.zeros(n), np.ones(n)/physical_scale[1]], axis=1),
        ])
    if order >= 2:
        columns.extend([
            np.stack([2*xi/physical_scale[0], np.zeros(n)], axis=1),
            np.stack([eta/physical_scale[0], xi/physical_scale[1]], axis=1),
            np.stack([np.zeros(n), 2*eta/physical_scale[1]], axis=1),
        ])
    return np.stack(columns, axis=1)


def strain_from_gradient(gradient):
    return np.stack([
        gradient[:, 0, 0],
        .5*(gradient[:, 0, 1]+gradient[:, 1, 0]),
        gradient[:, 1, 1],
    ], axis=1)


def fit(values, normalized_points, physical_scale, order):
    phi = basis(normalized_points, order)
    coefficients, *_ = np.linalg.lstsq(phi, values, rcond=None)
    low_value = phi@coefficients
    basis_dx = basis_gradient(normalized_points, order, physical_scale)
    low_gradient = np.einsum("nbx,bd->ndx", basis_dx, coefficients)
    return coefficients, low_value, low_gradient


def evaluate_polynomial(coefficients, normalized_points, physical_scale, order):
    value = basis(normalized_points, order)@coefficients
    gradient = np.einsum(
        "nbx,bd->ndx",
        basis_gradient(normalized_points, order, physical_scale), coefficients)
    return value, gradient, strain_from_gradient(gradient)


def normalized(points, xmin, xmax):
    center = .5*(xmin+xmax)
    scale = .5*(xmax-xmin)
    return (points-center)/scale, scale


def model_decomposition(params, roi_points, model):
    xmins, xmaxs = [np.asarray(value)
                    for value in params["static"]["decomposition"]
                    ["subdomain"]["params"][:2]]
    selected = np.all(
        (roi_points >= xmins[model]) & (roi_points <= xmaxs[model]), axis=1)
    points = roi_points[selected]
    ids = np.full(len(points), model, dtype=np.int32)
    full = _local_batched(params, points, ids)
    norm_points, scale = normalized(points, xmins[model], xmaxs[model])
    orders = {}
    for order in ORDERS:
        coefficients, low_value, low_gradient = fit(
            full["value"], norm_points, scale, order)
        high_value = full["value"]-low_value
        high_gradient = full["gradient"]-low_gradient
        low_strain = strain_from_gradient(low_gradient)
        high_strain = strain_from_gradient(high_gradient)
        orders[str(order)] = {
            "coefficients": coefficients,
            "low_value": low_value,
            "high_value": high_value,
            "low_gradient": low_gradient,
            "high_gradient": high_gradient,
            "low_strain": low_strain,
            "high_strain": high_strain,
            "metrics": {
                "total_displacement_rms": rms(full["value"]),
                "low_displacement_rms": rms(low_value),
                "orthogonal_displacement_rms": rms(high_value),
                "low_displacement_fraction": rms(low_value)/rms(full["value"]),
                "orthogonal_displacement_fraction": rms(high_value)/rms(full["value"]),
                "identity_displacement_rms": rms(
                    full["value"]-low_value-high_value),
                "total_strain_rms": rms(full["strain"]),
                "low_strain_rms": rms(low_strain),
                "orthogonal_strain_rms": rms(high_strain),
                "low_strain_fraction": rms(low_strain)/rms(full["strain"]),
                "orthogonal_strain_fraction": rms(high_strain)/rms(full["strain"]),
            },
        }
    return {
        "points": points, "full": full, "normalized": norm_points,
        "physical_scale": scale, "orders": orders,
    }


def interface_decomposition(params, roi_points, models):
    records = []
    for overlap in adjacent_overlap_points(params, roi_points):
        points = overlap["points"].astype(np.float32)
        left_id, right_id = overlap["left_model"], overlap["right_model"]
        full = []
        pieces = []
        for model in (left_id, right_id):
            values = _local_batched(
                params, points, np.full(len(points), model, dtype=np.int32))
            full.append(values)
            xmins, xmaxs = [np.asarray(value)[model]
                            for value in params["static"]["decomposition"]
                            ["subdomain"]["params"][:2]]
            norm_points, scale = normalized(points, xmins, xmaxs)
            by_order = {}
            for order in ORDERS:
                coefficients = models[model]["orders"][str(order)]["coefficients"]
                value, gradient, strain = evaluate_polynomial(
                    coefficients, norm_points, scale, order)
                by_order[str(order)] = {
                    "low_value": value,
                    "low_gradient": gradient,
                    "low_strain": strain,
                    "orthogonal_value": values["value"]-value,
                    "orthogonal_gradient": values["gradient"]-gradient,
                    "orthogonal_strain": values["strain"]-strain,
                }
            pieces.append(by_order)
        full_delta = {
            key: full[0][key]-full[1][key]
            for key in ("value", "gradient", "strain")}
        item = {
            "interface_x": overlap["x_interface"],
            "left_model": left_id,
            "right_model": right_id,
            "point_count": len(points),
            "full": {
                "displacement_mismatch_rms": rms(full_delta["value"]),
                "gradient_mismatch_rms": rms(full_delta["gradient"]),
                "strain_mismatch_rms": rms(full_delta["strain"]),
            },
            "orders": {},
        }
        for order in ORDERS:
            low = {name: pieces[0][str(order)][f"low_{name}"]
                   - pieces[1][str(order)][f"low_{name}"]
                   for name in ("value", "gradient", "strain")}
            high = {name: pieces[0][str(order)][f"orthogonal_{name}"]
                    - pieces[1][str(order)][f"orthogonal_{name}"]
                    for name in ("value", "gradient", "strain")}
            item["orders"][str(order)] = {
                "low_displacement_mismatch_rms": rms(low["value"]),
                "orthogonal_displacement_mismatch_rms": rms(high["value"]),
                "low_gradient_mismatch_rms": rms(low["gradient"]),
                "orthogonal_gradient_mismatch_rms": rms(high["gradient"]),
                "low_strain_mismatch_rms": rms(low["strain"]),
                "orthogonal_strain_mismatch_rms": rms(high["strain"]),
                "F_low_displacement": rms(low["value"])/rms(full_delta["value"]),
                "F_high_displacement": rms(high["value"])/rms(full_delta["value"]),
                "F_low_strain": rms(low["strain"])/rms(full_delta["strain"]),
                "F_high_strain": rms(high["strain"])/rms(full_delta["strain"]),
            }
        records.append(item)
    return records


def map_values(shape, points, values):
    image = np.full(shape, np.nan)
    image[points[:, 1].astype(int), points[:, 0].astype(int)] = values
    return image


def modal_figures(mask, models, interfaces):
    columns = (("full", None), ("P1 low", (1, "low")),
               ("P1 orth", (1, "high")), ("P2 low", (2, "low")),
               ("P2 orth", (2, "high")))
    for field, filename in (("value", "01_local_modal_decomposition.png"),
                            ("strain", "02_local_modal_strain.png")):
        fig, axes = plt.subplots(4, 5, figsize=(18, 10), dpi=160,
                                 constrained_layout=True)
        for model, data in models.items():
            candidates = []
            for _, spec in columns:
                if spec is None:
                    values = data["full"][field]
                else:
                    order, part = spec
                    values = data["orders"][str(order)][f"{part}_{field}"]
                candidates.append(np.linalg.norm(values, axis=1))
            limit = np.percentile(np.concatenate(candidates), 99)
            for col, ((label, _), values) in enumerate(zip(columns, candidates)):
                shown = axes[model, col].imshow(
                    map_values(mask.shape, data["points"], values),
                    vmin=0, vmax=limit, cmap="viridis")
                axes[model, col].set_title(f"local {model}: {label}", fontsize=8)
                axes[model, col].axis("off")
                fig.colorbar(shown, ax=axes[model, col], fraction=.03, pad=.01)
        fig.savefig(OUT / filename, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), dpi=160,
                             constrained_layout=True)
    for row, item in enumerate(interfaces):
        labels = ["P0", "P1", "P2"]
        for col, quantity in enumerate(("displacement", "strain")):
            full = item["full"][f"{quantity}_mismatch_rms"]
            low = [item["orders"][str(order)][f"low_{quantity}_mismatch_rms"]
                   for order in ORDERS]
            high = [item["orders"][str(order)][f"orthogonal_{quantity}_mismatch_rms"]
                    for order in ORDERS]
            x = np.arange(3)
            axes[row, col].bar(x-.2, low, width=.4, label="low mode")
            axes[row, col].bar(x+.2, high, width=.4, label="orthogonal")
            axes[row, col].axhline(full, color="black", linestyle="--",
                                   label="full")
            axes[row, col].set_xticks(x, labels)
            axes[row, col].set_title(
                f"x={item['interface_x']:g}: {quantity} mismatch")
            axes[row, col].grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT / "03_interface_modal_mismatch.png", bbox_inches="tight")
    plt.close(fig)


def serializable_models(models):
    result = {}
    for model, data in models.items():
        result[str(model)] = {
            "point_count": len(data["points"]),
            "physical_scale": data["physical_scale"].tolist(),
            "orders": {
                order: {
                    "coefficients": value["coefficients"].tolist(),
                    "metrics": value["metrics"],
                } for order, value in data["orders"].items()
            },
        }
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params, loss, checkpoint = _checkpoint("cosine_pou", 500)
    mask, roi_points = _roi(params)
    models = {
        model: model_decomposition(params, roi_points, model)
        for model in range(int(params["static"]["decomposition"]["m"]))
    }
    interfaces = interface_decomposition(params, roi_points, models)
    modal_figures(mask, models, interfaces)
    payload = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_loss": loss,
        "coordinate_system": "subdomain-local normalized xi,eta in [-1,1]",
        "weighting": "uniform over every valid ROI point in each subdomain",
        "models": serializable_models(models),
        "interfaces": interfaces,
    }
    path = OUT / "cosine500_local_modal_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "models": {model: {
            order: data["orders"][order]["metrics"]
            for order in ("0", "1", "2")}
            for model, data in payload["models"].items()},
        "interfaces": interfaces,
    }, indent=2))


if __name__ == "__main__":
    main()
