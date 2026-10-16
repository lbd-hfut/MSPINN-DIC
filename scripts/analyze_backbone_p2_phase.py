"""Build auditable Phase-A/B/C evidence from 500-step checkpoints only.

Classical POU values are copied from their original, unmodified 500-step
checkpoints.  Backbone variants are evaluated from this phase's own 500-step
checkpoints.  Every reported strain is automatic differentiation of the final
assembled displacement field.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from scripts.analyze_strain_interface_sweep import (
    ROOT, _assembled_batched, _field_metrics, _local_interface_metrics,
    _map_fields, _model_fns, _roi, geometry, rms,
)
from segpinndic.DIC_assemblies import (
    BackboneFirstCosineResidualAssembly, BackboneFirstP2ResidualAssembly,
    CosinePOUAssembly, GlobalBackboneLocalResidualAssembly,
    Hermite1DC1Assembly, QuinticPOUAssembly,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_field_diagnostics import strain_from_gradient
from segpinndic.DIC_trainers import get_inputs


PHASE_ROOT = ROOT / "case" / "bbf" / "1" / "results" / "backbone_p2_phase"
OUT = PHASE_ROOT / "analysis"
REFERENCE_OUT = PHASE_ROOT / "reference_500"
DIAGNOSTIC_OUT = PHASE_ROOT / "overfitting_diagnostic"
OLD_ANALYSIS = (ROOT / "case" / "bbf" / "1" / "results" /
                "strain_interface_phase" / "analysis")
OLD_METRICS = OLD_ANALYSIS / "strain_interface_metrics.json"

CLASSICAL = {
    "cosine_pou": CosinePOUAssembly,
    "quintic_pou": QuinticPOUAssembly,
    "hermite_1d_c1": Hermite1DC1Assembly,
}
BACKBONES = {
    "old_simultaneous": (GlobalBackboneLocalResidualAssembly, "backbone_old"),
    "backbone_direct_cosine": (BackboneFirstCosineResidualAssembly, "backbone_first_direct"),
    "backbone_progressive_cosine": (BackboneFirstCosineResidualAssembly, "backbone_first_cosine"),
    "backbone_progressive_p2": (BackboneFirstP2ResidualAssembly, "backbone_first_p2"),
}
LABELS = {
    "cosine_pou": "Cosine POU",
    "quintic_pou": "Quintic POU",
    "hermite_1d_c1": "Hermite 1D C1",
    "old_simultaneous": "Old simultaneous global + residual",
    "backbone_direct_cosine": "Backbone-first direct cosine",
    "backbone_progressive_cosine": "Backbone-first progressive cosine",
    "backbone_progressive_p2": "Backbone-first progressive P2",
}
RUN = {
    "old_simultaneous": "AdaptiveFCN_All_4x1_DIC_ZNSSD_global_backbone_local_residual",
    "backbone_direct_cosine": "AdaptiveFCN_All_4x1_DIC_ZNSSD_backbone_first_cosine_residual",
    "backbone_progressive_cosine": "AdaptiveFCN_All_4x1_DIC_ZNSSD_backbone_first_cosine_residual",
    "backbone_progressive_p2": "AdaptiveFCN_All_4x1_DIC_ZNSSD_backbone_first_p2_residual",
}
OLD_RUN = {method: f"AdaptiveFCN_All_4x1_DIC_ZNSSD_{method}" for method in CLASSICAL}


def _checkpoint(method, folder, step=500):
    path = PHASE_ROOT / folder / "models" / RUN[method] / f"model_{step:08d}.jax"
    with path.open("rb") as stream:
        saved_step, params, _state, _active, loss = pickle.load(stream)
    if int(saved_step) != step:
        raise ValueError(f"{path} has step {saved_step}, expected {step}")
    return params, float(np.asarray(loss)), path


def _runtime(method, folder):
    path = PHASE_ROOT / folder / "summaries" / RUN[method] / "run_metrics_roi0_pair1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _component_fields(params, points, assembly, batch_size=8192):
    """AD global and gated residual fields using the exact trainer plan."""
    result = {name: {"value": [], "gradient": []} for name in ("global", "residual")}
    fns = _model_fns(assembly)
    m = int(params["static"]["decomposition"]["m"])
    for begin in range(0, len(points), batch_size):
        query = jnp.asarray(points[begin:begin + batch_size])
        plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
            query, jnp.zeros(m, dtype=int), params,
            RectangularDecompositionND, assembly)
        cut = {"static": cut_all(params["static"]),
               "trainable": cut_active(params["trainable"])}

        def components(x):
            values = assembly.components(cut, x, plan, fns[:4])
            return values[0], values[1]

        tangent_x = jnp.tile(jnp.array([[1.0, 0.0]]), (len(query), 1))
        tangent_y = jnp.tile(jnp.array([[0.0, 1.0]]), (len(query), 1))
        values, dx = jax.jvp(components, (query,), (tangent_x,))
        _, dy = jax.jvp(components, (query,), (tangent_y,))
        for index, name in enumerate(("global", "residual")):
            result[name]["value"].append(np.asarray(values[index]))
            result[name]["gradient"].append(np.stack(
                [np.asarray(dx[index]), np.asarray(dy[index])], axis=-1))
    for name in result:
        result[name]["value"] = np.concatenate(result[name]["value"])
        result[name]["gradient"] = np.concatenate(result[name]["gradient"])
        result[name]["strain"] = np.asarray(strain_from_gradient(
            jnp.asarray(result[name]["gradient"])))
    return result


def _component_metrics(total, components):
    global_value, residual = components["global"]["value"], components["residual"]["value"]
    global_strain, residual_strain = components["global"]["strain"], components["residual"]["strain"]
    return {
        "global_displacement_rms": rms(global_value),
        "residual_displacement_rms": rms(residual),
        "residual_to_total_displacement_rms_ratio": rms(residual) / rms(total["value"]),
        "residual_to_global_displacement_rms_ratio": rms(residual) / rms(global_value),
        "global_strain_rms": rms(global_strain),
        "residual_strain_rms": rms(residual_strain),
        "residual_to_total_strain_rms_ratio": rms(residual_strain) / rms(total["strain"]),
        "residual_to_global_strain_rms_ratio": rms(residual_strain) / rms(global_strain),
    }


def _evaluate_backbone(method, assembly, folder):
    params, reported_loss, checkpoint = _checkpoint(method, folder)
    mask, points = _roi(params)
    fields = _assembled_batched(params, points, assembly)
    mapped = _map_fields(mask, fields)
    local = _local_interface_metrics(params, points, zero_offset=True)
    components = _component_fields(params, points, assembly)
    item = {
        "adam_budget": 500,
        "checkpoint_reported_loss": reported_loss,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "local_strain_mismatch_rms": rms([x["local_strain_mismatch_rms"] for x in local]),
        "local_gradient_mismatch_rms": rms([x["local_gradient_mismatch_rms"] for x in local]),
        "local_displacement_mismatch_rms": rms([x["local_displacement_mismatch_rms"] for x in local]),
        "local_interfaces": local,
        **_field_metrics(mapped, mask, geometry(mask)),
        "global_residual": _component_metrics(fields, components),
    }
    return item, mapped, components, params, fields


def _classical_reference(old):
    result, fields = {}, {}
    for method in CLASSICAL:
        item = old["methods"][method]["budgets"]["500"].copy()
        item["source"] = "original_raw_500_step_checkpoint"
        item["training"] = old["methods"][method]["training"]
        item["local_gradient_mismatch_rms"] = rms([
            entry["local_gradient_mismatch_rms"]
            for entry in item["local_interfaces"]])
        result[method] = item
        cache = OLD_ANALYSIS / f"{method}_adam500_fields.npz"
        if cache.exists():
            fields[method] = dict(np.load(cache))
            continue
        checkpoint = (ROOT / "case" / "bbf" / "1" / "results" /
                      "strain_interface_phase" / "fair_sweep" / "models" /
                      OLD_RUN[method] / "model_00000500.jax")
        with checkpoint.open("rb") as stream:
            _step, params, _state, _active, _loss = pickle.load(stream)
        mask, points = _roi(params)
        fields[method] = _map_fields(
            mask, _assembled_batched(params, points, CLASSICAL[method]))
    return result, fields


def _save_field(path_stem, mask, mapped):
    np.savez_compressed(OUT / f"{path_stem}.npz", **mapped)
    savemat(OUT / f"{path_stem}.mat", mapped)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=170, constrained_layout=True)
    for ax, name in zip(axes, ("exx", "exy", "eyy")):
        values = mapped[name][mask]
        limit = np.percentile(np.abs(values), 99.5)
        image = ax.imshow(np.where(mask, mapped[name], np.nan), cmap="coolwarm",
                          vmin=-limit, vmax=limit)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(OUT / f"{path_stem}.png", bbox_inches="tight")
    plt.close(fig)


def _comparison_figure(fields, mask, geom):
    methods = list(fields)
    fig, axes = plt.subplots(len(methods), 3, figsize=(14, 2.5*len(methods)), dpi=170,
                             constrained_layout=True)
    for col, name in enumerate(("exx", "exy", "eyy")):
        values = np.concatenate([fields[key][name][mask] for key in methods])
        low, high = np.percentile(values, (1, 99))
        for row, method in enumerate(methods):
            ax = axes[row, col]
            image = ax.imshow(np.where(mask, fields[method][name], np.nan), cmap="turbo",
                              vmin=low, vmax=high)
            for x in geom["interfaces"]:
                ax.axvline(x, color="white", linestyle="--", linewidth=.5)
            ax.set_title(f"{LABELS[method]} — {name}", fontsize=8)
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=.03, pad=.01)
    fig.savefig(OUT / "formal_500_strain_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _stage_evolution(method, assembly, folder):
    stage_metrics = {}
    for step in (0, 150, 300, 500):
        params, loss, _ = _checkpoint(method, folder, step)
        mask, points = _roi(params)
        total = _assembled_batched(params, points, assembly)
        components = _component_fields(params, points, assembly)
        gate = float(np.asarray(params["static"]["assembly"]["residual_gate"]))
        stage_metrics[str(step)] = {
            "gate": gate,
            "checkpoint_reported_loss": loss,
            **_component_metrics(total, components),
        }
        total_mapped = _map_fields(mask, total)
        global_mapped = _map_fields(mask, components["global"])
        residual_mapped = _map_fields(mask, components["residual"])
        component_arrays = {
            f"{prefix}_{name}": values
            for prefix, mapped in (("global", global_mapped),
                                   ("residual", residual_mapped),
                                   ("total", total_mapped))
            for name, values in mapped.items()
        }
        np.savez_compressed(OUT / f"{method}_stage_{step}_components.npz",
                            **component_arrays)
        savemat(OUT / f"{method}_stage_{step}_components.mat", component_arrays)
        if step in (0, 150, 300, 500):
            _save_field(f"{method}_stage_{step}", mask, total_mapped)
    return stage_metrics


def _interface_trace_figure(mask, fields, components, geom):
    """Three interface-normal y traces for the selected P2 decomposition."""
    y = np.arange(mask.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), dpi=170,
                             constrained_layout=True, sharex=True)
    bbox_x0 = geom["roi_bbox"][0]
    for row, interface in enumerate(geom["interfaces"]):
        x_index = int(round(interface))
        x_index = min(max(x_index, 0), mask.shape[1] - 1)
        valid = mask[:, x_index]
        if not np.any(valid):
            x_index = int(round(interface - bbox_x0))
            valid = mask[:, x_index]
        for col, name in enumerate(("exx", "exy", "eyy")):
            ax = axes[row, col]
            for label, mapped, color in (
                    ("global", components["global"], "#1f77b4"),
                    ("residual", components["residual"], "#d62728"),
                    ("total", fields, "#222222")):
                ax.plot(y[valid], mapped[name][valid, x_index], label=label,
                        color=color, linewidth=1.0)
            ax.set_title(f"interface {interface:g}: {name}", fontsize=9)
            ax.grid(alpha=.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.savefig(OUT / "backbone_progressive_p2_interface_y_traces.png",
                bbox_inches="tight")
    plt.close(fig)


def _diagnostic(old):
    cosine = old["methods"]["cosine_pou"]["budgets"]
    steps = np.array([500, 1000, 2000, 3000])
    loss = np.array([cosine[str(step)]["dic_loss_recomputed"] for step in steps])
    local = np.array([cosine[str(step)]["local_strain_mismatch_rms"] for step in steps])
    rough = np.array([cosine[str(step)]["interface_strain_roughness"] for step in steps])
    payload = {
        "diagnostic_only": True,
        "steps": steps.tolist(), "loss": loss.tolist(),
        "local_strain_mismatch_rms": local.tolist(),
        "interface_strain_roughness": rough.tolist(),
        "non_loss_indicator": "local_strain_mismatch_rms / value_at_500",
        "indicator_values": (local / local[0]).tolist(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), dpi=170, constrained_layout=True)
    for ax, values, title, ylabel in zip(
            axes, (loss, local, rough),
            ("Photometric loss", "Local strain mismatch", "Interface roughness"),
            ("ZNSSD", "RMS", "roughness")):
        ax.plot(steps, values, "o-")
        ax.set_title(title); ax.set_xlabel("Adam step (diagnostic only)"); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    fig.savefig(OUT / "overfitting_diagnostic.png", bbox_inches="tight")
    fig.savefig(DIAGNOSTIC_OUT / "overfitting_diagnostic.png", bbox_inches="tight")
    plt.close(fig)
    return payload


def _write_csv(results):
    rows = []
    for method, item in results.items():
        row = {"method": method, "label": LABELS[method],
               "reported_loss": item.get("checkpoint_reported_loss"),
               "field_loss": item.get("dic_loss_recomputed"),
               "interface_roughness": item["interface_strain_roughness"],
               "local_strain_mismatch": item["local_strain_mismatch_rms"],
               "local_gradient_mismatch": item["local_gradient_mismatch_rms"],
               "rms_strain_jump": item["rms_strain_jump"]}
        if "global_residual" in item:
            row.update(item["global_residual"])
        rows.append(row)
    with (OUT / "formal_500_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REFERENCE_OUT.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    reference = {
        method: old["methods"][method]["budgets"]["500"]
        for method in ("cosine_pou", "quintic_pou", "hermite_1d_c1",
                       "c1_minimal_correction_pou")
    }
    (REFERENCE_OUT / "reference_500_metrics.json").write_text(
        json.dumps(reference, indent=2, allow_nan=True), encoding="utf-8")
    results, fields = _classical_reference(old)
    stage, components_by_method = {}, {}
    for method, (assembly, folder) in BACKBONES.items():
        print(f"Evaluating {method}", flush=True)
        item, mapped, components, params, _total = _evaluate_backbone(method, assembly, folder)
        item["training"] = _runtime(method, folder)
        results[method] = item; fields[method] = mapped
        components_by_method[method] = {
            name: _map_fields(np.asarray(params["static"]["problem"]["mask"], dtype=bool), value)
            for name, value in components.items()
        }
        _save_field(f"{method}_adam500_fields", np.asarray(params["static"]["problem"]["mask"], dtype=bool), mapped)
    for method, assembly, folder in (
            ("backbone_progressive_cosine", BackboneFirstCosineResidualAssembly, "backbone_first_cosine"),
            ("backbone_progressive_p2", BackboneFirstP2ResidualAssembly, "backbone_first_p2")):
        print(f"Extracting stage snapshots for {method}", flush=True)
        stage[method] = _stage_evolution(method, assembly, folder)
    reference_params, _loss, _path = _checkpoint(
        "backbone_progressive_cosine", "backbone_first_cosine")
    mask = np.asarray(reference_params["static"]["problem"]["mask"], dtype=bool)
    _comparison_figure(fields, mask, geometry(mask))
    _interface_trace_figure(mask, fields["backbone_progressive_p2"],
                            components_by_method["backbone_progressive_p2"],
                            geometry(mask))
    diagnostic = _diagnostic(old)
    (DIAGNOSTIC_OUT / "overfitting_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8")
    _write_csv(results)
    payload = {
        "formal_budget": {"adam_steps": 500, "seed_train": 0, "lbfgs_steps": 0},
        "ground_truth_available": False,
        "strain_definition": "Automatic differentiation of final assembled displacement; no strain loss or smoothing postprocess.",
        "classical_reference_source": "Original raw 500-step checkpoints from strain_interface_phase; no retraining.",
        "methods": results,
        "stage_evolution": stage,
        "overfitting_diagnostic": diagnostic,
    }
    (OUT / "backbone_p2_metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({name: {"loss": value.get("checkpoint_reported_loss", value.get("dic_loss_recomputed")),
                             "roughness": value["interface_strain_roughness"]}
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
