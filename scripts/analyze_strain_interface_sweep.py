"""Evaluate Phase-3 Adam checkpoints and generate real-case evidence.

All strains are obtained by automatic differentiation of the checkpoint's
final assembled displacement.  No finite-difference strain enters the reported
field metrics; finite differences are used only to quantify field roughness.
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
from matplotlib.lines import Line2D
import numpy as np
from scipy.io import savemat
from tensorboard.backend.event_processing import event_accumulator

from scripts.analyze_assembly_results import (
    band_field_rms,
    finite_difference_roughness,
    geometry,
    interface_traces,
    rms,
)
from segpinndic.DIC_assemblies import (
    C1MinimalCorrectionPOUAssembly,
    CosinePOUAssembly,
    GlobalBackboneLocalResidualAssembly,
    Hermite1DC1Assembly,
    QuinticPOUAssembly,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_field_diagnostics import (
    adjacent_overlap_points,
    evaluate_assembled_fields,
    evaluate_local_fields,
    strain_from_gradient,
)
from segpinndic.DIC_networks import AdaptiveFCN
from segpinndic.DIC_problem import DIC_ZNSSD
from segpinndic.DIC_trainers import get_inputs


ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = ROOT / "case" / "bbf" / "1" / "results" / "strain_interface_phase"
PHASE = PHASE_ROOT / "fair_sweep"
OUT = PHASE_ROOT / "analysis"
BUDGETS = (500, 1000, 2000, 3000)
METHODS = {
    "cosine_pou": CosinePOUAssembly,
    "quintic_pou": QuinticPOUAssembly,
    "hermite_1d_c1": Hermite1DC1Assembly,
    "c1_minimal_correction_pou": C1MinimalCorrectionPOUAssembly,
    "global_backbone_local_residual": GlobalBackboneLocalResidualAssembly,
}
LABELS = {
    "cosine_pou": "Cosine POU",
    "quintic_pou": "Quintic POU",
    "hermite_1d_c1": "Hermite 1D C1",
    "c1_minimal_correction_pou": "C1 minimal correction",
    "global_backbone_local_residual": "Global + local residual",
}
RUN = {
    method: f"AdaptiveFCN_All_4x1_DIC_ZNSSD_{method}"
    for method in METHODS
}


def _model_fns(assembly):
    return (
        RectangularDecompositionND.norm_fn,
        AdaptiveFCN.network_fn,
        RectangularDecompositionND.unnorm_fn,
        RectangularDecompositionND.window_fn,
        assembly,
    )


def _checkpoint(method, budget):
    path = PHASE / "models" / RUN[method] / f"model_{budget:08d}.jax"
    with path.open("rb") as stream:
        item = pickle.load(stream)
    step, all_params, _opt_state, _active, checkpoint_loss = item
    if int(step) != budget:
        raise ValueError(f"{path} contains step {step}, expected {budget}")
    return all_params, float(np.asarray(checkpoint_loss)), path


def _roi(all_params):
    mask = np.asarray(all_params["static"]["problem"]["mask"], dtype=bool)
    ys, xs = np.where(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    return mask, points


def _assembled_batched(all_params, points, assembly, batch_size=8192):
    chunks = {"value": [], "gradient": [], "strain": []}
    fns = _model_fns(assembly)
    for begin in range(0, len(points), batch_size):
        current = points[begin:begin+batch_size]
        fields = evaluate_assembled_fields(
            all_params, current, RectangularDecompositionND, assembly, fns)
        for key in chunks:
            chunks[key].append(np.asarray(fields[key]))
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def _local_batched(
        all_params, points, model_indices, zero_offset=False, batch_size=8192):
    chunks = {"value": [], "gradient": [], "strain": []}
    local_fns = _model_fns(CosinePOUAssembly)[:4]
    for begin in range(0, len(points), batch_size):
        fields = evaluate_local_fields(
            all_params,
            points[begin:begin+batch_size],
            model_indices[begin:begin+batch_size],
            local_fns,
            zero_offset=zero_offset,
        )
        for key in chunks:
            chunks[key].append(np.asarray(fields[key]))
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def _local_interface_metrics(all_params, points, zero_offset=False):
    records = []
    for overlap in adjacent_overlap_points(all_params, points):
        sample = overlap["points"].astype(np.float32)
        left_ids = np.full(len(sample), overlap["left_model"], dtype=np.int32)
        right_ids = np.full(len(sample), overlap["right_model"], dtype=np.int32)
        left = _local_batched(
            all_params, sample, left_ids, zero_offset=zero_offset)
        right = _local_batched(
            all_params, sample, right_ids, zero_offset=zero_offset)
        delta_u = left["value"]-right["value"]
        delta_g = left["gradient"]-right["gradient"]
        delta_e = left["strain"]-right["strain"]
        records.append({
            "interface_x": overlap["x_interface"],
            "overlap_x": [overlap["x_left"], overlap["x_right"]],
            "point_count": int(len(sample)),
            "local_displacement_mismatch_rms": rms(delta_u),
            "local_gradient_mismatch_rms": rms(delta_g),
            "local_strain_mismatch_rms": rms(delta_e),
            "local_strain_mismatch_components_rms": {
                name: rms(delta_e[:, index])
                for index, name in enumerate(("exx", "exy", "eyy"))
            },
            "local_strain_mismatch_max": float(np.max(np.abs(delta_e))),
        })
    return records


def _map_fields(mask, fields):
    mapped = {}
    for name, values in {
        "u": fields["value"][:, 0],
        "v": fields["value"][:, 1],
        "exx": fields["strain"][:, 0],
        "exy": fields["strain"][:, 1],
        "eyy": fields["strain"][:, 2],
    }.items():
        image = np.zeros(mask.shape, dtype=np.float32)
        image[mask] = values
        mapped[name] = image
    return mapped


def _time_at_budget(runtime, budget):
    history = {int(step): float(seconds)
               for step, seconds in runtime["adam_time_history"]}
    if budget not in history:
        nearest = min(history, key=lambda step: abs(step-budget))
        raise KeyError(f"missing exact time for {budget}; nearest is {nearest}")
    return history[budget]


def _load_runtime(method):
    """Load normal run metrics, or recover an interrupted run from events."""

    runtime_path = PHASE / "summaries" / RUN[method] / "run_metrics_roi0_pair1.json"
    if runtime_path.exists():
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        payload["runtime_source"] = "run_metrics"
        return payload

    summary_dir = runtime_path.parent
    accumulator = event_accumulator.EventAccumulator(str(summary_dir))
    accumulator.Reload()
    scalars = accumulator.Scalars("loss/train")
    rates = {item.step: item.value for item in accumulator.Scalars("stats/rate")}
    if not scalars:
        raise FileNotFoundError(
            f"Neither run metrics nor loss/train TensorBoard events exist for {method}")
    first = scalars[0]
    initial_step_seconds = first.step/rates.get(first.step, 1.0)
    start_time = first.wall_time-initial_step_seconds
    return {
        "assembly_method": method,
        "adam_steps": int(scalars[-1].step),
        "lbfgs_steps": 0,
        "final_loss": float(scalars[-1].value),
        "total_seconds": float(scalars[-1].wall_time-start_time),
        "initial_local_parameter_sha256": (
            "dd612e86e2c9d893d271f29333466220a1cc68e53520023948be90f23159ef50"),
        "local_trainable_parameter_count": 13840,
        "extra_trainable_parameter_count": 0,
        "total_trainable_parameter_count": 13840,
        "adam_loss_history": [[int(item.step), float(item.value)] for item in scalars],
        "adam_time_history": [
            [int(item.step), float(item.wall_time-start_time)] for item in scalars],
        "lbfgs_loss_history": [],
        "runtime_source": "tensorboard_event_reconstruction",
    }


def _correction_metrics(all_params, points, final_fields):
    base = _assembled_batched(all_params, points, CosinePOUAssembly)
    correction = final_fields["value"]-base["value"]
    correction_gradient = final_fields["gradient"]-base["gradient"]
    correction_strain = strain_from_gradient(jnp.asarray(correction_gradient))
    correction_magnitude = np.linalg.norm(correction, axis=1)
    return {
        "correction_displacement_rms": rms(correction),
        "correction_u_rms": rms(correction[:, 0]),
        "correction_v_rms": rms(correction[:, 1]),
        "correction_displacement_max": float(np.max(np.abs(correction))),
        "correction_vector_max": float(np.max(correction_magnitude)),
        "correction_active_point_fraction": float(np.mean(
            correction_magnitude > 1.0e-10)),
        "correction_to_final_displacement_rms_ratio": (
            rms(correction)/rms(final_fields["value"])),
        "correction_strain_rms": rms(correction_strain),
        "correction_strain_max": float(np.max(np.abs(correction_strain))),
    }, correction, np.asarray(correction_strain)


def _global_components_batched(all_params, points, batch_size=8192):
    values = {"global": [], "residual": []}
    gradients = {"global": [], "residual": []}
    local_fns = _model_fns(GlobalBackboneLocalResidualAssembly)[:4]
    m = int(all_params["static"]["decomposition"]["m"])
    for begin in range(0, len(points), batch_size):
        current = jnp.asarray(points[begin:begin+batch_size])
        plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
            current, jnp.zeros(m, dtype=int), all_params,
            RectangularDecompositionND, GlobalBackboneLocalResidualAssembly)
        cut = {
            "static": cut_all(all_params["static"]),
            "trainable": cut_active(all_params["trainable"]),
        }

        def components(query):
            return GlobalBackboneLocalResidualAssembly.components(
                cut, query, plan, local_fns)[:2]

        tangent_x = jnp.tile(jnp.array([[1.0, 0.0]]), (len(current), 1))
        tangent_y = jnp.tile(jnp.array([[0.0, 1.0]]), (len(current), 1))
        component_values, dx = jax.jvp(components, (current,), (tangent_x,))
        _, dy = jax.jvp(components, (current,), (tangent_y,))
        for index, name in enumerate(("global", "residual")):
            values[name].append(np.asarray(component_values[index]))
            gradients[name].append(np.stack([
                np.asarray(dx[index]), np.asarray(dy[index])], axis=-1))
    output = {}
    for name in values:
        value = np.concatenate(values[name], axis=0)
        gradient = np.concatenate(gradients[name], axis=0)
        output[name] = {
            "value": value,
            "gradient": gradient,
            "strain": np.asarray(strain_from_gradient(jnp.asarray(gradient))),
        }
    return output


def _field_metrics(mapped, mask, geom):
    strains = [mapped[name] for name in ("exx", "exy", "eyy")]
    roughness = finite_difference_roughness(strains, mask, geom["interfaces"])
    traces = interface_traces(mapped, mask, geom["interfaces"])
    return {
        "global_strain_roughness": roughness["global"],
        "interface_strain_roughness": roughness["interface_band"],
        "interior_strain_roughness": roughness["interior"],
        "subdomain_periodic_artifact_ratio": roughness["periodic_artifact_ratio"],
        "strain_field_rms_by_band": band_field_rms(
            strains, mask, geom["interfaces"]),
        "rms_strain_jump": rms([item["J_strain_rms"] for item in traces]),
        "max_strain_jump": max(item["max_abs_strain_jump"] for item in traces),
        "interface_traces": traces,
    }


def evaluate():
    all_results = {}
    fields_final = {}
    correction_map = None
    correction_strain_map = None
    global_components = None
    common_mask = None
    common_geom = None

    for method, assembly in METHODS.items():
        runtime = _load_runtime(method)
        method_results = {}
        for budget in BUDGETS:
            print(f"Evaluating {method} at Adam {budget}...", flush=True)
            all_params, checkpoint_loss, checkpoint_path = _checkpoint(method, budget)
            mask, points = _roi(all_params)
            geom = geometry(mask)
            common_mask, common_geom = mask, geom
            fields = _assembled_batched(all_params, points, assembly)
            mapped = _map_fields(mask, fields)
            dic_loss = float(DIC_ZNSSD.loss_fn(
                all_params, jnp.asarray(points), jnp.asarray(fields["value"])))
            local_records = _local_interface_metrics(
                all_params, points,
                zero_offset=(method == "global_backbone_local_residual"))
            item = {
                "adam_budget": budget,
                "dic_loss_recomputed": dic_loss,
                "checkpoint_reported_loss": checkpoint_loss,
                "adam_elapsed_seconds": _time_at_budget(runtime, budget),
                "local_strain_mismatch_rms": rms([
                    record["local_strain_mismatch_rms"]
                    for record in local_records]),
                "local_displacement_mismatch_rms": rms([
                    record["local_displacement_mismatch_rms"]
                    for record in local_records]),
                "local_interfaces": local_records,
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                **_field_metrics(mapped, mask, geom),
            }
            local_mismatch = item["local_strain_mismatch_rms"]
            item["local_to_assembled_amplification_ratio"] = (
                item["rms_strain_jump"]/local_mismatch
                if local_mismatch > 0.0 else float("nan"))
            if method == "c1_minimal_correction_pou":
                correction_metrics, correction, correction_strain = (
                    _correction_metrics(all_params, points, fields))
                item["minimal_correction"] = correction_metrics
                if budget == BUDGETS[-1]:
                    correction_map = correction
                    correction_strain_map = correction_strain
            if method == "global_backbone_local_residual":
                components = _global_components_batched(all_params, points)
                global_value = components["global"]["value"]
                residual_value = components["residual"]["value"]
                global_strain = components["global"]["strain"]
                residual_strain = components["residual"]["strain"]
                item["global_residual"] = {
                    "global_displacement_rms": rms(global_value),
                    "global_u_rms": rms(global_value[:, 0]),
                    "global_v_rms": rms(global_value[:, 1]),
                    "residual_displacement_rms": rms(residual_value),
                    "residual_u_rms": rms(residual_value[:, 0]),
                    "residual_v_rms": rms(residual_value[:, 1]),
                    "residual_displacement_max": float(
                        np.max(np.linalg.norm(residual_value, axis=1))),
                    "residual_to_total_displacement_rms_ratio": (
                        rms(residual_value)/rms(fields["value"])),
                    "residual_to_global_displacement_rms_ratio": (
                        rms(residual_value)/rms(global_value)),
                    "residual_to_total_u_rms_ratio": (
                        rms(residual_value[:, 0])/rms(fields["value"][:, 0])),
                    "residual_to_total_v_rms_ratio": (
                        rms(residual_value[:, 1])/rms(fields["value"][:, 1])),
                    "global_strain_rms": rms(global_strain),
                    "residual_strain_rms": rms(residual_strain),
                    "residual_strain_max": float(np.max(np.abs(residual_strain))),
                    "residual_to_total_strain_rms_ratio": (
                        rms(residual_strain)/rms(fields["strain"])),
                    "residual_to_global_strain_rms_ratio": (
                        rms(residual_strain)/rms(global_strain)),
                }
                if budget == BUDGETS[-1]:
                    global_components = components
            method_results[str(budget)] = item
            if budget == BUDGETS[-1]:
                fields_final[method] = mapped
                np.savez_compressed(
                    OUT / f"{method}_adam{budget}_fields.npz", **mapped)
                savemat(OUT / f"{method}_adam{budget}_fields.mat", mapped)
        all_results[method] = {
            "training": runtime,
            "budgets": method_results,
        }
    return (all_results, fields_final, common_mask, common_geom,
            correction_map, correction_strain_map, global_components)


def _comparison_figure(fields, mask, geom):
    strain_names = ("exx", "exy", "eyy")
    limits = {}
    for name in strain_names:
        values = np.concatenate([item[name][mask] for item in fields.values()])
        limits[name] = tuple(np.percentile(values, (0.5, 99.5)))
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(15, 15), dpi=180,
                             constrained_layout=True)
    for row, method in enumerate(METHODS):
        for col, name in enumerate(strain_names):
            ax = axes[row, col]
            image = ax.imshow(np.where(mask, fields[method][name], np.nan),
                              cmap="turbo", vmin=limits[name][0],
                              vmax=limits[name][1])
            for x in geom["interfaces"]:
                ax.axvline(x, color="white", linestyle="--", linewidth=0.6)
            ax.set_title(f"{LABELS[method]} — {name}")
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    axes[0, 0].legend(
        handles=[Line2D([0], [0], color="white", linestyle="--",
                        linewidth=0.8, label="internal interface")],
        loc="lower right", fontsize=7, framealpha=0.7)
    fig.savefig(OUT / "assembly_method_strain_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _loss_figures(results):
    for scale, filename in (("linear", "loss_convergence_linear.png"),
                            ("log", "loss_convergence_log.png")):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
        for method in METHODS:
            history = np.asarray(
                results[method]["training"]["adam_loss_history"], dtype=float)
            ax.plot(history[:, 0], history[:, 1], label=LABELS[method])
        if scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("ZNSSD loss")
        ax.set_title(f"Photometric convergence ({scale} scale)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(OUT / filename, bbox_inches="tight")
        plt.close(fig)


def _tradeoff_figures(results):
    methods = list(METHODS)
    budget_markers = dict(zip(BUDGETS, ("o", "s", "^", "D")))
    for metric, xlabel, filenames in (
        ("dic_loss_recomputed", "Recomputed ZNSSD loss",
         ("photometric_vs_interface_tradeoff.png",
          "tradeoff_loss_vs_interface_roughness.png")),
        ("adam_elapsed_seconds", "Adam wall time [s]",
         ("runtime_vs_interface_tradeoff.png",
          "tradeoff_runtime_vs_interface_roughness.png")),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
        for method in methods:
            items = [results[method]["budgets"][str(b)] for b in BUDGETS]
            x = np.asarray([item[metric] for item in items])
            roughness = np.asarray([
                item["interface_strain_roughness"] for item in items])
            line, = ax.plot(x, roughness, linewidth=1.1, alpha=0.8,
                            label=LABELS[method])
            for budget, x_value, y_value in zip(BUDGETS, x, roughness):
                ax.scatter(x_value, y_value, marker=budget_markers[budget],
                           color=line.get_color(), s=45)
            ax.annotate(LABELS[method], (x[-1], roughness[-1]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Interface-band strain roughness")
        ax.grid(alpha=0.25)
        method_legend = ax.legend(loc="upper right", fontsize=7,
                                  title="method")
        ax.add_artist(method_legend)
        marker_handles = [Line2D([0], [0], marker=budget_markers[budget],
                                 color="gray", linestyle="none",
                                 label=str(budget)) for budget in BUDGETS]
        ax.legend(handles=marker_handles, loc="center right", fontsize=7,
                  title="Adam steps")
        for filename in filenames:
            fig.savefig(OUT / filename, bbox_inches="tight")
        plt.close(fig)


def _correction_figure(mask, correction, correction_strain):
    if correction is None:
        return
    components = [correction[:, 0], correction[:, 1],
                  correction_strain[:, 0], correction_strain[:, 1],
                  correction_strain[:, 2]]
    names = ("c_u", "c_v", "c_exx", "c_exy", "c_eyy")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=180,
                             constrained_layout=True)
    for ax, values, name in zip(axes, components, names):
        image_data = np.full(mask.shape, np.nan)
        image_data[mask] = values
        limit = np.percentile(np.abs(values), 99.5)
        image = ax.imshow(image_data, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(OUT / "minimal_correction_field.png", bbox_inches="tight")
    plt.close(fig)


def _global_figure(mask, components):
    if components is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180,
                             constrained_layout=True)
    for row, name in enumerate(("global", "residual")):
        for col, strain_name in enumerate(("exx", "exy", "eyy")):
            values = components[name]["strain"][:, col]
            image_data = np.full(mask.shape, np.nan)
            image_data[mask] = values
            limit = np.percentile(np.abs(values), 99.5)
            image = axes[row, col].imshow(
                image_data, cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[row, col].set_title(f"{name} — {strain_name}")
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.035, pad=0.02)
    fig.savefig(OUT / "global_backbone_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def _write_csv(results):
    rows = []
    for method in METHODS:
        training = results[method]["training"]
        for budget in BUDGETS:
            item = results[method]["budgets"][str(budget)]
            rows.append({
                "method": method,
                "adam_budget": budget,
                "dic_loss": item["dic_loss_recomputed"],
                "adam_seconds": item["adam_elapsed_seconds"],
                "interface_strain_roughness": item["interface_strain_roughness"],
                "interior_strain_roughness": item["interior_strain_roughness"],
                "artifact_ratio": item["subdomain_periodic_artifact_ratio"],
                "local_strain_mismatch_rms": item["local_strain_mismatch_rms"],
                "rms_strain_jump": item["rms_strain_jump"],
                "max_strain_jump": item["max_strain_jump"],
                "local_to_assembled_amplification_ratio": (
                    item["local_to_assembled_amplification_ratio"]),
                "local_parameter_count": training["local_trainable_parameter_count"],
                "extra_parameter_count": training["extra_trainable_parameter_count"],
                "initial_local_sha256": training["initial_local_parameter_sha256"],
            })
    with (OUT / "convergence_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (results, fields, mask, geom, correction, correction_strain,
     global_components) = evaluate()
    payload = {
        "ground_truth_available": False,
        "strain_definition": (
            "Automatic differentiation of each Adam checkpoint's final "
            "assembled displacement; no strain loss or smoothing postprocess."),
        "budget_definition": (
            "Checkpoints from one continuous, identically initialized 3000-step "
            "Adam trajectory per method; final L-BFGS is excluded from budgets."),
        "local_to_assembled_amplification_definition": (
            "RMS strain-jump indicator of the final autodiff strain divided by "
            "the pooled RMS adjacent-local-network strain mismatch evaluated "
            "at identical physical points. Values above one indicate that the "
            "assembled interface indicator exceeds the raw local mismatch; "
            "values below one indicate suppression by this diagnostic."),
        "geometry": {key: np.asarray(value).tolist() for key, value in geom.items()},
        "methods": results,
    }
    (OUT / "strain_interface_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    _write_csv(results)
    _comparison_figure(fields, mask, geom)
    _loss_figures(results)
    _tradeoff_figures(results)
    _correction_figure(mask, correction, correction_strain)
    _global_figure(mask, global_components)
    print(json.dumps({
        method: results[method]["budgets"][str(BUDGETS[-1])]
        for method in METHODS
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
