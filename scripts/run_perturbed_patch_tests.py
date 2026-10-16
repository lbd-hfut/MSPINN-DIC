"""Run deterministic perturbed P1/P2 patch tests for Phase 3 assemblies."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import polynomial_reproducing_reconstruction
from tests.test_perturbed_patch_tests import (
    METHODS,
    _base_strain,
    _base_value,
    perturbation,
    perturbed_patch_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "case" / "bbf" / "1" / "results" / "p2_bspline_scale_phase" / "patch_tests"


def reproducing_metrics(coefficients, order, amplitude=0.0, wavelength=11.0):
    """Value/strain error for the non-trainable P1/P2 reconstruction prototype."""

    # Use a grid and 19/11/7 px perturbation waves without a shared discrete
    # phase family; commensurate spacing can alias a wave into a constant.
    xx, yy = np.meshgrid(np.linspace(0.0, 40.0, 7),
                         np.linspace(0.0, 20.0, 4), indexing="xy")
    sample_points = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    qx, qy = np.meshgrid(np.linspace(0.0, 40.0, 41),
                         np.linspace(0.0, 20.0, 21), indexing="xy")
    query_points = jnp.asarray(np.stack([qx.ravel(), qy.ravel()], axis=1))
    sample_values = _base_value(coefficients, sample_points)
    phase = jnp.array([0.0, 0.7])
    sample_values = sample_values + amplitude*jnp.sin(
        (2*jnp.pi/wavelength)*sample_points[:, :1]+phase[None, :])

    def reconstruct(point):
        return polynomial_reproducing_reconstruction(
            sample_points, sample_values, point[None, :], order=order)[0][0]

    values = jax.vmap(reconstruct)(query_points)
    gradients = jax.vmap(jax.jacfwd(reconstruct))(query_points)
    strains = jnp.stack([
        gradients[:, 0, 0],
        0.5*(gradients[:, 0, 1]+gradients[:, 1, 0]),
        gradients[:, 1, 1],
    ], axis=1)
    value_error = values-_base_value(coefficients, query_points)
    strain_error = strains-_base_strain(coefficients, query_points)
    return {
        "order": order,
        "amplitude": amplitude,
        "wavelength": wavelength,
        "Disp_RMS": float(jnp.sqrt(jnp.mean(value_error**2))),
        "Disp_Max": float(jnp.max(jnp.abs(value_error))),
        "Strain_RMS": float(jnp.sqrt(jnp.mean(strain_error**2))),
        "Strain_Max": float(jnp.max(jnp.abs(strain_error))),
    }


def reproducing_report(p1, p2):
    report = {}
    for name, coefficients, order in (("P1", p1, 1), ("P2", p2, 2)):
        report[f"{name}_exact"] = reproducing_metrics(coefficients, order)
        for amplitude in (0.001, 0.005, 0.01):
            report[f"{name}_A{amplitude:g}_L11"] = reproducing_metrics(
                coefficients, order, amplitude, 11.0)
        for wavelength in (19.0, 11.0, 7.0):
            report[f"{name}_A0.005_L{wavelength:g}"] = reproducing_metrics(
                coefficients, order, 0.005, wavelength)
    return report


def main():
    p1 = np.array([
        [0.3, 0.02, -0.015, 0.0, 0.0, 0.0],
        [-0.4, 0.01, 0.025, 0.0, 0.0, 0.0],
    ])
    p2 = np.array([
        [0.2, 0.01, -0.008, 2e-4, -1.5e-4, 1e-4],
        [-0.1, 0.006, 0.012, -1e-4, 2.5e-4, -1.2e-4],
    ])
    scenarios = {}
    for level in (0.001, 0.005, 0.01):
        scenarios[f"P1_constant_{level:g}"] = (p1, perturbation("constant", level))
    for level in (1e-4, 5e-4, 1e-3):
        scenarios[f"P1_linear_{level:g}"] = (p1, perturbation("linear", level))
    for amplitude in (0.001, 0.005, 0.01):
        for wavelength in (20.0, 10.0, 5.0):
            scenarios[f"P1_highfreq_A{amplitude:g}_L{wavelength:g}"] = (
                p1, perturbation("high_frequency", amplitude, 2*np.pi/wavelength))
    scenarios["P2_highfreq_A0.005_L10"] = (
        p2, perturbation("high_frequency", 0.005, 2*np.pi/10.0))

    report = {}
    for scenario, (base, perturb) in scenarios.items():
        print(f"Running {scenario}...", flush=True)
        report[scenario] = {
            method: perturbed_patch_metrics(base, assembly, perturb)
            for method, assembly in METHODS.items()
        }
    prototype = reproducing_report(p1, p2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "perturbed_patch_metrics.json").write_text(
        json.dumps({
            "training_assemblies": report,
            "polynomial_reproducing_prototype": prototype,
        }, indent=2), encoding="utf-8")

    selected = [
        "P1_constant_0.005", "P1_linear_0.0005",
        "P1_highfreq_A0.005_L10", "P2_highfreq_A0.005_L10",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    x = np.arange(len(selected))
    width = 0.13
    for index, method in enumerate(METHODS):
        offset = (index-0.5*(len(METHODS)-1))*width
        records = [report[scenario][method] for scenario in selected]
        axes[0].bar(x+offset, [record["Disp_RMS"] for record in records], width, label=method)
        axes[1].bar(x+offset, [record["Strain_RMS"] for record in records], width)
        axes[2].bar(x+offset, [record["Interface_Strain_RMS"] for record in records], width)
    for axis, title in zip(axes, ("Displacement RMS error", "Strain RMS error", "Interface strain RMS error")):
        axis.set_title(title)
        axis.set_xticks(x, selected, rotation=20, ha="right", fontsize=8)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7)
    figure.savefig(OUT / "perturbed_patch_summary.png", dpi=180, facecolor="white")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in ("P1", "P2"):
        amplitudes = np.array([0.001, 0.005, 0.01])
        amplitude_errors = [
            prototype[f"{name}_A{value:g}_L11"]["Strain_RMS"]
            for value in amplitudes]
        wavelengths = np.array([19.0, 11.0, 7.0])
        wavelength_errors = [
            prototype[f"{name}_A0.005_L{value:g}"]["Strain_RMS"]
            for value in wavelengths]
        axes[0].loglog(amplitudes, amplitude_errors, "o-", label=name)
        axes[1].plot(1/wavelengths, wavelength_errors, "o-", label=name)
    axes[0].set_xlabel("Nodal perturbation amplitude")
    axes[0].set_ylabel("Reconstructed strain RMS error")
    axes[0].set_title("Amplitude trend (wavelength 11 px)")
    axes[1].set_xlabel("Spatial frequency [1/pixel]")
    axes[1].set_ylabel("Reconstructed strain RMS error")
    axes[1].set_title("Frequency trend (amplitude 0.005 px)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(OUT / "reproducing_prototype_trends.png", dpi=180,
                   facecolor="white")
    plt.close(figure)
    print(json.dumps(prototype, indent=2))


if __name__ == "__main__":
    main()
