import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import (
    C1MinimalCorrectionPOUAssembly,
    CosinePOUAssembly,
    GlobalBackboneLocalResidualAssembly,
    Hermite1DC1Assembly,
    P2FullFieldAssembly,
    QuinticPOUAssembly,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_field_diagnostics import (
    decompose_normalized_pou_gradient,
    evaluate_local_fields,
)
from segpinndic.DIC_trainers import (
    FBPINN_predict_forward,
    get_inputs,
    get_jmaps,
)


jax.config.update("jax_enable_x64", True)


METHODS = {
    "cosine_pou": CosinePOUAssembly,
    "quintic_pou": QuinticPOUAssembly,
    "hermite_1d_c1": Hermite1DC1Assembly,
    "c1_minimal_correction_pou": C1MinimalCorrectionPOUAssembly,
    "global_backbone_local_residual": GlobalBackboneLocalResidualAssembly,
    "p2_fullfield": P2FullFieldAssembly,
}
REQUIRED = (
    (0, ()), (1, ()),
    (0, (0,)), (0, (1,)), (1, (0,)), (1, (1,)),
)


def _base_value(coefficients, points):
    x, y = points[:, 0], points[:, 1]
    basis = jnp.stack([jnp.ones_like(x), x, y, x*x, x*y, y*y], axis=1)
    return basis@jnp.asarray(coefficients).T


def _base_strain(coefficients, points):
    c = jnp.asarray(coefficients)
    x, y = points[:, 0], points[:, 1]
    return jnp.stack([
        c[0, 1]+2*c[0, 3]*x+c[0, 4]*y,
        0.5*(c[0, 2]+c[0, 4]*x+2*c[0, 5]*y
             + c[1, 1]+2*c[1, 3]*x+c[1, 4]*y),
        c[1, 2]+c[1, 4]*x+2*c[1, 5]*y,
    ], axis=1)


def _fixture(base_coefficients, assembly, perturbation):
    nodes_x = np.linspace(0.0, 40.0, 5)
    nodes_y = np.linspace(0.0, 20.0, 2)
    centers = [0.5*(nodes_x[:-1]+nodes_x[1:]),
               0.5*(nodes_y[:-1]+nodes_y[1:])]
    widths = [1.6*np.diff(nodes_x), 1.6*np.diff(nodes_y)]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = assembly.init_params(correction_support_ratio=0.5)
    base = np.tile(np.asarray(base_coefficients)[None, :, :], (4, 1, 1))
    local_data = {
        "base": jnp.asarray(base),
        "offset": jnp.asarray(perturbation["offset"]),
        "slope": jnp.asarray(perturbation["slope"]),
        "amplitude": jnp.asarray(perturbation["amplitude"]),
        "frequency": jnp.asarray(perturbation["frequency"]),
        "phase": jnp.asarray(perturbation["phase"]),
    }
    trainable = {"network": {"subdomain": local_data}}
    if assembly is GlobalBackboneLocalResidualAssembly:
        trainable["network"]["subdomain"]["base"] = jnp.zeros_like(
            trainable["network"]["subdomain"]["base"])
        trainable["global_network"] = {
            "base": jnp.asarray(base_coefficients),
            "offset": jnp.zeros(2),
            "slope": jnp.zeros(2),
            "amplitude": jnp.zeros(2),
            "frequency": jnp.zeros(2),
            "phase": jnp.zeros(2),
            "physical_mu": jnp.array([20.0, 10.0]),
            "physical_sd": jnp.array([20.0, 10.0]),
        }
    params = {
        "static": {
            "assembly": assembly_static,
            "decomposition": decomposition_static,
            "domain": {
                "xmin": jnp.array([0.0, 0.0]),
                "xmax": jnp.array([40.0, 20.0]),
            },
        },
        "trainable": trainable,
    }

    def norm_fn(params, point):
        return point

    def network_fn(params, point):
        data = params["trainable"]["network"]["subdomain"]
        if "physical_mu" in data:
            point = point*data["physical_sd"]+data["physical_mu"]
        basis = jnp.array([
            1.0, point[0], point[1], point[0]**2,
            point[0]*point[1], point[1]**2,
        ])
        base_value = data["base"]@basis
        return (base_value + data["offset"] + data["slope"]*point[0]
                + data["amplitude"]*jnp.sin(
                    data["frequency"]*point[0]+data["phase"]))

    local_fns = (
        norm_fn, network_fn, lambda params, value: value,
        RectangularDecompositionND.window_fn)
    model_fns = local_fns+(assembly,)
    xx, yy = np.meshgrid(np.arange(0.0, 41.0), np.arange(0.0, 21.0),
                         indexing="xy")
    points = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, jnp.zeros(4, dtype=int), params,
        RectangularDecompositionND, assembly)
    cut_params = {
        "static": cut_all(params["static"]),
        "trainable": cut_active(params["trainable"]),
    }
    return cut_params, points, plan, local_fns, model_fns


def perturbation(kind, level, frequency=0.0):
    signs = np.array([
        [[1.0, -0.7], [-0.6, 0.8], [0.5, -1.0], [-0.8, 0.6]],
    ])[0]
    zeros = np.zeros((4, 2))
    data = {
        "offset": zeros.copy(), "slope": zeros.copy(),
        "amplitude": zeros.copy(), "frequency": zeros.copy(),
        "phase": np.array([[0.0, 0.4], [0.7, 1.1], [1.4, 1.8], [2.1, 2.5]]),
    }
    if kind == "constant":
        data["offset"] = level*signs
    elif kind == "linear":
        data["offset"] = 0.25*level*signs
        data["slope"] = level*signs
    elif kind == "high_frequency":
        data["amplitude"] = level*signs
        data["frequency"] = frequency*np.ones((4, 2))
    else:
        raise ValueError(kind)
    return data


def _rms(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2)))


def perturbed_patch_metrics(base_coefficients, assembly, perturbation_data):
    params, points, plan, local_fns, model_fns = _fixture(
        base_coefficients, assembly, perturbation_data)
    u, v, ux, uy, vx, vy = FBPINN_predict_forward(
        params, points, plan, model_fns, get_jmaps(REQUIRED))
    value = jnp.concatenate([u, v], axis=1)
    strain = jnp.concatenate([ux, 0.5*(uy+vx), vy], axis=1)
    exact_value = _base_value(base_coefficients, points)
    exact_strain = _base_strain(base_coefficients, points)
    displacement_error = value-exact_value
    strain_error = strain-exact_strain
    interfaces = jnp.array([10.0, 20.0, 30.0])
    distance = jnp.min(jnp.abs(points[:, 0, None]-interfaces[None, :]), axis=1)
    interface = distance <= 1.0
    interior = distance > 3.0

    local_displacement_mismatches = []
    local_strain_mismatches = []
    for left, interface_x in enumerate((10.0, 20.0, 30.0)):
        selected = jnp.abs(points[:, 0]-interface_x) <= 3.0
        selected_points = points[selected]
        left_field = evaluate_local_fields(
            params, selected_points,
            jnp.full(selected_points.shape[0], left), local_fns)
        right_field = evaluate_local_fields(
            params, selected_points,
            jnp.full(selected_points.shape[0], left+1), local_fns)
        local_displacement_mismatches.append(
            left_field["value"]-right_field["value"])
        local_strain_mismatches.append(
            left_field["strain"]-right_field["strain"])
    local_displacement_mismatch = jnp.concatenate(
        local_displacement_mismatches, axis=0)
    local_strain_mismatch = jnp.concatenate(local_strain_mismatches, axis=0)

    result = {
        "Disp_RMS": _rms(displacement_error),
        "Disp_Max": float(jnp.max(jnp.abs(displacement_error))),
        "Strain_RMS": _rms(strain_error),
        "Strain_Max": float(jnp.max(jnp.abs(strain_error))),
        "Interface_Strain_RMS": _rms(strain_error[interface]),
        "Interior_Strain_RMS": _rms(strain_error[interior]),
        "Interface_Interior_Ratio": (
            _rms(strain_error[interface])/_rms(strain_error[interior])
            if _rms(strain_error[interior]) else float("nan")),
        "Local_Network_Displacement_Mismatch_RMS": _rms(
            local_displacement_mismatch),
        "Local_Network_Strain_Mismatch_RMS": _rms(local_strain_mismatch),
        "P1_P2_Trend_Error": _rms(strain_error),
    }

    if assembly in (CosinePOUAssembly, QuinticPOUAssembly):
        decomposition = decompose_normalized_pou_gradient(
            params, points, RectangularDecompositionND, assembly, model_fns)
        result["Assembly_Induced_Strain_RMS"] = _rms(
            decomposition["window_strain"])
    if assembly is C1MinimalCorrectionPOUAssembly:
        correction = C1MinimalCorrectionPOUAssembly.assemble(
            params, points, plan, local_fns)[2]
        result.update({
            "Correction_RMS": _rms(correction),
            "Correction_Max": float(jnp.max(jnp.abs(correction))),
            "Correction_Field_RMS_Ratio": _rms(correction)/_rms(value),
        })
    if assembly is GlobalBackboneLocalResidualAssembly:
        global_value, residual, *_ = (
            GlobalBackboneLocalResidualAssembly.components(
                params, points, plan, local_fns))
        result.update({
            "Global_RMS": _rms(global_value),
            "Residual_RMS": _rms(residual),
            "Residual_Total_RMS_Ratio": _rms(residual)/_rms(value),
        })
    return result


class PerturbedPatchTests(unittest.TestCase):
    def setUp(self):
        self.p1 = np.array([
            [0.3, 0.02, -0.015, 0.0, 0.0, 0.0],
            [-0.4, 0.01, 0.025, 0.0, 0.0, 0.0],
        ])

    def test_all_training_assemblies_report_perturbed_metrics(self):
        data = perturbation("constant", 0.005)
        for name, assembly in METHODS.items():
            with self.subTest(method=name):
                metrics = perturbed_patch_metrics(self.p1, assembly, data)
                self.assertGreater(
                    metrics["Local_Network_Displacement_Mismatch_RMS"], 0.0)
                self.assertTrue(np.isfinite(metrics["Strain_RMS"]))

    def test_small_high_frequency_displacement_amplifies_strain_error(self):
        data = perturbation("high_frequency", 0.001, 2*np.pi/5.0)
        metrics = perturbed_patch_metrics(self.p1, CosinePOUAssembly, data)
        self.assertLess(metrics["Disp_RMS"], 0.002)
        self.assertGreater(metrics["Strain_RMS"], 1e-4)


if __name__ == "__main__":
    unittest.main()
