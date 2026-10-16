import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import (
    BicubicHermiteC1Assembly,
    CosinePOUAssembly,
    Hermite1DC1Assembly,
    QuinticPOUAssembly,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
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
    "bicubic_hermite_c1": BicubicHermiteC1Assembly,
}


def _patch_fixture(coefficients, assembly):
    x_nodes = np.linspace(0.0, 40.0, 5)
    y_nodes = np.linspace(0.0, 20.0, 2)
    centers = [0.5*(x_nodes[:-1]+x_nodes[1:]),
               0.5*(y_nodes[:-1]+y_nodes[1:])]
    widths = [1.6*np.diff(x_nodes), 1.6*np.diff(y_nodes)]
    decomp_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = assembly.init_params()
    model_coefficients = jnp.tile(jnp.asarray(coefficients)[None, :, :], (4, 1, 1))
    all_params = {
        "static": {"decomposition": decomp_static, "assembly": assembly_static},
        "trainable": {"network": {"subdomain": {
            "coefficients": model_coefficients}}},
    }

    def norm_fn(params, x):
        return x

    def network_fn(params, x):
        c = params["trainable"]["network"]["subdomain"]["coefficients"]
        basis = jnp.array([1.0, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])
        return c @ basis

    def unnorm_fn(params, value):
        return value

    model_fns = (
        norm_fn, network_fn, unnorm_fn,
        RectangularDecompositionND.window_fn, assembly)
    xx, yy = np.meshgrid(np.arange(0.0, 41.0), np.arange(0.0, 21.0),
                         indexing="xy")
    points = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=1))
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, jnp.zeros(4, dtype=int), all_params,
        RectangularDecompositionND, assembly)
    cut_params = {
        "static": cut_all(all_params["static"]),
        "trainable": cut_active(all_params["trainable"]),
    }
    return cut_params, points, plan, model_fns, x_nodes[1:-1]


def patch_metrics(coefficients, assembly):
    params, points, plan, model_fns, interfaces = _patch_fixture(
        coefficients, assembly)
    required = (
        (0, ()), (1, ()),
        (0, (0,)), (0, (1,)), (1, (0,)), (1, (1,)),
    )
    u, v, ux, uy, vx, vy = FBPINN_predict_forward(
        params, points, plan, model_fns, get_jmaps(required))
    assembled = jnp.concatenate([u, v], axis=1)
    strain = jnp.concatenate([ux, 0.5*(uy+vx), vy], axis=1)

    c = jnp.asarray(coefficients)
    x, y = points[:, 0], points[:, 1]
    basis = jnp.stack([jnp.ones_like(x), x, y, x**2, x*y, y**2], axis=1)
    exact = basis @ c.T
    exact_strain = jnp.stack([
        c[0, 1] + 2.0*c[0, 3]*x + c[0, 4]*y,
        0.5*((c[0, 2] + c[0, 4]*x + 2.0*c[0, 5]*y) +
             (c[1, 1] + 2.0*c[1, 3]*x + c[1, 4]*y)),
        c[1, 2] + c[1, 4]*x + 2.0*c[1, 5]*y,
    ], axis=1)
    displacement_error = assembled-exact
    strain_error = strain-exact_strain
    distance = jnp.min(jnp.abs(x[:, None]-jnp.asarray(interfaces)[None, :]), axis=1)
    interface = distance <= 1.0
    return {
        "max_displacement_error": float(jnp.max(jnp.abs(displacement_error))),
        "rms_displacement_error": float(jnp.sqrt(jnp.mean(displacement_error**2))),
        "max_strain_error": float(jnp.max(jnp.abs(strain_error))),
        "rms_strain_error": float(jnp.sqrt(jnp.mean(strain_error**2))),
        "interface_band_strain_rms": float(
            jnp.sqrt(jnp.mean(strain_error[interface]**2))),
    }


class AssemblyPatchTests(unittest.TestCase):
    def test_p1_affine_displacement(self):
        coefficients = np.array([
            [0.3, 0.02, -0.015, 0.0, 0.0, 0.0],
            [-0.4, 0.01, 0.025, 0.0, 0.0, 0.0],
        ])
        for name, assembly in METHODS.items():
            with self.subTest(method=name):
                metrics = patch_metrics(coefficients, assembly)
                self.assertLess(metrics["max_displacement_error"], 2e-10)
                self.assertLess(metrics["max_strain_error"], 2e-10)

    def test_p2_curved_displacement(self):
        coefficients = np.array([
            [0.2, 0.01, -0.008, 2e-4, -1.5e-4, 1e-4],
            [-0.1, 0.006, 0.012, -1e-4, 2.5e-4, -1.2e-4],
        ])
        for name, assembly in METHODS.items():
            with self.subTest(method=name):
                metrics = patch_metrics(coefficients, assembly)
                self.assertLess(metrics["max_displacement_error"], 3e-10)
                self.assertLess(metrics["max_strain_error"], 3e-10)


if __name__ == "__main__":
    unittest.main()
