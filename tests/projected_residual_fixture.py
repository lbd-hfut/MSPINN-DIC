"""Synthetic projected-residual assembly fixture."""

import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import BSplineProjectedCosineResidualAssembly
from segpinndic.DIC_coarse_fields import CoarseBSplineField
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_trainers import get_inputs


def fixture(order=1, gate=1.0):
    assembly = BSplineProjectedCosineResidualAssembly
    centers = [np.array([5., 15., 25., 35.]), np.array([10.])]
    widths = [np.full(4, 16.), np.array([32.])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = assembly.init_params(
        residual_gate=gate, local_projection_order=order,
        projection_point_count=64, projection_regularization=1e-8)
    global_static, global_trainable = CoarseBSplineField.init_params(
        (8, 8), 3, [0., 0.], [40., 20.])
    controls = global_trainable["controls"].at[..., 0].set(.25)
    controls = controls.at[..., 1].set(-.15)
    base = jnp.asarray([
        [[.30, .08, -.04, .03, -.02, .01, .025],
         [-.20, .03, .05, -.02, .01, .02, -.015]],
        [[.20, -.05, .03, .02, .01, -.02, -.020],
         [.10, .04, -.03, .01, -.02, .03, .018]],
        [[-.10, .03, .04, -.01, .02, .02, .022],
         [.15, -.02, .05, .03, .01, -.01, -.016]],
        [[.05, -.04, -.02, .02, -.01, .03, -.024],
         [-.12, .05, .02, -.03, .02, .01, .020]],
    ])
    params = {
        "static": {
            "assembly": assembly_static,
            "decomposition": decomposition_static,
            "global_field": global_static,
            "domain": {"xmin": jnp.array([0., 0.]),
                       "xmax": jnp.array([40., 20.])},
        },
        "trainable": {
            "network": {"subdomain": {"coefficients": base}},
            "global_field": {"controls": controls},
        },
    }

    def network_fn(params_, point):
        c = params_["trainable"]["network"]["subdomain"]["coefficients"]
        xi, eta = point
        features = jnp.stack([
            jnp.ones_like(xi), xi, eta, xi*xi, xi*eta, eta*eta, xi**3])
        return c@features

    local_fns = (
        RectangularDecompositionND.norm_fn, network_fn,
        lambda _params, value: value, RectangularDecompositionND.window_fn)
    points = jnp.array([
        [3., 4.], [9., 7.], [14., 9.], [18., 8.],
        [24., 12.], [31., 11.], [37., 15.]])
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, jnp.ones(4, dtype=int), params,
        RectangularDecompositionND, assembly)
    cut = {"static": cut_all(params["static"]),
           "trainable": cut_active(params["trainable"])}
    return cut, points, plan, local_fns
