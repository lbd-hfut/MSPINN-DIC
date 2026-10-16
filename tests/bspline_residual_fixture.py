"""Shared synthetic fixture for B-spline residual assembly tests."""

import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_coarse_fields import CoarseBSplineField
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_trainers import get_inputs


def fixture(assembly, *, local_scale=1.0, global_scale=1.0, freeze=False,
            gate=1.0, grid=(6, 6)):
    centers = [np.array([5.0, 15.0, 25.0, 35.0]), np.array([10.0])]
    widths = [np.full(4, 16.0), np.array([32.0])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = assembly.init_params(
        residual_gate=gate, freeze_backbone_stage2=freeze,
        p2_regularization=1e-12)
    global_static, global_trainable = CoarseBSplineField.init_params(
        grid, 3, [0., 0.], [40., 20.])
    controls = global_trainable["controls"].at[..., 0].set(0.4*global_scale)
    controls = controls.at[..., 1].set(-0.2*global_scale)
    local = local_scale*jnp.array([
        [[.02, .001, -.001], [-.01, .002, .001]],
        [[.01, -.001, .002], [.02, .001, -.002]],
        [[-.02, .002, .001], [.01, -.002, .001]],
        [[.01, -.001, -.002], [-.02, .001, .002]],
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
            "network": {"subdomain": {"coefficients": local}},
            "global_field": {"controls": controls},
        },
    }

    def norm_fn(_params, point):
        return point

    def network_fn(params_, point):
        c = params_["trainable"]["network"]["subdomain"]["coefficients"]
        return c[:, 0] + c[:, 1]*point[0] + c[:, 2]*point[1]

    local_fns = (norm_fn, network_fn, lambda _p, value: value,
                 RectangularDecompositionND.window_fn)
    points = jnp.array([[3., 4.], [9., 7.], [18., 8.], [31., 11.], [37., 15.]])
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, jnp.ones(4, dtype=int), params,
        RectangularDecompositionND, assembly)
    cut = {"static": cut_all(params["static"]),
           "trainable": cut_active(params["trainable"])}
    return cut, points, plan, local_fns

