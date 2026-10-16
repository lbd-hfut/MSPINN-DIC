"""Shared synthetic fixture for the full-local P2 assembly tests."""

import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import P2FullFieldAssembly
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_trainers import get_inputs


def fixture(coefficients=None):
    if coefficients is None:
        coefficients = [[.2, .03, .001], [-.1, -.02, -.002]]
    centers = [np.array([5.0, 15.0, 25.0, 35.0]), np.array([10.0])]
    widths = [np.full(4, 16.0), np.array([32.0])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = P2FullFieldAssembly.init_params(
        p2_regularization=1.0e-12)
    local = jnp.tile(jnp.asarray(coefficients)[None, :, :], (4, 1, 1))
    params = {
        "static": {
            "assembly": assembly_static,
            "decomposition": decomposition_static,
            "domain": {"xmin": jnp.array([0.0, 0.0]),
                       "xmax": jnp.array([40.0, 20.0])},
        },
        "trainable": {"network": {"subdomain": {"coefficients": local}}},
    }

    def norm_fn(_params, point):
        return point

    def network_fn(params_, point):
        c = params_["trainable"]["network"]["subdomain"]["coefficients"]
        return c[:, 0] + c[:, 1]*point[0] + c[:, 2]*point[0]**2

    local_fns = (norm_fn, network_fn, lambda _p, value: value,
                 RectangularDecompositionND.window_fn)
    points = jnp.array([[3., 4.], [9., 7.], [18., 8.], [31., 11.]],
                       dtype=jnp.float64)
    plan, _, (_, cut_active, cut_fixed, cut_all, _) = get_inputs(
        points, jnp.ones(4, dtype=int), params,
        RectangularDecompositionND, P2FullFieldAssembly)
    cut = {"static": cut_all(params["static"]),
           "trainable": cut_active(params["trainable"])}
    return cut, cut_fixed(params["trainable"]), points, plan, local_fns

