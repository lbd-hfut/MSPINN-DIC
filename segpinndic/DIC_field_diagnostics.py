"""Local-field and assembly-gradient diagnostics for MSPINN-DIC.

This module is intentionally read-only with respect to training.  It exposes
local neural displacement/gradient/strain and an exact normalized-POU gradient
decomposition without changing the assembled field or the photometric loss.
"""

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_trainers import (
    FBPINN_predict_forward,
    get_inputs,
    get_jmaps,
)


REQUIRED_FIRST_DERIVATIVES = (
    (0, ()), (1, ()),
    (0, (0,)), (0, (1,)),
    (1, (0,)), (1, (1,)),
)


def _take_model_params(all_params, model_indices):
    """Gather model-indexed pytree leaves for paired point/model evaluation."""

    gathered = {
        tree_key: {
            class_key: {
                key: jax.tree_util.tree_map(lambda p: p[model_indices], value)
                if key == "subdomain" else value
                for key, value in class_dict.items()
            }
            for class_key, class_dict in tree.items()
        }
        for tree_key, tree in all_params.items()
    }
    axes = {
        tree_key: {
            class_key: {
                key: jax.tree_util.tree_map(lambda _: 0, value)
                if key == "subdomain"
                else jax.tree_util.tree_map(lambda _: None, value)
                for key, value in class_dict.items()
            }
            for class_key, class_dict in tree.items()
        }
        for tree_key, tree in all_params.items()
    }
    return gathered, axes


def _local_field(params, point, local_fns, zero_offset=False):
    norm_fn, network_fn, unnorm_fn, _ = local_fns
    raw = network_fn(params, norm_fn(params, point))
    if zero_offset:
        output_unnorm = params["static"]["decomposition"]["subdomain"][
            "params"][5]
        return raw*output_unnorm[1]
    return unnorm_fn(params, raw)


def strain_from_gradient(gradient):
    """Convert displacement Jacobians ``(..., 2, 2)`` to engineering strain."""

    return jnp.stack(
        [gradient[..., 0, 0],
         0.5 * (gradient[..., 0, 1] + gradient[..., 1, 0]),
         gradient[..., 1, 1]],
        axis=-1,
    )


def evaluate_local_fields(
        all_params, points, model_indices, local_fns, zero_offset=False):
    """Evaluate paired local models at physical points, before any assembly.

    ``points[k]`` is evaluated only by ``model_indices[k]``.  Returned arrays
    have shapes ``value=(n,2)``, ``gradient=(n,2,2)``, ``strain=(n,3)``.
    Set ``zero_offset`` for local networks interpreted as residuals: output
    scale is retained but the displacement mean is not added.
    """

    points = jnp.asarray(points)
    model_indices = jnp.asarray(model_indices, dtype=jnp.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if model_indices.shape != (points.shape[0],):
        raise ValueError("model_indices must have shape (n,)")

    paired_params, axes = _take_model_params(all_params, model_indices)

    def value_and_gradient(params, point):
        field = lambda current: _local_field(
            params, current, local_fns, zero_offset=zero_offset)
        value = field(point)
        gradient = jax.jacfwd(field)(point)
        return value, gradient

    value, gradient = jax.vmap(
        value_and_gradient, in_axes=(axes, 0))(paired_params, points)
    return {
        "value": value,
        "gradient": gradient,
        "strain": strain_from_gradient(gradient),
    }


def evaluate_assembled_fields(
        all_params, points, decomposition, assembly, model_fns):
    """Evaluate final assembled displacement and its spatial derivatives."""

    points = jnp.asarray(points)
    active = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, active, all_params, decomposition, assembly)
    cut_params = {
        "static": cut_all(all_params["static"]),
        "trainable": cut_active(all_params["trainable"]),
    }
    u, v, ux, uy, vx, vy = FBPINN_predict_forward(
        cut_params, points, plan, model_fns,
        get_jmaps(REQUIRED_FIRST_DERIVATIVES))
    value = jnp.concatenate([u, v], axis=1)
    gradient = jnp.stack(
        [jnp.concatenate([ux, uy], axis=1),
         jnp.concatenate([vx, vy], axis=1)],
        axis=1,
    )
    return {
        "value": value,
        "gradient": gradient,
        "strain": strain_from_gradient(gradient),
    }


def decompose_normalized_pou_gradient(
        all_params, points, decomposition, assembly, model_fns):
    """Decompose a one-level normalized POU displacement gradient exactly.

    For ``u=A/W``, returns

    ``local = sum(w_i grad(u_i))/W``

    and

    ``window = sum(u_i outer grad(w_i))/W - outer(A,grad(W))/W**2``.

    The current decomposition has one POU level.  A clear error is raised for
    a multilevel decomposition instead of silently applying the wrong formula.
    """

    if not hasattr(assembly, "weight"):
        raise TypeError("gradient decomposition requires a normalized POU assembly")

    points = jnp.asarray(points)
    active = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, active, all_params, decomposition, assembly)
    m_take, n_take, _, _, npou = plan
    if int(npou) != 1:
        raise ValueError("POU gradient decomposition currently requires one POU level")

    cut_params = {
        "static": cut_all(all_params["static"]),
        "trainable": cut_active(all_params["trainable"]),
    }
    pair_points = points[n_take]
    paired_params, axes = _take_model_params(cut_params, m_take)
    local_fns = model_fns[:4]

    def pair_terms(params, point):
        field = lambda current: _local_field(params, current, local_fns)
        weight = lambda current: assembly.weight(params, current, local_fns)[0]
        value = field(point)
        gradient = jax.jacfwd(field)(point)
        w = weight(point)
        gradient_w = jax.grad(weight)(point)
        return value, gradient, w, gradient_w

    value_i, gradient_i, weight_i, gradient_weight_i = jax.vmap(
        pair_terms, in_axes=(axes, 0))(paired_params, pair_points)
    n_points = points.shape[0]

    segment_sum = lambda values: jax.ops.segment_sum(
        values, n_take, indices_are_sorted=False, num_segments=n_points)
    weight = segment_sum(weight_i)[:, None]
    gradient_weight = segment_sum(gradient_weight_i)
    numerator = segment_sum(weight_i[:, None] * value_i)
    local_numerator = segment_sum(
        weight_i[:, None, None] * gradient_i)
    value_window_numerator = segment_sum(
        value_i[:, :, None] * gradient_weight_i[:, None, :])

    local = local_numerator / weight[:, :, None]
    window = (
        value_window_numerator / weight[:, :, None]
        - numerator[:, :, None] * gradient_weight[:, None, :]
        / weight[:, :, None]**2
    )
    total = local + window
    displacement = numerator / weight
    return {
        "value": displacement,
        "total_gradient": total,
        "local_gradient": local,
        "window_gradient": window,
        "total_strain": strain_from_gradient(total),
        "local_strain": strain_from_gradient(local),
        "window_strain": strain_from_gradient(window),
        "weight_sum": weight,
    }


def adjacent_overlap_points(all_params, roi_points):
    """Return ROI points and model pairs for every adjacent x-chain overlap."""

    decomposition = all_params["static"]["decomposition"]
    nm = tuple(int(value) for value in decomposition["nm"])
    if len(nm) != 2 or nm[1] != 1:
        raise ValueError(f"x-chain diagnostics require [nx,1], received {nm}")
    xmins, xmaxs = decomposition["subdomain"]["params"][:2]
    xmins = np.asarray(xmins)[:, 0]
    xmaxs = np.asarray(xmaxs)[:, 0]
    roi_points = np.asarray(roi_points)
    overlaps = []
    for left in range(nm[0] - 1):
        right = left + 1
        overlap_left = max(xmins[left], xmins[right])
        overlap_right = min(xmaxs[left], xmaxs[right])
        selected = (
            (roi_points[:, 0] >= overlap_left)
            & (roi_points[:, 0] <= overlap_right)
        )
        overlaps.append({
            "left_model": left,
            "right_model": right,
            "x_left": float(overlap_left),
            "x_right": float(overlap_right),
            "x_interface": float(0.5 * (
                0.5*(xmins[left] + xmaxs[left])
                + 0.5*(xmins[right] + xmaxs[right]))),
            "points": roi_points[selected],
        })
    return overlaps
