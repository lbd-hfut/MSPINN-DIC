"""Explicit coarse global displacement fields.

The module exposes one deep interface: initialize a tensor-product cubic
B-spline field and evaluate it.  Knot construction, coordinate normalization,
basis recursion, boundary handling, and parameter layout remain internal.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _open_uniform_knots(n_controls, degree):
    if degree != 3:
        raise ValueError("The qualified coarse field uses cubic degree=3.")
    if n_controls < degree + 1:
        raise ValueError("A cubic B-spline requires at least four controls.")
    interior_count = n_controls-degree-1
    interior = (np.arange(1, interior_count+1, dtype=np.float32)
                /(interior_count+1))
    return np.concatenate([
        np.zeros(degree+1, dtype=np.float32),
        interior,
        np.ones(degree+1, dtype=np.float32),
    ])


def bspline_basis_1d(coordinate, knots, degree, n_controls):
    """Evaluate all clamped B-spline basis functions at one coordinate."""

    x = jnp.clip(coordinate, 0.0, 1.0)
    basis = ((x >= knots[:-1]) & (x < knots[1:])).astype(x.dtype)
    for current_degree in range(1, degree+1):
        count = knots.shape[0]-1-current_degree
        left_denominator = knots[current_degree:current_degree+count]-knots[:count]
        right_denominator = (
            knots[current_degree+1:current_degree+1+count]-knots[1:1+count])
        left = jnp.where(
            left_denominator > 0,
            (x-knots[:count])/jnp.where(
                left_denominator > 0, left_denominator, 1.0)*basis[:count],
            0.0)
        right = jnp.where(
            right_denominator > 0,
            (knots[current_degree+1:current_degree+1+count]-x)/jnp.where(
                right_denominator > 0, right_denominator, 1.0)*basis[1:count+1],
            0.0)
        basis = left+right
    basis = basis[:n_controls]
    endpoint = jnp.zeros_like(basis).at[n_controls-1].set(1.0)
    return jnp.where(x >= 1.0, endpoint, basis)


class CoarseBSplineField:
    """Tensor-product open-uniform clamped cubic B-spline displacement."""

    @staticmethod
    def init_params(grid_shape, degree, xmin, xmax):
        grid_shape = tuple(int(value) for value in grid_shape)
        if len(grid_shape) != 2:
            raise ValueError("bspline_grid must contain [nx, ny].")
        if degree != 3:
            raise ValueError("bspline_degree is fixed at 3 in this phase.")
        xmin = np.asarray(xmin, dtype=np.float32)
        xmax = np.asarray(xmax, dtype=np.float32)
        if xmin.shape != (2,) or xmax.shape != (2,) or np.any(xmax <= xmin):
            raise ValueError("B-spline xmin/xmax must define a positive 2-D box.")
        static = {
            "grid_shape": grid_shape,
            "degree": int(degree),
            "xmin": jnp.asarray(xmin),
            "xmax": jnp.asarray(xmax),
            "knots_x": jnp.asarray(_open_uniform_knots(grid_shape[0], degree)),
            "knots_y": jnp.asarray(_open_uniform_knots(grid_shape[1], degree)),
        }
        trainable = {
            "controls": jnp.zeros(grid_shape+(2,), dtype=jnp.float32),
        }
        return static, trainable

    @staticmethod
    def evaluate(static, trainable, point):
        normalized = (point-static["xmin"])/(static["xmax"]-static["xmin"])
        bx = bspline_basis_1d(
            normalized[0], static["knots_x"], static["degree"],
            static["grid_shape"][0])
        by = bspline_basis_1d(
            normalized[1], static["knots_y"], static["degree"],
            static["grid_shape"][1])
        return jnp.einsum("i,j,ijd->d", bx, by, trainable["controls"])

    @staticmethod
    def evaluate_batch(static, trainable, points):
        return jax.vmap(lambda point: CoarseBSplineField.evaluate(
            static, trainable, point))(points)
