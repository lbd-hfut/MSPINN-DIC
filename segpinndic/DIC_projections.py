"""Discrete polynomial projection for orthogonal local residual fields.

The module exposes one deep interface: build a geometry-only weighted-L2
projection plan, then project sampled values or evaluate their low polynomial
at query coordinates.  Rank checks, deterministic P2-to-P1 fallback,
conditioning, and the coefficient operator remain internal.  The tiny ridge
stabilizes linear algebra only; it is not a training objective.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


_BASIS_SIZE = {0: 1, 1: 3, 2: 6}


def polynomial_basis(points, order):
    """Return P0/P1/P2 basis values for normalized 2-D coordinates."""
    if order not in _BASIS_SIZE:
        raise ValueError("projection order must be 0, 1, or 2")
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("projection points must have shape (n, 2)")
    xi, eta = points[:, 0], points[:, 1]
    columns = [jnp.ones_like(xi)]
    if order >= 1:
        columns.extend([xi, eta])
    if order >= 2:
        columns.extend([xi*xi, xi*eta, eta*eta])
    return jnp.stack(columns, axis=1)


def _numpy_basis(points, order):
    return np.asarray(polynomial_basis(jnp.asarray(points), order),
                      dtype=np.float64)


def build_projection_plan(
        sample_points, order, weights=None, regularization=1.0e-8,
        allow_fallback=True):
    """Build a fixed weighted-L2 coefficient operator.

    ``sample_points`` must already be subdomain-local normalized coordinates.
    If an unregularized P2 design is rank deficient and fallback is enabled,
    P1 is attempted explicitly.  The returned metadata records the event.
    """
    requested_order = int(order)
    if requested_order not in _BASIS_SIZE:
        raise ValueError("projection order must be 0, 1, or 2")
    points = np.asarray(sample_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("sample_points must have non-empty shape (n, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("sample_points must be finite")
    if regularization <= 0 or regularization > 1.0e-4:
        raise ValueError("projection_regularization must be in (0, 1e-4]")
    if weights is None:
        weight = np.full(len(points), 1.0/len(points), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (len(points),) or np.any(weight <= 0):
            raise ValueError("weights must be positive with shape (n,)")
        weight = weight/weight.sum()

    effective_order = requested_order
    while True:
        phi = _numpy_basis(points, effective_order)
        rank = int(np.linalg.matrix_rank(phi))
        if rank == phi.shape[1]:
            break
        if not allow_fallback or effective_order == 0:
            raise ValueError(
                f"P{effective_order} projection design is rank deficient: "
                f"rank={rank}, expected={phi.shape[1]}")
        effective_order -= 1

    gram = phi.T@(weight[:, None]*phi)
    regularized = gram + regularization*np.eye(phi.shape[1])
    coefficient_operator = np.linalg.solve(
        regularized, phi.T*weight[None, :])
    return {
        "sample_points": jnp.asarray(points, dtype=jnp.float32),
        "sample_basis": jnp.asarray(phi, dtype=jnp.float32),
        "weights": jnp.asarray(weight, dtype=jnp.float32),
        "coefficient_operator": jnp.asarray(
            coefficient_operator, dtype=jnp.float32),
        "requested_order": requested_order,
        "effective_order": effective_order,
        "regularization": float(regularization),
        "rank": rank,
        "condition_number": float(np.linalg.cond(regularized)),
        "fallback": bool(effective_order != requested_order),
    }


def projection_coefficients(plan, sample_values):
    """Map sampled scalar/vector values to polynomial coefficients."""
    values = jnp.asarray(sample_values)
    if values.shape[0] != plan["sample_basis"].shape[0]:
        raise ValueError("sample_values leading dimension must match the plan")
    return plan["coefficient_operator"]@values


def project_low(plan, sample_values, query_points=None):
    """Evaluate the fitted low polynomial at sample or query coordinates."""
    coefficients = projection_coefficients(plan, sample_values)
    phi = (plan["sample_basis"] if query_points is None else
           polynomial_basis(query_points, plan["effective_order"]))
    return phi@coefficients


def project_orthogonal(plan, sample_values):
    """Return ``(I-P)q`` on the plan's sample coordinates."""
    values = jnp.asarray(sample_values)
    return values-project_low(plan, values)


def moment_residual(plan, residual_values):
    """Return normalized discrete polynomial moments ``Phi^T W r``."""
    residual = jnp.asarray(residual_values)
    if residual.shape[0] != plan["sample_basis"].shape[0]:
        raise ValueError("residual_values leading dimension must match the plan")
    return plan["sample_basis"].T@(
        plan["weights"][:, None]*residual)
