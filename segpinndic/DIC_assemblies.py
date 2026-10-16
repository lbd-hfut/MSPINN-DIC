"""Differentiable global-field assembly for multi-subdomain PINN-DIC.

The public seam is deliberately small:

``Assembly.build_plan(all_params, x_batch, takes, all_ims)`` prepares geometry
outside the jitted optimization step, and ``Assembly.assemble(...)`` evaluates
the global displacement inside seed, Adam, L-BFGS, and prediction paths.

The current Hermite qualification scope is a rectangular x-chain ``[nx, 1]``.
The bicubic primitive itself is fully two-dimensional and uses the standard 16
corner DOFs (q, qx, qy, qxy).  Its x-chain adapter obtains compatible corner
DOFs from the C1 strip field before constructing a shared bicubic cell.
"""

from segpinndic.DIC_importlib import jax, jnp, np, vmap
from segpinndic.DIC_coarse_fields import CoarseBSplineField
from segpinndic.DIC_projections import (
    build_projection_plan,
    polynomial_basis as projection_polynomial_basis,
)


ASSEMBLY_METHODS = (
    "cosine_pou",
    "quintic_pou",
    "hermite_1d_c1",
    "bicubic_hermite_c1",
    "c1_minimal_correction_pou",
    "global_backbone_local_residual",
    "backbone_first_cosine_residual",
    "backbone_first_p2_residual",
    "p2_fullfield",
    "bspline_cosine_residual",
    "bspline_p2_residual",
    "bspline_projected_cosine_residual",
)


def quintic_smootherstep(t):
    """Classic C2 smootherstep on [0, 1], clamped outside the interval."""

    t = jnp.clip(t, 0.0, 1.0)
    return t**3 * (t * (6.0 * t - 15.0) + 10.0)


def quintic_compact_window(xmin, xmax, wmin, wmax, x):
    """Tensor-product compact window with rise/core/fall in each dimension."""

    x = x.astype(xmin.dtype)
    eps = jnp.finfo(xmin.dtype).eps
    left_t = (x - xmin) / jnp.maximum(wmin, eps)
    right_t = (xmax - x) / jnp.maximum(wmax, eps)
    left = quintic_smootherstep(left_t)
    right = quintic_smootherstep(right_t)
    inside = (x >= xmin) & (x <= xmax)
    ws = jnp.where(inside, jnp.minimum(left, right), 0.0)
    return jnp.prod(ws, axis=0, keepdims=True)


def cubic_hermite(q0, dq0, q1, dq1, t, length):
    """Cubic Hermite interpolation with derivatives in physical coordinates."""

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h00 * q0 + h10 * length * dq0 + h01 * q1 + h11 * length * dq1


def cubic_hermite_dx(q0, dq0, q1, dq1, t, length):
    """Physical-coordinate derivative of :func:`cubic_hermite`."""

    dh00 = 6.0 * t**2 - 6.0 * t
    dh10 = 3.0 * t**2 - 4.0 * t + 1.0
    dh01 = -6.0 * t**2 + 6.0 * t
    dh11 = 3.0 * t**2 - 2.0 * t
    return (dh00 * q0 + dh10 * length * dq0 +
            dh01 * q1 + dh11 * length * dq1) / length


def bicubic_hermite(q, qx, qy, qxy, xi, eta, length_x, length_y):
    """Bicubic Hermite surface from four-corner physical-coordinate DOFs.

    Corner arrays use shape ``(2, 2, ...)`` and indexing ``[ix, iy]``.
    Interpolating first in x and then in y is algebraically equivalent to the
    standard Hermite matrix form and applies the Lx, Ly, and Lx*Ly scalings.
    """

    bottom = cubic_hermite(q[0, 0], qx[0, 0], q[1, 0], qx[1, 0], xi, length_x)
    top = cubic_hermite(q[0, 1], qx[0, 1], q[1, 1], qx[1, 1], xi, length_x)
    bottom_y = cubic_hermite(
        qy[0, 0], qxy[0, 0], qy[1, 0], qxy[1, 0], xi, length_x)
    top_y = cubic_hermite(
        qy[0, 1], qxy[0, 1], qy[1, 1], qxy[1, 1], xi, length_x)
    return cubic_hermite(bottom, bottom_y, top, top_y, eta, length_y)


def polynomial_basis(points, order, center=None, scale=None):
    """P1/P2 basis in normalized physical coordinates for reconstruction."""

    points = jnp.asarray(points)
    if center is None:
        center = jnp.mean(points, axis=0)
    if scale is None:
        scale = jnp.maximum(jnp.max(jnp.abs(points-center), axis=0), 1.0)
    normalized = (points-center)/scale
    x, y = normalized[..., 0], normalized[..., 1]
    if order == 1:
        return jnp.stack([jnp.ones_like(x), x, y], axis=-1)
    if order == 2:
        return jnp.stack(
            [jnp.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)
    raise ValueError("reconstruction order must be 1 or 2")


def polynomial_reproducing_reconstruction(
        sample_points, sample_values, query_points, order=1, weights=None):
    """Deterministic weighted least-squares P1/P2 reconstruction prototype.

    This is an assembly-space primitive, not a trainable loss.  It returns the
    reconstructed values and coefficients of the minimum Euclidean-norm least
    squares solution in a centered/scaled polynomial basis.
    """

    sample_points = jnp.asarray(sample_points)
    sample_values = jnp.asarray(sample_values)
    query_points = jnp.asarray(query_points)
    center = jnp.mean(sample_points, axis=0)
    scale = jnp.maximum(jnp.max(jnp.abs(sample_points-center), axis=0), 1.0)
    design = polynomial_basis(sample_points, order, center, scale)
    query_design = polynomial_basis(query_points, order, center, scale)
    if weights is None:
        weights = jnp.ones(sample_points.shape[0], dtype=sample_points.dtype)
    root_weight = jnp.sqrt(jnp.asarray(weights))[:, None]
    weighted_design = design*root_weight
    weighted_values = sample_values*root_weight
    coefficients = jnp.linalg.lstsq(
        weighted_design, weighted_values, rcond=None)[0]
    return query_design@coefficients, coefficients


def p2_chain_conditioning(centres, query_x, regularization=1.0e-8):
    """Return deterministic WLS conditioning statistics for an x-chain.

    This diagnostic is pure NumPy and is intentionally outside the training
    loss.  The same normalized basis and distance weights are used by the JAX
    reconstruction below.
    """

    centres = np.asarray(centres, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64).reshape(-1)
    centre = np.mean(centres)
    scale = max(float(np.max(np.abs(centres-centre))), 1.0)
    z = (centres-centre)/scale
    design = np.stack([np.ones_like(z), z, z*z], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    if rank < 3:
        raise ValueError(
            f"P2 x-chain requires at least three independent anchors; rank={rank}.")
    weights = 1.0/(1.0 + ((query_x[:, None]-centres[None, :])/scale)**2)
    normal = np.einsum("nm,mi,mj->nij", weights, design, design)
    regularized = normal + regularization*np.eye(3)[None, :, :]
    condition = np.linalg.cond(regularized)
    singular = np.linalg.svd(regularized, compute_uv=False)
    return {
        "anchor_count": int(centres.size),
        "design_rank": rank,
        "regularization": float(regularization),
        "condition_min": float(np.min(condition)),
        "condition_median": float(np.median(condition)),
        "condition_max": float(np.max(condition)),
        "singular_value_ratio_min": float(np.min(singular[:, -1]/singular[:, 0])),
    }


def _take_model_params(all_params, model_indices):
    """Gather model-indexed leaves and build the matching vmap in_axes tree."""

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
                if key == "subdomain" else jax.tree_util.tree_map(lambda _: None, value)
                for key, value in class_dict.items()
            }
            for class_key, class_dict in tree.items()
        }
        for tree_key, tree in all_params.items()
    }
    return gathered, axes


def _local_value_raw(params, x, local_fns):
    norm_fn, network_fn, unnorm_fn, _ = local_fns
    x_norm = norm_fn(params, x)
    raw = network_fn(params, x_norm)
    return unnorm_fn(params, raw), raw


def _local_value(params, x, local_fns):
    return _local_value_raw(params, x, local_fns)[0]


def _local_value_x(params, x, local_fns):
    tangent = jnp.array([1.0, 0.0], dtype=x.dtype)
    return jax.jvp(lambda z: _local_value(params, z, local_fns),
                   (x,), (tangent,))


def _local_dofs(params, x, local_fns):
    field = lambda z: _local_value(params, z, local_fns)
    tangent_x = jnp.array([1.0, 0.0], dtype=x.dtype)
    tangent_y = jnp.array([0.0, 1.0], dtype=x.dtype)
    q, qx = jax.jvp(field, (x,), (tangent_x,))
    _, qy = jax.jvp(field, (x,), (tangent_y,))
    qx_field = lambda z: jax.jvp(field, (z,), (tangent_x,))[1]
    _, qxy = jax.jvp(qx_field, (x,), (tangent_y,))
    return q, qx, qy, qxy


def _chain_geometry(all_params):
    decomposition = all_params["static"]["decomposition"]
    nm = tuple(int(v) for v in decomposition["nm"])
    if len(nm) != 2 or nm[1] != 1:
        raise ValueError(
            "Hermite x-chain assembly currently requires n_subdomains=[nx, 1]; "
            f"received {nm}."
        )
    params = decomposition["subdomain"]["params"]
    xmins = np.asarray(params[0])[:, 0]
    xmaxs = np.asarray(params[1])[:, 0]
    centers = 0.5 * (xmins + xmaxs)
    transition_left = xmins[1:]
    transition_right = xmaxs[:-1]
    if np.any(transition_right <= transition_left):
        raise ValueError("Hermite assembly requires positive adjacent overlap widths.")
    return centers, transition_left, transition_right


def _chain_plan_for_x(all_params, x_values, all_ims):
    centers, transition_left, transition_right = _chain_geometry(all_params)
    x_values = np.asarray(x_values, dtype=np.float64).reshape(-1)
    n = x_values.size
    transition_id = -np.ones(n, dtype=np.int32)
    for interface_id, (x_left, x_right) in enumerate(
            zip(transition_left, transition_right)):
        select = ((transition_id < 0) & (x_values >= x_left) &
                  (x_values <= x_right))
        transition_id[select] = interface_id

    core_global = np.argmin(np.abs(x_values[:, None] - centers[None, :]), axis=1)
    in_transition = transition_id >= 0
    left_global = np.where(in_transition, transition_id, core_global)
    right_global = np.where(in_transition, transition_id + 1, core_global)
    x_left = np.where(in_transition, transition_left[np.maximum(transition_id, 0)], x_values)
    x_right = np.where(in_transition, transition_right[np.maximum(transition_id, 0)], x_values)

    all_ims = np.asarray(all_ims, dtype=np.int32)
    lookup = -np.ones(len(centers), dtype=np.int32)
    lookup[all_ims] = np.arange(all_ims.size, dtype=np.int32)
    required = np.unique(np.concatenate([core_global, left_global, right_global]))
    missing = required[lookup[required] < 0]
    if missing.size:
        raise ValueError(
            "Assembly plan requires subdomains absent from the active parameter cut: "
            f"{missing.tolist()}"
        )

    return (
        lookup[core_global],
        lookup[left_global],
        lookup[right_global],
        in_transition,
        x_left,
        x_right,
    )


def _pou_plan_for_points(all_params, points, all_ims):
    """Build the historical one/multilevel POU take plan for arbitrary points."""

    points = np.asarray(points)
    decomposition = all_params["static"]["decomposition"]
    xmins, xmaxs = [np.asarray(value) for value in
                    decomposition["subdomain"]["params"][:2]]
    inside = np.all(
        (points[:, None, :] >= xmins[None, :, :])
        & (points[:, None, :] <= xmaxs[None, :, :]),
        axis=-1,
    )
    n_take, global_m_take = np.where(inside)
    if n_take.size == 0:
        raise ValueError("Assembly trace points are outside every subdomain support.")

    all_ims = np.asarray(all_ims, dtype=np.int32)
    lookup = -np.ones(int(decomposition["m"]), dtype=np.int32)
    lookup[all_ims] = np.arange(all_ims.size, dtype=np.int32)
    if np.any(lookup[global_m_take] < 0):
        missing = np.unique(global_m_take[lookup[global_m_take] < 0])
        raise ValueError(
            "Assembly trace plan requires subdomains absent from the parameter cut: "
            f"{missing.tolist()}")
    m_take = lookup[global_m_take]
    pous = np.asarray(decomposition["subdomain"]["pou"])[all_ims].astype(int)
    point_pous = np.stack([n_take, pous[m_take, 0]], axis=-1)
    unique_point_pous, p_take = np.unique(
        point_pous, axis=0, return_inverse=True)
    np_take = unique_point_pous[:, 0]
    npou = len(np.unique(np.asarray(
        decomposition["subdomain"]["pou"]).astype(int)))
    return tuple(jnp.asarray(value) for value in (
        m_take, n_take, p_take, np_take, npou))


def _hermite_strip_value(left_params, right_params, x, x_left, x_right,
                          local_fns):
    left_point = jnp.stack([x_left, x[1]])
    right_point = jnp.stack([x_right, x[1]])
    q_left, qx_left = _local_value_x(left_params, left_point, local_fns)
    q_right, qx_right = _local_value_x(right_params, right_point, local_fns)
    length = x_right - x_left
    t = (x[0] - x_left) / length
    return cubic_hermite(q_left, qx_left, q_right, qx_right, t, length)


def _hermite_strip_dofs(left_params, right_params, x, x_left, x_right,
                         local_fns):
    left_point = jnp.stack([x_left, x[1]])
    right_point = jnp.stack([x_right, x[1]])
    ql, qxl, qyl, qxyl = _local_dofs(left_params, left_point, local_fns)
    qr, qxr, qyr, qxyr = _local_dofs(right_params, right_point, local_fns)
    length = x_right - x_left
    t = (x[0] - x_left) / length
    q = cubic_hermite(ql, qxl, qr, qxr, t, length)
    qx = cubic_hermite_dx(ql, qxl, qr, qxr, t, length)
    qy = cubic_hermite(qyl, qxyl, qyr, qxyr, t, length)
    qxy = cubic_hermite_dx(qyl, qxyl, qyr, qxyr, t, length)
    return q, qx, qy, qxy


class Assembly:
    """Static interface implemented by all assembly adapters."""

    requires_complete_chain = False
    uses_backbone_schedule = False

    @staticmethod
    def init_params(eps=1e-8, bicubic_cell_size=4.0,
                    correction_support_ratio=0.5, residual_gate=1.0,
                    global_coordinate_scale=1.0,
                    p2_regularization=1.0e-8,
                    freeze_backbone_stage2=False,
                    local_projection_order=-1,
                    projection_regularization=1.0e-8,
                    projection_point_count=256):
        if bicubic_cell_size <= 0:
            raise ValueError("bicubic_cell_size must be positive.")
        if not 0.0 < correction_support_ratio <= 1.0:
            raise ValueError("correction_support_ratio must be in (0, 1].")
        if not 0.0 < p2_regularization <= 1.0e-4:
            raise ValueError("p2_regularization must be in (0, 1e-4].")
        if int(local_projection_order) not in (-1, 0, 1, 2):
            raise ValueError("local_projection_order must be none, 0, 1, or 2.")
        if not 0.0 < projection_regularization <= 1.0e-4:
            raise ValueError(
                "projection_regularization must be in (0, 1e-4].")
        projection_side = int(round(np.sqrt(projection_point_count)))
        if projection_side**2 != int(projection_point_count) or projection_side < 3:
            raise ValueError(
                "projection_point_count must be a square integer of at least 9.")
        return {
            "eps": jnp.asarray(eps, dtype=jnp.float32),
            "bicubic_cell_size": jnp.asarray(
                bicubic_cell_size, dtype=jnp.float32),
            "correction_support_ratio": jnp.asarray(
                correction_support_ratio, dtype=jnp.float32),
            "residual_gate": jnp.asarray(residual_gate, dtype=jnp.float32),
            "global_coordinate_scale": jnp.asarray(
                global_coordinate_scale, dtype=jnp.float32),
            "p2_regularization": jnp.asarray(
                p2_regularization, dtype=jnp.float32),
            "freeze_backbone_stage2": jnp.asarray(
                freeze_backbone_stage2, dtype=jnp.bool_),
            "local_projection_order": int(local_projection_order),
            "projection_regularization": float(projection_regularization),
            "projection_point_count": int(projection_point_count),
        }, {}

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        raise NotImplementedError

    @staticmethod
    def init_extra_params(key, network, **kwargs):
        """Optional representation parameters owned by an assembly adapter."""

        del key, network, kwargs
        return {}, {}

    @staticmethod
    def initialize_local_residual_params(trainable_network):
        """Optional v2 hook; ordinary local fields keep their initializer."""

        return trainable_network

    @staticmethod
    def assemble(all_params, x_batch, plan, local_fns):
        raise NotImplementedError


class _NormalizedPOUAssembly(Assembly):
    guard_denominator = False

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del all_params, x_batch, all_ims
        return takes

    @classmethod
    def weight(cls, params, x, local_fns):
        raise NotImplementedError

    @classmethod
    def assemble(cls, all_params, x_batch, plan, local_fns):
        m_take, n_take, p_take, np_take, npou = plan
        x_take = x_batch[n_take]
        params_take, axes = _take_model_params(all_params, m_take)

        def local_weighted(params, x):
            u, raw = _local_value_raw(params, x, local_fns)
            w = cls.weight(params, x, local_fns)
            return u * w, w, raw, u

        us, ws, us_raw, us_u = vmap(local_weighted, in_axes=(axes, 0))(
            params_take, x_take)
        summed = jax.ops.segment_sum(
            jnp.concatenate([us, ws], axis=1), p_take,
            indices_are_sorted=False, num_segments=len(np_take))
        wp = summed[:, -1:]
        if cls.guard_denominator:
            eps = all_params["static"]["assembly"]["eps"]
            denominator = jnp.maximum(wp, eps)
        else:
            denominator = wp
        u = summed[:, :-1] / denominator
        u = jax.ops.segment_sum(
            u, np_take, indices_are_sorted=False, num_segments=len(x_batch))
        u = u / npou
        return u, wp, us, ws, us_raw, us_u


class CosinePOUAssembly(_NormalizedPOUAssembly):
    """Numerically compatible adapter for the repository's cosine baseline."""

    @classmethod
    def weight(cls, params, x, local_fns):
        return local_fns[3](params, x)


class QuinticPOUAssembly(_NormalizedPOUAssembly):
    """Normalized POU with compact C2 quintic rise/core/fall windows."""

    guard_denominator = True

    @classmethod
    def weight(cls, params, x, local_fns):
        del local_fns
        values = params["static"]["decomposition"]["subdomain"]["params"]
        xmin, xmax, wmin, wmax, flag = values[:5]
        compact = quintic_compact_window(xmin, xmax, wmin, wmax, x)
        return flag * compact + (1.0 - flag)


class Hermite1DC1Assembly(Assembly):
    """Derivative-aware C1 transition assembly for an x-direction chain."""

    requires_complete_chain = True

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del takes
        plan = _chain_plan_for_x(all_params, np.asarray(x_batch)[:, 0], all_ims)
        return tuple(jnp.asarray(value) for value in plan)

    @staticmethod
    def assemble(all_params, x_batch, plan, local_fns):
        core_i, left_i, right_i, in_transition, x_left, x_right = plan
        core_params, axes = _take_model_params(all_params, core_i)
        left_params, _ = _take_model_params(all_params, left_i)
        right_params, _ = _take_model_params(all_params, right_i)

        def assemble_point(core, left, right, x, xl, xr, transition):
            return jax.lax.cond(
                transition,
                lambda _: _hermite_strip_value(left, right, x, xl, xr, local_fns),
                lambda _: _local_value(core, x, local_fns),
                operand=None,
            )

        u = vmap(assemble_point, in_axes=(axes, axes, axes, 0, 0, 0, 0))(
            core_params, left_params, right_params, x_batch,
            x_left, x_right, in_transition)
        ones = jnp.ones((u.shape[0], 1), dtype=u.dtype)
        return u, ones, u, ones, u, u


class BicubicHermiteC1Assembly(Assembly):
    """Two-dimensional bicubic Hermite field qualified on an x-chain.

    Query pixels are placed inside shared half-pixel cells.  Each corner gets
    q/qx/qy/qxy from the compatible x-chain C1 source field, so all 16 bicubic
    DOFs participate in the assembled displacement used by the loss.
    """

    requires_complete_chain = True

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del takes
        points = np.asarray(x_batch, dtype=np.float64)
        cell_size = float(all_params["static"]["assembly"]["bicubic_cell_size"])
        x_center = np.floor(points[:, 0]/cell_size + 0.5) * cell_size
        y_center = np.floor(points[:, 1]/cell_size + 0.5) * cell_size
        x0 = x_center - 0.5*cell_size
        y0 = y_center - 0.5*cell_size
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        corner_x = np.stack([x0, x1, x0, x1], axis=1)
        corner_y = np.stack([y0, y0, y1, y1], axis=1)
        corners = np.stack([corner_x, corner_y], axis=-1).reshape(-1, 2)
        unique_corners, corner_take = np.unique(
            corners, axis=0, return_inverse=True)
        chain = _chain_plan_for_x(all_params, unique_corners[:, 0], all_ims)
        geometry_dtype = all_params["static"]["decomposition"]["subdomain"][
            "params"][0].dtype
        return tuple(jnp.asarray(value) for value in chain) + (
            jnp.asarray(unique_corners, dtype=geometry_dtype),
            jnp.asarray(corner_take.reshape(-1, 4), dtype=jnp.int32),
            jnp.asarray(x0, dtype=geometry_dtype),
            jnp.asarray(x1, dtype=geometry_dtype),
            jnp.asarray(y0, dtype=geometry_dtype),
            jnp.asarray(y1, dtype=geometry_dtype),
        )

    @staticmethod
    def assemble(all_params, x_batch, plan, local_fns):
        (core_i, left_i, right_i, in_transition, x_left, x_right,
         corners, corner_take, x0, x1, y0, y1) = plan
        core_params, axes = _take_model_params(all_params, core_i)
        left_params, _ = _take_model_params(all_params, left_i)
        right_params, _ = _take_model_params(all_params, right_i)

        def source_dofs(core, left, right, point, xl, xr, transition):
            return jax.lax.cond(
                transition,
                lambda _: _hermite_strip_dofs(
                    left, right, point, xl, xr, local_fns),
                lambda _: _local_dofs(core, point, local_fns),
                operand=None,
            )

        q, qx, qy, qxy = vmap(
            source_dofs, in_axes=(axes, axes, axes, 0, 0, 0, 0))(
                core_params, left_params, right_params, corners,
                x_left, x_right, in_transition)
        q = q[corner_take]
        qx = qx[corner_take]
        qy = qy[corner_take]
        qxy = qxy[corner_take]

        def interpolate(q4, qx4, qy4, qxy4, point, xl, xr, yb, yt):
            def corners_2x2(values):
                return jnp.stack([
                    jnp.stack([values[0], values[2]], axis=0),
                    jnp.stack([values[1], values[3]], axis=0),
                ], axis=0)
            length_x = xr - xl
            length_y = yt - yb
            xi = (point[0] - xl) / length_x
            eta = (point[1] - yb) / length_y
            return bicubic_hermite(
                corners_2x2(q4), corners_2x2(qx4),
                corners_2x2(qy4), corners_2x2(qxy4),
                xi, eta, length_x, length_y)

        u = vmap(interpolate, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
            q, qx, qy, qxy, x_batch, x0, x1, y0, y1)
        ones = jnp.ones((u.shape[0], 1), dtype=u.dtype)
        return u, ones, u, ones, u, u


class C1MinimalCorrectionPOUAssembly(Assembly):
    """Cosine POU plus the minimum compact two-DOF C1 trace correction.

    At every x-chain interface and y coordinate, the compatible target is the
    arithmetic mean of the adjacent local value and x derivative.  The
    correction changes the cosine POU trace to that target and is joined to
    zero value/zero slope at both compact-support edges by the unique cubic
    Hermite segment on each side.  It does not replace the overlap field.
    """

    requires_complete_chain = True

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        centers, overlap_left, overlap_right = _chain_geometry(all_params)
        interfaces = 0.5 * (centers[:-1] + centers[1:])
        overlap_width = overlap_right - overlap_left
        ratio = float(all_params["static"]["assembly"][
            "correction_support_ratio"])
        half_supports = 0.5 * ratio * overlap_width

        points = np.asarray(x_batch, dtype=np.float64)
        distances = np.abs(points[:, 0, None] - interfaces[None, :])
        interface_id = np.argmin(distances, axis=1).astype(np.int32)
        active = distances[np.arange(points.shape[0]), interface_id] <= (
            half_supports[interface_id] + 1e-10)

        # Reuse one trace evaluation for every repeated (interface, y) pair.
        keys = np.stack([interface_id, points[:, 1]], axis=1)
        active_keys = keys[active]
        if active_keys.shape[0]:
            unique_keys, active_take = np.unique(
                active_keys, axis=0, return_inverse=True)
            trace_take = np.zeros(points.shape[0], dtype=np.int32)
            trace_take[active] = active_take
        else:
            # Keep all JAX array dimensions non-empty for sparse seed batches.
            unique_keys = np.array([[0.0, points[0, 1]]], dtype=np.float64)
            trace_take = np.zeros(points.shape[0], dtype=np.int32)

        trace_interface = unique_keys[:, 0].astype(np.int32)
        trace_points = np.stack(
            [interfaces[trace_interface], unique_keys[:, 1]], axis=1)
        trace_plan = _pou_plan_for_points(
            all_params, trace_points, all_ims)

        all_ims_np = np.asarray(all_ims, dtype=np.int32)
        lookup = -np.ones(len(centers), dtype=np.int32)
        lookup[all_ims_np] = np.arange(all_ims_np.size, dtype=np.int32)
        trace_left = lookup[trace_interface]
        trace_right = lookup[trace_interface + 1]
        if np.any(trace_left < 0) or np.any(trace_right < 0):
            raise ValueError("Minimal correction requires both adjacent local models.")

        point_interface = interfaces[interface_id]
        point_half_support = half_supports[interface_id]
        return (
            takes,
            trace_plan,
            jnp.asarray(trace_points, dtype=jnp.float32),
            jnp.asarray(trace_left),
            jnp.asarray(trace_right),
            jnp.asarray(trace_take),
            jnp.asarray(active),
            jnp.asarray(point_interface, dtype=jnp.float32),
            jnp.asarray(point_half_support, dtype=jnp.float32),
        )

    @staticmethod
    def assemble(all_params, x_batch, plan, local_fns):
        (base_plan, trace_plan, trace_points, trace_left, trace_right,
         trace_take, active, point_interface, half_support) = plan
        base = CosinePOUAssembly.assemble(
            all_params, x_batch, base_plan, local_fns)

        def trace_pou(current_points):
            return CosinePOUAssembly.assemble(
                all_params, current_points, trace_plan, local_fns)[0]

        tangent_x = jnp.tile(
            jnp.array([[1.0, 0.0]], dtype=trace_points.dtype),
            (trace_points.shape[0], 1))
        pou_value, pou_x = jax.jvp(
            trace_pou, (trace_points,), (tangent_x,))

        left_params, axes = _take_model_params(all_params, trace_left)
        right_params, _ = _take_model_params(all_params, trace_right)
        left_value, left_x = vmap(
            lambda params, point: _local_value_x(params, point, local_fns),
            in_axes=(axes, 0))(left_params, trace_points)
        right_value, right_x = vmap(
            lambda params, point: _local_value_x(params, point, local_fns),
            in_axes=(axes, 0))(right_params, trace_points)

        target_value = 0.5 * (left_value + right_value)
        target_x = 0.5 * (left_x + right_x)
        value_dof = (target_value - pou_value)[trace_take]
        derivative_dof = (target_x - pou_x)[trace_take]

        safe_half_support = jnp.where(active, half_support, 1.0)
        s = (x_batch[:, 0] - point_interface) / safe_half_support
        left_t = s + 1.0
        right_t = s
        zeros = jnp.zeros_like(value_dof)
        left_correction = cubic_hermite(
            zeros, zeros, value_dof, derivative_dof,
            left_t[:, None], safe_half_support[:, None])
        right_correction = cubic_hermite(
            value_dof, derivative_dof, zeros, zeros,
            right_t[:, None], safe_half_support[:, None])
        correction = jnp.where(
            (s <= 0.0)[:, None], left_correction, right_correction)
        correction = jnp.where(active[:, None], correction, 0.0)
        corrected = base[0] + correction
        return corrected, base[1], correction, base[3], base[4], base[5]


class GlobalBackboneLocalResidualAssembly(Assembly):
    """Global low-frequency neural field plus cosine-assembled local residuals."""

    @staticmethod
    def init_extra_params(key, network, layer_sizes, **kwargs):
        del kwargs
        static, trainable = network.init_params(key=key, layer_sizes=layer_sizes)
        return ({"global_network": static} if static else {},
                {"global_network": trainable})

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del all_params, x_batch, all_ims
        return takes

    @staticmethod
    def components(all_params, x_batch, plan, local_fns):
        norm_fn, network_fn, _, window_fn = local_fns
        m_take, n_take, p_take, np_take, npou = plan
        x_take = x_batch[n_take]
        params_take, axes = _take_model_params(all_params, m_take)

        def local_residual_weighted(params, point):
            raw = network_fn(params, norm_fn(params, point))
            unnorm = params["static"]["decomposition"]["subdomain"][
                "params"][5]
            residual = raw * unnorm[1]
            weight = window_fn(params, point)
            return residual * weight, weight, raw, residual

        weighted, weights, raw, local_residuals = vmap(
            local_residual_weighted, in_axes=(axes, 0))(
                params_take, x_take)
        summed = jax.ops.segment_sum(
            jnp.concatenate([weighted, weights], axis=1), p_take,
            indices_are_sorted=False, num_segments=len(np_take))
        weight_sum = summed[:, -1:]
        residual = summed[:, :-1] / weight_sum
        residual = jax.ops.segment_sum(
            residual, np_take, indices_are_sorted=False,
            num_segments=len(x_batch)) / npou

        global_trainable = all_params["trainable"]["global_network"]
        global_params = {
            "static": {
                "network": all_params["static"].get("global_network", {}),
            },
            "trainable": {
                "network": {"subdomain": global_trainable},
            },
        }
        domain = all_params["static"]["domain"]
        mu_x = 0.5 * (domain["xmin"] + domain["xmax"])
        sd_x = 0.5 * (domain["xmax"] - domain["xmin"])
        output_unnorm = all_params["static"]["decomposition"][
            "subdomain"]["params"][5][0]

        def global_field(point):
            normalized = (point - mu_x) / sd_x
            raw_global = network_fn(global_params, normalized)
            return raw_global * output_unnorm[1] + output_unnorm[0]

        global_value = vmap(global_field)(x_batch)
        return global_value, residual, weight_sum, raw, local_residuals

    @staticmethod
    def assemble(all_params, x_batch, plan, local_fns):
        global_value, residual, weight_sum, raw, local_residuals = (
            GlobalBackboneLocalResidualAssembly.components(
                all_params, x_batch, plan, local_fns))
        total = global_value + residual
        return total, weight_sum, residual, weight_sum, raw, local_residuals


class _BackboneFirstResidualAssembly(Assembly):
    """Shared global-low-frequency plus gated local-residual representation.

    The trainer changes only ``static[assembly][residual_gate]``.  This module
    owns the choice of residual assembly, zero-output local initialization, and
    global coordinate scale, keeping schedule mechanics outside the assembly.
    """

    requires_complete_chain = True
    uses_backbone_schedule = True

    @staticmethod
    def init_params(**kwargs):
        static, trainable = Assembly.init_params(**kwargs)
        scale = float(static["global_coordinate_scale"])
        if not 0.0 < scale <= 1.0:
            raise ValueError("global_coordinate_scale must be in (0, 1].")
        return static, trainable

    @staticmethod
    def init_extra_params(key, network, layer_sizes, **kwargs):
        del kwargs
        static, trainable = network.init_params(key=key, layer_sizes=layer_sizes)
        return ({"global_network": static} if static else {},
                {"global_network": trainable})

    @staticmethod
    def initialize_local_residual_params(trainable_network):
        """Zero the AdaptiveFCN output affine map, retaining hidden features."""

        layers = list(trainable_network["subdomain"]["layers"])
        final = layers[-1]
        if len(final) < 2:
            raise TypeError("v2 residual initialization expects affine output layers.")
        layers[-1] = (
            jnp.zeros_like(final[0]), jnp.zeros_like(final[1]), *final[2:])
        return {"subdomain": {"layers": layers}}

    @staticmethod
    def _global_value(all_params, x_batch, local_fns):
        _, network_fn, _, _ = local_fns
        global_params = {
            "static": {"network": all_params["static"].get("global_network", {})},
            "trainable": {"network": {
                "subdomain": all_params["trainable"]["global_network"]}},
        }
        domain = all_params["static"]["domain"]
        mu_x = 0.5*(domain["xmin"] + domain["xmax"])
        sd_x = 0.5*(domain["xmax"] - domain["xmin"])
        output_unnorm = all_params["static"]["decomposition"][
            "subdomain"]["params"][5][0]
        coordinate_scale = all_params["static"]["assembly"][
            "global_coordinate_scale"]

        def global_field(point):
            normalized = coordinate_scale*(point-mu_x)/sd_x
            raw = network_fn(global_params, normalized)
            return raw*output_unnorm[1] + output_unnorm[0]

        return vmap(global_field)(x_batch)

    @staticmethod
    def _cosine_residual(all_params, x_batch, takes, local_fns):
        norm_fn, network_fn, _, window_fn = local_fns
        m_take, n_take, p_take, np_take, npou = takes
        params_take, axes = _take_model_params(all_params, m_take)
        x_take = x_batch[n_take]

        def weighted_residual(params, point):
            raw = network_fn(params, norm_fn(params, point))
            output_unnorm = params["static"]["decomposition"][
                "subdomain"]["params"][5]
            value = raw*output_unnorm[1]
            weight = window_fn(params, point)
            return value*weight, weight, raw, value

        weighted, weights, raw, local_values = vmap(
            weighted_residual, in_axes=(axes, 0))(params_take, x_take)
        summed = jax.ops.segment_sum(
            jnp.concatenate([weighted, weights], axis=1), p_take,
            indices_are_sorted=False, num_segments=len(np_take))
        weight_sum = summed[:, -1:]
        residual = summed[:, :-1]/weight_sum
        residual = jax.ops.segment_sum(
            residual, np_take, indices_are_sorted=False,
            num_segments=len(x_batch))/npou
        return residual, weight_sum, raw, local_values

    @classmethod
    def components(cls, all_params, x_batch, plan, local_fns):
        global_value = cls._global_value(all_params, x_batch, local_fns)
        residual, weight_sum, raw, local_values = cls._residual(
            all_params, x_batch, plan, local_fns)
        gate = all_params["static"]["assembly"]["residual_gate"]
        residual = gate*residual
        return global_value, residual, weight_sum, raw, local_values

    @classmethod
    def assemble(cls, all_params, x_batch, plan, local_fns):
        global_value, residual, weight_sum, raw, local_values = cls.components(
            all_params, x_batch, plan, local_fns)
        return global_value+residual, weight_sum, residual, weight_sum, raw, local_values


class BackboneFirstCosineResidualAssembly(_BackboneFirstResidualAssembly):
    """Backbone-first representation with normalized cosine residual POU."""

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del all_params, x_batch, all_ims
        return takes

    @staticmethod
    def _residual(all_params, x_batch, plan, local_fns):
        return _BackboneFirstResidualAssembly._cosine_residual(
            all_params, x_batch, plan, local_fns)


def _build_p2_chain_plan(all_params, all_ims, dtype=jnp.float32):
    """Precompute anchor-only geometry shared by full-field and residual P2."""

    centres, _, _ = _chain_geometry(all_params)
    centres = np.asarray(centres)[np.asarray(all_ims)]
    centre = float(np.mean(centres))
    scale = max(float(np.max(np.abs(centres-centre))), 1.0)
    z = (centres-centre)/scale
    design = np.stack([np.ones_like(z), z, z*z], axis=1)
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("P2 x-chain reconstruction requires rank-3 anchor geometry.")
    return tuple(jnp.asarray(value, dtype=dtype) for value in (
        centres, centre, scale, design))


def _p2_chain_reconstruct(
        all_params, x_batch, plan, local_fns, *, zero_offset):
    """Reconstruct generic local vector fields with the shared P2 WLS module.

    For every query ``(x,y)``, local model ``i`` is sampled at
    ``(centre_i,y)``.  These are field samples, not values cached at a fixed y
    and not local predictions evaluated at the query x.  Only anchor geometry
    is precomputed; query-dependent weights remain in JAX so final-displacement
    spatial autodiff includes the derivative of the reconstruction itself.
    """

    centres, centre, scale, design = plan
    norm_fn, network_fn, unnorm_fn, _ = local_fns
    model_count = centres.shape[0]
    point_count = x_batch.shape[0]
    anchor_points = jnp.stack([
        jnp.repeat(centres, point_count),
        jnp.tile(x_batch[:, 1], model_count),
    ], axis=1)
    model_indices = jnp.repeat(jnp.arange(model_count), point_count)
    params, axes = _take_model_params(all_params, model_indices)

    def local_field(params_, point):
        raw = network_fn(params_, norm_fn(params_, point))
        if zero_offset:
            output_unnorm = params_["static"]["decomposition"][
                "subdomain"]["params"][5]
            value = raw*output_unnorm[1]
        else:
            value = unnorm_fn(params_, raw)
        return value, raw

    values, raw = vmap(local_field, in_axes=(axes, 0))(params, anchor_points)
    values = values.reshape(model_count, point_count, 2).transpose(1, 0, 2)
    query_z = (x_batch[:, 0]-centre)/scale
    query_design = jnp.stack(
        [jnp.ones_like(query_z), query_z, query_z*query_z], axis=1)
    weights = 1.0/(1.0 + ((x_batch[:, 0, None]-centres[None, :])/scale)**2)
    normal = jnp.einsum("nm,mi,mj->nij", weights, design, design)
    rhs = jnp.einsum("nm,mi,nmd->nid", weights, design, values)
    regularization = all_params["static"]["assembly"]["p2_regularization"]
    coefficients = jnp.linalg.solve(
        normal + regularization*jnp.eye(3, dtype=x_batch.dtype)[None, :, :],
        rhs)
    reconstructed = jnp.einsum("ni,nid->nd", query_design, coefficients)
    ones = jnp.ones((point_count, 1), dtype=x_batch.dtype)
    return reconstructed, ones, raw, values


class BackboneFirstP2ResidualAssembly(_BackboneFirstResidualAssembly):
    """Backbone-first representation with a differentiable P2 x-chain residual.

    At each query y, the four local residual traces are sampled at their
    subdomain centres.  A weighted quadratic fit in x is evaluated at the query
    x.  The small 3x3 solve is JIT-compatible and exactly reproduces P0/P1/P2
    residual fields represented consistently by the local traces.
    """

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del x_batch, takes
        return _build_p2_chain_plan(all_params, all_ims)

    @staticmethod
    def _residual(all_params, x_batch, plan, local_fns):
        return _p2_chain_reconstruct(
            all_params, x_batch, plan, local_fns, zero_offset=True)


class P2FullFieldAssembly(Assembly):
    """Full local displacement fields reconstructed by the shared P2 WLS."""

    requires_complete_chain = True

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del x_batch, takes
        return _build_p2_chain_plan(all_params, all_ims)

    @staticmethod
    def assemble(all_params, x_batch, plan, local_fns):
        value, ones, raw, local_values = _p2_chain_reconstruct(
            all_params, x_batch, plan, local_fns, zero_offset=False)
        return value, ones, value, ones, raw, local_values


class _BSplineBackboneResidualAssembly(Assembly):
    """Explicit coarse B-spline field plus structurally gated local residual."""

    uses_backbone_schedule = True
    residual_activates_at_stage1 = True

    @staticmethod
    def init_extra_params(
            key, network, layer_sizes, bspline_grid, bspline_degree,
            xmin, xmax, **kwargs):
        del key, network, layer_sizes, kwargs
        static, trainable = CoarseBSplineField.init_params(
            bspline_grid, bspline_degree, xmin, xmax)
        return {"global_field": static}, {"global_field": trainable}

    @staticmethod
    def initialize_local_residual_params(trainable_network):
        return _BackboneFirstResidualAssembly.initialize_local_residual_params(
            trainable_network)

    @staticmethod
    def _global_value(all_params, x_batch):
        controls = all_params["trainable"]["global_field"]
        gate = all_params["static"]["assembly"]["residual_gate"]
        freeze = all_params["static"]["assembly"]["freeze_backbone_stage2"]
        freeze_now = jnp.logical_and(freeze, gate > 0.0)
        controls = jax.tree_util.tree_map(
            lambda value: jax.lax.cond(
                freeze_now, lambda x: jax.lax.stop_gradient(x), lambda x: x,
                value),
            controls)
        return CoarseBSplineField.evaluate_batch(
            all_params["static"]["global_field"], controls, x_batch)

    @classmethod
    def components(cls, all_params, x_batch, plan, local_fns):
        global_value = cls._global_value(all_params, x_batch)
        residual, weight_sum, raw, local_values = cls._residual(
            all_params, x_batch, plan, local_fns)
        gate = all_params["static"]["assembly"]["residual_gate"]
        return global_value, gate*residual, weight_sum, raw, local_values

    @classmethod
    def assemble(cls, all_params, x_batch, plan, local_fns):
        global_value, residual, weight_sum, raw, local_values = cls.components(
            all_params, x_batch, plan, local_fns)
        return (global_value+residual, weight_sum, residual, weight_sum,
                raw, local_values)


class BSplineCosineResidualAssembly(_BSplineBackboneResidualAssembly):
    """Coarse cubic B-spline backbone plus cosine-POU local residual."""

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del all_params, x_batch, all_ims
        return takes

    @staticmethod
    def _residual(all_params, x_batch, plan, local_fns):
        return _BackboneFirstResidualAssembly._cosine_residual(
            all_params, x_batch, plan, local_fns)


class BSplineP2ResidualAssembly(_BSplineBackboneResidualAssembly):
    """Coarse cubic B-spline backbone plus P2-reconstructed local residual."""

    requires_complete_chain = True

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del x_batch, takes
        return _build_p2_chain_plan(all_params, all_ims)

    @staticmethod
    def _residual(all_params, x_batch, plan, local_fns):
        return _p2_chain_reconstruct(
            all_params, x_batch, plan, local_fns, zero_offset=True)


class BSplineProjectedCosineResidualAssembly(_BSplineBackboneResidualAssembly):
    """B-spline field plus hard P0/P1/P2-orthogonal cosine residuals."""

    @staticmethod
    def build_plan(all_params, x_batch, takes, all_ims):
        del x_batch
        static = all_params["static"]["assembly"]
        order = int(static["local_projection_order"])
        if order not in (0, 1, 2):
            raise ValueError(
                "projected residual assembly requires local_projection_order "
                "0, 1, or 2")
        point_count = int(static["projection_point_count"])
        side = int(round(np.sqrt(point_count)))
        axis = np.linspace(-1.0, 1.0, side, dtype=np.float32)
        normalized_points = np.stack(
            np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        projection = build_projection_plan(
            normalized_points, order,
            regularization=float(static["projection_regularization"]),
            allow_fallback=True)
        xmins, xmaxs = [
            np.asarray(value)[np.asarray(all_ims)]
            for value in all_params["static"]["decomposition"]
            ["subdomain"]["params"][:2]]
        centers = .5*(xmins+xmaxs)
        scales = .5*(xmaxs-xmins)
        physical_points = (
            centers[:, None, :]
            + scales[:, None, :]*normalized_points[None, :, :])
        return takes, {
            "physical_points": jnp.asarray(physical_points),
            "sample_basis": projection["sample_basis"],
            "weights": projection["weights"],
            "coefficient_operator": projection["coefficient_operator"],
            "effective_order": projection["effective_order"],
            "requested_order": projection["requested_order"],
            "rank": projection["rank"],
            "condition_number": projection["condition_number"],
            "fallback": projection["fallback"],
        }

    @staticmethod
    def projection_components(all_params, x_batch, plan, local_fns):
        norm_fn, network_fn, _, window_fn = local_fns
        takes, projection = plan
        m_take, n_take, p_take, np_take, npou = takes
        params_take, axes = _take_model_params(all_params, m_take)
        x_take = x_batch[n_take]

        def raw_residual(params, point):
            raw = network_fn(params, norm_fn(params, point))
            output_unnorm = params["static"]["decomposition"][
                "subdomain"]["params"][5]
            return raw, raw*output_unnorm[1]

        raw, raw_values = vmap(raw_residual, in_axes=(axes, 0))(
            params_take, x_take)
        model_count = projection["physical_points"].shape[0]
        model_params, model_axes = _take_model_params(
            all_params, jnp.arange(model_count))

        def sample_model(params, points):
            return vmap(lambda point: raw_residual(params, point)[1])(points)

        sample_values = vmap(
            sample_model, in_axes=(model_axes, 0))(
                model_params, projection["physical_points"])
        coefficients = jnp.einsum(
            "bk,mkd->mbd", projection["coefficient_operator"], sample_values)
        query_normalized = vmap(
            lambda params, point: norm_fn(params, point), in_axes=(axes, 0))(
                params_take, x_take)
        basis_order = {1: 0, 3: 1, 6: 2}[
            projection["coefficient_operator"].shape[0]]
        query_basis = projection_polynomial_basis(
            query_normalized, basis_order)
        low_values = jnp.einsum(
            "pb,pbd->pd", query_basis, coefficients[m_take])
        projected_values = raw_values-low_values
        weights = vmap(window_fn, in_axes=(axes, 0))(params_take, x_take)
        weighted = projected_values*weights
        summed = jax.ops.segment_sum(
            jnp.concatenate([weighted, weights], axis=1), p_take,
            indices_are_sorted=False, num_segments=len(np_take))
        weight_sum = summed[:, -1:]
        residual = summed[:, :-1]/weight_sum
        residual = jax.ops.segment_sum(
            residual, np_take, indices_are_sorted=False,
            num_segments=len(x_batch))/npou
        projected_samples = (
            sample_values
            - jnp.einsum("kb,mbd->mkd",
                         projection["sample_basis"], coefficients))
        global_value = BSplineProjectedCosineResidualAssembly._global_value(
            all_params, x_batch)
        gate = all_params["static"]["assembly"]["residual_gate"]
        return {
            "global": global_value,
            "residual": gate*residual,
            "weight_sum": weight_sum,
            "raw_network": raw,
            "raw_local_values": raw_values,
            "projected_local_values": projected_values,
            "sample_values": sample_values,
            "projected_sample_values": projected_samples,
            "coefficients": coefficients,
            "projection": projection,
        }

    @classmethod
    def components(cls, all_params, x_batch, plan, local_fns):
        result = cls.projection_components(
            all_params, x_batch, plan, local_fns)
        return (result["global"], result["residual"], result["weight_sum"],
                result["raw_network"], result["projected_local_values"])

    @classmethod
    def assemble(cls, all_params, x_batch, plan, local_fns):
        global_value, residual, weight_sum, raw, local_values = cls.components(
            all_params, x_batch, plan, local_fns)
        return (global_value+residual, weight_sum, residual, weight_sum,
                raw, local_values)


_ASSEMBLY_REGISTRY = {
    "cosine_pou": CosinePOUAssembly,
    "quintic_pou": QuinticPOUAssembly,
    "hermite_1d_c1": Hermite1DC1Assembly,
    "bicubic_hermite_c1": BicubicHermiteC1Assembly,
    "c1_minimal_correction_pou": C1MinimalCorrectionPOUAssembly,
    "global_backbone_local_residual": GlobalBackboneLocalResidualAssembly,
    "backbone_first_cosine_residual": BackboneFirstCosineResidualAssembly,
    "backbone_first_p2_residual": BackboneFirstP2ResidualAssembly,
    "p2_fullfield": P2FullFieldAssembly,
    "bspline_cosine_residual": BSplineCosineResidualAssembly,
    "bspline_p2_residual": BSplineP2ResidualAssembly,
    "bspline_projected_cosine_residual": (
        BSplineProjectedCosineResidualAssembly),
}


def get_assembly(method):
    try:
        return _ASSEMBLY_REGISTRY[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown assembly_method {method!r}; expected one of {ASSEMBLY_METHODS}."
        ) from exc
