import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import (
    BicubicHermiteC1Assembly,
    CosinePOUAssembly,
    Hermite1DC1Assembly,
    bicubic_hermite,
    cubic_hermite,
    quintic_compact_window,
    quintic_smootherstep,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_trainers import FBPINN_model, get_inputs


jax.config.update("jax_enable_x64", True)


class AssemblyMathematicsTests(unittest.TestCase):
    def test_quintic_endpoint_derivatives_and_compact_support(self):
        scalar = lambda value: quintic_smootherstep(value)
        first = jax.grad(scalar)
        second = jax.grad(first)
        for endpoint, expected in ((0.0, 0.0), (1.0, 1.0)):
            self.assertAlmostEqual(float(scalar(endpoint)), expected, places=13)
            self.assertAlmostEqual(float(first(endpoint)), 0.0, places=13)
            self.assertAlmostEqual(float(second(endpoint)), 0.0, places=13)

        xmin = jnp.array([0.0])
        xmax = jnp.array([10.0])
        width = jnp.array([2.0])
        values = [
            float(quintic_compact_window(xmin, xmax, width, width,
                                         jnp.array([x]))[0])
            for x in (0.0, 2.0, 5.0, 8.0, 10.0)
        ]
        np.testing.assert_allclose(values, [0.0, 1.0, 1.0, 1.0, 0.0], atol=1e-13)

    def test_cubic_hermite_reproduces_linear_and_cubic_fields(self):
        x0, x1 = -1.25, 2.75
        length = x1 - x0
        samples = jnp.linspace(x0, x1, 31)

        for coefficients in ((2.0, -0.7, 0.0, 0.0),
                             (1.2, -0.4, 0.35, -0.08)):
            a, b, c, d = coefficients
            field = lambda x: a + b*x + c*x**2 + d*x**3
            slope = jax.grad(field)
            reconstructed = jax.vmap(
                lambda x: cubic_hermite(
                    field(x0), slope(x0), field(x1), slope(x1),
                    (x-x0)/length, length))(samples)
            np.testing.assert_allclose(reconstructed, jax.vmap(field)(samples),
                                       rtol=1e-12, atol=1e-12)

            assembled = lambda x: cubic_hermite(
                field(x0), slope(x0), field(x1), slope(x1),
                (x-x0)/length, length)
            self.assertAlmostEqual(float(assembled(x0)), float(field(x0)), places=12)
            self.assertAlmostEqual(float(assembled(x1)), float(field(x1)), places=12)
            self.assertAlmostEqual(float(jax.grad(assembled)(x0)),
                                   float(slope(x0)), places=12)
            self.assertAlmostEqual(float(jax.grad(assembled)(x1)),
                                   float(slope(x1)), places=12)

    def test_bicubic_reproduces_bicubic_polynomial_and_gradient(self):
        x0, x1 = -1.5, 2.25
        y0, y1 = 0.75, 3.5
        lx, ly = x1-x0, y1-y0

        def field(point):
            x, y = point
            return (0.8 - 0.3*x + 0.2*y + 0.15*x*y + 0.07*x**2
                    - 0.04*y**2 + 0.02*x**3*y - 0.01*x*y**3
                    + 0.005*x**3*y**3)

        gradient = jax.grad(field)
        hessian = jax.jacfwd(gradient)
        q = np.empty((2, 2), dtype=np.float64)
        qx = np.empty_like(q)
        qy = np.empty_like(q)
        qxy = np.empty_like(q)
        for ix, x in enumerate((x0, x1)):
            for iy, y in enumerate((y0, y1)):
                point = jnp.array([x, y])
                q[ix, iy] = field(point)
                grad = gradient(point)
                qx[ix, iy], qy[ix, iy] = grad[0], grad[1]
                qxy[ix, iy] = hessian(point)[0, 1]
        q, qx, qy, qxy = map(jnp.asarray, (q, qx, qy, qxy))

        def reconstructed(point):
            xi = (point[0]-x0)/lx
            eta = (point[1]-y0)/ly
            return bicubic_hermite(q, qx, qy, qxy, xi, eta, lx, ly)

        points = jnp.array([
            [x0, y0], [x1, y1], [0.0, 1.0], [1.4, 2.8], [-0.7, 2.1]
        ])
        np.testing.assert_allclose(jax.vmap(reconstructed)(points),
                                   jax.vmap(field)(points), rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(jax.vmap(jax.grad(reconstructed))(points),
                                   jax.vmap(gradient)(points), rtol=2e-10, atol=2e-10)


class HermiteGradientPathTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        centers = [np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0])]
        widths = [1.6*np.ones(4), np.array([2.0])]
        decomp_static, _ = RectangularDecompositionND.init_params(
            centers, widths, (np.zeros(2), np.ones(2)))
        assembly_static, _ = Hermite1DC1Assembly.init_params()
        coeffs = jnp.array([
            [[0.1, 0.7, -0.2], [0.3, -0.4, 0.1]],
            [[0.4, -0.2, 0.3], [-0.1, 0.6, -0.2]],
            [[-0.2, 0.5, 0.2], [0.2, -0.1, 0.4]],
            [[0.6, -0.3, -0.1], [0.5, 0.2, -0.3]],
        ], dtype=jnp.float64)
        all_params = {
            "static": {"decomposition": decomp_static, "assembly": assembly_static},
            "trainable": {"network": {"subdomain": {"coeffs": coeffs}}},
        }

        def norm_fn(params, x):
            return x

        def network_fn(params, x):
            local = params["trainable"]["network"]["subdomain"]["coeffs"]
            return local[:, 0] + local[:, 1]*x[0] + local[:, 2]*x[1]

        def unnorm_fn(params, value):
            return value

        def unused_window(params, x):
            return jnp.ones((1,), dtype=x.dtype)

        local_fns = (norm_fn, network_fn, unnorm_fn, unused_window)
        points = jnp.array([[0.35, -0.2], [1.35, 0.4], [2.35, 0.8]],
                           dtype=jnp.float64)
        all_ims = jnp.arange(4)
        plan = Hermite1DC1Assembly.build_plan(
            all_params, points, takes=None, all_ims=all_ims)
        return all_params, points, plan, local_fns

    def test_grad_jvp_and_vjp_cross_hermite_assembly(self):
        all_params, points, plan, local_fns = self._fixture()
        coefficients = all_params["trainable"]["network"]["subdomain"]["coeffs"]

        def with_coefficients(current_coeffs, current_points=points):
            params = {
                "static": all_params["static"],
                "trainable": {"network": {"subdomain": {
                    "coeffs": current_coeffs}}},
            }
            return Hermite1DC1Assembly.assemble(
                params, current_points, plan, local_fns)[0]

        gradient = jax.grad(lambda c: jnp.sum(with_coefficients(c)))(coefficients)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        per_model = jnp.sum(jnp.abs(gradient), axis=(1, 2))
        self.assertTrue(bool(jnp.all(per_model > 0)))

        _, tangent = jax.jvp(
            lambda x: with_coefficients(coefficients, x),
            (points,), (jnp.ones_like(points)*0.1,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(tangent))))

        values, pullback = jax.vjp(with_coefficients, coefficients)
        cotangent = pullback(jnp.ones_like(values))[0]
        self.assertTrue(bool(jnp.all(jnp.isfinite(cotangent))))
        self.assertGreater(float(jnp.linalg.norm(cotangent)), 0.0)

    def test_actual_chain_matches_each_transition_endpoint_value_and_slope(self):
        all_params, _, _, local_fns = self._fixture()
        decomposition = all_params["static"]["decomposition"]
        xmins, xmaxs = decomposition["subdomain"]["params"][:2]
        transition_left = np.asarray(xmins)[1:, 0]
        transition_right = np.asarray(xmaxs)[:-1, 0]
        coefficients = all_params["trainable"]["network"]["subdomain"]["coeffs"]

        for interface, (left_x, right_x) in enumerate(
                zip(transition_left, transition_right)):
            for boundary, x_value, owner in (
                    ("left", left_x, interface),
                    ("right", right_x, interface + 1)):
                with self.subTest(interface=interface, boundary=boundary):
                    point = jnp.array([x_value, 0.37], dtype=jnp.float64)
                    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
                        point[None, :], jnp.zeros(4, dtype=int), all_params,
                        RectangularDecompositionND, Hermite1DC1Assembly)
                    params = {
                        "static": cut_all(all_params["static"]),
                        "trainable": cut_active(all_params["trainable"]),
                    }

                    def assembled(current_point):
                        return Hermite1DC1Assembly.assemble(
                            params, current_point[None, :], plan, local_fns)[0][0]

                    value = assembled(point)
                    slope = jax.jacfwd(assembled)(point)[:, 0]
                    local = coefficients[owner]
                    expected_value = (
                        local[:, 0] + local[:, 1]*point[0] + local[:, 2]*point[1])
                    expected_slope = local[:, 1]
                    np.testing.assert_allclose(
                        value, expected_value, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(
                        slope, expected_slope, rtol=1e-12, atol=1e-12)


class CosineRegressionTests(unittest.TestCase):
    def test_cosine_adapter_matches_direct_normalized_pou(self):
        centers = [np.array([0.0, 1.0]), np.array([0.0])]
        widths = [np.array([1.6, 1.6]), np.array([2.0])]
        decomp_static, _ = RectangularDecompositionND.init_params(
            centers, widths, (np.zeros(2), np.ones(2)))
        assembly_static, _ = CosinePOUAssembly.init_params()
        coefficients = jnp.array([
            [[0.2, 0.7, 0.1], [-0.3, 0.2, 0.4]],
            [[1.1, -0.5, 0.2], [0.8, -0.1, -0.2]],
        ])
        all_params = {
            "static": {"decomposition": decomp_static, "assembly": assembly_static},
            "trainable": {"network": {"subdomain": {
                "coeffs": coefficients}}},
        }

        def norm_fn(params, x):
            return x

        def network_fn(params, x):
            local = params["trainable"]["network"]["subdomain"]["coeffs"]
            return local[:, 0] + local[:, 1]*x[0] + local[:, 2]*x[1]

        def unnorm_fn(params, value):
            return value

        model_fns = (
            norm_fn, network_fn, unnorm_fn,
            RectangularDecompositionND.window_fn, CosinePOUAssembly)
        points = jnp.array([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0]])
        plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
            points, jnp.zeros(2, dtype=int), all_params,
            RectangularDecompositionND, CosinePOUAssembly)
        cut_params = {
            "static": cut_all(all_params["static"]),
            "trainable": cut_active(all_params["trainable"]),
        }
        actual = FBPINN_model(cut_params, points, plan, model_fns)[0]

        expected = []
        xmins, xmaxs = decomp_static["subdomain"]["params"][:2]
        for point in points:
            numerator = jnp.zeros(2)
            denominator = 0.0
            for model_id in range(2):
                if bool(jnp.all((point >= xmins[model_id]) &
                                (point <= xmaxs[model_id]))):
                    params = {
                        "static": {"decomposition": {
                            **decomp_static,
                            "subdomain": {
                                **decomp_static["subdomain"],
                                "params": jax.tree_util.tree_map(
                                    lambda value: value[model_id],
                                    decomp_static["subdomain"]["params"]),
                            },
                        }},
                        "trainable": {"network": {"subdomain": {
                            "coeffs": coefficients[model_id]}}},
                    }
                    value = network_fn(params, point)
                    weight = RectangularDecompositionND.window_fn(params, point)[0]
                    numerator = numerator + weight*value
                    denominator = denominator + weight
            expected.append(numerator/denominator)
        np.testing.assert_allclose(actual, jnp.stack(expected), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
