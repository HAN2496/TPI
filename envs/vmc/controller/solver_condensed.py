import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve, solve_discrete_are

from .solver import MPCOSQPSolver


class MPCCondensedOSQPSolver(MPCOSQPSolver):
    def __init__(self, *args, delta_u_constraint=False, delta_u_max=10.0, **kwargs):
        self.delta_u_constraint = bool(delta_u_constraint)
        self.delta_u_max = np.asarray(delta_u_max, dtype=np.float64).ravel()
        if self.delta_u_max.size == 0 or np.any(self.delta_u_max <= 0.0):
            raise ValueError("delta_u_max must be positive")

        super().__init__(*args, **kwargs)

        if np.any(np.isfinite(self.xmin)) or np.any(np.isfinite(self.xmax)):
            raise ValueError("Condensed MPC currently supports input constraints only")
        if self.delta_u_max.size == 1:
            self.delta_u_max = np.full(self.nu, self.delta_u_max.item(), dtype=np.float64)
        elif self.delta_u_max.size != self.nu:
            raise ValueError("delta_u_max must be scalar or have one value per input")

        self._last_controls = None
        self._previous_u = np.zeros(self.nu, dtype=np.float64)

    def setup_problem(self, Q, R, N, x_ref):
        self.N = int(N)
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        q_diag = np.diag(Q) if np.ndim(Q) == 2 else np.asarray(Q, dtype=np.float64)
        r_diag = np.diag(R) if np.ndim(R) == 2 else np.asarray(R, dtype=np.float64)

        self._build_prediction_matrices()
        P, q = self._build_condensed_cost(q_diag, r_diag)
        self._build_hessian_template(P.shape[0])
        P_upper = self._upper_matrix(P)

        control_count = self.N * self.nu
        input_matrix = sparse.eye(control_count, format="csc")
        input_lower = np.tile(self.umin, self.N)
        input_upper = np.tile(self.umax, self.N)

        if self.delta_u_constraint:
            difference = sparse.lil_matrix((control_count, control_count), dtype=np.float64)
            identity = sparse.eye(self.nu, format="lil")
            difference[:self.nu, :self.nu] = identity
            for index in range(1, self.N):
                current = slice(index * self.nu, (index + 1) * self.nu)
                previous = slice((index - 1) * self.nu, index * self.nu)
                difference[current, previous] = -identity
                difference[current, current] = identity
            self._difference_matrix = difference.tocsc()
            self.A_ = sparse.vstack([input_matrix, self._difference_matrix], format="csc")
            self.l_template = np.hstack([input_lower, -np.tile(self.delta_u_max, self.N)])
            self.u_template = np.hstack([input_upper, np.tile(self.delta_u_max, self.N)])
        else:
            self._difference_matrix = None
            self.A_ = input_matrix
            self.l_template = input_lower
            self.u_template = input_upper

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=P_upper,
            q=q,
            A=self.A_,
            l=self.l_template,
            u=self.u_template,
            warm_starting=True,
            verbose=False,
            max_iter=self.max_iter,
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
        )

        self.P_ = P_upper
        self.P_dense_ = P
        self.q_ = q
        self._lower = self.l_template.copy()
        self._upper = self.u_template.copy()
        self._last_controls = None
        self._previous_u.fill(0.0)
        self._cache_base_cost()

    def update_weights(self, q_diag, r_diag):
        P, q = self._build_condensed_cost(q_diag, r_diag)
        P_upper = self._upper_matrix(P)
        self.prob.update(Px=P_upper.data, q=q)
        self.P_ = P_upper
        self.P_dense_ = P
        self.q_ = q

    def set_base_weights(self, q_diag, r_diag):
        self.update_weights(q_diag, r_diag)
        self._cache_base_cost()

    def _cache_base_cost(self):
        self._base_P = self.P_.copy()
        self._base_P_dense = self.P_dense_.copy()
        self._base_q = self.q_.copy()
        self._base_linear_map = self._linear_map.copy()
        self._base_linear_offset = self._linear_offset.copy()
        self._base_fallback_gain = self._fallback_gain.copy()

    def reset(self):
        self.P_ = self._base_P.copy()
        self.P_dense_ = self._base_P_dense.copy()
        self.q_ = self._base_q.copy()
        self._linear_map = self._base_linear_map.copy()
        self._linear_offset = self._base_linear_offset.copy()
        self._fallback_gain = self._base_fallback_gain.copy()
        self._last_controls = None
        self._previous_u.fill(0.0)
        self._lower[:] = self.l_template
        self._upper[:] = self.u_template
        self.prob.update(Px=self.P_.data, q=self.q_, l=self._lower, u=self._upper)
        self.prob.warm_start(x=np.zeros(self.N * self.nu), y=np.zeros(len(self.l_template)))

    def solve(self, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        q = self._linear_map @ x0 + self._linear_offset
        self.prob.update(q=q)

        if self.delta_u_constraint:
            self._lower[:] = self.l_template
            self._upper[:] = self.u_template
            start = self.N * self.nu
            self._lower[start:start + self.nu] += self._previous_u
            self._upper[start:start + self.nu] += self._previous_u
            self.prob.update(l=self._lower, u=self._upper)

        if self._last_controls is not None:
            shifted = np.vstack([self._last_controls[1:], self._last_controls[-1:]])
            self.prob.warm_start(x=shifted.ravel())

        result = self.prob.solve()
        solved = result.info.status in ("solved", "solved inaccurate") and result.x is not None

        if solved:
            controls = np.asarray(result.x, dtype=np.float64).reshape(self.N, self.nu)
            control = controls[0].copy()
        else:
            control = np.clip(-self._fallback_gain @ (x0 - self.x_ref), self.umin, self.umax)
            controls = np.tile(np.asarray(control, dtype=np.float64), (self.N, 1))

        if self.delta_u_constraint:
            control = np.clip(
                control,
                self._previous_u - self.delta_u_max,
                self._previous_u + self.delta_u_max,
            )
            controls[0] = control

        control = np.clip(control, self.umin, self.umax)
        controls[0] = control
        self._last_controls = controls.copy()
        self._previous_u[:] = control

        prediction = {
            "solved": solved,
            "status": result.info.status,
            "iterations": result.info.iter,
            "fallback": not solved,
        }
        if self.test:
            prediction.update(self._prediction(x0, controls))

        return (float(control[0]) if self.nu == 1 else control), prediction

    def _build_prediction_matrices(self):
        control_count = self.N * self.nu
        self._phi = np.zeros(((self.N + 1) * self.nx, self.nx), dtype=np.float64)
        self._gamma = np.zeros(((self.N + 1) * self.nx, control_count), dtype=np.float64)
        self._phi[:self.nx] = np.eye(self.nx)

        for index in range(1, self.N + 1):
            current = slice(index * self.nx, (index + 1) * self.nx)
            previous = slice((index - 1) * self.nx, index * self.nx)
            control = slice((index - 1) * self.nu, index * self.nu)
            self._phi[current] = self.Ad @ self._phi[previous]
            self._gamma[current] = self.Ad @ self._gamma[previous]
            self._gamma[current, control] = self.Bd

    def _build_condensed_cost(self, q_diag, r_diag):
        q_diag = np.asarray(q_diag, dtype=np.float64).ravel() + 1e-12
        r_diag = np.asarray(r_diag, dtype=np.float64).ravel()
        Q = np.diag(q_diag)
        R = np.diag(r_diag)
        riccati = solve_discrete_are(self.Ad, self.Bd, Q, R)
        terminal = riccati if self.terminal_cost else np.zeros_like(Q)
        qbar = sparse.block_diag([Q] * self.N + [terminal], format="csc")
        rbar = sparse.kron(sparse.eye(self.N, format="csc"), R, format="csc")

        weighted_gamma = qbar @ self._gamma
        hessian = self._gamma.T @ weighted_gamma + rbar.toarray()
        hessian = 0.5 * (hessian + hessian.T)
        self._linear_map = 2.0 * self._gamma.T @ qbar @ self._phi
        reference = np.tile(self.x_ref, self.N + 1)
        self._linear_offset = -2.0 * self._gamma.T @ qbar @ reference
        self._fallback_gain = solve(
            self.Bd.T @ riccati @ self.Bd + R,
            self.Bd.T @ riccati @ self.Ad,
            assume_a="pos",
        )
        q = self._linear_map @ self.x0 + self._linear_offset
        return 2.0 * hessian, q

    def _build_hessian_template(self, size):
        self._hessian_template = sparse.triu(
            sparse.csc_matrix(np.ones((size, size), dtype=np.float64)),
            format="csc",
        )

    def _upper_matrix(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        template = self._hessian_template
        values = np.empty_like(template.data)
        for column in range(template.shape[1]):
            start, end = template.indptr[column:column + 2]
            rows = template.indices[start:end]
            values[start:end] = matrix[rows, column]
        return sparse.csc_matrix(
            (values, template.indices.copy(), template.indptr.copy()),
            shape=template.shape,
        )

    def _prediction(self, x0, controls):
        states = self._phi @ x0 + self._gamma @ controls.ravel()
        return {
            "times": np.arange(self.N + 1) * self.dt,
            "states": states.reshape(self.N + 1, self.nx),
            "controls": controls,
        }
