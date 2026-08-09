import numpy as np
import osqp
import scipy.linalg
from scipy import sparse
from scipy.linalg import solve_discrete_are

from ..env.components.vehicle_model import VehicleModel
from ..configs import Environment_Parameters


STATE_KEYS = [
    "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com",
    "z_com", "theta", "z_us_f", "z_us_r", "x_com",
]


def discretize_system(A, B, dt):
    nx, nu = B.shape
    matrix = np.zeros((nx + nu, nx + nu), dtype=np.float64)
    matrix[:nx, :nx] = A
    matrix[:nx, nx:] = B
    discrete = scipy.linalg.expm(matrix * dt)
    return discrete[:nx, :nx], discrete[:nx, nx:]


class LQRSolver:
    def __init__(self, vehicle=None, x0=None, ueq=None, zeq=None, dt=None):
        self.vehicle = VehicleModel() if vehicle is None else vehicle
        self.x0 = np.asarray(Environment_Parameters().x0 if x0 is None else x0, dtype=np.float64)
        self.dt = float(Environment_Parameters().dt_inner if dt is None else dt)
        self.ueq = float(0.0 if ueq is None else ueq)
        self.zeq = np.asarray(np.zeros(2) if zeq is None else zeq, dtype=np.float64)
        A, B = self.vehicle.get_ABmatrix(self.x0, self.ueq, self.zeq)
        self.Ad, self.Bd = discretize_system(A, B, self.dt)

    def get_K(self, Q=None, R=None):
        if Q is None:
            Q = np.diag([0, 8000, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float64)
        else:
            Q = np.asarray(Q, dtype=np.float64)
        R = np.array([[0.01]], dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
        return self.compute_lqr_gain(self.Ad, self.Bd, Q, R)

    @staticmethod
    def compute_lqr_gain(A, B, Q, R):
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        return scipy.linalg.solve(B.T @ P @ B + R, B.T @ P @ A, assume_a="pos")


class MPCOSQPSolver:
    def __init__(self, u_limit, test=False, vehicle=None, x0=None, ueq=None, zeq=None, dt=None,
                 max_iter=20000, eps_abs=0.01, eps_rel=0.01, constraints=None, terminal_cost=True):
        self.u_limit = float(u_limit)
        self.test = bool(test)
        self.max_iter = int(max_iter)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.constraints = constraints
        self.vehicle = VehicleModel() if vehicle is None else vehicle
        self.x0 = np.asarray(Environment_Parameters().x0 if x0 is None else x0, dtype=np.float64)
        self.dt = float(Environment_Parameters().dt_inner if dt is None else dt)
        self.ueq = float(0.0 if ueq is None else ueq)
        self.zeq = np.asarray(np.zeros(2) if zeq is None else zeq, dtype=np.float64)
        self.terminal_cost = bool(terminal_cost)

        A, B = self.vehicle.get_ABmatrix(self.x0, self.ueq, self.zeq)
        self.Ad, self.Bd = discretize_system(A, B, self.dt)
        self.nx, self.nu = self.Bd.shape

        self.N = None
        self.x_ref = None
        self.prob = None
        self.umin = np.full(self.nu, -self.u_limit, dtype=np.float64)
        self.umax = np.full(self.nu, self.u_limit, dtype=np.float64)
        self.xmin = np.full(self.nx, -np.inf, dtype=np.float64)
        self.xmax = np.full(self.nx, np.inf, dtype=np.float64)
        self._apply_constraints(constraints)
        self._sparse_cache = {}
        self._previous_u = np.zeros(self.nu, dtype=np.float64)

    @staticmethod
    def _bound_pair(value, name):
        if isinstance(value, dict):
            low = value.get("min", value.get("lo", value.get("lower", -np.inf)))
            high = value.get("max", value.get("hi", value.get("upper", np.inf)))
        else:
            if len(value) != 2:
                raise ValueError(f"{name} constraint must be [min, max]")
            low, high = value
        low = -np.inf if low is None else float(low)
        high = np.inf if high is None else float(high)
        if low > high:
            raise ValueError(f"{name} constraint lower bound exceeds upper bound")
        return low, high

    @staticmethod
    def _state_index(key):
        if isinstance(key, int):
            index = key
        elif isinstance(key, str) and key.lstrip("-").isdigit():
            index = int(key)
        else:
            if key not in STATE_KEYS:
                raise KeyError(f"Unknown MPC state constraint key: {key}")
            index = STATE_KEYS.index(key)
        if index < 0 or index >= len(STATE_KEYS):
            raise IndexError(f"MPC state constraint index out of range: {index}")
        return index

    def _apply_constraints(self, constraints):
        if constraints is None:
            return
        if not isinstance(constraints, dict):
            raise TypeError("MPC constraints must be a mapping")
        if constraints.get("enabled", True) is False:
            return

        input_bounds = constraints.get("u", constraints.get("input"))
        if input_bounds is not None:
            self.umin[:], self.umax[:] = self._bound_pair(input_bounds, "u")
        if "u_min" in constraints:
            self.umin[:] = float(constraints["u_min"])
        if "u_max" in constraints:
            self.umax[:] = float(constraints["u_max"])

        state_bounds = constraints.get("x", constraints.get("state"))
        if state_bounds is not None:
            if isinstance(state_bounds, dict):
                for key, bounds in state_bounds.items():
                    index = self._state_index(key)
                    self.xmin[index], self.xmax[index] = self._bound_pair(bounds, f"x[{key}]")
            else:
                if len(state_bounds) != self.nx:
                    raise ValueError(f"x constraints must provide {self.nx} [min, max] pairs")
                for index, bounds in enumerate(state_bounds):
                    self.xmin[index], self.xmax[index] = self._bound_pair(bounds, f"x[{index}]")

        if "x_min" in constraints:
            values = np.asarray(constraints["x_min"], dtype=np.float64)
            if values.size != self.nx:
                raise ValueError(f"x_min must have length {self.nx}")
            self.xmin[:] = values
        if "x_max" in constraints:
            values = np.asarray(constraints["x_max"], dtype=np.float64)
            if values.size != self.nx:
                raise ValueError(f"x_max must have length {self.nx}")
            self.xmax[:] = values

    def setup_problem(self, Q, R, N, x_ref):
        self.N = int(N)
        self.x_ref = np.asarray(x_ref, dtype=np.float64)
        q_diag = np.diag(Q) if np.ndim(Q) == 2 else np.asarray(Q, dtype=np.float64)
        r_diag = np.diag(R) if np.ndim(R) == 2 else np.asarray(R, dtype=np.float64)

        self._build_constraint_matrices()
        P, q = self._build_cost_matrices(q_diag, r_diag)
        self._build_hessian_template(P)
        P_upper = self._upper_matrix(P)

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
        self.q_ = q
        self._base_P = P_upper.copy()
        self._base_q = q.copy()
        self._lower = self.l_template.copy()
        self._upper = self.u_template.copy()
        self._previous_u.fill(0.0)

    def update_weights(self, q_diag, r_diag):
        P, q = self._build_cost_matrices(q_diag, r_diag)
        P_upper = self._upper_matrix(P)
        self.prob.update(Px=P_upper.data, q=q)
        self.P_ = P_upper
        self.q_ = q

    def set_base_weights(self, q_diag, r_diag):
        self.update_weights(q_diag, r_diag)
        self._base_P = self.P_.copy()
        self._base_q = self.q_.copy()

    def reset(self):
        self.P_ = self._base_P.copy()
        self.q_ = self._base_q.copy()
        self._previous_u.fill(0.0)
        self._lower[:] = self.l_template
        self._upper[:] = self.u_template
        self.prob.update(Px=self.P_.data, q=self.q_, l=self._lower, u=self._upper)
        variables = (self.N + 1) * self.nx + self.N * self.nu
        self.prob.warm_start(x=np.zeros(variables), y=np.zeros(len(self.l_template)))

    def _build_cost_matrices(self, q_diag, r_diag):
        q_diag = np.asarray(q_diag, dtype=np.float64).ravel() + 1e-12
        r_diag = np.asarray(r_diag, dtype=np.float64).ravel()
        Q = sparse.diags(q_diag, format="csc")
        R = sparse.diags(r_diag, format="csc")
        Q_full = np.diag(q_diag)
        R_full = np.diag(r_diag)
        terminal = solve_discrete_are(self.Ad, self.Bd, Q_full, R_full) if self.terminal_cost else np.zeros_like(Q_full)

        if "identity_horizon" not in self._sparse_cache:
            self._sparse_cache["identity_horizon"] = sparse.eye(self.N, format="csc")

        horizon_eye = self._sparse_cache["identity_horizon"]
        P = sparse.block_diag([
            sparse.kron(horizon_eye, Q),
            sparse.csc_matrix(terminal),
            sparse.kron(horizon_eye, R),
        ], format="csc") * 2.0

        state_linear = -2.0 * q_diag * self.x_ref
        q = np.hstack([
            np.tile(state_linear, self.N),
            -2.0 * terminal @ self.x_ref,
            np.zeros(self.N * self.nu, dtype=np.float64),
        ])
        return P, q

    def _build_hessian_template(self, P):
        template = sparse.triu(P, format="csc")
        offset = self.N * self.nx
        rows, columns = np.triu_indices(self.nx)
        terminal = sparse.csc_matrix(
            (np.ones(len(rows)), (offset + rows, offset + columns)),
            shape=P.shape,
        )
        template = (template + terminal).tocsc()
        template.data[:] = 1.0
        self._hessian_template = template

    def _upper_matrix(self, P):
        template = self._hessian_template
        values = np.empty_like(template.data)
        for column in range(template.shape[1]):
            start, end = template.indptr[column:column + 2]
            rows = template.indices[start:end]
            values[start:end] = np.asarray(P[rows, column].todense()).ravel()
        return sparse.csc_matrix(
            (values, template.indices.copy(), template.indptr.copy()),
            shape=template.shape,
        )

    def _build_constraint_matrices(self):
        N = self.N
        state_eye = sparse.eye(self.nx, format="csc")
        Ax = sparse.kron(sparse.eye(N + 1, format="csc"), -state_eye)
        Ax += sparse.kron(sparse.eye(N + 1, k=-1, format="csc"), sparse.csc_matrix(self.Ad))
        Bu = sparse.kron(
            sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N, format="csc")]),
            sparse.csc_matrix(self.Bd),
        )
        dynamics = sparse.hstack([Ax, Bu], format="csc")
        bounds = sparse.eye((N + 1) * self.nx + N * self.nu, format="csc")
        self.A_ = sparse.vstack([dynamics, bounds], format="csc")

        equality = np.hstack([-self.x0, np.zeros(N * self.nx, dtype=np.float64)])
        lower_bounds = np.hstack([np.tile(self.xmin, N + 1), np.tile(self.umin, N)])
        upper_bounds = np.hstack([np.tile(self.xmax, N + 1), np.tile(self.umax, N)])
        self.l_template = np.hstack([equality, lower_bounds])
        self.u_template = np.hstack([equality, upper_bounds])

    def solve(self, x0):
        x0 = np.asarray(x0, dtype=np.float64)
        self._lower[:self.nx] = -x0
        self._upper[:self.nx] = -x0
        self.prob.update(l=self._lower, u=self._upper)
        result = self.prob.solve()

        solved = result.info.status in ("solved", "solved inaccurate") and result.x is not None
        control_start = (self.N + 1) * self.nx

        if solved:
            control = np.asarray(result.x[control_start:control_start + self.nu], dtype=np.float64)
        else:
            control = self._previous_u.copy()

        control = np.clip(control, self.umin, self.umax)
        self._previous_u[:] = control

        prediction = {
            "solved": solved,
            "status": result.info.status,
            "iterations": result.info.iter,
            "fallback": not solved,
        }
        if self.test and solved:
            prediction.update(self._extract_prediction(result.x))

        return (float(control[0]) if self.nu == 1 else control), prediction

    def _extract_prediction(self, solution):
        state_end = (self.N + 1) * self.nx
        return {
            "times": np.arange(self.N + 1) * self.dt,
            "states": solution[:state_end].reshape(self.N + 1, self.nx),
            "controls": solution[state_end:].reshape(self.N, self.nu),
        }
