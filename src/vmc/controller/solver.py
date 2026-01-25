import numpy as np

import osqp
import scipy.linalg
from scipy import sparse
from scipy.linalg import solve_discrete_are

from src.vmc.env.components.vehicle_model import VehicleModel
from src.vmc.configs import Environment_Parameters

class LQRSolver:
    def __init__(self, vehicle=None, x0=None, ueq=None, zeq=None, dt=None):
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.x0 = x0           if x0      is not None else Environment_Parameters().x0
        self.dt = dt           if dt      is not None else Environment_Parameters().dt_inner
        self.ueq = ueq         if ueq     is not None else 0.0
        self.zeq = zeq         if zeq     is not None else np.zeros(2, dtype=np.float32)

        # Discrete-time system
        A, B = self.vehicle.get_ABmatrix(self.x0, self.ueq, self.zeq)
        self.Ad = A * self.dt + np.eye(A.shape[0])
        self.Bd = B * self.dt

    def get_K(self, Q=None, R=None):
        if Q is None:
            Q = np.diag([0, 8000, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float64)
        else:
            Q = np.asarray(Q, dtype=np.float64)
        if R is None:
            R = np.array([[0.01]], dtype=np.float64)
        return self.compute_lqr_gain(self.Ad, self.Bd, Q, R)

    @staticmethod
    def compute_lqr_gain(A, B, Q, R):
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        BtPB_R = B.T @ P @ B + R
        K = scipy.linalg.solve(BtPB_R, B.T @ P @ A, assume_a='pos')
        return K

"""
https://osqp.org/
참고링크: https://osqp.org/docs/examples/mpc.html
"""
class MPCOSQPSolver:
    def __init__(self, u_limit, test=False, vehicle=None, x0=None, ueq=None, zeq=None, dt=None,
                 max_iter=20000, eps_abs=0.01, eps_rel=0.01):
        self.u_limit = float(u_limit)
        self.test = bool(test)
        self.max_iter = int(max_iter)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)

        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.x0      = np.asarray(x0 if x0 is not None else Environment_Parameters().x0, dtype=np.float64)
        self.dt      = float(dt if dt is not None else Environment_Parameters().dt_inner)
        self.ueq     = float(ueq if ueq is not None else 0.0)
        self.zeq     = np.asarray(zeq if zeq is not None else np.zeros(2), dtype=np.float64)

        # Discrete-time system
        A, B = self.vehicle.get_ABmatrix(self.x0, self.ueq, self.zeq)
        self.Ad = np.asarray(A * self.dt + np.eye(A.shape[0]), dtype=np.float64)
        self.Bd = np.asarray(B * self.dt, dtype=np.float64)
        self.nx, self.nu = self.Bd.shape

        self.N = None
        self.x_ref = None
        self.prob = None

        # Constraints
        # constraints를 추가할 수 있지만 solve time으로 인해 무제한 (no constraints) 으로 설정
        self.umin = np.full(self.nu, -np.inf, dtype=np.float64)
        self.umax = np.full(self.nu, np.inf, dtype=np.float64)
        self.xmin = np.full(self.nx, -np.inf, dtype=np.float64)
        self.xmax = np.full(self.nx, np.inf, dtype=np.float64)

        # Cached sparse matrices
        self._sparse_cache = {}

    def update_weights(self, q_diag, r_diag):
        P, q = self._build_cost_matrices(q_diag, r_diag)

        # Update OSQP
        Px = sparse.triu(P).data
        self.prob.update(Px=Px, q=q)

        self.P_ = P
        self.q_ = q

    def setup_problem(self, Q, R, N, x_ref):
        """
        Full problem setup (called once at initialization).
        Builds constraint matrices and initializes cost matrices.
        """
        self.N = int(N)
        self.x_ref = np.asarray(x_ref, dtype=np.float64)

        Q = np.asarray(Q, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        q_diag = np.diag(Q) if Q.ndim == 2 else Q
        r_diag = np.diag(R) if R.ndim == 2 else R

        # Build constraint matrices
        self._build_constraint_matrices(N)

        # Build cost matrices using update_weights
        P, q = self._build_cost_matrices(q_diag, r_diag)

        # Create OSQP object
        self.prob = osqp.OSQP()
        self.prob.setup(P, q, self.A_, self.l_template, self.u_template,
                       warm_start=True, verbose=False,
                       max_iter=self.max_iter, eps_abs=self.eps_abs, eps_rel=self.eps_rel)


    def _build_cost_matrices(self, q_diag, r_diag):
        q_diag = np.asarray(q_diag, dtype=np.float64).ravel()
        r_diag = np.asarray(r_diag, dtype=np.float64).ravel()

        # Regularization
        q_diag_reg = q_diag + 1e-12

        # Stage cost matrices
        Q = sparse.diags(q_diag_reg, format='csc')
        R = sparse.diags(r_diag, format='csc')

        # Full matrices for DARE
        Q_full = np.diag(q_diag_reg)
        R_full = np.diag(r_diag)

        # Terminal cost
        P_terminal = solve_discrete_are(self.Ad, self.Bd, Q_full, R_full)
        QN = sparse.csc_matrix(P_terminal)

        # Cache kron(I_N, ·)
        if 'IN' not in self._sparse_cache:
            self._sparse_cache['IN'] = sparse.eye(self.N, format='csc')

        # Hessian P
        P = sparse.block_diag([
            sparse.kron(self._sparse_cache['IN'], Q),
            QN,
            sparse.kron(self._sparse_cache['IN'], R)
        ], format='csc') * 2.0

        # Linear term q
        q_vec = -2.0 * q_diag_reg * self.x_ref
        q_x  = np.tile(q_vec, self.N)
        q_xN = q_vec
        q_u  = np.zeros(self.N * self.nu, dtype=np.float64)
        q    = np.hstack([q_x, q_xN, q_u])

        return P, q

    def _build_constraint_matrices(self, N):
        """
        Build dynamics and bound constraint matrices (called once).
        """
        # Dynamics constraints: x_{k+1} = Ad*x_k + Bd*u_k
        Ax = sparse.kron(sparse.eye(N+1, format='csc'), -sparse.eye(self.nx, format='csc')) + \
             sparse.kron(sparse.eye(N+1, k=-1, format='csc'), sparse.csc_matrix(self.Ad))
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N, format='csc')]),
                        sparse.csc_matrix(self.Bd))
        Aeq = sparse.hstack([Ax, Bu], format='csc')

        # Bound constraints
        Aineq = sparse.eye((N+1)*self.nx + N*self.nu, format='csc')
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Constraint vectors (dtype float64)
        leq = np.hstack([-self.x0, np.zeros(N*self.nx, dtype=np.float64)])
        lineq = np.hstack([np.tile(self.xmin, N+1), np.tile(self.umin, N)])
        uineq = np.hstack([np.tile(self.xmax, N+1), np.tile(self.umax, N)])

        self.l_template = np.hstack([leq, lineq])
        self.u_template = np.hstack([leq, uineq])
        self.A_ = A

    def solve(self, x0):
        x0 = np.asarray(x0, dtype=np.float64)

        # Update initial state constraint
        l = self.l_template.copy()
        u = self.u_template.copy()
        l[:self.nx] = -x0
        u[:self.nx] = -x0

        # Solve QP
        self.prob.update(l=l, u=u)
        res = self.prob.solve()

        # Extract solution
        solved = res.info.status == 'solved'
        ctrl_start_idx = (self.N + 1) * self.nx
        u_opt = float(res.x[ctrl_start_idx]) if self.nu == 1 else res.x[ctrl_start_idx:ctrl_start_idx + self.nu]

        # Build prediction output
        prediction = {'solved': solved}
        if self.test:
            prediction.update(self._extract_prediction(res))

        return u_opt, prediction

    def _extract_prediction(self, res):
        pred_states = res.x[:(self.N + 1) * self.nx].reshape(self.N + 1, self.nx)
        pred_controls = res.x[(self.N + 1) * self.nx:].reshape(self.N, self.nu)
        pred_times = np.arange(self.N + 1) * self.dt

        return {
            'times': pred_times,
            'states': pred_states,
            'controls': pred_controls
        }