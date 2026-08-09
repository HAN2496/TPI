import numpy as np


STATE_KEYS = [
    "dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com",
    "z_com", "theta", "z_us_f", "z_us_r", "x_com",
]


class BaseController:
    kind = "base"

    def __init__(self, u_limit=145.0, control_dt=0.001):
        self.u_limit = float(u_limit)
        self.control_dt = float(control_dt)
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")

    def __call__(self, obs_dict, state_dict, state_vector=None):
        control = self.control(obs_dict, state_dict, state_vector)
        return float(np.clip(control, -self.u_limit, self.u_limit))

    def control(self, obs_dict, state_dict, state_vector=None):
        raise NotImplementedError

    def schedule(self, action):
        return

    def reset(self, *args):
        return

    def get_params(self):
        return {}

    def state_array(self, state_dict, state_vector=None):
        if state_vector is not None:
            return np.asarray(state_vector, dtype=np.float64)
        return np.fromiter((state_dict[key] for key in STATE_KEYS), dtype=np.float64, count=len(STATE_KEYS))


class EmptyController(BaseController):
    kind = "empty"

    def control(self, obs_dict, state_dict, state_vector=None):
        return 0.0


class HumanController(BaseController):
    kind = "human"

    def __init__(self, kp=400, v_ref=30 / 3.6):
        super().__init__()
        self.v_ref = float(v_ref)
        self.kp = float(kp)

    def control(self, obs_dict, state_dict, state_vector=None):
        velocity = state_vector[4] if state_vector is not None else state_dict["dx_com"]
        return (self.v_ref - velocity) * self.kp

    def reset(self, is_random=False):
        if is_random:
            self.v_ref = np.random.uniform(20 / 3.6, 40 / 3.6)
            self.kp = np.random.uniform(300, 500)

    def get_params(self):
        return {"kp": self.kp, "v_ref": self.v_ref}


class PController(BaseController):
    kind = "p"

    def __init__(self, kp, param_mapper=None, u_limit=145.0, control_dt=0.001):
        super().__init__(u_limit, control_dt)
        self.kp0 = float(kp)
        self.kp = float(kp)
        self.param_mapper = param_mapper

    def control(self, obs_dict, state_dict, state_vector=None):
        pitch_rate = state_vector[1] if state_vector is not None else state_dict["dtheta"]
        return -self.kp * pitch_rate

    def schedule(self, action):
        if self.param_mapper is not None:
            self.kp = float(self.param_mapper.update(action))

    def reset(self):
        self.kp = self.kp0
        if self.param_mapper is not None:
            self.param_mapper.reset()

    def get_params(self):
        return {"kp": self.kp}

    def set_params(self, params):
        self.kp = float(params)
        self.kp0 = self.kp


class LQRController(BaseController):
    kind = "lqr"

    def __init__(self, base_q_diag, param_mapper=None, u_limit=145.0, use_v_ref=False, control_dt=0.001):
        super().__init__(u_limit, control_dt)
        self.base_q_diag_0 = np.asarray(base_q_diag, dtype=np.float64).copy()
        self.q_diag = self.base_q_diag_0.copy()
        self.N = len(self.q_diag)
        self.param_mapper = param_mapper
        self.use_v_ref = bool(use_v_ref)
        self.v_ref = None

        from .solver import LQRSolver
        self.lqr_helper = LQRSolver(dt=self.control_dt)
        self.K0 = self.lqr_helper.get_K(Q=np.diag(self.q_diag))
        self.K = self.K0.copy()

    def control(self, obs_dict, state_dict, state_vector=None):
        state = self.state_array(state_dict, state_vector)
        if self.use_v_ref:
            state = state.copy()
            if self.v_ref is None:
                self.v_ref = state[4]
            state[4] -= self.v_ref
        return -float((self.K @ state).squeeze())

    def schedule(self, action):
        if self.param_mapper is None:
            return
        q_diag = np.asarray(self.param_mapper.update(action), dtype=np.float64)
        if np.array_equal(q_diag, self.q_diag):
            return
        self.q_diag = q_diag.copy()
        self.K = self.lqr_helper.get_K(Q=np.diag(self.q_diag))

    def reset(self):
        self.q_diag = self.base_q_diag_0.copy()
        self.K = self.K0.copy()
        self.v_ref = None
        if self.param_mapper is not None:
            self.param_mapper.reset()

    def get_params(self):
        return {"q_diag": self.q_diag.copy()}

    def set_params(self, params):
        self.q_diag = np.asarray(params, dtype=np.float64).copy()
        self.base_q_diag_0 = self.q_diag.copy()
        self.K0 = self.lqr_helper.get_K(Q=np.diag(self.q_diag))
        self.K = self.K0.copy()


class MPCController(BaseController):
    kind = "mpc"

    def __init__(self, base_q_diag, N, solver_type, solver_options, param_mapper=None, u_limit=145.0,
                 test=False, constraints=None, use_v_ref=False, control_dt=0.001):
        super().__init__(u_limit, control_dt)
        self.solver_options = dict(solver_options)
        configured_dt = float(self.solver_options.pop("dt", self.control_dt))
        if not np.isclose(configured_dt, self.control_dt):
            raise ValueError("solver_options.dt must match control_dt")
        self.solver_options["dt"] = self.control_dt

        self.base_q_diag_0 = np.asarray(base_q_diag, dtype=np.float64).copy()
        self.q_diag = self.base_q_diag_0.copy()
        self.r_diag = np.array([0.01], dtype=np.float64)
        self.solver_type = str(solver_type)
        self.param_mapper = param_mapper
        self.test = bool(test)
        self.N = int(N)
        self.constraints = constraints
        self.use_v_ref = bool(use_v_ref)
        self.v_ref = None

        if self.solver_type == "osqp":
            from .solver import MPCOSQPSolver
            solver_class = MPCOSQPSolver
        elif self.solver_type == "osqp_condensed":
            from .solver_condensed import MPCCondensedOSQPSolver
            solver_class = MPCCondensedOSQPSolver
        else:
            raise ValueError(f"Unknown MPC solver type: {self.solver_type}")

        self.solver = solver_class(
            u_limit=self.u_limit,
            test=self.test,
            constraints=self.constraints,
            **self.solver_options,
        )
        self.nx = self.solver.nx
        self.nu = self.solver.nu
        self.x_ref = np.zeros(self.nx, dtype=np.float64)
        self.last_prediction = None
        self.prob = None
        self.setup_solver()

    def setup_solver(self):
        self.solver.setup_problem(np.diag(self.q_diag), np.diag(self.r_diag), self.N, self.x_ref)
        self.prob = self.solver.prob

    def control(self, obs_dict, state_dict, state_vector=None):
        state = self.state_array(state_dict, state_vector)
        if self.use_v_ref:
            state = state.copy()
            if self.v_ref is None:
                self.v_ref = state[4]
            state[4] -= self.v_ref
        control, prediction = self.solver.solve(state)
        self.last_prediction = prediction
        return float(control)

    def schedule(self, action):
        if self.param_mapper is None:
            return
        q_diag = np.asarray(self.param_mapper.update(action), dtype=np.float64)
        if np.array_equal(q_diag, self.q_diag):
            return
        self.q_diag = q_diag.copy()
        self.solver.update_weights(self.q_diag, self.r_diag)

    def reset(self):
        self.q_diag = self.base_q_diag_0.copy()
        self.v_ref = None
        self.last_prediction = None
        if self.param_mapper is not None:
            self.param_mapper.reset()
        self.solver.reset()

    def get_params(self):
        return {"q_diag": self.q_diag.copy()}

    def set_params(self, params):
        self.q_diag = np.asarray(params, dtype=np.float64).copy()
        self.base_q_diag_0 = self.q_diag.copy()
        self.solver.set_base_weights(self.q_diag, self.r_diag)
        self.solver.reset()


class DirectRLController(BaseController):
    kind = "pure_rl"

    def __init__(self, param_mapper=None, u_limit=145.0, control_dt=0.001):
        super().__init__(u_limit, control_dt)
        self.param_mapper = param_mapper
        self.current_u = 0.0

    def control(self, obs_dict, state_dict, state_vector=None):
        return self.current_u

    def schedule(self, action):
        if self.param_mapper is not None:
            self.current_u = float(self.param_mapper.update(action))
        else:
            self.current_u = float(np.asarray(action).reshape(-1)[0])

    def reset(self):
        self.current_u = 0.0
        if self.param_mapper is not None:
            self.param_mapper.reset()

    def get_params(self):
        return {"u": self.current_u}
