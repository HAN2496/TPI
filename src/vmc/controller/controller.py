import numpy as np

class BaseController:
    kind: str
    def __init__(self, u_limit=145.0):
        self.u_limit = float(u_limit)
        self._prev_output = None

    def __call__(self, obs_dict, state_dict):
        u = self.control(obs_dict, state_dict)
        # if abs(u) > self.u_limit:
        #     print(f"Warning: Control output {u:.2f} exceeds limit {self.u_limit}. Clipping applied.")
        return float(np.clip(u, -self.u_limit, self.u_limit))

    def control(self, obs_dict, state_dict):
        raise NotImplementedError

    def schedule(self, action):
        pass

    def reset(self, *args):
        return

    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def obs_dict_to_array(self, obs_dict):
        return np.array(list(obs_dict.values()), dtype=np.float32)

class EmptyController(BaseController):
    kind = 'empty'
    def __init__(self):
        super().__init__()

    def control(self, obs_dict, state_dict):
        return 0.0

class HumanController(BaseController):
    kind = 'human'
    def __init__(self, kp=400, v_ref=30/3.6):
        super().__init__()
        self.v_ref = v_ref
        self.kp = kp

    def control(self, obs_dict, state_dict):
        return (self.v_ref - obs_dict["dx_com"]) * self.kp

    def reset(self, is_random=False):
        if is_random:
            self.v_ref = np.random.uniform(20/3.6, 40/3.6)
            self.kp = np.random.uniform(300, 500)

class PController(BaseController):
    kind = 'p'
    def __init__(self, kp, param_mapper=None, u_limit=145.0):
        super().__init__(u_limit=u_limit)
        self.kp0 = float(kp)
        self.kp = float(kp)
        self.param_mapper = param_mapper

    def _params(self):
        return {"kp": self.kp}

    def control(self, obs_dict, state_dict):
        dtheta = state_dict["dtheta"]
        return -self.kp * dtheta

    def schedule(self, action):
        if self.param_mapper is not None:
            self.kp = float(self.param_mapper.update(action))

    def reset(self):
        self.kp = self.kp0
        if self.param_mapper is not None:
            self.param_mapper.reset()
        self._prev_output = None

    def set_params(self, params):
        self.kp = float(params)
        self.kp0 = self.kp

class LQRController(BaseController):
    kind = 'lqr'
    def __init__(self, base_q_diag, param_mapper=None, u_limit=145.0):
        super().__init__(u_limit=u_limit)
        self.base_q_diag_0 = np.asarray(base_q_diag, dtype=np.float64)
        self.q_diag = self.base_q_diag_0.copy()
        self.N = len(self.base_q_diag_0)
        self.param_mapper = param_mapper

        from src.controller.solver import LQRSolver
        self.lqr_helper = LQRSolver()
        self.K = self.lqr_helper.get_K(Q=np.diag(self.q_diag))

    def control(self, obs_dict, state_dict):
        """Compute LQR control: u = -K(x)"""
        x = self.obs_dict_to_array(state_dict).astype(np.float64)
        u = -float((self.K @ x).squeeze())
        return u

    def schedule(self, action):
        """Schedule Q matrix based on RL action"""
        if self.param_mapper is not None:
            q_diag = np.asarray(self.param_mapper.update(action), dtype=np.float64)
            self.q_diag = q_diag
            self.K = self.lqr_helper.get_K(Q=np.diag(self.q_diag))

    def reset(self):
        """Reset controller to initial state"""
        self.q_diag = self.base_q_diag_0.copy()
        self.K = self.lqr_helper.get_K(Q=np.diag(self.q_diag))
        if self.param_mapper is not None:
            self.param_mapper.reset()
        self._prev_output = None

    def get_params(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        if self.param_mapper is not None and hasattr(self.param_mapper, 'idxs'):
            params['scheduled_indices'] = self.param_mapper.idxs
        return params
    
    def set_params(self, params):
        self.q_diag = np.asarray(params, dtype=np.float64)
        self.base_q_diag_0 = self.q_diag.copy()
        self.K = self.lqr_helper.get_K(Q=np.diag(self.q_diag))

class MPCController(BaseController):
    kind = "mpc"
    def __init__(self, base_q_diag, N, solver_type, solver_options, param_mapper=None, u_limit=145.0, test=False):
        super().__init__(u_limit=u_limit)
        self.solver_options = dict(solver_options)
        self.base_q_diag_0  = np.asarray(base_q_diag, dtype=np.float64)
        self.q_diag         = self.base_q_diag_0.copy()
        self.r_diag         = np.array([0.01], dtype=np.float64)
        self.solver_type    = str(solver_type)
        self.param_mapper   = param_mapper
        self.u_limit        = float(u_limit)
        self.test           = bool(test)
        self.N              = int(N)

        solver_opts = self.solver_options.copy()

        from src.controller.solver import MPCOSQPSolver
        self.solver = MPCOSQPSolver(
            u_limit=u_limit,
            test=test,
            **solver_opts
        )

        self.nx = self.solver.nx
        self.nu = self.solver.nu

        self.x_ref = np.zeros(self.nx, dtype=np.float64)
        self.prob = None
        self.last_prediction = None
        self.setup_solver()

    def setup_solver(self):
        """Setup MPC solver with current Q and R matrices"""
        Q = np.diag(self.q_diag)
        R = np.diag(self.r_diag)
        self.solver.setup_problem(Q, R, self.N, self.x_ref)
        self.prob = self.solver.prob

    def control(self, obs_dict, state_dict):
        """Compute control action using MPC"""
        x = self.obs_dict_to_array(state_dict).astype(np.float64)
        u, prediction = self.solver.solve(x)
        self.last_prediction = prediction
        return float(u)

    def schedule(self, action):
        """Schedule Q matrix based on RL action"""
        if self.param_mapper is not None:
            q_diag = np.asarray(self.param_mapper.update(action), dtype=np.float64)
            self.q_diag = q_diag
            self.solver.update_weights(self.q_diag, self.r_diag)

    def reset(self):
        self.q_diag = self.base_q_diag_0.copy()
        self.setup_solver()
        if self.param_mapper is not None:
            self.param_mapper.reset()
        self._prev_output = None

    def get_params(self):
        exclude_keys = {'solver', 'prob', 'last_prediction'}
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k not in exclude_keys}
        if self.param_mapper is not None and hasattr(self.param_mapper, 'idxs'):
            params['scheduled_indices'] = self.param_mapper.idxs
        return params

    def set_params(self, params):
        self.q_diag = np.asarray(params, dtype=np.float64)
        self.base_q_diag_0 = self.q_diag.copy()
        self.setup_solver()

class DirectRLController(BaseController):
    kind = 'pure_rl'
    def __init__(self, param_mapper=None, u_limit=145.0):
        super().__init__(u_limit=u_limit)
        self.param_mapper = param_mapper
        self.current_u = 0.0

    def control(self, obs_dict, state_dict):
        return self.current_u

    def schedule(self, action):
        if self.param_mapper is not None:
            self.current_u = float(self.param_mapper.update(action))
        else:
            self.current_u = float(action[0])

    def reset(self):
        self.current_u = 0.0
        if self.param_mapper is not None:
            self.param_mapper.reset()