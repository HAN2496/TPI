import numpy as np

from ..base import ControllableEnv, Trajectory
from .configs import Search_config
from .controller import LQRController, MPCController, PController
from .env import ErideEnv
from .env.plant import ACCEL_KEYS, JERK_KEYS, STATE_KEYS

CHANNELS = tuple(STATE_KEYS) + tuple(ACCEL_KEYS) + tuple(JERK_KEYS) + ("u_eride", "env_reward")


def make_controller(params):
    kind = params["controller"]
    if kind == "p":
        return PController(kp=params["kp"])
    if kind == "lqr":
        return LQRController(base_q_diag=params["q_diag"])
    return MPCController(base_q_diag=params["q_diag"], control_dt=0.01, **Search_config().mpc_settings)


class VMCEnv(ControllableEnv):

    def rollout(self, params, n, seed):
        env = ErideEnv(make_controller(params), mode="pure", record_inner=True)
        taus = []
        for episode in range(n):
            episode_seed = seed + episode
            env.reset(seed=episode_seed)
            inners = []
            truncated = False
            while not truncated:
                _, _, _, truncated, info = env.step(np.zeros(0, dtype=np.float32))
                inners += info["inner_infos"]
            channels = {key: np.asarray([i["state"][key] for i in inners]) for key in CHANNELS[:-1]}
            channels["env_reward"] = np.asarray([i["reward"] for i in inners])
            taus.append(Trajectory(
                channels=channels, dt=float(env.config.dt_inner),
                meta={"seed": episode_seed, "params": dict(params),
                      "horizon": env.observe_step, "terminated": False}))
        return taus
