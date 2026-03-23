import torch
import numpy as np

class EnvReward:
    def __init__(self, base_reward_fn=None, learned_reward_fn=None, mix=None):
        self.base_reward_fn = base_reward_fn
        self.learned_reward_fn = learned_reward_fn
        self.mix = mix
        self.reward_info = {
            "reward_env": 0.0,
            "reward_model": 0.0,
            "reward_mixed": 0.0
        }
        if self.base_reward_fn is None and self.learned_reward_fn is None:
            self.base_reward_fn = create_weighted_reward()

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        r_env = 0.0
        r_model = 0.0

        if self.learned_reward_fn is not None: # For Fine-tune RL
            r_model = self.learned_reward_fn(state, state_ddot, state_dddot, u_eride)
            if self.mix is not None: # mix environment and learned reward
                r_env = self.base_reward_fn(state, state_ddot, state_dddot, u_eride)
                r = self.mix * r_env + (1.0 - self.mix) * r_model
            else:
                r = r_model
        else: # For Pre-train RL
            r_env = self.base_reward_fn(state, state_ddot, state_dddot, u_eride)
            r = r_env

        self.reward_info = {
            "reward_env": r_env,
            "reward_model": r_model,
            "reward_mixed": r
        }
        return r

class BaseReward:
    def __init__(self, w_pitch=1.0, w_accel=0.0, w_control=0.0, w_bounce=0.0, w_exp=0.0, w_exp_threshold=0.5):
        self.w_pitch = w_pitch
        self.w_accel = w_accel
        self.w_control = w_control
        self.w_bounce = w_bounce
        self.w_exp = w_exp
        self.w_exp_threshold = w_exp_threshold
        self.reward_info = {"reward_env": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        reward = 0.0

        if self.w_pitch != 0:
            pitch_rate = state["dtheta"]
            reward -= self.w_pitch * (pitch_rate) ** 2

        if self.w_accel != 0:
            ddx = state_ddot["ddx_com"]
            reward -= self.w_accel * (ddx) ** 2
        if self.w_control != 0:
            reward -= self.w_control * (u_eride) ** 2

        if self.w_bounce != 0:
            ddz = state_ddot["ddz_com"]
            reward -= self.w_bounce * (ddz) ** 2

        if self.w_exp != 0:
            pitch_acc = state_ddot["ddtheta"]
            reward -= self.w_exp * (np.exp(np.maximum(0, self.w_exp_threshold - np.abs(pitch_acc)))-1) * u_eride**2

        self.reward_info = {"reward_env": float(reward)}
        return float(reward)

def create_weighted_reward(w_pitch=1.0, w_accel=0.0, w_control=0.0, w_bounce=0.0, w_exp=0., w_exp_threshold=0.5):
    return BaseReward(w_pitch=w_pitch, w_accel=w_accel, w_control=w_control, w_bounce=w_bounce, w_exp=w_exp, w_exp_threshold=w_exp_threshold)

class LearnedReward:
    def __init__(self, reward_model, state_keys, base_reward_fn=None, device="cpu"):
        self.reward_model = reward_model
        self.state_keys = state_keys
        self.device = device
        self.reward_info = {"reward_model": 0.0}

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        full_state = {}
        full_state.update(state)
        full_state.update(state_ddot)
        full_state.update(state_dddot)
        full_state["u_eride"] = u_eride
        x = np.array([full_state[key] for key in self.state_keys], dtype=np.float32)

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            r_model = self.reward_model(x_tensor).detach().cpu().numpy().item()

        self.reward_info = {"reward_model": float(r_model)}
        return float(r_model)

class SparseLearnedReward(LearnedReward):
    def __init__(self, reward_model, state_keys, device="cpu"):
        super().__init__(reward_model, state_keys, device=device)
        self.episode_buffer = []

    def reset(self):
        self.episode_buffer = []

    def __call__(self, state, state_ddot, state_dddot, u_eride):
        # Collect data
        full_state = {}
        full_state.update(state)
        full_state.update(state_ddot)
        full_state.update(state_dddot)
        full_state["u_eride"] = u_eride
        x = np.array([full_state[key] for key in self.state_keys], dtype=np.float32)
        self.episode_buffer.append(x)
        
        return 0.0 # Return 0 for intermediate steps

    def get_episode_reward(self):
        if not self.episode_buffer:
            return 0.0
            
        x_seq = np.array(self.episode_buffer) # (L, F)
        x_tensor = torch.tensor(x_seq, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, L, F)
        lengths = torch.tensor([len(x_seq)], device=self.device)

        with torch.no_grad():
            # EpisodeLSTM returns (B,) -> (1,)
            # Pass lengths=lengths to support models that require it
            if "Episode" in self.reward_model.__class__.__name__:
                 r_model = self.reward_model(x_tensor, lengths=lengths).detach().cpu().numpy().item()
            else:
                 r_model = self.reward_model(x_tensor).detach().cpu().numpy().item()
            
        self.reward_info = {"reward_model": float(r_model)}
        return float(r_model)
