import numpy as np
from typing import Dict, List, Union
from .components import Term, PitchTerm, LongitudinalTerm, DdxThresholdTerm, DthetaThresholdTerm

class Oracle:
    def __init__(self,
                 terms: List[Term],
                 weights: Dict[str, float]):
        self.terms = terms
        self.weights = weights
        # No normalization of weights to preserve absolute magnitude importance

    def get_step_reward_components(self, state):
        """
        Compute reward components for a single step or vectorized states.
        state: Dict[str, float] or Dict[str, np.ndarray]
        """
        components = {}
        for term in self.terms:
            value = term(state)
            weight = self.weights[term.name]
            components[term.name] = -weight * value
        return components

    def get_multi_step_components(self, recorder_states: Dict[str, Union[List, np.ndarray]]):
        """
        Vectorized computation of reward components over multiple steps.
        recorder_states: Dictionary of sequences (lists or arrays).
        """
        # Convert lists to numpy arrays if necessary for vectorization
        states_arr = {}
        n_steps = 0
        for k, v in recorder_states.items():
            arr = np.asarray(v)
            states_arr[k] = arr
            n_steps = len(arr)

        components = {}
        components_total = np.zeros(n_steps)

        for term in self.terms:
            # Term.__call__ is assumed to handle array inputs (vectorized)
            values = term(states_arr) # Shape: (T,)
            weight = self.weights[term.name]
            
            weighted_values = -weight * values
            components[term.name] = weighted_values
            components_total += weighted_values

        return components, components_total

    def calculate_episode_reward(self, trajectory: Dict[str, Union[List, np.ndarray]]) -> float:
        """
        Efficiently calculate the mean reward for an entire episode (trajectory).
        """
        _, step_rewards = self.get_multi_step_components(trajectory)
        return float(np.mean(step_rewards))

    def step_reward(self, state):
        components = self.get_step_reward_components(state)
        return sum(components.values())

    def diff_to_probability(self, r1, r2):
        return r1 / (np.exp(r1) + np.exp(r2))

    def sample_response(self, probability):
        return 1 if np.random.random() < probability else 0

    def compare(self, traj1: Dict[str, np.ndarray], traj2: Dict[str, np.ndarray]) -> int:
        """
        Compare two trajectories and return the label (0 or 1).
        1: traj1 is better (preferred)
        0: traj2 is better
        """
        r1 = self.calculate_episode_reward(traj1)
        r2 = self.calculate_episode_reward(traj2)
        
        prob_1_wins = self.diff_to_probability(r1, r2)
        return self.sample_response(prob_1_wins)

    def __call__(self, state_dict):
        """
        Calculate and return the reward for a given state or trajectory.
        Does NOT return probability or binary label for single input.
        """
        # Determine if input is single step or trajectory based on type of values
        is_trajectory = isinstance(next(iter(state_dict.values())), (list, np.ndarray)) and \
                        np.asarray(next(iter(state_dict.values()))).ndim > 0
        
        if is_trajectory:
            total_reward = self.calculate_episode_reward(state_dict)
        else:
            total_reward = self.step_reward(state_dict)

        return total_reward

    def get_weight_info(self):
        return self.weights.copy()


class OracleBuilder:
    def __init__(self):
        self.terms = []
        self.weights = {}
        self.dtheta_scale = 5.0
        self.ddx_scale = 1.78

    def add_pitch(self, weight=1.0, scale=None):
        if scale is None:
            scale = self.dtheta_scale
        self.terms.append(PitchTerm())
        self.weights["pitch"] = weight * scale
        return self

    def add_longitudinal(self, weight=1.0, scale=None):
        if scale is None:
            scale = self.ddx_scale
        self.terms.append(LongitudinalTerm())
        self.weights["longitudinal"] = weight * scale
        return self
    
    def add_ddx_threshold(self, weight, threshold, exp_scale, scale=None):
        if scale is None:
            scale = self.ddx_scale
        self.terms.append(DdxThresholdTerm(threshold, exp_scale))
        self.weights["ddx_threshold"] = weight * scale
        return self

    def add_dtheta_threshold(self, weight, threshold, exp_scale, scale=None):
        if scale is None:
            scale = self.dtheta_scale
        self.terms.append(DthetaThresholdTerm(threshold, exp_scale))
        self.weights["dtheta_threshold"] = weight * scale
        return self
    
    def build(self):
        return Oracle(self.terms, self.weights)


def create_oracle_from_config(config):
    builder = OracleBuilder()

    if config.w_dtheta is not None:
        builder.add_pitch(config.w_dtheta)
    if config.w_ddx is not None:
        builder.add_longitudinal(config.w_ddx)

    if config.w_ddx_threshold is not None:
        builder.add_ddx_threshold(
            config.w_ddx_threshold,
            config.ddx_threshold,
            config.ddx_exp_scale
        )

    if config.w_dtheta_threshold is not None:
        builder.add_dtheta_threshold(
            config.w_dtheta_threshold,
            config.dtheta_threshold,
            config.dtheta_exp_scale
        )

    return builder.build()
