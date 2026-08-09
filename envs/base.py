from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """공용 궤적 스키마 — 하류(phi, oracle, 추론)는 채널 이름만 본다."""
    channels: dict              # name -> (T,) per-step 관측 시계열 (이것이 obs)
    dt: float
    meta: dict                  # seed, params, horizon, terminated, ...
    obs: np.ndarray = None      # 선택 — raw 시뮬 관측 (T, D), 채널 재매핑용
    action: np.ndarray = None   # 선택 — logged 소스에는 없음

    def __len__(self):
        return len(next(iter(self.channels.values())))


class TrajectorySource:
    def episodes(self) -> list[Trajectory]:
        ...


class ControllableEnv(TrajectorySource):
    """④ 최적화(CMA-ES/RL)가 요구하는 능력 — logged 소스는 여기 못 들어온다."""

    def rollout(self, params, n, seed) -> list[Trajectory]:
        ...
