import math
import numpy as np


class Bump:
    __slots__ = ("base_specs", "bump_specs", "rng")

    def __init__(self):
    # (pos, half_width, height)
        self.base_specs = np.array([[3.0, 1.25, 0.09]], dtype=np.float64)
        self.bump_specs = self.base_specs.copy()
        self.rng = None

    def set_rng(self, rng):
        self.rng = rng

    def reset(self):
        self.bump_specs[:] = self.base_specs
        self.bump_specs[0, 1] += self.rng.uniform(-0.75, 0.75)
        self.bump_specs[0, 2] += self.rng.uniform(-0.03, 0.03)

    def __call__(self, position):
        bump_position, half_width, height = self.bump_specs[0]
        distance = position - bump_position
        if abs(distance) > half_width:
            return 0.0
        return height * (1.0 + math.cos(math.pi * distance / half_width)) * 0.5

    def set_x(self, position):
        self.base_specs[0, 0] = position
        self.bump_specs[0, 0] = position
