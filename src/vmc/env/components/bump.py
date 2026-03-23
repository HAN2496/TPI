import numpy as np

class Bump:
    # (pos, half_width, height)
    def __init__(self):
        self.base_specs = np.array([(3, 1.25, 0.09)])
        self.bump_specs = self.base_specs.copy()

    def set_rng(self, rng):
        self.rng = rng

    def reset(self):
        new_specs = np.copy(self.base_specs)
        new_specs[0, 1] += self.rng.uniform(-0.75, 0.75)
        new_specs[0, 2] += self.rng.uniform(-0.03, 0.03)
        self.bump_specs = new_specs           

    @staticmethod
    def _bump_shape(dist, half_width, height):  # sinusoidal
        return height * (1 + np.cos(np.pi * dist / half_width)) / 2

    def __call__(self, x):
        for pos, hw, h in self.bump_specs:
            d = x - pos
            if -hw <= d <= hw:
                return self._bump_shape(d, hw, h)
        return 0.0
    
    def set_x(self, x0):
        self.base_specs[:, 0] = x0
        self.bump_specs[:, 0] = x0