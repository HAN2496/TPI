import numpy as np


class BaseMapper:
    def __init__(self, initial_value, clip=None):
        self.clip = clip
        self.initial_param = np.asarray(initial_value, dtype=np.float32).copy()
        self.param = self.initial_param.copy()

    def initial_value(self):
        return self.initial_param.copy()

    def compute_target(self, action, current):
        raise NotImplementedError

    def reset(self):
        self.param = self.initial_param.copy()

    def update(self, action):
        target = self.compute_target(action, self.param)
        if self.clip is not None:
            low, high = self.clip
            target = np.clip(target, low, high)
        self.param = np.asarray(target, dtype=np.float32).copy()
        return self.param


class ScalarBaseMapper(BaseMapper):
    def __init__(self, initial_value, clip=None):
        self.action_dim = 1
        super().__init__(initial_value, clip)


class ScalarDirectMap(ScalarBaseMapper):
    def __init__(self, scale, clip=None, init=None):
        self.scale = float(scale)
        super().__init__(0.0 if init is None else init, clip)

    def compute_target(self, action, current):
        return self.scale * float(np.asarray(action).reshape(()))


class ScalarDerivMap(ScalarBaseMapper):
    def __init__(self, base_value, delta_per_action, clip=None):
        self.delta = float(delta_per_action)
        super().__init__(base_value, clip)

    def compute_target(self, action, current):
        return float(current) + self.delta * float(np.asarray(action).reshape(()))


class VectorBaseMapper(BaseMapper):
    def __init__(self, dim, indices=None, initial_value=None, clip=None):
        self.N = int(dim)
        self.idxs = list(range(self.N)) if indices is None else list(indices)
        self.action_dim = len(self.idxs)
        initial_value = np.zeros(self.N, dtype=np.float32) if initial_value is None else initial_value
        initial_value = np.asarray(initial_value, dtype=np.float32)
        if initial_value.size != self.N:
            raise ValueError("initial value length mismatch")
        super().__init__(initial_value, clip)

    def update(self, action):
        target = self.compute_target(action, self.param)
        if self.clip is not None:
            for index, bounds in zip(self.idxs, self.clip):
                target[index] = np.clip(target[index], bounds[0], bounds[1])
        self.param = np.asarray(target, dtype=np.float32).copy()
        return self.param


class VectorDirectMap(VectorBaseMapper):
    def __init__(self, dim, indices=None, direct_gain=1.0, init=None, clip=None):
        indices = list(range(int(dim))) if indices is None else list(indices)
        gain = np.asarray(direct_gain, dtype=np.float32).reshape(-1)
        if gain.size == 1:
            gain = np.full(len(indices), gain.item(), dtype=np.float32)
        if gain.size != len(indices):
            raise ValueError("direct_gain length mismatch")
        self.gain = gain
        initial = np.zeros(int(dim), dtype=np.float32) if init is None else np.asarray(init, dtype=np.float32)
        super().__init__(dim, indices, initial, clip)

    def compute_target(self, action, current):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != len(self.idxs):
            raise ValueError("action length must match indices")
        target = np.asarray(current, dtype=np.float32).copy()
        target[self.idxs] = self.gain * action
        return target


class VectorDerivMap(VectorBaseMapper):
    def __init__(self, dim, indices=None, base_value=1.0, delta_per_action=0.1, clip=None):
        indices = list(range(int(dim))) if indices is None else list(indices)
        base = np.asarray(base_value, dtype=np.float32).reshape(-1)
        if base.size == 1:
            base = np.full(int(dim), base.item(), dtype=np.float32)
        if base.size != int(dim):
            raise ValueError("base_value length mismatch")
        delta = np.asarray(delta_per_action, dtype=np.float32).reshape(-1)
        if delta.size == 1:
            delta = np.full(len(indices), delta.item(), dtype=np.float32)
        if delta.size != len(indices):
            raise ValueError("delta_per_action length mismatch")
        self.delta = delta
        super().__init__(dim, indices, base, clip)

    def compute_target(self, action, current):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != len(self.idxs):
            raise ValueError("action length must match indices")
        target = np.asarray(current, dtype=np.float32).copy()
        target[self.idxs] = target[self.idxs] + self.delta * action
        return target


class LQRProjectedDerivMap(VectorBaseMapper):
    def __init__(self, dim, indices, base_value, delta_per_action, region_path, dx_power=0.4, dx_offset=0.01, boundary_margin=1e-5):
        base = np.asarray(base_value, dtype=np.float64)
        if base.size != int(dim):
            raise ValueError("base_value length mismatch")
        delta = np.asarray(delta_per_action, dtype=np.float64)
        if delta.size != len(indices) or np.any(delta <= 0):
            raise ValueError("delta_per_action must match the scheduled indices")
        if len(indices) not in (2, 3):
            raise ValueError("LQRProjectedDerivMap requires [q_dtheta, q_theta] or [q_dtheta, q_theta, q_dx]")

        super().__init__(dim, indices, base, None)
        self.delta = delta
        self.dx_power = float(dx_power)
        self.dx_offset = float(dx_offset)
        self.boundary_margin = float(boundary_margin)

        data = np.load(region_path)
        self.axes = np.asarray(data["axes"], dtype=np.float64)
        self.surface = np.asarray(data["surface"], dtype=np.float64)
        self.scales = np.asarray(data["scales"], dtype=np.float64)
        active = self.surface > 0.0
        self.surface[active] = np.minimum(self.surface[active] + self.boundary_margin, self.axes[-1])
        self.q_max = self.scales * self.axes[-1] / (1.0 - self.axes[-1])

        if len(self.idxs) == 3:
            self.triangles = self._build_triangles()
        else:
            self.pitch_lows, self.pitch_highs = self._build_pitch_boxes()

    def _q_to_coord(self, q):
        q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
        r_dx = (q[2] + self.dx_offset) ** self.dx_power - self.dx_offset ** self.dx_power
        return np.array([q[0] / self.delta[0], q[1] / self.delta[1], r_dx / self.delta[2]])

    def _coord_to_q(self, coord):
        coord = np.maximum(np.asarray(coord, dtype=np.float64), 0.0)
        r_dx = coord[2] * self.delta[2] + self.dx_offset ** self.dx_power
        return np.array([
            coord[0] * self.delta[0],
            coord[1] * self.delta[1],
            max(r_dx ** (1.0 / self.dx_power) - self.dx_offset, 0.0),
        ])

    def _z_to_q(self, z):
        z = np.clip(np.asarray(z, dtype=np.float64), 0.0, self.axes[-1])
        return self.scales * z / (1.0 - z)

    def _build_triangles(self):
        vertices = np.empty((len(self.axes), len(self.axes), 3))
        for i, x in enumerate(self.axes):
            for j, y in enumerate(self.axes):
                vertices[i, j] = self._q_to_coord(self._z_to_q([x, y, self.surface[i, j]]))
        triangles = []
        for i in range(len(self.axes) - 1):
            for j in range(len(self.axes) - 1):
                a, b = vertices[i, j], vertices[i + 1, j]
                c, d = vertices[i + 1, j + 1], vertices[i, j + 1]
                triangles.extend([[a, b, c], [a, c, d]])
        return np.asarray(triangles)

    def _build_pitch_boxes(self):
        q_axes = [self.scales[i] * self.axes / (1.0 - self.axes) for i in range(2)]
        coord_axes = [q_axes[i] / self.delta[i] for i in range(2)]
        lows = []
        highs = []
        for i in range(len(self.axes) - 1):
            for j in range(len(self.axes) - 1):
                if np.all(self.surface[i:i + 2, j:j + 2] <= 0.0):
                    lows.append([coord_axes[0][i], coord_axes[1][j]])
                    highs.append([coord_axes[0][i + 1], coord_axes[1][j + 1]])
        if not lows:
            raise ValueError("No feasible q_dx=0 cells in projection region")
        return np.asarray(lows), np.asarray(highs)

    def _project_pitch(self, point):
        candidates = np.minimum(np.maximum(point, self.pitch_lows), self.pitch_highs)
        return candidates[np.argmin(np.sum((candidates - point) ** 2, axis=1))]

    def _boundary(self, q):
        z = q / (q + self.scales)
        x, y = np.clip(z[:2], self.axes[0], self.axes[-1])
        i = min(np.searchsorted(self.axes, x, side="right") - 1, len(self.axes) - 2)
        j = min(np.searchsorted(self.axes, y, side="right") - 1, len(self.axes) - 2)
        i = max(i, 0)
        j = max(j, 0)
        tx = (x - self.axes[i]) / (self.axes[i + 1] - self.axes[i])
        ty = (y - self.axes[j]) / (self.axes[j + 1] - self.axes[j])
        f00 = self.surface[i, j]
        f10 = self.surface[i + 1, j]
        f11 = self.surface[i + 1, j + 1]
        f01 = self.surface[i, j + 1]
        return (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + tx * ty * f11
            + (1.0 - tx) * ty * f01
        )

    def _contains(self, q):
        z = q / (q + self.scales)
        return bool(
            np.all(q >= 0.0)
            and np.all(q <= self.q_max)
            and z[2] >= self._boundary(q)
        )

    def _nearest_surface(self, point):
        a = self.triangles[:, 0]
        b = self.triangles[:, 1]
        c = self.triangles[:, 2]
        ab = b - a
        ac = c - a
        ap = point - a
        d1 = np.sum(ab * ap, axis=1)
        d2 = np.sum(ac * ap, axis=1)
        candidates = np.empty_like(a)
        remaining = np.ones(len(a), dtype=bool)

        mask = remaining & (d1 <= 0.0) & (d2 <= 0.0)
        candidates[mask] = a[mask]
        remaining[mask] = False

        bp = point - b
        d3 = np.sum(ab * bp, axis=1)
        d4 = np.sum(ac * bp, axis=1)
        mask = remaining & (d3 >= 0.0) & (d4 <= d3)
        candidates[mask] = b[mask]
        remaining[mask] = False

        vc = d1 * d4 - d3 * d2
        mask = remaining & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        v = d1[mask] / (d1[mask] - d3[mask])
        candidates[mask] = a[mask] + v[:, None] * ab[mask]
        remaining[mask] = False

        cp = point - c
        d5 = np.sum(ab * cp, axis=1)
        d6 = np.sum(ac * cp, axis=1)
        mask = remaining & (d6 >= 0.0) & (d5 <= d6)
        candidates[mask] = c[mask]
        remaining[mask] = False

        vb = d5 * d2 - d1 * d6
        mask = remaining & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        w = d2[mask] / (d2[mask] - d6[mask])
        candidates[mask] = a[mask] + w[:, None] * ac[mask]
        remaining[mask] = False

        va = d3 * d6 - d5 * d4
        mask = remaining & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        w = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
        candidates[mask] = b[mask] + w[:, None] * (c[mask] - b[mask])
        remaining[mask] = False

        denominator = va[remaining] + vb[remaining] + vc[remaining]
        v = vb[remaining] / denominator
        w = vc[remaining] / denominator
        candidates[remaining] = a[remaining] + v[:, None] * ab[remaining] + w[:, None] * ac[remaining]
        return candidates[np.argmin(np.sum((candidates - point) ** 2, axis=1))]

    def compute_target(self, action, current):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != len(self.idxs):
            raise ValueError("action length must match scheduled indices")

        if len(self.idxs) == 2:
            coord = np.maximum(np.asarray(current, dtype=np.float64)[self.idxs] / self.delta + action, 0.0)
            inside = np.any(np.all((self.pitch_lows <= coord) & (coord <= self.pitch_highs), axis=1))
            if not inside:
                coord = self._project_pitch(coord)
            target = np.asarray(current, dtype=np.float64).copy()
            target[self.idxs] = coord * self.delta
            return target

        q = np.asarray(current, dtype=np.float64)[self.idxs]
        coord = self._q_to_coord(q) + action
        candidate = np.minimum(self._coord_to_q(coord), self.q_max)
        if not self._contains(candidate):
            candidate = self._coord_to_q(self._nearest_surface(self._q_to_coord(candidate)))
            boundary = self._boundary(candidate)
            z = candidate / (candidate + self.scales)
            if z[2] < boundary:
                boundary = min(boundary + self.boundary_margin, self.axes[-1])
                candidate[2] = self.scales[2] * boundary / (1.0 - boundary)

        target = np.asarray(current, dtype=np.float64).copy()
        target[self.idxs] = np.minimum(candidate, self.q_max)
        return target
