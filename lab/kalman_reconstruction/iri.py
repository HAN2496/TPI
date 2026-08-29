import numpy as np
from scipy.linalg import expm


def spatialize(speed_kmh, roads, fs, step=0.1):
    velocity = np.maximum(speed_kmh, 0) / 3.6
    distance = np.r_[0, np.cumsum((velocity[:-1] + velocity[1:]) / (2 * fs))]
    position = (distance[:-1] + distance[1:]) / 2
    keep = np.r_[True, np.diff(position) > 1e-8]
    position, grid = position[keep], np.arange(step / 2, distance[-1], step)
    if len(position) < 2 or not len(grid):
        return grid, [np.full_like(grid, np.nan) for _ in roads]
    return grid, [np.interp(grid, position, road[1:][keep]) for road in roads]


def golden_iri(road, step=0.1, window=40):
    ms, mu, ks, cs, kt = 250, 37.5, 15825, 1500, 163250
    f = np.array([[0, 1, 0, 0], [-ks / ms, -cs / ms, ks / ms, cs / ms], [0, 0, 0, 1],
                  [ks / mu, cs / mu, -(ks + kt) / mu, -cs / mu]])
    b = np.array([[0], [0], [0], [kt / mu]])
    block = np.zeros((5, 5))
    block[:4, :4], block[:4, 4:] = f, b
    value = expm(block * step / (80 / 3.6))
    state, rattle = np.zeros(4), np.empty(len(road))
    for i, value_road in enumerate(road):
        state = value[:4, :4] @ state + value[:4, 4] * value_road
        rattle[i] = state[1] - state[3]
    n, iri = round(window / step), np.full(len(road), np.nan)
    if len(road) >= n:
        iri[n - 1:] = 1000 * step * np.convolve(np.abs(rattle), np.ones(n), "valid") / (window * (80 / 3.6))
    return iri


def spatial_results(speed, qc_road, hc_road, ids, fs, step=0.1, window=40):
    profiles, rows = [], []
    for i in range(len(speed)):
        distance, roads = spatialize(speed[i], [qc_road[i], hc_road[i, :, 0], hc_road[i, :, 1]], fs, step)
        iris = [golden_iri(road, step, window) for road in roads]
        median = lambda value: float(np.median(value[np.isfinite(value)])) if np.isfinite(value).any() else np.nan
        rows.append(dict(id=ids[i], distance_m=float(distance[-1] + step / 2) if len(distance) else 0,
                         qc2_iri_m_per_km=median(iris[0]), hc8_left_iri_m_per_km=median(iris[1]),
                         hc8_right_iri_m_per_km=median(iris[2])))
        profiles.append((distance, roads, iris))
    names = ("distance_m", "qc2_road_m", "hc8_left_road_m", "hc8_right_road_m",
             "qc2_iri_m_per_km", "hc8_left_iri_m_per_km", "hc8_right_iri_m_per_km")
    width = max((len(value[0]) for value in profiles), default=0)
    arrays = {name: np.full((len(profiles), width), np.nan, np.float32) for name in names}
    for i, (distance, roads, iris) in enumerate(profiles):
        n = len(distance)
        arrays["distance_m"][i, :n] = distance
        for name, value in zip(names[1:], roads + iris):
            arrays[name][i, :n] = value
    return arrays, rows
