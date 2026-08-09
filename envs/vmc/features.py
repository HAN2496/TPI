import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def p2p(x):
    return float(x.max() - x.min())


SCALE = {
    "pitch_rate_sq": 0.015,
    "long_accel_sq": 0.050,
    "pitch_rate_rms": 0.120,
    "pitch_rate_p2p": 0.500,
    "long_accel_rms": 0.200,
    "long_accel_p2p": 0.600,
}

STEP = {
    "pitch_rate_sq": lambda tau: tau.channels["dtheta"] ** 2 / SCALE["pitch_rate_sq"],
    "long_accel_sq": lambda tau: tau.channels["ddx_com"] ** 2 / SCALE["long_accel_sq"],
}

EPISODE = {
    "pitch_rate_rms": lambda tau: rms(tau.channels["dtheta"]) / SCALE["pitch_rate_rms"],
    "pitch_rate_p2p": lambda tau: p2p(tau.channels["dtheta"]) / SCALE["pitch_rate_p2p"],
    "long_accel_rms": lambda tau: rms(tau.channels["ddx_com"]) / SCALE["long_accel_rms"],
    "long_accel_p2p": lambda tau: p2p(tau.channels["ddx_com"]) / SCALE["long_accel_p2p"],
}

NAMES = tuple(STEP) + tuple(EPISODE)
