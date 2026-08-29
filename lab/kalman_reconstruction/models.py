import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from .state_space import GRAVITY, StateSpace, discretize, discretize_input, highpass

FRONT_LENGTH, REAR_LENGTH, MASS_RATIO = 1.45, 1.50, 20.0

LATENT_1DOF = {"oscillator": "none", "rw": "rw", "ou": "ou", "matern32": "matern32", "matern52": "matern52"}
PITCH = {"pitch_hc": ("none", False, False, False), "pitch_hc_ou": ("ou", False, False, False),
         "pitch_hc_osc": ("osc2", False, False, False), "pitch_road": ("none", "rw", False, False),
         "pitch_road_osc": ("osc2", "rw", False, False), "pitch_delay": ("none", "delay", False, False),
         "pitch_delay_osc": ("osc2", "delay", False, False), "pitch_tq": ("osc2", "rw", True, False),
         "pitch_ax": ("osc2", "rw", True, True), "pitch_axou": ("osc2", "ou", True, True),
         "pitch_eps": ("osc2", "rw", True, True)}
ROAD_STATES = {False: 0, "rw": 2, "ou": 2, "delay": 1}
SPLIT_EPS = {"pitch_eps"}


def disturbance(name, log_sigma2=None, log_lam=None, zeta=None):
    if name == "none":
        return np.zeros((0, 0)), 0.0, np.zeros((0, 0))
    sigma2 = np.exp(log_sigma2)
    if name == "rw":
        return np.zeros((1, 1)), sigma2, 10.0 * np.eye(1)
    lam = np.exp(log_lam)
    if name == "ou":
        f, q = np.array([[-lam]]), 2 * sigma2 * lam
    elif name == "matern32":
        f, q = np.array([[0, 1], [-lam**2, -2 * lam]], float), 4 * sigma2 * lam**3
    elif name == "matern52":
        f = np.array([[0, 1, 0], [0, 0, 1], [-lam**3, -3 * lam**2, -3 * lam]], float)
        q = 16 / 3 * sigma2 * lam**5
    elif name == "osc2":
        f = np.array([[0, 1], [-lam * lam, -2 * zeta * lam]], float)
        q = 4 * zeta * lam**3 * sigma2
    else:
        raise ValueError(name)
    qm = np.zeros_like(f)
    qm[-1, -1] = q
    return f, q, solve_continuous_lyapunov(f, -qm)


def augment(f, qc, P, entry, dist):
    fd, qd, Pd = dist
    m = len(fd)
    if not m:
        return f, qc, P
    n = len(f)
    f, qc, P = (np.pad(a, ((0, m), (0, m))) for a in (f, qc, P))
    f[entry, n] = 1.0
    f[n:, n:] = fd
    qc[-1, -1] = qd
    P[n:, n:] = Pd
    return f, qc, P


def onedof(p, fs, dist="none"):
    frequency, zeta, log_qv, log_qf, log_r = p[:5]
    w = 2 * np.pi * frequency
    f = np.array([[0, 1], [-w * w, -2 * zeta * w]], float)
    qc = np.diag([1e-10, np.exp(log_qv)])
    f, qc, P = augment(f, qc, np.eye(2), 1, disturbance(dist, log_qf, *p[5:6]))
    A, Q = discretize(f, qc, 1 / fs)
    return StateSpace(A, f[1:2].copy(), Q, np.array([[np.exp(log_r)]]), P)


def quarter_car(p, fs):
    frequency, zeta, log_gamma, log_road, log_r = p
    w, rho, gamma = 2 * np.pi * frequency, 20 / 3, np.exp(log_gamma)
    a, b = w * w, 2 * zeta * w
    f = np.array([[0, 1, 0, 0], [-a, -b, a, b], [0, 0, 0, 1],
                  [rho * a, rho * b, -rho * a * (1 + gamma), -rho * b]], float)
    b_road = np.array([[0], [0], [0], [rho * gamma * a]], float)
    A, G = discretize_input(f, b_road, 1 / fs)
    qr = np.exp(log_road)
    return StateSpace(A, f[1:2], qr * G @ G.T, np.array([[np.exp(log_r)]]), np.eye(4),
                      G=G, Qu=np.array([[qr]]))


def half_car(p, fs):
    frequency, zeta, log_gamma, log_j, log_road, log_rz, log_rl = p
    w, rho, gamma, j = 2 * np.pi * frequency, 20 / 3, np.exp(log_gamma), np.exp(log_j)
    a, b = w * w, 2 * zeta * w
    f = np.zeros((8, 8))
    f[0, 1], f[2, 3], f[4, 5], f[6, 7] = 1, 1, 1, 1
    f[1] = [-a, -b, 0, 0, a / 2, b / 2, a / 2, b / 2]
    f[3] = [0, 0, -2 * a / j, -2 * b / j, -a / j, -b / j, a / j, b / j]
    f[5] = [rho * a, rho * b, -rho * a, -rho * b, -rho * a * (1 + gamma), -rho * b, 0, 0]
    f[7] = [rho * a, rho * b, rho * a, rho * b, 0, 0, -rho * a * (1 + gamma), -rho * b]
    b_road = np.zeros((8, 2))
    b_road[5, 0] = b_road[7, 1] = rho * gamma * a
    A, G = discretize_input(f, b_road, 1 / fs)
    qu = 2 * np.exp(log_road) * np.eye(2)
    P = np.eye(8)
    P[4:, 4:] *= 2
    return StateSpace(A, f[[1, 3]], G @ qu @ G.T, np.diag(np.exp([log_rz, log_rl])), P, G=G, Qu=qu)


def pitch_half_car(p, fs, dist="none", road=False, torque=False, ax=False, split=False):
    frequency, zeta, eps, log_j, log_gamma, g_u, beta, lam_f, lam_r = p[:9]
    q_road, q_body, q_long, r_wheel, r_az = np.exp(p[9:14])
    w, j, gamma = 2 * np.pi * frequency, np.exp(log_j), np.exp(log_gamma)
    lf, lr, rho = FRONT_LENGTH, REAR_LENGTH, MASS_RATIO
    af, ar = w * w * (1 + eps) / 2, w * w * (1 - eps) / 2
    eps_c = p[-1] if split else eps
    bf, br = zeta * w * (1 + eps_c), zeta * w * (1 - eps_c)
    n = 9 + ROAD_STATES[road]
    front, rear = np.zeros(n), np.zeros(n)
    front[[0, 1, 2, 3, 4, 5]] = af, bf, af * lf, bf * lf, -af, -bf
    rear[[0, 1, 2, 3, 6, 7]] = ar, br, -ar * lr, -br * lr, -ar, -br
    f = np.zeros((n, n))
    f[0, 1] = f[2, 3] = f[4, 5] = f[6, 7] = 1
    f[1] = -(front + rear)
    f[3] = (-lf * front + lr * rear) / (j * lf * lr)
    f[5] = rho * front
    f[5, 4] -= rho * gamma * af
    f[7] = rho * rear
    f[7, 6] -= rho * gamma * ar
    qc = np.zeros((n, n))
    qc[1, 1] = qc[3, 3] = q_body
    qc[8, 8] = q_long
    if road:
        f[5, 9] = rho * gamma * af
        qc[9, 9] = q_road
        if road in ("rw", "ou"):
            f[7, 10] = rho * gamma * ar
            qc[10, 10] = q_road
        if road == "ou":
            lam_road = np.exp(p[-1])
            f[9, 9] = f[10, 10] = -lam_road
            qc[9, 9] = qc[10, 10] = 2 * q_road * lam_road
    else:
        qc[5, 5], qc[7, 7] = q_road * (rho * gamma * af) ** 2, q_road * (rho * gamma * ar) ** 2
    tail = () if dist == "none" else (p[15], p[14]) if dist == "ou" else (p[16], p[14], p[15])
    f, qc, P = augment(f, qc, np.eye(n), 3, disturbance(dist, *tail))
    m = len(f)
    idx = 14 + 2 * (dist == "ou") + 3 * (dist == "osc2")
    if torque:
        g_f, g_r, s_f, s_r = p[idx:idx + 4]
    if ax:
        log_lam_a, g_v, log_sig_a, log_q_grade, log_q_bias, log_r_ax = p[idx + 4 * torque:idx + 4 * torque + 6]
        lam_a, sig_a2 = np.exp(log_lam_a), np.exp(log_sig_a)
        f, qc, P = np.pad(f, ((0, 3), (0, 3))), np.pad(qc, ((0, 3), (0, 3))), np.pad(P, ((0, 3), (0, 3)))
        ia, igrade, ibias = m, m + 1, m + 2
        f[8, ia] = 1.0
        f[3, ia] = g_u
        f[ia, ia] = -lam_a
        qc[ia, ia] = 2 * sig_a2 * lam_a
        qc[igrade, igrade], qc[ibias, ibias] = np.exp(log_q_grade), np.exp(log_q_bias)
        P[ia, ia] = sig_a2
        P[igrade, igrade] = P[ibias, ibias] = .01
        m += 3
    b = np.zeros((m, 2 if ax else 1 + 2 * torque))
    D = None
    if ax:
        b[ia, 0] = b[ia, 1] = lam_a * g_v
        if torque:
            b[3, 0], b[3, 1] = g_f, g_r
            D = np.zeros((4, 2))
            D[0, 0], D[1, 1] = s_f, s_r
    else:
        b[3, 0], b[8, 0] = g_u, 1
        if torque:
            b[3, 1], b[3, 2] = g_f, g_r
            D = np.zeros((3, 3))
            D[0, 1], D[1, 2] = s_f, s_r
    h = np.zeros((3 + ax, m))
    h[0, 1], h[0, 3], h[0, 5], h[0, 8] = beta, beta * lf - lam_f, -beta, 1
    h[1, 1], h[1, 3], h[1, 7], h[1, 8] = beta, -beta * lr - lam_r, -beta, 1
    h[2] = f[1]
    R = [r_wheel, r_wheel, r_az]
    if ax:
        h[3, ia], h[3, 2], h[3, igrade], h[3, ibias] = 1.0, GRAVITY, GRAVITY, 1.0
        R.append(np.exp(log_r_ax))
    A, Q = discretize(f, qc, 1 / fs)
    if road == "delay":
        rear_road = np.zeros((m, 1))
        rear_road[7, 0] = rho * gamma * ar
        _, BE = discretize_input(f, np.c_[b, rear_road], 1 / fs)
        B, E = BE[:, :-1], BE[:, -1]
    else:
        _, B = discretize_input(f, b, 1 / fs)
        E = None
    return StateSpace(A, h, Q, np.diag(R), P, B=B, D=D, E=E)


def kinematic(fs, process_var, acceleration_var, velocity_gain=None, velocity_var=None):
    dt = 1 / fs
    A = np.array([[1, dt, .5 * dt**2], [0, 1, dt], [0, 0, 1]], float)
    gamma = np.array([.5 * dt**2, dt, 1.])
    Q = process_var * np.outer(gamma, gamma)
    if velocity_gain is None:
        H, R = np.array([[0, 0, 1.]]), np.array([[acceleration_var]])
    else:
        H = np.array([[0, velocity_gain, 0], [0, 0, 1.]], float)
        R = np.diag([velocity_var, acceleration_var])
    return StateSpace(A, H, Q, R, np.eye(3))


def model(name, p, fs):
    if name == "qc2":
        return quarter_car(p, fs)
    if name in PITCH:
        return pitch_half_car(p, fs, *PITCH[name], split=name in SPLIT_EPS)
    return onedof(p, fs, LATENT_1DOF[name])


def estimate(name, az_g, p, fs, smooth=False, road=False):
    y = ((az_g - 1) * GRAVITY)[..., None]
    return model(name, p, fs).filter(y, smooth=smooth, road=road)


def estimate_half(az_g, lat_g, p, fs, smooth=False, road=False, full=False, chunk=256):
    state_space = half_car(p, fs)
    output, inputs = [], []
    for start in range(0, len(az_g), chunk):
        stop = start + chunk
        y = np.stack(((az_g[start:stop] - 1) * GRAVITY,
                      highpass(lat_g[start:stop] * GRAVITY, 0.5, fs)), -1)
        value = state_space.filter(y, smooth=smooth, road=road)
        state, road_value = value if road else (value, None)
        output.append(state if full else state[..., [1, 3]])
        if road:
            inputs.append(road_value)
    result = np.concatenate(output)
    return (result, np.concatenate(inputs)) if road else result


def estimate_pitch(name, x, p, fs, smooth=False):
    front, rear = x[:, :, 5:7].mean(2) / 3.6, x[:, :, 7:9].mean(2) / 3.6
    ax_meas = x[:, :, 4] * GRAVITY
    _, _, torque, ax = PITCH[name]
    y = np.stack((front, rear, (x[:, :, 1] - 1) * GRAVITY) + ((ax_meas,) if ax else ()), -1)
    if ax:
        u = np.stack((x[:, :, 10], x[:, :, 9]), -1)
    elif torque:
        u = np.stack((ax_meas, x[:, :, 10], x[:, :, 9]), -1)
    else:
        u = ax_meas[..., None]
    state_space = model(name, p, fs)
    x0 = np.zeros((len(x), len(state_space.A)))
    x0[:, 8] = (front[:, 0] + rear[:, 0]) / 2
    if state_space.E is None:
        return state_space.filter(y, u, x0, smooth=smooth)
    distance = np.cumsum(np.maximum((front + rear) / 2, 0), 1) / fs
    wheelbase = FRONT_LENGTH + REAR_LENGTH
    rear_index = np.stack([np.searchsorted(d, d - wheelbase, side="left") for d in distance])
    rear_index[distance < wheelbase] = -1
    rows = np.arange(len(x))

    def replay(t, history):
        index = rear_index[:, t]
        valid = (index >= 0) & (index < t)
        rear_road = np.zeros(len(index))
        rear_road[valid] = history[rows[valid], index[valid], 9]
        return rear_road[:, None] * state_space.E

    return state_space.filter(y, u, x0, smooth=smooth, extra=replay)


def estimate_kinematic(acceleration, fs, process_var, acceleration_var=1., velocity=None,
                       velocity_gain=None, velocity_offset=0., velocity_var=None):
    state_space = kinematic(fs, process_var, acceleration_var, velocity_gain, velocity_var)
    observation = acceleration[..., None] if velocity is None else \
        np.stack((velocity - velocity_offset, acceleration), -1)
    return state_space.filter(observation)


def _az(name):
    return lambda x, p, fs, **kw: estimate(name, x[:, :, 1], p, fs, **kw)


def _pitch(name):
    return lambda x, p, fs, **kw: estimate_pitch(name, x, p, fs, **kw)


SPECS = {name: dict(target=0, output=1, run=_az(name)) for name in (*LATENT_1DOF, "qc2")}
SPECS |= {name: dict(target=2, output=3, run=_pitch(name)) for name in PITCH}


def model_spec(name):
    start = np.array([1.3, .3, -5, 0, -3])
    bounds = [(0.5, 8), (.05, 5), (-12, 3), (-12, 5), (-12, 4)]
    if name == "oscillator":
        return start, bounds
    if name == "qc2":
        return np.array([1.3, .4, np.log(10), -10, -3]), [(0.5, 3), (.05, 1.5),
               (np.log(3), np.log(20)), (-20, -4), (-12, 4)]
    if name in PITCH:
        dist, road, torque, ax = PITCH[name]
        start = np.r_[1.5, .4, 0, 0, np.log(10), .3, .13, .55, .55,
                      -10, np.log(.01), np.log(.5), np.log(2e-3), np.log(.6)]
        bounds = [(0.5, 4), (.05, 1.5), (-.9, .9), (np.log(.3), np.log(3)), (np.log(3), np.log(20)),
                  (-10, 10), (-2, 2), (-10, 10), (-10, 10), (-20, -2), (-16, 4), (-12, 6), (-14, 2), (-6, 6)]
        if dist == "ou":
            start = np.r_[start, np.log(2), 0.0]
            bounds += [(np.log(.05), np.log(100)), (-12, 8)]
        if dist == "osc2":
            start = np.r_[start, np.log(2 * np.pi), .7, np.log(4)]
            bounds += [(np.log(.3), np.log(60)), (.05, 2.0), (-12, 8)]
        if torque:
            start = np.r_[start, .01, .01, 1e-3, 1e-3]
            bounds += [(-.1, .1), (-.1, .1), (-.02, .02), (-.02, .02)]
        if ax:
            start = np.r_[start, np.log(20), .01, np.log(.2), -8, -10, -3]
            bounds += [(np.log(1), np.log(200)), (-.05, .05), (-8, 4), (-16, 0), (-16, 0), (-8, 4)]
        if road == "ou":
            start = np.r_[start, np.log(.5)]
            bounds += [(np.log(.01), np.log(50))]
        if name in SPLIT_EPS:
            start = np.r_[start, 0.0]
            bounds += [(-.9, .9)]
        return start, bounds
    if name != "rw":
        start, bounds = np.r_[start, np.log(2)], bounds + [(np.log(.05), np.log(100))]
    return start, bounds
