import yaml
from dataclasses import asdict, astuple, dataclass, field

@dataclass
class Parameters:
    def get_dict(self):
        return asdict(self)

    def get_tuple(self):
        return astuple(self)

    def get_yaml(self):
        return yaml.dump(self.get_dict())

@dataclass
class Environment_Parameters(Parameters):
    dt_inner: float = 0.001
    dt_outer: float = 0.001 # control timestep (see line 200 ~ in src/env/plant.py)

    t_end: float = 20.0
    x0: list = field(default_factory=lambda: [0, 0, 0, 0, 35/3.6, 0, 0, 0, 0, 0.0])

    # Single Scenario
    t_observe: float = 3.0

    is_multi_bump: bool = False


@dataclass(slots=True)
class Vehicle_Parameters(Parameters):
    # sprung
    k_s_f : float = 52741.0       # front sprung stiffness [N/m]
    k_s_r : float = 52741.0 
    c_s_f : float = 4029.0        # front damping coefficient [Ns/m]
    c_s_r : float = 4029.0

    # unsprung
    k_us_f: float = 220000.0      # front unsprung stiffness [N/m]
    k_us_r: float = 220000.0
    c_us_f: float = 0.0           # front unsprung damping coefficient [Ns/m]
    c_us_r: float = 0.0 

    # mass
    m_s: float    = 1710.0        # total sprung mass [kg]
    m_us_f: float = 100.0         # front unsprung mass [kg]
    m_us_r: float = 100.0
    m_tot : float = 1910.0        # total vehicle mass [kg]

    I: float   = 3900.0           # vehicle inertia [kg·m²]
    J_w: float = 1.6              # wheel inertia [kg·m²]

    # tire
    r_wheel: float = 0.37         # wheel radius [m]
    mu: float      = 0.90         # tire-road maximum static friction coefficient
    C_r_f: float   = 0.01         # front rolling resistance coefficient
    C_r_r: float   = 0.01

    # drivetrain
    r_transmission: float = 10.65 # transmission ratio
    eta_drive: float = 0.9        # drivetrain efficiency

    # aerodynamics
    rho_air: float = 1.225        # air density [kg/m³]
    C_d: float     = 0.288        # drag coefficient
    A: float       = 2.58         # frontal area [m²]

    # geometry
    l_f: float   = 1.5            # distance to front axle [m]
    l_r: float   = 1.5
    h_com: float = 0.53           # center of gravity height [m]
    h_f: float   = 0.5            # front height [m]
    h_r: float   = 0.5

    phi_f: float = 0.0110         # front tire slip angle [rad]
    phi_r: float = 0.1293

@dataclass
class Search_config(Parameters):
    n_seeds: int = 200
    batch_size: int = 64

    default_q_diag: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    mpc_settings: dict = field(default_factory=lambda: {
        'N': 5,
        'solver_type': 'osqp',
        'solver_options': {'dt': 0.01, 'max_iter': 4000, 'eps_abs': 1e-2, 'eps_rel': 1e-2}
    })

    p: dict = field(default_factory=lambda:{"range": (30, 300), "coarse_step": 15, "fine_range": 15})
    lqr: dict = field(default_factory=lambda: {"range": (794, 7940), "coarse_step": 397, "fine_range": 397})
    mpc: dict = field(default_factory=lambda: {"range": (742, 7420), "coarse_step": 371, "fine_range": 371})