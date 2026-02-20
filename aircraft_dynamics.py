# aircraft_dynamics.py
import numpy as np
from dataclasses import dataclass

@dataclass
class AircraftParams:
    # Longitudinal-ish xz-plane toy model params
    m: float = 8431.375
    A_w: float = 27.87
    c: float = 3.45
    I: float = 80187.14101685
    U0: float = 175.0
    rho: float = 1.0
    g: float = 9.81
    CD: float = 0.02
    CL_alpha: float = 5.0
    CL_gamma: float = 0.4
    CM_alpha: float = -0.7
    CM_thetadot: float = -10.0
    CM_gamma: float = -1.2

@dataclass
class ActuatorLimits:
    elev_max: float = np.deg2rad(25.0)
    throttle_max: float = 40e3
    elev_rate_max: float = np.deg2rad(200.0)   # rad/s
    throttle_rate_max: float = 8e3             # N/s
    use_engine_lag: bool = True
    tau_throttle: float = 0.5                  # s

def build_longitudinal_matrices(p: AircraftParams):
    """
    State x = [x, z, u, w, theta, q]
    Input u = [elevator, throttle]
    """
    A_33 = -p.CD * p.rho * p.U0 * p.A_w / p.m
    A_44 = -p.CL_alpha * p.rho * p.U0 * p.A_w / (2 * p.m)
    A_64 = p.CM_alpha * p.rho * p.U0 * p.c * p.A_w / (2 * p.I)
    A_66 = p.CM_thetadot * p.rho * p.U0 * p.c**2 * p.A_w / (4 * p.I)

    B_t = 1.0 / p.m
    B_4 = -p.CL_gamma * p.rho * p.U0**2 * p.A_w / (2 * p.m)
    B_6 = p.CM_gamma * p.rho * p.U0**2 * p.c * p.A_w / (2 * p.I)

    A6 = np.array([
        [0, 0,    1,    0,     0,    0],
        [0, 0,    0,    1,     0,    0],
        [0, 0, A_33,    0,    -p.g,  0],
        [0, 0,    0, A_44,     0,   p.U0],
        [0, 0,    0,    0,     0,    1],
        [0, 0,    0, A_64,     0, A_66],
    ], dtype=float)

    B6 = np.array([
        [0,   0  ],
        [0,   0  ],
        [0,   B_t],
        [B_4, 0  ],
        [0,   0  ],
        [B_6, 0  ],
    ], dtype=float)

    return A6, B6

def throttle_trim_force(p: AircraftParams) -> float:
    return 0.5 * p.rho * p.CD * p.U0**2 * p.A_w

def sat_rate(cmd: float, prev: float, rate_max: float, dt: float) -> float:
    return float(np.clip(cmd, prev - rate_max * dt, prev + rate_max * dt))

def rk4_step(x: np.ndarray, u: np.ndarray, dt: float, A6: np.ndarray, B6: np.ndarray) -> np.ndarray:
    def f(x_):
        return A6 @ x_ + B6 @ u
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class Aircraft:
    """
    Generic aircraft plant object with actuator memory + optional engine lag.
    """
    def __init__(self, x0, p: AircraftParams, limits: ActuatorLimits):
        self.p = p
        self.limits = limits
        self.A6, self.B6 = build_longitudinal_matrices(p)

        self.x = np.array(x0, dtype=float)

        self.throttle_trim = throttle_trim_force(p)

        # actuator memory
        self.elev_cmd_prev = 0.0
        self.throttle_cmd_prev = self.throttle_trim
        self.throttle_act_prev = self.throttle_trim

    def apply_actuators(self, elev_cmd_unsat: float, throttle_cmd_unsat: float, dt: float):
        # position saturation
        elev_cmd = float(np.clip(elev_cmd_unsat, -self.limits.elev_max, self.limits.elev_max))
        throttle_cmd = float(np.clip(throttle_cmd_unsat, -self.limits.throttle_max, self.limits.throttle_max))

        # rate limits
        elev_cmd = sat_rate(elev_cmd, self.elev_cmd_prev, self.limits.elev_rate_max, dt)
        throttle_cmd = sat_rate(throttle_cmd, self.throttle_cmd_prev, self.limits.throttle_rate_max, dt)

        # engine lag on throttle
        if self.limits.use_engine_lag:
            throttle_act = self.throttle_act_prev + (throttle_cmd - self.throttle_act_prev) * (dt / self.limits.tau_throttle)
        else:
            throttle_act = throttle_cmd

        ctrl_cmd = np.array([elev_cmd, throttle_cmd], dtype=float)
        ctrl_act = np.array([elev_cmd, throttle_act], dtype=float)

        # save memory
        self.elev_cmd_prev = elev_cmd
        self.throttle_cmd_prev = throttle_cmd
        self.throttle_act_prev = throttle_act

        return ctrl_cmd, ctrl_act

    def step(self, ctrl_act: np.ndarray, dt: float):
        self.x = rk4_step(self.x, ctrl_act, dt, self.A6, self.B6)
        return self.x