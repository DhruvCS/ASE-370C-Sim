# aircraft_controller.py
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import solve_continuous_are

from aircraft_dynamics import AircraftParams, build_longitudinal_matrices

# =========================
# Configs
# =========================

@dataclass
class OuterLoopConfig:
    dt_outer: float = 0.5

    # Speed reference law (forward-range standoff controller)
    # u_cmd = u_tgt + Kp*range_err + Kd*(u_tgt-u_own) + Ki*∫range_err dt + closure_bias*tanh(range_err/range_bias_scale)
    d_ref: float = 500.0
    Kv_speed: float = 0.015      # Kp on forward-range error (m/s per m)
    Kd_speed: float = 0.8        # Kd on forward-range-rate approx (m/s per (m/s))
    Ki_speed: float = 0.0        # start at 0; add later if steady-state offset remains

    range_int_min: float = -100.0
    range_int_max: float =  100.0

    closure_bias_gain: float = 3.0       # m/s
    closure_bias_scale: float = 200.0    # m, smoothness of tanh bias

    u_cmd_min: float = 140.0
    u_cmd_max: float = 220.0

    # Altitude PID -> theta_cmd
    Kp_alt: float = 0.0004
    Ki_alt: float = 0.0
    Kd_alt: float = 0.001

    theta_ref_min: float = np.deg2rad(-3.0)
    theta_ref_max: float = np.deg2rad( 3.0)
    theta_rate_lim: float = np.deg2rad(0.5)  # per outer tick
    alt_int_min: float = np.deg2rad(-2.0)
    alt_int_max: float = np.deg2rad( 2.0)

@dataclass
class ThrottlePIConfig:
    dt_inner: float = 0.005
    Kp_u: float = 1500.0
    Ki_u: float = 100.0
    u_int_min: float = -20.0
    u_int_max: float =  20.0
    antiwindup_gain: float = 0.1

@dataclass
class AttitudePDConfig:
    # Elevator sign convention chosen for B_4, B_6 < 0 in this model
    # delta_e = -K_theta*(theta_cmd-theta) + K_q*q + K_w*w
    K_theta: float = 1.5
    K_q: float = 1.0
    K_w: float = 0.0

@dataclass
class AttitudeLQRConfig:
    # x_att = [w, theta, q], input = elevator
    Q_att: np.ndarray = field(default_factory=lambda: np.diag([0.2, 20.0, 8.0]))
    R_elev: np.ndarray = field(default_factory=lambda: np.array([[4.0]]))

# =========================
# Base helper
# =========================

class BaseDecoupledController:
    """
    Common outer-loop + throttle PI logic.
    Subclasses implement elevator law.
    """
    def __init__(self, p: AircraftParams, outer_cfg: OuterLoopConfig, throttle_cfg: ThrottlePIConfig):
        self.p = p
        self.outer = outer_cfg
        self.thr = throttle_cfg

        self.u_cmd = p.U0
        self.theta_cmd = 0.0
        self.theta_cmd_prev = 0.0

        self.alt_integrator = 0.0
        self.u_integrator = 0.0

        self.range_integrator = 0.0

    def update_outer_from_target(self, x_own: np.ndarray, x_tgt: np.ndarray):
        """
        Uses target aircraft states directly.
        x = [x,z,u,w,theta,q]
        """

        # ======================================================
        # Forward-range standoff control (PD + optional I)
        # ======================================================
        range_err = (x_tgt[0] - x_own[0]) - self.outer.d_ref

        # Approx derivative of forward-range error in this toy model:
        # d/dt[(x_tgt - x_own) - d_ref] ≈ u_tgt - u_own
        range_rate_err = (x_tgt[2] - x_own[2])

        # Integrator (optional, start with Ki_speed = 0)
        self.range_integrator += range_err * self.outer.dt_outer
        self.range_integrator = float(np.clip(
            self.range_integrator,
            self.outer.range_int_min,
            self.outer.range_int_max
        ))

        # Smooth closure bias helps prevent "backing off too early"
        closure_bias = self.outer.closure_bias_gain * np.tanh(range_err / self.outer.closure_bias_scale)

        u_cmd_raw = (
            x_tgt[2]
            + self.outer.Kv_speed * range_err
            + self.outer.Kd_speed * range_rate_err
            + self.outer.Ki_speed * self.range_integrator
            + closure_bias
        )

        self.u_cmd = float(np.clip(
            u_cmd_raw,
            self.outer.u_cmd_min,
            self.outer.u_cmd_max
        ))

        # ======================================================
        # Altitude PID -> theta command
        # ======================================================
        dz_err = x_tgt[1] - x_own[1]
        dz_dot_err = x_tgt[3] - x_own[3]

        self.alt_integrator += self.outer.Ki_alt * dz_err * self.outer.dt_outer
        self.alt_integrator = float(np.clip(self.alt_integrator, self.outer.alt_int_min, self.outer.alt_int_max))

        theta_raw = self.outer.Kp_alt * dz_err + self.alt_integrator + self.outer.Kd_alt * dz_dot_err
        theta_raw = float(np.clip(theta_raw, self.outer.theta_ref_min, self.outer.theta_ref_max))

        dtheta = np.clip(theta_raw - self.theta_cmd_prev, -self.outer.theta_rate_lim, self.outer.theta_rate_lim)
        self.theta_cmd = float(self.theta_cmd_prev + dtheta)
        self.theta_cmd_prev = self.theta_cmd

        return self.u_cmd, self.theta_cmd

    def set_direct_commands(self, u_cmd=None, theta_cmd=None):
        if u_cmd is not None:
            self.u_cmd = float(u_cmd)
        if theta_cmd is not None:
            self.theta_cmd = float(np.clip(theta_cmd, self.outer.theta_ref_min, self.outer.theta_ref_max))
            self.theta_cmd_prev = self.theta_cmd

    def throttle_command(self, x6: np.ndarray, throttle_trim: float):
        """
        PI on speed u, returns unsaturated throttle command (N).
        """
        u_now = float(x6[2])
        u_err = self.u_cmd - u_now

        self.u_integrator += u_err * self.thr.dt_inner
        self.u_integrator = float(np.clip(self.u_integrator, self.thr.u_int_min, self.thr.u_int_max))

        throttle_unsat = throttle_trim + self.thr.Kp_u * u_err + self.thr.Ki_u * self.u_integrator
        return float(throttle_unsat), float(u_err)

    def throttle_antiwindup(self, throttle_cmd_applied: float, throttle_unsat: float):
        if self.thr.Ki_u > 1e-12:
            self.u_integrator += self.thr.antiwindup_gain * (throttle_cmd_applied - throttle_unsat) / self.thr.Ki_u
            self.u_integrator = float(np.clip(self.u_integrator, self.thr.u_int_min, self.thr.u_int_max))

    def elevator_command(self, x6: np.ndarray) -> float:
        raise NotImplementedError

    def control_unsat(self, x6: np.ndarray, throttle_trim: float):
        elev_unsat = self.elevator_command(x6)
        throttle_unsat, _ = self.throttle_command(x6, throttle_trim)
        return elev_unsat, throttle_unsat

# =========================
# Controller variants
# =========================

class DecoupledPIDController(BaseDecoupledController):
    """
    Throttle PI + Elevator PD (with q and w damping)
    """
    def __init__(self, p: AircraftParams, outer_cfg: OuterLoopConfig, throttle_cfg: ThrottlePIConfig, att_pd_cfg: AttitudePDConfig):
        super().__init__(p, outer_cfg, throttle_cfg)
        self.att = att_pd_cfg

    def elevator_command(self, x6: np.ndarray) -> float:
        w = float(x6[3])
        theta = float(x6[4])
        q = float(x6[5])

        theta_err = self.theta_cmd - theta
        # Sign convention for this toy model (B_4, B_6 < 0)
        elev_unsat = -self.att.K_theta * theta_err + self.att.K_q * q + self.att.K_w * w
        return float(elev_unsat)

class DecoupledLQRAttitudeController(BaseDecoupledController):
    """
    Throttle PI + Elevator LQR on x_att=[w, theta, q]
    """
    def __init__(self, p: AircraftParams, outer_cfg: OuterLoopConfig, throttle_cfg: ThrottlePIConfig, att_lqr_cfg: AttitudeLQRConfig):
        super().__init__(p, outer_cfg, throttle_cfg)
        self.att_cfg = att_lqr_cfg

        A6, B6 = build_longitudinal_matrices(p)
        idx = [3, 4, 5]  # [w, theta, q]
        self.idx_att = idx

        A_att = A6[np.ix_(idx, idx)]
        B_e = B6[np.ix_(idx, [0])]  # elevator column only

        P = solve_continuous_are(A_att, B_e, att_lqr_cfg.Q_att, att_lqr_cfg.R_elev)
        self.K_att = np.linalg.solve(att_lqr_cfg.R_elev, B_e.T @ P)  # (1,3)

    def elevator_command(self, x6: np.ndarray) -> float:
        x_att = x6[self.idx_att]                          # [w, theta, q]
        x_ref = np.array([0.0, self.theta_cmd, 0.0])     # desired [w,theta,q]
        x_tilde = x_att - x_ref
        elev_unsat = float(-(self.K_att @ x_tilde).item())
        return elev_unsat