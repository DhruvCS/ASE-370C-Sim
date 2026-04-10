"""
Trailing UAV Guidance & Autopilot — XPython3 Plugin
====================================================
Ported from MATLAB 6-DoF trailing simulation.

X-Plane provides the plant (6-DoF dynamics, sensors, actuators).
This plugin implements:
  - Virtual target trajectory generator (no real aircraft needed)
  - Trail-behind-at-distance guidance law (outer loop, 2 Hz)
  - Cascaded autopilot: heading → bank, altitude → pitch, speed → throttle
  - Inner attitude trackers: roll → aileron, pitch → elevator, sideslip → rudder

Place this file in:
  X-Plane 12/Resources/plugins/PythonPlugins/PI_trailing_uav.py

Bind 'trailuav/toggle' to a key in Settings → Keyboard to engage/disengage.
"""

from XPPython3 import xp
import math
import time as _time


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INNER_DT        = 0.05          # 20 Hz inner/mid loop (X-Plane flight loop)
OUTER_EVERY     = 10            # outer guidance runs every 10 inner ticks → 2 Hz
D_TRAIL         = 400.0         # desired trail distance (m)
DH_TRAIL        = 0.0           # altitude offset above target (m, + = above)

# Speed envelope (m/s)
V_MIN = 16.0
V_MAX = 35.0

# Guidance gains
GUID_KY  = 0.8
GUID_L1  = 500.0
GUID_KX  = 0.03
GUID_KDX = 0.40

# Rate limits on guidance commands (per second)
LIM_CHI_RATE = math.radians(20)
LIM_V_RATE   = 2.0
LIM_H_RATE   = 3.0

# Autopilot limits
PHI_MAX   = math.radians(35)
THETA_MAX = math.radians(15)

# Trim airspeed for reference (m/s)
V_TRIM = 25.0

# ─────────────────────────────────────────────────────────────────────────────
#  TARGET TRAJECTORY PROFILE
# ─────────────────────────────────────────────────────────────────────────────
# The "target" is a virtual point moving through space.
# Configure its behaviour here.
TGT_V0        = 25.0            # forward speed (m/s)
TGT_CHI0      = 0.0             # initial heading (rad, 0 = North)
TGT_H0        = 120.0           # initial altitude (m above ground)
TGT_TURN_AMP  = math.radians(8)
TGT_TURN_W    = 2 * math.pi / 60   # one full oscillation per 60 s
TGT_CLIMB_AMP = 0.5
TGT_CLIMB_W   = 2 * math.pi / 60


# ─────────────────────────────────────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rate_limit_scalar(x_prev, x_new, max_rate, dt):
    dx = clamp(x_new - x_prev, -max_rate * dt, max_rate * dt)
    return x_prev + dx


def rate_limit_angle(a_prev, a_new, max_rate, dt):
    da = clamp(wrap_pi(a_new - a_prev), -max_rate * dt, max_rate * dt)
    return wrap_pi(a_prev + da)


def kts_to_ms(kts):
    return kts * 0.514444


def ms_to_kts(ms):
    return ms / 0.514444


def ft_to_m(ft):
    return ft * 0.3048


def m_to_ft(m):
    return m / 0.3048


# ─────────────────────────────────────────────────────────────────────────────
#  VIRTUAL TARGET
# ─────────────────────────────────────────────────────────────────────────────
class VirtualTarget:
    """
    A fictitious trajectory point that moves through space.
    Position is tracked in a local NED frame with origin at the ownship's
    initial position when the plugin engages.
    """

    def __init__(self, north0, east0, alt0, chi0, V0):
        self.north = north0
        self.east  = east0
        self.alt   = alt0       # altitude (positive up)
        self.chi   = chi0       # heading (rad)
        self.V     = V0         # speed (m/s)
        self.t0    = 0.0        # sim time at creation

    def update(self, sim_time, dt):
        """Propagate the virtual target one timestep."""
        elapsed = sim_time - self.t0

        # Commanded heading (sinusoidal manoeuvre)
        chi_cmd = TGT_CHI0 + TGT_TURN_AMP * math.sin(TGT_TURN_W * elapsed)
        # Smoothly track commanded heading
        self.chi = rate_limit_angle(self.chi, chi_cmd, math.radians(15), dt)

        # Commanded altitude
        alt_cmd = TGT_H0 + TGT_CLIMB_AMP * math.sin(TGT_CLIMB_W * elapsed)
        self.alt = rate_limit_scalar(self.alt, alt_cmd, 2.0, dt)

        # Speed (constant for now)
        self.V = TGT_V0

        # Propagate position (flat earth, NED)
        self.north += self.V * math.cos(self.chi) * dt
        self.east  += self.V * math.sin(self.chi) * dt

    @property
    def pos_ne(self):
        """North-East position as a 2-tuple."""
        return (self.north, self.east)


# ─────────────────────────────────────────────────────────────────────────────
#  AUTOPILOT INTEGRATOR STATE
# ─────────────────────────────────────────────────────────────────────────────
class AutopilotState:
    def __init__(self):
        self.int_h     = 0.0
        self.int_V     = 0.0
        self.int_phi   = 0.0
        self.int_theta = 0.0
        self.int_beta  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PLUGIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class PythonInterface:

    def XPluginStart(self):
        self.Name = "Trailing UAV Autopilot"
        self.Sig  = "trailuav.guidance.autopilot"
        self.Desc = "Trail-behind guidance + cascaded autopilot for fixed-wing UAV"

        # ── State ──
        self.active       = False
        self.tick_count   = 0
        self.target       = None
        self.ap           = AutopilotState()
        self.engage_time  = 0.0

        # Previous guidance commands (for rate limiting)
        self.chi_cmd_prev = 0.0
        self.V_cmd_prev   = V_TRIM
        self.h_cmd_prev   = TGT_H0

        # Current guidance commands (held between outer-loop updates)
        self.chi_cmd = 0.0
        self.V_cmd   = V_TRIM
        self.h_cmd   = TGT_H0

        # Previous outputs for rate limiting actuator commands
        self.prev_da = 0.0
        self.prev_de = 0.0
        self.prev_dr = 0.0
        self.prev_dt = 0.5

        # Ownship local-frame origin (lat/lon/alt at engage time)
        self.origin_lat = 0.0
        self.origin_lon = 0.0
        self.origin_alt = 0.0
        self.cos_lat0   = 1.0

        # HUD display values
        self.disp_status    = "INACTIVE"
        self.disp_range     = 0.0
        self.disp_epar      = 0.0
        self.disp_eperp     = 0.0
        self.disp_chi_cmd   = 0.0
        self.disp_V_cmd     = 0.0
        self.disp_h_cmd     = 0.0
        self.disp_phi_cmd   = 0.0
        self.disp_theta_cmd = 0.0
        self.disp_alpha     = 0.0
        self.disp_beta      = 0.0
        self.disp_Va        = 0.0
        self.disp_alt       = 0.0

        # ── DataRefs — Sensors (read) ──
        self.dr_lat     = xp.findDataRef("sim/flightmodel/position/latitude")
        self.dr_lon     = xp.findDataRef("sim/flightmodel/position/longitude")
        self.dr_alt_msl = xp.findDataRef("sim/flightmodel/position/elevation")   # metres MSL
        self.dr_alt_agl = xp.findDataRef("sim/flightmodel/position/y_agl")       # metres AGL

        self.dr_phi     = xp.findDataRef("sim/flightmodel/position/phi")          # deg
        self.dr_theta   = xp.findDataRef("sim/flightmodel/position/theta")        # deg
        self.dr_psi     = xp.findDataRef("sim/flightmodel/position/psi")          # deg (true)

        self.dr_alpha   = xp.findDataRef("sim/flightmodel2/position/alpha")       # deg
        self.dr_beta    = xp.findDataRef("sim/flightmodel2/position/beta")        # deg

        self.dr_P       = xp.findDataRef("sim/flightmodel/position/P")            # deg/s
        self.dr_Q       = xp.findDataRef("sim/flightmodel/position/Q")            # deg/s
        self.dr_R       = xp.findDataRef("sim/flightmodel/position/R")            # deg/s

        self.dr_ias     = xp.findDataRef("sim/flightmodel/position/indicated_airspeed")  # m/s
        self.dr_tas     = xp.findDataRef("sim/flightmodel/position/true_airspeed")       # m/s

        self.dr_vx      = xp.findDataRef("sim/flightmodel/position/local_vx")     # m/s OGL
        self.dr_vy      = xp.findDataRef("sim/flightmodel/position/local_vy")     # m/s OGL
        self.dr_vz      = xp.findDataRef("sim/flightmodel/position/local_vz")     # m/s OGL

        self.dr_sim_time = xp.findDataRef("sim/time/total_running_time_sec")

        # ── DataRefs — Actuators (write) ──
        self.dr_yoke_roll   = xp.findDataRef("sim/cockpit2/controls/yoke_roll_ratio")    # -1..1
        self.dr_yoke_pitch  = xp.findDataRef("sim/cockpit2/controls/yoke_pitch_ratio")   # -1..1
        self.dr_yoke_yaw    = xp.findDataRef("sim/cockpit2/controls/yoke_heading_ratio") # -1..1
        self.dr_throttle    = xp.findDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all")  # 0..1

        # Override flags — take control away from default AI / joystick
        self.dr_override_js = xp.findDataRef("sim/operation/override/override_joystick")

        # ── Commands ──
        self.cmd_toggle  = xp.createCommand("trailuav/toggle",  "Toggle trailing autopilot")
        self.cmd_enable  = xp.createCommand("trailuav/enable",  "Enable trailing autopilot")
        self.cmd_disable = xp.createCommand("trailuav/disable", "Disable trailing autopilot")

        # ── Flight loop ──
        self.flight_loop = xp.createFlightLoop(self._flight_loop_cb)

        # ── Draw callback ──
        self.draw_cb = xp.registerDrawCallback(self._draw_hud, xp.Phase_Window, 0, None)

        print("[TrailUAV] Plugin loaded.")
        print(f"[TrailUAV] Trail distance = {D_TRAIL} m  |  Inner loop = {1/INNER_DT:.0f} Hz  |  Outer loop = {1/(INNER_DT*OUTER_EVERY):.0f} Hz")
        print("[TrailUAV] Bind 'trailuav/toggle' to a key in Settings → Keyboard")

        return self.Name, self.Sig, self.Desc

    # ── Plugin lifecycle ──────────────────────────────────────────────────────
    def XPluginStop(self):
        self._disengage()
        xp.destroyFlightLoop(self.flight_loop)
        xp.unregisterDrawCallback(self.draw_cb, xp.Phase_Window, 0, None)
        print("[TrailUAV] Plugin stopped.")

    def XPluginEnable(self):
        xp.registerCommandHandler(self.cmd_toggle,  self._cmd_handler, 1, None)
        xp.registerCommandHandler(self.cmd_enable,  self._cmd_handler, 1, None)
        xp.registerCommandHandler(self.cmd_disable, self._cmd_handler, 1, None)
        return 1

    def XPluginDisable(self):
        self._disengage()
        xp.unregisterCommandHandler(self.cmd_toggle,  self._cmd_handler, 1, None)
        xp.unregisterCommandHandler(self.cmd_enable,  self._cmd_handler, 1, None)
        xp.unregisterCommandHandler(self.cmd_disable, self._cmd_handler, 1, None)

    def XPluginReceiveMessage(self, fromWho, message, param):
        pass

    # ── Command handler ───────────────────────────────────────────────────────
    def _cmd_handler(self, cmd_ref, phase, refcon):
        if phase != xp.CommandBegin:
            return 1
        if cmd_ref == self.cmd_toggle:
            if self.active:
                self._disengage()
            else:
                self._engage()
        elif cmd_ref == self.cmd_enable:
            if not self.active:
                self._engage()
        elif cmd_ref == self.cmd_disable:
            if self.active:
                self._disengage()
        return 0

    # ── Engage / Disengage ────────────────────────────────────────────────────
    def _engage(self):
        # Capture current position as local-frame origin
        self.origin_lat = xp.getDataf(self.dr_lat)
        self.origin_lon = xp.getDataf(self.dr_lon)
        self.origin_alt = xp.getDataf(self.dr_alt_msl)
        self.cos_lat0   = math.cos(math.radians(self.origin_lat))

        sim_time = xp.getDataf(self.dr_sim_time)
        self.engage_time = sim_time

        # Get ownship current state for initialisation
        own_n, own_e = self._latlon_to_ne(self.origin_lat, self.origin_lon)
        own_alt = self.origin_alt
        own_psi = math.radians(xp.getDataf(self.dr_psi))

        # Place virtual target ahead of ownship along current heading
        tgt_n = own_n + D_TRAIL * math.cos(own_psi)
        tgt_e = own_e + D_TRAIL * math.sin(own_psi)
        tgt_alt = own_alt + DH_TRAIL

        self.target = VirtualTarget(tgt_n, tgt_e, tgt_alt, own_psi, TGT_V0)
        self.target.t0 = sim_time

        # Initialise autopilot
        self.ap = AutopilotState()
        self.tick_count = 0

        # Initialise guidance commands to current state
        Va = xp.getDataf(self.dr_tas)
        if Va < 1.0:
            Va = V_TRIM
        self.chi_cmd      = own_psi
        self.V_cmd        = Va
        self.h_cmd        = own_alt
        self.chi_cmd_prev = own_psi
        self.V_cmd_prev   = Va
        self.h_cmd_prev   = own_alt

        # Reset previous actuator outputs
        self.prev_da = 0.0
        self.prev_de = 0.0
        self.prev_dr = 0.0
        self.prev_dt = 0.5

        # Take over control surfaces
        xp.setDatai(self.dr_override_js, 1)

        self.active      = True
        self.disp_status = "ACTIVE"
        xp.scheduleFlightLoop(self.flight_loop, -1.0, 1)

        print(f"[TrailUAV] ENGAGED at lat={self.origin_lat:.4f} lon={self.origin_lon:.4f} alt={self.origin_alt:.0f}m")

    def _disengage(self):
        self.active      = False
        self.disp_status = "INACTIVE"

        # Release control surfaces
        try:
            xp.setDatai(self.dr_override_js, 0)
        except Exception:
            pass

        xp.scheduleFlightLoop(self.flight_loop, 0.0, 1)
        print("[TrailUAV] DISENGAGED — pilot has full control.")

    # ── Coordinate conversion ─────────────────────────────────────────────────
    def _latlon_to_ne(self, lat, lon):
        """Convert lat/lon to local North-East (metres) from origin."""
        R_earth = 6378137.0  # WGS84 semi-major axis
        dn = math.radians(lat - self.origin_lat) * R_earth
        de = math.radians(lon - self.origin_lon) * R_earth * self.cos_lat0
        return dn, de

    # ── Main flight loop ──────────────────────────────────────────────────────
    def _flight_loop_cb(self, since_last, elapsed, counter, refcon):
        if not self.active:
            return 0.0

        dt = since_last if since_last > 0.0 else INNER_DT
        sim_time = xp.getDataf(self.dr_sim_time)

        # ── Read sensors ──
        lat   = xp.getDataf(self.dr_lat)
        lon   = xp.getDataf(self.dr_lon)
        alt   = xp.getDataf(self.dr_alt_msl)

        phi   = math.radians(xp.getDataf(self.dr_phi))
        theta = math.radians(xp.getDataf(self.dr_theta))
        psi   = math.radians(xp.getDataf(self.dr_psi))

        alpha = math.radians(xp.getDataf(self.dr_alpha))
        beta  = math.radians(xp.getDataf(self.dr_beta))

        p_dps = xp.getDataf(self.dr_P)    # deg/s
        q_dps = xp.getDataf(self.dr_Q)
        r_dps = xp.getDataf(self.dr_R)
        p = math.radians(p_dps)
        q = math.radians(q_dps)
        r = math.radians(r_dps)

        Va = xp.getDataf(self.dr_tas)
        if Va < 1.0:
            Va = 1.0

        # Ownship NED velocity (X-Plane uses OpenGL: x=East, y=Up, z=South)
        vx_ogl = xp.getDataf(self.dr_vx)   # East
        vy_ogl = xp.getDataf(self.dr_vy)   # Up
        vz_ogl = xp.getDataf(self.dr_vz)   # South

        v_north = -vz_ogl
        v_east  =  vx_ogl
        v_down  = -vy_ogl
        hdot    = -v_down   # climb rate (positive up)

        # Ownship NED position in local frame
        own_n, own_e = self._latlon_to_ne(lat, lon)

        # Ownship ground-track heading
        chi_own = math.atan2(v_east, v_north)

        # ── Update virtual target ──
        self.target.update(sim_time, dt)

        # ── Outer loop: guidance (2 Hz) ──
        self.tick_count += 1
        if self.tick_count % OUTER_EVERY == 0:
            self._guidance_update(
                own_n, own_e, alt,
                v_north, v_east,
                chi_own, Va,
                dt * OUTER_EVERY   # outer_dt
            )

        # ── Mid loop: autopilot (heading→bank, alt→pitch, speed→throttle) ──
        chi   = psi  # no-wind assumption
        e_chi = wrap_pi(self.chi_cmd - chi)

        # Heading → bank angle (coordinated turn)
        chi_dot_des = clamp(2.0 * e_chi, -math.radians(15), math.radians(15))
        phi_cmd     = math.atan2(Va * chi_dot_des, 9.80665)
        phi_cmd     = clamp(phi_cmd, -PHI_MAX, PHI_MAX)

        # Altitude → pitch angle
        e_h = self.h_cmd - alt
        self.ap.int_h = clamp(self.ap.int_h + e_h * dt, -50, 50)
        theta_cmd = 0.04 * e_h + 0.0016 * self.ap.int_h - 0.16 * hdot
        theta_cmd = clamp(theta_cmd, -THETA_MAX, THETA_MAX)

        # Speed → throttle
        e_V = self.V_cmd - Va
        self.ap.int_V = clamp(self.ap.int_V + e_V * dt, -10, 10)
        dt_trim = 0.5   # rough trim throttle estimate
        dt_cmd  = dt_trim + 0.8 * e_V / (V_MAX - V_MIN) + 0.2 * self.ap.int_V / (V_MAX - V_MIN)
        dt_cmd  = clamp(dt_cmd, 0.0, 1.0)

        # ── Inner loop: attitude trackers → surface commands ──

        # Roll → aileron
        e_phi = phi_cmd - phi
        self.ap.int_phi = clamp(self.ap.int_phi + e_phi * dt, -5, 5)
        da_cmd = -2.0 * e_phi - 0.15 * self.ap.int_phi + 0.25 * p
        da_cmd = clamp(da_cmd, -1.0, 1.0)

        # Pitch → elevator
        e_theta = theta_cmd - theta
        self.ap.int_theta = clamp(self.ap.int_theta + e_theta * dt, -5, 5)
        de_cmd = -1.5 * e_theta - 0.12 * self.ap.int_theta + 0.6 * q
        de_cmd = clamp(de_cmd, -1.0, 1.0)

        # Sideslip → rudder (coordinated turn)
        beta_val = math.degrees(beta)  # use small beta in degrees for gain scaling
        self.ap.int_beta = clamp(self.ap.int_beta + beta_val * dt, -3, 3)
        dr_cmd = -0.08 * beta_val - 0.005 * self.ap.int_beta + 0.015 * r_dps
        dr_cmd = clamp(dr_cmd, -1.0, 1.0)

        # ── Speed protection near stall ──
        if Va < V_MIN + 1:
            phi_cmd   = 0.5 * phi_cmd
            theta_cmd = min(theta_cmd, math.radians(5))
            dt_cmd    = max(dt_cmd, 0.9)
            # Recompute with reduced demands
            e_phi_s = phi_cmd - phi
            da_cmd  = clamp(-2.0 * e_phi_s + 0.25 * p, -1.0, 1.0)
            e_theta_s = theta_cmd - theta
            de_cmd  = clamp(-1.5 * e_theta_s + 0.6 * q, -1.0, 1.0)

        # ── Rate limit actuator commands for smoothness ──
        max_surf_rate = 4.0   # full deflection per second
        max_thr_rate  = 1.0
        da_cmd = rate_limit_scalar(self.prev_da, da_cmd, max_surf_rate, dt)
        de_cmd = rate_limit_scalar(self.prev_de, de_cmd, max_surf_rate, dt)
        dr_cmd = rate_limit_scalar(self.prev_dr, dr_cmd, max_surf_rate, dt)
        dt_cmd = rate_limit_scalar(self.prev_dt, dt_cmd, max_thr_rate, dt)

        self.prev_da = da_cmd
        self.prev_de = de_cmd
        self.prev_dr = dr_cmd
        self.prev_dt = dt_cmd

        # ── Write actuator commands to X-Plane ──
        xp.setDataf(self.dr_yoke_roll,  da_cmd)
        xp.setDataf(self.dr_yoke_pitch, de_cmd)
        xp.setDataf(self.dr_yoke_yaw,   dr_cmd)
        xp.setDataf(self.dr_throttle,   dt_cmd)

        # ── Update HUD display values ──
        self.disp_alpha     = math.degrees(alpha)
        self.disp_beta      = math.degrees(beta)
        self.disp_Va        = Va
        self.disp_alt       = alt
        self.disp_chi_cmd   = math.degrees(self.chi_cmd)
        self.disp_V_cmd     = self.V_cmd
        self.disp_h_cmd     = self.h_cmd
        self.disp_phi_cmd   = math.degrees(phi_cmd)
        self.disp_theta_cmd = math.degrees(theta_cmd)

        # Log periodically (~1 Hz)
        if self.tick_count % 20 == 0:
            print(
                f"[TrailUAV] {self.disp_status} | "
                f"range={self.disp_range:.0f}m  e‖={self.disp_epar:.0f}  e⊥={self.disp_eperp:.0f} | "
                f"Va={Va:.1f}m/s  alt={alt:.0f}m | "
                f"da={da_cmd:+.2f} de={de_cmd:+.2f} dr={dr_cmd:+.2f} thr={dt_cmd:.2f}"
            )

        return INNER_DT

    # ── Guidance law (outer loop) ─────────────────────────────────────────────
    def _guidance_update(self, own_n, own_e, own_alt, v_n, v_e, chi_own, Va, outer_dt):
        """
        Trail-behind-at-distance guidance.
        Computes chi_cmd, V_cmd, h_cmd from virtual target state.
        """
        tgt = self.target

        # Target heading unit vector
        t_cos = math.cos(tgt.chi)
        t_sin = math.sin(tgt.chi)

        # Trail reference point (behind target along its heading)
        ref_n = tgt.north - D_TRAIL * t_cos
        ref_e = tgt.east  - D_TRAIL * t_sin

        # Altitude reference
        h_ref = tgt.alt + DH_TRAIL

        # Error in trail frame
        e_n = ref_n - own_n
        e_e = ref_e - own_e

        # Along-track (parallel to target heading)
        e_par = t_cos * e_n + t_sin * e_e

        # Cross-track (perpendicular, positive = left of track)
        n_cos = -t_sin
        n_sin =  t_cos
        e_perp = n_cos * e_n + n_sin * e_e

        # Closure rate (along track)
        v_tgt_n = tgt.V * t_cos
        v_tgt_e = tgt.V * t_sin
        de_par = t_cos * (v_tgt_n - v_n) + t_sin * (v_tgt_e - v_e)

        # Course command (base heading + cross-track correction)
        chi_base = math.atan2(t_sin, t_cos)
        chi_cmd_new = chi_base + GUID_KY * math.atan2(e_perp, GUID_L1)

        # Speed command (target speed + along-track PD)
        V_cmd_new = tgt.V + GUID_KX * e_par + GUID_KDX * de_par

        # Altitude command
        h_cmd_new = h_ref

        # Rate limit
        self.chi_cmd = rate_limit_angle(self.chi_cmd_prev, chi_cmd_new, LIM_CHI_RATE, outer_dt)
        self.V_cmd   = rate_limit_scalar(self.V_cmd_prev, V_cmd_new, LIM_V_RATE, outer_dt)
        self.h_cmd   = rate_limit_scalar(self.h_cmd_prev, h_cmd_new, LIM_H_RATE, outer_dt)

        self.V_cmd = clamp(self.V_cmd, V_MIN, V_MAX)

        self.chi_cmd_prev = self.chi_cmd
        self.V_cmd_prev   = self.V_cmd
        self.h_cmd_prev   = self.h_cmd

        # Update display
        dp_n = tgt.north - own_n
        dp_e = tgt.east  - own_e
        dp_alt = tgt.alt - own_alt
        self.disp_range = math.sqrt(dp_n**2 + dp_e**2 + dp_alt**2)
        self.disp_epar  = e_par
        self.disp_eperp = e_perp

    # ── On-screen HUD ─────────────────────────────────────────────────────────
    def _draw_hud(self, phase, after, refcon):
        x = 20
        y = 750

        if not self.active:
            r, g, b = 0.6, 0.6, 0.6
        elif abs(self.disp_epar) > 100 or abs(self.disp_eperp) > 100:
            r, g, b = 1.0, 0.7, 0.2   # orange — large tracking error
        else:
            r, g, b = 0.3, 1.0, 0.3   # green — tracking well

        lines = [
            f"TrailUAV [{self.disp_status}]",
            f"  Range to tgt : {self.disp_range:7.0f} m",
            f"  Along-track  : {self.disp_epar:+7.0f} m",
            f"  Cross-track  : {self.disp_eperp:+7.0f} m",
            f"  ─────────────────────────────────",
            f"  V cmd / act  : {self.disp_V_cmd:5.1f} / {self.disp_Va:5.1f} m/s",
            f"  h cmd / act  : {self.disp_h_cmd:5.0f} / {self.disp_alt:5.0f} m",
            f"  χ cmd        : {self.disp_chi_cmd:+6.1f} deg",
            f"  φ cmd        : {self.disp_phi_cmd:+6.1f} deg",
            f"  θ cmd        : {self.disp_theta_cmd:+6.1f} deg",
            f"  ─────────────────────────────────",
            f"  α = {self.disp_alpha:+5.1f}°   β = {self.disp_beta:+5.1f}°",
        ]

        for i, line in enumerate(lines):
            xp.drawString((r, g, b), x, y - i * 15, line, None, xp.Font_Proportional)

        return 1