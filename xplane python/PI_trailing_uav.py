"""
Trailing UAV Guidance & Autopilot — XPython3 Plugin
====================================================
Tuned for the Cirrus SR22 in X-Plane 12.

X-Plane provides the plant (6-DoF dynamics, sensors, actuators).
This plugin implements:
  - Virtual target trajectory generator (straight line, constant heading & altitude)
  - Trail-behind-at-distance guidance law (outer loop, 2 Hz)
  - Cascaded autopilot: heading → bank, altitude → pitch, speed → throttle
  - Inner attitude trackers: roll → aileron, pitch → elevator, sideslip → rudder

Place this file in:
  X-Plane 12/Resources/plugins/PythonPlugins/PI_trailing_uav.py

Bind 'trailuav/toggle' to a key in Settings → Keyboard to engage/disengage.

Cirrus SR22 key specs (used for tuning):
  Empty weight:  ~1025 kg     Gross weight:  ~1542 kg
  Wing area:     13.5 m²      Wing span:     11.7 m
  Stall (flaps): ~31 m/s (60 kts)
  Cruise (75%):  ~93 m/s (181 kts)
  Vne:           ~95 m/s (185 kts)
  310 HP Continental IO-550-N
  Higher wing loading (~114 kg/m²) → more inertia, less twitchy than ultralights
  Composite airframe, side-yoke controls
"""

from XPPython3 import xp
import math
import time as _time


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INNER_DT        = 0.05          # 20 Hz inner/mid loop (X-Plane flight loop)
OUTER_EVERY     = 10            # outer guidance runs every 10 inner ticks → 2 Hz
D_TRAIL         = 500.0         # desired trail distance (m) — larger for fast aircraft
DH_TRAIL        = 0.0           # altitude offset above target (m, + = above)

# ── Speed envelope (m/s) — Cirrus SR22 ──
# Stall ~31 m/s (60 kts), cruise ~93 m/s (181 kts), Vne ~95 m/s (185 kts)
V_MIN   = 36.0                 # operational floor (above stall with margin, ~70 kts)
V_MAX   = 90.0                 # operational ceiling (below Vne, ~175 kts)
V_TRIM  = 75.0                 # nominal cruise (~146 kts)

# ── Guidance gains (outer loop) ──
# The SR22 is fast and has good energy — can handle more aggressive guidance
GUID_KY  = 0.6                 # cross-track → heading correction
GUID_L1  = 600.0               # L1 lookahead distance (larger for higher speed)
GUID_KX  = 0.015               # along-track position → speed (gentle — big speed range)
GUID_KDX = 0.25                # along-track rate → speed

# ── Rate limits on guidance commands (per second) ──
LIM_CHI_RATE = math.radians(12)    # heading command rate
LIM_V_RATE   = 2.0                 # speed command rate m/s²
LIM_H_RATE   = 4.0                 # altitude command rate m/s (SR22 climbs well)

# ── Autopilot limits ──
PHI_MAX   = math.radians(30)       # bank limit — standard for GA autopilot
THETA_MAX = math.radians(15)       # pitch limit

# ── Inner-loop gains ──
# The SR22 has more inertia than an ultralight so needs firmer gains,
# but still moderate since yoke_ratio 1.0 = full deflection.

# Roll tracker: da = -Kp_phi * e_phi - Ki_phi * int_phi + Kd_phi * p
KP_PHI  = 0.55                 # proportional on roll error
KI_PHI  = 0.04                 # integral on roll error
KD_PHI  = 0.12                 # derivative (roll rate damping)

# Pitch tracker: de = -Kp_theta * e_theta - Ki_theta * int_theta + Kd_theta * q
KP_THETA = 0.60                # proportional on pitch error
KI_THETA = 0.04                # integral on pitch error
KD_THETA = 0.20                # derivative (pitch rate damping)

# Sideslip tracker: dr = -Kp_beta * beta - Ki_beta * int_beta + Kd_beta * r
KP_BETA  = 0.06                # proportional on sideslip
KI_BETA  = 0.004               # integral on sideslip
KD_BETA  = 0.015               # yaw rate damping

# Altitude mid-loop: theta_cmd = Kp_h * e_h + Ki_h * int_h - Kd_h * hdot
KP_H    = 0.025                # altitude error → pitch
KI_H    = 0.0012               # altitude integral
KD_H    = 0.12                 # climb rate damping

# Speed mid-loop: throttle = trim + Kp_V * e_V + Ki_V * int_V
KP_V    = 0.04                 # speed error → throttle (smaller range since V_MAX-V_MIN is large)
KI_V    = 0.010                # speed integral → throttle

# Heading mid-loop: chi error → desired turn rate → bank angle
K_CHI   = 1.5                  # heading error → desired turn rate
CHI_DOT_MAX = math.radians(12) # max turn rate command

# ── Actuator rate limits ──
MAX_SURF_RATE = 2.0             # max surface rate (ratio/s) — SR22 can handle faster
MAX_THR_RATE  = 0.6             # max throttle rate (ratio/s)

# ─────────────────────────────────────────────────────────────────────────────
#  TARGET TRAJECTORY PROFILE
# ─────────────────────────────────────────────────────────────────────────────
# The "target" is a virtual point moving in a straight line.
# Heading and altitude are locked to whatever the ownship has at engage time.
# Speed is set to SR22 cruise.
TGT_V0        = 75.0                # forward speed (m/s) — SR22 cruise (~146 kts)
TGT_CHI0      = 0.0                 # overridden at engage to match ownship heading
TGT_H0        = 300.0               # overridden at engage to match ownship altitude
TGT_TURN_AMP  = 0.0                 # NO turning — straight line
TGT_TURN_W    = 0.0                 # (zero frequency = constant heading)
TGT_CLIMB_AMP = 0.0                 # NO climbing — constant altitude
TGT_CLIMB_W   = 0.0


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
    A fictitious trajectory point that moves in a straight line.
    Heading and altitude are fixed at creation time.
    Position is tracked in a local NED frame with origin at the ownship's
    initial position when the plugin engages.
    """

    def __init__(self, north0, east0, alt0, chi0, V0):
        self.north = north0
        self.east  = east0
        self.alt   = alt0       # altitude (positive up, metres MSL)
        self.chi   = chi0       # heading (rad) — fixed for straight line
        self.V     = V0         # speed (m/s)
        self.t0    = 0.0        # sim time at creation

    def update(self, sim_time, dt):
        """Propagate the virtual target one timestep — straight line."""
        # Heading and altitude are constant (straight line)
        # Just propagate position forward
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
        self.Desc = "Trail-behind guidance + cascaded autopilot (Cirrus SR22 tuning)"

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
        self.disp_da        = 0.0
        self.disp_de        = 0.0
        self.disp_dr        = 0.0
        self.disp_dt        = 0.0

        # ── DataRefs — Sensors (read) ──
        self.dr_lat     = xp.findDataRef("sim/flightmodel/position/latitude")
        self.dr_lon     = xp.findDataRef("sim/flightmodel/position/longitude")
        self.dr_alt_msl = xp.findDataRef("sim/flightmodel/position/elevation")
        self.dr_alt_agl = xp.findDataRef("sim/flightmodel/position/y_agl")

        self.dr_phi     = xp.findDataRef("sim/flightmodel/position/phi")
        self.dr_theta   = xp.findDataRef("sim/flightmodel/position/theta")
        self.dr_psi     = xp.findDataRef("sim/flightmodel/position/psi")

        self.dr_alpha   = xp.findDataRef("sim/flightmodel2/position/alpha")
        self.dr_beta    = xp.findDataRef("sim/flightmodel2/position/beta")

        self.dr_P       = xp.findDataRef("sim/flightmodel/position/P")
        self.dr_Q       = xp.findDataRef("sim/flightmodel/position/Q")
        self.dr_R       = xp.findDataRef("sim/flightmodel/position/R")

        self.dr_ias     = xp.findDataRef("sim/flightmodel/position/indicated_airspeed")
        self.dr_tas     = xp.findDataRef("sim/flightmodel/position/true_airspeed")

        self.dr_vx      = xp.findDataRef("sim/flightmodel/position/local_vx")
        self.dr_vy      = xp.findDataRef("sim/flightmodel/position/local_vy")
        self.dr_vz      = xp.findDataRef("sim/flightmodel/position/local_vz")

        self.dr_sim_time = xp.findDataRef("sim/time/total_running_time_sec")

        # ── DataRefs — Actuators (write) ──
        self.dr_yoke_roll   = xp.findDataRef("sim/cockpit2/controls/yoke_roll_ratio")
        self.dr_yoke_pitch  = xp.findDataRef("sim/cockpit2/controls/yoke_pitch_ratio")
        self.dr_yoke_yaw    = xp.findDataRef("sim/cockpit2/controls/yoke_heading_ratio")
        self.dr_throttle    = xp.findDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all")

        # Override flags
        self.dr_override_js = xp.findDataRef("sim/operation/override/override_joystick")

        # ── Commands ──
        self.cmd_toggle  = xp.createCommand("trailuav/toggle",  "Toggle trailing autopilot")
        self.cmd_enable  = xp.createCommand("trailuav/enable",  "Enable trailing autopilot")
        self.cmd_disable = xp.createCommand("trailuav/disable", "Disable trailing autopilot")

        # ── Flight loop ──
        self.flight_loop = xp.createFlightLoop(self._flight_loop_cb)

        # ── Draw callback ──
        self.draw_cb = xp.registerDrawCallback(self._draw_hud, xp.Phase_Window, 0, None)

        print("[TrailUAV] Plugin loaded — tuned for Cirrus SR22.")
        print(f"[TrailUAV] Trail dist = {D_TRAIL} m  |  V range = [{V_MIN:.0f}, {V_MAX:.0f}] m/s")
        print(f"[TrailUAV] Inner = {1/INNER_DT:.0f} Hz  |  Outer = {1/(INNER_DT*OUTER_EVERY):.0f} Hz")
        print("[TrailUAV] Bind 'trailuav/toggle' to a key in Settings -> Keyboard")

        return self.Name, self.Sig, self.Desc

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

    def _engage(self):
        self.origin_lat = xp.getDataf(self.dr_lat)
        self.origin_lon = xp.getDataf(self.dr_lon)
        self.origin_alt = xp.getDataf(self.dr_alt_msl)
        self.cos_lat0   = math.cos(math.radians(self.origin_lat))

        sim_time = xp.getDataf(self.dr_sim_time)
        self.engage_time = sim_time

        own_n, own_e = self._latlon_to_ne(self.origin_lat, self.origin_lon)
        own_alt = self.origin_alt
        own_psi = math.radians(xp.getDataf(self.dr_psi))

        tgt_n = own_n + D_TRAIL * math.cos(own_psi)
        tgt_e = own_e + D_TRAIL * math.sin(own_psi)
        tgt_alt = own_alt + DH_TRAIL

        # Read airspeed first
        Va = xp.getDataf(self.dr_tas)
        if Va < 1.0:
            Va = V_TRIM

        # Target flies straight line at ownship's current speed and heading
        tgt_speed = clamp(Va, V_MIN, V_MAX)
        self.target = VirtualTarget(tgt_n, tgt_e, tgt_alt, own_psi, tgt_speed)
        self.target.t0 = sim_time

        self.ap = AutopilotState()
        self.tick_count = 0

        self.chi_cmd      = own_psi
        self.V_cmd        = clamp(Va, V_MIN, V_MAX)
        self.h_cmd        = own_alt
        self.chi_cmd_prev = own_psi
        self.V_cmd_prev   = self.V_cmd
        self.h_cmd_prev   = own_alt

        # Bumpless transfer: start from current actuator positions
        self.prev_da = xp.getDataf(self.dr_yoke_roll)
        self.prev_de = xp.getDataf(self.dr_yoke_pitch)
        self.prev_dr = xp.getDataf(self.dr_yoke_yaw)
        self.prev_dt = xp.getDataf(self.dr_throttle)

        xp.setDatai(self.dr_override_js, 1)

        self.active      = True
        self.disp_status = "ACTIVE"
        xp.scheduleFlightLoop(self.flight_loop, -1.0, 1)

        print(f"[TrailUAV] ENGAGED at lat={self.origin_lat:.4f} lon={self.origin_lon:.4f} "
              f"alt={self.origin_alt:.0f}m Va={Va:.1f}m/s hdg={math.degrees(own_psi):.0f} deg")

    def _disengage(self):
        self.active      = False
        self.disp_status = "INACTIVE"
        try:
            xp.setDatai(self.dr_override_js, 0)
        except Exception:
            pass
        xp.scheduleFlightLoop(self.flight_loop, 0.0, 1)
        print("[TrailUAV] DISENGAGED — pilot has full control.")

    def _latlon_to_ne(self, lat, lon):
        R_earth = 6378137.0
        dn = math.radians(lat - self.origin_lat) * R_earth
        de = math.radians(lon - self.origin_lon) * R_earth * self.cos_lat0
        return dn, de

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN FLIGHT LOOP (20 Hz)
    # ══════════════════════════════════════════════════════════════════════════
    def _flight_loop_cb(self, since_last, elapsed, counter, refcon):
        if not self.active:
            return 0.0

        dt = since_last if since_last > 0.0 else INNER_DT
        dt = min(dt, 0.2)  # guard against pause/resume spikes

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

        p_dps = xp.getDataf(self.dr_P)
        q_dps = xp.getDataf(self.dr_Q)
        r_dps = xp.getDataf(self.dr_R)
        p = math.radians(p_dps)
        q = math.radians(q_dps)
        r = math.radians(r_dps)

        Va = xp.getDataf(self.dr_tas)
        if Va < 1.0:
            Va = 1.0

        vx_ogl = xp.getDataf(self.dr_vx)
        vy_ogl = xp.getDataf(self.dr_vy)
        vz_ogl = xp.getDataf(self.dr_vz)

        v_north = -vz_ogl
        v_east  =  vx_ogl
        v_down  = -vy_ogl
        hdot    = -v_down

        own_n, own_e = self._latlon_to_ne(lat, lon)

        # ── Update virtual target ──
        self.target.update(sim_time, dt)

        # ── Outer loop: guidance (2 Hz) ──
        self.tick_count += 1
        if self.tick_count % OUTER_EVERY == 0:
            self._guidance_update(own_n, own_e, alt, v_north, v_east, Va,
                                  dt * OUTER_EVERY)

        # ── Mid loop: heading → bank angle ──
        chi   = psi
        e_chi = wrap_pi(self.chi_cmd - chi)

        chi_dot_des = clamp(K_CHI * e_chi, -CHI_DOT_MAX, CHI_DOT_MAX)
        phi_cmd     = math.atan2(Va * chi_dot_des, 9.80665)
        phi_cmd     = clamp(phi_cmd, -PHI_MAX, PHI_MAX)

        # ── Mid loop: altitude → pitch angle ──
        e_h = self.h_cmd - alt
        self.ap.int_h = clamp(self.ap.int_h + e_h * dt, -30, 30)
        theta_cmd = KP_H * e_h + KI_H * self.ap.int_h - KD_H * hdot
        theta_cmd = clamp(theta_cmd, -THETA_MAX, THETA_MAX)

        # ── Mid loop: speed → throttle ──
        e_V = self.V_cmd - Va
        self.ap.int_V = clamp(self.ap.int_V + e_V * dt, -8, 8)
        dt_cmd = 0.60 + KP_V * e_V + KI_V * self.ap.int_V
        dt_cmd = clamp(dt_cmd, 0.05, 1.0)

        # ── Inner loop: roll → aileron ──
        e_phi = phi_cmd - phi
        self.ap.int_phi = clamp(self.ap.int_phi + e_phi * dt, -2, 2)
        da_cmd = -(KP_PHI * e_phi + KI_PHI * self.ap.int_phi) + KD_PHI * p
        da_cmd = clamp(da_cmd, -0.7, 0.7)

        # ── Inner loop: pitch → elevator ──
        e_theta = theta_cmd - theta
        self.ap.int_theta = clamp(self.ap.int_theta + e_theta * dt, -2, 2)
        de_cmd = -(KP_THETA * e_theta + KI_THETA * self.ap.int_theta) + KD_THETA * q
        de_cmd = clamp(de_cmd, -0.6, 0.6)

        # ── Inner loop: sideslip → rudder ──
        beta_deg = math.degrees(beta)
        self.ap.int_beta = clamp(self.ap.int_beta + beta_deg * dt, -2, 2)
        dr_cmd = -(KP_BETA * beta_deg + KI_BETA * self.ap.int_beta) + KD_BETA * r_dps
        dr_cmd = clamp(dr_cmd, -0.5, 0.5)

        # ── Speed protection near stall ──
        if Va < V_MIN + 5:
            phi_cmd_prot   = 0.3 * phi_cmd
            theta_cmd_prot = min(theta_cmd, math.radians(3))
            dt_cmd         = max(dt_cmd, 0.90)
            e_phi_p   = phi_cmd_prot - phi
            da_cmd    = clamp(-(KP_PHI * e_phi_p) + KD_PHI * p, -0.5, 0.5)
            e_theta_p = theta_cmd_prot - theta
            de_cmd    = clamp(-(KP_THETA * e_theta_p) + KD_THETA * q, -0.5, 0.5)

        # ── Rate limit actuator commands ──
        da_cmd = rate_limit_scalar(self.prev_da, da_cmd, MAX_SURF_RATE, dt)
        de_cmd = rate_limit_scalar(self.prev_de, de_cmd, MAX_SURF_RATE, dt)
        dr_cmd = rate_limit_scalar(self.prev_dr, dr_cmd, MAX_SURF_RATE, dt)
        dt_cmd = rate_limit_scalar(self.prev_dt, dt_cmd, MAX_THR_RATE, dt)

        self.prev_da = da_cmd
        self.prev_de = de_cmd
        self.prev_dr = dr_cmd
        self.prev_dt = dt_cmd

        # ── Write to X-Plane ──
        xp.setDataf(self.dr_yoke_roll,  da_cmd)
        xp.setDataf(self.dr_yoke_pitch, de_cmd)
        xp.setDataf(self.dr_yoke_yaw,   dr_cmd)
        xp.setDataf(self.dr_throttle,   dt_cmd)

        # ── Update HUD ──
        self.disp_alpha     = math.degrees(alpha)
        self.disp_beta      = math.degrees(beta)
        self.disp_Va        = Va
        self.disp_alt       = alt
        self.disp_chi_cmd   = math.degrees(self.chi_cmd)
        self.disp_V_cmd     = self.V_cmd
        self.disp_h_cmd     = self.h_cmd
        self.disp_phi_cmd   = math.degrees(phi_cmd)
        self.disp_theta_cmd = math.degrees(theta_cmd)
        self.disp_da        = da_cmd
        self.disp_de        = de_cmd
        self.disp_dr        = dr_cmd
        self.disp_dt        = dt_cmd

        if self.tick_count % 20 == 0:
            print(
                f"[TrailUAV] {self.disp_status} | "
                f"rng={self.disp_range:.0f}m  e||={self.disp_epar:+.0f}  eT={self.disp_eperp:+.0f} | "
                f"Va={Va:.1f} alt={alt:.0f} | "
                f"phi_c={math.degrees(phi_cmd):+.1f} th_c={math.degrees(theta_cmd):+.1f} | "
                f"da={da_cmd:+.3f} de={de_cmd:+.3f} dr={dr_cmd:+.3f} thr={dt_cmd:.2f}"
            )

        return INNER_DT

    # ── Guidance law (outer loop) ─────────────────────────────────────────────
    def _guidance_update(self, own_n, own_e, own_alt, v_n, v_e, Va, outer_dt):
        tgt = self.target

        t_cos = math.cos(tgt.chi)
        t_sin = math.sin(tgt.chi)

        ref_n = tgt.north - D_TRAIL * t_cos
        ref_e = tgt.east  - D_TRAIL * t_sin
        h_ref = tgt.alt + DH_TRAIL

        e_n = ref_n - own_n
        e_e = ref_e - own_e

        e_par  = t_cos * e_n + t_sin * e_e
        n_cos  = -t_sin
        n_sin  =  t_cos
        e_perp = n_cos * e_n + n_sin * e_e

        v_tgt_n = tgt.V * t_cos
        v_tgt_e = tgt.V * t_sin
        de_par  = t_cos * (v_tgt_n - v_n) + t_sin * (v_tgt_e - v_e)

        chi_base    = math.atan2(t_sin, t_cos)
        chi_cmd_new = chi_base + GUID_KY * math.atan2(e_perp, GUID_L1)

        V_cmd_new = tgt.V + GUID_KX * e_par + GUID_KDX * de_par
        h_cmd_new = h_ref

        self.chi_cmd = rate_limit_angle(self.chi_cmd_prev, chi_cmd_new, LIM_CHI_RATE, outer_dt)
        self.V_cmd   = rate_limit_scalar(self.V_cmd_prev, V_cmd_new, LIM_V_RATE, outer_dt)
        self.h_cmd   = rate_limit_scalar(self.h_cmd_prev, h_cmd_new, LIM_H_RATE, outer_dt)
        self.V_cmd   = clamp(self.V_cmd, V_MIN, V_MAX)

        self.chi_cmd_prev = self.chi_cmd
        self.V_cmd_prev   = self.V_cmd
        self.h_cmd_prev   = self.h_cmd

        dp_n = tgt.north - own_n
        dp_e = tgt.east  - own_e
        dp_alt = tgt.alt - own_alt
        self.disp_range = math.sqrt(dp_n**2 + dp_e**2 + dp_alt**2)
        self.disp_epar  = e_par
        self.disp_eperp = e_perp

    # ── On-screen HUD ─────────────────────────────────────────────────────────
    def _draw_hud(self, phase, after, refcon):
        x = 20
        y = 780

        if not self.active:
            r, g, b = 0.6, 0.6, 0.6
        elif self.disp_Va < V_MIN + 5:
            r, g, b = 1.0, 0.2, 0.2
        elif abs(self.disp_epar) > 80 or abs(self.disp_eperp) > 80:
            r, g, b = 1.0, 0.7, 0.2
        else:
            r, g, b = 0.3, 1.0, 0.3

        lines = [
            f"TrailUAV [{self.disp_status}]  (Cirrus SR22)",
            f"  Range to tgt : {self.disp_range:7.0f} m",
            f"  Along-track  : {self.disp_epar:+7.0f} m",
            f"  Cross-track  : {self.disp_eperp:+7.0f} m",
            f"  ---------------------------------",
            f"  V cmd / act  : {self.disp_V_cmd:5.1f} / {self.disp_Va:5.1f} m/s  ({ms_to_kts(self.disp_Va):.0f} kts)",
            f"  h cmd / act  : {self.disp_h_cmd:5.0f} / {self.disp_alt:5.0f} m",
            f"  chi cmd      : {self.disp_chi_cmd:+6.1f} deg",
            f"  phi cmd      : {self.disp_phi_cmd:+6.1f} deg",
            f"  theta cmd    : {self.disp_theta_cmd:+6.1f} deg",
            f"  ---------------------------------",
            f"  alpha = {self.disp_alpha:+5.1f} deg   beta = {self.disp_beta:+5.1f} deg",
            f"  da={self.disp_da:+.3f}  de={self.disp_de:+.3f}  dr={self.disp_dr:+.3f}  thr={self.disp_dt:.2f}",
        ]

        for i, line in enumerate(lines):
            xp.drawString((r, g, b), x, y - i * 15, line, None, xp.Font_Proportional)

        return 1