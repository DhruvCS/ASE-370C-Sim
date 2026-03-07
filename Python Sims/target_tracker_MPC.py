"""
Aircraft Pursuit Simulator — Robust Multi-Rate MPC
===================================================

THREE-RATE LOOP
---------------
  200 Hz  RK4 plant integration + actuator ZOH
   10 Hz  MPC QP solve  (lsq_linear BVLS, ~0.4 ms, budget 100 ms)
    2 Hz  Outer guidance (range + altitude PID)

Implementation cooked up by Claude Sonnet 4.6
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, expm
from scipy.optimize import lsq_linear
import time as _time

# ─────────────────────────────────────────────────────────────
#  PLANT
# ─────────────────────────────────────────────────────────────
m           = 8431.375
A_w         = 27.87
c           = 3.45
I           = 80187.14101685
U0          = 185.0
rho         = 1.0
g           = 9.81
CD          = 0.02
CL_alpha    = 5.0
CL_gamma    = 0.4
CM_alpha    = -0.7
CM_thetadot = -10.0
CM_gamma    = -1.2

A_33 = -CD  * rho * U0 * A_w / m
A_44 = -CL_alpha * rho * U0 * A_w / (2*m)
A_64 = CM_alpha  * rho * U0 * c * A_w / (2*I)
A_66 = CM_thetadot * rho * U0 * c**2 * A_w / (4*I)
B_t  = 1.0 / m
B_4  = -CL_gamma * rho * U0**2 * A_w / (2*m)
B_6  = CM_gamma  * rho * U0**2 * c * A_w / (2*I)

# 6-state full plant [x, z, u, w, theta, q]
A6 = np.array([
    [0, 0,    1,    0,     0,    0],
    [0, 0,    0,    1,     0,    0],
    [0, 0, A_33,    0,    -g,    0],
    [0, 0,    0, A_44,     0,   U0],
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

A4 = A6[2:6, 2:6]   # attitude subsystem [u, w, theta, q]
B4 = B6[2:6, :]
nx, nu = 4, 2

# ─────────────────────────────────────────────────────────────
#  TRIM FEEDFORWARD  (key robustness fix)
# ─────────────────────────────────────────────────────────────
def trim_feedforward(u_cmd, theta_cmd):
    """
    Compute the equilibrium control (u_ff) needed to hold [u_cmd, 0, theta_cmd, 0]
    as a steady state of the LINEARISED system.

    From A4 @ x4_ref + B4 @ u_ff = 0:
      Row 0:  A_33*u_cmd - g*theta_cmd + B_t*thr_ff = 0
              → thr_ff = (g*theta_cmd - A_33*u_cmd) / B_t
      Row 1:  B_4*elev_ff = 0  (w_ref=0, q_ref=0)
              → elev_ff = 0
    """
    thr_ff  = (g * theta_cmd - A_33 * u_cmd) / B_t
    elev_ff = 0.0
    return np.array([elev_ff, thr_ff])

# ─────────────────────────────────────────────────────────────
#  TIMING
# ─────────────────────────────────────────────────────────────
dt_inner    = 0.005   # 200 Hz — integration
dt_mpc      = 0.1     # 10  Hz — MPC solve
dt_outer    = 0.5     #  2  Hz — guidance
mpc_every   = int(round(dt_mpc   / dt_inner))   # 20
outer_every = int(round(dt_outer / dt_inner))   # 100

# ─────────────────────────────────────────────────────────────
#  DISCRETISE PLANT AT MPC PREDICTION RATE (dt_mpc = 0.1 s)
# ─────────────────────────────────────────────────────────────
Mexp = np.zeros((nx + nu, nx + nu))
Mexp[:nx, :nx] = A4
Mexp[:nx, nx:] = B4
eMexp = expm(Mexp * dt_mpc)
Ad = eMexp[:nx, :nx]
Bd = eMexp[:nx, nx:]

# ─────────────────────────────────────────────────────────────
#  MPC SETUP
# ─────────────────────────────────────────────────────────────
N = 20   # horizon: 20 × 0.1 s = 2 s

Q_mpc = np.diag([2.0, 0.5, 20.0, 10.0])    # [u, w, theta, q]
R_mpc = np.diag([3.0, 0.3])                # [elevator, throttle]
P_lqr = solve_continuous_are(A4, B4, Q_mpc, R_mpc)   # terminal cost

# Absolute actuator limits
elev_max     = np.deg2rad(25)
throttle_max = 30e3

# Rate limits on the MPC deviation variable Δu between consecutive MPC solves
# (dt_mpc = 0.1 s between solves)
delta_elev_max_mpc = np.deg2rad(200) * dt_mpc   # 20 deg per MPC step
delta_thr_max_mpc  = 8e3 * dt_mpc               # 800 N per MPC step

# Pre-build prediction matrices (constant — LTI plant)
Su = np.zeros((N * nx, N * nu))
Sx = np.zeros((N * nx, nx))
Ai = np.eye(nx)
for k in range(N):
    Ai_new = Ad @ Ai
    Sx[k*nx:(k+1)*nx, :] = Ai_new
    for j in range(k + 1):
        Su[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(Ad, k - j) @ Bd
    Ai = Ai_new

Q_bar = np.zeros((N * nx, N * nx))
for k in range(N - 1):
    Q_bar[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = Q_mpc
Q_bar[(N-1)*nx:, (N-1)*nx:] = P_lqr   # terminal LQR cost

R_bar = np.kron(np.eye(N), R_mpc)

H_qp  = Su.T @ Q_bar @ Su + R_bar
H_qp  = (H_qp + H_qp.T) / 2
L_chol = np.linalg.cholesky(H_qp)

def solve_mpc(e0, du_prev, u_ff):
    """
    Solve the deviation MPC QP.

    State error:  e0 = x4_current - x4_ref
    Control var:  Δu = u_mpc (deviation from feedforward)
    u_total = Δu + u_ff  (applied to plant)

    Bounds on Δu derived from residual actuator authority:
      Δelev   ∈ [-elev_max - u_ff[0], elev_max - u_ff[0]]
      Δthr    ∈ [-thr_max  - u_ff[1], thr_max  - u_ff[1]]
    Rate limit on first step relative to previous Δu (du_prev).
    """
    f_qp = Su.T @ Q_bar @ (Sx @ e0)
    d    = np.linalg.solve(L_chol.T, -f_qp)

    # Position bounds in Δu space (residual authority)
    U_lo = np.tile([-elev_max - u_ff[0], -throttle_max - u_ff[1]], N)
    U_hi = np.tile([ elev_max - u_ff[0],  throttle_max - u_ff[1]], N)

    # Rate limit on first step (du_prev is the previous Δu[0])
    U_lo[0] = max(U_lo[0], du_prev[0] - delta_elev_max_mpc)
    U_hi[0] = min(U_hi[0], du_prev[0] + delta_elev_max_mpc)
    U_lo[1] = max(U_lo[1], du_prev[1] - delta_thr_max_mpc)
    U_hi[1] = min(U_hi[1], du_prev[1] + delta_thr_max_mpc)

    res = lsq_linear(L_chol, d, bounds=(U_lo, U_hi),
                     method='bvls', lsq_solver='exact', verbose=0)
    return res.x[:nu]   # Δu[0] — receding horizon

# ─────────────────────────────────────────────────────────────
#  OUTER LOOP PARAMETERS
# ─────────────────────────────────────────────────────────────
d_ref     = 200.0
Kv_speed  = 0.05
u_cmd_min = 140.0
u_cmd_max = 220.0

Kp_alt    = -0.003
Ki_alt    = 0.0
Kd_alt    = -0.003

theta_ref_min  = np.deg2rad(-5.0)
theta_ref_max  = np.deg2rad( 5.0)
theta_rate_lim = np.deg2rad(1.5)
alt_int_min    = np.deg2rad(-3.0)
alt_int_max    = np.deg2rad( 3.0)

# ─────────────────────────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────────────────────────
Tf = 120.0
n  = int(Tf / dt_inner) + 1
t  = np.linspace(0, Tf, n)

target_x0, target_z0 = 800.0, 2000.0
target_vx   = 200.0
target_amp  = 10.0
target_freq = 0.001

x_tgt    = target_x0 + target_vx * t
z_tgt    = target_z0 + target_amp * np.sin(2 * np.pi * target_freq * t)
zdot_tgt = target_amp * 2 * np.pi * target_freq * np.cos(2 * np.pi * target_freq * t)

X               = np.zeros((6, n))
X[:, 0]         = np.array([0.0, 2000.0, U0, 0.0, 0.0, 0.0])
U_ctrl          = np.zeros((2, n))
U_ff_hist       = np.zeros((2, n))
u_ref_hist      = np.zeros(n)
theta_ref_hist  = np.zeros(n)
range_hist      = np.zeros(n)
dz_hist         = np.zeros(n)
outer_tick_hist = np.zeros(n, dtype=bool)
mpc_tick_hist   = np.zeros(n, dtype=bool)
solve_time_hist = np.zeros(n)

# State
u_cmd          = U0
theta_cmd      = 0.0
theta_cmd_prev = 0.0
u_ff           = trim_feedforward(U0, 0.0)   # initialise at trim
du_prev        = np.zeros(nu)                 # previous MPC deviation
du_mpc         = np.zeros(nu)                 # held between solves (ZOH)
alt_integrator = 0.0

for i in range(n - 1):
    x6 = X[:, i]

    # ── Outer loop: 2 Hz ─────────────────────────────────────
    if i % outer_every == 0:
        range_err = (x_tgt[i] - x6[0]) - d_ref
        dz_err    =  z_tgt[i] - x6[1]
        dz_dot    = zdot_tgt[i] - x6[3]

        u_cmd = np.clip(target_vx + Kv_speed * range_err, u_cmd_min, u_cmd_max)

        alt_integrator += Ki_alt * dz_err * dt_outer
        alt_integrator  = np.clip(alt_integrator, alt_int_min, alt_int_max)

        theta_raw  = np.clip(Kp_alt * dz_err + alt_integrator + Kd_alt * dz_dot,
                             theta_ref_min, theta_ref_max)
        delta_t        = np.clip(theta_raw - theta_cmd_prev, -theta_rate_lim, theta_rate_lim)
        theta_cmd      = theta_cmd_prev + delta_t
        theta_cmd_prev = theta_cmd

        # Update trim feedforward for new reference
        u_ff = trim_feedforward(u_cmd, theta_cmd)
        outer_tick_hist[i] = True

        target_vx += 0.1

    u_ref_hist[i]     = u_cmd
    theta_ref_hist[i] = theta_cmd
    range_hist[i]     = x_tgt[i] - x6[0]
    dz_hist[i]        = z_tgt[i] - x6[1]

    # ── MPC: 10 Hz solve, 200 Hz apply ───────────────────────
    if i % mpc_every == 0:
        x4_ref = np.array([u_cmd, 0.0, theta_cmd, 0.0])
        e0     = x6[2:6] - x4_ref

        t0      = _time.perf_counter()
        du_mpc  = solve_mpc(e0, du_prev, u_ff)
        solve_time_hist[i] = (_time.perf_counter() - t0) * 1000

        du_prev = du_mpc.copy()
        mpc_tick_hist[i] = True

    # Total control = feedforward trim + MPC deviation
    ctrl = u_ff + du_mpc
    # Safety clamp (should never be active with correct bounds, but belt-and-suspenders)
    ctrl[0] = np.clip(ctrl[0], -elev_max,     elev_max)
    ctrl[1] = np.clip(ctrl[1], -throttle_max, throttle_max)

    U_ctrl[:, i]   = ctrl
    U_ff_hist[:, i] = u_ff

    # ── RK4 at 200 Hz ────────────────────────────────────────
    def f(x): return A6 @ x + B6 @ ctrl
    k1 = f(x6)
    k2 = f(x6 + 0.5 * dt_inner * k1)
    k3 = f(x6 + 0.5 * dt_inner * k2)
    k4 = f(x6 +       dt_inner * k3)
    X[:, i+1] = x6 + (dt_inner / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Fill last column
U_ctrl[:, -1]   = U_ctrl[:, -2]
U_ff_hist[:, -1] = U_ff_hist[:, -2]
u_ref_hist[-1]  = u_ref_hist[-2]
theta_ref_hist[-1] = theta_ref_hist[-2]
range_hist[-1]  = range_hist[-2]
dz_hist[-1]     = dz_hist[-2]

# ─────────────────────────────────────────────────────────────
#  DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
elev_deg  = np.rad2deg(U_ctrl[0, :])
elev_rate = np.abs(np.diff(elev_deg)) / dt_inner
fft_p     = np.abs(np.fft.rfft(elev_deg - elev_deg.mean()))**2
freqs_fft = np.fft.rfftfreq(n, d=dt_inner)
dom_freq  = freqs_fft[np.argmax(fft_p)]
st        = solve_time_hist[mpc_tick_hist]

print("=" * 60)
print("ROBUST MPC SIMULATION RESULTS")
print("=" * 60)
print(f"Loop rates: Integration {1/dt_inner:.0f} Hz | "
      f"MPC {1/dt_mpc:.0f} Hz | Guidance {1/dt_outer:.0f} Hz")
print(f"MPC horizon: N={N} × {dt_mpc:.2f}s = {N*dt_mpc:.1f}s")
print(f"Solve time:  mean={st.mean():.3f} ms  max={st.max():.3f} ms  "
      f"(budget {dt_mpc*1000:.0f} ms,  margin {dt_mpc*1000/st.mean():.0f}×)")
print()
print(f"Elevator:    max={np.abs(elev_deg).max():.2f}°  "
      f"std={elev_deg.std():.2f}°  sat={np.sum(np.abs(elev_deg)>=24.9)/n*100:.2f}%")
print(f"Elev rate:   max={elev_rate.max():.1f}°/s  mean={elev_rate.mean():.2f}°/s")
print(f"Dom. freq:   {dom_freq:.3f} Hz")
print(f"Max |pitch|: {np.rad2deg(np.abs(X[4,:])).max():.2f}°")
print(f"Alt error:   std={dz_hist.std():.1f} m  max={np.abs(dz_hist).max():.1f} m")
ff_thr_kN = U_ff_hist[1, :] / 1e3
print(f"Thr ff:      mean={ff_thr_kN.mean():.2f} kN  "
      f"range=[{ff_thr_kN.min():.2f}, {ff_thr_kN.max():.2f}] kN")
print(f"Standoff:    final={range_hist[-1]:.1f} m  (target {d_ref} m)  "
      f"std={range_hist.std():.1f} m")

# ─────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────
fig, axs = plt.subplots(3, 3, figsize=(18, 13), dpi=140)
fig.suptitle(
    f"Robust MPC  |  Solve {1/dt_mpc:.0f} Hz, N={N} ({N*dt_mpc:.1f}s horizon)  |  "
    f"Solve: {st.mean():.2f} ms avg  |  "
    f"Max pitch: {np.rad2deg(np.abs(X[4,:])).max():.1f}°  |  "
    f"Alt err std: {dz_hist.std():.1f} m",
    fontsize=10, fontweight='bold')

ax = axs[0, 0]
ax.plot(X[0,:], X[1,:], 'b', lw=2, label='Ownship')
ax.plot(x_tgt, z_tgt, 'r--', lw=2, label='Target')
ax.scatter([X[0,0]], [X[1,0]], c='b', zorder=5)
ax.scatter([x_tgt[0]], [z_tgt[0]], c='r', zorder=5)
ax.set_title("XZ Trajectory"); ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
ax.legend(); ax.grid(True); ax.set_aspect('equal', 'datalim')

ax = axs[0, 1]
slant = np.sqrt((x_tgt - X[0,:])**2 + (z_tgt - X[1,:])**2)
ax.plot(t, slant, 'purple', lw=2, label='Slant range')
ax.axhline(d_ref, color='k', ls='--', label=f'Standoff ({d_ref:.0f} m)')
ax.set_title("Slant Range"); ax.set_ylabel("m"); ax.legend(); ax.grid(True)

ax = axs[0, 2]
ax.plot(t, X[1,:], 'b', lw=2, label='Ownship z')
ax.plot(t, z_tgt, 'r--', lw=2, label='Target z')
ax.set_title("Altitude Tracking"); ax.set_ylabel("m"); ax.legend(); ax.grid(True)

ax = axs[1, 0]
ax.plot(t, range_hist, 'teal', lw=2, label='Forward range')
ax.axhline(d_ref, color='k', ls='--', label=f'Standoff ({d_ref:.0f} m)')
for ot in t[outer_tick_hist]:
    ax.axvline(ot, color='gray', alpha=0.15, lw=0.5)
ax.set_title(f"Forward Range (gray = guidance ticks @ {1/dt_outer:.0f} Hz)")
ax.set_ylabel("m"); ax.legend(); ax.grid(True)

ax = axs[1, 1]
ax.plot(t, dz_hist, 'orange', lw=2, label='Alt error')
ax.axhline(0, color='k', ls='--')
ax.set_title("Altitude Error (z_tgt − z_own)"); ax.set_ylabel("m")
ax.legend(); ax.grid(True)

ax = axs[1, 2]
ax.plot(t, X[2,:], 'b', lw=2, label='u actual')
ax.plot(t, u_ref_hist, 'b--', lw=1.5, label='u_ref')
ax2 = ax.twinx()
ax2.plot(t, np.rad2deg(X[4,:]), 'g', lw=2, label='θ actual')
ax2.plot(t, np.rad2deg(theta_ref_hist), 'g--', lw=1.5, label='θ_ref')
ax2.set_ylabel("deg", color='g')
ax.set_title("Speed (blue) & Pitch (green)")
ax.set_ylabel("m/s", color='b')
ax.legend(loc='upper left'); ax2.legend(loc='lower right'); ax.grid(True)

ax = axs[2, 0]
ax.plot(t, np.rad2deg(X[4,:]), 'b', lw=2, label='θ actual')
ax.plot(t, np.rad2deg(theta_ref_hist), 'b--', lw=1.5, label='θ_ref')
ax.set_title("Pitch Angle"); ax.set_xlabel("s"); ax.set_ylabel("deg")
ax.legend(); ax.grid(True)

ax = axs[2, 1]
ax.plot(t, elev_deg, 'r', lw=1.2, label='Elevator (total)')
ax.plot(t, np.rad2deg(U_ff_hist[0,:]), 'r--', lw=1, alpha=0.5, label='Elev FF')
ax.axhline( np.rad2deg(elev_max), color='r', ls=':', alpha=0.5, label=f'±{np.rad2deg(elev_max):.0f}° limit')
ax.axhline(-np.rad2deg(elev_max), color='r', ls=':', alpha=0.5)
ax.set_title(f"Elevator  (dom. {dom_freq:.3f} Hz)")
ax.set_xlabel("s"); ax.set_ylabel("deg"); ax.legend(fontsize=8); ax.grid(True)

ax = axs[2, 2]
ax.plot(t, U_ctrl[1,:]/1e3, 'darkorange', lw=1.5, label='Throttle (total)')
ax.plot(t, U_ff_hist[1,:]/1e3, 'darkorange', lw=1, ls='--', alpha=0.5, label='Throttle FF')
ax.set_title("Throttle (solid=total, dashed=FF)"); ax.set_xlabel("s"); ax.set_ylabel("kN")
ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
plt.savefig("./media/pursuit_sim_mpc.png", dpi=140, bbox_inches='tight')
plt.show()