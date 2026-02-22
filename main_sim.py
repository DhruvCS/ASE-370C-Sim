# main_sim.py
import numpy as np
import matplotlib.pyplot as plt

from aircraft_dynamics import AircraftParams, ActuatorLimits, Aircraft
from aircraft_controller import (
    OuterLoopConfig, ThrottlePIConfig, AttitudePDConfig, AttitudeLQRConfig,
    DecoupledPIDController, DecoupledLQRAttitudeController
)

def main():
    # =========================================================
    # Global sim timing
    # =========================================================
    dt_inner = 0.005   # 200 Hz
    dt_outer = 0.5     # 2 Hz
    Tf = 60.0
    n = int(Tf / dt_inner) + 1
    t = np.linspace(0, Tf, n)
    outer_every = int(round(dt_outer / dt_inner))

    # =========================================================
    # Shared aircraft params and limits
    # =========================================================
    p = AircraftParams()
    limits = ActuatorLimits(
        elev_max=np.deg2rad(25.0),
        throttle_max=40e3,
        elev_rate_max=np.deg2rad(200.0),
        throttle_rate_max=8e3,
        use_engine_lag=True,
        tau_throttle=0.5
    )

    # =========================================================
    # Aircraft objects (same dynamics)
    # =========================================================
    ownship = Aircraft(
        x0=[0.0, 2000.0, p.U0, 0.0, 0.0, 0.0],
        p=p,
        limits=limits
    )
    target = Aircraft(
        x0=[800.0, 2000.0, 160.0, 0.0, 0.0, 0.0],
        p=p,
        limits=limits
    )

    # =========================================================
    # Controllers
    # =========================================================
    # Ownship: pursuit outer loop + decoupled PID inner
    own_outer = OuterLoopConfig(
        dt_outer=dt_outer,
        d_ref=500.0,

        # Standoff speed loop (patched)
        Kv_speed=0.08,
        Kd_speed=1.2,
        Ki_speed=0.05,              # start at 0 first
        range_int_min=-100.0,
        range_int_max=100.0,
        closure_bias_gain=3.0,
        closure_bias_scale=200.0,

        u_cmd_min=140.0,
        u_cmd_max=220.0,

        # Altitude loop
        Kp_alt=0.0004,
        Ki_alt=0.0,
        Kd_alt=0.001,
        theta_ref_min=np.deg2rad(-3.0),
        theta_ref_max=np.deg2rad( 3.0),
        theta_rate_lim=np.deg2rad(0.5),
        alt_int_min=np.deg2rad(-2.0),
        alt_int_max=np.deg2rad( 2.0),
    )
    own_thr = ThrottlePIConfig(dt_inner=dt_inner, Kp_u=1500.0, Ki_u=100.0, u_int_min=-20.0, u_int_max=20.0)
    own_att = AttitudeLQRConfig(
        Q_att=np.diag([0.2, 20.0, 7.0]),
        R_elev=np.array([[4.0]])
    )
    own_ctrl = DecoupledLQRAttitudeController(p, own_outer, own_thr, own_att)

    # Target: same outer-loop structure but used as "direct command holder"; can still use LQR attitude
    tgt_outer = OuterLoopConfig(
        dt_outer=dt_outer,
        d_ref=500.0,        # unused for direct commands
        Kv_speed=0.0,       # unused for direct commands
        u_cmd_min=120.0,
        u_cmd_max=220.0,
        Kp_alt=0.0005,
        Ki_alt=0.0,
        Kd_alt=0.001,
        theta_ref_min=np.deg2rad(-3.0),
        theta_ref_max=np.deg2rad( 3.0),
        theta_rate_lim=np.deg2rad(0.5),
        alt_int_min=np.deg2rad(-2.0),
        alt_int_max=np.deg2rad( 2.0),
    )
    tgt_thr = ThrottlePIConfig(dt_inner=dt_inner, Kp_u=1200.0, Ki_u=80.0, u_int_min=-20.0, u_int_max=20.0)
    tgt_att_lqr = AttitudeLQRConfig(
        Q_att=np.diag([0.2, 20.0, 8.0]),
        R_elev=np.array([[4.0]])
    )
    target_ctrl = DecoupledLQRAttitudeController(p, tgt_outer, tgt_thr, tgt_att_lqr)
    # If you want target PD instead, swap to DecoupledPIDController(...)

    # =========================================================
    # Histories
    # =========================================================
    X_own = np.zeros((6, n))
    X_tgt = np.zeros((6, n))
    Ucmd_own = np.zeros((2, n))
    Uact_own = np.zeros((2, n))
    Ucmd_tgt = np.zeros((2, n))
    Uact_tgt = np.zeros((2, n))

    own_u_ref_hist = np.zeros(n)
    own_theta_ref_hist = np.zeros(n)
    tgt_u_ref_hist = np.zeros(n)
    tgt_theta_ref_hist = np.zeros(n)

    forward_range_hist = np.zeros(n)
    alt_err_hist = np.zeros(n)

    X_own[:, 0] = ownship.x
    X_tgt[:, 0] = target.x

    # =========================================================
    # Target command profiles (scenario logic lives here)
    # =========================================================
    def target_u_ref_cmd(tt):
        return 160.0 + 5.0 * np.sin(2*np.pi*0.004*tt)

    def target_z_ref_cmd(tt):
        return 2000.0 + 10.0 * np.sin(2*np.pi*0.01*tt)

    def target_zdot_ref_cmd(tt):
        return 10.0 * (2*np.pi*0.01) * np.cos(2*np.pi*0.01*tt)

    # =========================================================
    # Main simulation loop
    # =========================================================
    for k in range(n - 1):
        xo = ownship.x
        xt = target.x

        # ---------- Target outer update (direct imposed commands) ----------
        if k % outer_every == 0:
            # use target's own controller outer law for altitude->theta shaping,
            # but direct u_cmd is imposed from scenario
            z_ref_t = target_z_ref_cmd(t[k])
            z_dot_ref_t = target_zdot_ref_cmd(t[k])
            u_ref_t = target_u_ref_cmd(t[k])

            # manual outer update for target (altitude shaping)
            # do altitude loop using target states:
            target_ctrl.alt_integrator += target_ctrl.outer.Ki_alt * (z_ref_t - xt[1]) * dt_outer
            target_ctrl.alt_integrator = np.clip(target_ctrl.alt_integrator,
                                                 target_ctrl.outer.alt_int_min,
                                                 target_ctrl.outer.alt_int_max)

            theta_raw_t = (target_ctrl.outer.Kp_alt * (z_ref_t - xt[1]) +
                           target_ctrl.alt_integrator +
                           target_ctrl.outer.Kd_alt * (z_dot_ref_t - xt[3]))
            theta_raw_t = np.clip(theta_raw_t,
                                  target_ctrl.outer.theta_ref_min,
                                  target_ctrl.outer.theta_ref_max)

            dtheta_t = np.clip(theta_raw_t - target_ctrl.theta_cmd_prev,
                               -target_ctrl.outer.theta_rate_lim,
                               target_ctrl.outer.theta_rate_lim)
            theta_cmd_t = float(target_ctrl.theta_cmd_prev + dtheta_t)
            target_ctrl.theta_cmd = theta_cmd_t
            target_ctrl.theta_cmd_prev = theta_cmd_t
            target_ctrl.u_cmd = float(np.clip(u_ref_t, target_ctrl.outer.u_cmd_min, target_ctrl.outer.u_cmd_max))

        # ---------- Target inner control ----------
        elev_unsat_t, thr_unsat_t = target_ctrl.control_unsat(xt, target.throttle_trim)
        ctrl_cmd_t, ctrl_act_t = target.apply_actuators(elev_unsat_t, thr_unsat_t, dt_inner)
        target_ctrl.throttle_antiwindup(ctrl_cmd_t[1], thr_unsat_t)
        target.step(ctrl_act_t, dt_inner)

        # ---------- Ownship outer loop (pursuit using target state) ----------
        if k % outer_every == 0:
            own_ctrl.update_outer_from_target(xo, xt)

        # ---------- Ownship inner control ----------
        elev_unsat_o, thr_unsat_o = own_ctrl.control_unsat(xo, ownship.throttle_trim)
        ctrl_cmd_o, ctrl_act_o = ownship.apply_actuators(elev_unsat_o, thr_unsat_o, dt_inner)
        own_ctrl.throttle_antiwindup(ctrl_cmd_o[1], thr_unsat_o)
        ownship.step(ctrl_act_o, dt_inner)

        # ---------- Log ----------
        X_own[:, k+1] = ownship.x
        X_tgt[:, k+1] = target.x
        Ucmd_own[:, k] = ctrl_cmd_o
        Uact_own[:, k] = ctrl_act_o
        Ucmd_tgt[:, k] = ctrl_cmd_t
        Uact_tgt[:, k] = ctrl_act_t

        own_u_ref_hist[k] = own_ctrl.u_cmd
        own_theta_ref_hist[k] = own_ctrl.theta_cmd
        tgt_u_ref_hist[k] = target_ctrl.u_cmd
        tgt_theta_ref_hist[k] = target_ctrl.theta_cmd

        forward_range_hist[k] = xt[0] - xo[0]
        alt_err_hist[k] = xt[1] - xo[1]

        # Safety envelope for this linear toy model
        if abs(ownship.x[4]) > np.deg2rad(25) or abs(target.x[4]) > np.deg2rad(25):
            print(f"Pitch envelope exceeded at t={t[k+1]:.2f}s. Stopping early.")
            X_own = X_own[:, :k+2]
            X_tgt = X_tgt[:, :k+2]
            Ucmd_own = Ucmd_own[:, :k+2]
            Uact_own = Uact_own[:, :k+2]
            Ucmd_tgt = Ucmd_tgt[:, :k+2]
            Uact_tgt = Uact_tgt[:, :k+2]
            own_u_ref_hist = own_u_ref_hist[:k+2]
            own_theta_ref_hist = own_theta_ref_hist[:k+2]
            tgt_u_ref_hist = tgt_u_ref_hist[:k+2]
            tgt_theta_ref_hist = tgt_theta_ref_hist[:k+2]
            forward_range_hist = forward_range_hist[:k+2]
            alt_err_hist = alt_err_hist[:k+2]
            t = t[:k+2]
            n_local = len(t)
            break
    else:
        n_local = n

    # Fill final logs
    if n_local > 1:
        Ucmd_own[:, n_local-1] = Ucmd_own[:, n_local-2]
        Uact_own[:, n_local-1] = Uact_own[:, n_local-2]
        Ucmd_tgt[:, n_local-1] = Ucmd_tgt[:, n_local-2]
        Uact_tgt[:, n_local-1] = Uact_tgt[:, n_local-2]
        own_u_ref_hist[n_local-1] = own_u_ref_hist[n_local-2]
        own_theta_ref_hist[n_local-1] = own_theta_ref_hist[n_local-2]
        tgt_u_ref_hist[n_local-1] = tgt_u_ref_hist[n_local-2]
        tgt_theta_ref_hist[n_local-1] = tgt_theta_ref_hist[n_local-2]
        forward_range_hist[n_local-1] = forward_range_hist[n_local-2]
        alt_err_hist[n_local-1] = alt_err_hist[n_local-2]

    # =========================================================
    # Diagnostics
    # =========================================================
    print(f"Inner: {1/dt_inner:.0f} Hz | Outer: {1/dt_outer:.0f} Hz | Ratio: {outer_every}x")
    print(f"Ownship max |pitch|: {np.rad2deg(np.abs(X_own[4,:])).max():.2f} deg")
    print(f"Target  max |pitch|: {np.rad2deg(np.abs(X_tgt[4,:])).max():.2f} deg")
    print(f"Final forward range: {X_tgt[0,-1] - X_own[0,-1]:.1f} m (target {own_outer.d_ref} m)")
    print(f"Own throttle cmd range: {Ucmd_own[1,:].min()/1e3:.1f} to {Ucmd_own[1,:].max()/1e3:.1f} kN")
    print(f"Tgt throttle cmd range: {Ucmd_tgt[1,:].min()/1e3:.1f} to {Ucmd_tgt[1,:].max()/1e3:.1f} kN")

    # =========================================================
    # Plots
    # =========================================================
    fig, axs = plt.subplots(3, 3, figsize=(18, 13), dpi=140)
    fig.suptitle("Pursuit Sim (Modular): Shared Aircraft Dynamics + Reusable Controllers", fontsize=12)

    # XZ trajectories
    ax = axs[0,0]
    ax.plot(X_own[0,:], X_own[1,:], label='Ownship', lw=2)
    ax.plot(X_tgt[0,:], X_tgt[1,:], '--', label='Target', lw=2)
    ax.scatter([X_own[0,0]], [X_own[1,0]], c='C0')
    ax.scatter([X_tgt[0,0]], [X_tgt[1,0]], c='C1')
    ax.set_title("XZ Trajectories")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.legend(); ax.grid(True); ax.set_aspect('equal', 'datalim')

    # forward range
    ax = axs[0,1]
    ax.plot(t[:len(forward_range_hist)], forward_range_hist, lw=2, label='Forward range')
    ax.axhline(own_outer.d_ref, color='k', ls='--', label='Standoff ref')
    ax.set_title("Forward Range")
    ax.set_ylabel("m")
    ax.legend(); ax.grid(True)

    # altitude error
    ax = axs[0,2]
    ax.plot(t[:len(alt_err_hist)], alt_err_hist, lw=2, label='z_tgt - z_own')
    ax.axhline(0, color='k', ls='--')
    ax.set_title("Altitude Error")
    ax.set_ylabel("m")
    ax.legend(); ax.grid(True)

    # surge speeds
    ax = axs[1,0]
    ax.plot(t[:X_own.shape[1]], X_own[2,:], label='u own', lw=2)
    ax.plot(t[:len(own_u_ref_hist)], own_u_ref_hist, '--', label='u_ref own', lw=1.5)
    ax.plot(t[:X_tgt.shape[1]], X_tgt[2,:], label='u tgt', lw=2)
    ax.plot(t[:len(tgt_u_ref_hist)], tgt_u_ref_hist, '--', label='u_ref tgt', lw=1.0)
    ax.set_title("Surge Speeds")
    ax.set_ylabel("m/s")
    ax.legend(); ax.grid(True)

    # pitch
    ax = axs[1,1]
    ax.plot(t[:X_own.shape[1]], np.rad2deg(X_own[4,:]), label='theta own', lw=2)
    ax.plot(t[:len(own_theta_ref_hist)], np.rad2deg(own_theta_ref_hist), '--', label='theta_ref own', lw=1.5)
    ax.plot(t[:X_tgt.shape[1]], np.rad2deg(X_tgt[4,:]), label='theta tgt', lw=2)
    ax.plot(t[:len(tgt_theta_ref_hist)], np.rad2deg(tgt_theta_ref_hist), '--', label='theta_ref tgt', lw=1.0)
    ax.set_title("Pitch Angles")
    ax.set_ylabel("deg")
    ax.legend(); ax.grid(True)

    # heave velocities
    ax = axs[1,2]
    ax.plot(t[:X_own.shape[1]], X_own[3,:], label='w own', lw=2)
    ax.plot(t[:X_tgt.shape[1]], X_tgt[3,:], label='w tgt', lw=2)
    ax.set_title("Heave Velocities")
    ax.set_ylabel("m/s")
    ax.legend(); ax.grid(True)

    # own elevator
    ax = axs[2,0]
    ax.plot(t[:Ucmd_own.shape[1]], np.rad2deg(Ucmd_own[0,:]), label='Own elev cmd')
    ax.plot(t[:Uact_own.shape[1]], np.rad2deg(Uact_own[0,:]), '--', label='Own elev act')
    ax.set_title("Own Elevator")
    ax.set_xlabel("s"); ax.set_ylabel("deg")
    ax.legend(); ax.grid(True)

    # own throttle
    ax = axs[2,1]
    ax.plot(t[:Ucmd_own.shape[1]], Ucmd_own[1,:]/1e3, label='Own throttle cmd')
    ax.plot(t[:Uact_own.shape[1]], Uact_own[1,:]/1e3, '--', label='Own throttle act')
    ax.axhline(ownship.throttle_trim/1e3, color='k', ls=':', label='Trim')
    ax.set_title("Own Throttle")
    ax.set_xlabel("s"); ax.set_ylabel("kN")
    ax.legend(); ax.grid(True)

    # target throttle
    ax = axs[2,2]
    ax.plot(t[:Ucmd_tgt.shape[1]], Ucmd_tgt[1,:]/1e3, label='Target throttle cmd')
    ax.plot(t[:Uact_tgt.shape[1]], Uact_tgt[1,:]/1e3, '--', label='Target throttle act')
    ax.axhline(target.throttle_trim/1e3, color='k', ls=':', label='Trim')
    ax.set_title("Target Throttle")
    ax.set_xlabel("s"); ax.set_ylabel("kN")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig("./media/pursuit_sim.png", dpi=140, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()