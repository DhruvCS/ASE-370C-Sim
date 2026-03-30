function trailing_uav_sim()
% TRAILING_UAV_6DOF
% Full 6-DoF rigid-body fixed-wing UAV trailing simulation.
%
% Both ownship and target are modelled as 6-DoF rigid bodies with:
%   - 13-state dynamics: [x y z u v w q0 q1 q2 q3 p q r]
%     (NED position, body velocities, quaternion attitude, body rates)
%   - Conventional aerodynamics (stability-derivative model)
%   - Propulsion (thrust vs throttle & airspeed)
%   - Servo actuator dynamics (aileron, elevator, rudder, throttle)
%   - Standard atmosphere density model
%   - Gravity resolved in body frame via quaternion
%
% Autopilot architecture (per vehicle):
%   Inner loop  (200 Hz): surface commands -> servo dynamics -> 6DoF plant
%   Mid loop    (200 Hz): attitude tracking (SAS) -> surface commands
%   Outer loop  (  2 Hz): guidance -> attitude & throttle commands
%
% Coordinate frame: NED (x North, y East, z Down).
%   Altitude h = -z.
%
% The target flies a pre-programmed gentle manoeuvring profile.
% The ownship uses trail-behind-at-distance guidance.
%
% Designed for straightforward Simulink conversion: each sub-function maps
% to a Simulink block / subsystem.

clc; close all;

%% =====================================================================
%  Simulation timing
%  =====================================================================
dt       = 0.005;            % 200 Hz
Tend     = 300;
N        = round(Tend/dt) + 1;
t        = (0:N-1)' * dt;
g        = 9.80665;

outer_hz = 2;                % guidance rate
outer_dt = 1/outer_hz;
outer_k  = round(outer_dt/dt);

%% =====================================================================
%  Aircraft parameters (identical for target & ownship — small UAV)
%  =====================================================================
ac = aircraft_params(g);

%% =====================================================================
%  Target initial state & guidance profile
%  =====================================================================
tgt = init_state();
tgt.pos   = [0; 0; -120];          % NED  (h = 120 m)
tgt.Vb    = [25; 0; 0];            % body-frame velocity (u,v,w)
tgt.quat  = euler2quat(0, 0, 0);   % wings-level heading North
tgt.omega = [0; 0; 0];             % p q r
tgt.surf  = [0; 0; 0; 0.5];        % da de dr dt (trim throttle)

% Target autopilot holds a prescribed manoeuvring profile
tgt_profile.V0       = 25;
tgt_profile.chi0     = 0;
tgt_profile.h0       = 120;
tgt_profile.turn_amp = deg2rad(8);
tgt_profile.turn_w   = 0*pi/60;     % one full oscillation per 60 s
tgt_profile.climb_amp= 0.5;
tgt_profile.climb_w  = 0*pi/60;

%% =====================================================================
%  Ownship initial state
%  =====================================================================
own = init_state();
own.pos   = [200; -400; -100];      % NED  (h=100 m, offset east & south)
own.Vb    = [22; 0; 0];
own.quat  = euler2quat(0, 0, deg2rad(20));
own.omega = [0; 0; 0];
own.surf  = [0; 0; 0; 0.5];

%% =====================================================================
%  Trailing objective
%  =====================================================================
d_trail  = 400;     % desired trail distance (m)
dh_trail = 0;       % altitude offset above target (m, positive = above)

%% =====================================================================
%  Guidance gains (ownship outer loop)
%  =====================================================================
guid.ky   = 0.8;
guid.L1   = 500;
guid.kx   = 0.03;
guid.kdx  = 0.40;

%% =====================================================================
%  Allocate logging arrays
%  =====================================================================
log.t     = t;
log.pos_t = zeros(N,3);  log.pos_o = zeros(N,3);
log.V_t   = zeros(N,1);  log.V_o   = zeros(N,1);
log.euler_t = zeros(N,3); log.euler_o = zeros(N,3);
log.alpha_o = zeros(N,1); log.beta_o  = zeros(N,1);
log.surf_o  = zeros(N,4); % da de dr dt
log.range   = zeros(N,1);
log.epar    = zeros(N,1); log.eperp = zeros(N,1);
log.V_cmd   = zeros(N,1); log.chi_cmd = zeros(N,1); log.h_cmd = zeros(N,1);

%% =====================================================================
%  Autopilot integrator states  (one set per vehicle)
%  =====================================================================
ap_tgt = init_autopilot_state();
ap_own = init_autopilot_state();

% Guidance rate-limited previous commands
chi_cmd = heading_from_quat(own.quat);
V_cmd   = norm(own.Vb);
h_cmd   = -own.pos(3);
chi_cmd_prev = chi_cmd;
V_cmd_prev   = V_cmd;
h_cmd_prev   = h_cmd;

lim_chi_rate = deg2rad(20);
lim_V_rate   = 2.0;
lim_h_rate   = 3.0;

%% =====================================================================
%  Main loop
%  =====================================================================
for k = 1:N
    tk = t(k);

    % ---- Target commanded profile -----------------------------------
    chi_t_cmd = tgt_profile.chi0 + tgt_profile.turn_amp * sin(tgt_profile.turn_w * tk);
    V_t_cmd   = tgt_profile.V0;
    h_t_cmd   = tgt_profile.h0 + tgt_profile.climb_amp * sin(tgt_profile.climb_w * tk);

    % ---- Target autopilot ------------------------------------------
    [tgt.surf, ap_tgt] = autopilot(tgt, ac, chi_t_cmd, V_t_cmd, h_t_cmd, ap_tgt, dt);

    % ---- Target 6-DoF dynamics -------------------------------------
    tgt = step_6dof(tgt, ac, g, dt);

    % ---- Ownship guidance (outer loop, low rate) --------------------
    if mod(k-1, outer_k) == 0
        % Target inertial velocity for trail direction
        R_t = quat2dcm(tgt.quat);
        v_t_ned = R_t' * tgt.Vb;
        chi_t = atan2(v_t_ned(2), v_t_ned(1));
        V_t   = norm(tgt.Vb);

        t_hat = [cos(chi_t); sin(chi_t)];

        % Trail reference point (NED horizontal)
        p_ref_h = tgt.pos(1:2) - d_trail * t_hat;
        h_ref   = (-tgt.pos(3)) + dh_trail;   % altitude (positive up)

        % Ownship inertial velocity
        R_o = quat2dcm(own.quat);
        v_o_ned = R_o' * own.Vb;

        % Errors in trail frame
        e_h = p_ref_h - own.pos(1:2);
        e_par  = t_hat' * e_h;
        n_hat  = [-t_hat(2); t_hat(1)];
        e_perp = n_hat' * e_h;

        % Closure rate
        de_par = t_hat' * (v_t_ned(1:2) - v_o_ned(1:2));

        % Course command
        chi_base    = atan2(t_hat(2), t_hat(1));
        chi_cmd_new = chi_base + guid.ky * atan2(e_perp, guid.L1);

        % Speed command
        V_cmd_new = V_t + guid.kx * e_par + guid.kdx * de_par;

        % Altitude command
        h_cmd_new = h_ref;

        % Rate-limit
        chi_cmd = rate_limit_angle(chi_cmd_prev, chi_cmd_new, lim_chi_rate, outer_dt);
        V_cmd   = rate_limit_scalar(V_cmd_prev, V_cmd_new, lim_V_rate, outer_dt);
        h_cmd   = rate_limit_scalar(h_cmd_prev, h_cmd_new, lim_h_rate, outer_dt);

        V_cmd = clamp(V_cmd, ac.V_min, ac.V_max);

        chi_cmd_prev = chi_cmd;
        V_cmd_prev   = V_cmd;
        h_cmd_prev   = h_cmd;
    end

    % ---- Ownship autopilot -----------------------------------------
    [own.surf, ap_own] = autopilot(own, ac, chi_cmd, V_cmd, h_cmd, ap_own, dt);

    % ---- Ownship 6-DoF dynamics ------------------------------------
    own = step_6dof(own, ac, g, dt);

    % ---- Logging ----------------------------------------------------
    log.pos_t(k,:)   = tgt.pos';
    log.pos_o(k,:)   = own.pos';
    log.V_t(k)       = norm(tgt.Vb);
    log.V_o(k)       = norm(own.Vb);
    log.euler_t(k,:) = quat2euler(tgt.quat)';
    log.euler_o(k,:) = quat2euler(own.quat)';
    [a,b]            = aero_angles(own.Vb);
    log.alpha_o(k)   = a;
    log.beta_o(k)    = b;
    log.surf_o(k,:)  = own.surf';
    log.V_cmd(k)     = V_cmd;
    log.chi_cmd(k)   = chi_cmd;
    log.h_cmd(k)     = h_cmd;

    dp = tgt.pos - own.pos;
    log.range(k) = norm(dp);

    R_t2 = quat2dcm(tgt.quat);
    v_t2 = R_t2' * tgt.Vb;
    chi_t2 = atan2(v_t2(2), v_t2(1));
    th2 = [cos(chi_t2); sin(chi_t2)];
    pr2 = tgt.pos(1:2) - d_trail*th2;
    eh2 = pr2 - own.pos(1:2);
    log.epar(k)  = th2' * eh2;
    log.eperp(k) = [-th2(2);th2(1)]' * eh2;
end

%% =====================================================================
%  Plots
%  =====================================================================
plot_results(log);

end % trailing_uav_6dof


%% =====================================================================
%  AIRCRAFT PARAMETER MODEL
%  =====================================================================
function ac = aircraft_params(g)
% Small fixed-wing UAV (~25 kg, 3 m span). Stability-derivative model.

    % Mass / inertia
    ac.mass = 25;                       % kg
    ac.Ixx  = 1.80;  ac.Iyy = 3.20;  ac.Izz = 4.50;
    ac.Ixz  = 0.30;
    ac.I    = [ac.Ixx   0     -ac.Ixz;
               0        ac.Iyy  0;
              -ac.Ixz   0      ac.Izz];
    ac.Iinv = inv(ac.I);

    % Reference geometry
    ac.S     = 1.5;                     % wing area (m^2)
    ac.b     = 3.0;                     % span (m)
    ac.cbar  = 0.50;                    % mean aerodynamic chord (m)

    % ---- Longitudinal aero coefficients ----
    ac.CL0   =  0.28;
    ac.CLa   =  4.80;                   % per rad
    ac.CLq   =  7.50;                   % per (rad·c/(2V))
    ac.CLde  =  0.36;                   % per rad
    ac.CLad  =  1.80;                   % CLalphadot per (rad·c/(2V))

    ac.CD0   =  0.03;
    ac.CDa   =  0.30;                   % parabolic drag coeff (CD = CD0 + CDa*alpha^2)
    ac.CDde  =  0.02;                   % per rad (small)

    ac.Cm0   = -0.02;
    ac.Cma   = -0.56;                   % pitch stiffness (stable < 0)
    ac.Cmq   = -12.0;                   % pitch damping
    ac.Cmde  = -1.10;                   % elevator effectiveness
    ac.Cmad  = -5.20;                   % alpha-dot damping

    % ---- Lateral-directional aero coefficients ----
    ac.CYb   = -0.31;
    ac.CYp   =  0.0;
    ac.CYr   =  0.21;
    ac.CYda  =  0.0;
    ac.CYdr  =  0.19;

    ac.Clb   = -0.089;                  % roll-due-to-sideslip (dihedral effect)
    ac.Clp   = -0.47;                   % roll damping
    ac.Clr   =  0.096;                  % roll-due-to-yaw
    ac.Clda  = -0.178;                  % aileron effectiveness
    ac.Cldr  =  0.0144;

    ac.Cnb   =  0.065;                  % yaw stiffness (weathercock)
    ac.Cnp   = -0.03;
    ac.Cnr   = -0.099;                  % yaw damping
    ac.Cnda  = -0.053;                  % adverse yaw
    ac.Cndr  = -0.069;                  % rudder effectiveness

    % ---- Propulsion ----
    ac.Tmax  = 18;                      % max thrust (N) at sea level
    ac.tau_t = 0.30;                    % throttle first-order lag (s)

    % ---- Servo actuator limits ----
    ac.da_max = deg2rad(25);
    ac.de_max = deg2rad(20);
    ac.dr_max = deg2rad(25);
    ac.tau_s  = 0.05;                   % servo time constant (s)

    % ---- Trim alpha (approximate) ----
    % At trim: CL0 + CLa*alpha_trim = mg/(qbar*S). Computed at V=25.
    rho0 = 1.225;
    qbar0 = 0.5*rho0*25^2;
    CL_trim = ac.mass*g / (qbar0*ac.S);
    ac.alpha_trim = (CL_trim - ac.CL0)/ac.CLa;

    % ---- Speed envelope ----
    ac.V_min = 16;
    ac.V_max = 35;
end


%% =====================================================================
%  STATE INITIALISATION
%  =====================================================================
function s = init_state()
    s.pos   = [0;0;0];          % NED position (m)
    s.Vb    = [25;0;0];         % body velocity [u;v;w] (m/s)
    s.quat  = [1;0;0;0];       % quaternion [q0;q1;q2;q3]
    s.omega = [0;0;0];         % body rates [p;q;r] (rad/s)
    s.surf  = [0;0;0;0.5];     % actuator states [da;de;dr;dt]
    s.alpha_prev = 0;           % for alpha-dot estimation
end


%% =====================================================================
%  AUTOPILOT INTEGRATOR STATE
%  =====================================================================
function ap = init_autopilot_state()
    ap.int_h   = 0;
    ap.int_V   = 0;
    ap.int_phi = 0;
    ap.int_theta = 0;
    ap.int_beta  = 0;
end


%% =====================================================================
%  6-DOF RIGID BODY STEP (RK4)
%  =====================================================================
function s = step_6dof(s, ac, g, dt)
% Propagate the 13-state + actuator dynamics by one timestep using RK4.

    y = state2vec(s);

    % RK4 integration of rigid-body + actuator dynamics
    k1 = dynamics(y, s, ac, g);
    k2 = dynamics(y + 0.5*dt*k1, s, ac, g);
    k3 = dynamics(y + 0.5*dt*k2, s, ac, g);
    k4 = dynamics(y + dt*k3, s, ac, g);
    y  = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

    % Write back and normalise quaternion
    s = vec2state(y, s);
    s.quat = s.quat / norm(s.quat);

    % Enforce actuator limits
    s.surf(1) = clamp(s.surf(1), -ac.da_max, ac.da_max);
    s.surf(2) = clamp(s.surf(2), -ac.de_max, ac.de_max);
    s.surf(3) = clamp(s.surf(3), -ac.dr_max, ac.dr_max);
    s.surf(4) = clamp(s.surf(4), 0, 1);

    % Store alpha for alpha-dot estimation next step
    [alpha, ~] = aero_angles(s.Vb);
    s.alpha_prev = alpha;
end


%% =====================================================================
%  STATE VECTOR CONVERSION
%  =====================================================================
function y = state2vec(s)
    y = [s.pos; s.Vb; s.quat; s.omega; s.surf];   % 17x1
end

function s = vec2state(y, s)
    s.pos   = y(1:3);
    s.Vb    = y(4:6);
    s.quat  = y(7:10);
    s.omega = y(11:13);
    s.surf  = y(14:17);
end


%% =====================================================================
%  DYNAMICS  (returns dy/dt for the 17-state vector)
%  =====================================================================
function ydot = dynamics(y, s_ref, ac, g)
% y = [pos(3); Vb(3); quat(4); omega(3); surf(4)]  = 17 states
%
% s_ref carries the commanded surface deflections in s_ref.surf (target)
% and alpha_prev for alpha-dot estimation.

    pos   = y(1:3);
    Vb    = y(4:6);
    quat  = y(7:10); quat = quat/norm(quat);
    omega = y(11:13);
    surf  = y(14:17);   % actual actuator states

    u = Vb(1); v = Vb(2); w = Vb(3);
    p = omega(1); q_rate = omega(2); r = omega(3);

    da = surf(1); de = surf(2); dr = surf(3); dt_act = surf(4);

    % ---- Airspeed, alpha, beta ----
    Va = norm(Vb);
    if Va < 1e-3, Va = 1e-3; end
    alpha = atan2(w, u);
    beta  = asin(clamp(v/Va, -1, 1));

    % Alpha-dot estimate (finite difference, using previous alpha)
    alpha_dot = (alpha - s_ref.alpha_prev) / 0.005;  % use dt
    alpha_dot = clamp(alpha_dot, -2, 2);              % limit

    % ---- Atmosphere ----
    h   = -pos(3);                                    % altitude (m)
    rho = atmosphere(h);
    qbar = 0.5 * rho * Va^2;

    % ---- Non-dimensional rates ----
    phat = p * ac.b / (2*Va);
    qhat = q_rate * ac.cbar / (2*Va);
    rhat = r * ac.b / (2*Va);
    alpha_dot_hat = alpha_dot * ac.cbar / (2*Va);

    % ---- Aerodynamic coefficients ----
    CL = ac.CL0 + ac.CLa*alpha + ac.CLq*qhat + ac.CLde*de + ac.CLad*alpha_dot_hat;
    CD = ac.CD0 + ac.CDa*alpha^2 + ac.CDde*abs(de);
    CY = ac.CYb*beta + ac.CYp*phat + ac.CYr*rhat + ac.CYda*da + ac.CYdr*dr;

    Cl = ac.Clb*beta + ac.Clp*phat + ac.Clr*rhat + ac.Clda*da + ac.Cldr*dr;
    Cm = ac.Cm0 + ac.Cma*alpha + ac.Cmq*qhat + ac.Cmde*de + ac.Cmad*alpha_dot_hat;
    Cn = ac.Cnb*beta + ac.Cnp*phat + ac.Cnr*rhat + ac.Cnda*da + ac.Cndr*dr;

    % ---- Aero forces in stability frame -> body frame ----
    ca = cos(alpha); sa = sin(alpha);
    Xa = qbar*ac.S*(-CD*ca + CL*sa);   % body x (forward)
    Ya = qbar*ac.S* CY;                 % body y (right)
    Za = qbar*ac.S*(-CD*sa - CL*ca);   % body z (down)

    La = qbar*ac.S*ac.b * Cl;           % roll moment
    Ma = qbar*ac.S*ac.cbar * Cm;        % pitch moment
    Na = qbar*ac.S*ac.b * Cn;           % yaw moment

    % ---- Propulsion (body x-axis thrust) ----
    T = dt_act * ac.Tmax * (rho/1.225);  % thrust scaled by density ratio

    % ---- Gravity in body frame ----
    R = quat2dcm(quat);
    grav_ned = [0; 0; g];               % NED
    grav_b   = R * grav_ned;            % body frame

    % ---- Translational dynamics (body frame) ----
    F = [Xa + T; Ya; Za] + ac.mass * grav_b;
    udot = F(1)/ac.mass + r*v - q_rate*w;
    vdot = F(2)/ac.mass - r*u + p*w;
    wdot = F(3)/ac.mass + q_rate*u - p*v;

    % ---- Rotational dynamics ----
    M_aero = [La; Ma; Na];
    H = ac.I * omega;
    Mdot = M_aero - cross(omega, H);
    omega_dot = ac.Iinv * Mdot;

    % ---- Quaternion kinematics ----
    Omega = [ 0,  -p,     -q_rate, -r;
              p,   0,      r,      -q_rate;
              q_rate, -r,  0,       p;
              r,   q_rate, -p,      0];
    quat_dot = 0.5 * Omega * quat;

    % ---- Position kinematics (NED) ----
    pos_dot = R' * Vb;

    % ---- Actuator dynamics (first-order lag toward commanded) ----
    surf_cmd = s_ref.surf;     % commanded deflections
    surf_dot = (surf_cmd - surf) / ac.tau_s;
    % Throttle has its own time constant
    surf_dot(4) = (surf_cmd(4) - surf(4)) / ac.tau_t;

    % ---- Assemble ydot ----
    ydot = [pos_dot;             % 1-3
            udot; vdot; wdot;    % 4-6
            quat_dot;            % 7-10
            omega_dot;           % 11-13
            surf_dot];           % 14-17
end


%% =====================================================================
%  AUTOPILOT  (SAS + mid-level attitude loops)
%  =====================================================================
function [surf_cmd, ap] = autopilot(s, ac, chi_cmd, V_cmd, h_cmd, ap, dt)
% Cascaded autopilot:
%   Outer:  chi_cmd, V_cmd, h_cmd  (from guidance)
%   Mid:    phi_cmd, theta_cmd, beta_cmd = 0
%   Inner:  da_cmd, de_cmd, dr_cmd, dt_cmd

    g = 9.80665;
    [phi, theta, psi] = deal(0,0,0);
    eul = quat2euler(s.quat);
    phi = eul(1); theta = eul(2); psi = eul(3);

    u = s.Vb(1); v = s.Vb(2); w = s.Vb(3);
    Va = norm(s.Vb);
    if Va < 1, Va = 1; end
    [alpha, beta] = aero_angles(s.Vb);

    p = s.omega(1); q = s.omega(2); r = s.omega(3);

    h = -s.pos(3);          % altitude (positive up)

    % Heading (ground-track approximation from body heading + wind = 0)
    chi = psi;               % no wind assumption

    % ---- Heading hold -> bank angle command ----
    e_chi = wrapToPi(chi_cmd - chi);
    phi_max = deg2rad(35);

    % Desired turn rate
    chi_dot_des = clamp(2.0 * e_chi, -deg2rad(15), deg2rad(15));
    % Bank angle for coordinated turn: phi = atan(V*chi_dot/g)
    phi_cmd = atan(Va * chi_dot_des / g);
    phi_cmd = clamp(phi_cmd, -phi_max, phi_max);

    % ---- Altitude hold -> pitch angle command ----
    e_h = h_cmd - h;
    ap.int_h = clamp(ap.int_h + e_h*dt, -50, 50);

    hdot = -s.Vb(1)*sin(theta) + s.Vb(3)*cos(theta);  % approximate
    % hdot from NED velocity:
    R = quat2dcm(s.quat);
    v_ned = R' * s.Vb;
    hdot = -v_ned(3);

    theta_max = deg2rad(15);
    theta_cmd = 0.04*e_h + 0.0016*ap.int_h - 0.16*hdot;
    theta_cmd = clamp(theta_cmd, -theta_max, theta_max);

    % ---- Speed hold -> throttle command ----
    e_V = V_cmd - Va;
    ap.int_V = clamp(ap.int_V + e_V*dt, -10, 10);

    % Trim throttle estimate
    rho = atmosphere(h);
    qbar = 0.5*rho*Va^2;
    CD_est = ac.CD0 + ac.CDa*alpha^2;
    D_est  = qbar * ac.S * CD_est;
    T_trim = D_est + ac.mass*g*sin(theta);
    dt_trim = clamp(T_trim / (ac.Tmax * rho/1.225), 0, 1);

    dt_cmd = dt_trim + 0.8*e_V/(ac.V_max - ac.V_min) + 0.2*ap.int_V/(ac.V_max - ac.V_min);
    dt_cmd = clamp(dt_cmd, 0, 1);

    % ---- Roll attitude tracker -> aileron ----
    e_phi = phi_cmd - phi;
    ap.int_phi = clamp(ap.int_phi + e_phi*dt, -5, 5);
    da_cmd = -2.0*e_phi - 0.15*ap.int_phi + 0.25*p;   % PD + rate damping
    da_cmd = clamp(da_cmd, -ac.da_max, ac.da_max);

    % ---- Pitch attitude tracker -> elevator ----
    e_theta = theta_cmd - theta;
    ap.int_theta = clamp(ap.int_theta + e_theta*dt, -5, 5);

    % Trim elevator estimate
    de_trim = -(ac.Cm0 + ac.Cma*ac.alpha_trim) / ac.Cmde;

    de_cmd = de_trim - 1.5*e_theta - 0.12*ap.int_theta + 0.6*q;
    de_cmd = clamp(de_cmd, -ac.de_max, ac.de_max);

    % ---- Sideslip regulator -> rudder (coordinated turn) ----
    ap.int_beta = clamp(ap.int_beta + beta*dt, -3, 3);
    dr_cmd = -1.5*beta - 0.1*ap.int_beta + 0.3*r;
    dr_cmd = clamp(dr_cmd, -ac.dr_max, ac.dr_max);

    % ---- Speed protection: reduce demands if near stall ----
    if Va < ac.V_min + 1
        phi_cmd = 0.5 * phi_cmd;
        theta_cmd = min(theta_cmd, deg2rad(5));
        dt_cmd = max(dt_cmd, 0.9);   % push throttle
    end

    surf_cmd = [da_cmd; de_cmd; dr_cmd; dt_cmd];
end


%% =====================================================================
%  ATMOSPHERE MODEL
%  =====================================================================
function rho = atmosphere(h)
% ISA troposphere density (h in metres, h >= 0)
    h = max(h, 0);
    T = 288.15 - 0.0065*h;
    rho = 1.225 * (T/288.15)^4.2561;
end


%% =====================================================================
%  QUATERNION / EULER UTILITIES
%  =====================================================================
function q = euler2quat(phi, theta, psi)
% Euler (3-2-1 / ZYX) -> quaternion [q0; q1; q2; q3]
    cp = cos(phi/2);   sp = sin(phi/2);
    ct = cos(theta/2); st = sin(theta/2);
    cs = cos(psi/2);   ss = sin(psi/2);

    q = [cp*ct*cs + sp*st*ss;
         sp*ct*cs - cp*st*ss;
         cp*st*cs + sp*ct*ss;
         cp*ct*ss - sp*st*cs];
    q = q / norm(q);
end

function eul = quat2euler(q)
% Quaternion -> Euler [phi; theta; psi]  (3-2-1)
    q0=q(1); q1=q(2); q2=q(3); q3=q(4);
    phi   = atan2(2*(q0*q1+q2*q3), 1-2*(q1^2+q2^2));
    theta = asin(clamp(2*(q0*q2-q3*q1), -1, 1));
    psi   = atan2(2*(q0*q3+q1*q2), 1-2*(q2^2+q3^2));
    eul = [phi; theta; psi];
end

function R = quat2dcm(q)
% Quaternion -> DCM  (R rotates NED -> body:  v_body = R * v_ned)
    q0=q(1); q1=q(2); q2=q(3); q3=q(4);
    R = [q0^2+q1^2-q2^2-q3^2,  2*(q1*q2+q0*q3),      2*(q1*q3-q0*q2);
         2*(q1*q2-q0*q3),      q0^2-q1^2+q2^2-q3^2,   2*(q2*q3+q0*q1);
         2*(q1*q3+q0*q2),      2*(q2*q3-q0*q1),        q0^2-q1^2-q2^2+q3^2];
end

function chi = heading_from_quat(q)
% Extract heading (psi) from quaternion
    eul = quat2euler(q);
    chi = eul(3);
end


%% =====================================================================
%  AERO ANGLE HELPERS
%  =====================================================================
function [alpha, beta] = aero_angles(Vb)
    u = Vb(1); v = Vb(2); w = Vb(3);
    Va = norm(Vb);
    if Va < 1e-3, Va = 1e-3; end
    alpha = atan2(w, u);
    beta  = asin(clamp(v/Va, -1, 1));
end


%% =====================================================================
%  GENERIC HELPERS
%  =====================================================================
function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end

function a = wrapToPi(a)
    a = mod(a + pi, 2*pi) - pi;
end

function x = rate_limit_scalar(x_prev, x_new, max_rate, dt)
    dx = clamp(x_new - x_prev, -max_rate*dt, max_rate*dt);
    x  = x_prev + dx;
end

function a = rate_limit_angle(a_prev, a_new, max_rate, dt)
    da = clamp(wrapToPi(a_new - a_prev), -max_rate*dt, max_rate*dt);
    a  = wrapToPi(a_prev + da);
end


%% =====================================================================
%  PLOTTING
%  =====================================================================
function plot_results(log)
    t = log.t;

    figure('Name','Ground Tracks','Position',[100 600 700 500]);
    plot(log.pos_t(:,2), log.pos_t(:,1), 'b-', 'LineWidth',1.5); hold on;
    plot(log.pos_o(:,2), log.pos_o(:,1), 'r-', 'LineWidth',1.5);
    axis equal; grid on;
    xlabel('East (m)'); ylabel('North (m)');
    legend('Target','Ownship');
    title('Ground Tracks (NED)');

    figure('Name','Engagement Metrics','Position',[100 100 700 600]);
    subplot(3,1,1);
    plot(t, log.range, 'LineWidth',1.2); grid on;
    ylabel('Range (m)'); title('Engagement Metrics');
    subplot(3,1,2);
    plot(t, log.epar, 'LineWidth',1.2); grid on;
    ylabel('Along-track e_{\parallel} (m)');
    subplot(3,1,3);
    plot(t, log.eperp, 'LineWidth',1.2); grid on;
    ylabel('Cross-track e_{\perp} (m)'); xlabel('Time (s)');

    figure('Name','Speed & Altitude','Position',[850 600 700 500]);
    subplot(2,1,1);
    plot(t, log.V_cmd,'b--','LineWidth',1.2); hold on;
    plot(t, log.V_o,  'r-', 'LineWidth',1.2);
    plot(t, log.V_t,  'g-', 'LineWidth',1.0); grid on;
    ylabel('Airspeed (m/s)'); legend('V_{cmd}','V_{own}','V_{tgt}');
    subplot(2,1,2);
    plot(t, log.h_cmd, 'b--','LineWidth',1.2); hold on;
    plot(t, -log.pos_o(:,3), 'r-', 'LineWidth',1.2);
    plot(t, -log.pos_t(:,3), 'g-', 'LineWidth',1.0); grid on;
    ylabel('Altitude (m)'); xlabel('Time (s)');
    legend('h_{cmd}','h_{own}','h_{tgt}');

    figure('Name','Euler Angles (Ownship)','Position',[850 100 700 600]);
    subplot(3,1,1);
    plot(t, rad2deg(log.euler_o(:,1)), 'LineWidth',1.2); grid on;
    ylabel('\phi (deg)'); title('Ownship Euler Angles');
    subplot(3,1,2);
    plot(t, rad2deg(log.euler_o(:,2)), 'LineWidth',1.2); grid on;
    ylabel('\theta (deg)');
    subplot(3,1,3);
    plot(t, rad2deg(log.euler_o(:,3)), 'LineWidth',1.2); grid on;
    ylabel('\psi (deg)'); xlabel('Time (s)');

    figure('Name','Aero Angles & Surfaces (Ownship)','Position',[1600 600 700 600]);
    subplot(3,1,1);
    plot(t, rad2deg(log.alpha_o), 'LineWidth',1.2); hold on;
    plot(t, rad2deg(log.beta_o),  'LineWidth',1.2); grid on;
    ylabel('deg'); legend('\alpha','\beta'); title('Aero Angles');
    subplot(3,1,2);
    plot(t, rad2deg(log.surf_o(:,1)), 'LineWidth',1.2); hold on;
    plot(t, rad2deg(log.surf_o(:,2)), 'LineWidth',1.2);
    plot(t, rad2deg(log.surf_o(:,3)), 'LineWidth',1.2); grid on;
    ylabel('Deflection (deg)'); legend('\delta_a','\delta_e','\delta_r');
    subplot(3,1,3);
    plot(t, log.surf_o(:,4), 'LineWidth',1.2); grid on;
    ylabel('Throttle'); xlabel('Time (s)');
end