function trailing_uav_sim()
% TRAILING_UAV_SIM
% Fixed-wing "trail behind at distance" simulator with:
% - Outer loop (guidance): chi_cmd, V_cmd, h_cmd (2 Hz)
% - Autopilot shaping: chi_cmd->phi_cmd, h_cmd->theta_cmd, V_cmd->throttle
% - Inner loop (attitude): phi,theta track commands with rate limits (200 Hz)
% - Plant kinematics: coordinated turn approximation + speed/altitude dynamics
%
% Coordinate frame: ENU (x East, y North, z Up). Altitude = z.

clc; close all;

%% =========================
%  Simulation parameters
%  =========================
dt       = 0.005;           % 200 Hz inner loop
Tend     = 120;             % seconds
N        = round(Tend/dt)+1;
t        = (0:N-1)'*dt;
g        = 9.81;

% Outer loop update rate
outer_hz = 2;               % 2 Hz guidance
outer_dt = 1/outer_hz;
outer_k  = round(outer_dt/dt);

%% =========================
%  Scenario / target motion
%  =========================
% Target flies a gentle maneuvering path (speed + heading changes).
target.V0      = 25;        % m/s
target.z0      = 120;       % m (altitude)
target.turnAmp = deg2rad(8);% heading oscillation amplitude (rad)
target.turnW   = 2*pi/60;   % heading oscillation frequency (rad/s)
target.climbAmp= 0.5;         % m altitude oscillation amplitude
target.climbW  = 2*pi/60;   % rad/s

%% =========================
%  Trailing objective
%  =========================
d_trail = 400;              % desired trailing distance (m)
dh_trail= 0;                % desired vertical offset relative to target (m)

%% =========================
%  Ownship initial state
%  =========================
own.p = [ -200; -400; 100];  % position [x;y;z] (m)
own.V = 22;                  % speed (m/s)
own.chi = deg2rad(20);       % course/track angle (rad)
own.phi = 0;                 % roll/bank (rad)
own.theta = 0;               % pitch (rad)
own.hdot = 0;

% Simple first-order attitude dynamics time constants
tau_phi   = 0.25;            % roll response time constant (s)
tau_theta = 0.40;            % pitch response time constant (s)

% Speed / altitude response time constants
tau_V     = 2.0;             % speed response (s)
tau_z     = 3.0;             % altitude response to theta (s)

%% =========================
%  Limits / protections
%  =========================
lim.phi_max   = deg2rad(35);   % bank limit
lim.phi_rate  = deg2rad(80);   % max roll rate (rad/s) for command shaping
lim.theta_max = deg2rad(12);   % pitch limit
lim.V_min     = 16;            % stall-ish protection
lim.V_max     = 35;

% Command rate limits (to prevent outer loop steps)
lim.chi_rate_cmd = deg2rad(20); % rad/s
lim.V_rate_cmd   = 2.0;         % m/s^2
lim.h_rate_cmd   = 3.0;         % m/s

%% =========================
%  Guidance gains (outer loop)
%  =========================
% Lateral: steer back to trail line (cross-track)
ky      = 0.8;               % course correction gain
y0      = 120;               % cross-track normalization (m) for sat()
L1      = 500;

% Longitudinal: along-track spacing control -> speed command
kx      = 0.03;              % (m -> m/s) gain
kdx     = 0.40;              % damping on along-track closure

% Altitude tracking
kh      = 0.04;              % (m -> rad) pitch cmd gain (through shaping)
khdot   = 0.12;              % vertical speed damping

%% =========================
%  Autopilot gains (shaping layer)
%  =========================
% Course hold -> bank command
kchi    = 2.5;               % bank per course error
kchidot = 0.6;               % bank per course rate error (optional)

% Speed hold -> "throttle" command (here mapped to V dynamics directly)
kV      = 0.8;
kIV     = 0.2;               % slow integral (optional)
intV    = 0;

% Altitude hold -> pitch command
kH      = 0.04;
kIH     = 0.0020;
intH    = 0;

%% =========================
%  Logs
%  =========================
P_t  = zeros(N,3); Vt = zeros(N,1); chit = zeros(N,1); zt = zeros(N,1);
P_o  = zeros(N,3); Vo = zeros(N,1); chio= zeros(N,1);
phi  = zeros(N,1); theta = zeros(N,1);

chi_cmd_log = zeros(N,1); V_cmd_log = zeros(N,1); h_cmd_log = zeros(N,1);
phi_cmd_log = zeros(N,1); theta_cmd_log = zeros(N,1);

r_log = zeros(N,1);
epar_log = zeros(N,1);
eperp_log = zeros(N,1);

%% =========================
%  Initialize guidance commands
%  =========================
chi_cmd = own.chi;
V_cmd   = own.V;
h_cmd   = own.p(3);

chi_cmd_prev = chi_cmd;
V_cmd_prev   = V_cmd;
h_cmd_prev   = h_cmd;

%% =========================
%  Main simulation
%  =========================
for k = 1:N
    tk = t(k);

    % --- Target "truth"
    % Heading oscillation around north (y+). Use chi_t as course.
    chi_t = target.turnAmp * sin(target.turnW * tk);
    V_t   = target.V0 + 0.0*sin(2*pi/25*tk);  % constant speed (can vary)
    z_t   = target.z0 + target.climbAmp * sin(target.climbW * tk);

    % Target position integration (simple)
    if k == 1
        p_t = [0; 0; z_t];
    else
        p_t = P_t(k-1,:)';
    end
    v_t_vec = V_t * [cos(chi_t); sin(chi_t); 0];   % ENU: x East, y North
    p_t = p_t + dt * v_t_vec;
    p_t(3) = z_t; % enforce altitude profile

    % --- Outer loop guidance update (every outer_k steps)
    if mod(k-1, outer_k) == 0
        % Build trail point behind target along its velocity direction
        t_hat = v_t_vec;
        if norm(t_hat(1:2)) < 1e-6
            t_hat = [cos(chi_t); sin(chi_t); 0];
        end
        t_hat = t_hat / norm(t_hat(1:2));  % horizontal unit

        p_ref = p_t - d_trail * [t_hat(1); t_hat(2); 0];
        p_ref(3) = z_t + dh_trail;

        % Error from ownship to trail point
        e = p_ref - own.p;

        % Along-track error (positive means trail point ahead of us)
        e_par = t_hat(1:2)' * e(1:2);

        % Signed cross-track error (left/right of target direction)
        n_hat = [-t_hat(2); t_hat(1)];  % +90 deg
        e_perp = n_hat' * e(1:2);

        % Along-track closure (project relative velocity)
        v_o_vec = own.V * [cos(own.chi); sin(own.chi); 0];
        de_par  = t_hat(1:2)' * (v_t_vec(1:2) - v_o_vec(1:2));

        % Guidance: course command = target course + cross-track correction
        chi_base = atan2(t_hat(2), t_hat(1));
        chi_cmd_new = chi_base + ky * atan2(e_perp, L1);  % bounded correction

        % Guidance: speed command to regulate spacing
        V_cmd_new = V_t + kx * e_par + kdx * de_par;

        % Guidance: altitude command
        h_cmd_new = p_ref(3);

        % Rate limit guidance commands to avoid steps
        chi_cmd = rate_limit_angle(chi_cmd_prev, chi_cmd_new, lim.chi_rate_cmd, outer_dt);
        V_cmd   = rate_limit_scalar(V_cmd_prev, V_cmd_new, lim.V_rate_cmd, outer_dt);
        h_cmd   = rate_limit_scalar(h_cmd_prev, h_cmd_new, lim.h_rate_cmd, outer_dt);

        % Clamp speed command
        V_cmd = min(max(V_cmd, lim.V_min), lim.V_max);

        chi_cmd_prev = chi_cmd;
        V_cmd_prev   = V_cmd;
        h_cmd_prev   = h_cmd;

        if k > N/2
            d_trail = 300;
        end
    end

    % --- Autopilot shaping (continuous)
    % Course hold -> bank cmd
    e_chi = wrapToPi(chi_cmd - own.chi);

    % Optional course-rate error (approximate own chi_dot from phi)
    % Coordinated turn approximation: chi_dot ~ g/V * tan(phi)
    chi_dot_est = (g/max(own.V,1e-3)) * tan(own.phi);
    chi_dot_cmd = 0; % could be derived from guidance; set to 0 for now

    phi_cmd = kchi * e_chi + kchidot * (chi_dot_cmd - chi_dot_est);
    phi_cmd = sat(phi_cmd / lim.phi_max) * lim.phi_max;  % saturate
    phi_cmd = rate_limit_scalar(own.phi, phi_cmd, lim.phi_rate, dt); % roll cmd smoothing
    phi_cmd = min(max(phi_cmd, -lim.phi_max), lim.phi_max);

    % Altitude hold -> pitch cmd
    e_h = h_cmd - own.p(3);
    % vertical speed estimate from state derivative approx (we don't keep v_z)
    % Use simple estimate from last step:
    if k < 3
        hdot = 0;
    else
        hdot = (P_o(k-1,3) - P_o(k-2,3)) / dt;
    end
    intH = intH + e_h*dt;
    theta_cmd = kH*e_h + kIH*intH - khdot*hdot;
    theta_cmd = min(max(theta_cmd, -lim.theta_max), lim.theta_max);

    % Speed hold -> throttle surrogate (here: desired acceleration via PI)
    e_V = V_cmd - own.V;
    intV = intV + e_V*dt;
    a_V_cmd = kV*e_V + kIV*intV;   % desired accel (m/s^2) surrogate

    % Speed protection: if too slow, reduce bank and pitch demand
    if own.V < lim.V_min + 1.0
        phi_cmd   = 0.5 * phi_cmd;
        theta_cmd = min(theta_cmd, deg2rad(5));
    end

    % --- Inner attitude dynamics (simplified first-order tracking)
    own.phi   = own.phi   + (dt/tau_phi)  * (phi_cmd   - own.phi);
    own.theta = own.theta + (dt/tau_theta)* (theta_cmd - own.theta);

    % --- Plant kinematics (coordinated turn + climb)
    % Turn rate from bank
    chi_dot = (g/max(own.V,1e-3)) * tan(own.phi);

    % Speed dynamics (first-order toward commanded accel)
    own.V = own.V + dt * sat(a_V_cmd / 5.0) * 5.0; % accel limited to +/-5 m/s^2
    own.V = min(max(own.V, lim.V_min), lim.V_max);

    % Altitude dynamics: use pitch to create climb rate ~ V*sin(theta)
    % Then low-pass with tau_z
    hdot_cmd = own.V * sin(own.theta);
    own.hdot = own.hdot + (dt/tau_z)*(hdot_cmd - own.hdot); % smooth
    own.p(3) = own.p(3) + dt * own.hdot;

    % Horizontal position update with current course and speed
    own.chi = wrapToPi(own.chi + dt * chi_dot);
    v_o_xy = own.V * [cos(own.chi); sin(own.chi)];
    own.p(1:2) = own.p(1:2) + dt * v_o_xy;

    % --- Log
    P_t(k,:) = p_t'; Vt(k) = V_t; chit(k)=chi_t; zt(k)=z_t;
    P_o(k,:) = own.p'; Vo(k)=own.V; chio(k)=own.chi;
    phi(k)=own.phi; theta(k)=own.theta;

    chi_cmd_log(k)=chi_cmd; V_cmd_log(k)=V_cmd; h_cmd_log(k)=h_cmd;
    phi_cmd_log(k)=phi_cmd; theta_cmd_log(k)=theta_cmd;

    % Metrics: true range and decomposition w.r.t target direction
    dp = (p_t - own.p);
    r_log(k) = norm(dp);
    % compute trail-frame errors for plotting at full rate
    t_hat2 = v_t_vec(1:2);
    if norm(t_hat2) < 1e-6
        t_hat2 = [cos(chi_t); sin(chi_t)];
    end
    t_hat2 = t_hat2 / norm(t_hat2);
    p_ref2 = p_t - d_trail*[t_hat2;0]; p_ref2(3)=z_t+dh_trail;
    e2 = p_ref2 - own.p;
    epar_log(k)  = t_hat2' * e2(1:2);
    n_hat2 = [-t_hat2(2); t_hat2(1)];
    eperp_log(k) = n_hat2' * e2(1:2);
end

%% =========================
%  Plots
%  =========================
figure; plot(P_t(:,1), P_t(:,2), 'LineWidth', 1.5); hold on;
plot(P_o(:,1), P_o(:,2), 'LineWidth', 1.5);
axis equal; grid on;
xlabel('x East (m)'); ylabel('y North (m)');
legend('Target','Ownship');
title('Ground Tracks');

figure; subplot(3,1,1);
plot(t, r_log, 'LineWidth', 1.2); grid on;
ylabel('Range (m)'); title('Engagement Metrics');

subplot(3,1,2);
plot(t, epar_log, 'LineWidth', 1.2); grid on;
ylabel('Along-track e_{\parallel} (m)');

subplot(3,1,3);
plot(t, eperp_log, 'LineWidth', 1.2); grid on;
ylabel('Cross-track e_{\perp} (m)'); xlabel('Time (s)');

figure; subplot(3,1,1);
plot(t, wrapToPi(chi_cmd_log - chio), 'LineWidth', 1.2); grid on;
ylabel('e_\chi (rad)'); title('Commands & Attitude');

subplot(3,1,2);
plot(t, phi_cmd_log, 'LineWidth', 1.2); hold on;
plot(t, phi, 'LineWidth', 1.2); grid on;
ylabel('\phi (rad)'); legend('\phi_{cmd}','\phi');

subplot(3,1,3);
plot(t, V_cmd_log, 'LineWidth', 1.2); hold on;
plot(t, Vo, 'LineWidth', 1.2); grid on;
ylabel('V (m/s)'); xlabel('Time (s)'); legend('V_{cmd}','V');

figure; subplot(2,1,1);
plot(t, h_cmd_log, 'LineWidth', 1.2); hold on;
plot(t, P_o(:,3), 'LineWidth', 1.2); grid on;
ylabel('Altitude z (m)'); legend('h_{cmd}','z');

subplot(2,1,2);
plot(t, theta_cmd_log, 'LineWidth', 1.2); hold on;
plot(t, theta, 'LineWidth', 1.2); grid on;
ylabel('\theta (rad)'); xlabel('Time (s)'); legend('\theta_{cmd}','\theta');

end

%% ========= Helpers =========
function y = sat(x)
% Saturation to [-1,1]
y = min(max(x, -1), 1);
end

function a = wrapToPi(a)
% Wrap angle to [-pi, pi]
a = mod(a + pi, 2*pi) - pi;
end

function x = rate_limit_scalar(x_prev, x_new, max_rate, dt)
dx = x_new - x_prev;
dx = min(max(dx, -max_rate*dt), max_rate*dt);
x  = x_prev + dx;
end

function a = rate_limit_angle(a_prev, a_new, max_rate, dt)
da = wrapToPi(a_new - a_prev);
da = min(max(da, -max_rate*dt), max_rate*dt);
a  = wrapToPi(a_prev + da);
end