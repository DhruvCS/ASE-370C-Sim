import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# ================= SYSTEM PARAMETERS ==================== #
m = 9200
A_w = 27.87
c = 3.45
I = 75e3
U0 = 175.0
rho = 1.0
g = 9.81

CD = 0.02
CL_alpha = 5.0
CL_gamma = 0.4
CM_alpha = -0.7
CM_thetadot = -10.0
CM_gamma = -1.2

A_33 = -CD*rho*U0*A_w/m
A_44 = -CL_alpha*rho*U0*A_w/(2*m)
A_64 = CM_alpha*rho*U0*c*A_w/(2*I)
A_66 = CM_thetadot*rho*U0*c**2*A_w/(4*I)

B_t = 1.0/m
B_4 = -CL_gamma*rho*U0**2*A_w/(2*m)
B_6 = CM_gamma*rho*U0**2*c*A_w/(2*I)

# ================ FULL 6-STATE PLANT =================== #
# x6 = [x, z, u, w, theta, q]
A6 = np.array([[0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, A_33, 0, -g, 0],
               [0, 0, 0, A_44, 0, U0],
               [0, 0, 0, 0, 0, 1],
               [0, 0, 0, A_64, 0, A_66]], dtype=float)

B6 = np.array([[0,   0],
               [0,   0],
               [0,   B_t],   # throttle -> u_dot
               [B_4, 0],     # elevator -> w_dot
               [0,   0],
               [B_6, 0]], dtype=float)

# ============= CONTROLLABLE 4-STATE PLANT ============== #
A4 = A6[2:6, 2:6]
B4 = B6[2:6, :]

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]])

# LQR Controller
Q = np.diag([1.0, 1.0, 200.0, 10.0])
R = np.diag([2.0, 1.0])

P = solve_continuous_are(A4, B4, Q, R)
K = np.linalg.solve(R, B4.T@P) # LQR Controller Gain

A_CL = A4 - B4@K

# Feedforward Controller
G = C@np.linalg.inv(A_CL)@B4
N = -np.linalg.pinv(G) # Feedforward Compensator Gain

# ==================== SIMULATION ===================== #
Tf = 20.0
dt = 0.01
n = int(Tf/dt + 1)
t = np.linspace(0,10.0,n)

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

X = np.zeros((np.size(A6,1),np.size(t)))
X[:,0] = np.array([0.0, 200.0, U0, 0.0, 0.0, 0.0])

U = np.zeros((np.size(B6,1),np.size(t)))

# Reference States
u_ref = U0*np.ones_like(t)
u_ref[t>=0.5] = 200.0
w_ref = np.zeros_like(t)
theta_ref = np.zeros_like(t)

# Control Limits
elev_max = np.deg2rad(25)
throttle_max = 40e3

for i in range(0, n-1):
    x6 = X[:,i]
    x4 = x6[2:6]
    r = np.array([u_ref[i], w_ref[i], theta_ref[i]], dtype=float)

    u_ctrl = -K@x4 + N@r
    
    u_ctrl[0] = np.clip(u_ctrl[0], -elev_max, elev_max)
    u_ctrl[1] = np.clip(u_ctrl[1], -throttle_max, throttle_max)
    U[:,i] = u_ctrl

    x6_dot = A6@x6 + B6@u_ctrl
    X[:,i+1] = X[:,i] + x6_dot*dt

U[:,-1] = U[:,-2]

# ================== PLOTS ================== #
fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=150)

axs[0,0].plot(t, X[0,:], label="x")
axs[0,0].plot(t, X[1,:], label="z")
axs[0,0].set_title("Position")
axs[0,0].set_ylabel("m")
axs[0,0].legend(); axs[0,0].grid(True)

axs[0,1].plot(t, X[2,:], label="u")
axs[0,1].plot(t, u_ref, "--", label="u_ref")
axs[0,1].plot(t, X[3,:], label="w")
axs[0,1].plot(t, w_ref, "--", label="w_ref")
axs[0,1].set_title("Tracked velocities")
axs[0,1].set_ylabel("m/s")
axs[0,1].legend(); axs[0,1].grid(True)

axs[1,0].plot(t, np.rad2deg(X[4,:]), label="theta")
axs[1,0].plot(t, np.rad2deg(theta_ref), "--", label="theta_ref")
axs[1,0].set_title("Tracked pitch")
axs[1,0].set_xlabel("s")
axs[1,0].set_ylabel("deg")
axs[1,0].legend(); axs[1,0].grid(True)

axs[1,1].plot(t, np.rad2deg(U[0,:]), label="delta_e (deg)")
axs[1,1].plot(t, U[1,:], label="throttle (N)")
axs[1,1].set_title("Controls")
axs[1,1].set_xlabel("s")
axs[1,1].legend(); axs[1,1].grid(True)

plt.tight_layout()
plt.show()