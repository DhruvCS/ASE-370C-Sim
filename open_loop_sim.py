import numpy as np
import matplotlib.pyplot as plt

# ================= SYSTEM PARAMETERS ==================== #
m = 9200 # Mass of Aircraft, kg
A_w = 27.87 # Wing Surface Area, m^2
c = 3.45 # Chord Length, m
I = 75e3 # Pitch Inertia, kg*m^2
U = 175 # Nominal Velocity, m/s
rho = 1 # Air Density, kg/m^3
g = 9.81  # Gravitational acceleration, m/s^2

CD = 0.02 # Drag Coefficient
CL_alpha = 5 # Lift Coefficient per unit Attack Angle, /rad
CL_gamma = 0.4 # Lift Coefficient per unit Elevator, /rad
CM_alpha = 0.2 # Moment Coefficient per unit Attack Angle, /rad
CM_thetadot = -10 # Moment Coefficient per unit Pitch Rate, s/rad
CM_gamma = -1.2 # Moment Coefficient per unit Elevator, /rad

A_33 = -CD*rho*U*A_w/m
A_44 = -CL_alpha*rho*U*A_w/(2*m)
A_64 = CM_alpha*rho*U*c*A_w/(2*I)
A_66 = CM_thetadot*rho*U*c**2*A_w/(4*I)
B_t = 1/m
B_4 = -CL_gamma*rho*U**2*A_w/(2*m)
B_6 = CM_gamma*rho*U**2*c*A_w/(2*I)

# ================ OPEN-LOOP SIMULATOR =================== #

""" State = [x z u w theta theta_dot]^T
    Control Inputs: [elevator throttle]^T"""

A = np.array([[0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, A_33, 0, -g, 0],
              [0, 0, 0, A_44, 0, U],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, A_64, 0, A_66]])

B = np.array([[0, 0],
              [0, 0],
              [0, B_t],
              [B_4,0],
              [0, 0],
              [B_6, 0]])

T = 10.0
dt = 0.01
t = np.linspace(0,T,int(T/dt+1))
q0 = np.transpose(np.array([0, 200, U, 0, 0, 0]))
U_in = np.zeros((np.size(B,1),np.size(t)))
U_in[0,1:int(T/(2*dt))] += 0.05
U_in[1,:] += 10000

X = np.zeros((np.size(q0),np.size(t)))
X[:,0] = q0

for i in range(0,np.size(t)-1):
    dxdt = A@X[:,i] + B@U_in[:,i]
    X[:,i+1] = X[:,i] + dxdt*dt

fig, axs = plt.subplots(2,2,figsize=(16,10),dpi=150)

axs[0,0].plot(t,X[0,:])
axs[0,0].plot(t,X[1,:])
axs[0,0].set_ylabel("Position (m)")
axs[0,0].legend(["Surge", "Heave"])
axs[0,0].set_title("Position of Aircraft for Throttle and Elevator Step Inputs")

axs[0,1].plot(t,X[2,:])
axs[0,1].plot(t,X[3,:])
axs[0,1].set_ylabel("Velocity (m/s)")
axs[0,1].legend(["Surge", "Heave"])
axs[0,1].set_title("Velocity of Aircraft for Throttle and Elevator Step Inputs")

axs[1,0].plot(t,180/np.pi*X[4,:])
axs[1,0].set_xlabel("Time (s)")
axs[1,0].set_ylabel("Pitch (deg)")
axs[1,0].set_title("Pitch of Aircraft for Throttle and Elevator Step Inputs")

axs[1,1].plot(t,X[5,:])
axs[1,1].set_xlabel("Time (s)")
axs[1,1].set_ylabel("Pitch Rate (rad/s)")
axs[1,1].set_title("Pitch Rate of Aircraft for Throttle and Elevator Step Inputs")

plt.show()