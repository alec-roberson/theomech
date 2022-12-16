import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

x0 = 1
omega = 10
g = 1
l = 1

TIME_MAX = 100
INTERVALS = 10000

times = np.linspace(0, TIME_MAX, INTERVALS)

# initial_states = [np.array([0,0])]
initial_states = []

for phi0 in np.linspace(0.1, 0.11, 100):
    initial_states.append([phi0,0 ])

def func(tau, s):
    phi, phid = s
    phidd = g/(l*omega**2)*np.sin(phi) - x0/l * np.sin(tau)*np.cos(phi)
    return np.array([phid, phidd])

plt.figure(figsize=(8,8))

for s0 in initial_states:
    sol = solve_ivp(func, [0, TAU_MAX], s0, t_eval=taus)
    color =  np.random.rand(3)
    plt.subplot(311)
    plt.plot(times, sol.y[0], color=color)
    plt.subplot(312)
    plt.plot(times, sol.y[1]*omega, color=color)
    plt.subplot(313)
    plt.plot(sol.y[0], sol.y[1], color=color)



# plt.plot(times, np.mod(sol.y[0],2*np.pi))
plt.subplot(311)
plt.xlabel('Time (s)')
plt.ylabel('φ Displacement (rad)')

plt.subplot(312)
plt.xlabel('Time (s)')
plt.ylabel('φ Velocity (rad/s)')

plt.subplot(313)
plt.xlabel('φ Displacement (rad)')
plt.ylabel('φ Velocity (rad/s)')

# plt.subplot(314)
# plt.xlabel('Time (s)')
# plt.ylabel('x Displacement')
# plt.plot(times, x0*np.sin(taus))

plt.tight_layout(pad=1)
plt.show()

