import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

x0 = 10
omega = 100
g = 9.81
l = 10

phi0 = 0.1
phi0_dot = 0.0

times = np.linspace(0, 100, 10000)
state = np.array([phi0, phi0_dot])

def func(t, y):
    phi, phid = y
    phidd = g/(l*omega**2)*np.sin(phi) + x0/l * np.sin(t)*np.cos(phi)
    return np.array([phid, phidd])

sol = solve_ivp(func, (times[0], times[-1]), state, t_eval=times)

# plt.plot(times, np.mod(sol.y[0],2*np.pi))
plt.plot(times/omega, sol.y[0])
plt.xlabel('Time (s)')
plt.ylabel('φ Displacement (rad)')
plt.show()

