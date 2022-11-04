import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

import utils

nw = utils.network(N=3)

# Simulate in closed loop
nsim = 15
trajectory = np.zeros((nw.nx, nsim + 1))
trajectory[:, 0] = nw.x
control_inputs = np.zeros((nw.nu, nsim))
for i in range(nsim):
    u_centr = nw.centr_solve()
    # u = nw.fdfbs_solve()
    nw.simulate_timestep(u_centr)
    trajectory[:, i+1] = nw.x
    control_inputs[:, i] = np.hstack(u_centr)

for i in range(nw.M):
    plt.plot(trajectory[4*i, :], trajectory[4*i+2, :], marker='x')

plt.show()