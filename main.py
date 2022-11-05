import numpy as np
import matplotlib.pyplot as plt

import utils

nw = utils.network(N=5)

# Simulate in closed loop
nsim = 15
trajectory = np.zeros((nw.nx, nsim + 1))
trajectory[:, 0] = nw.x
control_inputs = np.zeros((nw.nu, nsim))
for i in range(nsim):
    # u_centr = nw.centr_solve()
    u_centr2 = nw.centr_solve2()
    # u = nw.fdfbs_solve()
    nw.simulate_timestep(u_centr2)
    trajectory[:, i+1] = nw.x
    control_inputs[:, i] = np.hstack(u_centr2)

for i in range(nw.M):
    plt.plot(trajectory[4*i, :], trajectory[4*i+2, :], marker='x')

plt.show()
print('success')