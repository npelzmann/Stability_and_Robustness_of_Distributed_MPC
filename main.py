from cvxpy import *
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

import utils

nw = utils.network()

# Prediction horizon
N = 5

# Define problem
u = Variable((nw.nu, N))
x = Variable((nw.nx, N+1))
x_init = Parameter(nw.nx)
objective = 0
constraints = [x[:,0] == x_init]
for k in range(N):
    objective += quad_form(x[:,k] - nw.xt, nw.Q) + quad_form(u[:,k], nw.R)
    constraints += [x[:, k+1] == nw.A @ x[:, k] + nw.B @ u[:, k]]
    constraints += [nw.E_x @ x[:, k] + nw.E_u @ u[:, k] <= nw.b]
objective += quad_form(x[:,N] - nw.xt, nw.P)
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
x0 = nw.x0
nsim = 15
trajectory = np.zeros((nw.nx, nsim + 1))
trajectory[:, 0] = x0
applied_u = np.zeros((nw.nu, nsim))
for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    x0 = nw.A.dot(x0) + nw.B.dot(u[:,0].value)
    trajectory[:, i+1] = x0
    applied_u[:, i] = u[:,0].value

for i in range(nw.M):
    plt.plot(trajectory[4*i, :], trajectory[4*i+2, :], marker='x')

plt.show()
print('tst')