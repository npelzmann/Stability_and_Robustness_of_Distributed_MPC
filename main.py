from cvxpy import *
import numpy as np
import scipy as sp
from scipy import sparse

import utils

nw = utils.network()

# Discrete time model of a quadcopter
Ad = nw.A
Bd = nw.B
nx = nw.nx
nu = nw.nu

# Constraints
u0 = np.zeros(nu)

# Objective function
Q = nw.Q
P = nw.P
R = nw.R

# Initial and reference states
x0 = np.zeros(nx)
xr = np.zeros(nx)
xr[0:2:-1] = 1

# Prediction horizon
N = 10

# Define problem
u = Variable((nu, N))
x = Variable((nx, N+1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:,0] == x_init]
for k in range(N):
    objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k], R)
    constraints += [x[:, k+1] == Ad*x[:, k] + Bd*u[:, k]]
    constraints += [nw.E_x @ x[:, k] + nw.E_u @ u[:, k] <= nw.b]
objective += quad_form(x[:,N] - xr, P)
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
nsim = 15
for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)