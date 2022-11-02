import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy import eye
import cvxpy


def condensed_form(A: np.array, B: np.array, 
                   Q: np.array, R: np.array, 
                   P: np.array, E_x: np.array, 
                   E_u: np.array, C_x: np.array, 
                   C_u: np.array, C_N: np.array,
                   c_x: np.array, c_u: np.array, 
                   c_N: np.array, N: int) -> np.array:
    nx, nu = B.shape
    I_N = np.eye(N)
    Bh = np.zeros((N*nx, N*nu))
    Ah = np.zeros((N*nx, N*nx))
    for j in range(1, N+1):
        mul = np.eye(N, N, j)
        Bh += sp.kron(mul, np.linalg.matrix_power(A, j-1) @ B)
        mul = np.eye(N, 1, j)
        Ah += sp.kron(mul, np.linalg.matrix_power(A, j))

    Hh = sp.block_diag(sp.kron(I_N, Q), P)

    H = Bh.T @ Hh @ Bh + sp.kron(I_N, R)
    G = Bh.T @ Hh @ Ah 
    W = Q + Ah.T @ Hh @ Ah 

    D = sp.block_diag(0, (sp.kron(I_N, C_x) @ Ah))
    C_ = np.vstack(sp.kron(np.eye(N-1), C_x), C_N)
    C = np.vstack(sp.kron(I_N, C_u), C_ @ Bh)
    ch = np.vstack(sp.kron(np.ones(N, 1), c_u), 
                   sp.kron(np.ones(N-1, 1), c_x),
                   c_N)
    b = sp.kron(np.ones(N, 1), b)
    F = sp.kron(I_N, E_x) @ Ah 
    E = sp.kron(I_N, E_x) @ Bh + sp.kron(I_N, E_u)

    M = np.vstack(np.hstack(H, G), np.hstack(G.T, W))

    return M, C, D, E, F, ch, b


class node(object):

    def __init__(self, 
                 x0: list = [0., 0., 0., 0.],
                 xt: list = [0., 0., 0., 0.]) -> None:
        
        self.x = x0
        self.xt = xt
        self.A = sp.kron(eye(2), np.array([[1., 1.], [0., 1.]])).todense()
        self.B = sp.kron(eye(2), np.array([[0.], [1.]])).todense()
        
        self.nx, self.nu = np.shape(self.B)

        self.R = eye(self.nu) * 5
        self.Q = eye(self.nx)
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)

    def simulate_timestep(self, u: np.array):
        self.x = np.squeeze(np.asarray(self.A.dot(self.x) + self.B.dot(u)))


class network(object):

    def __init__(self) -> None:
        self.nodes = np.array([node(x0=[0., 0.5, 0., -0.5], xt=[1., 0.,  1.,  0.]), 
                               node(x0=[1.5, -0.5, 0., 0.], xt=[2.,   0.,  1.,  0.]), 
                               node(x0=[0.5, 0.2, 0.7, -0.2], xt=[1.5, 0., 1 + 1/np.sqrt(2), 0.])])
        self.D = np.array([[1, 0, 1], [-1, 1, 0], [0, -1, -1]])
        N = 5

        self.M = len(self.nodes)
        self.nu = np.sum([n.nu for n in self.nodes])
        self.nx = np.sum([n.nx for n in self.nodes])
        
        self.xt = np.hstack([n.xt for n in self.nodes])
        self.A = la.block_diag(*[n.A for n in self.nodes])
        self.B = la.block_diag(*[n.B for n in self.nodes])
        self.R = la.block_diag(*[n.R for n in self.nodes])
        self.Q = la.block_diag(*[n.Q for n in self.nodes])
        self.P = la.block_diag(*[n.P for n in self.nodes])
        sel_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.E_x= la.kron(self.D.T, sel_mat)
        self.E_u = np.zeros((np.shape(self.E_x)[0], self.nu))
        self.b = np.ones(np.shape(self.E_x)[0])

        self.x_init = cvxpy.Parameter(self.nx)
        self.u_var = cvxpy.Variable((self.nu, N))
        self.centr_prob_def(N)

    @property
    def x(self):
        return np.hstack([n.x for n in self.nodes])

    def simulate_timestep(self, u: np.array):
        u_split = np.split(u, np.cumsum([n.nu for n in self.nodes]))
        for i, n in enumerate(self.nodes):
            n.simulate_timestep(u_split[i])
    
    def centr_solve(self) -> np.array:
        self.x_init.value = self.x
        self.centr_prob.solve(solver=cvxpy.OSQP, warm_start=True)
        u = self.u_var[:,0].value
        return u

    def fdfbs_solve(self) -> np.array:
        pass

    def centr_prob_def(self, N: int = 5) -> None:
        # Define problem
        x = cvxpy.Variable((self.nx, N+1))
        objective = 0
        constraints = [x[:,0] == self.x_init]
        for k in range(N):
            objective += cvxpy.quad_form(x[:,k] - self.xt, self.Q) + cvxpy.quad_form(self.u_var[:,k], self.R)
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ self.u_var[:, k]]
            constraints += [self.E_x @ x[:, k] + self.E_u @ self.u_var[:, k] <= self.b]
        objective += cvxpy.quad_form(x[:,N] - self.xt, self.P)

        self.centr_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)