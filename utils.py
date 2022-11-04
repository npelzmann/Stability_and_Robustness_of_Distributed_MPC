import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy import eye
import cvxpy
from typing import List


def condensed_form(A: np.array, B: np.array, 
                   Q: np.array, R: np.array, 
                   P: np.array, E_x: np.array, 
                   E_u: np.array, bb: np.array, C_x: np.array = None, 
                   C_u: np.array = None, C_N: np.array = None,
                   c_x: np.array = None, c_u: np.array = None, 
                   c_N: np.array = None, N: int = 2) -> np.array:
    nx, nu = B.shape
    I_N = np.eye(N)
    I_Np = np.eye(N+1)
    Bh = np.zeros(((N+1)*nx, N*nu))
    Ah = sp.kron(np.eye(N+1,1), np.eye(nx)).todense()
    for j in range(1, N+1):
        mul = np.eye(N+1, N, -j)
        Bh += sp.kron(mul, np.linalg.matrix_power(A, j-1) @ B)
        mul = np.eye(N+1, 1, -j)
        Ah += sp.kron(mul, np.linalg.matrix_power(A, j))

    Hh = sp.block_diag((sp.kron(I_N, Q), P)).todense()

    H = Bh.T @ Hh @ Bh + sp.kron(I_N, R)
    G = Bh.T @ Hh @ Ah 
    W = Q + Ah.T @ Hh @ Ah 

    b = sp.kron(np.ones((N, 1)), bb.reshape(-1, 1))
    F = sp.kron(I_Np, E_x) @ Ah 
    E = sp.kron(I_Np, E_x) @ Bh + sp.kron(np.vstack((I_N, np.zeros((1, N)))), E_u)

    M = np.block([[H, G], [G.T, W]])

    if C_x is None and C_u is None and C_N is None:
        D = C = ch = None
    elif C_x is None:
        C = sp.kron(I_N, C_u)
        ch = sp.kron(np.ones((N, 1)), c_u.reshape(-1, 1))
        D = None
    elif C_u is None:
        if C_N is None:
            C_N = C_x
            c_N = c_x
        D = sp.block_diag(0, (sp.kron(I_N, C_x) @ Ah))
        C = np.vstack(sp.kron(np.eye(N-1), C_x), C_N) @ Bh
        ch = np.vstack(sp.kron(np.ones(N-1, 1), c_x),
                       c_N)
    else:
        D = sp.block_diag(0, (sp.kron(I_N, C_x) @ Ah))
        C_ = np.vstack(sp.kron(np.eye(N-1), C_x), C_N)
        C = np.vstack(sp.kron(I_N, C_u), C_ @ Bh)
        ch = np.vstack(sp.kron(np.ones(N, 1), c_u), 
                       sp.kron(np.ones(N-1, 1), c_x),
                       c_N)

    return M, E, F, b, C, D, ch


class node(object):

    def __init__(self, 
                 x0: list = [0., 0., 0., 0.],
                 xt: list = [0., 0., 0., 0.],
                 N: int = 2) -> None:
        self.N = N
        self.x = x0
        self.xt = xt
        self.A = sp.kron(eye(2), np.array([[1., 1.], [0., 1.]])).todense()
        self.B = sp.kron(eye(2), np.array([[0.], [1.]])).todense()
        self.C_u = np.array([[1., 0.], [0., -1.]])
        self.c_u = np.array([1, 1])
        
        self.nx, self.nu = np.shape(self.B)

        self.R = eye(self.nu) * 5
        self.Q = eye(self.nx)
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)

        self.M = self.E = self.F = self.b = None

    def dense_problem_matrices(self, E_x: np.array, E_u: np.array, bb: np.array):
        self.M, self.E, self.F, self.b, _, _, _ = condensed_form(self.A, self.B, 
                                                                 self.Q, self.R, self.P,
                                                                 E_x, E_u, bb,
                                                                 C_u=self.C_u, c_u=self.c_u, N=self.N)

    def simulate_timestep(self, u: np.array):
        self.x = np.squeeze(np.asarray(self.A.dot(self.x) + self.B.dot(u)))

    def formulate_primal(self):
        self.lda = cvxpy.Parameter()
        self.x_par = cvxpy.Parameter(self.nx)
        self.xi = cvxpy.Variable(self.nu * N)
        objective = self.xi.T @ self.M @ self.xi + (2 * self.x_par.T @ self.G.T + self.lda.T @ self.E) @ self.xi
        if self.D:
            constraints = [self.D @ self.x_par + self.C @ self.xi <= self.c]
        self.node_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def solve_primal(self, lda: np.array):
        if self.node_prob is None:
            self.formulate_primal()
        self.lda.value = lda
        self.x_par.value = self.x
        self.node_prob.solve(solver=cvxpy.OSQP, warm_start=True)
        u = self.xi.value
        return u, self.F @ self.x + self.E @ u


class network(object):

    def __init__(self, 
                 N: int = 2, 
                 fdfbs_iterations: int = 100,
                 fdfbs_eps: float = 1e-4) -> None:
        self.nodes = np.array([node(x0=[0., 0.5, 0., -0.5], xt=[1., 0.,  1.,  0.], N=N), 
                               node(x0=[1.5, -0.5, 0., 0.], xt=[2.,   0.,  1.,  0.], N=N), 
                               node(x0=[0.5, 0.2, 0.7, -0.2], xt=[1.5, 0., 1 + 1/np.sqrt(2), 0.], N=N)])
        self.D = np.array([[1, 0, 1], [-1, 1, 0], [0, -1, -1]])

        self.N = N
        self.lb = fdfbs_iterations

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
        self.E_x = la.kron(self.D.T, sel_mat)
        self.E_u = np.zeros((np.shape(self.E_x)[0], self.nu))
        self.C_u = la.block_diag(*[n.C_u for n in self.nodes])
        self.c_u = np.hstack([n.c_u for n in self.nodes])
        self.bb = np.ones(np.shape(self.E_x)[0])

        self.x_init = cvxpy.Parameter(self.nx)
        self.u_var = cvxpy.Variable((self.nu, N))
        self.centr_prob_def()

        self.dense_agent_problem_matrices()

    @property
    def x(self):
        return np.hstack([n.x for n in self.nodes])

    def dense_agent_problem_matrices(self):
        E_x_split = np.split(self.E_x, np.cumsum([n.nx for n in self.nodes]), axis=1)
        E_u_split = np.split(self.E_u, np.cumsum([n.nu for n in self.nodes]), axis=1)
        for i, n in enumerate(self.nodes):
            n.dense_problem_matrices(E_x_split[i], E_u_split[i], self.bb)

    def simulate_timestep(self, u: List[np.array]):
        for i, n in enumerate(self.nodes):
            n.simulate_timestep(u[i])
    
    def centr_solve(self) -> List[np.array]:
        self.x_init.value = self.x
        self.centr_prob.solve(solver=cvxpy.OSQP, warm_start=True)
        u = self.u_var[:,0].value
        u_split = np.split(u, np.cumsum([n.nu for n in self.nodes]))
        return u_split[:-1]

    def fdfbs_solve(self) -> List[np.array]:
        lda0 = np.zeros(self.N * self.E_x.shape[0])
        lda = [lda0]
        mu = lda0
        theta = 1.
        for l in range(self.lb):
            dlda = 0
            u = []
            for n in self.nodes:
                _u, _dlda = n.solve_primal(lda[l])
                dlda += _dlda - self.b + self.epsilon * lda[l]
                u.append(_u)
            mu_new = np.maximum(np.zeros(lda.shape), lda[l] + self.alpha * dlda)
            theta_new = 0.5 * (1 + np.sqrt(1. + 4. * theta**2.))
            lda[l+1] = mu + (theta - 1) / theta_new * (mu_new - mu)
            theta = theta_new, mu = mu_new

        return u

    def centr_prob_def(self) -> None:
        # Define problem
        x = cvxpy.Variable((self.nx, self.N+1))
        objective = 0
        constraints = [x[:,0] == self.x_init]
        for k in range(self.N):
            objective += cvxpy.quad_form(x[:,k] - self.xt, self.Q) + cvxpy.quad_form(self.u_var[:,k], self.R)
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ self.u_var[:, k]]
            constraints += [self.E_x @ x[:, k] + self.E_u @ self.u_var[:, k] <= self.bb]
            constraints += [self.C_u @ self.u_var[:, k] <= self.c_u]
        objective += cvxpy.quad_form(x[:,self.N] - self.xt, self.P)

        self.centr_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)