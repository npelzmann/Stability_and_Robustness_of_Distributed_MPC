import os
import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy import eye
import cvxpy
from typing import List
import matplotlib.pyplot as plt
import copy

plt.rcParams.update({
    'text.usetex': True
})


def condensed_form(A: np.array, B: np.array, 
                   Q: np.array, R: np.array, 
                   P: np.array, E_x: np.array, 
                   E_u: np.array, bb: np.array, C_x: np.array = None, 
                   C_u: np.array = None, C_N: np.array = None,
                   c_x: np.array = None, c_u: np.array = None, 
                   c_N: np.array = None, x_t: np.array = None, N: int = 2) -> np.array:
    nx, nu = B.shape
    x_t = np.zeros(nx) if x_t is None else x_t
    I_N = np.eye(N)
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
    gxt = (-2 * Bh.T @ Hh.T @ sp.kron(np.ones(N+1), x_t).reshape(-1, 1)).T

    b = np.kron(np.ones(N-1), bb)
    xi_mul = sp.hstack((sp.kron(I_N[1:,:], E_x), sp.csr_matrix(np.zeros(((N-1)*E_x.shape[0], nx)))))
    F = xi_mul @ Ah 
    E = xi_mul @ Bh + sp.kron(I_N[1:,:], E_u)

    M = np.block([[H, G], [G.T, W]])

    if C_x is None and C_u is None and C_N is None:
        D = C = ch = None
    elif C_x is None:
        C = sp.kron(I_N, C_u)
        ch = np.kron(np.ones(N), c_u)
        D = np.zeros((C.shape[0], (N+1)*nx)) @ Ah
    elif C_u is None:
        if C_N is None:
            C_N = C_x
            c_N = c_x
        D = np.vstack(np.zeros((1, (N+1)*nx)), sp.block_diag((sp.kron(I_N, C_x), C_N))) @ Ah
        C = np.vstack(sp.kron(I_N, C_x), C_N) @ Bh
        ch = np.hstack(np.kron(np.ones(N-1), c_x),
                       c_N)
    else:
        D = np.vstack(np.zeros((1, (N+1)*nx)), sp.block_diag((sp.kron(I_N, C_x), C_N))) @ Ah
        C_ = sp.block_diag((sp.kron(I_N, C_x), C_N))
        C = np.vstack(sp.kron(I_N, C_u), C_ @ Bh)
        ch = np.hstack(np.kron(np.ones(N), c_u), 
                       np.kron(np.ones(N), c_x),
                       c_N)

    return M, H, G, gxt, E, F, b, C, D, ch


class node(object):

    def __init__(self, 
                 x0: list = [0., 0., 0., 0.],
                 xt: list = [0., 0., 0., 0.],
                 v_max: float = 1.0,
                 N: int = 5) -> None:
        self.N = N
        self.x = x0
        self.xt = xt
        self.A = sp.kron(eye(2), np.array([[1., 1.], [0., 1.]])).todense()
        self.B = sp.kron(eye(2), np.array([[0.], [1.]])).todense()
        self.C_u = np.array([[1., 0.], [0., -1.]])
        self.c_u = np.array([1, 1])
        self.C_x = sp.kron(np.array([[1, 0], [0, -1]]), np.array([[0, 1, 0, 0], [0, 0, 0, 1]]))
        self.c_x = np.array([1, 1, 1, 1]) * v_max
        
        self.nx, self.nu = np.shape(self.B)

        self.R = eye(self.nu) * 100
        self.Q = eye(self.nx)
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)

        self.node_prob = None
        self.M = self.H = self.G = self.gxt = None
        self.E = self.F = self.b = None
        self.C = self.D = self.ch = None
        self.x_par = self.x_par_centr = self.z = self.xi = self.lda = None

    def dense_problem_matrices(self, E_x: np.array, E_u: np.array, bb: np.array):
        _, self.H, self.G, self.gxt, self.E, self.F, \
        self.b, self.C, self.D, self.c = condensed_form(self.A, self.B, 
                                                        self.Q, self.R, self.P,
                                                        E_x, E_u, bb,
                                                        C_u=self.C_u, c_u=self.c_u, 
                                                        x_t=self.xt, N=self.N)
        
    def set_up_cvx_vars(self):
        self.x_par_centr = cvxpy.Parameter(self.nx)
        self.z = cvxpy.Variable(self.nu * self.N)

    def simulate_timestep(self, u: np.array):
        self.x = np.squeeze(np.asarray(self.A.dot(self.x) + self.B.dot(u)))

    def formulate_primal(self):
        self.lda = cvxpy.Parameter(self.E.shape[0])
        self.x_par = cvxpy.Parameter(self.nx)
        self.xi = cvxpy.Variable(self.nu * self.N)
        objective = cvxpy.quad_form(self.xi, self.H)
        objective += 2 * self.x_par.T @ self.G.T @ self.xi
        objective += self.gxt @ self.xi
        objective += self.lda.T @ self.E @ self.xi
        constraints = [self.D @ self.x_par + self.C @ self.xi <= self.c]
        self.node_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def solve_node_primal(self, lda: np.array):
        if self.node_prob is None:
            self.formulate_primal()
        self.lda.value = lda
        self.x_par.value = self.x
        self.node_prob.solve(solver=cvxpy.OSQP, warm_start=True)
        u = self.xi.value
        return u[:self.nu], np.asarray(self.F @ self.x + self.E @ u).ravel()


class network(object):

    def __init__(self, 
                 nodes: List[node],
                 incidence_matrix: np.array,
                 distance_bounds: np.array = None,
                 N: int = 2, 
                 ada_iterations: int = 100,
                 ada_eps: float = 1e-8,
                 ada_alpha: float = 0.9) -> None:
        self.nodes = nodes
        self.D = incidence_matrix
        self.N = N
        self.lb = ada_iterations
        self.epsilon = ada_eps
        self.alpha = ada_alpha
        for n in self.nodes:
            n.N = N

        self.M = len(self.nodes)
        self.nu = np.sum([n.nu for n in self.nodes])
        self.nx = np.sum([n.nx for n in self.nodes])
        
        self.xt = np.hstack([n.xt for n in self.nodes])
        self.A = la.block_diag(*[n.A for n in self.nodes])
        self.B = la.block_diag(*[n.B for n in self.nodes])
        self.R = la.block_diag(*[n.R for n in self.nodes])
        self.Q = la.block_diag(*[n.Q for n in self.nodes])
        self.P = la.block_diag(*[n.P for n in self.nodes])
        sel_mat = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0]])
        self.E_x = la.kron(self.D.T, sel_mat)
        self.E_u = np.zeros((np.shape(self.E_x)[0], self.nu))
        self.C_u = la.block_diag(*[n.C_u for n in self.nodes])
        self.c_u = np.hstack([n.c_u for n in self.nodes])
        if distance_bounds is not None:
            self.bb = np.kron([1, 1, 1, 1], distance_bounds).ravel()
        else:
            self.bb = np.ones(np.shape(self.E_x)[0])

        self.x_init = cvxpy.Parameter(self.nx)
        self.u_var = cvxpy.Variable((self.nu, N))
        self.lda_ada = self.lda_centr = None

        self.dense_agent_problem_matrices()
        
        self.centr_prob_def()
        self.centr_prob_def2()

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
    
    def centr_solve2(self) -> List[np.array]:
        for n in self.nodes:
            n.x_par_centr.value = n.x
        self.centr_prob2.solve(solver=cvxpy.OSQP, warm_start=True)
        u = []
        for n in self.nodes:
            u.append(n.z[0:n.nu].value)
        self.lda_centr = self.centr_prob2.constraints[-1].dual_value
        return u, self.lda_centr

    def ada_solve(self) -> List[np.array]:
        assert self.lb >= 1, 'ADA needs at least one iteration.'
        lda0 = np.zeros((self.N-1) * self.E_x.shape[0]) if self.lda_ada is None else self.lda_ada[:, -1]
        lda = np.zeros((lda0.shape[0], self.lb + 1))
        lda[:, 0] = lda0
        mu = lda0
        theta = 1.
        for l in range(self.lb):
            dlda = np.zeros(lda0.shape)
            u = []
            for n in self.nodes:
                u_, dlda_ = n.solve_node_primal(lda[:, l])
                dlda += dlda_
                u.append(u_)
            dlda += -n.b + self.epsilon * lda[:, l]
            mu_new = np.maximum(np.zeros(lda0.shape), lda[:, l] + self.alpha * dlda)
            theta_new = 0.5 * (1 + np.sqrt(1. + 4. * theta**2.))
            lda[:, l+1] = mu + (theta - 1) / theta_new * (mu_new - mu)
            theta = theta_new
            mu = mu_new
        
        self.lda_ada = lda

        return u, lda[:, -1]

    def centr_prob_def(self) -> None:
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
        
    def centr_prob_def2(self) -> None:
        objective = 0
        coupling_sum = 0
        constraints = []
        for n in self.nodes:
            n.set_up_cvx_vars()
            objective += cvxpy.quad_form(n.z, n.H)
            objective += 2 * n.x_par_centr.T @ n.G.T @ n.z
            objective += n.gxt @ n.z
            constraints += [n.D @ n.x_par_centr + n.C @ n.z <= n.c]
            coupling_sum += n.F @ n.x_par_centr + n.E @ n.z
        constraints += [coupling_sum <= self.nodes[0].b]
        
        self.centr_prob2 = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        
    def plot_convergence(self) -> None:
        lda_err = (self.lda_ada.T - self.lda_centr).T
        lda_err_norm = np.linalg.norm(lda_err, ord=1, axis=0)
        plt.figure(figsize=(6, 6))
        plt.plot(lda_err_norm)
        plt.yscale('log')
        plt.xlabel('Iterations $\ell$')
        plt.ylabel(r'$||\lambda_{\epsilon, l} - \lambda_{optimal}||_1$')
        plt.grid(which='both')

        if not os.path.exists(os.path.join(os.getcwd(), 'figures')):
            os.mkdir(os.path.join(os.getcwd(), 'figures'))
        plt.savefig(os.path.join(os.getcwd(), 'figures', 'lambda_convergence.pdf'))
        plt.close()

    def calcualte_metrics(self) -> float:
        constraint_violations = np.sum(np.maximum(self.E_x @ self.x - self.bb , 0))
        try:
            lda_dist = np.linalg.norm(self.lda_ada[:, -1] - self.lda_centr, ord=1)
        except TypeError:
            lda_dist = np.nan
        return constraint_violations, lda_dist
        

def simulate_trajectory(nw: network, 
                        nsim: int,
                        plot_convergence_at: int = None):
    use_centr_contr = True if nw.lb <= 0 else False

    trajectory = np.zeros((nw.nx, nsim + 1))
    trajectory[:, 0] = nw.x
    constraint_violations = np.zeros(nsim)
    lda_dist = np.zeros(nsim)
    for i in range(nsim):
        u_centr, _ = nw.centr_solve2()
        if use_centr_contr:
            nw.simulate_timestep(u_centr)
        else:
            u, _ = nw.ada_solve()
            nw.simulate_timestep(u)

        trajectory[:, i+1] = nw.x
        cv, ldad = nw.calcualte_metrics()
        if not use_centr_contr:
            constraint_violations[i] = cv
            lda_dist[i] = ldad

        if i == plot_convergence_at and not use_centr_contr:
            nw.plot_convergence()

    return trajectory, constraint_violations, lda_dist


def sweep_iterations(iterations: List[int], 
                     nodes: np.array,
                     incidence_matrix: np.array,
                     distance_bounds: np.array,
                     N: int = 8, nsim: int = 15, 
                     ada_eps: float = 1e-6, ada_alpha: float = 0.9):
    trajectories = []
    constraint_violations = []
    lda_dist = []
    plt.figure(figsize=(7.3,5))
    for iter in iterations:
        nw = network(nodes=copy.deepcopy(nodes), 
                     incidence_matrix=incidence_matrix, 
                     distance_bounds=distance_bounds,
                     N=N, 
                     ada_eps=ada_eps, 
                     ada_iterations=iter, 
                     ada_alpha=ada_alpha)
        trajectory, cv, ldad = simulate_trajectory(nw, nsim=nsim)
        trajectories.append(trajectory)
        constraint_violations.append(cv)
        lda_dist.append(ldad)

        if iter < 0:
            lbl = 'optimal'
        else:
            lbl = f'$\ell = {iter}$'
        plt.plot(cv, label=lbl)
    
    plt.legend()
    plt.xlabel('Time $t$')
    #plt.ylabel(r'$\sum_{m=0}^{p} max{0, E_{x,m} x}$')
    plt.ylabel('Constraint Violation')
    if not os.path.exists(os.path.join(os.getcwd(), 'figures')):
        os.mkdir(os.path.join(os.getcwd(), 'figures'))
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'constraint_violations.pdf'))
    plt.close()
    return trajectories, constraint_violations, lda_dist


def plot_trajectories(trajectories: List[np.array], 
                      iterations: List[int], 
                      nodes: np.array, 
                      nx: int = 4, 
                      selection: List[int] = None,
                      colors: List[str] = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']):
    assert len(trajectories) == len(iterations), \
        "The recorded trajectories and iterations must have the same length"
        
    nnodes = nodes.shape[0]
        
    if selection:
        trajectories = [trajectories[i] for i in selection]
        iterations = [iterations[i] for i in selection]

    for t, trajectory in enumerate(trajectories):
        ax = plt.gca()
        for i in range(nnodes):
            if i == 0:
                if iterations[t] < 0:
                    lbl = 'optimal'
                else:
                    lbl = f'$\ell = {iterations[t]}$'
            else:
                lbl = None

            if t == 0 and i == 0:
                start_label = 'start'
                target_label = 'target'
            else:
                start_label = None
                target_label = None
                
            if iterations[t] < 0:
                ax.scatter(trajectory[nx*i, 0], trajectory[nx*i+2, 0], 
                            marker='^', 
                            color=colors[i],
                            label=start_label)   
                ax.scatter(nodes[i].xt[0], nodes[i].xt[2], 
                            marker='v', 
                            color=colors[i],
                            label=target_label)

            ax.plot(trajectory[nx*i, :], trajectory[nx*i+2, :], 
                    marker='x',
                    linestyle='--', 
                    dashes=(2, t * 1),
                    label=lbl,
                    color=colors[i])
    
    plt.legend()
    leg = ax.get_legend()
    for hdl in leg.legendHandles[2:]:
        hdl.set_color('black')
    if not os.path.exists(os.path.join(os.getcwd(), 'figures')):
        os.mkdir(os.path.join(os.getcwd(), 'figures'))
    plt.savefig(os.path.join(os.getcwd(), 'figures', 'trajectories.pdf'))
    plt.close()
