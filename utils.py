import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy import eye


class node(object):

    def __init__(self, 
                 x0: list = [0., 0., 0., 0.],
                 xt: list = [0., 0., 0., 0.]) -> None:
        
        self.x0 = x0
        self.xt = xt
        self.A = sp.kron(eye(2), np.array([[1., 1.], [0., 1.]])).todense()
        self.B = sp.kron(eye(2), np.array([[0.], [1.]])).todense()
        
        self.nx, self.nu = np.shape(self.B)

        self.R = eye(self.nu) * 5
        self.Q = eye(self.nx)
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)


class network(object):

    def __init__(self) -> None:
        self.nodes = np.array([node(x0=[0., 0.5, 0., -0.5], xt=[1., 0.,  1.,  0.]), 
                               node(x0=[1.5, -0.5, 0., 0.], xt=[2.,   0.,  1.,  0.]), 
                               node(x0=[0.5, 0.2, 0.7, -0.2], xt=[1.5, 0., 1 + 1/np.sqrt(2), 0.])])
        self.D = np.array([[1, 0, 1], [-1, 1, 0], [0, -1, -1]])
        
        self.M = len(self.nodes)
        self.nu = np.sum([n.nu for n in self.nodes])
        self.nx = np.sum([n.nx for n in self.nodes])
        
        self.x0 = np.hstack([n.x0 for n in self.nodes])
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
    