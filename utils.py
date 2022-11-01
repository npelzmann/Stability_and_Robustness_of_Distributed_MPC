import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy import eye


class node(object):

    def __init__(self) -> None:
        self.A = sp.kron(eye(2), eye(2,2,1)).todense()
        self.B = sp.kron(eye(2), np.array([[1.], [0.]])).todense()
        
        self.nx, self.nu = np.shape(self.B)

        self.R = eye(self.nu)
        self.Q = eye(self.nx)
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)


class network(object):

    def __init__(self) -> None:
        self.nodes = np.array([node(), node(), node()])
        self.D = np.array([[1, 0, 1], [-1, 1, 0], [0, -1, -1]])
        
        nu = A = np.sum(*list(n.nu for n in self.nodes))
        nx = A = np.sum(*list(n.nx for n in self.nodes))

        A = la.block_diag(*list(n.A for n in self.nodes))
        B = la.block_diag(*list(n.B for n in self.nodes))
        R = la.block_diag(*list(n.R for n in self.nodes))
    
        sel_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.E_x= la.kron(self.D, sel_mat)
        self.E_u = np.zeros((np.shape(self.E_x)[0], self.nu))
        self.b = np.ones(np.shape(self.E_x)[0])
    