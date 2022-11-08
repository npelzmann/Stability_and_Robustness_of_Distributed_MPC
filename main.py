import numpy as np
import matplotlib.pyplot as plt

from utils import network, node

# # just three nodes
# nodes = np.array([node(x0=[1.5, -0.3, 0., 0.], xt=[1.,   0.,  1.,  0.]),
#                   node(x0=[0.5, -0.3, 0., -0.4], xt=[2., 0.,  1.,  0.]),  
#                   node(x0=[0.5,  0.2, 0.7, -0.2], xt=[1.5, 0., 2., 0.])])
# D = np.array([[1, 0, 1], 
#               [-1, 1, 0], 
#               [0, -1, -1]])

X0 = np.array([[0.0, -0.4, 0.,   0.4],
               [1.0,  0.4, 0.5,  0.4],
               [0.5,  0.3, 1.0, -0.3],
               [1.5, -0.4, 1.5,  0.4],
               [3.0,  0.2, 0.0, -0.2]])
# rotate and shift
phi_rot = 30.   # in degrees
c, s = np.cos(phi_rot), np.sin(phi_rot)
R = np.array(((c, -s), (s, c)))
Xt = np.zeros(X0.shape)
Xt[:,[0,2]] = (R @ (X0[:,[0,2]]).T).T + np.array([1, 1])

nodes = np.array([node(x0=X0[0,:], xt=Xt[0,:]),
                  node(x0=X0[1,:], xt=Xt[1,:]),
                  node(x0=X0[2,:], xt=Xt[2,:]),
                  node(x0=X0[3,:], xt=Xt[3,:]),
                  node(x0=X0[4,:], xt=Xt[4,:])])
incidence_matrix = np.array([[ 1,  1,  0,  0,  0,  0,  1], 
                             [-1,  0,  1,  1,  0,  0,  0],
                             [ 0, -1, -1,  0,  1,  0,  0],
                             [ 0,  0,  0, -1, -1,  1,  0], 
                             [ 0,  0,  0,  0,  0, -1, -1]])
# d_max
distance_bounds = np.array([[2.0],
                            [2.0],
                            [2.0],
                            [2.0],
                            [2.0],
                            [4.0],
                            [4.5]])

nw = network(nodes=nodes, incidence_matrix=incidence_matrix, distance_bounds=distance_bounds,
             N=8, ada_eps=1e-3, ada_iterations=10, ada_alpha=0.9)

# Simulate in closed loop
nsim = 15
trajectory = np.zeros((nw.nx, nsim + 1))
trajectory[:, 0] = nw.x
control_inputs = np.zeros((nw.nu, nsim))
for i in range(nsim):
    # u_centr = nw.centr_solve()
    u_centr2, lda_centr2 = nw.centr_solve2()
    print(nw.centr_prob2.status)
    print('lambda norm: ', np.linalg.norm(lda_centr2, ord=1))
    u, lda = nw.ada_solve()
    nw.simulate_timestep(u_centr2)
    trajectory[:, i+1] = nw.x
    control_inputs[:, i] = np.hstack(u_centr2)
    # if i == 0:
    #     nw.plot_convergence()

for i in range(nw.M):
    plt.plot(trajectory[4*i, :], trajectory[4*i+2, :], marker='x')

plt.show()
print('success')