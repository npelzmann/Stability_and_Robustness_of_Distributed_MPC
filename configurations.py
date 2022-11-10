# # configuration with 5 nodes
# X0 = np.array([[0.0, -0.4, 0.,   0.4],
#                [1.0,  0.4, 0.5,  0.4],
#                [0.5,  0.3, 1.0, -0.3],
#                [1.5, -0.4, 1.5,  0.4],
#                [3.0,  0.2, 0.0, -0.2]])
# # rotate and shift
# phi_rot = 30.   # in degrees
# c, s = np.cos(phi_rot), np.sin(phi_rot)
# R = np.array(((c, -s), (s, c)))
# Xt = np.zeros(X0.shape)
# Xt[:,[0,2]] = (R @ (X0[:,[0,2]]).T).T + np.array([1, 1])

# nodes = np.array([node(x0=X0[0,:], xt=Xt[0,:]),
#                   node(x0=X0[1,:], xt=Xt[1,:]),
#                   node(x0=X0[2,:], xt=Xt[2,:]),
#                   node(x0=X0[3,:], xt=Xt[3,:]),
#                   node(x0=X0[4,:], xt=Xt[4,:])])
# incidence_matrix = np.array([[ 1,  1,  0,  0,  0,  0,  1], 
#                              [-1,  0,  1,  1,  0,  0,  0],
#                              [ 0, -1, -1,  0,  1,  0,  0],
#                              [ 0,  0,  0, -1, -1,  1,  0], 
#                              [ 0,  0,  0,  0,  0, -1, -1]])
# # d_max
# distance_bounds = np.array([[2.0],
#                             [2.0],
#                             [2.0],
#                             [2.0],
#                             [2.0],
#                             [4.0],
#                             [4.5]])