import numpy as np

from utils import network, node
import utils

if __name__ == '__main__':
    # just four nodes
    nodes = np.array([node(x0=[1.4, -0.3, 0., 0.],    xt=[1.,   0.0,  1.,  0.], v_max=0.1),
                    node(x0=[0.6, -0.1, 0.,  -0.4],   xt=[1.5,  0.0,  1.,  0.], v_max=0.5),  
                    node(x0=[0.5,  0.1, 0.7, -0.1],   xt=[1.,   0.0,  1.5, 0.], v_max=0.1),
                    node(x0=[0.5, -0.2, 0.3,  0.2],   xt=[0.7,  0.0,  0.7, 0.], v_max=1.0)
                    ])
    incidence_matrix = np.array([[ 1,  0,  1,  1,  0,  0], 
                                [-1,  1,  0,  0,  0,  0], 
                                [ 0, -1, -1,  0,  1,  0],
                                [ 0,  0,  0, -1, -1, -1]])
    distance_bounds = np.array([[1.], [1.], [1.], [1.], [1.], [1.]])

    # nw = network(nodes=nodes, incidence_matrix=incidence_matrix, distance_bounds=distance_bounds,
    #              N=8, ada_eps=1e-6, ada_iterations=10, ada_alpha=0.95)

    # utils.simulate_trajectory(nw, nsim=15, plot_convergence_at=0)

    iterations = [-1, 1, 10, 2500]
    trajectories, _, _ = utils.sweep_iterations(iterations, nodes, incidence_matrix, distance_bounds, nsim=15)
    utils.plot_trajectories(trajectories, iterations, nodes)

    print('success')