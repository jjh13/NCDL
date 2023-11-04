def visualize_lattice(lt, channel=-1):
    """
    Visualizes a lattice tensor.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from itertools import product

    fig = plt.figure()
    
    if lt.parent.dimension == 2:
        ax = fig.add_subplot()
    elif lt.parent.dimension == 3:
        ax = fig.add_subplot(projection='3d')
    
    for idx in range(lt.parent.coset_count):    
        region = lt.coset(idx).shape[2:]
        shift = lt.coset_vectors[idx]
        pts = [(np.array(__, dtype='int')*lt.parent.coset_scale + shift).tolist() for __ in  product(*[range(0, _) for _ in region])]
        
        components = list(zip(*pts))

        if len(components) == 2:
            if idx != 0:
                ax.quiver(shift[0:1], shift[1:])
            ax.scatter(components[0], components[1], marker='o')

            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')

        if len(components) == 3:
            
            if idx != 0:
                ax.plot([0, shift[0]], [0, shift[1]], [0, shift[2]])
                
            ax.scatter(components[0], components[1], components[2], marker='o')

            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')

    plt.show()
