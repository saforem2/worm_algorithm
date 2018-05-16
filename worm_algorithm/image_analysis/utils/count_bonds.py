import numpy as np 

def count_bonds(image_set):
    """Method for counting 'bonds' in a set of worm-type images."""
    w = image_set[0].shape[0]
    bond_idxs = [(i, j) for i in range(w)
                 for j in range(w)
                 if (i + j) % 2 == 1]
    bc_arr = np.array([np.sum([image[i] for i in bond_idxs])
                       for image in image_set])
    bc2_arr = bc_arr ** 2
    Nb_avg = np.mean(bc_arr)
    Nb2_avg = np.mean(bc2_arr)
    Nb_avg2 = Nb_avg ** 2
    delta_Nb = Nb2_avg - Nb_avg2
    return Nb_avg, delta_Nb
