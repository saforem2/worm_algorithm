import numpy as np
from operator import xor

def block_image(image, double_bonds_value=None):
    """Create blocked image from grayscale 'worm'-type image representing
    boundaries between black and white pixel regions. 
    
    Args:
        image (array-like):
            Flattened array of length 4L ** 2 ( = 2L * 2L ) of pixels
            representing the original image.
        double_bonds_value (int, default=None):
            Value assigned to 'double bonds' exiting a given block. If
            double_bonds_value is None, we use the 1 + 1 = 0 (approximate)
            blocking scheme, which preserves the closed path requirement in the
            blocked image.
    Returns:
        blocked_image (np.ndarray, shape=(L ** 2,)):
            Flattened image obtained after a single blocking step.
    """
    L = int(np.sqrt(len(image.flatten())) / 2)
    if image.shape != (2*L, 2*L):
        image = image.reshape(2*L, 2*L)
    blocked_image = np.zeros((L, L), dtype=int)
    blocked_sites = [(2*i, 2*j) for i in range(L//2) for j in range(L//2)]
    for site in blocked_sites:
        i = site[0]
        j = site[1]
        # look at the number of active external bonds leaving the block to the
        # right (ext_x_bonds) and upwards (ext_y_bonds)
        ext_x_bonds = [image[2*i, 2*j+3], image[2*i+2, 2*j+3]]
        ext_y_bonds = [image[2*i+3, 2*j], image[2*i+3, 2*j+2]]
        if double_bonds_value is None:
            ext_x_bonds_active = xor(ext_x_bonds[0], ext_x_bonds[1])
            ext_y_bonds_active = xor(ext_y_bonds[0], ext_y_bonds[1])
            active_site = ext_x_bonds_active or ext_y_bonds_active
        else:
            if ext_x_bonds == [1, 1]:
                ext_x_bonds_active = double_bonds_value
            if ext_y_bonds == [1, 1]:
                ext_y_bonds_active = double_bonds_value
            if ext_x_bonds_active or ext_y_bonds_active:
                active_site = double_bonds_value
        blocked_image[i, j] = active_site
        blocked_image[i, j+1] = ext_x_bonds_active
        blocked_image[i+1, j] = ext_y_bonds_active

    for site in blocked_sites:
        i = site[0]
        j = site[1]
        if blocked_image[i, j-1] or blocked_image[i-1, j]:
            blocked_image[site] = 1
    return blocked_image.flatten()

