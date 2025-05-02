import os
import numpy as np
from matplotlib.testing.compare import compare_images


def are_equivalent_ss(ss_1, ss_2, atol=1e-08):
    assert ss_1.shape == ss_2.shape

    is_reachable = np.tri(*ss_1.shape)

    for i in range(1, ss_1.shape[0]):
        for j in range(0, i + 1):
            if j == 0:
                parent_js = [0]
            else:
                parent_js = [j - 1, j]
            
            parent_thetas = ss_1[i - 1, parent_js]
            parents_are_reachable = is_reachable[i - 1, parent_js]
            parents_are_passable = (parent_thetas < 1 - atol) * parents_are_reachable
            
            if not parents_are_passable.any():
                is_reachable[i, j] = 0
    
    return np.allclose(
        ss_1 * is_reachable,
        ss_2 * is_reachable,
        atol=atol
    )


def compare_images_with_cleanup(expected_path, actual_path, tol):
    garbage_paths = []

    for path in [expected_path, actual_path]:
        path_without_ext, ext = os.path.splitext(path)
        if ext != ".png":
            garbage_paths.append(
                path_without_ext + "_" + ext[1:] + ".png"
            )

    try:
        return compare_images(expected_path, actual_path, tol)
    finally:
        for garbage_path in garbage_paths:
            if os.path.exists(garbage_path):
                os.remove(garbage_path)
