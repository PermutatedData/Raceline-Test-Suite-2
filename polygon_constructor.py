import numpy as np
import math

TOLERANCE = 1E-5

# TODO: projection sorting, something more advanced if necessary. Centroid angle sorting likely not good enough for FSAE?

def score_cone(current, heading, cone, max_dist, weight_dist, weight_angle, min_spacing, max_spacing, max_search_angle_dot):
    """_summary_

    Args:
        current (_type_): _description_
        heading (_type_): _description_
        cone (_type_): _description_
        weight_dist (_type_): _description_
        weight_angle (_type_): _description_
        min_spacing (_type_): _description_
        max_spacing (_type_): _description_
        max_search_angle (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if not np.isclose(np.linalg.norm(heading), 1.0, atol=TOLERANCE):
        raise ValueError(heading, "non-unit vector")
    
    dist_vec = cone - current
    dist = np.linalg.norm(dist_vec)
    
    if dist < min_spacing or dist > max_spacing:
        return np.inf
    
    dot = np.dot(heading, dist_vec / dist)
    
    if dot <= max_search_angle_dot:
        return np.inf
    
    return weight_dist * (dist / max_dist) + weight_angle * (1 - dot)

# In reality, car pos should be 0
# In particularly bad cases, ordering will fail rather than try a bs solution
def order_boundary_weighted(cones: np.ndarray, car_pos, car_heading=0, weight_dist=0.6, weight_angle=0.4, min_spacing=0.5, max_spacing=6, max_search_angle=70) -> np.ndarray:
    """
    Nearest neighbor walk

    Parameters
    ----------
    points : (x, y) Numpy array

    Returns
    -------
    sorted points : (x, y) Numpy array
    """

    n = len(cones)
    
    if n < 2:
        return cones

    visited = set()
    ordered = []

    cos_limit = np.cos(np.deg2rad(max_search_angle))

    # start: closest cone to car
    dists = np.linalg.norm(cones - car_pos, axis=1)
    current = np.argmin(dists)

    ordered.append(cones[current])
    visited.add(current)

    prev = None

    while len(visited) < n:
        # Fix
        if prev is None:
            direction = np.array((np.cos(np.deg2rad(car_heading)), np.sin(np.deg2rad(car_heading))))
            
        else:
            # v1 = cones[prev] - cones[prev2]
            # v2 = cones[current] - cones[prev]

            # v1 = v1 / np.linalg.norm(v1)
            # v2 = v2 / np.linalg.norm(v2)

            # direction = 0.4 * v1 + 0.6 * v2
            
            direction = cones[current] - cones[prev]
        
        print(cones[current], len(visited))
        
        norm = np.linalg.norm(direction)

        direction_normalized = direction / norm

        best_idx = -1
        best_score = np.inf

        # compute max candidate distance for normalization
        max_dist = 0
        for idx in range(n):
            if idx not in visited:
                d = np.linalg.norm(cones[idx] - cones[current])
                max_dist = max(max_dist, d)

        for idx in range(n):
            if idx in visited:
                continue

            score = score_cone(cones[current], direction_normalized, cones[idx], max_dist, weight_dist=weight_dist, weight_angle=weight_angle, min_spacing=min_spacing, max_spacing=max_spacing, max_search_angle_dot=cos_limit)

            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            print('break 2')
            break

        ordered.append(cones[best_idx])
        visited.add(best_idx)

        prev = current
        current = best_idx

    return np.array(ordered)
