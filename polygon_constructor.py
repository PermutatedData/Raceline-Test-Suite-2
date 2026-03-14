import numpy as np
import math

TOLERANCE = 1E-5

def score_cone(current, heading, cone, max_dist, weight_angle, min_spacing, max_spacing, max_search_angle_dot) -> float: 
    """
    Returns weighted sum between two positions factoring change in heading and spacing
    Maximum possible value is 1 + weight_angle
    Has cutoffs

    Args:
        current (_type_): current position
        heading (_type_): heading vector (must be normalized)
        cone (_type_): next position (cone)
        weight_angle: weight applied to angle compared to distance (1)
        min_spacing (_type_): min spacing between cones
        max_spacing (_type_): max spacing between cones
        max_search_angle (_type_): max change in heading

    Returns:
        float: score
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
    
    return dist / max_dist + weight_angle * (1 - dot)


# Typical values
# weight_angle / weight_dist = 0.67-ish
# weight_future_look = 0.4–0.7

# In reality, car pos should be 0
# In particularly bad cases, ordering will fail rather than try a bs solution
# TODO: ensure initial set value of future_score is not cooked

def order_boundary_weighted(cones: np.ndarray, car_pos, car_heading=0, weight_angle=0.66, weight_future_look=0.5, min_spacing=0.5, max_spacing=6, max_search_angle=70) -> np.ndarray:
    """
    Returns weighted sum between two positions factoring change in heading and spacing
    Has cutoffs

    Args:
        cones (_type_): current position
        car_pos (_type_): heading vector (must be normalized)
        car_heading (_type_): next position (cone)
        weight_angle (_type_): weight applied to angle compared to distance (1)
        weight_future_look (_type_): weight applied to future prediction compared to current
        min_spacing (_type_): min spacing between cones
        max_spacing (_type_): max spacing between cones
        max_search_angle (_type_): max change in heading

    Returns:
        float: score
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

    direction = np.array((np.cos(np.deg2rad(car_heading)), np.sin(np.deg2rad(car_heading))))
    direction_normalized = direction / np.linalg.norm(direction)

    while len(visited) < n:
        # print(cones[current], len(visited))

        best_idx = -1
        best_score = np.inf

        max_dist = 0
        for idx in range(n):
            if idx not in visited:
                d = np.linalg.norm(cones[idx] - cones[current])
                max_dist = max(max_dist, d)

        for idx in range(n):
            if idx in visited:
                continue

            score = score_cone(cones[current], direction_normalized, cones[idx], max_dist, weight_angle=weight_angle, min_spacing=min_spacing, max_spacing=max_spacing, max_search_angle_dot=cos_limit)
            
            # print(score)
            
            # Should ensure this is greater than what's feasible
            future_score = 1 + weight_angle + 1

            if score == np.inf:
                continue
            
            # Lookahead step
            for idx2 in range(n):
                if idx2 in visited or idx2 == idx:
                    continue
                
                score2 = score_cone(cones[idx], direction_normalized, cones[idx2], max_dist, weight_angle=weight_angle, min_spacing=min_spacing, max_spacing=max_spacing, max_search_angle_dot=cos_limit)
                
                if score2 < future_score:
                    future_score = score2
            
            combined_score = score + weight_future_look * future_score
              
            if combined_score < best_score:
                    best_score = combined_score
                    best_idx = idx

        if best_idx == -1:
            print("No new cones within bounds. Cone sorting stopped early")
            break

        ordered.append(cones[best_idx])
        visited.add(best_idx)

        direction = cones[best_idx] - cones[current]
        direction_normalized = direction / np.linalg.norm(direction)
        
        current = best_idx

    return np.array(ordered)


def get_good_polygon(left_cones: np.ndarray, right_cones: np.ndarray) -> np.ndarray:
    """
    Using the shorter of the two cone lists, finds the closest point of the other list and connects to it to create vertices of a CCW polygon
    Assumes cones are sorted

    Args:
        left_cones (np.ndarray): _description_
        right_cones (np.ndarray): _description_
    """
    
    last_left_cone = left_cones[-1]
    dists = np.linalg.norm(right_cones - last_left_cone, axis=1)
    index_right = np.argmin(dists)
    
    last_right_cone = right_cones[-1]
    dists2 = np.linalg.norm(left_cones - last_right_cone, axis=1)
    index_left = np.argmin(dists2)
        
    min_on_left = min(dists[index_right], dists2[index_left]) == dists[index_left]
    min_dist = 0
    
    if min_on_left:
        min_dist = dists[index_left]

        right_cones = right_cones[:index_right + 1]
        
    else:
        min_dist = dists2[index_right]
        
        left_cones = left_cones[:index_left + 1]
        
    # TODO: separation check with min_dist
    
    return np.vstack((right_cones, left_cones[::-1]))


def polygon_pipeline(left_cones: np.ndarray, right_cones: np.ndarray, car_pos, car_heading=0, weight_angle=0.66, weight_future_look=0.5, min_spacing=0.5, max_spacing=6, max_search_angle=70) -> np.ndarray:
    left_cones_ordered = left_cones
    right_cones_ordered = right_cones
    
    if len(left_cones_ordered) < 2 and len(right_cones_ordered) < 2:
        raise ValueError("Too few cones")
    
    return get_good_polygon(left_cones_ordered, right_cones_ordered)