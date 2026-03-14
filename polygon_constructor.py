import numpy as np
import math

TOLERANCE = 1E-5

# To use: 
# WEIGHT_ANGLE = 0.66
# WEIGHT_FUTURE_LOOK = 0.5
# MIN_SPACING = 0.5
# MAX_SPACING = 6
# MAX_SEARCH_ANGLE = 70

WEIGHT_ANGLE = 0.66
WEIGHT_FUTURE_LOOK = 0.5
MIN_SPACING = 0.5
MAX_SPACING = 20
MAX_SEARCH_ANGLE = 70

def score_cone(current, heading, cone, max_search_angle_dot) -> float: 
    """
    Returns weighted sum between two positions factoring change in heading and spacing
    Maximum possible value is 1 + WEIGHT_ANGLE
    Has cutoffs

    Args:
        current (_type_): current position
        heading (_type_): heading vector (must be normalized)
        cone (_type_): next position (cone)
        max_dist (_type_): maximum distance from current cone to any cone
        max_search_angle_dot (_type_): dot of MAX_SEARCH_ANGLE

    Returns:
        float: score
    """
    
    if not np.isclose(np.linalg.norm(heading), 1.0, atol=TOLERANCE):
        raise ValueError(heading, "non-unit vector")
    
    dist_vec = cone - current
    dist = np.linalg.norm(dist_vec)
    
    if dist < MIN_SPACING or dist > MAX_SPACING:
        return np.inf
    
    dot = np.dot(heading, dist_vec / dist)
    
    # print(dist_vec, heading)
    
    if dot <= max_search_angle_dot:
        return np.inf
    
    # print(dist / MAX_SPACING + WEIGHT_ANGLE * (1 - dot), 1 - dot)
    
    return dist / MAX_SPACING + WEIGHT_ANGLE * (1 - dot)


# Typical values
# weight_angle / weight_dist = 0.67-ish
# weight_future_look = 0.4–0.7

# In reality, car pos should be 0
# In particularly bad cases, ordering will fail rather than try a bs solution
# TODO: ensure initial set value of future_score is not cooked

def order_boundary_weighted(cones: np.ndarray, car_pos, car_heading_vector) -> np.ndarray:
    """
    Returns weighted sum between two positions factoring change in heading and spacing
    Has cutoffs

    Args:
        cones (_type_): current position
        car_pos (_type_): next position (cone)
        car_heading (_type_): heading vector (must be normalized)
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

    cos_limit = np.cos(np.deg2rad(MAX_SEARCH_ANGLE))

    # start: closest cone to car
    dists = np.linalg.norm(cones - car_pos, axis=1)
    current = np.argmin(dists)

    ordered.append(cones[current])
    visited.add(current)

    direction_normalized = car_heading_vector

    while len(visited) < n:
        # print()
        # print(cones[current], len(visited))

        best_idx = -1
        best_score = np.inf

        for idx in range(n):
            if idx in visited:
                continue

            # print("Current score: ", cones[idx])
            
            score = score_cone(cones[current], direction_normalized, cones[idx], cos_limit)
            
            if score == np.inf:
                continue
            
            # Should ensure this is greater than what's feasible
            future_score = 1 + WEIGHT_ANGLE + 1
            
            # Lookahead step
            for idx2 in range(n):
                if idx2 in visited or idx2 == idx:
                    continue
                
                # print("Future score: ", cones[idx2])
                
                direction_future = cones[idx2] - cones[idx]
                score2 = score_cone(cones[idx], direction_future / np.linalg.norm(direction_future), cones[idx2], cos_limit)
                
                if score2 < future_score:
                    future_score = score2
            
            combined_score = score + WEIGHT_FUTURE_LOOK * future_score
            
            # print(cones[idx], score, future_score, combined_score)
            # print()
            
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
    dists_from_right = np.linalg.norm(right_cones - last_left_cone, axis=1)
    index_right = np.argmin(dists_from_right)
    
    last_right_cone = right_cones[-1]
    dists_from_left = np.linalg.norm(left_cones - last_right_cone, axis=1)
    index_left = np.argmin(dists_from_left)
        
    min_on_left = min(dists_from_right[index_right], dists_from_left[index_left]) == dists_from_right[index_right]
    min_dist = 0

    if min_on_left:
        min_dist = dists_from_right[index_right]

        right_cones = right_cones[:index_right + 1]
        
    else:
        min_dist = dists_from_left[index_left]
        
        left_cones = left_cones[:index_left + 1]
        
    # TODO: separation check with min_dist
    
    return np.vstack((right_cones, left_cones[::-1]))


def polygon_pipeline(left_cones: np.ndarray, right_cones: np.ndarray, car_pos, car_heading_vector) -> np.ndarray:
    left_cones_ordered = order_boundary_weighted(left_cones, car_pos, car_heading_vector)
    
    print("\n")
    
    right_cones_ordered = order_boundary_weighted(right_cones, car_pos, car_heading_vector)
    
    if len(left_cones_ordered) < 2 and len(right_cones_ordered) < 2:
        raise ValueError("Too few cones")
    
    return get_good_polygon(left_cones_ordered, right_cones_ordered)