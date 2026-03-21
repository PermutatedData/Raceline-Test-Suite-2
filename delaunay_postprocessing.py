import numpy as np

from scipy.spatial import Delaunay

import helpers

MIN_SPACING = 1
MAX_SPACING = 20

tri = None
labels = None
points = None
midpoints = []

def create_delaunay(left_points, right_points):
    global tri, labels, points
    
    # Combine points
    points = np.vstack((left_points, right_points))
    
    labels = np.hstack((
        np.zeros(len(left_points)),      # 0 = left
        np.ones(len(right_points))       # 1 = right
    ))

    # Delaunay triangulation
    tri = Delaunay(points)

def simplices():
    return tri.simplices

def get_points():
    return points

def prostprocess():
    global midpoints
    """
    Process Delaunay triangulation

    Returns
    -------
    midline : (K,2) array
    simplices 
    """
    
    simplices_filtered = [] # I could use simplex indices instead. Interesting
    used_edges = set()
    
    for simplex in tri.simplices:
        triangle_included = False
        
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]

            edge = tuple(sorted((a, b)))
            
            if edge in used_edges:
                continue
            
            used_edges.add(edge)

            # Only keep edges connecting left-right
            if labels[a] == labels[b]:
                continue
            
            # Potential checks: intersections, edges aligned, isolated midpoints?
            
            pa = points[a]
            pb = points[b]
            
            width = np.linalg.norm(pa - pb)

            if MIN_SPACING <= width and width <= MAX_SPACING:
                midpoint = (pa + pb) / 2.0
                midpoints.append(midpoint)

                triangle_included = True

        if triangle_included:
            simplices_filtered.append(simplex)
        
    midpoints = np.array(midpoints)

    if len(midpoints) < 3:
        raise RuntimeError("Not enough midpoints found.")

    return np.array(simplices_filtered)

def greedy_intersection_removal(simplices):
    """
    Assumes longer edge is the bad one
    """

# Repeated assembly and use of kdTree to find nearest neighbor is more efficient for large scale but not here
def ordered_midpoint(start_pos):
    global midpoints
    """
    start_pos: starting index
    
    returns:
        path: ordered points
    """
    
    ordered = []
    
    n = len(midpoints)
    visited_indices = [] # Rather than constantly appending to list, a fixed array of correct size could be used. Interesting

    current = start_pos

    while len(visited_indices) < n:
        dist = np.linalg.norm(midpoints - current, axis=1)
        dist[visited_indices] = np.inf
        
        nearest_index = np.argmin(dist)
        nearest = midpoints[nearest_index]

        visited_indices.append(nearest_index)
        ordered.append(nearest)
        
        current = nearest

    return np.array(ordered)