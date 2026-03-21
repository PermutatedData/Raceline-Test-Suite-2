import numpy as np

from scipy.spatial import Delaunay

import helpers

MIN_SPACING = 1
MAX_SPACING = 25

tri = None
labels = None
points = None

# TODO: figure out what constraints a Delaunay triangulation imposes
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
    """
    Process Delaunay triangulation

    Returns
    -------
    midline : (K,2) array
    simplices 
    """
    
    simplices_filtered = [] # I could use simplex indices instead. Interesting
    bad_edges = set()

    for simplex in tri.simplices:
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]

            edge = tuple(sorted((a, b)))
            
            if edge in bad_edges:
                continue
            
            # Only keep edges connecting left-right
            if labels[a] == labels[b]:
                bad_edges.add(edge)
                continue
            
            pa = points[a]
            pb = points[b]
            
            width = np.linalg.norm(pa - pb)

            # Potential checks: intersections, edges aligned, isolated midpoints?
            # Consider: skinniness score. Score added to all edges; interior edges get contributions from up to 2 triangles

            if MIN_SPACING > width or width > MAX_SPACING:
                bad_edges.add(edge)
                break

            # Triangle is valid if just one edge is valid. Requiring two is an option
            simplices_filtered.append(simplex)
            break

    # Is this even necessary?
    if len(simplices_filtered) < 3:
        raise RuntimeError("Not enough midpoints found.")

    return np.array(simplices_filtered)

# By nature of Delaunay triangulations, this likely does nothing
def greedy_intersection_removal(simplices):
    """
    Assumes longer edge is the bad one
    Destroys triangulation and must be plotted separately
    
    Args:
        simplices(list): simplices of triangles
    """
    selected = []

    e1 = simplices[:, [0, 1]]
    e2 = simplices[:, [1, 2]]
    e3 = simplices[:, [2, 0]]
    
    # Directly derived from simplices. Gotta remove the boundaries
    edges_raw = np.vstack([e1, e2, e3])
    
    edges = edges_raw[labels[edges_raw].sum(axis=1) == 1]
    
    # Duplicate edges (a, b) and (b, a) won't be detected without this. Also just more consistent
    edges = np.sort(edges, axis=1)
    
    # remove duplicates
    edges = np.unique(edges, axis=0)

    lengths = {
        tuple(e): np.linalg.norm(points[e[0]] - points[e[1]]) for e in edges
    }

    for e in sorted(edges, key=lambda e: lengths[tuple(e)]):
        p1_index, p2_index = e
        
        intersects_existing = False
        for e2 in selected:
            q1_index, q2_index = e2
            
            if p1_index == q1_index or p1_index == q2_index or p2_index == q1_index or p2_index == q2_index:
                continue
            
            if helpers.segments_intersect(points[p1_index], points[p2_index], points[q1_index], points[q2_index]):
                intersects_existing = True
                break
        
        if not intersects_existing:
            selected.append(e)
    
    return np.array(selected)

def indices_to_points(indices):
    return points[indices]

# Repeated assembly and use of kdTree to find nearest neighbor is more efficient for large scale but not here
def ordered_midpoint_from_edge_indices(start_pos, edges):
    """
    start_pos: starting index
    
    returns:
        path: ordered points
    """
    
    ordered = []
    
    n = len(edges)
    visited_indices = [] # Rather than constantly appending to list, a fixed array of correct size could be used. Interesting

    edge_points = indices_to_points(edges) # Numpy magic. Works with not just list of indices, but list of edge indices
    midpoints = (edge_points[:,0] + edge_points[:,1]) / 2

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