# ERROR: allows lines to cross edges. Not sure why

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


def midline(left_points, right_points, 
                    max_width=None,
                    smooth=False,
                    spline_s=0.0,
                    num_output_points=500):
    """
    Compute racetrack midline using Delaunay triangulation.

    Parameters
    ----------
    left_points : (N,2) array
    right_points : (M,2) array
    max_width : float or None
        Optional maximum allowed track width
    smooth : bool
        Whether to spline smooth the result
    spline_s : float
        Spline smoothing factor
    num_output_points : int
        Number of output midline samples

    Returns
    -------
    midline : (K,2) array
    """

    # Combine points
    points = np.vstack((left_points, right_points))
    labels = np.hstack((
        np.zeros(len(left_points)),      # 0 = left
        np.ones(len(right_points))       # 1 = right
    ))

    # Delaunay triangulation
    tri = Delaunay(points)

    # Extract cross-boundary edges
    midpoints = []
    triangles = []
    used_edges = set()
    
    for simplex in tri.simplices:
        triangle_edges = []
        
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]

            edge = tuple(sorted((a, b)))
            if edge in used_edges:
                continue
            used_edges.add(edge)

            # Only keep edges connecting left-right
            if labels[a] != labels[b]:
                pa = points[a]
                pb = points[b]
                length = np.linalg.norm(pa - pb)

                if max_width is None or length <= max_width:
                    midpoint = (pa + pb) / 2.0
                    midpoints.append(midpoint)
                    
                    triangle_edges.append([pa, pb])

        # Triangles. I have no idea how the fuck that worked
        if len(triangle_edges) != 0:
            triangles.append(points[simplex])
        
    midpoints = np.array(midpoints)

    if len(midpoints) < 3:
        raise RuntimeError("Not enough midpoints found.")

    # Order midpoints using nearest neighbor chaining
    ordered = order_points(midpoints)

    # Close loop
    # ordered = np.vstack([ordered, ordered[0]])

    return ordered, np.array(triangles)

# Repeatedly assembles and uses kdTree to find nearest neighbor
# More efficient for large scale but not here
def order_points(points):
    """
    Order points into continuous loop using greedy nearest neighbor.
    """
    points = points.copy()
    ordered = [points[0]]
    remaining = points[1:]

    tree = cKDTree(remaining)

    while len(remaining) > 0:
        last = ordered[-1]
        dist, idx = tree.query(last)
        next_point = remaining[idx]

        ordered.append(next_point)
        remaining = np.delete(remaining, idx, axis=0)

        if len(remaining) > 0:
            tree = cKDTree(remaining)

    return np.array(ordered)