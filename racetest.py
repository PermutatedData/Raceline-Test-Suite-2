import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import triangle

from scipy.interpolate import CubicSpline

INPUT_FILE_NAME = "test_track.csv" 

# def insert function here lmao

def find_midline(left_x, left_y, right_x, right_y, num_midpoints=None):
    """
    Given left and right boundary points, returns midline points by averaging.
    Handles different number of points by resampling via interpolation.

    Args:
        left_x, left_y: np.arrays of left boundary points
        right_x, right_y: np.arrays of right boundary points
        num_midpoints: number of points in the midline (default=max of left/right points)

    Returns:
        mid_x, mid_y: np.arrays of midline points
    """
    if num_midpoints is None:
        num_midpoints = max(len(left_x), len(right_x))

    # Parameterize boundaries from 0 to 1 by cumulative distance
    def parameterize_curve(x, y):
        dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumdist = np.insert(np.cumsum(dist), 0, 0)
        return cumdist / cumdist[-1]

    t_left = parameterize_curve(left_x, left_y)
    t_right = parameterize_curve(right_x, right_y)

    # Interpolate left boundary to uniform t
    t_uniform = np.linspace(0, 1, num_midpoints)
    left_x_interp = np.interp(t_uniform, t_left, left_x)
    left_y_interp = np.interp(t_uniform, t_left, left_y)

    # Interpolate right boundary to uniform t
    right_x_interp = np.interp(t_uniform, t_right, right_x)
    right_y_interp = np.interp(t_uniform, t_right, right_y)

    # Average to get midline
    mid_x = (left_x_interp + right_x_interp) / 2
    mid_y = (left_y_interp + right_y_interp) / 2

    return mid_x, mid_y

# 2-large sliding window across left cones that finds the nearest right cone to the first left cone and turns that into a triangle
# Some right cones are never connected
# Trash, not super efficient, but who cares lol
def basic_triangulation(left_x, left_y, right_x, right_y,):
    test_x = []
    test_y = []
    
    triangles = []
    
    for i, _ in enumerate(left_x[2:]):
        nearest_right = [1000, 1000]
        
        for j, element in enumerate(right_x):
            if (left_x[i] - right_x[j]) ** 2 + (left_y[i] - right_y[j]) ** 2 < (left_x[i] - nearest_right[0]) ** 2 + (left_y[i] - nearest_right[1]) ** 2:
                nearest_right = [right_x[j], right_y[j]]
        
        triangles.append([[left_x[i], left_y[i]], [left_x[i + 1], left_y[i + 1]], nearest_right])
    
        # Midpoint of first leg
        test_x.append((left_x[i] + nearest_right[0]) / 2)
        test_y.append((left_y[i] + nearest_right[1]) / 2)
        
        # Midpoint of second leg
        test_x.append((left_x[i + 1] + nearest_right[0]) / 2)
        test_y.append((left_y[i + 1] + nearest_right[1]) / 2)
            
    return np.array(test_x), np.array(test_y), np.array(triangles)

def basic_triangulation_complete(left_x, left_y, right_x, right_y):
    test_x = []
    test_y = []
    
    triangles = []
    
    for i, _ in enumerate(left_x[2:]):
        nearest_right = [1000, 1000]
        
        for j, element in enumerate(right_x):
            if (left_x[i] - right_x[j]) ** 2 + (left_y[i] - right_y[j]) ** 2 < (left_x[i] - nearest_right[0]) ** 2 + (left_y[i] - nearest_right[1]) ** 2:
                nearest_right = [right_x[j], right_y[j]]
        
        triangles.append([[left_x[i], left_y[i]], [left_x[i + 1], left_y[i + 1]], nearest_right])
    
        test_x.append((left_x[i] + nearest_right[0]) / 2)
        test_y.append((left_y[i] + nearest_right[1]) / 2)
        
        test_x.append((left_x[i + 1] + nearest_right[0]) / 2)
        test_y.append((left_y[i + 1] + nearest_right[1]) / 2)
            
    return np.array(test_x), np.array(test_y), np.array(triangles)

def delaunay_library(left_x, left_y, right_x, right_y):
    left = np.stack((left_x, left_y), axis=1)
    right = np.stack((right_x, right_y), axis=1)
    
    temp = np.arange(len(left) - 1)
    temp2 = np.arange(len(left), len(left) + len(right) - 1)
    
    segments1 = np.concatenate((np.column_stack((temp, temp + 1)), [[len(left) - 1, 0]]))
    segments2 = np.concatenate((np.column_stack((temp2, temp2 + 1)), [[len(left) + len(right) - 1, len(left)]]))
    
    data = {"vertices": np.concatenate((left, right)), 
            "segments": np.concatenate((segments1, segments2))
        }
    
    triangulation = triangle.triangulate(data, "p")
    
    triangle.compare(plt, data, triangulation)

if __name__ == "__main__":
    df = pd.read_csv('./tracks/' + INPUT_FILE_NAME)

    left_df = df[df['type'] == 'left']
    right_df = df[df['type'] == 'right']

    left_x = left_df['x'].to_numpy()
    left_y = left_df['y'].to_numpy()
    right_x = right_df['x'].to_numpy()
    right_y = right_df['y'].to_numpy()
    
    # mid_x, mid_y = find_midline(left_x, left_y, right_x, right_y)
    mid_x, mid_y, triangles = basic_triangulation(left_x, left_y, right_x, right_y)
    
    plt.figure(figsize=(10,6))
    plt.plot(left_x, left_y, color='blue', marker='o')
    plt.plot(right_x, right_y, color='gold', marker='o')
    plt.plot(mid_x, mid_y, color='green', marker='x')
    plt.grid(True)
    
    ax = plt.gca()
    ax.set_aspect("equal")
    
    poly_collection = PolyCollection(triangles, facecolors=(1,0,0,0), edgecolors='black', linewidths=1)
    ax.add_collection(poly_collection)
    
    # -------------------------------------------------------
    # 2. Arc-Length Parameterization
    # -------------------------------------------------------

    mid_points_duplicates = np.column_stack((mid_x, mid_y))
    
    _, idx = np.unique(mid_points_duplicates, axis=0, return_index=True)
    mid_points = mid_points_duplicates[np.sort(idx)]
    sorted_points = mid_points

    def arc_length_param(points):
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        return s

    s = arc_length_param(sorted_points)

    # -------------------------------------------------------
    # 3. Non-Periodic Cubic Spline
    # -------------------------------------------------------

    # bc_type options:
    # 'natural' → second derivative = 0 at endpoints
    # 'clamped' → specify slope at endpoints
    # 'not-a-knot' → default behavior

    cs_x = CubicSpline(s, sorted_points[:,0], bc_type='natural')
    cs_y = CubicSpline(s, sorted_points[:,1], bc_type='natural')

    # Fine sampling
    s_fine = np.linspace(0, s[-1], 200)
    x_fine = cs_x(s_fine)
    y_fine = cs_y(s_fine)

    # -------------------------------------------------------
    # 4. Curvature Computation
    # -------------------------------------------------------

    dx = cs_x(s_fine, 1)
    dy = cs_y(s_fine, 1)
    ddx = cs_x(s_fine, 2)
    ddy = cs_y(s_fine, 2)

    curvature = (dx*ddy - dy*ddx) / (dx*dx + dy*dy)**1.5

    # -------------------------------------------------------
    # 5. Plot
    # -------------------------------------------------------
    
    plt.plot(x_fine, y_fine, '-', label="Spline")
    plt.legend()
    
    delaunay_library(left_x, left_y, right_x, right_y)
    
    plt.show()