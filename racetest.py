import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import pandas as pd
import triangle
import json

from scipy.interpolate import CubicSpline

import first_delaunay_midline
import second_delaunay_midline

import polygon_constructor

import shapely

# Starts from 1
TEST_CASE = 3

UNORDERED_INPUTS = True

def get_data_for_test_case():
    data = {}
    
    try:
        with open('test_configs.json', 'r') as file:
            data = json.load(file)
        
    except FileNotFoundError:
        print("Error: The file 'test_configs.json' was not found.")
    
    return data[TEST_CASE - 1]


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
# Assumes order
# Some right cones are never connected
# Trash, not super efficient, but who cares lol
def basic_triangulation(left_x, left_y, right_x, right_y,):
    test_x = []
    test_y = []
    
    triangles = []
    
    for i, _ in enumerate(left_x[1:]):
        nearest_right = [1000, 1000]
        
        for j, _ in enumerate(right_x):
            if (left_x[i] - right_x[j]) ** 2 + (left_y[i] - right_y[j]) ** 2 < (left_x[i] - nearest_right[0]) ** 2 + (left_y[i] - nearest_right[1]) ** 2:
                nearest_right = [right_x[j], right_y[j]]
        
        triangles.append([[left_x[i], left_y[i]], [left_x[i + 1], left_y[i + 1]], nearest_right])
    
        midpoint_1_x = (left_x[i] + nearest_right[0]) / 2
        midpoint_1_y = (left_y[i] + nearest_right[1]) / 2
        midpoint_2_x = (left_x[i + 1] + nearest_right[0]) / 2
        midpoint_2_y = (left_y[i + 1] + nearest_right[1]) / 2
    
        # Midpoint of first leg
        if i == 0 or (test_x[-1] != midpoint_1_x and test_y[-1] != midpoint_1_y):
            test_x.append(midpoint_1_x)
            test_y.append(midpoint_1_y)
        
        # Midpoint of second leg
        # if i == 0 or (test_x[-1] != midpoint_2_x and test_y[-1] != midpoint_2_y):
        test_x.append(midpoint_2_x)
        test_y.append(midpoint_2_y)
            
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


def cubic_spline(points):
    # Arc length parameterization, whatever that means
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])

    # -------------------------------------------------------
    # 3. Non-Periodic Cubic Spline
    # -------------------------------------------------------

    # bc_type options:
    # 'natural' → second derivative = 0 at endpoints
    # 'clamped' → specify slope at endpoints
    # 'not-a-knot' → default behavior

    cs_x = CubicSpline(s, points[:,0], bc_type='natural')
    cs_y = CubicSpline(s, points[:,1], bc_type='natural')

    # Fine sampling
    s_fine = np.linspace(0, s[-1], 200)
    x_fine = cs_x(s_fine)
    y_fine = cs_y(s_fine)

    # # -------------------------------------------------------
    # # 4. Curvature Computation
    # # -------------------------------------------------------

    # dx = cs_x(s_fine, 1)
    # dy = cs_y(s_fine, 1)
    # ddx = cs_x(s_fine, 2)
    # ddy = cs_y(s_fine, 2)

    # curvature = (dx*ddy - dy*ddx) / (dx*dx + dy*dy)**1.5
    
    return x_fine, y_fine


if __name__ == "__main__":
    test_case_data = get_data_for_test_case()
    file_name = test_case_data["name"]
    car_pos = test_case_data["position"]
    
    heading_angle = np.deg2rad(test_case_data["heading"])
    heading_vector = np.array([np.cos(heading_angle), np.sin(heading_angle)])
    
    df = pd.read_csv('./tracks/' + file_name + '.csv')

    left_df = df[df['type'] == 'left']
    right_df = df[df['type'] == 'right']

    left_x = left_df['x'].to_numpy()
    left_y = left_df['y'].to_numpy()
    right_x = right_df['x'].to_numpy()
    right_y = right_df['y'].to_numpy()
    
    left = np.column_stack((left_x, left_y)) # (left_x, left_y) is not a tuple but an NDarray infuriatingly
    right = np.column_stack((right_x, right_y))
    
    if UNORDERED_INPUTS:
        left_unsorted = np.random.permutation(left)
        right_unsorted = np.random.permutation(right)
        
        left = polygon_constructor.order_boundary_weighted(left_unsorted, car_pos, heading_vector)
        right = polygon_constructor.order_boundary_weighted(right_unsorted, car_pos, heading_vector)
    
    polygon = polygon_constructor.polygon_pipeline(left_unsorted, right_unsorted, car_pos, heading_vector)
    
    # Old midline
    # mid_x, mid_y = find_midline(left_x, left_y, right_x, right_y)
    # mid_x, mid_y, triangles = basic_triangulation(left_x, left_y, right_x, right_y)
    # mid = np.column_stack((mid_x, mid_y))
    
    # poly_collection = PolyCollection(triangles, facecolors=(1,0,0,0), edgecolors='black', linewidths=1)
    
    # x_fine, y_fine = cubic_spline(mid)
    
    # delaunay_midline, delaunay_triangles = first_delaunay_midline.midline(left, right)
    # delaunay_midline, delaunay_triangles = map(np.array, first_delaunay_midline.midline(left.tolist(), right.tolist())) # Ensure everything is in numpy arrays
    
    # delaunay_midline_x = delaunay_midline[:,0]
    # delaunay_midline_y = delaunay_midline[:,1]
    
    # delaunay_poly_collection = PolyCollection(delaunay_triangles, facecolors=(1,0,0,0), edgecolors='black', linewidths=1)
    
    # cubic_spline_x_2, cubic_spline_y_2 = cubic_spline(delaunay_midline)
    
    
    # Matplotlib stuff
    plt.figure(figsize=(10,6))
    plt.grid(True)
    
    plt.plot(left_x, left_y, color='blue', marker='o')
    plt.plot(right_x, right_y, color='gold', marker='o')
    
    plt.arrow(car_pos[0], car_pos[1], 2 * heading_vector[0], 2 * heading_vector[1], color="red", linewidth=2, head_width=1)
    
    plt.plot(left[:,0], left[:, 1], color="red")
    plt.plot(right[:,0], right[:, 1], color="green")
    
    polygon_closed = np.vstack((polygon, np.array(polygon[0])))
    plt.plot(polygon_closed[:,0], polygon_closed[:,1], color="orange", linestyle="--")
    
    ax = plt.gca()
    ax.set_aspect("equal")
    
    # Naive midline
    # plt.plot(mid_x, mid_y, color='green', marker='x')
    # ax.add_collection(poly_collection)
    
    # plt.plot(x_fine, y_fine, '-', label="Spline")
    
    # Delaunay midline
    # plt.plot(delaunay_midline_x, delaunay_midline_y, color='forestgreen', marker='x')
    # ax.add_collection(delaunay_poly_collection)

    # plt.plot(cubic_spline_x_2, cubic_spline_y_2, '-', label="Spline 2")
    
    triangles = shapely.constrained_delaunay_triangles(shapely.Polygon(polygon_closed))
    
    for tri in triangles.geoms:
        x, y = tri.exterior.xy
        plt.plot(x, y, "k-")
    
    plt.legend()
    
    # delaunay_library(left_x, left_y, right_x, right_y)
    
    plt.show()