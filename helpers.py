MIN_VALID_AREA = 1e-9
TOLERANCE = 1e-5

# Handling of list vs tuple issues is inconsistent at best

# ---------------------------------------------------------
# List/tuple handling
# ---------------------------------------------------------

def points_equal(a, b):
    return a[0] == b[0] and a[1] == b[1]


def edges_equal(e1, e2):
    if type(e1) != tuple or type(e2) != tuple:
        print("Not tuple", e1, e2)
    
    return (points_equal(e1[0], e2[0]) and points_equal(e1[1], e2[1])) or (points_equal(e1[0], e2[1]) and points_equal(e1[1], e2[0]))


# ---------------------------------------------------------
# Triangle class
# ---------------------------------------------------------

class Triangle:
    """
    Simple collection of 3 points
    Automatically enforces CCW orientation and checks for degeneracy
    """

    def __init__(self, a, b, c):
        if abs(orient(a,b,c)) < MIN_VALID_AREA:
            raise ValueError("degenerate triangle")

        if orient(a,b,c) < 0:
            self.v = (a, c, b)
            print("reforce")

        else:
            self.v = (a, b, c)

    def edges(self):
        return [
            (self.v[0], self.v[1]),
            (self.v[1], self.v[2]),
            (self.v[2], self.v[0])
        ]
        

def is_triangle_edge(t: Triangle, edge):
    """
    Checks if edge is an edge of the triangle
    Operation: checks if edge or reversed edge is in triangle edges

    Args:
        t (Triangle): triangle
        edge (tuple): tuple of two points
    """
    
    for e in t.edges():
        if edges_equal(e, edge):
            return True
        
    return False


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

def orient(a, b, c):
    """ 
    Calculates |AB x AC|

    Args:
        a (list): x, y pair
        b (list): x, y pair
        c (list): x, y pair

    Returns:
        int: |AB x AC|. Positive if CCW, negative if CW
    """
    
    # For my own sake: AB = b - a, AC = c - a
    # Then, 2D cross product formula
    
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4):
    """
    Another bit of comp geometry magic
    Assumes both segments have non-zero length
    
    Args:
        p1 (list): segment 1 endpoint
        p2 (list): segment 1 endpoint
        p3 (list): segment 2 endpoint
        p4 (list): segment 2 endpoint
    """
    
    # Sharing endpoints is completely fine. Using == for this might be a bit sus
    if points_equal(p1, p3) or points_equal(p1, p4) or points_equal(p2, p3) or points_equal(p2, p4):
        return False
    
    o1 = orient(p1,p2,p3)
    o2 = orient(p1,p2,p4)
    o3 = orient(p3,p4,p1)
    o4 = orient(p3,p4,p2)

    # Opposite signs apparently. One positive, one negative
    return o1 * o2 < 0 and o3 * o4 < 0


def points_approx_equal(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 <= TOLERANCE ** 2

def in_xy_bounds(bound1, bound2, c):
    return (min(bound1[0], bound2[0]) <= c[0] <= max(bound1[0], bound2[0]) and
            min(bound1[1], bound2[1]) <= c[1] <= max(bound1[1], bound2[1]))

def segments_intersect2(p1, p2, p3, p4):
    p1_shared =  points_approx_equal(p1, p3) or points_approx_equal(p1, p4)
    p2_shared = points_approx_equal(p2, p3) or points_approx_equal(p2, p4)
    
    if not p1_shared == p2_shared:
        return False
    
    o1 = orient(p1,p2,p3)
    o2 = orient(p1,p2,p4)
    o3 = orient(p3,p4,p1)
    o4 = orient(p3,p4,p2)


def is_degenerate(t: Triangle):
    return abs(orient(t.v[0], t.v[1], t.v[2])) < MIN_VALID_AREA


def is_triangle_CCW(t: Triangle):
    """
    Checks if points in triangle are listed in CCW order
    Theoretically, points should only be CW or CCW; if not, something has gone seriously wrong
    Because Triangle should force CCW orientation, this should never return False 

    Args:
        t (Triangle)
    """
    ret = orient(t.v[0], t.v[1], t.v[2]) > 0
    if not ret:
        raise ValueError("Triangle is not CCW")
        
    return ret


def in_circumcircle(t: Triangle, point):
    """
    Calculates determinant of matrix to check if point is in triangle's circumcircle
    Big-brain technique. For explanation, see: https://ianthehenry.com/posts/delaunay/
    Algorithm assumes points are put counterclockwise, so extra check is added

    Args:
        t (Triangle): triangle
        point (list): x, y pair

    Returns:
        bool: whether or not point is in circumcircle
    """
    if not is_triangle_CCW(t):
        raise ValueError("Triangle is not CCW")
    
    a, b, c = t.v[0], t.v[1], t.v[2]
    
    ax, ay = a[0] - point[0], a[1] - point[1]
    bx, by = b[0] - point[0], b[1] - point[1]
    cx, cy = c[0] - point[0], c[1] - point[1]

    det = (
        (ax*ax + ay*ay)*(bx*cy - by*cx)
        - (bx*bx + by*by)*(ax*cy - ay*cx)
        + (cx*cx + cy*cy)*(ax*by - ay*bx)
    )
            
    return det > 0


# ---------------------------------------------------------
# Polygon utilities
# ---------------------------------------------------------

def polygon_area(poly):
    area = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1*y2 - x2*y1
    return area/2


def ensure_ccw(poly):
    if polygon_area(poly) < 0:
        return poly[::-1]
    return poly


# ---------------------------------------------------------
# Super triangle
# ---------------------------------------------------------

def super_triangle(points):

    minx = min(p[0] for p in points)
    miny = min(p[1] for p in points)
    maxx = max(p[0] for p in points)
    maxy = max(p[1] for p in points)

    dx = maxx - minx
    dy = maxy - miny
    d = max(dx, dy) * 10

    # Apparently, guards against degeneracy?
    if d == 0:
        d = 1

    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2

    return [
        (midx - d,midy - d),
        (midx, midy + d),
        (midx + d,midy - d)
    ]