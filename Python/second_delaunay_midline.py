from helpers import *

# Annoying note: no standardized point format: either tuple or list
# Edges may also similarly be unstandardized

# Understanding this will be a personal project for later


# ---------------------------------------------------------
# Delaunay triangulation (Bowyer-Watson)
# ---------------------------------------------------------

# Degenerate triangles: when triangle forms a line. Therefore, deleting them is the same as deleting a line in a list of triangles: just cleaning up data

def delaunay(points):

    st_vertices = super_triangle(points)
    st = Triangle(st_vertices[0], st_vertices[1], st_vertices[2])
    
    triangles = [st]

    for point in points:

        bad=[] 

        for t in triangles:
            a,b,c = t.v
            
            if in_circumcircle(t, point):
                bad.append(t)

        boundary=[]

        for t in bad:
            for e in t.edges():

                shared = False

                for ot in bad:
                    if ot == t:
                        continue

                    # Since for triangle edges, order matters. Note to self: fix this?
                    if is_triangle_edge(ot, e):
                        shared=True
                        break

                if not shared:
                    boundary.append(e)

        for t in bad:
            triangles.remove(t)

        for e in boundary:
            t_new = Triangle(e[0], e[1], point)
            
            if not is_degenerate(t_new):
                triangles.append(Triangle(e[0], e[1], point))
                
        print("triangles before insert:", len(triangles))
        print("bad:", len(bad))
        print("boundary:", len(boundary))

    # Probably the super triangle deletion step?
    triangles = [t for t in triangles if not any(v in st.v for v in t.v)]

    return triangles


# ---------------------------------------------------------
# Edge utilities
# ---------------------------------------------------------

def edge_exists(triangles, a, b):
    for t in triangles:
        if is_triangle_edge(t, (a, b)):
            return True

    return False


def flip_edge(triangles: list, edge):

    t1=None
    t2=None

    for t in triangles:
        for e in t.edges():
            if edges_equal(e, edge):
                if t1 is None:
                    t1=t
                else:
                    t2=t
                    break

    if not t1 or not t2:
        return

    a, b = edge

    c=next(v for v in t1.v if v not in edge)
    d=next(v for v in t2.v if v not in edge)

    triangles.remove(t1)
    triangles.remove(t2)

    triangles.append(Triangle(c,d,a))
    triangles.append(Triangle(c,d,b))


# ---------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------

def enforce_constraints(triangles,constraints):
    changed = True

    while changed:
        changed=False

        for a, b in constraints:
            
            if edge_exists(triangles,a,b):
                print("edge is part of constraint")
                continue

            print("does this ever run")

            for t in triangles:

                for e in t.edges():

                    if segments_intersect(a,b,e[0],e[1]):

                        flip_edge(triangles,e)
                        changed=True
                        break

                if changed:
                    break

    return triangles


# ---------------------------------------------------------
# Build FSAE CDT
# ---------------------------------------------------------

def build_fsae_cdt(left_sorted, right_sorted):
    polygon = left_sorted + right_sorted[::-1] # LIST CONCATENATION???? WTF
    polygon = ensure_ccw(polygon)

    constraints=[]

    # Handling list-to-tuple is annoying as f
    for i in range(len(polygon)):
        constraints.append(
            (tuple(polygon[i]), tuple(polygon[(i + 1) % len(polygon)]))
        )

    tris = delaunay(polygon)
    tris = enforce_constraints(tris,constraints)

    return tris, left_sorted, right_sorted


# ---------------------------------------------------------
# Centerline extraction
# ---------------------------------------------------------

def extract_centerline(triangles,left_set,right_set):

    centers=[]

    print("reached 1")

    for t in triangles:

        for a,b in t.edges():

            if (a in left_set and b in right_set) or \
               (b in left_set and a in right_set):

                centers.append(
                    ((a[0]+b[0])/2,(a[1]+b[1])/2)
                )

    return centers


# ---------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------

def midline(left_cones_sorted: list, right_cones_sorted: list):
    """
    Constructs midline with Delaunay

    Args:
        left_cones_sorted (list): sorted list of left cone coordinates
        right_cones_sorted (list): sorted list of right cone coordinates

    Returns:
        list: centerline
    """
    left_cones_normalized = [tuple(p) for p in left_cones_sorted]
    right_cones_normalized = [tuple(p) for p in right_cones_sorted]
    
    tris,left,right = build_fsae_cdt(left_cones_normalized, right_cones_normalized)

    # do sets really do anything here
    centerline = extract_centerline(tris, left, right)

    return centerline, tris