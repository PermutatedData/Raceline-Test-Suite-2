# BUSTED
 
from helpers import *

def bowyer_watson(points):
    st_vertices = super_triangle(points)
    st = Triangle(st_vertices[0], st_vertices[1], st_vertices[2])
    
    triangles = [st]

    for p in points:

        bad = []

        for tri in triangles:
            if in_circumcircle(tri, p):
                bad.append(tri)

        edge_count = {}

        for tri in bad:
            for e in tri.edges():
                k = frozenset(e)
                edge_count[k] = edge_count.get(k,0) + 1

        polygon = []

        for e, count in edge_count.items():
            if count == 1:
                polygon.append(tuple(e))

        for tri in bad:
            triangles.remove(tri)

        for e in polygon:
            a,b = e
            triangles.append(Triangle(a, b, p))

    result = []

    for tri in triangles:
        if any(v in st for v in tri):
            continue
        result.append(tri)

    return result


def recover_edge(triangles, edge):
    a, b = edge

    changed = True

    while changed:
        changed = False

        for t1 in triangles:
            for t2 in triangles:

                if t1 == t2:
                    continue

                e1 = frozenset(t1.edges())
                e2 = frozenset(t2.edges())

                shared = e1 & e2

                if len(shared) != 1:
                    continue

                shared_edge = list(shared)[0]

                if not segments_intersect(*shared_edge,a,b):
                    continue

                c = [v for v in t1 if v not in shared_edge][0]
                d = [v for v in t2 if v not in shared_edge][0]

                triangles.remove(t1)
                triangles.remove(t2)

                triangles.append(Triangle(c,a,b))
                triangles.append(Triangle(d,a,b))

                changed = True
                break

            if changed:
                break


def constrained_delaunay(points, constraints):

    triangles = bowyer_watson(points)

    for edge in constraints:
        recover_edge(triangles, edge)

    return triangles