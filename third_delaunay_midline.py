from helpers import *


def delaunay(points):
    st_vertices = super_triangle(points)
    super_tri = Triangle(st_vertices[0], st_vertices[1], st_vertices[2])

    triangles = [super_tri]

    for p in points:
        bad = []
        circles = {}

        for t in triangles:
            c = circumcircle(t)
            circles[t] = c
            if in_circle(p, c):
                bad.append(t)

        polygon = []
        for t in bad:
            for edge in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                if edge in polygon:
                    polygon.remove(edge)
                else:
                    polygon.append(edge)

        for t in bad:
            triangles.remove(t)

        for edge in polygon:
            triangles.append((edge[0], edge[1], p))

    result = []
    for t in triangles:
        if any(v in super_tri for v in t):
            continue
        result.append(t)

    return result