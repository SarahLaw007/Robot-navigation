import sys

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

from spr import *

'''
Set up matplotlib to create a plot with an empty square
'''


def setupPlot():
    fig = plt.figure(num=None, dpi=120, facecolor='w', edgecolor='k')
    ax = fig.subplots()
    ax.set_axisbelow(True)
    ax.grid(which='minor', linestyle=':', alpha=0.2)
    ax.grid(which='major', linestyle=':', alpha=0.5)
    return fig, ax


'''
Make a patch for a single pology
'''


def createPolygonPatch(polygon):
    verts = []
    codes = []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0], xy[1]))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='gray', lw=1)

    return patch


'''
Make a patch for the robot
'''


def createPolygonPatchForRobot(polygon):
    verts = []
    codes = []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0], xy[1]))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='gray', lw=1)

    return patch


'''
Render polygon obstacles
'''


def drawPolygons(polygons, fig, ax):
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p])
        ax.add_patch(patch)


if __name__ == "__main__":

    # Retrive file name for input data
    if(len(sys.argv) < 2):
        print("Please provide input tfile: python visualize.py [env-file]")
        exit()

    filename = sys.argv[1]

    # Read data and parse polygons
    lines = [line.rstrip('\n') for line in open(filename)]
    polygons = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            polygon.append([float(i) for i in xys[p].split(',')])
        polygons.append(polygon)

    # Print out the data
    print("Pologonal obstacles:")
    for p in range(0, len(polygons)):
        print(str(polygons[p]))

    # Setup
    fig, ax = setupPlot()

    # Draw the polygons
    drawPolygons(polygons, fig, ax)

    # Extra visualization elements goes here

    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Compute path data
    reflexVertices = findReflexiveVertices(polygons)
    vertexMap, adjListMap = computeSPRoadmap(polygons, reflexVertices)
    start, goal, updatedALMap = updateRoadmap(
        polygons, vertexMap, adjListMap, x1, y1, x2, y2)
    path, length = uniformCostSearch(updatedALMap, start, goal)

    path_edges = set()

    for i in range(len(path) - 1):
        path_edges.add((path[i], path[i + 1]))

    plt.plot(x1, y1)
    plt.plot(x2, y2)

    for (v_num, edges) in updatedALMap.items():
        for (w_num, _) in edges:
            if v_num == 0:
                v = (x1, y1)
            elif v_num == -1:
                v = (x2, y2)
            else:
                v = vertexMap[v_num]

            if w_num == 0:
                w = (x1, y1)
            elif w_num == -1:
                w = (x2, y2)
            else:
                w = vertexMap[w_num]

            if (v_num, w_num) in path_edges or (w_num, v_num) in path_edges:
                plt.plot([v[0], w[0]], [v[1], w[1]], color="red")
            else:
                plt.plot([v[0], w[0]], [v[1], w[1]], color="green")

    plt.show()
