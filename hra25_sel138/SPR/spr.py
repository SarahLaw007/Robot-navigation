import sys
import numpy as np
import itertools

from collections import defaultdict
from typing import Dict, Optional, List, Tuple
from heapq import *

'''
Report reflexive vertices
'''


def angle(x: np.array, y: np.array) -> float:
    """
    Find angle between two vectors.
    """
    return np.arccos(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def is_inner(x: np.array, y: np.array) -> bool:
    """
    Returns True iff the angle
    between vectors x, y is the
    inner angle within the polygon.
    Since the points are given in
    clockwise order, it's an inner
    angle if the cross product is
    negative.
    """
    return np.cross(x, y) < 0


def is_reflexive(p1: np.array, p2: np.array, p3: np.array) -> bool:
    """
    Returns True iff the p2 is a reflexive vertex.
    """
    x = p2 - p1
    y = p3 - p2
    theta = angle(x, y)

    # make sure theta equals the exterior angle
    if is_inner(x, y):
        theta = 2 * np.pi - theta

    return theta > np.pi


def findReflexiveVertices(
        polygons: List[List[List[float]]]) -> List[List[float]]:
    vertices = []

    # Your code goes here
    # You should return a list of (x,y) values as lists, i.e.
    # vertices = [[x1,y1],[x2,y2],...]

    for polygon in polygons:
        for i in range(len(polygon)):
            p1 = np.array(polygon[i - 1])
            p2 = np.array(polygon[i])
            p3 = np.array(polygon[0] if i == (
                len(polygon) - 1) else polygon[i + 1])

            if is_reflexive(p1, p2, p3):
                vertices.append(polygon[i])

    return vertices


'''
Compute the roadmap graph
'''


def perp(a: np.array) -> np.array:
    """
    Compute orthogonal (i.e. perpendicular)
    vector of a.
    """
    b = np.empty_like(a)

    b[0] = -a[1]
    b[1] = a[0]

    return b


def intersect(
        x1: np.array,
        x2: np.array,
        y1: np.array,
        y2: np.array) -> np.array:
    """
    Find intersection of lines induced by vectors
    xi, yi.
    """
    dx = x2 - x1
    dy = y2 - y1

    dp = x1 - y1
    dxp = perp(dx)

    num = np.dot(dxp, dp)
    denom = np.dot(dxp, dy)

    if denom == 0:
        return np.array([])

    return (num / denom.astype(float)) * dy + y1


def in_same_polygon(polygons: List[List[List[float]]],
                    target: List[float],
                    possible_neighbor: List[float]) -> List[List[float]]:
    """
    If target and possible_combinations
    are in the same polygon, then return
    list that represents that polygon.
    Otherwise, return None.
    """
    for polygon in polygons:
        if target in polygon:
            if possible_neighbor in polygon:
                return polygon
            else:
                return None
    raise ValueError("Parameter \"target\" not in any polygon.")


def is_within_bounds(
        x1: List[float],
        x2: List[float],
        y1: List[float],
        y2: List[float],
        int_pt: List[float]):
    """
    Return true iff int_pt is within bounds of
    the line segments induced by xi, yi.
    """
    ix, iy = int_pt

    if not min(x1[0], x2[0]) <= ix <= max(x1[0], x2[0]):
        return False
    elif not min(x1[1], x2[1]) <= iy <= max(x1[1], x2[1]):
        return False
    elif not min(y1[0], y2[0]) <= ix <= max(y1[0], y2[0]):
        return False
    elif not min(y1[1], y2[1]) <= iy <= max(y1[1], y2[1]):
        return False
    else:
        return True


def visible_in_polygon(polygon: List[List[float]],
                       target: List[float],
                       possible_neighbor: List[float],
                       clockwise: bool = True) -> bool:
    """
    Assumes that target and possible_neighbor are
    vertices of the same polygon, and sees if
    possible_neighbor is visible from target.
    """
    n = len(polygon)
    i = polygon.index(target)
    result = None

    for num_inc in range(1, n):
        if clockwise:
            j = i + num_inc
        else:
            j = i - num_inc

            if j < 0:
                j = n - j

        j %= n

        if is_reflexive(np.array(polygon[j - 1]),
                        np.array(polygon[j]),
                        np.array(polygon[(j + 1) % n])):
            result = polygon[j]
            break

    return result == possible_neighbor


def visible_outside_polygon(
        polygons: List[List[List[float]]], target: List[float], possible_neighbor: List[float]):
    """
    Assumes that target and possible_neighbor are
    vertices of different polygons, and sees if
    possible_neighbor is visible from target.
    """
    for polygon in polygons:
        n = len(polygon)
        for i in range(n):
            p_i = polygon[i]
            p_j = polygon[(i + 1) % n]

            if p_i == target or p_i == possible_neighbor:
                continue

            if p_j == target or p_j == possible_neighbor:
                continue

            int_pt = list(
                intersect(
                    np.array(p_i),
                    np.array(p_j),
                    np.array(target),
                    np.array(possible_neighbor)))
            if int_pt and is_within_bounds(
                    target, possible_neighbor, p_i, p_j, int_pt):
                return False

    return True


def computeSPRoadmap(polygons: List[List[List[float]]],
                     reflexVertices: List[List[float]]) -> Tuple[Dict[int,
                                                                      List[float]],
                                                                 Dict[int,
                                                                      List[List[float]]]]:
    vertexMap = dict(enumerate(reflexVertices, 1))
    adjacencyListMap = defaultdict(list)

    # Your code goes here
    # You should check for each pair of vertices whether the
    # edge between them should belong to the shortest path
    # roadmap.
    #
    # Your vertexMap should look like
    # {1: [5.2,6.7], 2: [9.2,2.3], ... }
    #
    # and your adjacencyListMap should look like
    # {1: [[2, 5.95], [3, 4.72]], 2: [[1, 5.95], [5,3.52]], ... }
    #
    # The vertex labels used here should start from 1

    for ((target_index, target), (possible_neighbor_index, possible_neighbor)
         ) in itertools.combinations(vertexMap.items(), 2):
        polygon = in_same_polygon(polygons, target, possible_neighbor)
        if polygon:
            if visible_in_polygon(
                    polygon,
                    target,
                    possible_neighbor) or visible_in_polygon(
                    polygon,
                    target,
                    possible_neighbor,
                    clockwise=False):
                target_vec = np.array(target)
                neighbor_vec = np.array(possible_neighbor)

                adjacencyListMap[target_index].append(
                    [possible_neighbor_index, np.linalg.norm(target_vec - neighbor_vec)])
                adjacencyListMap[possible_neighbor_index].append(
                    [target_index, np.linalg.norm(target_vec - neighbor_vec)])
        elif visible_outside_polygon(polygons, target, possible_neighbor):
            target_vec = np.array(target)
            neighbor_vec = np.array(possible_neighbor)

            adjacencyListMap[target_index].append(
                [possible_neighbor_index, np.linalg.norm(target_vec - neighbor_vec)])
            adjacencyListMap[possible_neighbor_index].append(
                [target_index, np.linalg.norm(target_vec - neighbor_vec)])

    return vertexMap, adjacencyListMap


'''
Perform uniform cost search
'''


def uniformCostSearch(adjListMap, start, goal):
    # Your code goes here. As the result, the function should
    # return a list of vertex labels, e.g.
    #
    # path = [23, 15, 9, ..., 37]
    #
    # in which 23 would be the label for the start and 37 the
    # label for the goal.

    queue = [(0, start, ())]
    seen = set()
    mins = {start: 0}

    while queue:
        (cost, v1, path) = heappop(queue)
        if v1 not in seen:
            seen.add(v1)
            path += (v1,)

            if v1 == goal:
                return (list(path), cost)

            for v2, c in adjListMap.get(v1, ()):
                if v2 in seen:
                    continue

                prev = mins.get(v2, None)
                nxt = cost + c

                if not prev or nxt < prev:
                    mins[v2] = nxt
                    heappush(queue, (nxt, v2, path))

    return [], float("inf")


'''
Agument roadmap to include start and goal
'''


def updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2):
    updatedALMap = defaultdict(list)
    startLabel = 0
    goalLabel = -1

    # Your code goes here. Note that for convenience, we
    # let start and goal have vertex labels 0 and -1,
    # respectively. Make sure you use these as your labels
    # for the start and goal vertices in the shortest path
    # roadmap. Note that what you do here is similar to
    # when you construct the roadmap.

    updatedALMap.update(adjListMap)

    for (possible_neighbor_index, possible_neighbor) in vertexMap.items():
        if visible_outside_polygon(polygons, [x1, y1], possible_neighbor):
            dist = np.linalg.norm(
                np.array([x1, y1]) - np.array(possible_neighbor))

            updatedALMap[possible_neighbor_index].append([startLabel, dist])
            updatedALMap[startLabel].append([possible_neighbor_index, dist])

        if visible_outside_polygon(polygons, [x2, y2], possible_neighbor):
            dist = np.linalg.norm(
                np.array([x2, y2]) - np.array(possible_neighbor))

            updatedALMap[possible_neighbor_index].append([goalLabel, dist])
            updatedALMap[goalLabel].append([possible_neighbor_index, dist])

    return startLabel, goalLabel, updatedALMap


if __name__ == "__main__":
    # Retrive file name for input data
    if(len(sys.argv) < 6):
        print(
            "Five arguments required: python spr.py [env-file] [x1] [y1] [x2] [y2]")
        exit()

    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

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
    print("")

    # Compute reflex vertices
    reflexVertices = findReflexiveVertices(polygons)
    print("Reflexive vertices:")
    print(str(reflexVertices))
    print("")

    # Compute the roadmap
    vertexMap, adjListMap = computeSPRoadmap(polygons, reflexVertices)
    print("Vertex map:")
    print(str(vertexMap))
    print("")
    print("Base roadmap:")
    print(dict(adjListMap))
    print("")

    # Update roadmap
    start, goal, updatedALMap = updateRoadmap(
        polygons, vertexMap, adjListMap, x1, y1, x2, y2)
    print("Updated roadmap:")
    print(dict(updatedALMap))
    print("")

    # Search for a solution
    path, length = uniformCostSearch(updatedALMap, start, goal)
    print("Final path:")
    print(str(path))
    print("Final path length:" + str(length))
