from __future__ import print_function

import numpy as np
import cv2   as cv
import glob
import os

# Returns the coordinates of the union of the two rectangles.
def union(a,b):
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[2], b[2])
    y2 = max(a[3], b[3])
    return [x1, y1, x2, y2]

# Returns the coordinates of the intersection of the two rectangles.
def intersection(a,b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]) 
    y2 = min(a[3], b[3])
    w = min(a[2], b[2]) - x1
    h = min(a[3], b[3]) - y1
    if w < 0 or h < 0:
        return [0,0,0,0]
    return [x1, y1, x2, y2]

# Returns the area of the intersection of the given rectangles.
def area_of_intersection(a,b):
    intersect = intersection(a,b)
    if intersect == [0,0,0,0]:
        return 0
    dx = intersect[2] - intersect[0]
    dy = intersect[3] - intersect[1]
    return dx * dy

# Returns the area of the given rectangle.
def areaof(a):
    dx = a[2] - a[0]
    dy = a[3] - a[1]
    return dx * dy

# Takes a given rectangle and returns the rectangle comprising of the bottom third widthwise and middle third lengthwise.
def thirdof(a):
    (x1, y1, x2, y2) = a
    w = x2 - x1
    h = y2 - y1
    thirdx1 = x1 + (w // 3)
    thirdy1 = y1 + ((h // 3) * 2)
    thirdx2 = x2 - (w // 3)
    thirdy2 = y2
    return [thirdx1, thirdy1, thirdx2, thirdy2]

# Takes a given rectangle and returns the point in the centre of the bottom side.
def centreof(a):
    w = a[2] - a[0]
    h = a[3] - a[1]
    x = a[0] + (w // 2)
    y = a[3]
    return (x, y)

# Given a list of rectangles representing a modeled path and objects, finds the intersections between each and returns as list of (idx1,idx2,area).
def find_intersections(path, objects, intersections):
    new_intersections = []
    for idx1, rect1 in enumerate(path):
        for idx2, rect2 in enumerate(objects):
            # If there is an intersection, store (idx1,idx2,area) in a list that is passed on to the next frame.
            intersect_area = area_of_intersection(rect1,rect2)
            # Iterate through intersections list and check whether rect1, rect2 are already in it.
            inlist = False
            for index, (i1,i2,a,d) in enumerate(intersections):
                if (idx1 == i1) and (idx2 == i2):
                    inlist = True
                    if intersect_area == a:
                        if d == 0:
                            new_intersections.append((idx1,idx2,intersect_area,1))
                        else: 
                            new_intersections.append((idx1,idx2,intersect_area,2))
            if (not inlist) and (intersect_area != 0):
                new_intersections.append((idx1,idx2,intersect_area,0))
    return new_intersections

# Determine foreground objects from rectangles detected by background subtraction and HOG feature descriptor.
def find_fg_objects(pick_fg, pick_hog, threshold):
    pick_total = []
    to_delete = []
    for idx, rect1 in enumerate(pick_fg):
        for rect2 in pick_hog:
            # If there is an intersection that takes up at least the given threshold of pick_hog rectangle, take union of both boxes.
            if intersection(rect1, rect2) != [0,0,0,0]:
                fraction = area_of_intersection(rect1, rect2) / areaof(rect2)
                if fraction >= threshold:
                    union_of = union(rect1, rect2)
                    pick_total.append(union_of)
                    # Store a list of indexes to remove from pick_fg.
                    to_delete.append(idx)
    to_delete = list(set(to_delete))
    if to_delete != []:
        pick_fg = np.delete(pick_fg, to_delete, axis=0)
    return pick_total, pick_fg