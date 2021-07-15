from __future__ import print_function
import numpy as np

def union(a,b):
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[2], b[2])
    y2 = max(a[3], b[3])
    return [x1, y1, x2, y2]

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

# Returns the are of the given rectangle.
def area(a):
    dx = a[2] - a[0]
    dy = a[3] - a[1]
    return dx * dy

# Determine foreground objects from rectangles detected by background subtraction and HOG feature descriptor.
def find_fg_objects(pick_fg, pick_hog, threshold):
    pick_total = []
    for rect1 in pick_fg:
        for rect2 in pick_hog:
            # If there is an intersection that takes up at least the given threshold of pick_hog rectangle, take union of both boxes.
            if intersection(rect1, rect2) != [0,0,0,0]:
                fraction = area_of_intersection(rect1, rect2) / area(rect2)
                if fraction >= threshold:
                    union_of = union(rect1, rect2)
                    pick_total.append(union_of)
    return pick_total