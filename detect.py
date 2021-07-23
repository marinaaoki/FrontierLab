from __future__               import print_function
from imutils.object_detection import non_max_suppression
from helper                   import find_fg_objects, thirdof, centreof, find_intersections

import numpy as np
import cv2 as cv
import os

def detect_people(folder, source, access_lvl=3):
    # Model background using Gaussian Mixture Model.
    backSub = cv.createBackgroundSubtractorMOG2()

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    tracker = cv.legacy.TrackerMOSSE_create()
    
    # Access the image stream.
    cap = cv.VideoCapture(os.path.join(source,'%03d.png'), cv.CAP_IMAGES)
    firstFrame = None
    # Rectangles modeling frequently used path.
    path = []
    # Centres of rectangles to enable trajectory modeling.
    centres = []
    # Tuples of (idx1,idx2,area) storing intersections between objects and path rectangles.
    intersections = []

    if not cap.isOpened():
        print('Unable to open')
        exit(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        # Resize frame for faster processing.
        resized = cv.resize(frame, (400, 300))
        # Greyscale of resized image.
        grey = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        # Extract height and width for full-body detection.
        (HEIGHT, WIDTH) = resized.shape[:2]

        # Initialise first frame.
        if firstFrame is None:
            firstFrame = grey
            cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
            cv.putText(resized, str(cap.get(cv .CAP_PROP_POS_FRAMES)), (15,15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv.imwrite(os.path.join(folder, "frame_" + str((cap.get(cv .CAP_PROP_POS_FRAMES))) + ".png"), resized)
            continue

        ### --- Static background subtraction --- ###
        img_diff = cv.absdiff(firstFrame, grey)
        # Threshold image
        ret, thresh = cv.threshold(img_diff, 25, 255, cv.THRESH_BINARY)
        thresh = cv.dilate(thresh, None, iterations=2)
        contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        # Draw bounding rectangles around detected diff.
        boxes = []
        for c in contours:
            if cv.contourArea(c) < 500:
                continue
            (x, y, w, h) = cv.boundingRect(c)
            boxes.append((x, y, x+w, y+h))
        boxes = np.array(boxes)
        # Non-maxima suppression to get rid of overlapping boxes.
        pick = non_max_suppression(boxes, overlapThresh=0.3)
        # Get rid of rectangles that are the same size as the entire frame.
        x_max = resized.shape[1]
        y_max = resized.shape[0]
        for idx, lst in enumerate(pick):
            if (lst[2] >= x_max) and (lst[3] >= y_max):
                pick = np.delete(pick, obj=idx, axis=0)
        
        ### --- HOG people detection --- ###
        rects, weights = hog.detectMultiScale(grey, winStride=(4,4), padding=(8,8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick_hog = non_max_suppression(rects, overlapThresh=0.65)
           
        # Check whether "humans" detected by HOG are part of detected foreground.
        pick_human, pick = find_fg_objects(pick, pick_hog, 0.3)
        pick_human = np.array([[x, y, w, h] for [x, y, w, h] in pick_human])
        pick_human = non_max_suppression(pick_human, overlapThresh=0.65)

        ### --- Path modeling --- ###
        for rect in pick_human:
            # Take bottom third of pick_human rectangles to model path.
            # Only if the rectangle is at least a certain height (1/3 of the total frame height)
            h = rect[3] - rect[1]
            w = rect[2] - rect[0]
            if h <= HEIGHT // 3 or w <= WIDTH // 3:
                step = thirdof(rect)
                path.append(step)
                centres.append(centreof(step))

        ### -- Drawing trajectory --- ###
        #for p1, p2 in zip(centres, centres[1:]):
        #    cv.arrowedLine(resized, p1, p2, (255,0,0), 2)
                
        ### --- Dangerous object detection --- ###
        # Checking intersection of each rectangle in path and pick.
        if path != []:
            intersections = find_intersections(path, pick, intersections)

        # Background subtraction in red
        #for (startX, startY, endX, endY) in pick:
        #            cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        # HOG for human detection in blue
        #for (startX, startY, endX, endY) in pick_hog:
        #            cv.rectangle(resized, (startX, startY), (endX, endY), (255, 0, 0), 2)

        # Union in green
        #for (startX, startY, endX, endY) in pick_human:
        #    cv.rectangle(resized, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Rectangles used to model path in white
        #for (startX, startY, endX, endY) in path:
        #        cv.rectangle(resized, (startX, startY), (endX, endY), (255, 255, 255), 2)

        ### --- INFORMATION DISCLOSURE --- ###
        blurred = cv.blur(resized, (15,15), 0)
        # Tier 1: Nothing changes
        if access_lvl == 2:
            # Tier 2: Blur out everything except for the dangerous object that was detected.
            mask = np.zeros(resized.shape[:2], dtype="uint8")
            for (idx1,idx2,area,danger) in intersections:
                if danger:
                    (startX, startY, endX, endY) = pick[idx2]
                    cv.rectangle(mask, (startX, startY), (endX, endY), (255, 255, 255), -1)
            mask = cv.bitwise_not(mask)
            resized[mask>0] = blurred[mask>0]
        elif access_lvl == 3:
            # Tier 3: Send out only textual data -> blur everything and place text on top?
            resized = blurred

        # Iterate through the intersections to find objects that have been determined to be dangerous.
        for (idx1,idx2,area,danger) in intersections:
            if danger:
                (startX, startY, endX, endY) = pick[idx2]
                # Draw black rectangle around detected dangerous object.
                cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv.putText(resized, "DANGEROUS OBJECT DETECTED", (15,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        
        cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
        cv.putText(resized, str(cap.get(cv .CAP_PROP_POS_FRAMES)), (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        
        cv.imshow("Frame", resized)
        #cv.imwrite(os.path.join(folder, "frame_" + str((cap.get(cv .CAP_PROP_POS_FRAMES))) + ".png"), resized)
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        
    cap.release()

