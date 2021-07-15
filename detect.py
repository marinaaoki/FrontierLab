from __future__               import print_function
from typing                   import Type
from imutils.convenience import resize
from imutils.object_detection import non_max_suppression
from helper                   import find_fg_objects

import numpy as np
import cv2 as cv
import imutils

def detect_people():
    # Model background using Gaussian Mixture Model.
    backSub = cv.createBackgroundSubtractorMOG2()

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    tracker = cv.legacy.TrackerMOSSE_create()
    
    # Access the image stream.
    cap = cv.VideoCapture('C:\\Users\\aokim\\Documents\\Bachelorarbeit\\opencv\\data\\%03d.png', cv.CAP_IMAGES)
    initBB = None
    firstFrame = True

    if not cap.isOpened():
        print('Unable to open')
        exit(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        # Resize frame for faster processing.
        resized = imutils.resize(frame, width=min(400, frame.shape[1]))
        # Greyscale of resized image.
        grey = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        (H, W) = resized.shape[:2]

        ### --- GMM background subtraction --- ###
        fgMask = backSub.apply(resized)
        contours = cv.findContours(fgMask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
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
        pick_human = find_fg_objects(pick, pick_hog, 0.25)
        pick_human = np.array([[x, y, w, h] for [x, y, w, h] in pick_human])
        pick_human = non_max_suppression(pick_human, overlapThresh=0.65)

        ### --- Object on ground detection --- ###
        path = []


        # GMM background subtraction in red
        #for (startX, startY, endX, endY) in pick:
                    #cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # HOG for human detection in blue
        #for (startX, startY, endX, endY) in pick_hog:
                    #cv.rectangle(resized, (startX, startY), (endX, endY), (255, 0, 0), 2)

        # Union in green
        #if pick_human != []:
        #    for (startX, startY, endX, endY) in pick_human:
        #        cv.rectangle(resized, (startX, startY), (endX, endY), (0, 255, 0), 2)

        ### --- MOSSE tracker --- ###
        # Check whether an object is already being tracked.
        if initBB is not None:
            (success, box) = tracker.update(resized)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                # Yellow rectangle around succesfully-tracked objects.
                cv.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # TODO: Approximate walking path by taking lower eighth of rectangle and half the width in the centre.

                # TODO: Save walking path and pass on to object detection.

            else:
                initBB = None
                #print("Tracking failed! Restarting initialisation...")

        # If a moving object has been detected, start tracking it.
        if pick_human != [] and firstFrame == False:
            #print("Initialising tracker...")
            #print("pick_human in frame " + str(cap.get(cv .CAP_PROP_POS_FRAMES)) + ": " + str(pick_human))
            initBB = [pick_human[0][0], pick_human[0][1], pick_human[0][2], pick_human[0][3]]
            #print("initBB: " + str(initBB))
            # Purple rectangle around detected objects.
            cv.rectangle(resized, (pick_human[0,0], pick_human[0,1]), (pick_human[0,2],pick_human[0,3]), (128,0,128), 2)
            tracker.init(resized, initBB)

        cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
        cv.putText(resized, str(cap.get(cv .CAP_PROP_POS_FRAMES)), (15,15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        
        cv.imshow("Frame", resized)
        #cv.imwrite("C:\\Users\\aokim\\Documents\\Bachelorarbeit\\opencv\\result_track\\frame_" + str((cap.get(cv .CAP_PROP_POS_FRAMES))) + ".png", resized)
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

        # Prevent first frame being picked up by tracker.
        firstFrame = False
        
    cap.release()

