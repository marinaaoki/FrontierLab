from __future__               import print_function
from imutils.object_detection import non_max_suppression
from helper                   import find_fg_objects, thirdof, centreof, find_intersections, areaof
from fuzzy                    import fuzzy_infer
from sklearn.preprocessing    import normalize

import numpy as np
import cv2   as cv
import os

def detect_people(folder, source, disclose_all=True, threshold=0.3):
    # Initialise HOG feature descriptor.
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    # Access the image stream.
    cap = cv.VideoCapture(os.path.join(source,'%03d.png'), cv.CAP_IMAGES)
    firstFrame = None
    # Rectangles modeling frequently used path.
    path = []
    # Centres of rectangles to enable trajectory modeling.
    centres = []
    # Tuples of (idx1,idx2,area) storing intersections between objects and path rectangles.
    intersections = []

    # Initialise risk level.
    risk_lvl = -1

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
            blurred = cv.blur(resized, (15,15), 0)
            if not disclose_all:
                resized = blurred

            # Initialise matrix represenation of image with 0s.
            matrix = np.zeros((WIDTH, HEIGHT))

            cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
            cv.putText(resized, "Frame " + str(cap.get(cv .CAP_PROP_POS_FRAMES)) + ", threshold=" + str(threshold), (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv.imshow("Frame", resized)
            #cv.imwrite(os.path.join(folder, "frame_" + str((cap.get(cv .CAP_PROP_POS_FRAMES))) + ".png"), resized)
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
        pick_human, pick = find_fg_objects(pick, pick_hog, threshold)
        pick_human = np.array(pick_human)
        pick_human = non_max_suppression(pick_human, overlapThresh=0.65)

        ### --- Path modeling --- ###
        for rect in pick_human:
            # Take bottom third of pick_human rectangles to model path.
            # Only if the rectangle is at least a certain height (1/3 of the total frame height)
            (x1, y1, x2, y2) = rect
            H = x2 - x1
            W = y2 - y1
            #if H <= HEIGHT // 3 or W <= WIDTH // 3:
            #    step = thirdof(rect)
            #    path.append(step)
            #    centres.append(centreof(step))

            step = thirdof(rect)
            path.append(step)
            centres.append(centreof(step))

            #Increment pixel values of matrix by 1 as counter of path frequency.
            for i in range(x1, x2):
                for j in range(y1, y2):
                    #print("x1: " + str(x1) + ", y1: " + str(y1) + "\nx2: " + str(x2) + ", y2: " + str(y2) + "\n")
                    matrix[i-1][j-1] += 1

        if not np.all((matrix == 0)) and pick_human != []: 
            # Normalise matrix.
            normed = normalize(matrix, axis=1, norm='l2')

        ### -- Drawing trajectory --- ###
        #for p1, p2 in zip(centres, centres[1:]):
        #    cv.arrowedLine(resized, p1, p2, (255,0,0), 2)
                
        ### --- Dangerous object detection --- ###
        # Checking intersection of each rectangle in path and pick.
        if path != []:
            intersections = find_intersections(path, pick, intersections)

        # Rectangles used to model path in white
        #for (startX, startY, endX, endY) in path:
        #        cv.rectangle(resized, (startX, startY), (endX, endY), (255, 255, 255), 2)

        # Background subtraction in red
        #for (startX, startY, endX, endY) in pick:
        #            cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        # HOG for human detection in blue
        #for (startX, startY, endX, endY) in pick_hog:
        #            cv.rectangle(resized, (startX, startY), (endX, endY), (255, 0, 0), 2)

        # Union in green
        #for (startX, startY, endX, endY) in pick_human:
        #    cv.rectangle(resized, (startX, startY), (endX, endY), (0, 255, 0), 2)

        ### --- INFORMATION DISCLOSURE --- ###
        blurred = cv.blur(resized, (15,15), 0)
        if disclose_all:
            # Iterate through the intersections to find objects that have been determined to be dangerous.
            for (idx1,idx2,area,danger) in intersections:
                if danger == 2:
                    (startX, startY, endX, endY) = pick[idx2]

                    ### --- FUZZY INFERENCE --- ###
                    sliced = normed[startX:endX, startY:endY]
                    risk_lvl = fuzzy_infer(sliced, pick[idx2])
                    
                    # Draw black rectangle around detected dangerous object.
                    cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 0), 2)
                    cv.putText(resized, "DANGEROUS OBJECT DETECTED", (15,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                    
        else:
            # Blur out everything except for the dangerous object that was detected.
            mask = np.zeros(resized.shape[:2], dtype="uint8")
            # Iterate through the intersections to find objects that have been determined to be dangerous.
            for (idx1,idx2,area,danger) in intersections:
                if danger == 2:
                    (startX, startY, endX, endY) = pick[idx2]
                    # Draw filled rectangle around detected dangerous object to change mask.
                    cv.rectangle(mask, (startX, startY), (endX, endY), (255, 255, 255), -1)

            mask = cv.bitwise_not(mask)
            resized[mask>0] = blurred[mask>0]
            # Iterate through the intersections to find objects that have been determined to be dangerous.
            for (idx1,idx2,area,danger) in intersections:
                if danger == 2:
                    (startX, startY, endX, endY) = pick[idx2]

                    ### --- FUZZY INFERENCE --- ###
                    sliced = normed[startX:endX, startY:endY]
                    risk_lvl = fuzzy_infer(sliced, pick[idx2])

                    # Draw black rectangle around detected dangerous object.
                    cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 0), 2)
                    cv.putText(resized, "DANGEROUS OBJECT DETECTED", (15,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        
        cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
        cv.putText(resized, "Frame " + str(cap.get(cv .CAP_PROP_POS_FRAMES)) + ", threshold=" + str(threshold), (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        
        cv.imshow("Frame", resized)
        #cv.imwrite(os.path.join(folder, "frame_" + str((cap.get(cv .CAP_PROP_POS_FRAMES))) + ".png"), resized)
        
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        
    cap.release()
    return risk_lvl
    

def risk_notification(folder, source, disclose_all, threshold):
    risk_lvl = detect_people(folder, source, disclose_all, threshold)
    ### --- RISK LEVEL-BASED INFORMATION DISCLOSURE --- #
    if risk_lvl == 0:
        print("\nLow risk level.\nSend update in weekly report.")
    elif risk_lvl == 1:
        print("\nMedium risk level.\nNotify primary caregiver.")
    elif risk_lvl == 2:
        print("\nHigh risk level.\nSend urgent warning notification to primary caregiver.")

