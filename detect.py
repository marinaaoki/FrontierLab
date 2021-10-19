from __future__               import print_function
from imutils.object_detection import non_max_suppression
from helper                   import find_fg_objects, thirdof, find_intersections, areaof
from fuzzy                    import fuzzy_infer

import numpy as np
import cv2   as cv
import os

def detect_people(source: str, disclose_all: bool = True, threshold: float = 0.3) -> int:
    # Access the image stream.
    cap = cv.VideoCapture(os.path.join(source,'%03d.png'), cv.CAP_IMAGES)
    firstFrame = None

    if not cap.isOpened():
        print('Unable to open')
        exit(0)
    
    # Intialise HOG feature descriptor.
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    path = []
    intersections = []
    sizes = []
    human_size = 1
    risk_lvl = -1

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        resized = cv.resize(frame, (400, 300))
        grey = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        (HEIGHT, WIDTH) = resized.shape[:2]

        # Initialise first frame.
        if firstFrame is None:
            firstFrame = grey
            blurred = cv.blur(resized, (15,15), 0)
            if not disclose_all:
                resized = blurred

            # Initialise matrix represenation of image with 0s.
            matrix = np.zeros((WIDTH, HEIGHT))

            # Frame display.
            cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
            cv.putText(resized, "Frame " + str(cap.get(cv .CAP_PROP_POS_FRAMES)) + ", threshold=" + str(threshold), (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
            cv.imshow("Frame", resized)
            continue

        ### --- Static background subtraction --- ###
        img_diff = cv.absdiff(firstFrame, grey)
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
            (x1, y1, x2, y2) = rect
            H = x2 - x1
            W = y2 - y1
            # Take bottom third of pick_human rectangles to model path.
            step = thirdof(rect)
            [thirdx1, thirdy1, thirdx2, thirdy2] = step
            path.append(step)

            # Calculate the area of each detected human.
            size = areaof(rect)
            sizes.append(size)

            #Increment pixel values of matrix by 1 as counter of path frequency.
            for i in range(thirdx1, thirdx2):
                for j in range(thirdy1, thirdy2):
                    matrix[i-1][j-1] += 1
        
        if sizes != []:
            # Calculate average human size for fuzzy inference.
            human_size = np.mean(sizes)

        if not np.all((matrix == 0)) and pick_human != []: 
            normed = matrix / np.linalg.norm(matrix)

        ### --- Dangerous object detection --- ###
        if path != []:
            intersections = find_intersections(path, pick, intersections)

        ### --- INFORMATION DISCLOSURE --- ###
        blurred = cv.blur(resized, (15,15), 0)
        if disclose_all:
            for (idx1,idx2,area,danger) in intersections:
                if danger == 2:
                    (startX, startY, endX, endY) = pick[idx2]

                    ### --- FUZZY INFERENCE --- ###
                    sliced = normed[startX:endX, startY:endY]
                    prop = areaof(pick[idx2]) / human_size

                    risk_lvl = fuzzy_infer(sliced, prop)
                    
                    # Display dangerous objects.
                    cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 0), 2)
                    cv.putText(resized, "OBJECT DETECTED" + str(risk_lvl), (15,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                    
        else:
            mask = np.zeros(resized.shape[:2], dtype="uint8")
            for (idx1,idx2,area,danger) in intersections:
                if danger == 2:
                    (startX, startY, endX, endY) = pick[idx2]
                    # Draw filled rectangle around detected dangerous object to change mask.
                    cv.rectangle(mask, (startX, startY), (endX, endY), (255, 255, 255), -1)

            mask = cv.bitwise_not(mask)
            resized[mask>0] = blurred[mask>0]

            for (idx1,idx2,area,danger) in intersections:
                if danger == 2:
                    (startX, startY, endX, endY) = pick[idx2]

                    ### --- FUZZY INFERENCE --- ###
                    sliced = normed[startX:endX, startY:endY]
                    prop = areaof(pick[idx2]) / human_size

                    risk_lvl = fuzzy_infer(sliced, prop)

                    # Display dangerous objects.
                    cv.rectangle(resized, (startX, startY), (endX, endY), (0, 0, 0), 2)
                    cv.putText(resized, "OBJECT DETECTED" + str(risk_lvl), (15,30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        
        # Frame display.
        cv.rectangle(resized, (10,2), (100,2), (255,255,255), -1)
        cv.putText(resized, "Frame " + str(cap.get(cv .CAP_PROP_POS_FRAMES)) + ", threshold=" + str(threshold), (15,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv.imshow("Frame", resized)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        
    cap.release()
    return risk_lvl

def risk_notification(source: str, disclose_all: bool, threshold: float):
    risk_lvl = detect_people(source, disclose_all, threshold)
    ### --- RISK LEVEL-BASED INFORMATION DISCLOSURE --- #
    if risk_lvl == 0:
        print("\nLow risk level.\nSend update in weekly report.")
    elif risk_lvl == 1:
        print("\nMedium risk level.\nSend update in daily report.")
    elif risk_lvl == 2:
        print("\nHigh risk level.\nSend urgent warning notification to primary caregiver.")

