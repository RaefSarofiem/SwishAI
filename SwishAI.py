'''
SwishAI, your jumpshot coach in your backpocket
'''

### importing necessary libraries

import cv2
import mediapipe as mpipe
import numpy

mpipe_drawing=mpipe.solutions.drawing_utils
mpipe_pose=mpipe.solutions.pose

### variables to change for testing

detect_conf_level=0.5 # higher is more accurate, but also less likely to work (must detect perfect body)
track_conf_level=0.5
capture_device=0


### video feed capture

capture=cv2.VideoCapture(capture_device) #setting up video capture device (webcam)
##!TODO: check if this is the right webcam input (now and when you get the temp webcam from walmart)

with mpipe_pose.Pose(min_detection_confidence=detect_conf_level,min_tracking_confidence=track_conf_level) as pose:

    #begin capturing when the webcam is running, stop when user exits
    while capture.isOpened():
        rt, frame = capture.read()

        # switch from BGR to RGB, mediapipe only works with rgb

        image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable= False # makes it impossible to change color order. saves memory

        results= pose.process(image) # feed to mediapipe

        # switch back to BGR, cv2 prefers BGR when rendering
        image.flags.writeable= True # making color order changeable again
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #rendering
        mpipe_drawing.draw_landmarks(image,results.pose_landmarks, mpipe_pose.POSE_CONNECTIONS)


        cv2.imshow('SwishAI Feed', image) # popup window
        if cv2.waitKey(10) & 0xFF == ord('q'): #if "q" (for quit) is pressed, stop the loop
            break

# exited webcam capturing, stop capturing and close the window
capture.release()
cv2.destroyAllWindows()


