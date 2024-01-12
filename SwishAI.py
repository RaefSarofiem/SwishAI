'''
SwishAI, your jumpshot coach in your backpocket
'''

### importing necessary libraries

import cv2
import mediapipe as mpipe
import numpy

mpipe_drawing=mpipe.solutions.drawing_utils
mpipe_pose=mpipe.solutions.pose

### video feed capture

capture=cv2.VideoCapture(0) #setting up video capture device (webcam)
##!TODO: check if this is the right webcam input (now and when you get the temp webcam from walmart)

#begin capturing when the webcam is running, stop when user exits
while capture.isOpened():
    rt, frame = capture.read()
    cv2.imshow('SwishAI Feed', frame) # popup window
    if cv2.waitKey(10) & 0xFF == ord('q'): #if the window is closed or "q" (for quit) is pressed, stop the loop
        break

# exited webcam capturing, stop capturing and close the window
capture.release()
cv2.destroyAllWindows()


