'''
SwishAI, your jumpshot coach in your backpocket
'''
###TODO: Fix Knee/ crouch angle detection
###TODO: implement hand/finger detection

### importing necessary libraries

import cv2
import mediapipe as mpipe
import numpy

mpipe_drawing=mpipe.solutions.drawing_utils
mpipe_pose=mpipe.solutions.pose
mpipe_hands=mpipe.solutions.hands


### variables to change for testing
capture_device=0
#POSE
detect_conf_level_pose=0.5 # higher is more accurate, but also less likely to work (must detect perfect body)
track_conf_level_pose=0.5


#HANDS
detect_conf_level_hands=0.5
track_conf_level_hands=0.5
max_num_hands=0

### functions

# calculates angle between joints for analyzing displacement
def angle_between(a,b,c):
    #in order of joints closest to your core to farthest (ex. hip then knee then foot)
    a = numpy.array(a)
    b = numpy.array(b)
    c = numpy.array (c)
    #arctan2 is used intead of arctan because arctan cannot handle x being 0
    radians= numpy.arctan2(c[1]-b[1], c[0]-b[0]) - numpy.arctan2(a[1]-b[1],a[0]-b[0])
    angle = numpy.abs(radians*180.0/numpy.pi)

    if angle > 180.0:
        angle= 360 - angle
    
    return angle


### video feed capture
capture=cv2.VideoCapture(capture_device) ##seeting up video capture device (webcam)
with mpipe_pose.Pose(min_detection_confidence=detect_conf_level_pose,min_tracking_confidence=track_conf_level_pose) as pose:
    with mpipe_hands.Hands(min_detection_confidence=detect_conf_level_hands,min_tracking_confidence=track_conf_level_hands) as hands:
        
        #begin capturing when the webcam is running, stop when user exits
        while capture.isOpened():
            rt, frame = capture.read()
            

            # switch from BGR to RGB, mediapipe only works with rgb

            image= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable= False # makes it impossible to change color order. saves memory

            tracked_pose= pose.process(image) # feed to mediapipe
            tracked_hands=hands.process(image)

            # switch back to BGR, cv2 prefers BGR when rendering
            image.flags.writeable= True # making color order changeable again
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            #extracting landmarks/nodes for calculations
            try: ##in a try except statement to ignore any index errors that pop up when a body or hand are not found
                #might need if statemnt around each one to check if they are detected first.
                nodes_pose= tracked_pose.pose_landmarks.landmark
                nodes_hand1=tracked_hands.multi_hand_landmarks[0].landmark
                nodes_hand2=tracked_hands.multi_hand_landmarks[1].landmark

                ##### finding coordinates of pose nodes
                right_shoulder= [nodes_pose[mpipe_pose.PoseLandmark.RIGHT_SHOULDER.value].x,nodes_pose[mpipe_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow=[nodes_pose[mpipe_pose.PoseLandmark.RIGHT_ELBOW.value].x,nodes_pose[mpipe_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist=[nodes_pose[mpipe_pose.PoseLandmark.RIGHT_WRIST.value].x,nodes_pose[mpipe_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip=[nodes_pose[mpipe_pose.PoseLandmark.RIGHT_HIP.value].x,nodes_pose[mpipe_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_shoulder= [nodes_pose[mpipe_pose.PoseLandmark.LEFT_SHOULDER.value].x,nodes_pose[mpipe_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip=[nodes_pose[mpipe_pose.PoseLandmark.LEFT_HIP.value].x,nodes_pose[mpipe_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee=[nodes_pose[mpipe_pose.PoseLandmark.LEFT_KNEE.value].x,nodes_pose[mpipe_pose.PoseLandmark.LEFT_KNEE.value].y]
                ##ankle maybe

                


                ###Acceleration
                ###Use 

                #calculating angle of body parts
                angle_shoulder=angle_between(right_shoulder,right_elbow,right_wrist) 
                angle_hip=angle_between(right_hip,right_shoulder,right_elbow)
                angle_smooth=angle_between(left_knee,left_hip,left_shoulder)

                #displaying angle on screen

                #right elbow is placeholder, put in its place all the midpoints
                #TODO replace 700,700 with dimensions of web cam
                cv2.putText(image,str(numpy.round(angle_shoulder)), tuple(numpy.multiply(right_elbow, [640,480]).astype(int)), 
                                                    cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
            
                cv2.putText(image,str(numpy.round(angle_hip)), tuple(numpy.multiply(right_shoulder, [640,480]).astype(int)), 
                                                    cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
            
                cv2.putText(image,str(numpy.round(angle_smooth)), tuple(numpy.multiply(left_hip, [640,480]).astype(int)), 
                                                    cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
                

                print(nodes_pose)
            except:
                pass
        

            #rendering
            try:
                mpipe_drawing.draw_landmarks(image,tracked_pose.pose_landmarks, mpipe_pose.POSE_CONNECTIONS)
                mpipe_drawing.draw_landmarks(image,tracked_hands.multi_hand_landmarks[0], mpipe_hands.HAND_CONNECTIONS)
                mpipe_drawing.draw_landmarks(image,tracked_hands.multi_hand_landmarks[1], mpipe_hands.HAND_CONNECTIONS)
            except:
                pass    
            
            #image size
            width=1280
            
            aspect_ratio= image.shape[1] / image.shape[0] #this is width/ height respectively

            calc_height=int(width/ aspect_ratio)
            
            
            resized_image=cv2.resize(image, (width,calc_height))
            cv2.imshow('SwishAI Feed', resized_image) # popup window
            if cv2.waitKey(10) & 0xFF == ord('q'): #if "q" (for quit) is pressed, stop the loop
                break
#
# exited webcam capturing, stop capturing and close the window
capture.release()
cv2.destroyAllWindows()


