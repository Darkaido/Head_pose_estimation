import cv2
import mediapipe as mp
import numpy as np



import math

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

def relative(landmark, shape):
    x = int(landmark.x * shape[1])
    y =int(landmark.y * shape[0])

    return (x,y)


# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(landmark, right_indices, left_indices,frame):
    # Right eyes 
    # horizontal line 
    rh_right = relative(landmark[right_indices[0]], frame.shape)
    rh_left = relative(landmark[right_indices[8]], frame.shape)
    # vertical line 
    rv_top = relative(landmark[right_indices[12]], frame.shape)
    rv_bottom = relative(landmark[right_indices[4]], frame.shape)
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = relative(landmark[left_indices[0]], frame.shape)
    lh_left = relative(landmark[left_indices[8]], frame.shape)

    # vertical line 
    lv_top = relative(landmark[left_indices[12]], frame.shape)
    lv_bottom = relative(landmark[left_indices[4]], frame.shape)
     
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)
    # print(rhDistance,rvDistance , "   right eyes")

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    # print(lvDistance,lhDistance, "  left eyes")

    if ((rvDistance < 4.0) and (lvDistance < 4.0)):
        # Add the text on the image
        
        text = "Eyes are closed"
        print(text)
        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        










iris=[473,468]
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
fps = 0

while cap.isOpened():
    success, image = cap.read()
    fps+=1
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:

        # right eyes indices
        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

        # Left eyes indices 
        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        
        # ratio = blinkRatio(results.multi_face_landmarks[0].landmark, RIGHT_EYE,LEFT_EYE,image)

        # left_pupil = relative(results.multi_face_landmarks[iris[0]].landmark, image.shape)
        # right_pupil = relative(results.multi_face_landmarks[iris[1]].landmark, image.shape)
  
        # cv2.circle(frame,right_pupil,3,(0,0,255))
        # cv2.circle(frame,left_pupil,3,(0,0,255))

        for face_landmarks in results.multi_face_landmarks:

            

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 468:
                    print('yes')
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 :
                # if idx == 468 or idx == 473:
                    if idx == 1:
                    # if idx == 468:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

        
            # print(ratio)
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)


            
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # print(y)

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif x > 20:
                text = "Looking Upward"  
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]))
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Head Pose Estimation', image)
    key = cv2.waitKey(2)
    if key==ord('q') or key ==ord('Q'):
        break
cv2.destroyAllWindows()
cap.release()
    
            