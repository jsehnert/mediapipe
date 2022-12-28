from mymediapipe import MyFaceMeshToy, MyFaceToy, MyPoseToy
from videocam import MyWebCam

import cv2
import numpy as np
import time


###############################################################################

# COMMENT: With python 3.6, the face detection class does not support model selection

#
# This identifies L eye, R eye, Nose, Chin, L ear, R ear
# The frame-to-frame results to appear a bit unstable
#
# NOTE: a detection object encapsulates a face with
#       - a label ID (likely an indexer to enumerate faces in picture)
#       - a detction score
#       - a bounding box (in relative coords)
#       - 6 key points (in relative coords)
#
###############################################################################

#
# Resources:
# https://www.youtube.com/watch?v=-toNMaS4SeQ
#

#
# Open the video camera and return the opened device



# camera object
#
def getLook( x,y,z):
    text = 'Looking Forward' 
    if y < -10: text = 'Looking Left'
    elif y > 10: text = 'Looking Right'
    elif x < -10: text = 'Looking Down'
    elif x > 10: text = 'Looking Up'
    
    return text

###############################################################################
########## MAIN ##########
###############################################################################
if __name__=='__main__':

# 33  - lateral corner of left eye (as looking at person)
# 199 - chin
# 263 - lateral corner of right eye (as looking at person)
# 61  - lateral left corner of mouth
# 10  - middle of forehead

#TESTING
    #f_detect = mp.solutions.face_detection.Toy)
    window_title="MediaPipe Fun..."
    camera = MyWebCam(fps=200)
    camera.open(0)
    face_mesh_toy = MyFaceMeshToy(still=False, numFaces=3, tol1=0.5, drawMesh=True)
    face_toy = MyFaceToy()
    pose_toy = MyPoseToy()

    frame_count = 0
    lm_index=0
    upperLimit=468
    lowerLimit=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0,0,255)
    fontSize = .2*3
    fontThickness = 2 

    cv2.namedWindow('Trackbars')
    cv2.moveWindow('Trackbars', 550,100)
    cv2.resizeWindow('Trackbars', 400,150)
    def setLower(value):
        global lowerLimit
        lowerLimit=value
    
    def setHigher(value):
        global higherLimit
        higherLimit=value
        
    cv2.createTrackbar('lowerLimit', 'Trackbars', 0, 468, setLower)
    cv2.createTrackbar('higherLimit', 'Trackbars', 468, 468, setHigher)

    while camera.isOpened():
        frame_count += 1
        success, image = camera.getFrame()
        if success == False: continue

        #to track FPS
        start = time.time()        
        
        #convert to RGB and flip as if looking in moirror
        
        image = cv2.flip(image, 1)
        img_h, img_w, img_c = image.shape
        #
        # Face detection stuff
        #
        faceLoc = face_toy.marks(image)
        for face in faceLoc:
            cv2.rectangle(image, pt1= face[0], pt2=face[1], color=(0,255,255), thickness=3)
        #
        # End face detection stuff
        #

        #
        # Face pose stuff
        #
        poseLM = pose_toy.marks(image)
        if poseLM != []:
            lms=0
            for lm in poseLM:
                cv2.circle(image, lm, 10, color=(0,255,0),thickness=-1)
                lms+=1
                if lms >= 11: break
        #
        # End Face Pose stuff
        #

        #
        # Face mesh stuff
        #
        facesMeshLM = face_mesh_toy.marks(image)
        for faceMeshLM in facesMeshLM:
            cnt=0
            for lm in faceMeshLM:
                if cnt>=lowerLimit and cnt <=higherLimit:
                    pass#cv2.putText(image, str(cnt),lm, font, fontSize, fontColor, fontThickness)
                cnt += 1

        '''image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_toy.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = True

        pointsOfInterest = {}
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if True:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                        pointsOfInterest[idx] = (lm.x * img_w, lm.y * img_h)
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                #Foca length has no particular effect other than scaling
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h/2],
                                      [0, focal_length, img_w/2 ],
                                      [0, 0, 1]])
 
                #the distortion parameters
                dist_coeffs = np.zeros((4,1), dtype=np.float64)

                #Solve PnP - finds the rotation and translation that minimizes the reprojection error from #d-2D point correspondences
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)

                #get the rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                #get angles, the 3x3 upper triangular matrix R, 3x3 orthogonal matrix Q, the 3 rotation matrices around the canonical axes
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x, y, z  =angles[0]*360, angles[1]*360, angles[2]*360
                
                #see where the head is facing
                text = getLook(x,y,z)
        
                #draw line starting on tip of nose and pointing into the pose direction
                #nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_coeffs)
                
                #print(nose_2d)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                #cv2.line(image, p1, p2, (255,0,0), 3)

                p = pointsOfInterest[lm_index]
                p = (int(p[0]), int(p[1]))
                cv2.line(image, (p[0]-1, p[1]), (p[0]+1, p[1]), (0,255,0), 5)

                #draw other captured face points
                #for x,y in

                #Place text onto the image
                #cv2.putText(image, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                #cv2.putText(image, "x: " + str(np.round(x,2)), (500,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #cv2.putText(image, "y: " + str(np.round(y,2)), (500,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                #cv2.putText(image, "z: " + str(np.round(z,2)), (500,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(image, "Landmark: " + str(lm_index), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                #
                # End - face mesh stuff
                #

            end = time.time()
            totalTime = end - start

            fps = 1.0 / totalTime
            #print("FPS: ", fps)
            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

            #face_mesh_toy.drawLandmarks(image, face_landmarks)'''


        cv2.imshow(window_title, image)#cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        elif key==ord('u'):
            lm_index +=1
        elif key==ord('d'):
            lm_index -=1

        if( lm_index < 0): lm_index = 467
        elif( lm_index >= 468 ): lm_index = 0
   
    camera.close()
    
    cv2.destroyAllWindows()
    print(-1) 
    
   