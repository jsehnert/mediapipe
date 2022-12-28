from mymediapipe import MyFaceMeshToy, MyFaceToy, MyPoseToy
from videocam import MyWebCam

import cv2
import numpy as np
import time

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

    window_title="MediaPipe Fun..."
    camera = MyWebCam(fps=200)
    camera.open(0)
    face_toy = MyFaceToy()

    while camera.isOpened():
        success, image = camera.getFrame()
        if success == False: continue 
        #convert to RGB and flip as if looking in moirror
        
        image = cv2.flip(image, 1)
        img_h, img_w, img_c = image.shape
        #
        # Face detection stuff
        #
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_toy.process(imageRGB)
        if results.detections != None:
            for det in results.detections:
                print(det)
                print('###')

        #print((results))
        break
        #imageRGB = face_toy.processAndMarkImage(imageRGB)
        #image = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        #faceLoc = face_toy.marks(imageRGB)
        #del imageRGB
        #for face in faceLoc:
            #cv2.rectangle(image, pt1= face[0], pt2=face[1], color=(0,255,255), thickness=3)
        #
        # End face detection stuff
        #

        cv2.imshow(window_title, image)#cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(5)
        if key == ord('q'):
            break
     
    camera.close()
    
    cv2.destroyAllWindows()
    print(-1) 
    
   