# 23 DEC 2022
# Jim Sehnert
#

class MyFaceToy:
    import mediapipe as mp
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.face_detection = self.mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_detection_confidence,
                                                                   model_selection=model_selection)
        self.mp_drawing = self.mp.solutions.drawing_utils

    def marks(self, image):
        results = self.process(image)
        faceBoundBoxes = []
        if results.detections != None:
            img_h, img_w, _ = image.shape
            for face in results.detections:
                bBox = face.location_data.relative_bounding_box
                topLeft = (int(bBox.xmin * img_w), int(bBox.ymin * img_h))
                botRight = (int((bBox.xmin+bBox.width) * img_w), int((bBox.ymin+bBox.height) * img_h))
                faceBoundBoxes.append((topLeft,botRight))

        return faceBoundBoxes

    def processAndMarkImage(self, image):
        results = self.process(image)
        if results.detections != None:
            for det in results.detections:
                self.mp_drawing.draw_detection(image, det)

        return image
        

    def process(self, image):
        return self.face_detection.process(image)
#
# Class to work with the media-pipe body landmarks
#
class MyPoseToy:
    import mediapipe as mp
    def __init__(self, still=False, model_complexity=1, smooth=True, enable_segmentation=False, tol1=0.5, tol2=0.5):
        self.myPose = self.mp.solutions.pose.Pose(static_image_mode=still, 
                                                  model_complexity = 1,
                                                  smooth_landmarks = smooth,
                                                  enable_segmentation = enable_segmentation,
                                                  smooth_segmentation = smooth, 
                                                  min_detection_confidence = tol1,
                                                  min_tracking_confidence = tol2)
    
    def marks(self, image):
        results = self.myPose.process(image)
        poseLandmarks = []
        if results.pose_landmarks:
            img_h, img_w, _ = image.shape
            for lm in results.pose_landmarks.landmark:
                p = (int(lm.x * img_w), int(lm.y * img_h))
                poseLandmarks.append(p)
        
        return poseLandmarks
#
# Class to work with the media-pipe face mesh
#
class MyFaceMeshToy:
    import mediapipe as mp
    def __init__(self, still=False, numFaces=3, tol1=0.5, tol2=0.5, drawMesh=True):
        #Media pipe inits
        self.face_mesh = self.mp.solutions.face_mesh.FaceMesh(
            static_image_mode = still,
            max_num_faces = numFaces,                                           
            min_detection_confidence=tol1,
            min_tracking_confidence=tol2)

        self.drawer = self.mp.solutions.drawing_utils
        self.drawing_spec = self.mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(40,224,40))
        self.draw = drawMesh

    def marks(self, image):
        img_h, img_w, _ = image.shape
        #global img_w
        #global img_h
        results = self.face_mesh.process(image)
        facesMeshLandmarks = []
        if results.multi_face_landmarks != None:
            for faceMesh in results.multi_face_landmarks:
                faceMeshLandmarks = []
                for lm in faceMesh.landmark:
                    loc = (int(lm.x*img_w), int(lm.y*img_h))
                    faceMeshLandmarks.append(loc)
                
                facesMeshLandmarks.append(faceMeshLandmarks)
                if self.draw == True:
                    self.drawLandmarks(image, faceMesh)
        
        return facesMeshLandmarks

    def process(self, image):
        results = self.face_mesh.process(image)
        
        return results

    def drawLandmarks(self, image, face_landmarks):
            self.drawer.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                #connections = self.mp.solutions.face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec = self.drawing_spec,
                connection_drawing_spec = self.drawing_spec
            )
