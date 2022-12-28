#
#

#
# Get the mesh of face points from the image
#
class MyWebCam:
    import cv2
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camNum=-1
        self.cam = None

    def open(self, camera_number=0):
        if not self.cam == None:
            if camera_number == self.camNum:
                return True
            self.cam.release()
            self.cam=None

        self.cam = self.cv2.VideoCapture(camera_number)
        self.camNum = camera_number
        if self.cam.isOpened():
            self.cam.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cam.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cam.set(self.cv2.CAP_PROP_FPS, self.fps)
            return True
        else:
            self.cam.release()
            self.cam=None
            self.camNum=-1
            return False
    def isOpened(self):
        if self.cam == None:
            return False

        return self.cam.isOpened()

    def getFrame(self):
        success, image = self.cam.read()
        return success, image

    def close(self):
        if self.cam != None:
            self.cam.release()
            self.cam = None
            self.camNum = -1
        