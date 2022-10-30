import cv2

class Grabber():
    def __init__(self, camera_captures_list):
        self.camera_captures_list = camera_captures_list

    def Grab(self):
        grabbed_images = []
        for camera_capture in self.camera_captures_list:
            ret_val, image = camera_capture.read()
            if ret_val == True:
                grabbed_images.append(image)
            else:
                print("Grabber.Grab(): Could not grab an image")
                grabbed_images.append(None)
        return grabbed_images