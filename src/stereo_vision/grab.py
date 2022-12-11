import cv2

class Grabber():
    def __init__(self, cameraCaptures_id_list, camera_id_backup_list):
        self.cameraCaptures_id_list = cameraCaptures_id_list
        self.camera_id_backup_list = camera_id_backup_list

    def Grab(self):
        grabbed_images = []
        for camera_capture, id in self.cameraCaptures_id_list:
            ret_val, image = camera_capture.read()
            if ret_val == True:
                grabbed_images.append(image)
            else:
                print("Grabber.Grab(): Could not grab an image")
                grabbed_images.append(None)
                self.RestartCameras()
        return grabbed_images

    def RestartCameras(self):
        print(f"Grabber.RestartCameras()")
        new_camera_backup_id_list = []
        for camera_capture, camera_id in self.cameraCaptures_id_list:
            new_camera_backup_id_list.append(camera_id)
            camera_capture.release()
        self.cameraCaptures_id_list = []
        for camera_id in self.camera_id_backup_list:
            video_capture = cv2.VideoCapture(camera_id)
            self.cameraCaptures_id_list.append((video_capture, camera_id))
        self.camera_id_backup_list = new_camera_backup_id_list
