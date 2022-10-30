import cv2
import logging
import argparse
import ast
from stereo_vision.grab import Grabber

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        imageSizeHW,
        cameraIDList,
        capturesPeriod
):
    logging.info("stereo_display.main()")

    video_captures_list = []
    for camera_id in cameraIDList:
        video_capture = cv2.VideoCapture(camera_id)
        if video_capture.isOpened():
            video_captures_list.append(video_capture)
    grabber = Grabber(video_captures_list)

    camera_names = []
    for camera_id in cameraIDList:
        camera_names.append('camera_' + str(camera_id))

    while True:
        images = grabber.Grab()
        key = None
        for camera_name_ndx in range(len(camera_names)):
            if images[camera_name_ndx] is not None:
                cv2.imshow(camera_names[camera_name_ndx], images[camera_name_ndx])

                key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageSizeHW', help="The image resize (Height, Width), if desired. Default: 'None'",
                        default='None')
    parser.add_argument('--cameraIDList', help="The list of camera ID. Default: '[1, 2]'", default='[1, 2]')
    parser.add_argument('--capturesPeriod', help="The number of captures used to compute frame rate. Defaut: 50",
                        type=int, default=50)
    args = parser.parse_args()

    if args.imageSizeHW.upper() == 'NONE':
        imageSizeHW = None
    else:
        imageSizeHW = ast.literal_eval(args.imageSizeHW)

    cameraIDList = ast.literal_eval(args.cameraIDList)
    main(
        imageSizeHW,
        cameraIDList,
        args.capturesPeriod
    )