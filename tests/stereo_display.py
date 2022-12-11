import cv2
import logging
import argparse
import ast
from datetime import datetime
import os
from stereo_vision.grab import Grabber

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        imageSizeHW,
        cameraIDList,
        cameraIDBackupList,
        capturesPeriod,
        outputDirectory
):
    logging.info("stereo_display.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    video_captures_id_list = []
    for camera_id in cameraIDList:
        video_capture = cv2.VideoCapture(camera_id)
        if video_capture.isOpened():
            video_captures_id_list.append((video_capture, camera_id))
    logging.debug(f"type(video_captures_id_list) = {type(video_captures_id_list)}")
    grabber = Grabber(video_captures_id_list, cameraIDBackupList)

    camera_names = []
    for camera_id in cameraIDList:
        camera_names.append('camera_' + str(camera_id))

    while True:
        images = grabber.Grab()
        key = None
        for camera_name_ndx in range(len(camera_names)):
            if images[camera_name_ndx] is not None:
                cv2.imshow(camera_names[camera_name_ndx], images[camera_name_ndx])

                key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Save the image
            dateTime_obj = datetime.now()
            timestamp = dateTime_obj.strftime("%Y%m%d-%H:%M:%S")
            image_filepath0 = os.path.join(outputDirectory, camera_names[0] + '_' + timestamp) + ".png"
            logging.info(f"Saving {image_filepath0}")
            cv2.imwrite(image_filepath0, images[0])

            image_filepath1 = os.path.join(outputDirectory, camera_names[1] + '_' + timestamp) + ".png"
            logging.info(f"Saving {image_filepath1}")
            cv2.imwrite(image_filepath1, images[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageSizeHW', help="The image resize (Height, Width), if desired. Default: 'None'",
                        default='None')
    parser.add_argument('--cameraIDList', help="The list of camera ID. Default: '[1, 2]'", default='[1, 2]')
    parser.add_argument('--cameraIDBackupList', help="The list of backup camera ID. Default: '[3, 4]'", default='[3, 4]')
    parser.add_argument('--capturesPeriod', help="The number of captures used to compute frame rate. Defaut: 50",
                        type=int, default=50)
    parser.add_argument('--outputDirectory', help="The output directory. Defaut: './stereo_display_output'",
                        default='./stereo_display_output')
    args = parser.parse_args()

    if args.imageSizeHW.upper() == 'NONE':
        imageSizeHW = None
    else:
        imageSizeHW = ast.literal_eval(args.imageSizeHW)

    cameraIDList = ast.literal_eval(args.cameraIDList)
    cameraIDBackupList = ast.literal_eval(args.cameraIDBackupList)
    main(
        imageSizeHW,
        cameraIDList,
        cameraIDBackupList,
        args.capturesPeriod,
        args.outputDirectory,
    )
