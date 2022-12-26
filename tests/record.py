import cv2
import logging
import argparse
import ast
from datetime import datetime, timedelta
import os
from stereo_vision.grab import Grabber
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
    cameraIDList,
    outputDirectory,
    recordTime
):
    logging.info(f"record.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    video_captures_id_list = []
    for camera_id in cameraIDList:
        video_capture = cv2.VideoCapture(camera_id)
        if video_capture.isOpened():
            video_captures_id_list.append((video_capture, camera_id))
    logging.debug(f"type(video_captures_id_list) = {type(video_captures_id_list)}")
    grabber = Grabber(video_captures_id_list)

    camera_names = []
    for camera_id in cameraIDList:
        camera_names.append('camera_' + str(camera_id))

    start_time = datetime.now()
    logging.debug(f"start_time = {start_time}")
    current_time = datetime.now()
    while current_time < start_time + timedelta(seconds=recordTime):
        images = grabber.Grab()
        for image_ndx in range(len(images)):
            img_filepath = os.path.join(outputDirectory, camera_names[image_ndx] + "_" + \
                                        str(current_time).replace(' ', '_').replace(':', '') + '.png')
            cv2.imwrite(img_filepath, images[image_ndx])
        current_time = datetime.now()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameraIDList', help="The list of camera ID. Default: '[1, 2]'", default='[1, 2]')
    parser.add_argument('--outputDirectory', help="The output directory. Defaut: './output_record'",
                        default='./output_record')
    parser.add_argument('--recordTime', help="Record time, in seconds. Default: 15.0", type=float, default=15.0)
    args = parser.parse_args()
    cameraIDList = ast.literal_eval(args.cameraIDList)
    main(
        cameraIDList,
        args.outputDirectory,
        args.recordTime
    )