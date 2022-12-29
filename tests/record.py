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
    recordTime,
    warmupTime,
    grabDelays,
    exposure
):
    logging.info(f"record.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    video_captures_id_list = []
    for camera_id in cameraIDList:
        video_capture = cv2.VideoCapture(camera_id)
        if video_capture.isOpened():
            video_capture.set(cv2.CAP_PROP_FPS, 30)
            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            video_capture.set(cv2.CAP_PROP_AUTO_WB, 0)
            video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            video_capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
            video_captures_id_list.append((video_capture, camera_id))
    logging.debug(f"type(video_captures_id_list) = {type(video_captures_id_list)}")
    grabber = Grabber(video_captures_id_list, grabDelays)

    camera_names = []
    for camera_id in cameraIDList:
        camera_names.append('camera_' + str(camera_id))

    warmup_is_over = False
    start_time = datetime.now()
    logging.debug(f"start_time = {start_time}")
    current_time = datetime.now()
    while current_time < start_time + timedelta(seconds=warmupTime) + timedelta(seconds=recordTime):
        images = grabber.Grab()
        if not warmup_is_over and current_time >= start_time + timedelta(seconds=warmupTime):
            warmup_is_over = True
            logging.info("Warmup is over!")
        if warmup_is_over:
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
    parser.add_argument('--warmupTime', help="Warmup time, in seconds, where no images are recorded. Default: 5.0", type=float, default=5.0)
    parser.add_argument('--grabDelays', help="The delays, in seconds, to compensate for cameras grab speed differences. The first camera grabs without delay, and the other ones are delayed. Default: '[0]'", default='[0]')
    parser.add_argument('--exposure', help="The value for the parameter CAP_PROP_EXPOSURE. The meaning depends on the camera model. Default: 400", type=float, default=400)
    args = parser.parse_args()
    cameraIDList = ast.literal_eval(args.cameraIDList)
    grabDelays = ast.literal_eval(args.grabDelays)
    if len(grabDelays) != len(cameraIDList) - 1:
        raise ValueError(f"len(grabDelays) ({len(grabDelays)}) != len(cameraIDList) - 1 ({len(cameraIDList) - 1})")
    main(
        cameraIDList,
        args.outputDirectory,
        args.recordTime,
        args.warmupTime,
        grabDelays,
        args.exposure
    )