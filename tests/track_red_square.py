import cv2
import logging
import argparse
import ast
import os
import numpy as np
import red_square
import copy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        inputImagesFilepathPrefix,
        outputDirectory,
        cameraIDList,
        redSquareDetectorBlueDelta,
        redSquareDetectorBlueDilationSize,
        redSquareDetectorRedDelta,
        redSquareDetectorRedDilationSize
):
    logging.info("track_red_square.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    timestamp_to_imageFilepathsList = TimestampToImageFilepathsList(cameraIDList, inputImagesFilepathPrefix)
    #logging.debug(f"timestamp_to_imageFilepathsList = {timestamp_to_imageFilepathsList}")

    red_square_detector = red_square.Detector(
        blue_delta=redSquareDetectorBlueDelta,
        blue_mask_dilation_kernel_size=redSquareDetectorBlueDilationSize,
        red_delta=redSquareDetectorRedDelta,
        red_mask_dilation_kernel_size=redSquareDetectorRedDilationSize,
        debug_directory=None
    )

    with open(os.path.join(outputDirectory, "red_square_coordinates.csv"), 'w') as coords_file:
        header = "timestamp"
        for camera_ID_ndx in range(1, len(cameraIDList) + 1):
            header += f",x_{str(camera_ID_ndx)},y_{str(camera_ID_ndx)}"
        header += "\n"
        coords_file.write(header)
        timestamps = list(timestamp_to_imageFilepathsList.keys())
        #logging.debug(f"timestamps = {timestamps}")
        timestamps.sort()
        #for timestamp, image_filepaths_list in timestamp_to_imageFilepathsList.items():
        for timestamp in timestamps:
            image_filepaths_list = timestamp_to_imageFilepathsList[timestamp]
            coords_file.write(timestamp)
            images = []
            for image_filepath in image_filepaths_list:
                image = cv2.imread(image_filepath)
                images.append(image)
            img_shapeHWC = images[0].shape
            mosaic_img = np.zeros((img_shapeHWC[0], len(images) * img_shapeHWC[1], img_shapeHWC[2]), dtype=np.uint8)
            for image_ndx in range(len(images)):
                image = images[image_ndx]
                annotated_img = copy.deepcopy(image)
                center = red_square_detector.Detect(image)
                center_rounded = (round(center[0]), round(center[1]))
                cv2.line(annotated_img, (center_rounded[0] - 5, center_rounded[1]), (center_rounded[0] + 5, center_rounded[1]), (255, 0, 0),
                         thickness=3)
                cv2.line(annotated_img, (center_rounded[0], center_rounded[1] - 5),
                         (center_rounded[0], center_rounded[1] + 5), (255, 0, 0),
                         thickness=3)
                mosaic_img[:, image_ndx * img_shapeHWC[1]: (image_ndx + 1) * img_shapeHWC[1], :] = annotated_img
                #camera_ID = camera_ID[image_ndx]
                coords_file.write(f",{center[0]},{center[1]}")
            coords_file.write("\n")
            mosaic_img_filepath = os.path.join(outputDirectory, 'stereo_' + timestamp + '.png')
            cv2.imwrite(mosaic_img_filepath, mosaic_img)

def TimestampToImageFilepathsList(camera_ID_list, images_filepath_prefix):
    extensions = ['.PNG']
    images_directory = os.path.dirname(images_filepath_prefix)
    timestamp_to_imageFilepathsList = {}
    filepaths_in_directory = [os.path.join(images_directory, f) for f in os.listdir(images_directory) \
                              if os.path.isfile(os.path.join(images_directory, f))]

    image_filepaths = [filepath for filepath in filepaths_in_directory \
                       if filepath.upper()[-4:] in extensions]

    first_camera_image_filepaths = [filepath for filepath in image_filepaths \
                                    if filepath.startswith(images_filepath_prefix + str(camera_ID_list[0]) + '_')]
    first_camera_ID = camera_ID_list[0]
    first_camera_prefix = images_filepath_prefix + str(first_camera_ID) + '_'
    #print(f"first_camera_image_filepaths = {first_camera_image_filepaths}")
    #print(f"len(first_camera_image_filepaths) = {len(first_camera_image_filepaths)}")
    for filepath in first_camera_image_filepaths:
        timestamp = filepath[len(first_camera_prefix): -4]
        #logging.debug(f"timestamp = {timestamp}")
        image_filepaths_list = [filepath]
        for other_camera_ID_ndx in range(1, len(camera_ID_list)):
            other_camera_ID = camera_ID_list[other_camera_ID_ndx]
            other_camera_filepath = images_filepath_prefix + str(other_camera_ID) + '_' + timestamp + '.png'
            #logging.debug(f"other_camera_filepath = {other_camera_filepath}")
            if not os.path.exists(other_camera_filepath):
                raise FileNotFoundError(f"TimestampToImageFilepathsList(): Could not find file '{other_camera_filepath}'")
            image_filepaths_list.append(other_camera_filepath)
        timestamp_to_imageFilepathsList[timestamp] = image_filepaths_list
    return timestamp_to_imageFilepathsList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputImagesFilepathPrefix', help="The filepath prefix of the input images")
    parser.add_argument('--outputDirectory', help="The output directory. Defaut: './output_track_red_square'",
                        default='./output_track_red_square')
    parser.add_argument('--cameraIDList', help="The list of camera ID. Default: '[1, 2]'", default='[1, 2]')
    parser.add_argument('--redSquareDetectorBlueDelta', help="For the red square detector, the blue delta. Default: 15", type=int, default=15)
    parser.add_argument('--redSquareDetectorBlueDilationSize', help="For the red square detector, the blue dilation size. Default: 45", type=int, default=45)
    parser.add_argument('--redSquareDetectorRedDelta', help="For the red square detector, the red delta. Default: 70", type=int, default=70)
    parser.add_argument('--redSquareDetectorRedDilationSize', help="For the red square detector, the red dilation size. Default: 13", type=int, default=13)
    args = parser.parse_args()
    cameraIDList = ast.literal_eval(args.cameraIDList)
    main(
        args.inputImagesFilepathPrefix,
        args.outputDirectory,
        cameraIDList,
        args.redSquareDetectorBlueDelta,
        args.redSquareDetectorBlueDilationSize,
        args.redSquareDetectorRedDelta,
        args.redSquareDetectorRedDilationSize
    )