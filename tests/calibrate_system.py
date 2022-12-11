import cv2
import logging
import argparse
import ast
import os
import camera_distortion_calibration.checkerboard as checkerboard
import camera_distortion_calibration.radial_distortion as radial_dist
import copy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

camera1_filepath_to_z = {
    "calibration_images/camera_1_60cm.png": 60,
    "calibration_images/camera_1_70cm.png": 70,
    "calibration_images/camera_1_80cm.png": 80,
    "calibration_images/camera_1_90cm.png": 90,
    "calibration_images/camera_1_100cm.png": 100,
    "calibration_images/camera_1_110cm.png": 110,
    "calibration_images/camera_1_120cm.png": 120
}

camera2_filepath_to_z = {
    "calibration_images/camera_2_60cm.png": 60,
    "calibration_images/camera_2_70cm.png": 70,
    "calibration_images/camera_2_80cm.png": 80,
    "calibration_images/camera_2_90cm.png": 90,
    "calibration_images/camera_2_100cm.png": 100,
    "calibration_images/camera_2_110cm.png": 110,
    "calibration_images/camera_2_120cm.png": 120
}

camera1_radial_distortion = "radial_distortion/calibration_left.pkl"
camera2_radial_distortion = "radial_distortion/calibration_right.pkl"

output_directory = "./output_calibrate_system"

def main():
    logging.info(f"calibrate_system.main()")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    checkerboard_intersections = checkerboard.CheckerboardIntersections(
        adaptive_threshold_block_side=19,
        adaptive_threshold_bias=-5,
        correlation_threshold=0.65,
        debug_directory=output_directory
    )

    for image_filepath, distance in camera1_filepath_to_z.items():
        image = cv2.imread(image_filepath)
        annotated_img = copy.deepcopy(image)
        intersections_list = checkerboard_intersections.FindIntersections(image)
        logging.debug(f"len(intersections_list) = {len(intersections_list)}")
        for p in intersections_list:
            cv2.circle(annotated_img, (round(p[0]), round(p[1])), 3, (0, 255, 0), thickness=2)
        cv2.imshow("Intersections", annotated_img)
        cv2.waitKey()

if __name__ == '__main__':
    main()