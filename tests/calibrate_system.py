import cv2
import logging
import argparse
import ast
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

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

def main():
    logging.info(f"calibrate_system.main()")


if __name__ == '__main__':
    main()