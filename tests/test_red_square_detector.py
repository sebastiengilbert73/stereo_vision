import cv2
import numpy as np
import red_square
import os

def main():
    output_directory = "./output_test_red_square_detector"
    input_img_filepath = "./output_record_redSquare/camera_2_2022-12-29_092012.303990.png"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    detector = red_square.Detector(
        blue_delta=15,
        blue_mask_dilation_kernel_size=45,
        red_delta=50,
        red_mask_dilation_kernel_size=13,
        debug_directory=output_directory)
    image = cv2.imread(input_img_filepath)
    center_of_mass = detector.Detect(image)
    print(f"center_of_mass = {center_of_mass}")


if __name__ == '__main__':
    main()