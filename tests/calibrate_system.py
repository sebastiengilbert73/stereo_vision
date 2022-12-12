import cv2
import logging
import argparse
import ast
import os
import camera_distortion_calibration.checkerboard as checkerboard
import camera_distortion_calibration.radial_distortion as radial_dist
import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
import math

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

    camera1_image_to_intersectionsList = InteractivelyFilterBadPoints(camera1_filepath_to_z, checkerboard_intersections)

def InteractivelyFilterBadPoints(cameraFilepath_to_z, checkerboard_intersections):
    imageFilepath_to_intersectionsList = {}
    for image_filepath, distance in camera1_filepath_to_z.items():
        user_is_satisfied = False
        while not user_is_satisfied:
            image = cv2.imread(image_filepath)
            annotated_img = copy.deepcopy(image)
            intersections_list = checkerboard_intersections.FindIntersections(image)
            logging.info(f"len(intersections_list) = {len(intersections_list)}")
            for pt_ndx in range(len(intersections_list)):
                p = intersections_list[pt_ndx]
                color = ((pt_ndx * 17)%256, (pt_ndx * 117)%256, (pt_ndx*1117)%256)
                cv2.circle(annotated_img, (round(p[0]), round(p[1])), 3, color, thickness=2)
                cv2.circle(annotated_img, (round(p[0]), round(p[1])), 5, (255, 255, 255), thickness=1)
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0]), round(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0]), round(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)


            input_is_accepted = False
            while not input_is_accepted:
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                print("Enter outlier indices. Ex: [0, 21, 7]. For an empty list, press Enter.\nThen, close the image window.")
                plt.imshow(annotated_img_rgb)
                plt.show()
                outlier_indices_str = input()
                outlier_indices = []
                if outlier_indices_str != '':
                    try:
                        print(f"outlier_indices_str = {outlier_indices_str}")
                        outlier_indices = ast.literal_eval(outlier_indices_str)
                        if type(outlier_indices) is list:
                            input_is_accepted = True
                        else:
                            print("Please enter indices again.\n")
                    except Exception as e:
                        print(f"Caught exception '{e}'. Please enter indices again.\n")
                else:
                    input_is_accepted = True
                plt.close()
            # Filter bad points
            filtered_points = []
            for pt_ndx in range(len(intersections_list)):
                if pt_ndx not in outlier_indices:
                    filtered_points.append(intersections_list[pt_ndx])

            annotated_img = copy.deepcopy(image)
            for pt_ndx in range(len(filtered_points)):
                p = filtered_points[pt_ndx]
                color = ((pt_ndx * 17)%256, (pt_ndx * 117)%256, (pt_ndx*1117)%256)
                cv2.circle(annotated_img, (round(p[0]), round(p[1])), 3, color, thickness=2)
                cv2.circle(annotated_img, (round(p[0]), round(p[1])), 5, (255, 255, 255), thickness=1)
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0]), round(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0]), round(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            print("If there is an error (ignore duplicates), type 'n [Enter]' and close the image.\nOtherwise, just close the image and type Enter.")
            plt.imshow(annotated_img_rgb)
            plt.show()
            user_response = input()
            if user_response.upper() != 'N':
                user_is_satisfied = True
            plt.close()
        logging.info(f"Before RemoveDuplicates(): len(filtered_points) = {len(filtered_points)}")
        filtered_points = RemoveDuplicates(filtered_points)
        logging.info(f"After RemoveDuplicates(): len(filtered_points) = {len(filtered_points)}")
        imageFilepath_to_intersectionsList[image_filepath] = filtered_points
    return imageFilepath_to_intersectionsList

def RemoveDuplicates(points_list, threshold_in_pixels=5):
    no_duplicates_points_list = []
    duplicate_indices_list = []
    for pt_ndx in range(len(points_list)):
        candidate_pt = points_list[pt_ndx]
        for neighbor_ndx in range(pt_ndx + 1, len(points_list)):
            neighbor_pt = points_list[neighbor_ndx]
            distance = math.sqrt((candidate_pt[0] - neighbor_pt[0])**2 + (candidate_pt[1] - neighbor_pt[1])**2)
            if distance < threshold_in_pixels:
                duplicate_indices_list.append(neighbor_ndx)
    for pt_ndx in range(len(points_list)):
        if pt_ndx not in duplicate_indices_list:
            no_duplicates_points_list.append(points_list[pt_ndx])
    return no_duplicates_points_list

if __name__ == '__main__':
    main()