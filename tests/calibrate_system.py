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
import pickle
import pandas as pd
from stereo_vision.projection import ProjectionMatrix

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

camera1_filepath_to_z = {
    "calibration_images/camera_1_60cm.png": -60,
    "calibration_images/camera_1_70cm.png": -70,
    "calibration_images/camera_1_80cm.png": -80,
    "calibration_images/camera_1_90cm.png": -90,
    "calibration_images/camera_1_100cm.png": -100,
    "calibration_images/camera_1_110cm.png": -110,
    "calibration_images/camera_1_120cm.png": -120
}

camera2_filepath_to_z = {
    "calibration_images/camera_2_60cm.png": -60,
    "calibration_images/camera_2_70cm.png": -70,
    "calibration_images/camera_2_80cm.png": -80,
    "calibration_images/camera_2_90cm.png": -90,
    "calibration_images/camera_2_100cm.png": -100,
    "calibration_images/camera_2_110cm.png": -110,
    "calibration_images/camera_2_120cm.png": -120
}

camera1_radial_distortion_filepath = "radial_distortion/calibration_left.pkl"
camera2_radial_distortion_filepath = "radial_distortion/calibration_right.pkl"
grid_shapeHW = (6, 6)
calibration_pattern_xy_filepath = 'calibration_images/calibration_pattern_xy.csv'

output_directory = "./output_calibrate_system"

def main():
    logging.info(f"calibrate_system.main()")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    """checkerboard_intersections = checkerboard.CheckerboardIntersections(
        adaptive_threshold_block_side=19,
        adaptive_threshold_bias=-5,
        correlation_threshold=0.6,
        debug_directory=output_directory
    )

    camera1_imageFilepath_to_intersectionsList = InteractivelyFilterBadPoints(camera1_filepath_to_z, checkerboard_intersections)
    with open(os.path.join(output_directory, "camera1_intersections.pkl"), 'wb') as intersections_file:
        pickle.dump(camera1_imageFilepath_to_intersectionsList, intersections_file, pickle.HIGHEST_PROTOCOL)
    
    camera2_imageFilepath_to_intersectionsList = InteractivelyFilterBadPoints(camera2_filepath_to_z,
                                                                              checkerboard_intersections)
    with open(os.path.join(output_directory, "camera2_intersections.pkl"), 'wb') as intersections_file:
        pickle.dump(camera2_imageFilepath_to_intersectionsList, intersections_file, pickle.HIGHEST_PROTOCOL)
    """
    camera1_imageFilepath_to_intersectionsList = {}
    with open(os.path.join(output_directory, "camera1_intersections.pkl"), 'rb') as obj_file:
        camera1_imageFilepath_to_intersectionsList = pickle.load(obj_file)
    camera2_imageFilepath_to_intersectionsList = {}
    with open(os.path.join(output_directory, "camera2_intersections.pkl"), 'rb') as obj_file:
        camera2_imageFilepath_to_intersectionsList = pickle.load(obj_file)

    # Undistort the points
    camera1_radial_distortion = None
    with open(camera1_radial_distortion_filepath, 'rb') as obj_file:
        camera1_radial_distortion = pickle.load(obj_file)
    camera2_radial_distortion = None
    with open(camera2_radial_distortion_filepath, 'rb') as obj_file:
        camera2_radial_distortion = pickle.load(obj_file)

    camera1_imageFilepath_to_undistorted_intersectionsList = UndistortIntersections(
        camera1_imageFilepath_to_intersectionsList, camera1_radial_distortion
    )
    camera2_imageFilepath_to_undistorted_intersectionsList = UndistortIntersections(
        camera2_imageFilepath_to_intersectionsList, camera2_radial_distortion
    )

    # Sort the points, per lines
    camera1_imageFilepath_to_undistorted_intersectionsList = \
        SortPointsPerLine(camera1_imageFilepath_to_undistorted_intersectionsList,
                          camera1_radial_distortion,
                          grid_shapeHW)
    camera2_imageFilepath_to_undistorted_intersectionsList = \
        SortPointsPerLine(camera2_imageFilepath_to_undistorted_intersectionsList,
                          camera2_radial_distortion,
                          grid_shapeHW)

    # Load the calibration pattern (x, y) coordinates
    calibration_pattern_xy_df = pd.read_csv(calibration_pattern_xy_filepath)
    #print(f"calibration_patter_xy_df: \n{calibration_pattern_xy_df}")

    # Match the pixel points with the 3D points
    camera1_xy_XYZ_tuples = MatchPixelsWith3D(
        camera1_imageFilepath_to_undistorted_intersectionsList, camera1_filepath_to_z,
        calibration_pattern_xy_df
    )
    #logging.info(f"camera1_xy_XYZ_tuples = {camera1_xy_XYZ_tuples}")

    # Compute the projection matrix
    projection_mtx1 = ProjectionMatrix(camera1_xy_XYZ_tuples)
    logging.info(f"projection_mtx1.matrix = \n{projection_mtx1.matrix}")

    # Debug images
    for image_filepath, intersections_list in camera1_imageFilepath_to_undistorted_intersectionsList.items():
        annotated_img = cv2.imread(image_filepath)
        for point_ndx in range(len(intersections_list)):
            p = intersections_list[point_ndx]
            cv2.circle(annotated_img, (round(p[0]), round(p[1])), 3, (255, 0, 0), thickness=2)
            cv2.putText(annotated_img, str(point_ndx), (round(p[0]), round(p[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), thickness=1)

        # Projections
        Z = camera1_filepath_to_z[image_filepath]
        for point_ndx in range(len(calibration_pattern_xy_df)):
            XYZ = (calibration_pattern_xy_df.iloc[point_ndx].x, calibration_pattern_xy_df.iloc[point_ndx].y, Z)
            projected_p = projection_mtx1.Project(XYZ, must_round=True)
            cv2.circle(annotated_img, projected_p, 6, (0, 255, 255), thickness=2)

        annotated_img_filepath = os.path.join(output_directory, "calibrateSystem_main_" + os.path.basename(
            image_filepath) + "UndistortedIntersections.png")
        cv2.imwrite(annotated_img_filepath, annotated_img)

    for image_filepath, intersections_list in camera2_imageFilepath_to_undistorted_intersectionsList.items():
        annotated_img = cv2.imread(image_filepath)
        for point_ndx in range(len(intersections_list)):
            p = intersections_list[point_ndx]
            cv2.circle(annotated_img, (round(p[0]), round(p[1])), 3, (255, 0, 0), thickness=2)
            cv2.putText(annotated_img, str(point_ndx), (round(p[0]), round(p[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), thickness=1)
        annotated_img_filepath = os.path.join(output_directory, "calibrateSystem_main_" + os.path.basename(image_filepath) + "UndistortedIntersections.png")
        cv2.imwrite(annotated_img_filepath, annotated_img)



def InteractivelyFilterBadPoints(cameraFilepath_to_z, checkerboard_intersections):
    imageFilepath_to_intersectionsList = {}
    for image_filepath, distance in cameraFilepath_to_z.items():
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
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0] - 10), round(p[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0] - 10), round(p[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)


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
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0] - 10), round(p[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
                cv2.putText(annotated_img, str(pt_ndx), (round(p[0] - 10), round(p[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            print("If there is an error (ignore duplicates), type 'n [Enter]' and close the image.\nOtherwise, just close the image and type [Enter].")
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

def UndistortIntersections(
        imageFilepath_to_intersectionsList, radial_distortion
    ):
    imageFilepath_to_undistorted_intersectionsList = {}
    for image_filepath, intersections_list in imageFilepath_to_intersectionsList.items():
        undistorted_intersections_list = []
        for p in intersections_list:
            undistorted_p = radial_distortion.UndistortPoint(p, must_be_rounded=False)
            undistorted_intersections_list.append(undistorted_p)
        imageFilepath_to_undistorted_intersectionsList[image_filepath] = undistorted_intersections_list
    return imageFilepath_to_undistorted_intersectionsList

def SortPointsPerLine(imageFilepath_to_intersectionsList, radial_distortion, grid_shapeHW):
    imageFilepath_to_sortedIntersectionsList = {}
    for image_filepath, intersections_list in imageFilepath_to_intersectionsList.items():
        horizontal_lines, vertical_lines = radial_distortion.GroupCheckerboardPoints(
            intersections_list, grid_shapeHW
        )
        sorted_intersections_list = []
        for horizontal_line in horizontal_lines:
            horizontal_line = sorted(horizontal_line, key=lambda xy: xy[0])
            sorted_intersections_list += horizontal_line
        imageFilepath_to_sortedIntersectionsList[image_filepath] = sorted_intersections_list
    return imageFilepath_to_sortedIntersectionsList

def MatchPixelsWith3D(imageFilepath_to_pixelPointsList,
                      imageFilepath_to_z,
                      calibration_pattern_xy_df):
    xy_XYZ_tuples = []
    for image_filepath, xy_list in imageFilepath_to_pixelPointsList.items():
        logging.info(f"image_filepath: {image_filepath}")
        if not image_filepath in imageFilepath_to_z:
            raise ValueError(f"MatchPixelsWith3D(): image filepath '{image_filepath}' was not found in imageFilepath_to_z:\n{imageFilepath_to_z}")
        Z = imageFilepath_to_z[image_filepath]
        if len(xy_list) != len(calibration_pattern_xy_df):
            raise ValueError(f"len(xy_list) ({len(xy_list)}) != len(calibration_pattern_xy_df) ({len(calibration_pattern_xy_df)})")
        for point_ndx in  range(len(xy_list)):
            xy = xy_list[point_ndx]
            XY = (calibration_pattern_xy_df.iloc[point_ndx].x, calibration_pattern_xy_df.iloc[point_ndx].y)
            xy_XYZ_tuples.append((xy, (XY[0], XY[1], Z)))

    return xy_XYZ_tuples

if __name__ == '__main__':
    main()