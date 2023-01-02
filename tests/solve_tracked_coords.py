import copy
import cv2
import logging
import argparse
import os
import stereo_vision.projection as proj
import pickle
import pandas as pd
import imageio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
    inputImagesFilepathPrefix,
    outputDirectory,
    projectionMatrix1Filepath,
    projectionMatrix2Filepath,
    coordinatesFilepath
):
    logging.info("solve_tracked_coords.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    timestamp_imageFilepath_list = TimestampAndImageFilepaths(inputImagesFilepathPrefix)

    # Load the projection matrices
    P1 = None
    P2 = None
    with open(projectionMatrix1Filepath, 'rb') as P1_file:
        P1 = pickle.load(P1_file)
    with open(projectionMatrix2Filepath, 'rb') as P2_file:
        P2 = pickle.load(P2_file)
    projection_matrices = [P1, P2]
    stereo_system = proj.StereoVisionSystem(projection_matrices)

    # Load the coordinates
    coords_df = pd.read_csv(coordinatesFilepath)

    annotated_images_list = []
    for timestamp, image_filepath in timestamp_imageFilepath_list:
        row = coords_df[coords_df['timestamp'] == timestamp]
        if len(row) != 1:
            raise ValueError(f"len(row) ({len(row)}) != 1 for timestamp '{timestamp}'")
        #logging.debug(f"row: {row}")
        coords = [(row.iloc[0]['x_1'], row.iloc[0]['y_1']), (row.iloc[0]['x_2'], row.iloc[0]['y_2'])]
        XYZ = stereo_system.SolveXYZ(coords)
        image = cv2.imread(image_filepath)
        annotated_img = copy.deepcopy(image)
        uv = coords[0]
        cv2.putText(annotated_img, "({:.2f}, {:.2f}, {:.2f})".format(XYZ[0], XYZ[1], XYZ[2]), (round(uv[0]) + 10, round(uv[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(outputDirectory, timestamp + ".png"), annotated_img)
        annotated_images_list.append(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))  # imageio expects RGB images

    # Build an animated gif
    animated_gif_filepath = os.path.join(outputDirectory, "animation.gif")
    imageio.mimsave(animated_gif_filepath, annotated_images_list)

def TimestampAndImageFilepaths(images_filepath_prefix):
    extensions = ['.PNG']
    timestamp_imageFilepath_list = []
    images_directory = os.path.dirname(images_filepath_prefix)
    filepaths_in_directory = [os.path.join(images_directory, f) for f in os.listdir(images_directory) \
                              if os.path.isfile(os.path.join(images_directory, f))]

    image_filepaths = [filepath for filepath in filepaths_in_directory \
                       if filepath.upper()[-4:] in extensions
                       and filepath.startswith(images_filepath_prefix)]
    for filepath in image_filepaths:
        timestamp = filepath[len(images_filepath_prefix): -4]
        timestamp_imageFilepath_list.append((timestamp, filepath))
    # Sort by increasing timestamp
    timestamp_imageFilepath_list.sort(key=lambda x: x[0])
    #logging.debug(f"timestamp_imageFilepath_list = {timestamp_imageFilepath_list}")
    return timestamp_imageFilepath_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputImagesFilepathPrefix', help="The filepath prefix of the input images. Default: './output_track_red_square/stereo_'",
                        default='./output_track_red_square/stereo_')
    parser.add_argument('--outputDirectory', help="The output directory. Defaut: './output_solve_tracked_coords'",
                        default='./output_solve_tracked_coords')
    parser.add_argument('--projectionMatrix1Filepath', help="Filepath of the projection matrix 1. Defualt: './output_calibrate_system/camera1.projmtx'",
                        default="./output_calibrate_system/camera1.projmtx")
    parser.add_argument('--projectionMatrix2Filepath', help="Filepath of the projection matrix 2. Default: './output_calibrate_system/camera2.projmtx'",
                        default="./output_calibrate_system/camera2.projmtx")
    parser.add_argument('--coordinatesFilepath', help="The filepath for the tracked coordinates. Default: './output_track_red_square/red_square_coordinates.csv'",
                        default="./output_track_red_square/red_square_coordinates.csv")
    args = parser.parse_args()

    main(
        args.inputImagesFilepathPrefix,
        args.outputDirectory,
        args.projectionMatrix1Filepath,
        args.projectionMatrix2Filepath,
        args.coordinatesFilepath
    )