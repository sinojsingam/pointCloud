# import laspy
# import sys
# import numpy as np
# from scipy.spatial import cKDTree

# import time

# import send_email
# import concurrent.futures
# from concurrent.futures import ProcessPoolExecutor
# import argparse
# import sys
# from pathlib import Path
# from typing import List, Optional
# from scipy.spatial import cKDTree




# def recursive_split(x_min, y_min, x_max, y_max, max_x_size, max_y_size):
#     """
#     This function recursively splits the spatial extent of the LAS file into smaller bounds 
#     until each sub-bound's size is below the specified maximum x and y sizes. 

#     It takes the minimum and maximum x and y coordinates of the current bound, along with the maximum sizes 
#     for both dimensions. It returns a list of tuples, where each tuple represents the coordinates of a sub-bound.
#     """
#     x_size = x_max - x_min
#     y_size = y_max - y_min

#     if x_size > max_x_size:
#         left = recursive_split(
#             x_min, y_min, x_min + (x_size // 2), y_max, max_x_size, max_y_size
#         )
#         right = recursive_split(
#             x_min + (x_size // 2), y_min, x_max, y_max, max_x_size, max_y_size
#         )
#         return left + right
#     elif y_size > max_y_size:
#         up = recursive_split(
#             x_min, y_min, x_max, y_min + (y_size // 2), max_x_size, max_y_size
#         )
#         down = recursive_split(
#             x_min, y_min + (y_size // 2), x_max, y_max, max_x_size, max_y_size
#         )
#         return up + down
#     else:
#         return [(x_min, y_min, x_max, y_max)]


# def tuple_size(string):
#     """
#     A helper function that parses a size argument given in the form "numberxnumber" (e.g., "50x64.17") 
#     and returns it as a tuple of floats. 
#     This is used to parse the command-line argument specifying the maximum size of each sub-bound.
#     """
#     try:
#         return tuple(map(float, string.split("x")))
#     except:
#         raise ValueError("Size must be in the form of numberxnumber eg: 50.0x65.14")


# def main():
#     """
#     It uses argparse to parse command-line arguments, 
#     including the input LAS file, output directory, the maximum size for sub-bounds, and an optional argument specifying 
#     the number of points to process in each iteration for efficiency.
#     """
#     parser = argparse.ArgumentParser(
#         "LAS recursive splitter", description="Splits a las file bounds recursively"
#     )
#     parser.add_argument("input_file")
#     parser.add_argument("output_dir")
#     parser.add_argument("size", type=tuple_size, help="eg: 50x64.17")
#     parser.add_argument("--points-per-iter", default=10**6, type=int)

#     args = parser.parse_args()
#     #Opens the input LAS file and calculates the sub-bounds 
#     with laspy.open(sys.argv[1]) as file:
#         sub_bounds = recursive_split(
#             file.header.x_min,
#             file.header.y_min,
#             file.header.x_max,
#             file.header.y_max,
#             args.size[0],
#             args.size[1],
#         )
#         #Prepares a list of LasWriter objects for writing the points to new files, initially filled with None.
#         writers: List[Optional[laspy.LasWriter]] = [None] * len(sub_bounds)
#         try:
#             R=0.5
#             count = 0
#             for points in file.chunk_iterator(args.points_per_iter):
#                 print(f"{count / file.header.point_count * 100}%")

#                 # For performance we need to use copy
#                 # so that the underlying arrays are contiguous
#                 x, y, z,nz = points.x.copy(), points.y.copy(), points.z.copy(), points['normal z'].copy()
#                 tree = cKDTree(np.c_[x, y, z, nz])
#                 #eigenvalues_list = []
#                 #point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
#                 points_3d = np.c_[x, y, z]
                
#                 translated_coords = geometricFeatures.translate_coords(points_3d)
#                 linearity_list = geometricFeatures.compute_linearity(translated_coords, tree, R)
#                 count += len(points)
#             print(f"{count / file.header.point_count * 100}%")
#         finally:
#             #Closes all open LasWriter objects to ensure all data is written and files are properly closed.
#             for writer in writers:
#                 if writer is not None:
#                     writer.close()


# if __name__ == "__main__":
#     main()
    




####
    
import laspy
import numpy as np
from scipy.spatial import cKDTree
from numpy.linalg import eigh
import calculateFeatures
import geometricFeatures
import os

def process_chunk(chunk, radius):
    x, y, z = chunk.x, chunk.y, chunk.z
    points = np.c_[chunk.x, chunk.y, chunk.z]
    tree = cKDTree(points)
    eigenvalues_list = np.zeros((len(x), 3))
    #omnivariance_list = np.zeros(len(x))
    features_list = []

    for idx in range(len(x)):
        print(x[idx])
        indices = tree.query_ball_point([x[idx], y[idx], z[idx]], r=radius)
        if len(indices) > 2:
            
            neighbors = np.c_[x[indices], y[indices], z[indices]]
            covariance_matrix = calculateFeatures.compute_covariance_matrix(neighbors)
            eigenvalues = calculateFeatures.compute_eigenvalues(covariance_matrix)
            eigenvalues_list[idx] = eigenvalues
            #omnivariance_list[idx] = compute_omnivariance(eigenvalues)

            lambda_1, lambda_2, lambda_3 = eigenvalues
            features = [
                calculateFeatures.compute_omnivariance(eigenvalues),
                calculateFeatures.compute_eigenentropy(eigenvalues),
                calculateFeatures.compute_anisotropy(lambda_1, lambda_2, lambda_3),
                calculateFeatures.compute_linearity(lambda_1, lambda_2, lambda_3),
                calculateFeatures.compute_planarity(lambda_2, lambda_3),
                calculateFeatures.compute_curvature(lambda_1, lambda_2, lambda_3),
                calculateFeatures.compute_sphericity(lambda_1, lambda_3)
            ]
        else:
            eigenvalues_list[idx] = [np.nan, np.nan, np.nan]
            #omnivariance_list[idx] = np.nan
            features = [np.nan] * 6
        features_list.append(features)

    features_array = np.array(features_list)
    return eigenvalues_list, 'omnivariance_list', features_array

def main(input_file, output_file, radius):
    with laspy.open(input_file) as infile:
        header = infile.header
        # header.add_extra_dim(name="eigenvalue_1", type=np.float64)
        # header.add_extra_dim(name="eigenvalue_2", type=np.float64)
        # header.add_extra_dim(name="eigenvalue_3", type=np.float64)
        header.add_extra_dim(laspy.ExtraBytesParams(name="omnivariance", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="eigenentropy", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="anisotropy", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="linearity", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="planarity", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="curvature", type=np.float64))
        header.add_extra_dim(laspy.ExtraBytesParams(name="sphericity", type=np.float64))

        all_eigenvalues = np.zeros((header.point_count, 3))
        #all_omnivariance = np.zeros(header.point_count)
        all_features = np.zeros((header.point_count, 6))
        points_count = int(header.point_count)
        start_index = 0

        for chunk in infile.chunk_iterator(1000):
            eigenvalues_chunk, features_chunk = process_chunk(chunk, radius)
            end_index = start_index + len(eigenvalues_chunk)
            all_eigenvalues[start_index:end_index] = eigenvalues_chunk
            #all_omnivariance[start_index:end_index] = omnivariance_chunk
            all_features[start_index:end_index] = features_chunk
            start_index = end_index

        with laspy.create(output_file, header=header) as outfile:
            outfile.points = infile.read()
            # outfile.point.record['eigenvalue_1'] = all_eigenvalues[:, 0]
            # outfile.point.record['eigenvalue_2'] = all_eigenvalues[:, 1]
            # outfile.point.record['eigenvalue_3'] = all_eigenvalues[:, 2]
            outfile.point.record['omnivariance'] = all_features[:, 0].astype(np.float64)
            outfile.point.record['eigenentropy'] = all_features[:, 1].astype(np.float64)
            outfile.point.record['anisotropy'] = all_features[:, 2].astype(np.float64)
            outfile.point.record['linearity'] = all_features[:, 3].astype(np.float64)
            outfile.point.record['planarity'] = all_features[:, 4].astype(np.float64)
            outfile.point.record['curvature'] = all_features[:, 5].astype(np.float64)
            outfile.point.record['sphericity'] = all_features[:, 6].astype(np.float64)

if __name__ == "__main__":
    input_las_path = 'working/isolated_features/car_training.las'#sys.argv[1]
    subfolder = "chunk"
    #check if input is a las file
    if os.path.splitext(input_las_path)[-1].lower() == ".las":
        LAS_name_original = os.path.splitext(os.path.basename(input_las_path))[0]
        LAS_name = LAS_name_original.split('_')[0] + f'_{subfolder}.las'
        output_las_path = os.path.join('working',subfolder,LAS_name)
        print(f"Now calculating for: {LAS_name_original}...")
    else:
        print("ERROR: input is not a las file, quitting.")
        exit()

    #if working doesnt exist, create it with
    #subfolder geom else the func does nothing
    geometricFeatures.createWorkingDir(sub_folder= subfolder)

    radius = 0.5  # search radius
    main(input_las_path, output_las_path, radius)
