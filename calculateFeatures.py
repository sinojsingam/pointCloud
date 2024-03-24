import numpy as np
from numpy.linalg import eigh
import colorsys
import pandas as pd
import laspy
import geometricFeatures
import os
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import cv2

decimal_digits = 8

def grid_subsampling_with_color(points, voxel_size):
    #Poux F.
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel= np.unique(((points[:, :3] - np.min(points[:, :3], axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)
    voxel_grid={}
    grid_barycenter,grid_candidate_center=[],[]
    last_seen=0

    for idx,vox in enumerate(non_empty_voxel_keys):
        pts_in_vox = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        voxel_grid[tuple(vox)]=pts_in_vox
        grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)][:, :3]-np.mean(voxel_grid[tuple(vox)][:, :3],axis=0),axis=1).argmin()])
        last_seen+=nb_pts_per_voxel[idx]
    data_array = np.array(grid_candidate_center)
    return data_array

def grid_subsampling(points, voxel_size):
    #Poux F.
    nb_vox=np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel= np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)
    voxel_grid={}
    grid_barycenter,grid_candidate_center=[],[]
    last_seen=0

    for idx,vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)]=points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
        grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
        last_seen+=nb_pts_per_voxel[idx]
    data_array = np.array(grid_barycenter)
    return data_array

def getRadii_voxelSizes(scales=10,smallest_radius=0.1, growth_factor=2, density=5):
    r_scales = []
    grid_sizes = []
    for s in range(scales):
        r_s = smallest_radius * (growth_factor)**s
        grid_size = r_s/density
        grid_sizes.append(grid_size)
        r_scales.append(r_s)
    return r_scales, grid_sizes

def compute_covariance_matrix(neighbors):
    return np.cov(neighbors.T)

def compute_eigenvalues(covariance_matrix):
    eigenvalues, _ = eigh(covariance_matrix) #it gives eigen values and vectors as tuple
    return np.flip(np.sort(eigenvalues)) #l1>l2>l3

def compute_omnivariance(lambda_1, lambda_2, lambda_3):
    #array = np.cbrt(np.prod(eigenvalues))
    array = np.cbrt(lambda_1 * lambda_2 * lambda_3)
    return array

def compute_eigenentropy(eigenvalues, lambda_1, lambda_2, lambda_3):
    eigenvalues = eigenvalues[eigenvalues > 0]
    if lambda_1 <= 0 or lambda_2 <= 0 or lambda_3 <= 0:
        return np.nan
    else:
    # Add the constant to the eigenvalues before taking the logarithm
        array = -( (lambda_1*np.log(lambda_1)) + (lambda_2*np.log(lambda_2)) + (lambda_3*np.log(lambda_3)))
        return array

def compute_anisotropy(lambda_1, lambda_3):
    array = np.round((lambda_1 - lambda_3) / lambda_1 if lambda_1 > 0 else 0, decimals=decimal_digits)
    return array

def compute_linearity(lambda_1, lambda_2):
    array = np.round((lambda_1 - lambda_2) / lambda_1 if lambda_1 > 0 else 0,decimals=decimal_digits)
    return array

def compute_planarity(lambda_1, lambda_2, lambda_3):
    array = np.round((lambda_2 - lambda_3) / lambda_1 if lambda_1 > 0 else 0,decimals=decimal_digits)
    return array

def compute_curvature(lambda_1, lambda_2, lambda_3):
    sum = lambda_1 + lambda_2 + lambda_3
    array = np.round(lambda_3 / sum if sum > 0 else 0,decimals=decimal_digits)
    return array

def compute_sphericity(lambda_1, lambda_3):
    array =  np.round(lambda_3 / lambda_1 if lambda_1 > 0 else 0,decimals=decimal_digits) 
    return array

def compute_verticality(points):
    verticality_values = list(map(lambda point: 1 - point[3], points))
    return np.round(np.array(verticality_values),decimals=decimal_digits)

def compute_height(point,neighbors):
    z_point = point[2] # z of point
    z_neighbors = neighbors[:,2] # z of neighbors
    min, max = np.min(z_neighbors), np.max(z_neighbors)
    range = round(max - min, 2).astype(np.float32)
    average_height = round(np.mean(z_neighbors),2).astype(np.float32)
    height_above = round(max - z_point,2).astype(np.float32)
    height_below = round(z_point - min,2).astype(np.float32)
    return range, average_height, height_below, height_above

def rgb_to_hsv(colors_array):
    return np.round(np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_array]),decimals=2)


def addDimsToLAS(laspyLASObject):
    dim_names = [f'omnivariance', #0
                 f'eigenentropy', #1
                 f'anisotropy', #2
                 f'linearity', #3
                 f'curvature', #4
                 f'sphericity',#5
                 f'planarity', #6
                 f'verticality'] #7
    
    data_type = np.float32
    #adding metadata to LAS
    laspyLASObject.add_extra_dims([laspy.ExtraBytesParams(name=dim_names[0], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[1], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[2], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[3], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[4], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[5], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[6], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[7], type=data_type)
                        ])
    return "dims added"

def saveDF_as_LAS(df,reference_LAS,radius,output_file):
    output_file = f"{output_file}_{radius}.las"
    output_file_path = os.path.join("../results/testing",output_file)
    # Create a new header
    header = laspy.LasHeader(point_format=reference_LAS.header.point_format, version=reference_LAS.header.version)
    header.offsets = reference_LAS.header.offsets
    header.scales = reference_LAS.header.scales
    # header.add_extra_dim(laspy.ExtraBytesParams(name=f"RF", type=np.float32))
    # header.add_extra_dim(laspy.ExtraBytesParams(name=f"GBT", type=np.float32))
    addDimsToLAS(header)
    #retrieve color info from las file
    # rgb_non_normalised = np.vstack((reference_LAS.red,reference_LAS.green,reference_LAS.blue)).transpose() * 65535.0 
    # Create a LasWriter and a point record, then write it
    with laspy.open(output_file_path, mode="w", header=header) as writer:

        point_record = laspy.ScaleAwarePointRecord.zeros(df.shape[0], header=header)
        # point_record.x = np.array(df['X'] + header.offsets[0])
        # point_record.y = np.array(df['Y'] + header.offsets[1])
        # point_record.z = np.array(df['Z'])
        point_record.x = df.get('X')
        point_record.y = df.get('Y')
        point_record.z = df.get('Z')
        point_record.omnivariance = df.get('omnivariance')
        point_record.eigenentropy = df.get('eigenentropy')
        point_record.anisotropy = df.get('anisotropy')
        point_record.linearity = df.get('linearity')
        point_record.planarity = df.get('planarity')
        point_record.curvature = df.get('curvature')
        point_record.sphericity = df.get('sphericity')
        point_record.verticality = df.get('verticality')
        #point_record.red = rgb_non_normalised[:, 0]
        #point_record.green = rgb_non_normalised[:, 1]
        #point_record.blue = rgb_non_normalised[:, 2]
        # point_record.RF = df.get('predictions_RF')
        # point_record.GBT = df.get('predictions_GBT')
        writer.write_points(point_record)



def saveNP_as_LAS(data_to_save,reference_LAS,output_file):
    # Create a new header
    header = laspy.LasHeader(point_format=reference_LAS.header.point_format, version=reference_LAS.header.version)
    header.offsets = reference_LAS.header.offsets
    header.scales = reference_LAS.header.scales
    # radius = str(0.5)
    # header.add_extra_dim(lp.ExtraBytesParams(name=f"RF_{radius}", type=np.float32))
    # header.add_extra_dim(lp.ExtraBytesParams(name=f"GBT_{radius}", type=np.float32))
    #retrieve color info from las file
    # rgb_non_normalised = np.vstack((point_cloud.red,point_cloud.green,point_cloud.blue)).transpose() * 65535.0 
    # Create a LasWriter and a point record, then write it
    with laspy.open(output_file, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(data_to_save.shape[0], header=header)
        # point_record.x = np.array(df['X'] + header.offsets[0])
        # point_record.y = np.array(df['Y'] + header.offsets[1])
        # point_record.z = np.array(df['Z'])
        point_record.x = data_to_save[:, 0]
        point_record.y = data_to_save[:, 1]
        point_record.z = data_to_save[:, 2]
        point_record['normal z'] = data_to_save[:, 3]
        point_record.red = data_to_save[:, 4]
        point_record.green = data_to_save[:, 5]
        point_record.blue = data_to_save[:, 6]
        # point_record.RF = RF_array
        # point_record.GBT = GBT_array
        writer.write_points(point_record)


def calculateGeometricFeatures(data_array,neighborhood_radius, data_type = np.float32, save=False, output_file=None):
    """
    Iterates over each point and calculates the geometric features for each point and its neighbors in a spherical neighborhood.
    """
    colors_rgb = (data_array[:, 5:8] / 65535.0).astype(data_type) #normalise
    colors_hsv = np.round(np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_rgb]),decimals=2).astype(data_type)
    translated_3d_color = np.hstack([data_array, colors_hsv])
    tree = cKDTree(translated_3d_color[:, :3])
    pc_length = translated_3d_color.shape[0]
    #initiating np arrays
    #values for each neighbor#
    omniList = np.zeros(pc_length, dtype=data_type)
    eigenList = np.zeros(pc_length, dtype=data_type)
    anisoList = np.zeros(pc_length, dtype=data_type)
    linList = np.zeros(pc_length, dtype=data_type)
    planarList = np.zeros(pc_length, dtype=data_type)
    curveList = np.zeros(pc_length, dtype=data_type)
    sphereList = np.zeros(pc_length, dtype=data_type)
    heightRangeList = np.zeros(pc_length, dtype=data_type)
    heightAvgList = np.zeros(pc_length, dtype=data_type)
    heightBelowList = np.zeros(pc_length, dtype=data_type)
    heightAboveList = np.zeros(pc_length, dtype=data_type)
    neighboringHList = np.zeros(pc_length, dtype=data_type)
    neighboringSList = np.zeros(pc_length, dtype=data_type)
    neighboringVList = np.zeros(pc_length, dtype=data_type)
    #values for each point#
    xList = data_array[:, 0]
    yList = data_array[:, 1]
    zList = data_array[:, 2]
    #color values#
    H_List = colors_hsv[:, 0].astype(data_type)
    S_List = colors_hsv[:, 1].astype(data_type)
    V_List = colors_hsv[:, 2].astype(data_type)
    #calculate verticality#
    verticalityList = compute_verticality(data_array)

    #Loops only once for all calculations according to neighbors
    for i, point in enumerate(translated_3d_color):
        indices = tree.query_ball_point(point[: 3], neighborhood_radius) #query just the coordinates XYZ coordinates and radius
        neighbors = translated_3d_color[indices]
         # Need at least 4 points to compute a meaningful covariance matrix
        if len(neighbors) < 4:
            omniList[i] = np.nan
            eigenList[i] = np.nan
            anisoList[i] = np.nan
            linList[i] = np.nan
            planarList[i] = np.nan
            curveList[i] = np.nan
            sphereList[i] = np.nan
            heightRangeList[i] = np.nan
            heightAvgList[i] = np.nan
            heightBelowList[i] = np.nan
            heightAboveList[i] = np.nan
            neighboringHList[i] = np.nan
            neighboringSList[i] = np.nan
            neighboringVList[i] = np.nan
        else:
            heightRange,average_height, heightBelow, heightAbove = compute_height(point, neighbors)
            cov_matrix = compute_covariance_matrix(neighbors[:, :3]).astype(data_type)
            eigenvalues = compute_eigenvalues(cov_matrix)
            sum_eigenvalues = np.sum(eigenvalues) + 0.001
            #normalise eigenvalues
            lambda_1 = eigenvalues[0] / sum_eigenvalues
            lambda_2 = eigenvalues[1] / sum_eigenvalues
            lambda_3 = eigenvalues[2] / sum_eigenvalues
            #Geometric features
            omni = compute_omnivariance(lambda_1, lambda_2, lambda_3)
            eigen = compute_eigenentropy(eigenvalues, lambda_1, lambda_2, lambda_3)
            aniso = compute_anisotropy(lambda_1, lambda_3)
            linear = compute_linearity(lambda_1, lambda_2)
            planar = compute_planarity(lambda_1, lambda_2, lambda_3)
            curve = compute_curvature(lambda_1, lambda_2, lambda_3)
            sphere = compute_sphericity(lambda_1, lambda_3)
            #Retrieve average neighboring colors value
            k_H, k_S, k_V = np.round(np.mean(neighbors[...,-3:], axis=0), decimals=2)
            #Assign values to lists
            omniList[i] = omni
            eigenList[i] = eigen
            anisoList[i] = aniso
            linList[i] = linear
            planarList[i] = planar
            curveList[i] = curve
            sphereList[i] = sphere
            heightRangeList[i] = heightRange
            heightAvgList[i] = average_height
            heightBelowList[i] = heightBelow
            heightAboveList[i] = heightAbove
            neighboringHList[i] = k_H
            neighboringSList[i] = k_S
            neighboringVList[i] = k_V

    #Create a dictionary with all the values
    pointsDict_with_nan = {
            "X": xList,
            "Y": yList,
            "Z": zList,
            "H": H_List,
            "S": S_List,
            "V": V_List,
            "classification": data_array[:, 4],
            "normal z": data_array[:, 3],
            "omnivariance": omniList,
            "eigenentropy": eigenList,
            "anisotropy": anisoList,
            "linearity": linList,
            "planarity": planarList,
            "curvature": curveList,
            "sphericity": sphereList,
            "verticality": verticalityList,
            "height_range":heightRangeList,
            "height_below": heightBelowList,
            "height_above": heightAboveList,
            "neighbor_H": neighboringHList,
            "neighbor_S": neighboringSList,
            "neighbor_V": neighboringVList,  
        }
    df = pd.DataFrame(pointsDict_with_nan)
    df = df.dropna()
    pointsDict = df.to_dict(orient='list')
    if save:
        ref_las = laspy.read('../working/classification/multiscale/classified_sample.las')
        output_path = '../results/testing/'
        saveDF_as_LAS(pd.DataFrame(pointsDict), ref_las, neighborhood_radius, output_path+output_file)
    return pointsDict