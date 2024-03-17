import numpy as np
from numpy.linalg import eigh
import colorsys
import pandas as pd
import laspy
import geometricFeatures
decimal_digits = 8


def compute_covariance_matrix(neighbors):
    return np.round(np.cov(neighbors.T),decimals=decimal_digits)

def compute_eigenvalues(covariance_matrix):
    eigenvalues, _ = eigh(covariance_matrix) #it gives eigen values and vectors as tuple
    return np.round(np.flip(np.sort(eigenvalues)),decimals=decimal_digits) #l1>l2>l3

def compute_omnivariance(eigenvalues):
    array = np.round(np.cbrt(np.prod(eigenvalues)),decimals=decimal_digits)
    return array

def compute_eigenentropy(eigenvalues):
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:  # Check if all eigenvalues were filtered out
        return 0
    else:
        array = np.round(-np.sum(eigenvalues * np.log(eigenvalues)),decimals=decimal_digits)
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
    height_above = round(max - z_point,2).astype(np.float32)
    height_below = round(z_point - min,2).astype(np.float32)
    return range, height_below, height_above

def rgb_to_hsv(colors_array):
    return np.round(np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_array]),decimals=2)
#np.mean(colorsofneighbors, axis=0) for average of the colors

def save_as_LAS(df, reference_LAS,radius):
    # Create a new header
    header = laspy.LasHeader(point_format=reference_LAS.header.point_format, version=reference_LAS.header.version)
    header.offsets = reference_LAS.header.offsets
    header.scales = reference_LAS.header.scales
    radius = str(0.5)
    header.add_extra_dim(laspy.ExtraBytesParams(name="RF", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="GBT", type=np.int32))
    
    rgb_non_normalised = np.vstack((las.red,las.green,las.blue)).transpose() * 65535.0 
    # Create a LasWriter and a point record, then write it
    with laspy.open("../working/car_training/write_testing.las", mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(translated_coords.shape[0], header=header)
        point_record.x = translated_coords[:, 0] + header.offsets[0]
        point_record.y = translated_coords[:, 1] + header.offsets[1]
        point_record.z = translated_coords[:, 2]
        point_record.red = rgb_non_normalised[:, 0]
        point_record.green = rgb_non_normalised[:, 1]
        point_record.blue = rgb_non_normalised[:, 2]
        point_record.Verticality = verticalityList
        writer.write_points(point_record)


    las_version = reference_LAS.header.version
    point_format = reference_LAS.header.point_format

    new_las = laspy.create(point_format=point_format, file_version=las_version)

    new_las.x = np.array(df['X']+667000.0) #add translation
    new_las.y = np.array(df['Y']+650000.0)
    new_las.z = np.array(df['Z'])
    #fix
    new_las.red = np.array(reference_LAS.red)
    new_las.green = np.array(reference_LAS.green)
    new_las.blue = np.array(reference_LAS.blue)
    new_las.classification = np.array(df['classification'])
    new_las.add_extra_dim(laspy.ExtraBytesParams(name='omnivariance', type=np.float64))
    geometricFeatures.addDimsToLAS(las,radius)
    new_las.omnivariance = np.array(df['omnivariance'])
    new_las.write('../working/car_training/car.las')