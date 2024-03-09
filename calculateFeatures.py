import numpy as np
from numpy.linalg import eigh
import colorsys


decimal_digits = 8

def compute_covariance_matrix(neighbors):
    return np.round(np.cov(neighbors.T),decimals=decimal_digits)

def compute_eigenvalues(covariance_matrix):
    eigenvalues, _ = eigh(covariance_matrix) #it gives eigen values and vectors as tuple
    return np.round(np.flip(np.sort(eigenvalues)),decimals=decimal_digits) #l1>l2>l3

def compute_omnivariance(eigenvalues):
    return np.round(np.cbrt(np.prod(eigenvalues)),decimals=decimal_digits)

def compute_eigenentropy(eigenvalues):
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:  # Check if all eigenvalues were filtered out
        return 0
    else:
        return np.round(-np.sum(eigenvalues * np.log(eigenvalues)),decimals=decimal_digits)

def compute_anisotropy(lambda_1, lambda_3):
    return np.round((lambda_1 - lambda_3) / lambda_1 if lambda_1 > 0 else 0, decimals=decimal_digits)

def compute_linearity(lambda_1, lambda_2):
    return np.round((lambda_1 - lambda_2) / lambda_1 if lambda_1 > 0 else 0,decimals=decimal_digits)

def compute_planarity(lambda_1, lambda_2, lambda_3):
    return np.round((lambda_2 - lambda_3) / lambda_1 if lambda_1 > 0 else 0,decimals=decimal_digits)

def compute_curvature(lambda_1, lambda_2, lambda_3):
    sum = lambda_1 + lambda_2 + lambda_3
    return np.round(lambda_3 / sum if sum > 0 else 0,decimals=decimal_digits)

def compute_sphericity(lambda_1, lambda_3):
    return np.round(lambda_3 / lambda_1 if lambda_1 > 0 else 0,decimals=decimal_digits)

def compute_verticality(points):
    verticality_values = list(map(lambda point: 1 - point[3], points))
    return np.round(np.array(verticality_values),decimals=decimal_digits)

def compute_height(point,neighbors):
    z_point = point[2] # z of point
    z_neighbors = neighbors[:,2] # z of neighbors
    min, max = np.min(z_neighbors), np.max(z_neighbors)
    range = round(max - min, 2)
    height_above = round(max - z_point,2)
    height_below = round(z_point - min,2)
    return range, height_below, height_above

def rgb_to_hsv(colors_array):
    return np.round(np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_array]),decimals=2)
#np.mean(colorsofneighbors, axis=0) for average of the colors