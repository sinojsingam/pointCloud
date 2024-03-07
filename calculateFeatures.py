import numpy as np
from numpy.linalg import eigh

def compute_covariance_matrix(neighbors):
    return np.cov(neighbors.T)

def compute_eigenvalues(covariance_matrix):
    eigenvalues, _ = eigh(covariance_matrix)
    return np.flip(np.sort(eigenvalues)) #l1>l2>l3

def compute_omnivariance(eigenvalues):
    return np.cbrt(np.product(eigenvalues))

def compute_eigenentropy(eigenvalues):
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:  # Check if all eigenvalues were filtered out
        return 0
    else:
        return -np.sum(eigenvalues * np.log(eigenvalues))

def compute_anisotropy(lambda_1, lambda_3):
    return (lambda_1 - lambda_3) / lambda_1 if lambda_1 > 0 else 0

def compute_linearity(lambda_1, lambda_2):
    return (lambda_1 - lambda_2) / lambda_1 if lambda_1 > 0 else 0

def compute_planarity(lambda_1, lambda_2, lambda_3):
    return (lambda_2 - lambda_3) / lambda_1 if lambda_1 > 0 else 0

def compute_curvature(lambda_1, lambda_2, lambda_3):
    sum = lambda_1 + lambda_2 + lambda_3
    return lambda_3 / sum if sum > 0 else 0

def compute_sphericity(lambda_1, lambda_3):
    return lambda_3 / lambda_1 if lambda_1 > 0 else 0

def compute_verticality(points):
    verticality_values = list(map(lambda point: 1 - point[3], points))
    return np.array(verticality_values)