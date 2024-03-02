import laspy
import CSF
import sklearn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial import KDTree
from scipy.spatial import cKDTree


def compute_omnivariance(points, radius):
    """
    Calculate omnivariance for each point in the point cloud using a spherical neighborhood of a given radius.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius to define the spherical neighborhood.
    :return: Array of omnivariance values for each point.
    """
    omnivariances = []
    # Compute pairwise distances between points
    distances = pairwise_distances(points)
    
    for i, point in enumerate(points):
        # Find points within the spherical neighborhood
        within_radius = distances[i] <= radius
        neighbors = points[within_radius]
        
        if len(neighbors) < 3:  # Need at least 3 points to form a plane
            omnivariances.append(0)
            continue
        
        # Compute covariance matrix of neighbors
        cov_matrix = np.cov(neighbors.T)
        
        # Compute eigenvalues and calculate omnivariance
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        omnivariance = np.cbrt(np.product(eigenvalues))
        omnivariances.append(omnivariance)
    
    return np.array(omnivariances)

def compute_eigenentropy(points, radius):
    """
    Calculate Eigenentropy for each point in the point cloud using a spherical neighborhood.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius to define the spherical neighborhood.
    :return: A NumPy array of Eigenentropy values for each point.
    """
    tree = cKDTree(points)
    eigenentropy_values = []

    for point in points:
        # Find points within the spherical neighborhood
        idx = tree.query_ball_point(point, radius)
        neighbors = points[idx]
        
        if len(neighbors) < 3:  # Ensure there are enough points to compute a covariance matrix
            eigenentropy_values.append(0)
            continue
        
        # Compute the covariance matrix of the neighborhood
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and filter out non-positive values to avoid log domain errors
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        if eigenvalues.size == 0:  # Check if all eigenvalues were filtered out
            eigenentropy_values.append(0)
            continue
        
        # Calculate Eigenentropy
        eigenentropy = -np.sum(eigenvalues * np.log(eigenvalues))
        eigenentropy_values.append(eigenentropy)

    return np.array(eigenentropy_values)

def compute_anisotropy(points, radius):
    """
    Compute Anisotropy for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Anisotropy values for each point.
    """
    distances = pairwise_distances(points)
    anisotropy_values = []

    for i, point in enumerate(points):
        # Identify points within the spherical neighborhood
        within_radius = distances[i] <= radius
        neighbors = points[within_radius]
        
        if len(neighbors) < 4:  # Ensuring enough points for a valid covariance matrix
            anisotropy_values.append(0)
            continue
        
        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate the eigenvalues (sorted in ascending order by default)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate Anisotropy
        lambda_1, _, lambda_3 = eigenvalues[-1], eigenvalues[1], eigenvalues[0]
        anisotropy = (lambda_1 - lambda_3) / lambda_1 if lambda_1 > 0 else 0
        anisotropy_values.append(anisotropy)
    
    return np.array(anisotropy_values)

def compute_linearity(points, radius):
    """
    Compute Linearity for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Linearity values for each point.
    """
    # Calculate pairwise distances between points
    distances = pairwise_distances(points)
    linearity_values = []

    for i, point in enumerate(points):
        # Find points within the spherical neighborhood
        within_radius = distances[i] <= radius
        neighbors = points[within_radius]
        
        if len(neighbors) < 4:  # Need at least 4 points to compute a covariance matrix
            linearity_values.append(0)
            continue

        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and sort them in descending order
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = sorted(eigenvalues, reverse=True)
        
        # Calculate Linearity
        lambda_1, lambda_2, _ = eigenvalues
        linearity = (lambda_1 - lambda_2) / lambda_1 if lambda_1 > 0 else 0
        linearity_values.append(linearity)
    
    return np.array(linearity_values)

def compute_surface_variation(points, radius):
    """
    Compute Surface Variation for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Surface Variation values for each point.
    """
    # Calculate pairwise distances between points
    distances = pairwise_distances(points)
    surface_variation_values = []

    for i, point in enumerate(points):
        # Find points within the spherical neighborhood
        within_radius = distances[i] <= radius
        neighbors = points[within_radius]
        
        if len(neighbors) < 4:  # Need at least 4 points to compute a meaningful covariance matrix
            surface_variation_values.append(0)
            continue

        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and sort them in ascending order
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate Surface Variation
        lambda_1, lambda_2, lambda_3 = eigenvalues
        total_variance = sum(eigenvalues)
        surface_variation = lambda_3 / total_variance if total_variance > 0 else 0
        surface_variation_values.append(surface_variation)
    
    return np.array(surface_variation_values)

#CHECK formula
def compute_sphericity(points, radius):
    """
    Compute Sphericity for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Sphericity values for each point.
    """
    # Calculate pairwise distances between points
    distances = pairwise_distances(points)
    sphericity_values = []

    for i, point in enumerate(points):
        # Find points within the spherical neighborhood
        within_radius = distances[i] <= radius
        neighbors = points[within_radius]
        
        if len(neighbors) < 4:  # Need at least 4 points for a meaningful covariance matrix
            sphericity_values.append(0)
            continue

        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and sort them in ascending order
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate Sphericity
        lambda_1, lambda_2, lambda_3 = eigenvalues
        total_variance = sum(eigenvalues)
        sphericity = (3 * lambda_3) / total_variance if total_variance > 0 else 0
        sphericity_values.append(sphericity)
    
    return np.array(sphericity_values)

#check formula
def compute_verticality(points, radius):
    """
    Compute Verticality for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Verticality values for each point.
    """
    distances = pairwise_distances(points)
    verticality_values = []

    for i, point in enumerate(points):
        # Find points within the spherical neighborhood
        within_radius = distances[i] <= radius
        neighbors = points[within_radius]
        
        if len(neighbors) < 4:  # Require at least 4 points for a meaningful covariance matrix
            verticality_values.append(0)
            continue
        
        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigvalsh(cov_matrix)
        
        # The eigenvector corresponding to the smallest eigenvalue
        normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Calculate the angle between the normal vector and the vertical
        # Assuming the z-axis is [0, 0, 1]
        verticality = np.arccos(np.abs(normal_vector[2]))
        
        # Convert from radians to degrees for easier interpretation
        verticality_degrees = np.degrees(verticality)
        verticality_values.append(verticality_degrees)
    
    return np.array(verticality_values)

def compute_first_order_moments(points, radius):
    """
    Compute the 1st order moments for each point in the point cloud within a radius-based neighborhood,
    oriented along the principal eigenvector.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: Radius to define the neighborhood around each point.
    :return: A list of 1st order moments for each point's neighborhood.
    """
    tree = KDTree(points)
    moments = []

    for i, point in enumerate(points):
        # Find points within the radius
        indices = tree.query_ball_point(point, radius)
        if len(indices) < 3:  # Need at least 3 points to compute a meaningful covariance matrix
            moments.append(np.array([0, 0, 0]))
            continue

        # Extract the neighborhood points
        neighborhood = points[indices]
        
        # Center the neighborhood points
        centered_neighborhood = neighborhood - neighborhood.mean(axis=0)
        
        # Compute the covariance matrix of the neighborhood
        cov_matrix = np.cov(centered_neighborhood, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Find the principal eigenvector (associated with the largest eigenvalue)
        principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Compute the 1st order moment along the principal eigenvector
        # This is essentially the projection of the points onto the eigenvector
        # and then finding the mean of these projections.
        projections = np.dot(centered_neighborhood, principal_eigenvector)
        first_order_moment = np.mean(projections)
        
        # Store the moment
        moments.append(first_order_moment)

    return moments

#color manipulation
def rgb_to_hsv(r, g, b):
    r_prime = r / 255.0
    g_prime = g / 255.0
    b_prime = b / 255.0

    c_max = max(r_prime, g_prime, b_prime)
    c_min = min(r_prime, g_prime, b_prime)
    delta = c_max - c_min

    # Hue calculation
    if delta == 0:
        h = 0
    elif c_max == r_prime:
        h = 60 * (((g_prime - b_prime) / delta) % 6)
    elif c_max == g_prime:
        h = 60 * (((b_prime - r_prime) / delta) + 2)
    elif c_max == b_prime:
        h = 60 * (((r_prime - g_prime) / delta) + 4)

    # Saturation calculation
    if c_max == 0:
        s = 0
    else:
        s = (delta / c_max) * 100

    # Value calculation
    v = c_max * 100

    return h, s, v


def average_hsv_neighborhood_colors(points, colors, radius):
    """
    Compute the average HSV color for each point in a point cloud within a spherical neighborhood.
    
    :param points: NumPy array of shape (N, 3) representing the point cloud coordinates.
    :param colors: NumPy array of shape (N, 3) representing the HSV colors of the points.
    :param radius: The radius defining the spherical neighborhood around each point.
    :return: A NumPy array of the averaged HSV colors for each point.
    """
    tree = KDTree(points)
    averaged_colors = np.zeros_like(colors)
    
    for i, point in enumerate(points):
        # Indices of points within the radius including the point itself
        indices = tree.query_ball_point(point, radius)
        
        # Sum and average the HSV colors of the neighborhood
        neighborhood_colors = colors[indices]
        averaged_color = np.mean(neighborhood_colors, axis=0)
        
        # Assign the averaged color to the current point
        averaged_colors[i] = averaged_color
    
    return averaged_colors


def compute_verticality2(points, radius):
    """
    Compute Verticality for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Verticality values for each point.
    """
    tree = KDTree(points)

    #distances = pairwise_distances(points)
    verticality_values = []
    for i, point in enumerate(points):
        # Indices of points within the radius including the point itself
        indices = tree.query_ball_point(point, radius)
        neighbors = points[indices]
        if len(neighbors) < 4:  # Require at least 4 points for a meaningful covariance matrix
            verticality_values.append(0)
            continue
        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The eigenvector corresponding to the smallest eigenvalue
        v_min = eigenvectors[:, np.argmin(eigenvalues)]
        verticality = 1 - np.abs(v_min[2])
        verticality_values.append(verticality)
    
    return np.array(verticality_values)