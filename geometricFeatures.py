import numpy as np

def compute_omnivariance(points, cKDTree, radius):
    """
    Calculate omnivariance for each point in the point cloud using a spherical neighborhood of a given radius.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param KDTree: cKDTree from scipy
    :param radius: The radius to define the spherical neighborhood.
    :return: Array of omnivariance values for each point.
    """
    omnivariance_values = []
    #tree = cKDTree(points)
    for i, point in enumerate(points):

        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
        if len(neighbors) < 3:  # Need at least 3 points to form a plane
            omnivariance_values.append(0)
            continue
        
        # Compute covariance matrix of neighbors
        cov_matrix = np.cov(neighbors.T)
        # Compute eigenvalues and calculate omnivariance
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        #math calculation
        omnivariance = np.cbrt(np.product(eigenvalues))
        omnivariance_values.append(omnivariance)
    
    return np.array(omnivariance_values)

def compute_eigenentropy(points, cKDTree, radius):
    """
    Calculate Eigenentropy for each point in the point cloud using a spherical neighborhood.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius to define the spherical neighborhood.
    :return: A NumPy array of Eigenentropy values for each point.
    """
    eigenentropy_values = []
    #tree = cKDTree(points)
    
    for i, point in enumerate(points):
        
        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
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

def compute_anisotropy(points, cKDTree, radius):
    """
    Compute Anisotropy for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Anisotropy values for each point.
    """
    anisotropy_values = []

    for i, point in enumerate(points):
        
        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
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

def compute_linearity(points, cKDTree, radius):
    """
    Compute Linearity for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Linearity values for each point.
    """
    linearity_values = []

    for i, point in enumerate(points):
        
        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
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

def compute_surface_variation(points, cKDTree, radius):
    """
    Compute Surface Variation for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Surface Variation values for each point.
    """
    # Calculate pairwise distances between points
    surface_variation_values = []

    for i, point in enumerate(points):
        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
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

def compute_sphericity(points, cKDTree, radius):
    """
    Compute Sphericity for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Sphericity values for each point.
    """

    sphericity_values = []

    for i, point in enumerate(points):

        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
        if len(neighbors) < 4:  # Need at least 4 points for a meaningful covariance matrix
            sphericity_values.append(0)
            continue

        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and sort them in ascending order
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate Sphericity
        lambda_1, _, lambda_3 = eigenvalues
        sphericity = lambda_3 / lambda_1 if lambda_1 > 0 else 0
        sphericity_values.append(sphericity)
    
    return np.array(sphericity_values)

def compute_first_order_moments(points, cKDTree, radius):
    """
    Compute the 1st order moments for each point in the point cloud within a radius-based neighborhood,
    oriented along the principal eigenvector.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: Radius to define the neighborhood around each point.
    :return: A list of 1st order moments for each point's neighborhood.
    """
    moments = []

    for i, point in enumerate(points):
        # Find points within the radius
        indices = cKDTree.query_ball_point(point, radius)
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

def compute_curvature(points, cKDTree, radius):
    """
    Compute Curvature for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of curvature values for each point.
    """

    curvature_values = []

    for i, point in enumerate(points):

        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
        if len(neighbors) < 4:  # Need at least 4 points for a meaningful covariance matrix
            curvature_values.append(0)
            continue

        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and sort them in ascending order
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate Sphericity
        lambda_1, lambda_2, lambda_3 = eigenvalues
        curvature = lambda_3 / lambda_1 + lambda_2 + lambda_3 
        curvature_values.append(curvature)
    
    return np.array(curvature_values)

#color manipulation
def normalized_rgb_to_hsv(normalized_colors):
    """
    Convert an array of normalized RGB values to HSV.
    
    Parameters:
    - normalized_colors: numpy array of shape (n, 3) where each row contains normalized 16-bit R, G, B values.
    
    Returns:
    - hsv_colors: numpy array of shape (n, 3) containing HSV values.
    """
    r, g, b = normalized_colors[:, 0], normalized_colors[:, 1], normalized_colors[:, 2]
    c_max = np.max(normalized_colors, axis=1)
    c_min = np.min(normalized_colors, axis=1)
    delta = c_max - c_min

    # Hue calculation
    hue = np.zeros_like(c_max)
    mask_r = (c_max == r) & (delta != 0)
    mask_g = (c_max == g) & (delta != 0)
    mask_b = (c_max == b) & (delta != 0)
    
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    # Saturation calculation
    saturation = np.where(c_max == 0, 0, delta / c_max)

    # Value calculation
    value = c_max

    hsv_colors = np.vstack((hue, saturation * 100, value * 100)).transpose()

    return hsv_colors

def compute_verticality(points):
    """
    Compute Verticality for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Verticality values for each point.
    """

    verticality_values = []

    for point in points:
        nz = point[3]

        verticality = 1 - nz #3rd dimension
        verticality_values.append(verticality)
    
    return np.array(verticality_values)

def translate_coords(numpy_coords_array):
    """
    Translates array to make computations easier
    :param array: NumPy array of shape (N, 3) representing the point cloud.
    :return: A NumPy array of with translated coordinates.
    """
    X = numpy_coords_array[:,0]
    Y = numpy_coords_array[:,1]
    Z = numpy_coords_array[:,2]
    #NZ = numpy_coords_array[:,3]
    baseX = X[0] // 1000
    baseY = Y[0] // 1000   # Find the base of the first element
    bases_x = set(map(lambda x: x // 100000, X))
    #bases_y = set(map(lambda x: x // 100000, Y))
    if len(bases_x) == 1:
        offset = (baseX*1000,baseY*1000)
        point_coords = np.vstack((X - offset[0], Y - offset[1], Z)).transpose()
        print(f'Translated with {offset}')
        return point_coords