import numpy as np
import os
import time
import laspy

def addDimsToLAS(laspyLASObject,radius,dims=None):
    if dims!=None:
        pass
    dim_names = [f'Omnivariance ({radius})', #0
                 f'Eigenentropy ({radius})', #1
                 f'Anisotropy ({radius})', #2
                 f'Linearity ({radius})', #3
                 f'Curvature ({radius})', #4
                 f'Sphericity ({radius})',#5
                 f'Planarity ({radius})', #6
                 f'Verticality', #7
                 f'Height Range ({radius})', #8
                 f'Height Below ({radius})', #9
                 f'Height Above ({radius})', #10
                 f'Color H', #11
                 f'Color S', #12
                 f'Color V']#13
    
    
    #adding metadata to LAS
    laspyLASObject.add_extra_dims([laspy.ExtraBytesParams(name=dim_names[0], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[1], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[2], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[3], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[4], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[5], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[6], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[7], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[8], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[9], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[10], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[11], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[12], type=np.float64),
                        laspy.ExtraBytesParams(name=dim_names[13], type=np.float64)
    ])
    return "dims added"

def saveLASFile(laspyLASObject, dim_name, numpyArray, output_las_path):
    """
    Recursive function for geometric Features computation. If compute functions have save=True,
    then the a las file will be specified output path with an additional dimension pertaining to the geometric feature.
    :param laspyLASObject: a laspy LAS object coming from laspy.read()
    :dim_name (str): a name for the new dimension to be added to the LAS file
    :numpyArray (array): a numpy array containing the values coming from the compute algorithm
    :output_las_path (str): path for the file to be saved in.
    :return: None
    """
    print(f"Writing LAS file to {output_las_path}")
    #adding metadata to LAS
    laspyLASObject.add_extra_dim(laspy.ExtraBytesParams(name=dim_name, type=np.float64))
    #write the calculated data onto the las file then onto disk
    laspyLASObject[dim_name] = numpyArray
    laspyLASObject.write(output_las_path)


def compute_omnivariance(points, cKDTree, radius, save=False,laspyLASObject=None, output_las_path=None):
    """
    Calculate omnivariance for each point in the point cloud using a spherical neighborhood of a given radius.

    :param points (ndarray): NumPy array of shape (N, 3) representing the point cloud.
    :param cKDTree (cKDTree): cKDTree object from scipy
    :param radius (int): The radius to define the spherical neighborhood.
    :save (Bool): if True, the resulting numpy array will be added to a new dimension and saved onto disk
    :kwargs: parameters needed to save file, if file isnt being saved, they can be ignored
    :return: Array of omnivariance values for each point.
    """
    dim_name = f'Omnivariance ({radius})'
    start = time.time()
    omnivariance_values = []
    
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
    end = time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    if save:
        saveLASFile(laspyLASObject, dim_name, np.array(omnivariance_values), output_las_path)
    return dim_name, np.array(omnivariance_values)

def compute_eigenentropy(points, cKDTree, radius,save=False,laspyLASObject=None,output_las_path=None):
    """
    Calculate Eigenentropy for each point in the point cloud using a spherical neighborhood.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius to define the spherical neighborhood.
    :return: A NumPy array of Eigenentropy values for each point.
    """
    dim_name = f'Eigenentropy ({radius})'
    start = time.time()
    eigenentropy_values = []
    
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
    end = time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    if save:
        saveLASFile(laspyLASObject, dim_name, np.array(eigenentropy_values), output_las_path)
    return dim_name, np.array(eigenentropy_values)

def compute_anisotropy(points, cKDTree, radius,save=False,laspyLASObject=None, output_las_path=None):
    """
    Compute Anisotropy for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Anisotropy values for each point.
    """
    dim_name = f'Anisotropy ({radius})'
    start = time.time()
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
    end = time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    if save:
        saveLASFile(laspyLASObject, dim_name, np.array(anisotropy_values), output_las_path)
    return dim_name, np.array(anisotropy_values)

def compute_linearity(points, cKDTree, radius,save=False,laspyLASObject=None, output_las_path=None):
    """
    Compute Linearity for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Linearity values for each point.
    """
    dim_name = f'Linearity ({radius})'
    start=time.time()
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
    end=time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    if save:
        saveLASFile(laspyLASObject, dim_name, np.array(linearity_values), output_las_path)
    return dim_name, np.array(linearity_values)

def compute_curvature(points, cKDTree, radius,save=False,laspyLASObject=None, output_las_path=None):
    """
    Compute curvature or surface variation or  for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Surface Variation values for each point.
    """
    dim_name = f'Curvature ({radius})'
    start = time.time()
    # Calculate pairwise distances between points
    curvature_values = []

    for i, point in enumerate(points):
        indices = cKDTree.query_ball_point(point, radius)
        neighbors = points[indices]
        
        if len(neighbors) < 4:  # Need at least 4 points to compute a meaningful covariance matrix
            curvature_values.append(0)
            continue

        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors.T)
        
        # Calculate eigenvalues and sort them in ascending order
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Calculate Surface Variation
        lambda_1, lambda_2, lambda_3 = eigenvalues
        total_variance = sum(eigenvalues)
        curvature = lambda_3 / total_variance if total_variance > 0 else 0
        curvature_values.append(curvature)
    end = time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    if save:
        saveLASFile(laspyLASObject, dim_name, np.array(curvature_values), output_las_path)
    return dim_name, np.array(curvature_values)

def compute_sphericity(points, cKDTree, radius,save=False,laspyLASObject=None, output_las_path=None):
    """
    Compute Sphericity for each point in the point cloud using spherical neighborhoods.

    :param points: NumPy array of shape (N, 3) representing the point cloud.
    :param radius: The radius of the spherical neighborhoods.
    :return: A NumPy array of Sphericity values for each point.
    """
    dim_name = f'Sphericity ({radius})'
    start = time.time()
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
    end = time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    if save:
        saveLASFile(laspyLASObject, dim_name, np.array(sphericity_values), output_las_path)
    return dim_name, np.array(sphericity_values)

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
    start = time.time()
    dim_name = f'Verticality'
    verticality_values = []
    verticality_values = list(map(lambda point: 1 - point[3], points))
    end = time.time()
    printTimeElapsed(dim_name, round((end-start)/60,2))
    return dim_name, np.array(verticality_values)
#translation
def translate_coords(numpy_coords_array):
    """
    Translates array to make computations easier
    :param array: NumPy array of shape (N, 3) representing the point cloud.
    :return: A NumPy array of with translated coordinates.
    """
    X = numpy_coords_array[:,0]
    Y = numpy_coords_array[:,1]
    Z = numpy_coords_array[:,2]
    NZ = numpy_coords_array[:,3]
    baseX = X[0] // 1000
    baseY = Y[0] // 1000   # Find the base of the first element
    bases_x = set(map(lambda x: x // 100000, X))
    #bases_y = set(map(lambda x: x // 100000, Y))
    if len(bases_x) == 1:
        offset = (baseX*1000,baseY*1000)
        point_coords = np.vstack((X - offset[0], Y - offset[1], Z, NZ)).transpose()
        print(f"""\nTranslated with {offset}\ne.g for X {X[0]} - {offset[0]} -> {round(X[0] - offset[0],2)}\nand for Y {Y[0]} - {offset[1]} -> {round(Y[0] - offset[1],2)}\n""")
        return point_coords

#print table
def printTimeElapsed(title,time):
    """
    Print table with elapsed times for the geometric features
    :param timeElapsed: str of times elapsed for the different geometric features

    :return: print statement in the form of a table
    """
    #headers = ["Geometric feature", "Time elapsed (mins)"]
    # headers
    #header_row = "|".join(f"{header:^25}" for header in headers)
    #print(header_row)
    #print("-" * len(header_row))
    # rows
    timeList = [title,str(time)+' mins']
    #print("|".join(f"{str(row):^25}") for row in timeList)
    row_table = "|".join(f"{row:^25}" for row in timeList)
    print(row_table)
#create files
def createWorkingDir(sub_folder, main_folder="working"):
    """
    This function checks if there is working directory in cwd and if yes, it doesnt do anything
    if not, it will create working directory with a subfolder with user-specified name.
    :param sub_folder (str): sub-directory withing the main directory
    :param main_folder (str): default name for main directory is working
    :return: directory with subdirectory
    """
    # Check for the working folder
    main_folder_path = os.path.join(os.getcwd(), main_folder)
    if not os.path.exists(main_folder_path):
        os.mkdir(main_folder_path)
        print(f"Working folder '{main_folder}' created.")
    else:
        pass #already exists
    # Check for the subfolder 
    subfolder_path = os.path.join(main_folder_path, sub_folder)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
        print(f"Subfolder '{sub_folder}' created in '{main_folder}'.")
    else:
        pass #already exists