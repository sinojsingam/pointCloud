import numpy as np # type: ignore
from numpy.linalg import eigh # type: ignore
import colorsys
import pandas as pd # type: ignore
import laspy # type: ignore
from scipy.spatial import cKDTree # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.ensemble import RandomForestClassifier  #type: ignore
from sklearn.model_selection import train_test_split  #type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  #type: ignore
import matplotlib.pyplot as plt  #type: ignore
from rasterio.transform import rowcol # type: ignore
import rasterio # type: ignore
import time

decimal_digits = 8

def print_progress(iteration, total, prefix='Progress', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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
    scale_tuples= []
    for s in range(scales):
        r_s = smallest_radius * (growth_factor)**s
        grid_size = r_s/density
        scale_tuples.append((grid_size, r_s))

    print(f"Scale tuples (grid, r): {scale_tuples}")
    return scale_tuples

def compute_covariance_matrix(neighbors):
    return np.cov(neighbors.T)

def compute_eigenvalues(covariance_matrix):
    #eigh returns them in ascending order
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    return eigenvalues, eigenvectors

def compute_omnivariance(lambda_1, lambda_2, lambda_3):
    #array = np.cbrt(np.prod(eigenvalues))
    value = np.cbrt(lambda_1 * lambda_2 * lambda_3)
    return np.float32(value)

def compute_eigenentropy(lambda_1, lambda_2, lambda_3):
    if lambda_1 <= 0 or lambda_2 <= 0 or lambda_3 <= 0:
        return 0.0
    else:
    # Add the constant to the eigenvalues before taking the logarithm
        value = -( (lambda_1*np.log(lambda_1)) + (lambda_2*np.log(lambda_2)) + (lambda_3*np.log(lambda_3)))
        return np.float32(value)

def compute_anisotropy(lambda_1, lambda_3):
    value = (lambda_1 - lambda_3) / lambda_1 if lambda_1 > 0. else 0.0
    return np.float32(value)

def compute_linearity(lambda_1, lambda_2):
    value = (lambda_1 - lambda_2) / lambda_1 if lambda_1 > 0. else 0.0
    return np.float32(value)

def compute_planarity(lambda_1, lambda_2, lambda_3):
    value = (lambda_2 - lambda_3) / lambda_1 if lambda_1 > 0. else 0.0
    return np.float32(value)

def compute_curvature(lambda_1, lambda_2, lambda_3):
    sum = lambda_1 + lambda_2 + lambda_3
    value = lambda_3 / sum if sum > 0. else 0.0
    return np.float32(value)

def compute_sphericity(lambda_1, lambda_3):
    value =  lambda_3 / lambda_1 if lambda_1 > 0. else 0.0
    return np.float32(value)

def compute_verticality(eigenvectors):
    third_eigenvector = eigenvectors[:, 0] #smallest eigenvector
    vertical_direction = np.array([0, 0, 1])
    dot_product = np.dot(third_eigenvector, vertical_direction)
    value = 1 - np.abs(dot_product)
    return np.float32(value)
    #old vertical measure 1 - nz
    # verticality_values = list(map(lambda point: 1 - point[3], points))
    # return np.round(np.array(verticality_values),decimals=decimal_digits)

def compute_moments(point, neighbors, eigenvectors):
    differences = neighbors - point
    first_vector = np.dot(differences, eigenvectors[:, 2]) #biggest
    second_vector = np.dot(differences, eigenvectors[:, 0]) #smallest
    first_order_first_vector = np.sum(first_vector).astype(np.float32)
    first_order_second_vector = np.sum(second_vector).astype(np.float32)
    second_order_first_vector = np.sum(first_vector**2).astype(np.float32)
    second_order_second_vector = np.sum(second_vector**2).astype(np.float32)
    return first_order_first_vector, first_order_second_vector, second_order_first_vector, second_order_second_vector

def compute_height(point,neighbors):
    z_point = point[2] # z of point
    z_neighbors = neighbors[:,2] # z of neighbors
    min, max = np.min(z_neighbors), np.max(z_neighbors)
    range = round(max - min, 2).astype(np.float32)
    average_height = round(np.mean(z_neighbors),2).astype(np.float32)
    height_below = round(z_point - min,2).astype(np.float32).astype(np.float32)
    height_above = round(max - z_point,2).astype(np.float32).astype(np.float32)
    return range, average_height, height_below, height_above

def compute_relative_height(rel_z_point,rel_z_neighbors):
    min, max = np.min(rel_z_neighbors), np.max(rel_z_neighbors)
    range = round(max - min, 2).astype(np.float32)
    average_height = round(np.mean(rel_z_neighbors),2).astype(np.float32)
    height_below = round(rel_z_point - min,2).astype(np.float32).astype(np.float32)
    height_above = round(max - rel_z_point,2).astype(np.float32).astype(np.float32)
    return range, average_height, height_below, height_above

def rgb_to_hsv(colors_array):
    return np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_array]).astype(np.float32)

def addDimsToLAS(laspyLASObject):
    dim_names = [f'omnivariance', #0
                 f'eigenentropy', #1
                 f'anisotropy', #2
                 f'linearity', #3
                 f'curvature', #4
                 f'sphericity',#5
                 f'planarity', #6
                 f'verticality', #7
                 f'F_1_vector', #8
                 f'F_2_vector', #9
                 f'S_1_vector', #10
                 f'S_2_vector'] #11
    
    data_type = np.float32
    #adding metadata to LAS
    laspyLASObject.add_extra_dims([laspy.ExtraBytesParams(name=dim_names[0], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[1], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[2], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[3], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[4], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[5], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[6], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[7], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[8], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[9], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[10], type=data_type),
                        laspy.ExtraBytesParams(name=dim_names[11], type=data_type)
                        ])
    return "dims added"

def saveDF_as_LAS(df,reference_LAS,output_file):
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
    with laspy.open(output_file, mode="w", header=header) as writer:

        point_record = laspy.ScaleAwarePointRecord.zeros(len(df.get('X')), header=header)
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
        point_record.F_1_vector = df.get('first_order_first_vector')
        point_record.F_2_vector = df.get('first_order_second_vector')
        point_record.S_1_vector = df.get('second_order_first_vector')
        point_record.S_2_vector = df.get('second_order_second_vector')
        #point_record.red = rgb_non_normalised[:, 0]
        #point_record.green = rgb_non_normalised[:, 1]
        #point_record.blue = rgb_non_normalised[:, 2]
        # point_record.RF = df.get('predictions_RF')
        # point_record.GBT = df.get('predictions_GBT')
        writer.write_points(point_record)

def saveNP_as_LAS(data_to_save,reference_LAS,output_file,RF_array, geom=False,geomList=[],height=False, heightList = []):
    # Create a new header
    header = laspy.LasHeader(point_format=reference_LAS.header.point_format, version=reference_LAS.header.version)
    header.offsets = reference_LAS.header.offsets
    header.scales = reference_LAS.header.scales
    # radius = str(0.5)
    header.add_extra_dim(laspy.ExtraBytesParams(name=f"RF", type=np.float32))
    #header.add_extra_dim(laspy.ExtraBytesParams(name=f"SVM", type=np.float32))
    if geom:
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Omnivariance", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Eigenentropy", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Anisotropy", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Linearity", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Planarity", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Curvature", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Sphericity", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Verticality", type=np.float32))
    if height:
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Relative", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Height_below", type=np.float32))
        header.add_extra_dim(laspy.ExtraBytesParams(name=f"Height_above", type=np.float32))
    #retrieve color info from las file
    # rgb_non_normalised = np.vstack((point_cloud.red,point_cloud.green,point_cloud.blue)).transpose() * 65535.0 
    # Create a LasWriter and a point record, then write it
    with laspy.open(output_file, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(data_to_save.shape[0], header=header)
        point_record.x = data_to_save[:, 0]
        point_record.y = data_to_save[:, 1]
        point_record.z = data_to_save[:, 2]
        # point_record['normal z'] = data_to_save[:, 3]
        #point_record.red = data_to_save[:, 4]
        #point_record.green = data_to_save[:, 5]
        #point_record.blue = data_to_save[:, 6]
        point_record.RF = RF_array
        #point_record.SVM = SVM_array
        if geom:
            point_record.Omnivariance = geomList[0]
            point_record.Eigenentropy = geomList[1]
            point_record.Anisotropy = geomList[2]
            point_record.Linearity = geomList[3]
            point_record.Planarity = geomList[4]
            point_record.Curvature = geomList[5]
            point_record.Sphericity = geomList[6]
            point_record.Verticality = geomList[7]
        if height:
            point_record.Relative = heightList[0]
            point_record.Height_below = heightList[1]
            point_record.Height_above = heightList[2]
        writer.write_points(point_record)

def calculateGeometricFeatures(data_array,neighborhood_radius,dtm = None, data_type = np.float32, loader=False, save=False, output_file=None, ref_las=None):
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
    verticalityList = np.zeros(pc_length, dtype=data_type)
    #moments
    first_order_first_vectorList = np.zeros(pc_length, dtype=data_type)
    first_order_second_vectorList = np.zeros(pc_length, dtype=data_type)
    second_order_first_vectorList = np.zeros(pc_length, dtype=data_type)
    second_order_second_vectorList = np.zeros(pc_length, dtype=data_type)
    #height values
    if dtm is not None:
        transform = dtm.transform
        heightRelativeList = np.zeros(pc_length, dtype=data_type)
    heightRangeList = np.zeros(pc_length, dtype=data_type)
    heightAvgList = np.zeros(pc_length, dtype=data_type)
    heightBelowList = np.zeros(pc_length, dtype=data_type)
    heightAboveList = np.zeros(pc_length, dtype=data_type)
    #color values
    neighboringHList = np.zeros(pc_length, dtype=data_type)
    neighboringSList = np.zeros(pc_length, dtype=data_type)
    neighboringVList = np.zeros(pc_length, dtype=data_type)
    # coordinate values for each point #
    xList = data_array[:, 0]
    yList = data_array[:, 1]
    zList = data_array[:, 2]
    redList = (data_array[:, 5] / 65535.0).astype(data_type)
    greenList = (data_array[:, 6] / 65535.0).astype(data_type)
    blueList = (data_array[:, 7] / 65535.0).astype(data_type)
    # color values #
    H_List = colors_hsv[:, 0].astype(data_type)
    S_List = colors_hsv[:, 1].astype(data_type)
    V_List = colors_hsv[:, 2].astype(data_type)
    scaler = MinMaxScaler()
    #Loops only once for all calculations according to neighbors
    for i, point in enumerate(translated_3d_color):
        indices = tree.query_ball_point(point[: 3], neighborhood_radius) #query just the coordinates XYZ coordinates and radius
        neighbors = translated_3d_color[indices]
        if dtm is not None: #calculates the height of the point relative to the ground
            row, col = rowcol(transform, point[0], point[1])
            dtm_value = dtm.read(1)[row, col]
            relative_z = point[2] - dtm_value
            heightRelativeList[i] = relative_z
            relative_heights = []
            # Loop through each neighbor point
            for neighbor in neighbors:
                # Convert XY coordinates to row and column in the DTM array
                rowN, colN = rowcol(transform, neighbor[0], neighbor[1])
                # Fetch the DTM value
                dtm_valueN = dtm.read(1)[rowN, colN]
                # Calculate the relative height by subtracting the DTM value from the Z-coordinate
                relative_height = neighbor[2] - dtm_valueN
                # Append the relative height to the list
                relative_heights.append(relative_height)
         # Need at least 4 points to compute a meaningful covariance matrix
        if len(neighbors) < 4:
            omniList[i] = 0.0
            eigenList[i] = 0.0
            anisoList[i] = 0.0
            linList[i] = 0.0
            planarList[i] = 0.0
            curveList[i] = 0.0
            sphereList[i] = 0.0
            heightRangeList[i] = 0.0
            heightAvgList[i] = 0.0
            heightBelowList[i] = 0.0
            heightAboveList[i] = 0.0
            neighboringHList[i] = 0.0
            neighboringSList[i] = 0.0
            neighboringVList[i] = 0.0
            verticalityList[i] = 0.0
            first_order_first_vectorList[i] = 0.0
            first_order_second_vectorList[i] = 0.0
            second_order_first_vectorList[i] = 0.0
            second_order_second_vectorList[i] = 0.0
        else:
            if dtm is not None:
                heightRange,average_height, heightBelow, heightAbove = compute_height(point, neighbors)
                #heightRange,average_height, heightBelow, heightAbove = compute_relative_height(relative_z, relative_heights)
            else:
                heightRange,average_height, heightBelow, heightAbove = compute_height(point, neighbors)

            cov_matrix = compute_covariance_matrix(neighbors[:, :3]).astype(data_type)
            eigenvalues,eigenvectors = compute_eigenvalues(cov_matrix)
            sum_eigenvalues = np.sum(eigenvalues) + 0.001 
            #normalise eigenvalues
            lambda_1 = eigenvalues[2] / sum_eigenvalues
            lambda_2 = eigenvalues[1] / sum_eigenvalues
            lambda_3 = eigenvalues[0] / sum_eigenvalues
            #Geometric features
            omni = compute_omnivariance(lambda_1, lambda_2, lambda_3)
            eigen = compute_eigenentropy(lambda_1, lambda_2, lambda_3)
            aniso = compute_anisotropy(lambda_1, lambda_3)
            linear = compute_linearity(lambda_1, lambda_2)
            planar = compute_planarity(lambda_1, lambda_2, lambda_3)
            curve = compute_curvature(lambda_1, lambda_2, lambda_3)
            sphere = compute_sphericity(lambda_1, lambda_3)
            verticality = compute_verticality(eigenvectors)
            #Moments
            first_order_first_vector, first_order_second_vector, second_order_first_vector, second_order_second_vector = compute_moments(point[: 3], neighbors[:, :3], eigenvectors)
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
            verticalityList[i] = verticality
            first_order_first_vectorList[i] = first_order_first_vector
            first_order_second_vectorList[i] = first_order_second_vector
            second_order_first_vectorList[i] = second_order_first_vector
            second_order_second_vectorList[i] = second_order_second_vector
        if loader:
            print_progress(i + 1, pc_length)
    #transforms data to values between 0 and 1
    #scaler.fit_transform()
    #Create a dictionary with all the values
    pointsDict_with_zeros = {
            "X": xList,
            "Y": yList,
            "Z": zList,
            "Z_scaled": scaler.fit_transform(zList.reshape(-1, 1)).ravel(),
            "H": H_List,
            "S": S_List,
            "V": V_List,
            "classification": (data_array[:, 4]),
            "normal z": (data_array[:, 3]),
            "omnivariance": omniList,
            "eigenentropy": scaler.fit_transform(eigenList.reshape(-1, 1)).ravel(),
            "anisotropy": anisoList,
            "linearity": linList,
            "planarity": planarList,
            "curvature": curveList,
            "sphericity": sphereList,
            "verticality": verticalityList,
            "first_order_first_vector": first_order_first_vectorList,
            "first_order_second_vector": first_order_second_vectorList,
            "second_order_first_vector": second_order_first_vectorList,
            "second_order_second_vector": second_order_second_vectorList,
            "height_range":scaler.fit_transform(heightRangeList.reshape(-1, 1)).ravel(),
            #"height_relative": heightRelativeList,#scaler.fit_transform(heightRelativeList.reshape(-1, 1)).ravel(),
            "height_avg": scaler.fit_transform(heightAvgList.reshape(-1, 1)).ravel(),
            "height_below": scaler.fit_transform(heightBelowList.reshape(-1, 1)).ravel(),
            "height_above": scaler.fit_transform(heightAboveList.reshape(-1, 1)).ravel(),
            "neighbor_H": neighboringHList,
            "neighbor_S": neighboringSList,
            "neighbor_V": neighboringVList,
            "red": redList,
            "green": greenList,
            "blue": blueList
        }
    if dtm is not None:
        pointsDict_with_zeros["height_relative"] = scaler.fit_transform(heightRelativeList.reshape(-1, 1)).ravel()
    # df = pd.DataFrame(pointsDict_with_nan)
    # df = df.dropna()
    # pointsDict = df.to_dict(orient='list')
    if save:
        output_path = '../results/testing/'
        if ref_las is None:
            df = df.astype('float32')
            df.to_csv(f"{output_file}_{neighborhood_radius}.csv", index=False)
        else:
            ref_las = laspy.read('../working/classification/multiscale/classified_sample.las')
            #saveDF_as_LAS(pd.DataFrame(pointsDict), ref_las, neighborhood_radius, output_path+output_file)
    return pointsDict_with_zeros

def classifyPointCloud(modelname, classified_features, nonClassified_features, features, labels, errorTXTPath,importances_png_path):
    processing_times = {'model':modelname}
    trainingBegin = time.time()
    
    # data to train on
    X_train, X_test, y_train, y_test = train_test_split(classified_features, labels, test_size=0.2, random_state=42)
    # RF model
    rf_model = RandomForestClassifier(n_estimators=50)
    # Train the models
    rf_model.fit(X_train, y_train)
    # Evaluate model
    y_pred_rf = rf_model.predict(X_test)
    # RF model report
    report_RF = classification_report(y_test, y_pred_rf)
    matrix_RF = confusion_matrix(y_test, y_pred_rf)
    accuracy_RF = accuracy_score(y_test, y_pred_rf)
    importances = rf_model.feature_importances_

    #Get accuracy results and write to file
    with open(errorTXTPath, 'w') as f:
        f.write('Classification Report for Random Forests:\n')
        f.write(report_RF)
        f.write('\nConfusion Matrix:\n')
        f.write(str(matrix_RF))
        f.write(f'\nAccuracy: {accuracy_RF * 100:.2f}%')
        f.write('\nFeature ranking:\n')
        for f_index in range(len(features)):
            f.write(f"{features[f_index]}: {importances[f_index]}\n")
    trainingEnd = time.time()
    predictions_RF = rf_model.predict(nonClassified_features)
    predictionsEnd = time.time()
    #save plot of importances
    try:
        #create dictionary
        combined_dict = {features[i]: importances[i] for i in range(len(features))}
        #sort the values
        sorted_dict = dict(sorted(combined_dict.items(), key=lambda item: item[1]))
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_dict.keys(), sorted_dict.values(), color='#0a9396')
        plt.style.use('fast')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        # plt.title('Importance of features')
        plt.xticks(rotation=45, ha='right')
        plt.savefig(importances_png_path, bbox_inches='tight')
    except:
        print('chart was not saved')
    processing_times['trainingTime'] = trainingEnd - trainingBegin
    processing_times['predictingTime'] = predictionsEnd - trainingEnd

    return processing_times, predictions_RF