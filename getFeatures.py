import laspy
import sys
import numpy as np
from scipy.spatial import cKDTree
import geometricFeatures
import time
import os
import send_email
import calculateFeatures
import colorsys
import resource
import pandas as pd

init_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Initial memory usage: {init_memory_usage / (1024 * 1024)} mb")


start = time.time()

input_las_path =sys.argv[1]

#check if input is a las file
if os.path.splitext(input_las_path)[-1].lower() == ".las":
    LAS_name_original = os.path.splitext(os.path.basename(input_las_path))[0]
    subfolder = LAS_name_original
    LAS_name = LAS_name_original.split('_')[0] + '_geom.csv'
    output_las_path = os.path.join('working',subfolder,LAS_name)
    print(f"Now calculating for: {LAS_name_original}...")
else:
    print("ERROR: input is not a las file, quitting.")
    exit()

radius = 0.5 #search radius

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
                 f'Color V', #13
                 f'NeighborColor H ({radius})', #14
                 f'NeighborColor S ({radius})', #15
                 f'NeighborColor V ({radius})',] #16

#if working doesnt exist, create it with
#subfolder geom else the func does nothing
try:
    geometricFeatures.createWorkingDir(sub_folder= subfolder)
except:
    output_las_path = os.path.join('working',LAS_name)
    print(f"! Creating subfolder didn't work, result is being saved in {output_las_path}")

las = laspy.read(input_las_path)

#add dimensions to las
#get info
point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
translated_coords = geometricFeatures.translate_coords(point_coords)
translated_3d = translated_coords[..., :-1]
colors_rgb = np.vstack((las.red,las.green,las.blue)).transpose() / 65535.0 #normalise
colors_hsv = np.round(np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_rgb]),decimals=2)
translated_3d_color = np.hstack([translated_3d, colors_hsv])
tree = cKDTree(translated_3d)

#GEOMETRIC

#initiating np array
#per neighbor
omniList = np.zeros(len(las.x))
eigenList = np.zeros(len(las.x))
anisoList = np.zeros(len(las.x))
linList = np.zeros(len(las.x))
planarList = np.zeros(len(las.x))
curveList = np.zeros(len(las.x))
sphereList = np.zeros(len(las.x))
heightRangeList = np.zeros(len(las.x))
heightBelowList = np.zeros(len(las.x))
heightAboveList = np.zeros(len(las.x))
neighboringHList = np.zeros(len(las.x))
neighboringSList = np.zeros(len(las.x))
neighboringVList = np.zeros(len(las.x))
#perpoint
xList = np.array(las.x)
yList = np.array(las.y)
zList = np.array(las.z)
H_List = colors_hsv[:,:1].ravel()
S_List = colors_hsv[:,1:2].ravel()
V_List = colors_hsv[:,2:3].ravel()
verticalityList = calculateFeatures.compute_verticality(translated_coords)

#loops only once for all calculations according to neighbors
for i, point in enumerate(translated_3d_color):
    indices = tree.query_ball_point(point[..., :3], radius) #query just the coordinates
    neighbors = translated_3d_color[indices]
    if len(neighbors) < 4:  # Need at least 4 points to compute a meaningful covariance matrix
        pass
    else:
        heightRange, heightBelow, heightAbove = calculateFeatures.compute_height(point, neighbors)
        cov_matrix = calculateFeatures.compute_covariance_matrix(neighbors[...,:3]) #just the coordinates are enough
        eigenvalues = calculateFeatures.compute_eigenvalues(cov_matrix)
        lambda_1, lambda_2, lambda_3 = eigenvalues #where l1>l2>l3
        omni = calculateFeatures.compute_omnivariance(eigenvalues)
        eigen = calculateFeatures.compute_eigenentropy(eigenvalues)
        aniso = calculateFeatures.compute_anisotropy(lambda_1, lambda_3)
        linear = calculateFeatures.compute_linearity(lambda_1, lambda_2)
        planar = calculateFeatures.compute_planarity(lambda_1, lambda_2, lambda_3)
        curve = calculateFeatures.compute_curvature(lambda_1, lambda_2, lambda_3)
        sphere = calculateFeatures.compute_sphericity(lambda_1, lambda_3)
        k_H, k_S, k_V = np.round(np.mean(neighbors[...,-3:], axis=0), decimals=2) #retrieve neighboring colors
        omniList[i] = omni
        eigenList[i] = eigen
        anisoList[i] = aniso
        linList[i] = linear
        planarList[i] = planar
        curveList[i] = curve
        sphereList[i] = sphere
        heightRangeList[i] = heightRange
        heightBelowList[i] = heightBelow
        heightAboveList[i] = heightAbove
        neighboringHList[i] = k_H
        neighboringSList[i] = k_S
        neighboringVList[i] = k_V


pointsDict = {
        "X": xList,
        "Y": yList,
        "Z": zList,
        "H": H_List,
        "S": S_List,
        "V": V_List,
        "classification": np.array(las.classification),
        "normal z": np.array(las['normal z']),
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

df = pd.DataFrame(pointsDict)
df.to_csv(output_las_path, sep=',')

end = time.time()
print_message=f'Time elapsed: {(end-start)/60} mins.'
print(print_message)
#add mailme to CLI and get an email notification sent when scipt is done
try:
    if len(sys.argv) >2:
        if sys.argv[2]=='mailme':
            send_email.sendNotification(f'Process finished. {print_message}')
except:
    print("mail was not send, due to API key error")

max_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Maximum memory usage: {max_memory_usage / (1024 * 1024)} mb")

print(f"Delta memory usage: {(max_memory_usage - init_memory_usage)/(1024 * 1024)} mb")


# import pandas as pd
# df = pd.read_csv('../working/car_training/car_car_training.csv')
# las_version = las.header.version
# point_format = las.header.point_format

# new_las = laspy.create(point_format=point_format, file_version=las_version)

# new_las.x = np.array(df['X']+667000.0) #add translation
# new_las.y = np.array(df['Y']+650000.0)
# new_las.z = np.array(df['Z'])
# new_las.red = np.array(las.red)
# new_las.green = np.array(las.green)
# new_las.blue = np.array(las.blue)
# new_las.classification = np.array(df['classification'])
# new_las.add_extra_dim(laspy.ExtraBytesParams(name='omnivariance', type=np.float64))
#geometricFeatures.addDimsToLAS(las,radius)
# new_las.omnivariance = np.array(df['omnivariance'])
# new_las.write('../working/car_training/car.las')