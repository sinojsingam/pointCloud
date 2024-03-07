import laspy
import sys
import numpy as np
from scipy.spatial import cKDTree
import geometricFeatures
import time
import os
import send_email
import calculateFeatures

start = time.time()

input_las_path =sys.argv[1]
subfolder = "test"
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
las = laspy.read(input_las_path)
radius = 0.5 #search radius
geometricFeatures.addDimsToLAS(las,radius)
point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
translated_coords = geometricFeatures.translate_coords(point_coords)
translated_3d = translated_coords[..., :-1]
tree = cKDTree(translated_3d)
# colors = np.vstack((las.red,las.green,las.blue)).transpose() / 65535.0 
# point_cloud = np.hstack((point_coords, colors))

#GEOMETRIC
omniList = []
eigenList= []
ansioList= []
linList= []
planarList= []
curveList= []
sphereList= []

for i, point in enumerate(translated_3d):
    indices = tree.query_ball_point(point, radius)
    neighbors = translated_3d[indices]
    if len(neighbors) < 4:  # Need at least 4 points to compute a meaningful covariance matrix
        omniList.append(0.)
        eigenList.append(0.)
        ansioList.append(0.)
        linList.append(0.)
        planarList.append(0.)
        curveList.append(0.)
        sphereList.append(0.)
        continue
    cov_matrix = calculateFeatures.compute_covariance_matrix(neighbors)
    eigenvalues = calculateFeatures.compute_eigenvalues(cov_matrix)
    lambda_1, lambda_2, lambda_3 = eigenvalues
    omni = calculateFeatures.compute_omnivariance(eigenvalues)
    eigen = calculateFeatures.compute_omnivariance(eigenvalues)
    aniso = calculateFeatures.compute_anisotropy(lambda_1, lambda_3)
    linear = calculateFeatures.compute_linearity(lambda_1, lambda_2, lambda_3)
    planar = calculateFeatures.compute_planarity(lambda_2, lambda_3)
    curve = calculateFeatures.compute_curvature(lambda_1, lambda_2, lambda_3)
    sphere = calculateFeatures.compute_sphericity(lambda_1, lambda_3)

    omniList.append(omni)
    eigenList.append(eigen)
    ansioList.append(aniso)
    linList.append(linear)
    planarList.append(planar)
    curveList.append(curve)
    sphereList.append(sphere)
    

dim_names = [f'Omnivariance ({radius})', f'Eigenentropy ({radius})', f'Anisotropy ({radius})',
    f'Linearity ({radius})', f'Curvature ({radius})', f'Sphericity ({radius})',f'Planarity ({radius})',f'Verticality']

# #write the calculated data onto the las file
las[dim_names[0]] = omniList
las[dim_names[1]] = eigenList
las[dim_names[2]] = ansioList
las[dim_names[3]] = linList
las[dim_names[4]] = curveList
las[dim_names[5]] = sphereList
las[dim_names[6]] = planarList
las[dim_names[7]] = calculateFeatures.compute_verticality(translated_coords)
# #output_las_path = '../working/geom_values/car_geom.las'
las.write(output_las_path)
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