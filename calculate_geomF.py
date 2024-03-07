import laspy
import sys
import numpy as np
from scipy.spatial import cKDTree
import geometricFeatures
import time
import os
import send_email

input_las_path = sys.argv[1]

#check if input is a las file
if os.path.splitext(input_las_path)[-1].lower() == ".las":
    LAS_name_original = os.path.splitext(os.path.basename(input_las_path))[0]
    LAS_name = LAS_name_original.split('_')[0] + '_geom.las'
    output_las_path = os.path.join('working','geom',LAS_name)
    print(f"Now calculating for: {LAS_name_original}...")
else:
    print("ERROR: input is not a las file, quitting.")
    exit()

#if working doesnt exist, create it with
#subfolder geom else the func does nothing
#geometricFeatures.createWorkingDir(sub_folder= "geom")
las = laspy.read(input_las_path)
R = 0.5
geometricFeatures.addDimsToLAS(las,R)
point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
translated_coords = geometricFeatures.translate_coords(point_coords)
translated_3d = translated_coords[..., :-1]
tree = cKDTree(translated_3d)

# colors = np.vstack((las.red,las.green,las.blue)).transpose() / 65535.0 
# point_cloud = np.hstack((point_coords, colors))

#GEOMETRIC FEATURES CALCULATION
start = time.time()
# print('Calculating omnivariance...')
omnivariance = geometricFeatures.compute_verticality(translated_coords)
# omniTime = time.time()

# print('Calculating eigenentropy...')
# eigenentropy = geometricFeatures.compute_eigenentropy(translated_3d, tree, R)
# eigenTime = time.time()

# print('Calculating anisotropy...')
# anisotropy = geometricFeatures.compute_anisotropy(translated_3d, tree, R)
# anisoTime = time.time()

# print('Calculating linearity...')
# linearity = geometricFeatures.compute_linearity(translated_3d, tree, R)
# linearityTime = time.time()

# print('Calculating curvature...')
# curvature = geometricFeatures.compute_curvature(translated_3d, tree, R)
# surVarTime = time.time()

# print('Calculating sphericity...')
# sphericity = geometricFeatures.compute_sphericity(translated_3d, tree, R)
# spherTime= time.time()

print('Calculating verticality...')
verticality = list(map(lambda point: 1 - point[3], translated_coords))
vertTime= time.time()
#print_message = f'Geometric calculations are done, time elapsed for functions: {(end - start)/60} mins.'
print_message = f"""
Geometric calculations are done. Time elapsed for functions: {round((vertTime - start)/60,2)} mins.
"""
# times = [round((omniTime - start)/60,2), 
#          round((eigenTime - omniTime)/60,2),
#          round((anisoTime - eigenTime)/60,2),
#          round((linearityTime - anisoTime)/60,2),
#          round((surVarTime - linearityTime)/60,2),
#          round((spherTime - surVarTime)/60,2),
#          round((vertTime - spherTime)/60,2)]

print(print_message)

dim_names = [f'Omnivariance ({R})', f'Eigenentropy ({R})', f'Anisotropy ({R})',
f'Linearity ({R})', f'Curvature ({R})', f'Sphericity ({R})',f'Verticality ({R})']
print(f"Writing LAS file to {output_las_path}")

#write the calculated data onto the las file
las[dim_names[0]] = omnivariance
# las[dim_names[1]] = eigenentropy[1]
# las[dim_names[2]] = anisotropy[1]
# las[dim_names[3]] = linearity[1]
# las[dim_names[4]] = curvature[1]
# las[dim_names[5]] = sphericity[1]
las[dim_names[6]] = verticality

#output_las_path = '../working/geom_values/car_geom.las'
las.write(output_las_path)


#add mailme to CLI and get an email notification sent when scipt is done
try:
    if len(sys.argv) >2:
        if sys.argv[2]=='mailme':
            send_email.sendNotification(f'Process finished. {print_message}')
except:
    print("mail was not send, due to API key error")