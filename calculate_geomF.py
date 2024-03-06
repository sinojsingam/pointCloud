import laspy
import sys
import numpy as np
from scipy.spatial import cKDTree
import geometricFeatures
import time
import os
#import send_email

input_las_path = sys.argv[1]
output_directory = sys.argv[2]
LAS_name = os.path.basename(input_las_path)[:-13]



def check_or_create_subfolder(main_folder="test", subfolder="geom"):
    # Check for the working folder
    main_folder_path = os.path.join(os.getcwd(), main_folder)
    if not os.path.exists(main_folder_path):
        os.mkdir(main_folder_path)
        print(f"Working folder '{main_folder}' created.")
    else:
        pass #already exists

    # Check for the subfolder geom within the working folder
    subfolder_path = os.path.join(main_folder_path, subfolder)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
        print(f"Subfolder '{subfolder}' created in '{main_folder}'.")
    else:
        pass #already exists


check_or_create_subfolder()



base_name = LAS_name + '_geom.las'
output_las_path = os.path.join(output_directory,base_name)



las = laspy.read(input_las_path)
R = 0.5
point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
translated_coords = geometricFeatures.translate_coords(point_coords)
translated_3d = translated_coords[..., :-1]
tree = cKDTree(translated_3d)

# colors = np.vstack((las.red,las.green,las.blue)).transpose() / 65535.0 
# point_cloud = np.hstack((point_coords, colors))

#GEOMETRIC FEATURES CALCULATION
start = time.time()
print('Calculating omnivariance...')
omnivariance = geometricFeatures.compute_omnivariance(translated_3d, tree, R)
omniTime = time.time()

print('Calculating eigenentropy...')
eigenentropy = geometricFeatures.compute_eigenentropy(translated_3d, tree, R)
eigenTime = time.time()

print('Calculating anisotropy...')
anisotropy = geometricFeatures.compute_anisotropy(translated_3d, tree, R)
anisoTime = time.time()

print('Calculating linearity...')
linearity = geometricFeatures.compute_linearity(translated_3d, tree, R)
linearityTime = time.time()

print('Calculating curvature...')
curvature = geometricFeatures.compute_curvature(translated_3d, tree, R)
surVarTime = time.time()

print('Calculating sphericity...')
sphericity = geometricFeatures.compute_sphericity(translated_3d, tree, R)
spherTime= time.time()

print('Calculating verticality...')
verticality = list(map(lambda point: 1 - point[3], translated_coords))
vertTime= time.time()
#print_message = f'Geometric calculations are done, time elapsed for functions: {(end - start)/60} mins.'
print_message = f"""
Geometric calculations are done. Time elapsed for functions: {round((vertTime - start)/60,2)} mins.
"""
times = [round((omniTime - start)/60,2), 
         round((eigenTime - omniTime)/60,2),
         round((anisoTime - eigenTime)/60,2),
         round((linearityTime - anisoTime)/60,2),
         round((surVarTime - linearityTime)/60,2),
         round((spherTime - surVarTime)/60,2),
         round((vertTime - spherTime)/60,2)]
geometricFeatures.printTimeElapsed(times)
print(print_message)

dim_names = [f'Omnivariance ({R})', f'Eigenentropy ({R})', f'Anisotropy ({R})',
f'Linearity ({R})', f'Curvature ({R})', f'Sphericity ({R})',f'Verticality ({R})']
print("Writing LAS file")
#adding metadata to LAS
las.add_extra_dims([laspy.ExtraBytesParams(name=dim_names[0], type=np.float64),
                    laspy.ExtraBytesParams(name=dim_names[1], type=np.float64),
                    laspy.ExtraBytesParams(name=dim_names[2], type=np.float64),
                    laspy.ExtraBytesParams(name=dim_names[3], type=np.float64),
                    laspy.ExtraBytesParams(name=dim_names[4], type=np.float64),
                    laspy.ExtraBytesParams(name=dim_names[5], type=np.float64),
                    laspy.ExtraBytesParams(name=dim_names[6], type=np.float64)
])

#write the calculated data onto the las file
las[dim_names[0]] = omnivariance
las[dim_names[1]] = eigenentropy
las[dim_names[2]] = anisotropy
las[dim_names[3]] = linearity
las[dim_names[4]] = curvature
las[dim_names[5]] = sphericity
las[dim_names[6]] = verticality

#output_las_path = '../working/geom_values/car_geom.las'
las.write(output_las_path)


#add mailme to CLI and get an email notification sent when scipt is done
# if len(sys.argv) >1:
#     if sys.argv[3]=='mailme':
#         send_email.sendNotification(f'Process finished. {print_message}')