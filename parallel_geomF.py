import laspy
import sys
import numpy as np
from scipy.spatial import cKDTree
import geometricFeatures
import time
import os
import send_email
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


input_las_path = sys.argv[1]
subfolder = "geom"
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
R = 0.5 #search radius

point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
translated_coords = geometricFeatures.translate_coords(point_coords)
translated_3d = translated_coords[..., :-1]
tree = cKDTree(translated_3d)
# colors = np.vstack((las.red,las.green,las.blue)).transpose() / 65535.0 
# point_cloud = np.hstack((point_coords, colors))

#GEOMETRIC FEATURES CALCULATION
functions_to_run = [geometricFeatures.compute_omnivariance,
                    geometricFeatures.compute_eigenentropy,
                    geometricFeatures.compute_anisotropy,
                    geometricFeatures.compute_linearity,
                    geometricFeatures.compute_curvature,
                    geometricFeatures.compute_sphericity,
                    ]
#PARAMETERS
geometricFeatures.addDimsToLAS(las,R)

start = time.time()
results={}
# parallel processing
with ProcessPoolExecutor() as executor:
    #run in parallel
    futures = [executor.submit(func, translated_3d, tree, R) for func in functions_to_run]
    futures.append(executor.submit(geometricFeatures.compute_verticality, translated_coords))
    f, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    
    for future in futures:
        try:
            feature, result_array = future.result() # This blocks until the function completes
            #print(feature)
            results[feature] = result_array
            #las[feature] = result_array
            #verticality = list(map(lambda point: 1 - point[3], translated_coords))
            #print_message = f'Geometric calculations are done, time elapsed for functions: {(end - start)/60} mins.'
            # dim_names = [f'Omnivariance ({R})', f'Eigenentropy ({R})', f'Anisotropy ({R})',
            # f'Linearity ({R})', f'Curvature ({R})', f'Sphericity ({R})',f'Verticality ({R})']
            # print(f"Writing LAS file to {output_las_path}")
            # print('Done')
        except Exception as e:
            print(f"Function execution failed with error: {e}")
for k,v in results.items():
    las[k] = v
las.write(output_las_path)
end = time.time()
print_message=f"Parallel calculations are done, time elapsed: {round((end-start)/60,2)} mins."
print(print_message)
#add mailme to CLI and get an email notification sent when scipt is done
try:
    if len(sys.argv) >2:
        if sys.argv[2]=='mailme':
            send_email.sendNotification(f'Process finished. {print_message}')
except:
    print("mail was not send, due to API key error")
