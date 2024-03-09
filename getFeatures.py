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


init_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Initial memory usage: {init_memory_usage / (1024 * 1024)} mb")


start = time.time()

input_las_path =sys.argv[1]
subfolder = "car"
#check if input is a las file
if os.path.splitext(input_las_path)[-1].lower() == ".las":
    LAS_name_original = os.path.splitext(os.path.basename(input_las_path))[0]
    LAS_name = LAS_name_original.split('_')[0] + f'_{subfolder}.las'
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
geometricFeatures.createWorkingDir(sub_folder= subfolder)
las = laspy.read(input_las_path)

#add dimensions to las
#geometricFeatures.addDimsToLAS(las,radius)
#get info
point_coords = np.vstack((las.x, las.y, las.z, las['normal z'])).transpose()
translated_coords = geometricFeatures.translate_coords(point_coords)
translated_3d = translated_coords[..., :-1]
colors_rgb = np.vstack((las.red,las.green,las.blue)).transpose() / 65535.0 #normalise
colors_hsv = np.round(np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors_rgb]),decimals=2)
translated_3d_color = np.hstack([translated_3d, colors_hsv])
tree = cKDTree(translated_3d)

#GEOMETRIC
omniList = []
eigenList = []
anisoList = []
linList = []
planarList = []
curveList = []
sphereList = []
heighRangeList = []
heighBelowList = []
heighAboveList = []
neighboringHList = []
neighboringSList = []
neighboringVList = []
xList = []
yList = []
zList = []
#loops only once for all calculations
for i, point in enumerate(translated_3d_color):
    indices = tree.query_ball_point(point[..., :3], radius)
    neighbors = translated_3d_color[indices]
    if len(neighbors) < 4:  # Need at least 4 points to compute a meaningful covariance matrix
        omniList.append(0.)
        eigenList.append(0.)
        anisoList.append(0.)
        linList.append(0.)
        planarList.append(0.)
        curveList.append(0.)
        sphereList.append(0.)
        heighRangeList.append(0.)
        heighBelowList.append(0.)
        heighAboveList.append(0.)
        neighboringHList.append(0.)
        neighboringSList.append(0.)
        neighboringVList.append(0.)
        continue
    heighRange, heighBelow, heighAbove = calculateFeatures.compute_height(point, neighbors)
    cov_matrix = calculateFeatures.compute_covariance_matrix(neighbors[...,:3]) #just the coordinates
    eigenvalues = calculateFeatures.compute_eigenvalues(cov_matrix)
    lambda_1, lambda_2, lambda_3 = eigenvalues #l1>l2>l3
    omni = calculateFeatures.compute_omnivariance(eigenvalues)
    eigen = calculateFeatures.compute_eigenentropy(eigenvalues)
    aniso = calculateFeatures.compute_anisotropy(lambda_1, lambda_3)
    linear = calculateFeatures.compute_linearity(lambda_1, lambda_2)
    planar = calculateFeatures.compute_planarity(lambda_1, lambda_2, lambda_3)
    curve = calculateFeatures.compute_curvature(lambda_1, lambda_2, lambda_3)
    sphere = calculateFeatures.compute_sphericity(lambda_1, lambda_3)
    k_H, k_S, k_V = np.round(np.mean(neighbors[...,-3:], axis=0), decimals=2) #retrieve neighboring colors
    omniList.append(omni)
    eigenList.append(eigen)
    anisoList.append(aniso)
    linList.append(linear)
    planarList.append(planar)
    curveList.append(curve)
    sphereList.append(sphere)
    heighRangeList.append(heighRange)
    heighBelowList.append(heighBelow)
    heighAboveList.append(heighAbove)
    neighboringHList.append(k_H)
    neighboringSList.append(k_S)
    neighboringVList.append(k_V)
    xList.append(point[0])
    yList.append(point[1])
    zList.append(point[2])


#output_las_path = os.path.join('working',subfolder,"omnivariance.csv")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_x.csv"), xList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_y.csv"), yList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_z.csv"), zList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_omnivariance.csv"), omniList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_eigen.csv"), eigenList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_aniso.csv"), anisoList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_linear.csv"), linList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_planar.csv"), planarList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_curvature.csv"), curveList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_spheri.csv"), sphereList,fmt='%.8f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_heightRange.csv"), heighRangeList,fmt='%.2f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_heightBelow.csv"), heighBelowList,fmt='%.2f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_heightAbove.csv"), heighAboveList,fmt='%.2f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_neighboringHList.csv"), neighboringHList,fmt='%.2f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_neighboringSList.csv"), neighboringSList,fmt='%.2f', delimiter=",")
np.savetxt(os.path.join('working',subfolder,f"{LAS_name_original}_neighboringVList.csv"), neighboringVList,fmt='%.2f', delimiter=",")

#write the calculated data onto the las file
# las[dim_names[0]] = omniList
# las[dim_names[1]] = eigenList
# las[dim_names[2]] = ansioList
# las[dim_names[3]] = linList
# las[dim_names[4]] = curveList
# las[dim_names[5]] = sphereList
# las[dim_names[6]] = planarList
# # verticality is independent of the neighbors 1 - normal z
# # using translated_coords since it has the normal z
# las[dim_names[7]] = calculateFeatures.compute_verticality(translated_coords)
# las[dim_names[8]] = heighRangeList
# las[dim_names[9]] = heighBelowList
# las[dim_names[10]] = heighAboveList
# #HSV colors
# las[dim_names[11]] = colors_hsv[:,0] #H
# las[dim_names[12]] = colors_hsv[:,1] #S
# las[dim_names[13]] = colors_hsv[:,2] #V

# las[dim_names[14]] = neighboringHList #H
# las[dim_names[15]] = neighboringSList #S
# las[dim_names[16]] = neighboringVList #V
#las.write(output_las_path)
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