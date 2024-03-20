import laspy as lp
import numpy as np
import calculateFeatures
from scipy.spatial import cKDTree
import pandas as pd
import colorsys

#read file
input_path = '../working/classification/multiscale/'
dataname = 'classified_sample.las'
point_cloud = lp.read(input_path+dataname)
points = np.vstack((point_cloud.x, point_cloud.y,point_cloud.z,point_cloud['normal z'],point_cloud.classification,point_cloud.red, point_cloud.green,point_cloud.blue)).transpose()
# colors = np.vstack((point_cloud.red, point_cloud.green,point_cloud.blue)).transpose()

#subsample
data_array = calculateFeatures.grid_subsampling_with_color(points, 0.64) #grid size
#calculate features
dictResult = calculateFeatures.calculateGeometricFeatures(data_array, 3.2) #radius
#save to csv
df = pd.DataFrame(dictResult)
df.to_csv('../working/classification/multiscale/multiscale_features.csv', index=False)
