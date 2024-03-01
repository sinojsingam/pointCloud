import numpy as np
import laspy
import sklearn
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

fp = os.environ['LAS_FOLDER_PATH']
laspath =  os.environ['LAS_FILE_PATH']
os.chdir(fp)
pc_file = f'{laspath}/lln_ground.las'

print('reading las file')
las = laspy.read(pc_file)
print('imported las file')

#x, y, z = las.x, las.y, las.z

points = np.vstack((las.x, las.y, las.z)).transpose()
colors = np.vstack((las.red, las.green, las.blue)).transpose()

# Geometric features extraction
def extract_geometric_features(points):
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(points)  # Fit using x, y, z coordinates
    distances, indices = neigh.kneighbors(points)
    avg_distance = np.mean(distances, axis=1)
    return avg_distance

# Color features extraction (assuming colors are in 16-bit)
def extract_color_features(colors):
    # Normalize colors to 8-bit if necessary
    colors_normalized = colors / 65535 * 255
    avg_color = np.mean(colors_normalized, axis=1)
    return avg_color

# Extract features
geometric_features = extract_geometric_features(points)
color_features = extract_color_features(colors)
print('kmeans is done')
# Combine features for classification input
features = np.column_stack((geometric_features, color_features))

print(features[:, 0])