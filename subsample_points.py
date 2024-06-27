import laspy as lp # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import calculateFeatures

classified_pointCloudPath = '../working/training/classified_smaller.las' #change
nonClassified_pointCloudPath = '../working/training/lln_clean.las' #change

#create output txt files
outputErrorRF = '../results/error_RF_multi.txt'
outputErrorSVM = '../results/error_SVM_multi.txt'
#create output csv file
output_path = '../results/multiscale_classified_points.csv'

# Read LAS data
classified_pointCloud = lp.read(classified_pointCloudPath)
nonClassified_pointCloud = lp.read(nonClassified_pointCloudPath)
classified_points_array = np.vstack((classified_pointCloud.x,
                               classified_pointCloud.y,
                               classified_pointCloud.z,
                               classified_pointCloud['normal z'],
                               classified_pointCloud.classification,
                               classified_pointCloud.red,
                               classified_pointCloud.green,
                               classified_pointCloud.blue)).transpose()

nonClassified_pointCloud = lp.read(nonClassified_pointCloudPath)
nonClassified_points_array = np.vstack((nonClassified_pointCloud.x,
                               nonClassified_pointCloud.y,
                               nonClassified_pointCloud.z,
                               nonClassified_pointCloud['normal z'],
                               nonClassified_pointCloud.classification,
                               nonClassified_pointCloud.red,
                               nonClassified_pointCloud.green,
                               nonClassified_pointCloud.blue)).transpose()
# Scales and Radii
#grid_sizes = [0.2, 0.4]
#radii = [1.0, 2.0]
grid_sizes = [0.1, 0.2, 0.4]
radii = [0.5, 1.0, 2.0]
# Subsample the data
classified_subsampled_s1 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[0])
classified_subsampled_s2 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[1])
classified_subsampled_s3 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[2])

nonClassified_subsampled_s1 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[0])
nonClassified_subsampled_s2 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[1])
nonClassified_subsampled_s3 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[2])


# Data for the number of points in each subsampled point cloud
classified_scales = [classified_subsampled_s1.shape[0], classified_subsampled_s2.shape[0], classified_subsampled_s3.shape[0]]
non_classified_scales = [nonClassified_subsampled_s1.shape[0], nonClassified_subsampled_s2.shape[0], nonClassified_subsampled_s3.shape[0]]

# Original point counts
original_points = [classified_points_array.shape[0], nonClassified_points_array.shape[0]]

# Labels for the x-axis
labels = ['Training point cloud', 'Predicted point cloud']

# Stack the data for each point cloud
data = np.array([classified_scales, non_classified_scales]).T

# Create the bar graph
fig, ax = plt.subplots(figsize=(5, 5))

# Number of bars
N = len(labels)

# X locations for the groups
ind = np.arange(N)

# Define the width of the bars and the gap
bar_width = 0.35
gap = 0.1

# Plot original points
ax.bar(ind - (bar_width + gap) / 2, original_points, width=bar_width, label='Original Points', color='gray')

# Plot each stack
bottoms = np.zeros(N)
for i in range(data.shape[0]):
    ax.bar(ind + (bar_width + gap) / 2, data[i], width=bar_width, bottom=bottoms, label=f'Scale {i+1}')
    bottoms += data[i]

# Add labels, title, and adjust layout
ax.set_ylabel('Number of Points', color='#4D4D4D')
#ax.set_title('Number of Points in Original and Subsampled Point Clouds')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
plt.xticks(rotation=45, ha='right')
# Position the legend outside the plot
ax.legend(bbox_to_anchor=(1.02, .3), loc='upper left', borderaxespad=0.,edgecolor='white', labelcolor='#4D4D4D')
# Adjust layout to make space for the legend
#plt.tight_layout(rect=[0, 0, 0.75, 1])
ax.tick_params(colors='#4D4D4D')
#set color for the graph borders
for spine in ax.spines.values():
    spine.set_color('#4D4D4D')
# Show the plot
fig.savefig('../results_final/subsample/numPoints.png')