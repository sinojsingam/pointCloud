import calculateFeatures
import numpy as np
import laspy as lp
import sys
import send_email
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
import seaborn as sns

classified_pointCloudPath = '../working/geom/classified_smaller.las' #change

# Read LAS data
# send_email.sendUpdate('Script has begun. Reading LAS files...')
classified_pointCloud = lp.read(classified_pointCloudPath)
classified_points_array = np.vstack((classified_pointCloud.x,
                               classified_pointCloud.y,
                               classified_pointCloud.z,
                               classified_pointCloud['normal z'],
                               classified_pointCloud.classification,
                               classified_pointCloud.red,
                               classified_pointCloud.green,
                               classified_pointCloud.blue)).transpose()


grid_sizes = [0.1, 0.2, 0.4]
radii = [0.5, 1.0, 2.0]
# Subsample the data
classified_subsampled_s3 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[2])
classified_features_s3 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s3, radii[2],loader=True)

df = pd.DataFrame(classified_features_s3)
filtered_df = df[df['classification'] == 6] #get only facades
data = pd.DataFrame({
    'Omnivariance': filtered_df['omnivariance'],
    'Verticality': filtered_df['verticality'],
    'Planarity': filtered_df['planarity'],
    'Eigenentropy': filtered_df['eigenentropy']
})
mapping = {
    2.0: 'Ground',
    3.0: 'Low Vegetation',
    4.0: 'Medium Vegetation',
    5.0: 'High Vegetation',
    6.0: 'Building',
    7.0: 'Facade',
    12.0: 'Vehicle'}
df['classification'] = df['classification'].map(mapping)

# plot HSV side to side

palette = ['#d4a373','#a3b18a','#588157','#344e41','#c0d0d5','#fefae0','#555555']
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.kdeplot(data=df, 
            x="H", 
            hue="classification", 
            palette=palette,
            multiple='stack',ax=axs[0],legend=False,lw=0.5)
axs[0].set_ylim(0, 4.5)
sns.despine(ax=axs[0])
sns.kdeplot(data=df, 
            x="S", 
            hue="classification", 
            palette=palette,
            multiple='stack',ax=axs[1],legend=False,lw=0.5).set_ylabel('')
axs[1].set_ylim(0, 4.5)
axs[1].set_yticks([])
sns.despine(ax=axs[1],left=True) 
sns.kdeplot(data=df, 
            x="V", 
            hue="classification", 
            palette=palette,
            multiple='stack',ax=axs[2],lw=0.5).set_ylabel('')
axs[2].set_ylim(0, 4.5)
axs[2].set_yticks([])
sns.despine(ax=axs[2],left=True)
# Display the figure
plt.tight_layout()
plt.show()