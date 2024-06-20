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
hue_order= ['Ground', 'Low Vegetation', 'Medium Vegetation', 'High Vegetation', 'Building', 'Facade', 'Vehicle']
palette = ['#d4a373','#a3b18a','#588157','#344e41','#c0d0d5','#fefae0','#555555']

#paths
HSV_plot_path = f'../working/geom/HSV_plot_{additional_text}.png'
heights_plot_path = f'../working/geom/heights_plot_{additional_text}.png'
geomFeatures_plot_path = f'../working/geom/geomFeatures_plot_{additional_text}.png' 

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
mapping = {
    2.0: 'Ground',
    3.0: 'Low Vegetation',
    4.0: 'Medium Vegetation',
    5.0: 'High Vegetation',
    6.0: 'Roof',
    7.0: 'Facade',
    12.0: 'Vehicle'
    }

df['classification'] = df['classification'].map(mapping)

# plot HSV side to side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#plot kernel density estimate
sns.kdeplot(data=df, #data
            x="H", #value to plot
            hue="classification", #color by classification
            hue_order=hue_order, #order of classification
            palette=palette, #color palette
            multiple= 'stack', #stacked KDE
            legend=False, #no legend
            lw=0.5, #line width
            ax=axs[0]) #plot on first subplot
axs[0].set_ylim(0, 4.5) #set y-axis limits
sns.despine(ax=axs[0])
sns.kdeplot(data=df, 
            x="S", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[1],legend=False,lw=0.5).set_ylabel('')
axs[1].set_ylim(0, 4.5)
axs[1].set_yticks([])
sns.despine(ax=axs[1],left=True) 
sns.kdeplot(data=df, 
            x="V", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[2],lw=0.5).set_ylabel('')
axs[2].set_ylim(0, 4.2)
axs[2].set_yticks([])
sns.despine(ax=axs[2],left=True)
# Display the figure
plt.tight_layout()
plt.savefig(HSV_plot_path)
plt.clf()

# plot heights

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
ylimit = 10.5
sns.kdeplot(data=df,
            x="Z_scaled", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[0],legend=False,lw=0.5)
axs[0].set_ylim(0, ylimit)
sns.despine(ax=axs[0])
sns.kdeplot(data=df, 
            x="height_below", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[1],legend=False,lw=0.5).set_ylabel('')
axs[1].set_ylim(0, ylimit)
axs[1].set_yticks([])
sns.despine(ax=axs[1],left=True) 
sns.kdeplot(data=df, 
            x="height_above", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[2],lw=0.5).set_ylabel('')
axs[2].set_ylim(0, ylimit)
axs[2].set_yticks([])
sns.despine(ax=axs[2],left=True)
# Display the figure
plt.tight_layout()
plt.savefig(heights_plot_path)
plt.clf()

# plot geometric features

fig, axs = plt.subplots(4, 2, figsize=(15, 10))
ylimit = 10.5
sns.kdeplot(data=df,x="omnivariance", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[0][0],legend=False,lw=0.5).set_ylabel('')
#axs[0][0].set_ylim(0, ylimit)
#axs[0][0].set_yticks([])
sns.despine(ax=axs[0][0])

sns.kdeplot(data=df, 
            x="eigenentropy", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[0][1],legend=False,lw=0.5).set_ylabel('')
#axs[0][1].set_ylim(0, ylimit)
#axs[0][1].set_yticks([])
sns.despine(ax=axs[0][1])

sns.kdeplot(data=df, 
            x="anisotropy", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[1][0],legend=False,lw=0.5).set_ylabel('')
#axs[1][0].set_ylim(0, ylimit)
#axs[1][0].set_yticks([])
sns.despine(ax=axs[1][0])

sns.kdeplot(data=df, 
            x="linearity", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[1][1],legend=False,lw=0.5).set_ylabel('')
#axs[1][1].set_ylim(0, ylimit)
#axs[1][1].set_yticks([])
sns.despine(ax=axs[1][1])

sns.kdeplot(data=df, 
            x="planarity", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[2][0],legend=False,lw=0.5).set_ylabel('')
#axs[2][0].set_ylim(0, ylimit)
#axs[2][0].set_yticks([])
sns.despine(ax=axs[2][0])

sns.kdeplot(data=df, 
            x="curvature", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[2][1],lw=0.5).set_ylabel('')
axs[2][1].set_ylim(0, ylimit)
#axs[2][1].set_yticks([])
sns.despine(ax=axs[2][1])

sns.kdeplot(data=df, 
            x="sphericity", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[3][0],legend=False,lw=0.5).set_ylabel('')
#axs[3][0].set_ylim(0, ylimit)
#axs[3][0].set_yticks([])
sns.despine(ax=axs[3][0])

sns.kdeplot(data=df, 
            x="verticality", 
            hue="classification",
            hue_order=hue_order,
            palette=palette,
            multiple='stack',ax=axs[3][1],legend=False,lw=0.5).set_ylabel('')
#axs[3][1].set_ylim(0, ylimit)
#axs[3][1].set_yticks([])
sns.despine(ax=axs[3][1])

# Display the figure
plt.tight_layout()
plt.savefig(geomFeatures_plot_path)
