import calculateFeatures
import numpy as np #type: ignore
import laspy as lp  #type: ignore
import csv
import send_email
import time
from sklearn.ensemble import RandomForestClassifier  #type: ignore
from sklearn.model_selection import train_test_split  #type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  #type: ignore
import matplotlib.pyplot as plt  #type: ignore
import rasterio  #type: ignore
import pandas as pd #type: ignore
import seaborn as sns #type: ignore

additional_text = "REL_HEI_TEST"
print(f"Classifying data for {additional_text}") #change
start_read = time.time()
# Get current current time
def get_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time
# FILE PATHS
#LAS files
print(f'Reading LAS files... {get_time()}')
classified_pointCloudPath = '../working/nonGroundClassification/offGround_classified.las' #change
nonClassified_pointCloudPath = '../working/nonGroundClassification/lln_nonGround.las'
#DTM files
dtmClassified = rasterio.open("../working/nonGroundClassification/merged_dtm.tif")
dtmNonClassified = rasterio.open("../working/nonGroundClassification/lln_ground_FILLED.tif")
#create output txt files
outputErrorRF = f'../results_final/{additional_text}/rf_{additional_text}.txt'
# outputErrorSVM = '../results/error_SVM_multi_reduced_gpu_rf.txt'
#create output csv file
output_path_png = f'../results_final/{additional_text}/rf_importances_{additional_text}.png'
output_path_csv = f'../results_final/{additional_text}/rf_{additional_text}.csv'
output_path_las = f'../results_final/{additional_text}/rf_{additional_text}.las'

#KDE plot variables
hue_order = [ 'Low Vegetation', 'Medium Vegetation', 'High Vegetation', 'Roof', 'Facade', 'Vehicle']
palette = ['#a3b18a','#588157','#344e41','#c0d0d5','#fefae0','#555555']
HSV_plot_path = f'../results_final/{additional_text}/HSV_plot_{additional_text}.png'
heights_plot_path = f'../results_final/{additional_text}/heights_plot_{additional_text}.png'
geomFeatures_plot_path = f'../results_final/{additional_text}/geomFeatures_plot_{additional_text}.png' 


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
#Scale tuples (grid, r): [(0.04, 0.2), (0.08, 0.4), (0.16, 0.8)] #accuracy 90%
#Scale tuples (grid, r): [(0.1, 0.5), (0.2, 1.0), (0.4, 2.0)] #accuracy 93% with 50 trees

grid_sizes = [0.1, 0.2, 0.4]
radii = [0.5, 1.0, 2.0]

print(f'Subsampling classified pc... {get_time()}')
# Subsample the data
classified_subsampled_s1 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[0])
classified_subsampled_s2 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[1])
classified_subsampled_s3 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[2])

print(f'Subsampling nonclassified pc... {get_time()}')
nonClassified_subsampled_s1 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[0])
nonClassified_subsampled_s2 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[1])
nonClassified_subsampled_s3 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[2])

print(f'Calculating geometric features for classified pc... {get_time()}')

# Calculate geometric features for both with DTM information
classified_features_s1 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s1, radii[0],dtm=dtmClassified)
classified_features_s2 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s2, radii[1],dtm=dtmClassified)
classified_features_s3 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s3, radii[2],dtm=dtmClassified)

print(f'Calculating geometric features for nonclassified pc... {get_time()}')
nonClassified_features_s1 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s1, radii[0],dtm=dtmNonClassified)
nonClassified_features_s2 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s2, radii[1],dtm=dtmNonClassified)
nonClassified_features_s3 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s3, radii[2],dtm=dtmNonClassified)

print(f'Concatenating features... {get_time()}')
# Concatenate the features for classified
classified_Z = np.concatenate([classified_features_s1.get('Z'),classified_features_s2.get('Z'),classified_features_s3.get('Z')])
classified_Z_scaled = np.concatenate([classified_features_s1.get('Z_scaled'),classified_features_s2.get('Z_scaled'),classified_features_s3.get('Z_scaled')])
classified_omnivariance = np.concatenate([classified_features_s1.get('omnivariance'),classified_features_s2.get('omnivariance'),classified_features_s3.get('omnivariance')])
classified_eigenentropy = np.concatenate([classified_features_s1.get('eigenentropy'),classified_features_s2.get('eigenentropy'),classified_features_s3.get('eigenentropy')])
classified_anisotropy = np.concatenate([classified_features_s1.get('anisotropy'),classified_features_s2.get('anisotropy'),classified_features_s3.get('anisotropy')])
classified_linearity = np.concatenate([classified_features_s1.get('linearity'),classified_features_s2.get('linearity'),classified_features_s3.get('linearity')])
classified_planarity = np.concatenate([classified_features_s1.get('planarity'),classified_features_s2.get('planarity'),classified_features_s3.get('planarity')])
classified_curvature = np.concatenate([classified_features_s1.get('curvature'),classified_features_s2.get('curvature'),classified_features_s3.get('curvature')])
classified_sphericity = np.concatenate([classified_features_s1.get('sphericity'),classified_features_s2.get('sphericity'),classified_features_s3.get('sphericity')])
classified_verticality = np.concatenate([classified_features_s1.get('verticality'),classified_features_s2.get('verticality'),classified_features_s3.get('verticality')])
classified_height_range = np.concatenate([classified_features_s1.get('height_range'),classified_features_s2.get('height_range'),classified_features_s3.get('height_range')])
classified_height_rel = np.concatenate([classified_features_s1.get('height_relative'),classified_features_s2.get('height_relative'),classified_features_s3.get('height_relative')])
classified_height_below = np.concatenate([classified_features_s1.get('height_below'),classified_features_s2.get('height_below'),classified_features_s3.get('height_below')])
classified_height_above = np.concatenate([classified_features_s1.get('height_above'),classified_features_s2.get('height_above'),classified_features_s3.get('height_above')])
classified_neighbor_H = np.concatenate([classified_features_s1.get('neighbor_H'),classified_features_s2.get('neighbor_H'),classified_features_s3.get('neighbor_H')])
classified_neighbor_S = np.concatenate([classified_features_s1.get('neighbor_S'),classified_features_s2.get('neighbor_S'),classified_features_s3.get('neighbor_S')])
classified_neighbor_V = np.concatenate([classified_features_s1.get('neighbor_V'),classified_features_s2.get('neighbor_V'),classified_features_s3.get('neighbor_V')])
classified_H_values = np.concatenate([classified_features_s1.get('H'),classified_features_s2.get('H'),classified_features_s3.get('H')])
classified_S_values = np.concatenate([classified_features_s1.get('S'),classified_features_s2.get('S'),classified_features_s3.get('S')])
classified_V_values = np.concatenate([classified_features_s1.get('V'),classified_features_s2.get('V'),classified_features_s3.get('V')])
# Concatenate the features for nonclassified
nonClassified_X = np.concatenate([nonClassified_features_s1.get('X'),nonClassified_features_s2.get('X'),nonClassified_features_s3.get('X')])
nonClassified_Y = np.concatenate([nonClassified_features_s1.get('Y'),nonClassified_features_s2.get('Y'),nonClassified_features_s3.get('Y')])
nonClassified_Z = np.concatenate([nonClassified_features_s1.get('Z'),nonClassified_features_s2.get('Z'),nonClassified_features_s3.get('Z')])
nonClassified_Z_scaled = np.concatenate([nonClassified_features_s1.get('Z_scaled'),nonClassified_features_s2.get('Z_scaled'),nonClassified_features_s3.get('Z_scaled')])

nonClassified_omnivariance = np.concatenate([nonClassified_features_s1.get('omnivariance'),nonClassified_features_s2.get('omnivariance'),nonClassified_features_s3.get('omnivariance')])
nonClassified_eigenentropy = np.concatenate([nonClassified_features_s1.get('eigenentropy'),nonClassified_features_s2.get('eigenentropy'),nonClassified_features_s3.get('eigenentropy')])
nonClassified_anisotropy = np.concatenate([nonClassified_features_s1.get('anisotropy'),nonClassified_features_s2.get('anisotropy'),nonClassified_features_s3.get('anisotropy')])
nonClassified_linearity = np.concatenate([nonClassified_features_s1.get('linearity'),nonClassified_features_s2.get('linearity'),nonClassified_features_s3.get('linearity')])
nonClassified_planarity = np.concatenate([nonClassified_features_s1.get('planarity'),nonClassified_features_s2.get('planarity'),nonClassified_features_s3.get('planarity')])
nonClassified_curvature = np.concatenate([nonClassified_features_s1.get('curvature'),nonClassified_features_s2.get('curvature'),nonClassified_features_s3.get('curvature')])
nonClassified_sphericity = np.concatenate([nonClassified_features_s1.get('sphericity'),nonClassified_features_s2.get('sphericity'),nonClassified_features_s3.get('sphericity')])
nonClassified_verticality = np.concatenate([nonClassified_features_s1.get('verticality'),nonClassified_features_s2.get('verticality'),nonClassified_features_s3.get('verticality')])
nonClassified_height_range = np.concatenate([nonClassified_features_s1.get('height_range'),nonClassified_features_s2.get('height_range'),nonClassified_features_s3.get('height_range')])
nonClassified_height_rel = np.concatenate([nonClassified_features_s1.get('height_relative'),nonClassified_features_s2.get('height_relative'),nonClassified_features_s3.get('height_relative')])
nonClassified_height_below = np.concatenate([nonClassified_features_s1.get('height_below'),nonClassified_features_s2.get('height_below'),nonClassified_features_s3.get('height_below')])
nonClassified_height_above = np.concatenate([nonClassified_features_s1.get('height_above'),nonClassified_features_s2.get('height_above'),nonClassified_features_s3.get('height_above')])
nonClassified_neighbor_H = np.concatenate([nonClassified_features_s1.get('neighbor_H'),nonClassified_features_s2.get('neighbor_H'),nonClassified_features_s3.get('neighbor_H')])
nonClassified_neighbor_S = np.concatenate([nonClassified_features_s1.get('neighbor_S'),nonClassified_features_s2.get('neighbor_S'),nonClassified_features_s3.get('neighbor_S')])
nonClassified_neighbor_V = np.concatenate([nonClassified_features_s1.get('neighbor_V'),nonClassified_features_s2.get('neighbor_V'),nonClassified_features_s3.get('neighbor_V')])
nonClassified_H_values = np.concatenate([nonClassified_features_s1.get('H'),nonClassified_features_s2.get('H'),nonClassified_features_s3.get('H')])
nonClassified_S_values = np.concatenate([nonClassified_features_s1.get('S'),nonClassified_features_s2.get('S'),nonClassified_features_s3.get('S')])
nonClassified_V_values = np.concatenate([nonClassified_features_s1.get('V'),nonClassified_features_s2.get('V'),nonClassified_features_s3.get('V')])

#Stack features for classification
print(f'Stacking features... {get_time()}')
classified_features = np.vstack((
                    classified_omnivariance,
                    classified_eigenentropy,
                    classified_anisotropy,
                    classified_linearity,
                    classified_planarity,
                    classified_curvature,
                    classified_sphericity,
                    classified_verticality,
                    classified_height_range,
                    classified_height_rel,
                    classified_height_below,
                    classified_height_above,
                    classified_neighbor_H,
                    classified_neighbor_S,
                    classified_neighbor_V,
                    classified_H_values,
                    classified_S_values,
                    classified_V_values
                    )).transpose()

nonClassified_features = np.vstack((
                    nonClassified_omnivariance,
                    nonClassified_eigenentropy,
                    nonClassified_anisotropy,
                    nonClassified_linearity,
                    nonClassified_planarity,
                    nonClassified_curvature,
                    nonClassified_sphericity,
                    nonClassified_verticality,
                    nonClassified_height_range,
                    nonClassified_height_rel,
                    nonClassified_height_below,
                    nonClassified_height_above,
                    nonClassified_neighbor_H,
                    nonClassified_neighbor_S,
                    nonClassified_neighbor_V,
                    nonClassified_H_values,
                    nonClassified_S_values,
                    nonClassified_V_values
                    )).transpose()
features = [
        'Omnivariance',
        'Eigenentropy',
        'Anisotropy',
        "Linearity",
        "Planarity",
        "Curvature",
        "Sphericity",
        "Verticality",
        "Height range",
        "Height relative",
        "Height below",
        "Height above",
        "Neighbor H",
        "Neighbor S",
        "Neighbor V",
        "Hue",
        "Saturation",
        "Value"
    ]
# Labels

labels = np.concatenate([classified_features_s1.get('classification'),classified_features_s2.get('classification'),classified_features_s3.get('classification')])

#-------#
send_email.sendUpdate(f'Classification has begun for {additional_text}')
# Train a classifier
processing_times = {'model':{additional_text}}
trainingBegin = time.time()
X_train, X_test, y_train, y_test = train_test_split(classified_features, labels, test_size=0.2, random_state=42)

# Machine learning model
rf_model = RandomForestClassifier(n_estimators=50)

# Train the models
print(f'Training models... {get_time()}')
rf_model.fit(X_train, y_train)
#svm_model.fit(X_train, y_train)
# Evaluate model
send_email.sendUpdate('Training done. Evaluating models...')
print(f'Evaluating models... {get_time()}')
y_pred_rf = rf_model.predict(X_test)
#y_pred_svm = svm_model.predict(X_test)
#Predict the non-classified area
# RF model report
print(f'Writing RF classification performance to file... {get_time()}')
report_RF = classification_report(y_test, y_pred_rf)
matrix_RF = confusion_matrix(y_test, y_pred_rf)
accuracy_RF = accuracy_score(y_test, y_pred_rf)
importances = rf_model.feature_importances_
#Get accuracy results and write to file
with open(outputErrorRF, 'w') as f:
    f.write('Classification Report for Random Forests:\n')
    f.write(report_RF)
    f.write('\nConfusion Matrix:\n')
    f.write(str(matrix_RF))
    f.write(f'\nAccuracy: {accuracy_RF * 100:.2f}%')
    f.write('\nFeature ranking:\n')
    for f_index in range(len(features)):
        f.write(f"{features[f_index]}: {importances[f_index]}\n")

print(f'Predicting non-classified pc... {get_time()}')
trainingEnd = time.time()
predictions_RF = rf_model.predict(nonClassified_features)
predictionsEnd = time.time()
processing_times['trainingTime'] = trainingEnd - trainingBegin
processing_times['predictingTime'] = predictionsEnd - trainingEnd
# predictions_SVM = svm_model.predict(nonClassified_features)
try:
    fieldnames = ["model", "trainingTime", "predictingTime"]

    # Open the CSV file in append mode
    with open("../working/times/times.csv", "a", newline='') as csvfile:
        # Create a DictWriter object, passing the file object and the fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Check if the file is empty to write the header
        csvfile.seek(0, 2)  # Move the cursor to the end of the file
        if csvfile.tell() == 0:
            # Write the header only if the file is empty
            writer.writeheader()
        
        # Write the dictionary to the CSV file
        writer.writerow(processing_times)
except:
    print('writing to times.csv did not work properly')
result_output_array= np.vstack((nonClassified_X,
                                nonClassified_Y,
                                nonClassified_Z,
                                predictions_RF,
                                )).T

print(f'Saving CSV file... {get_time()}')
#write to txt file in case las write didnt work
#np.savetxt(output_path_csv,result_output_array, delimiter=',',header='X,Y,Z,RF,GBT',comments='')

try:
    print(f'Saving classified points as LAS... {get_time()}')
    calculateFeatures.saveNP_as_LAS(result_output_array, # Array with X,Y,Z values
                                    nonClassified_pointCloud, # Reference pc with headers
                                    output_path_las, # output path
                                    predictions_RF, # RF values
                                    height=True,
                                    heightList=[nonClassified_height_rel,nonClassified_height_below,nonClassified_height_above]) #place holder second ML values
except Exception as e:
    print(e)
    send_email.sendNotification('Error in saving classified points as LAS')


#PLOTS
#save plot of importances
try:
    #create dictionary
    combined_dict = {features[i]: importances[i] for i in range(len(features))}
    #sort the values
    sorted_dict = dict(sorted(combined_dict.items(), key=lambda item: item[1]))
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_dict.keys(), sorted_dict.values())
    plt.style.use('fast')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Importance of features')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(output_path_png, bbox_inches='tight')
except:
    print('Importances chart was not saved')
try:
    extended_features = features.copy()
    extended_features.append('classification')

    # gch = geometric values, color values and height values
    # array that includes the gch and predictions for plotting
    full_value_array = np.vstack((nonClassified_features.T, predictions_RF)).T

    df = pd.DataFrame(full_value_array, columns=extended_features)
    # convert numeric labels to semantic labels
    semantic_labels = {
        #2.0: 'Ground',
        3.0: 'Low Vegetation',
        4.0: 'Medium Vegetation',
        5.0: 'High Vegetation',
        6.0: 'Roof',
        7.0: 'Facade',
        12.0: 'Vehicle'
        }
    df['classification'] = df['classification'].map(semantic_labels)

    # plot HSV side to side
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    #plot kernel density estimate
    sns.kdeplot(data=df, #data
                x="Hue", #value to plot
                hue="classification", #color by classification
                hue_order=hue_order, #order of classification
                palette=palette, #color palette
                multiple= 'stack', #stacked KDE
                legend=False, #no legend
                lw=0.5, #line width
                ax=axs1[0]) #plot on first subplot
    #axs1[0].set_ylim(0, 4.5) #set y-axis limits
    sns.despine(ax=axs1[0])
    sns.kdeplot(data=df, 
                x="Saturation", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs1[1],legend=False,lw=0.5).set_ylabel('')
    #axs1[1].set_ylim(0, 4.5)
    #axs1[1].set_yticks([])
    #sns.despine(ax=axs1[1],left=True)
    sns.despine(ax=axs1[1])  
    sns.kdeplot(data=df, 
                x="Value", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs1[2],lw=0.5).set_ylabel('')
    #axs1[2].set_ylim(0, 4.2)
    #axs1[2].set_yticks([])
    #sns.despine(ax=axs1[2],left=True)
    sns.despine(ax=axs1[2])
    # Display the figure
    fig1.tight_layout()
    fig1.savefig(HSV_plot_path)


    # plot heights
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    ylimit = 10.5
    sns.kdeplot(data=df,
                x="Height relative", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs2[0],legend=False,lw=0.5)
    #axs2[0].set_ylim(0, ylimit)
    sns.despine(ax=axs2[0])
    sns.kdeplot(data=df, 
                x="Height below", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs2[1],legend=False,lw=0.5).set_ylabel('')
    #axs2[1].set_ylim(0, ylimit)
    #axs2[1].set_yticks([])
    #sns.despine(ax=axs2[1],left=True) 
    sns.despine(ax=axs2[1]) 
    sns.kdeplot(data=df, 
                x="Height above", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs2[2],lw=0.5).set_ylabel('')
    #axs2[2].set_ylim(0, ylimit)
    #axs2[2].set_yticks([])
    #sns.despine(ax=axs2[2],left=True)
    sns.despine(ax=axs2[2])
    # Display the figure
    fig2.tight_layout()
    fig2.savefig(heights_plot_path)

    # plot geometric features
    fig3, axs3 = plt.subplots(4, 2, figsize=(15, 10))
    ylimit = 10.5
    sns.kdeplot(data=df,x="Omnivariance", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[0][0],legend=False,lw=0.5).set_ylabel('')
    #axs[0][0].set_ylim(0, ylimit)
    #axs[0][0].set_yticks([])
    sns.despine(ax=axs3[0][0])

    sns.kdeplot(data=df, 
                x="Eigenentropy", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[0][1],legend=False,lw=0.5).set_ylabel('')
    #axs[0][1].set_ylim(0, ylimit)
    #axs[0][1].set_yticks([])
    sns.despine(ax=axs3[0][1])

    sns.kdeplot(data=df, 
                x="Anisotropy", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[1][0],legend=False,lw=0.5).set_ylabel('')
    #axs[1][0].set_ylim(0, ylimit)
    #axs[1][0].set_yticks([])
    sns.despine(ax=axs3[1][0])

    sns.kdeplot(data=df, 
                x="Linearity", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[1][1],legend=False,lw=0.5).set_ylabel('')
    #axs[1][1].set_ylim(0, ylimit)
    #axs[1][1].set_yticks([])
    sns.despine(ax=axs3[1][1])

    sns.kdeplot(data=df, 
                x="Planarity",
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[2][0],legend=False,lw=0.5).set_ylabel('')
    #axs[2][0].set_ylim(0, ylimit)
    #axs[2][0].set_yticks([])
    sns.despine(ax=axs3[2][0])

    sns.kdeplot(data=df, 
                x="Curvature", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[2][1],lw=0.5).set_ylabel('')
    #axs3[2][1].set_ylim(0, ylimit)
    #axs[2][1].set_yticks([])
    sns.despine(ax=axs3[2][1])

    sns.kdeplot(data=df, 
                x="Sphericity", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[3][0],legend=False,lw=0.5).set_ylabel('')
    #axs[3][0].set_ylim(0, ylimit)
    #axs[3][0].set_yticks([])
    sns.despine(ax=axs3[3][0])

    sns.kdeplot(data=df,
                x="Verticality", 
                hue="classification",
                hue_order=hue_order,
                palette=palette,
                multiple='stack',ax=axs3[3][1],legend=False,lw=0.5).set_ylabel('')
    #axs[3][1].set_ylim(0, ylimit)
    #axs[3][1].set_yticks([])
    sns.despine(ax=axs3[3][1])

    # Display the figure
    fig3.tight_layout()
    fig3.savefig(geomFeatures_plot_path)
except:
    print('Density charts were not saved')


done_time = time.time()

eval_time = round((done_time - start_read)/3600,2)
print(f"Evaluated time for {additional_text}: {eval_time} hours")