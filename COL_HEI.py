import calculateFeatures
import numpy as np  # type: ignore
import laspy as lp  # type: ignore
import csv
import send_email
import time
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
import matplotlib.pyplot as plt # type: ignore


additional_text = "COL_HEI"
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
classified_pointCloudPath = '../working/training/classified_smaller.las' #change
nonClassified_pointCloudPath = '../working/training/lln_not_classified.las' #change

#create output txt files
outputErrorRF = f'../results_final/{additional_text}/rf_{additional_text}.txt'
# outputErrorSVM = '../results/error_SVM_multi_reduced_gpu_rf.txt'
#create output csv file
importances_path_png = f'../results_final/{additional_text}/rf_importances_{additional_text}.png'
output_path_csv = f'../results_final/{additional_text}/rf_{additional_text}.csv'
output_path_las = f'../results_final/{additional_text}/rf_{additional_text}.las'
print(f"Outpath: {output_path_las}")
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
# Calculate geometric features for both
classified_features_s1 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s1, radii[0])
classified_features_s2 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s2, radii[1])
classified_features_s3 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s3, radii[2])

print(f'Calculating geometric features for nonclassified pc... {get_time()}')
nonClassified_features_s1 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s1, radii[0])
nonClassified_features_s2 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s2, radii[1])
nonClassified_features_s3 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s3, radii[2])

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
classified_height_avg = np.concatenate([classified_features_s1.get('height_avg'),classified_features_s2.get('height_avg'),classified_features_s3.get('height_avg')])
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
nonClassified_height_avg = np.concatenate([nonClassified_features_s1.get('height_avg'),nonClassified_features_s2.get('height_avg'),nonClassified_features_s3.get('height_avg')])
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
                    # classified_omnivariance,
                    # classified_eigenentropy,
                    # classified_anisotropy,
                    # classified_linearity,
                    # classified_planarity,
                    # classified_curvature,
                    # classified_sphericity,
                    # classified_verticality,
                    classified_height_range,
                    classified_height_avg,
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
                    # nonClassified_omnivariance,
                    # nonClassified_eigenentropy,
                    # nonClassified_anisotropy,
                    # nonClassified_linearity,
                    # nonClassified_planarity,
                    # nonClassified_curvature,
                    # nonClassified_sphericity,
                    # nonClassified_verticality,
                    nonClassified_height_range,
                    nonClassified_height_avg,
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
        # 'omnivariance',
        # 'eigenentropy',
        # 'anisotropy',
        # "linearity",
        # "planarity",
        # "curvature",
        # "sphericity",
        # "verticality",
        "Height range",
        "Height mean",
        "Height below",
        "Height above",
        "Neighbor H",
        "Neighbor S",
        "Neighbor V",
        "H values",
        "S values",
        "V values"
    ]
# Labels

labels = np.concatenate([classified_features_s1.get('classification'),classified_features_s2.get('classification'),classified_features_s3.get('classification')])

times, predictions_RF = calculateFeatures.classifyPointCloud(additional_text, 
                                                             classified_features, 
                                                             nonClassified_features, 
                                                             features, 
                                                             labels, 
                                                             outputErrorRF,
                                                             importances_path_png)

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
    writer.writerow(times)


# predictions_SVM = svm_model.predict(nonClassified_features)
result_output_array= np.vstack((nonClassified_X,
                                nonClassified_Y,
                                nonClassified_Z,
                                predictions_RF,
                                nonClassified_verticality #place holder second ML values
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
                                    )
except Exception as e:
    print(e)
    send_email.sendNotification('Error in saving classified points as LAS')


done_time = time.time()

eval_time = round((done_time - start_read)/3600,2)
print(f"Evaluated time for {additional_text}: {eval_time} hours")