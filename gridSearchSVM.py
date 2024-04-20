import calculateFeatures
import numpy as np
import laspy as lp
import send_email
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

start_read = time.time()
# Get current current time
def get_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time

# FILE PATHS
#LAS files
print(f'Reading LAS files... {get_time()}')
classified_pointCloudPath = '../working/multiscale/classified_sample.las' #change
nonClassified_pointCloudPath = '../working/multiscale/nonClassified_sample.las' #change

# Read LAS data
send_email.sendUpdate('Script has begun. Reading LAS files...')
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
grid_sizes = [0.2, 0.4]
radii = [1.0, 2.0]

print(f'Subsampling classified pc... {get_time()}')
# Subsample the data
classified_subsampled_s1 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[0])
classified_subsampled_s2 = calculateFeatures.grid_subsampling_with_color(classified_points_array, grid_sizes[1])

print(f'Subsampling nonclassified pc... {get_time()}')
nonClassified_subsampled_s1 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[0])
nonClassified_subsampled_s2 = calculateFeatures.grid_subsampling_with_color(nonClassified_points_array, grid_sizes[1])
# send_email.sendUpdate('Subsampling done. Starting to calculate geometric features...')
print(f'Calculating geometric features for classified pc... {get_time()}')
# Calculate geometric features for both
classified_features_s1 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s1, radii[0])
classified_features_s2 = calculateFeatures.calculateGeometricFeatures(classified_subsampled_s2, radii[1])

print(f'Calculating geometric features for nonclassified pc... {get_time()}')
nonClassified_features_s1 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s1, radii[0])
nonClassified_features_s2 = calculateFeatures.calculateGeometricFeatures(nonClassified_subsampled_s2, radii[1])

print(f'Concatenating features... {get_time()}')
# Concatenate the features
classified_Z = np.concatenate([classified_features_s1.get('Z'),classified_features_s2.get('Z')])
classified_omnivariance = np.concatenate([classified_features_s1.get('omnivariance'),classified_features_s2.get('omnivariance')])
classified_eigenentropy = np.concatenate([classified_features_s1.get('eigenentropy'),classified_features_s2.get('eigenentropy')])
classified_anisotropy = np.concatenate([classified_features_s1.get('anisotropy'),classified_features_s2.get('anisotropy')])
classified_linearity = np.concatenate([classified_features_s1.get('linearity'),classified_features_s2.get('linearity')])
classified_planarity = np.concatenate([classified_features_s1.get('planarity'),classified_features_s2.get('planarity')])
classified_curvature = np.concatenate([classified_features_s1.get('curvature'),classified_features_s2.get('curvature')])
classified_sphericity = np.concatenate([classified_features_s1.get('sphericity'),classified_features_s2.get('sphericity')])
classified_verticality = np.concatenate([classified_features_s1.get('verticality'),classified_features_s2.get('verticality')])
classified_height_range = np.concatenate([classified_features_s1.get('height_range'),classified_features_s2.get('height_range')])
classified_height_avg = np.concatenate([classified_features_s1.get('height_avg'),classified_features_s2.get('height_avg')])
classified_height_below = np.concatenate([classified_features_s1.get('height_below'),classified_features_s2.get('height_below')])
classified_height_above = np.concatenate([classified_features_s1.get('height_above'),classified_features_s2.get('height_above')])
classified_neighbor_H = np.concatenate([classified_features_s1.get('neighbor_H'),classified_features_s2.get('neighbor_H')])
classified_neighbor_S = np.concatenate([classified_features_s1.get('neighbor_S'),classified_features_s2.get('neighbor_S')])
classified_neighbor_V = np.concatenate([classified_features_s1.get('neighbor_V'),classified_features_s2.get('neighbor_V')])
classified_H_values = np.concatenate([classified_features_s1.get('H'),classified_features_s2.get('H')])
classified_S_values = np.concatenate([classified_features_s1.get('S'),classified_features_s2.get('S')])
classified_V_values = np.concatenate([classified_features_s1.get('V'),classified_features_s2.get('V')])

print(f'Concatenating features... {get_time()}')
nonClassified_X = np.concatenate([nonClassified_features_s1.get('X'),nonClassified_features_s2.get('X')])
nonClassified_Y = np.concatenate([nonClassified_features_s1.get('Y'),nonClassified_features_s2.get('Y')])
nonClassified_Z = np.concatenate([nonClassified_features_s1.get('Z'),nonClassified_features_s2.get('Z')])
nonClassified_omnivariance = np.concatenate([nonClassified_features_s1.get('omnivariance'),nonClassified_features_s2.get('omnivariance')])
nonClassified_eigenentropy = np.concatenate([nonClassified_features_s1.get('eigenentropy'),nonClassified_features_s2.get('eigenentropy')])
nonClassified_anisotropy = np.concatenate([nonClassified_features_s1.get('anisotropy'),nonClassified_features_s2.get('anisotropy')])
nonClassified_linearity = np.concatenate([nonClassified_features_s1.get('linearity'),nonClassified_features_s2.get('linearity')])
nonClassified_planarity = np.concatenate([nonClassified_features_s1.get('planarity'),nonClassified_features_s2.get('planarity')])
nonClassified_curvature = np.concatenate([nonClassified_features_s1.get('curvature'),nonClassified_features_s2.get('curvature')])
nonClassified_sphericity = np.concatenate([nonClassified_features_s1.get('sphericity'),nonClassified_features_s2.get('sphericity')])
nonClassified_verticality = np.concatenate([nonClassified_features_s1.get('verticality'),nonClassified_features_s2.get('verticality')])
nonClassified_height_range = np.concatenate([nonClassified_features_s1.get('height_range'),nonClassified_features_s2.get('height_range')])
nonClassified_height_avg = np.concatenate([nonClassified_features_s1.get('height_avg'),nonClassified_features_s2.get('height_avg')])
nonClassified_height_below = np.concatenate([nonClassified_features_s1.get('height_below'),nonClassified_features_s2.get('height_below')])
nonClassified_height_above = np.concatenate([nonClassified_features_s1.get('height_above'),nonClassified_features_s2.get('height_above')])
nonClassified_neighbor_H = np.concatenate([nonClassified_features_s1.get('neighbor_H'),nonClassified_features_s2.get('neighbor_H')])
nonClassified_neighbor_S = np.concatenate([nonClassified_features_s1.get('neighbor_S'),nonClassified_features_s2.get('neighbor_S')])
nonClassified_neighbor_V = np.concatenate([nonClassified_features_s1.get('neighbor_V'),nonClassified_features_s2.get('neighbor_V')])
nonClassified_H_values = np.concatenate([nonClassified_features_s1.get('H'),nonClassified_features_s2.get('H')])
nonClassified_S_values = np.concatenate([nonClassified_features_s1.get('S'),nonClassified_features_s2.get('S')])
nonClassified_V_values = np.concatenate([nonClassified_features_s1.get('V'),nonClassified_features_s2.get('V')])

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
                    classified_Z,
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
                    nonClassified_omnivariance,
                    nonClassified_eigenentropy,
                    nonClassified_anisotropy,
                    nonClassified_linearity,
                    nonClassified_planarity,
                    nonClassified_curvature,
                    nonClassified_sphericity,
                    nonClassified_verticality,
                    nonClassified_Z,
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
# Labels
labels = np.concatenate([classified_features_s1.get('classification'),classified_features_s2.get('classification')])

send_email.sendUpdate('SVM grid search CV has begun')
# Train classifier
X_train, X_test, y_train, y_test = train_test_split(classified_features, labels, test_size=0.2, random_state=42)

# Machine learning models
svm = SVC()

# Train the models
print(f'Training models... {get_time()}')
# Define the parameter grid to search through
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.1, 0.01, 0.001, 0.0001],  # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly']  # Kernel type
}

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)



print(f'Evaluating models... {get_time()}')

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
accuracy = best_model.score(X_test, y_test)
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)

eval_time = round((time.time() - start_read)/3600,2)
print(f"Evaluated time: {eval_time} hours")
