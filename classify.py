import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import laspy
import time

print("Reading...")
start_read = time.time()
full = laspy.read('results/lln_geom2.las')
training = laspy.read('results/training.las')

# for dimension in full.point_format.dimensions:
#     print(dimension.name)

training_labels = training.classification

features_train = np.vstack((training['Omnivariance (0.5)'], 
                            training['Eigenentropy (0.5)'],
                            training['Anisotropy (0.5)'],
                            training['Planarity (0.5)'],
                            training['Linearity (0.5)'],
                            training['1st order moment (0.5)'],
                            training['Surface variation (0.5)'],
                            training['Sphericity (0.5)'],
                            training['Verticality (0.5)'])).T

features_lln = np.vstack((full['Omnivariance (0.5)'], 
                            full['Eigenentropy (0.5)'],
                            full['Anisotropy (0.5)'],
                            full['Planarity (0.5)'],
                            full['Linearity (0.5)'],
                            full['1st order moment (0.5)'],
                            full['Surface variation2'],
                            full['Sphericity (0.5)'],
                            full['Verticality (0.5)'])).T

end_read = time.time()
print(f"Read in {(end_read-start_read)/60} mins. Now training...")
start_train = time.time()
# Random Forest classifier
clf = RandomForestClassifier(n_estimators=300)
clf.fit(features_train, training_labels)
# gradient boosted trees
gb_clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.2,
                                    max_depth=3, random_state=0)

end_train = time.time()
print(f"Trained in {(end_train-start_train)/60} mins. Now fitting...")

start_fit = time.time()
gb_clf.fit(features_train, training_labels)
end_fit = time.time()
print(f"Fit in {(end_fit-start_fit)/60} mins. Now predicting...")
start_pred = time.time()
# result

predictions_GBT = gb_clf.predict(features_lln)
predictions_RF = clf.predict(features_lln)
end_pred = time.time()
print(f"Fit in {(end_pred-start_pred)/60} mins. Now writing...")
# For evaluation
# accuracy = accuracy_score(full_dataset_labels, predictions)
# print("Accuracy:", accuracy)
# Assigning predictions back to full dataset points (for visualization or further analysis)
# add predictions to LAS data

full.add_extra_dims([laspy.ExtraBytesParams(name='RF', type=np.float64),
                    laspy.ExtraBytesParams(name='GBT', type=np.float64)
])

full['RF'] = predictions_RF
full['GBT'] = predictions_GBT

full.write("results/lln_classified.las")

done_time = time.time()

print(f"Writing took {(end_pred - done_time)/60} mins. Classification without color data is done.\nThe whole process elapsed {(done_time - start_read)/3600} hours.\nClassified LAS file saved in ./results/lln_classified.las.\nGoodbye.")