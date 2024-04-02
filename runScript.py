
if __name__ == '__main__':
    import calculateFeatures
    import numpy as np
    import pandas as pd
    import laspy as lp
    import sys
    import send_email
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    start_read = time.time()
    
    #Paths for the pre-calculated features for the nonclassified area at 0.5, 1, 2 spherical neighborhood
    path_nonC_1 = "../working/chunks/features_0_5.csv"
    path_nonC_2 = "../working/chunks/features_1_0.csv"
    path_nonC_3 = "../working/chunks/features_2_0.csv"
    #classified point cloud path
    input_path = '../working/classification/multiscale/classified_sample.las'

    #read the pre-calculated features for the non-classified area
    nonC_features_s1 = pd.read_csv(path_nonC_1)
    nonC_features_s2 = pd.read_csv(path_nonC_2)
    nonC_features_s3 = pd.read_csv(path_nonC_3)

    #path for the classified point cloud, read it and get the dimensions for calculation
    point_cloud = lp.read(input_path)
    points = np.vstack((point_cloud.x, point_cloud.y,point_cloud.z,point_cloud['normal z'],point_cloud.classification,point_cloud.red, point_cloud.green,point_cloud.blue)).transpose()
    
    #subsample the point cloud at different grid sizes (_s1,_s2,_s3 = scale 1,2,3)
    data_array_s1 = calculateFeatures.grid_subsampling_with_color(points,0.1)
    data_array_s2 = calculateFeatures.grid_subsampling_with_color(points,0.2)
    data_array_s3 = calculateFeatures.grid_subsampling_with_color(points,0.4)

    #calculate the features for each scale at different radii
    features_s1 = calculateFeatures.calculateGeometricFeatures(data_array_s1,0.5)
    features_s2 = calculateFeatures.calculateGeometricFeatures(data_array_s2,1.0)
    features_s3 = calculateFeatures.calculateGeometricFeatures(data_array_s3,2.0)

    #concatenate the features calculated
    omnivariance = np.concatenate([features_s1.get('omnivariance'),features_s2.get('omnivariance'),features_s3.get('omnivariance')])
    eigenentropy = np.concatenate([features_s1.get('eigenentropy'),features_s2.get('eigenentropy'),features_s3.get('eigenentropy')])
    anisotropy = np.concatenate([features_s1.get('anisotropy'),features_s2.get('anisotropy'),features_s3.get('anisotropy')])
    linearity = np.concatenate([features_s1.get('linearity'),features_s2.get('linearity'),features_s3.get('linearity')])
    planarity = np.concatenate([features_s1.get('planarity'),features_s2.get('planarity'),features_s3.get('planarity')])
    curvature = np.concatenate([features_s1.get('curvature'),features_s2.get('curvature'),features_s3.get('curvature')])
    sphericity = np.concatenate([features_s1.get('sphericity'),features_s2.get('sphericity'),features_s3.get('sphericity')])
    verticality = np.concatenate([features_s1.get('verticality'),features_s2.get('verticality'),features_s3.get('verticality')])
    height_range = np.concatenate([features_s1.get('height_range'),features_s2.get('height_range'),features_s3.get('height_range')])
    height_avg = np.concatenate([features_s1.get('height_avg'),features_s2.get('height_avg'),features_s3.get('height_avg')])
    height_below = np.concatenate([features_s1.get('height_below'),features_s2.get('height_below'),features_s3.get('height_below')])
    height_above = np.concatenate([features_s1.get('height_above'),features_s2.get('height_above'),features_s3.get('height_above')])
    neighbor_H = np.concatenate([features_s1.get('neighbor_H'),features_s2.get('neighbor_H'),features_s3.get('neighbor_H')])
    neighbor_S = np.concatenate([features_s1.get('neighbor_S'),features_s2.get('neighbor_S'),features_s3.get('neighbor_S')])
    neighbor_V = np.concatenate([features_s1.get('neighbor_V'),features_s2.get('neighbor_V'),features_s3.get('neighbor_V')])
    H_values = np.concatenate([features_s1.get('H'),features_s2.get('H'),features_s3.get('H')])
    S_values = np.concatenate([features_s1.get('S'),features_s2.get('S'),features_s3.get('S')])
    V_values = np.concatenate([features_s1.get('V'),features_s2.get('V'),features_s3.get('V')])
    #stack for machine learning input
    features = np.vstack((omnivariance,
                        eigenentropy,
                        anisotropy,
                        linearity,
                        planarity,
                        curvature,
                        sphericity,
                        verticality,
                        height_range,
                        height_avg,
                        height_below,
                        height_above,
                        neighbor_H,
                        neighbor_S,
                        neighbor_V,
                        H_values,
                        S_values,
                        V_values)).transpose()
    
    #concatenate classification labels from the classified point cloud
    labels = np.concatenate([features_s1.get('classification'),features_s2.get('classification'),features_s3.get('classification')])

    #get the features that were precalculated for the non-classified area
    omnivariance_nonC = np.concatenate([nonC_features_s1.get('omnivariance'),nonC_features_s2.get('omnivariance'),nonC_features_s3.get('omnivariance')])
    eigenentropy_nonC = np.concatenate([nonC_features_s1.get('eigenentropy'),nonC_features_s2.get('eigenentropy'),nonC_features_s3.get('eigenentropy')])
    anisotropy_nonC = np.concatenate([nonC_features_s1.get('anisotropy'),nonC_features_s2.get('anisotropy'),nonC_features_s3.get('anisotropy')])
    linearity_nonC = np.concatenate([nonC_features_s1.get('linearity'),nonC_features_s2.get('linearity'),nonC_features_s3.get('linearity')])
    planarity_nonC = np.concatenate([nonC_features_s1.get('planarity'),nonC_features_s2.get('planarity'),nonC_features_s3.get('planarity')])
    curvature_nonC = np.concatenate([nonC_features_s1.get('curvature'),nonC_features_s2.get('curvature'),nonC_features_s3.get('curvature')])
    sphericity_nonC = np.concatenate([nonC_features_s1.get('sphericity'),nonC_features_s2.get('sphericity'),nonC_features_s3.get('sphericity')])
    verticality_nonC = np.concatenate([nonC_features_s1.get('verticality'),nonC_features_s2.get('verticality'),nonC_features_s3.get('verticality')])
    height_range_nonC = np.concatenate([nonC_features_s1.get('height_range'),nonC_features_s2.get('height_range'),nonC_features_s3.get('height_range')])
    height_avg_nonC = np.concatenate([nonC_features_s1.get('height_avg'),nonC_features_s2.get('height_avg'),nonC_features_s3.get('height_avg')])
    height_below_nonC = np.concatenate([nonC_features_s1.get('height_below'),nonC_features_s2.get('height_below'),nonC_features_s3.get('height_below')])
    height_above_nonC = np.concatenate([nonC_features_s1.get('height_above'),nonC_features_s2.get('height_above'),nonC_features_s3.get('height_above')])
    neighbor_H_nonC = np.concatenate([nonC_features_s1.get('neighbor_H'),nonC_features_s2.get('neighbor_H'),nonC_features_s3.get('neighbor_H')])
    neighbor_S_nonC = np.concatenate([nonC_features_s1.get('neighbor_S'),nonC_features_s2.get('neighbor_S'),nonC_features_s3.get('neighbor_S')])
    neighbor_V_nonC = np.concatenate([nonC_features_s1.get('neighbor_V'),nonC_features_s2.get('neighbor_V'),nonC_features_s3.get('neighbor_V')])
    H_values_nonC = np.concatenate([nonC_features_s1.get('H'),nonC_features_s2.get('H'),nonC_features_s3.get('H')])
    S_values_nonC = np.concatenate([nonC_features_s1.get('S'),nonC_features_s2.get('S'),nonC_features_s3.get('S')])
    V_values_nonC = np.concatenate([nonC_features_s1.get('V'),nonC_features_s2.get('V'),nonC_features_s3.get('V')])
    
    #stack for machine learning input
    nonC_features = np.vstack((omnivariance_nonC,
                        eigenentropy_nonC,
                        anisotropy_nonC,
                        linearity_nonC,
                        planarity_nonC,
                        curvature_nonC,
                        sphericity_nonC,
                        verticality_nonC,
                        height_range_nonC,
                        height_avg_nonC,
                        height_below_nonC,
                        height_above_nonC,
                        neighbor_H_nonC,
                        neighbor_S_nonC,
                        neighbor_V_nonC,
                        H_values_nonC,
                        S_values_nonC,
                        V_values_nonC)).transpose()
    
    #split the data into training and testing sets (20% test size)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    #machine learning models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gbt_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.2,max_depth=3, random_state=0)
    #Train the models
    rf_model.fit(X_train, y_train)
    gbt_model.fit(X_train, y_train)
    #Evaluate model
    y_pred_rf = rf_model.predict(X_test)
    y_pred_gbt = gbt_model.predict(X_test)
    #Predict the non-classified area
    predictions_RF = rf_model.predict(nonC_features)
    predictions_GBT = gbt_model.predict(nonC_features)
    
    output_csv = np.vstack((nonC_features_s1.get('X'),nonC_features_s1.get('Y'),nonC_features_s1.get('Z'),predictions_RF,predictions_GBT)).T
    np.savetxt('../working/classification/multiscale/classified_points.csv',output_csv,delimiter=',',header='X,Y,Z,RF,GBT',comments='')

    done_time = time.time()
    # #add mailme to CLI and get an email notification sent when scipt is done
    try:
        if len(sys.argv) >1:
            if sys.argv[1]=='mailme':
                send_email.sendNotification(f"""Process finished. Classification of data is done.
                                            \nThe whole process elapsed {(done_time - start_read)/3600} hours""")
    except:
        print("mail was not send, due to API key error")

