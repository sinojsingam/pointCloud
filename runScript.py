
if __name__ == '__main__':
    import calculateFeatures
    import numpy as np
    import laspy as lp

    # Load the point cloud data

    input_path = '../working/classification/multiscale/'
    dataname = 'classified_sample.las'
    point_cloud = lp.read(input_path+dataname)
    points = np.vstack((point_cloud.x, 
                        point_cloud.y,
                        point_cloud.z,
                        point_cloud['normal z'],
                        point_cloud.classification,
                        point_cloud.red, 
                        point_cloud.green,
                        point_cloud.blue)).transpose()
    # colors = np.vstack((point_cloud.red, point_cloud.green,point_cloud.blue)).transpose()
    calculateFeatures.calculateGeometricFeatures(points,0.5,save=True,output_file='TEST1_geomFeat.las')