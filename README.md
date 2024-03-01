# Classifying pointcloud data

These set of functions use pointcloud libraries such as [laspy](https://laspy.readthedocs.io/en/latest/) and [CSF](https://github.com/jianboqi/CSF) for point cloud manipulation.

The workflow is as follows:
- First, I used Cloth Simulation Filter (CSF) to properly identify ground points.
- Using CloudCompare, using only non-ground points, I isolated and segmented features for training data
- Features such as buildings, cars, and vegetation at varying heights
- Using the training data, I calculated the geometric features
- With the geometric feature, color and location information for each point in the point cloud, I trained the classifier
    
K-means clustering and Random Forest algorithms are used.
