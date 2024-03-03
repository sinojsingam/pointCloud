# Classifying pointcloud data
## Still under construction :construction_worker:
These set of functions use pointcloud libraries such as [laspy](https://laspy.readthedocs.io/en/latest/) and [CSF](https://github.com/jianboqi/CSF) for point cloud manipulation.

The workflow is as follows:
- First, I used Cloth Simulation Filter (CSF) to properly identify ground points.
- Using CloudCompare, using only non-ground points, I isolated and segmented features for training data
- Features such as buildings, cars, and vegetation at varying heights
- Using the training data, I calculated the geometric features
- With the geometric feature, color and location information for each point in the point cloud, I trained the classifier
    
What are the geometric features:
- Omnivariance $\sqrt[3]{\lambda_1 \cdot \lambda_2 \cdot \lambda_3}$
- Eigenentropy $-\sum_{i=1}^{3} \lambda_i \cdot \ln(\lambda_i)$
- Anisotropy $(\lambda_{\text{min}} - \lambda_{\text{max}})/\lambda_{\text{min}}$
- Sphericity 
- Linearity
- Surface variation
- Verticality
- Neighborhood colors
K-means clustering and Random Forest algorithms are used.
