import os
import open3d as o3d
import laspy
import numpy as np

if __name__ == "__main__":
    # Read the bunny statue point cloud using numpy's loadtxt
    point_cloud_path = os.path.join('..','working','isolated_features','car_training.las')
    point_cloud = laspy.read(point_cloud_path)

    # Separate the into points, colors and normals array
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
    normals = np.vstack((point_cloud['normal x'], point_cloud['normal y'], point_cloud['normal z'])).transpose()

    # Initialize a point cloud object
    pcd = o3d.geometry.PointCloud()
    # Add the points, colors and normals as Vectors
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Create a voxel grid from the point cloud with a voxel_size of 0.01
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.01)

    # Initialize a visualizer object
    o3d.visualization.draw_geometries([pcd])
