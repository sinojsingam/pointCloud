import time
import laspy
import numpy as np
from scipy.spatial import cKDTree

offset=(667000, 650000, 0)

def main():
    # Load the training data LAS file
    training_las = laspy.read("merge/buildings.las")
    # Load the full point cloud LAS file
    full_las = laspy.read("merge/lln.las")

    # Extract coordinates from the training data and full dataset
    training_coords = np.vstack((training_las.x - offset[0], training_las.y- offset[1], training_las.z)).transpose()
    full_coords = np.vstack((full_las.x - offset[0], full_las.y - offset[1], full_las.z)).transpose()

    # Create a cKDTree for the training data
    tree = cKDTree(training_coords)

    # Define a tolerance for considering two points as the same location
    tolerance = 0.01  # Adjust this value based on the precision of your data

    # Query the full dataset points in the training data tree to find exact matches within the tolerance
    distances, indices = tree.query(full_coords, distance_upper_bound=tolerance)

    # Find indices of the full dataset points that have an exact match in the training data
    matched_indices = indices < len(training_las.points)  # Filter out indices beyond training data length, indicating no match

    # Update classifications only for matched points
    full_las.classification[matched_indices] = training_las.classification[indices[matched_indices]]

    full_coords[:, 0] += offset[0]  # Retranslate
    full_coords[:, 1] += offset[1]
    # Save the modified full LAS file with updated classifications for matched points
    full_las.write("merge/lln_merged2.las")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f'{(end-start)/60} mins')

