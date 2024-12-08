import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Paths
input_folder = "AffixData/ply-inputs"
output_folder = "AffixData/npy-pointCloud"
os.makedirs(output_folder, exist_ok=True)


def get_latest_file(folder):
    """Get the latest file by DateOfCreation from a folder."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".ply")]
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def visualize_segment(points):
    """Visualize a point cloud segment."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def clean_point_cloud(points):
    """Remove NaN and infinite values from a point cloud.   THE mechmind software generated .ply come with a lot of garbage points"""
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def segment_point_cloud(pcd, eps=0.1, min_samples=3):
    points = np.asarray(pcd.points)

    # Clean the point cloud
    points = clean_point_cloud(points)

    if points.size == 0:
        print("No valid points after cleaning. (NaN and infinite values)")
        return [], []

    # Print diagnostic information
    print(f"Point cloud stats:")
    print(f"Number of points: {len(points)}")
    # print(f"Min values: {np.min(points, axis=0)}")
    # print(f"Max values: {np.max(points, axis=0)}")
    # print(f"Mean values: {np.mean(points, axis=0)}")

    # Normalize the points
    scaler = StandardScaler()
    points_normalized = scaler.fit_transform(points)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1  # Use all available cores
    ).fit(points_normalized)

    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"eps={eps}, min_samples={min_samples}: "
          f"found {n_clusters} clusters, "
          f"{n_noise} noise points "
          f"({n_noise / len(points) * 100:.1f}% noise)")

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    segmented_objects = []
    certainties = []

    for label in unique_labels:
        mask = labels == label
        segment_points = points[mask]  # Use original points, not normalized

        # Calculate certainty based on ratio of core points
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        certainty = np.mean(core_samples_mask[mask])

        segmented_objects.append(segment_points)
        certainties.append(certainty)

        print(f"Cluster {label}: "
              f"{len(segment_points)} points, "
              f"Certainty: {certainty:.2f}")

        # Optional: Calculate and print cluster properties
        centroid = np.mean(segment_points, axis=0)
        print(f"    Centroid: {centroid}")
        print(f"    Bounding box: "
              f"min={np.min(segment_points, axis=0)}, "
              f"max={np.max(segment_points, axis=0)}")

    return segmented_objects, certainties


def clean_point_cloud(points):
    """
    Clean the point cloud by removing invalid points.
    Modify this function based on your specific cleaning needs.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(points).any(axis=1)

    # Remove infinite values
    valid_mask &= ~np.isinf(points).any(axis=1)

    # Optional: Remove outliers based on statistical properties
    # You might want to adjust or remove these depending on your data
    for dim in range(points.shape[1]):
        dim_data = points[:, dim]
        dim_mean = np.mean(dim_data[valid_mask])
        dim_std = np.std(dim_data[valid_mask])
        valid_mask &= np.abs(dim_data - dim_mean) <= 3 * dim_std  # 3 sigma rule

    cleaned_points = points[valid_mask]

    if len(cleaned_points) < len(points):
        print(f"Removed {len(points) - len(cleaned_points)} invalid points")

    return cleaned_points


def main(eps, min_samples, calibrate, visualizeSegmentedObjs, min_certainty):
    # Get the latest file
    latest_file = get_latest_file(input_folder)
    print(f"Processing file: {latest_file}")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(latest_file)

    # Segment the point cloud
    if calibrate:
        segmented_objects, certainties = find_best_paramethers_for_segment_point_cloud(pcd)
    else:
        segmented_objects, certainties = segment_point_cloud(pcd, eps, min_samples)

    # remove objects with certainty below min_certainty
    segmented_objects = [obj_points for obj_points, certainty in zip(segmented_objects, certainties) if
                         certainty >= min_certainty]

    # Save each segment as a .npy file and output certainty
    for i, (obj_points, certainty) in enumerate(zip(segmented_objects, certainties)):
        # Visualize the segment

        if visualizeSegmentedObjs:
            print(f"Visualizing Object {i} with Certainty: {certainty:.2f}")
            visualize_segment(obj_points)

        # Save the segment
        output_path = os.path.join(output_folder,
                                   f"{os.path.splitext(os.path.basename(latest_file))[0]}_object_{i}.npy")
        np.save(output_path, obj_points)
        print(f"File: {latest_file}, Object {i}, Certainty: {certainty:.2f}")


"""
TRAINING & RESEARCH
"""


def find_best_paramethers_for_segment_point_cloud(pcd):
    points = np.asarray(pcd.points)

    # Clean the point cloud
    points = clean_point_cloud(points)

    if points.size == 0:
        print("No valid points after cleaning. (NaN and infinite values)")
        return [], []

    print("----------------------------")
    print(
        "you are shown a visualization of the point cloud count the number of objects (object have separation from one another or gap in the poit cloud , this gap can be in any 3d direction)")
    print(
        "\n KEEP IN MIND THAT THE THE RESULTING CLUSTERS(FOUND OBJECTS) are not you number of objects in your bin , so chose paramethers that match closely the number of objects in the visualization  ")
    print(
        "\n when you close the visualization the system will give you results with variety of paramethers , the script will exit after that")
    print(
        "set min_samples_values based in your depth camera`s resolution  generally 50 to 100 is a good start for a mechmind PRO S-GL, if you want to capture smaller objects you need to lower the  min_samples_values but generally not less then 20")

    # visualize cleaned points
    visualize_segment(points)

    # Print diagnostic information
    print(f"Point cloud stats:")
    print(f"Number of points: {len(points)}")
    print(f"Min values: {np.min(points, axis=0)}")
    print(f"Max values: {np.max(points, axis=0)}")
    print(f"Mean values: {np.mean(points, axis=0)}")

    # Normalize the points
    scaler = StandardScaler()
    points_normalized = scaler.fit_transform(points)

    # Try different parameter combinations
    eps_values = [0.02, 0.05, 0.1, 0.2, 0.5]
    min_samples_values = [25, 30, 50, 150, 250]

    best_n_clusters = 0
    best_params = None
    best_labels = None
    best_clustering = None

    print("\nTesting DBSCAN parameters...")

    for eps in eps_values:
        for min_samples in min_samples_values:
            clustering = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                n_jobs=-1  # Use all available cores
            ).fit(points_normalized)

            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            print(f"eps={eps}, min_samples={min_samples}: "
                  f"found {n_clusters} clusters, "
                  f"{n_noise} noise points "
                  f"({n_noise / len(points) * 100:.1f}% noise)")

            if n_clusters > best_n_clusters:
                best_n_clusters = n_clusters
                best_params = (eps, min_samples)
                best_labels = labels
                best_clustering = clustering

    if best_params is None:
        print("\nNo clusters found with any parameters.")
        return [], []

    print(f"\nBest parameters: eps={best_params[0]}, min_samples={best_params[1]}")
    print(f"Number of clusters: {best_n_clusters}")

    # Process clusters using best parameters
    unique_labels = set(best_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    segmented_objects = []
    certainties = []

    print("\nCluster statistics:")
    for label in unique_labels:
        mask = best_labels == label
        segment_points = points[mask]  # Use original points, not normalized

        # Calculate certainty based on ratio of core points
        core_samples_mask = np.zeros_like(best_labels, dtype=bool)
        core_samples_mask[best_clustering.core_sample_indices_] = True
        certainty = np.mean(core_samples_mask[mask])

        segmented_objects.append(segment_points)
        certainties.append(certainty)

        print(f"Cluster {label}: "
              f"{len(segment_points)} points, "
              f"Certainty: {certainty:.2f}")

        # Optional: Calculate and print cluster properties
        centroid = np.mean(segment_points, axis=0)
        print(f"    Centroid: {centroid}")
        print(f"    Bounding box: "
              f"min={np.min(segment_points, axis=0)}, "
              f"max={np.max(segment_points, axis=0)}")

    return segmented_objects, certainties


if __name__ == "__main__":
    import argparse

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run AffixObjectSegmentation with optional parameters.")
    parser.add_argument("--eps", type=float, default=0.094, help="Epsilon value (default: 0.1)")
    parser.add_argument("--min_samples", type=int, default=150, help="Minimum samples (default: 3)")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate flag (default: False)")
    parser.add_argument("--visualizeSegmentedObjs", default=True, help="Calibrate flag (default: True)")
    parser.add_argument("--min_certainty", type=int, default=0.48, help="Minimum certainty (default: 0.48)")

    # Parse arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(eps=args.eps, min_samples=args.min_samples, calibrate=args.calibrate,
         visualizeSegmentedObjs=args.visualizeSegmentedObjs, min_certainty=args.min_certainty)
