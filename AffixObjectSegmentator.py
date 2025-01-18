import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.transform import resize

# Paths
input_folder = "AffixData/ply-inputs"
output_folder = "AffixData/npy-pointCloud"
mask_folder = "AffixData/npy-pointCloud"
os.makedirs(output_folder, exist_ok=True)
IM_HEIGHT = 480
IM_WIDTH = 640

def point_cloud_to_depth_image(points, height=IM_HEIGHT, width=IM_WIDTH):
    """
    Convert point cloud to depth image with proper depth preservation
    
    Args:
        points: Nx3 array of points
        height: desired height of depth image
        width: desired width of depth image
    
    Returns:
        depth_image: height x width depth image
    """
    if len(points) == 0:
        return np.zeros((height, width))

    # Create empty depth image
    depth_image = np.zeros((height, width))

    # Get depth values (Z coordinate)
    depths = points[:, 2]

    # Normalize X and Y to image coordinates while preserving aspect ratio
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Calculate aspect ratio of point cloud
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    point_cloud_aspect = x_range / y_range if y_range != 0 else 1.0

    # Calculate image aspect ratio
    image_aspect = width / height

    # Adjust scaling to preserve aspect ratio
    if point_cloud_aspect > image_aspect:
        # Point cloud is wider than image
        scale = width / x_range
        x_offset = 0
        y_offset = (height - y_range * scale) / 2
    else:
        # Point cloud is taller than image
        scale = height / y_range
        x_offset = (width - x_range * scale) / 2
        y_offset = 0

    # Scale coordinates to image space
    x_pixels = ((x_coords - np.min(x_coords)) * scale + x_offset).astype(int)
    y_pixels = ((y_coords - np.min(y_coords)) * scale + y_offset).astype(int)

    # Ensure coordinates are within image bounds
    valid_points = (x_pixels >= 0) & (x_pixels < width) & (y_pixels >= 0) & (y_pixels < height)
    x_pixels = x_pixels[valid_points]
    y_pixels = y_pixels[valid_points]
    depths = depths[valid_points]

   # Normalize depths to [0, 1] range for better visualization
    if len(depths) > 0:
        depth_min = np.min(depths)
        depth_max = np.max(depths)
        if depth_max > depth_min:
            depths = (depths - depth_min) / (depth_max - depth_min)

        # Fill in depth values
        for x, y, d in zip(x_pixels, y_pixels, depths):
            if depth_image[y, x] == 0 or d < depth_image[y, x]:
                depth_image[y, x] = d

    # Fill small holes
    from scipy.ndimage import median_filter
    depth_image = median_filter(depth_image, size=3)

    return depth_image

def visualize_depth_and_mask(depth_image, mask, points=None):
    """
    Visualize depth image, mask, and optionally point cloud
    """
    n_plots = 3 if points is not None else 2
    plt.figure(figsize=(5*n_plots, 5))

    if points is not None:
        plt.subplot(1, n_plots, 1)
        plt.title("Point Cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

    plt.subplot(1, n_plots, n_plots-1)
    plt.title("Depth Image")
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Depth')

    plt.subplot(1, n_plots, n_plots)
    plt.title("Segmentation Mask")
    plt.imshow(mask, cmap='gray')

    plt.tight_layout()
    plt.show()


def create_2d_mask(points, height=IM_HEIGHT, width=IM_WIDTH):
    """
    Create a 2D binary mask from 3D points
    
    Args:
        points: Nx3 array of points
        height: desired height of mask
        width: desired width of mask
    
    Returns:
        2D binary mask
    """
    # Project points to 2D
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Normalize to image coordinates
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    x_pixels = ((x_coords - x_min) / x_range * (width - 1)).astype(int)
    y_pixels = ((y_coords - y_min) / y_range * (height - 1)).astype(int)

    # Fill mask
    for x_pix, y_pix in zip(x_pixels, y_pixels):
        if 0 <= x_pix < width and 0 <= y_pix < height:
            mask[y_pix, x_pix] = 255

    # Dilate mask to fill small gaps
    mask = binary_dilation(mask, iterations=2).astype(np.uint8) * 255

    return mask
def get_latest_file(folder):
    """Get the latest file by DateOfCreation from a folder."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".ply")]
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def visualize_segment(points, mask=None):
    """Visualize a point cloud segment and its 2D mask if provided."""
    if mask is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    # if mask is not None:
    #     plt.subplot(122)
    #     plt.imshow(mask, cmap='gray')
    #     plt.title('2D Segmentation Mask')
    #     plt.axis('off')
    #     plt.show()



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



def temp(pcd):
    
    base_filename = "fullCloud"

    points = np.asarray(pcd.points)

    # Clean the point cloud
    points = clean_point_cloud(points)
    depth_image = point_cloud_to_depth_image(points)
    print(f"Depth image stats - Min: {np.min(depth_image):.3f}, Max: {np.max(depth_image):.3f}, Mean: {np.mean(depth_image):.3f}")

    # Save depth image as NPY
    npy_path = os.path.join(output_folder, f"{base_filename}_object_0.npy")
    np.save(npy_path, depth_image)

    # Create and save mask
    mask = create_2d_mask(points)
    mask_path = os.path.join(mask_folder, f"{base_filename}_object_0_mask.png")
    plt.imsave(mask_path, mask, cmap='gray')

def main(eps=0.094, min_samples=150, calibrate=False, visualizeSegmentedObjs=False, min_certainty=0.48):
    # Get the latest file
    latest_file = get_latest_file(input_folder)
    print(f"Processing file: {latest_file}")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(latest_file)
    
    temp(pcd)

    # Segment the point cloud
    if calibrate:
        segmented_objects, certainties = find_best_paramethers_for_segment_point_cloud(pcd)
    else:
        segmented_objects, certainties = segment_point_cloud(pcd, eps, min_samples)

    # Filter objects based on certainty
    filtered_objects = [(obj, cert) for obj, cert in zip(segmented_objects, certainties)
                        if cert >= min_certainty]

    if not filtered_objects:
        print("No objects found meeting the certainty threshold")
        return

    segmented_objects, certainties = zip(*filtered_objects)

    # Save each segment and its mask
    for i, (obj_points, certainty) in enumerate(zip(segmented_objects, certainties)):
        # Generate base filename
        base_filename = os.path.splitext(os.path.basename(latest_file))[0]

        # Convert point cloud to depth image
        depth_image = point_cloud_to_depth_image(obj_points)
        print(f"Depth image stats - Min: {np.min(depth_image):.3f}, Max: {np.max(depth_image):.3f}, Mean: {np.mean(depth_image):.3f}")

        # Save depth image as NPY
        npy_path = os.path.join(output_folder, f"{base_filename}_object_{i}.npy")
        np.save(npy_path, depth_image)

        # Create and save mask
        mask = create_2d_mask(obj_points)
        mask_path = os.path.join(mask_folder, f"{base_filename}_object_{i}_mask.png")
        plt.imsave(mask_path, mask, cmap='gray')

        print(f"Processed object {i}:")
        print(f"  Depth image saved to: {npy_path}")
        print(f"  Depth image shape: {depth_image.shape}")
        print(f"  Mask saved to: {mask_path}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Certainty: {certainty:.2f}")

        # if visualizeSegmentedObjs:
        #     visualize_depth_and_mask(depth_image, mask, obj_points)

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
    parser.add_argument("--visualizeSegmentedObjs", default=True, help="visualizeSegmentedObjs flag for DEBUG when Calibrating (default: True)")
    parser.add_argument("--min_certainty", type=int, default=0.48, help="Minimum certainty (default: 0.48)")

    # Parse arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(eps=args.eps, min_samples=args.min_samples, calibrate=args.calibrate,
         visualizeSegmentedObjs=args.visualizeSegmentedObjs, min_certainty=args.min_certainty)
