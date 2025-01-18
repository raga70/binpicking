import glob
import os
import numpy as np
import cv2
from PIL import Image
import torch

# ---------------------------
# Import from SAM2
# ---------------------------
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ---------------------------
# 1. Device selection
# ---------------------------
device = "cpu"  # set to "cuda" or "cuda:0" if you want to use GPU
print(f"Using device: {device}")

# ---------------------------
# 2. Load SAM2 model
# ---------------------------
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(
    model_cfg,
    sam2_checkpoint,
    device=device,
    apply_postprocessing=True  # Turn on postprocessing for smoother mask edges
)

# ---------------------------
# 3. Automatic mask generator
# ---------------------------
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.90,
    min_mask_region_area=150
)

# ---------------------------
# 4. Paths
# ---------------------------

def get_latest_image(folder, prefix="rgb_", suffix=".png"):
    search_pattern = os.path.join(folder, f"{prefix}*{suffix}")
    files = glob.glob(search_pattern)
    if not files:
        return None  # No files found
    latest_file = max(files, key=os.path.getctime)  # Get the file with the latest creation time
    return latest_file

input_image_folder = "../AffixData/ply-inputs"

input_image_path = get_latest_image(input_image_folder) #latest file in the folder with name starting with "rgb_" and ending on .png
point_cloud_dir = "../AffixData/npy-pointCloud"
output_dir = "../AffixData/RGB_Segmented"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 5. Load input image
# ---------------------------
image_pil = Image.open(input_image_path).convert("RGB")
image = np.array(image_pil)

# ---------------------------
# 6. Generate masks
# ---------------------------
masks = mask_generator.generate(image)

# ---------------------------
# 7. Compute bounding boxes for masks
# ---------------------------
def get_mask_bbox(segmentation):
    """
    Returns the bounding box (x_min, y_min, x_max, y_max) for a given segmentation mask.
    If the mask is empty, returns None.
    """
    rows, cols = np.where(segmentation)
    if len(rows) == 0 or len(cols) == 0:
        # Empty mask
        return None
    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)
    return (x_min, y_min, x_max, y_max)

mask_bboxes = []
for m in masks:
    bbox = get_mask_bbox(m["segmentation"])
    mask_bboxes.append(bbox)

# ---------------------------
# 8. Load .npy files (point clouds) and match them
# ---------------------------
point_cloud_files = sorted(f for f in os.listdir(point_cloud_dir) if f.endswith(".npy"))

# If there are no .npy files, create dummy names according to the number of masks
if not point_cloud_files:
    point_cloud_files = [f"object_{i+1}.npy" for i in range(len(masks))]

def bbox_iou(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    Each box is given as (x_min, y_min, x_max, y_max).
    Returns 0.0 if one of them is None.
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_pointcloud_bbox_2d(points):
    """
    Returns (x_min, y_min, x_max, y_max) for the given point cloud.
    If the points are 3D, we take the x,y columns and ignore z.
    If the point cloud is empty, returns None.
    """
    if points.shape[1] == 3:
        # 3D -> use x, y only
        x_vals = points[:, 0]
        y_vals = points[:, 1]
    else:
        # 2D
        x_vals = points[:, 0]
        y_vals = points[:, 1]

    if len(x_vals) == 0 or len(y_vals) == 0:
        return None

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    return (int(x_min), int(y_min), int(x_max), int(y_max))

# ---------------------------
# 9. Find the best mask for each point cloud and save
# ---------------------------
def save_masks_for_pointcloud(
    pc_file, pc_data, masks, mask_bboxes, original_image, output_dir
):
    """
    1) Compute the bounding box of the loaded point cloud (pc_data).
    2) For each mask, compute IoU with the point cloud bbox.
    3) Choose the mask with the highest IoU.
    4) Save two outputs:
       - A 3-channel BW mask with the object in white (255,255,255) and a red (0,0,255) dot on its centroid.
       - An RGB mask that shows the original colors of the object.
    """
    pc_bbox = get_pointcloud_bbox_2d(pc_data)

    # If there's no bounding box, skip
    if pc_bbox is None:
        print(f"[WARNING] No valid bounding box found for {pc_file}.")
        return

    best_iou = 0.0
    best_mask_idx = None
    for i, bbox in enumerate(mask_bboxes):
        iou_val = bbox_iou(pc_bbox, bbox)
        if iou_val > best_iou:
            best_iou = iou_val
            best_mask_idx = i

    if best_mask_idx is None:
        print(f"[WARNING] No matching mask found for {pc_file} (IoU=0).")
        return

    # Select the best mask
    chosen_mask = masks[best_mask_idx]
    segmentation = chosen_mask["segmentation"]

    # --- 1) 3-channel BW mask with a red dot on the centroid ---
    bw_mask = np.zeros_like(original_image)  # (H, W, 3)
    bw_mask[segmentation] = (255, 255, 255)  # object in white

    # find contours, then draw a red dot at the centroid
    contours, _ = cv2.findContours(
        segmentation.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(bw_mask, (cx, cy), 5, (0, 0, 255), -1)  # red dot

    bw_output_path = os.path.join(
        output_dir,
        pc_file.replace(".npy", "_BW.png")
    )
    Image.fromarray(bw_mask).save(bw_output_path)

    # --- 2) RGB mask ---
    rgb_mask = np.zeros_like(original_image)
    rgb_mask[segmentation] = original_image[segmentation]

    rgb_output_path = os.path.join(
        output_dir,
        pc_file.replace(".npy", "_RGB.png")
    )
    Image.fromarray(rgb_mask).save(rgb_output_path)

    print(f"[OK] {pc_file} -> best_mask_idx={best_mask_idx}, IoU={best_iou:.3f}")
    print(f"     Saved: {bw_output_path} and {rgb_output_path}")

# ---------------------------
# 10. Loop over .npy files, load and save outputs
# ---------------------------
for pc_file in point_cloud_files:
    pc_path = os.path.join(point_cloud_dir, pc_file)
    if os.path.isfile(pc_path):
        try:
            pc_data = np.load(pc_path)  # shape (N, 2) or (N, 3)
            save_masks_for_pointcloud(pc_file, pc_data, masks, mask_bboxes, image, output_dir)
        except Exception as e:
            print(f"[ERROR] Could not read {pc_file}. {e}")
    else:
        print(f"[WARNING] {pc_path} not found, skipping.")

print("All done. Check the 'output_objects' folder for results.")

