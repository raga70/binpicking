import os
from ultralytics import YOLO

rgb_segmented_folder = "../AffixData/RGB_Segmented"
def classify_object(npy_file):
    """
    Classifies an object based on the corresponding image of a given .npy file.

    Args:
        npy_file (str): The file name of the .npy file (e.g., 'name.npy').

    Returns:
        str: The classification label, or an error message if the process fails.
    """
    # Derive the image file path
    npy_filename = os.path.basename(npy_file)
    image_filename = os.path.join(rgb_segmented_folder, f"{npy_filename[:-4]}.png")    # search for the exact fileName with .png extension in the AffixData/RGB_Segmented folder
    

    # Check if the image file exists
    if not os.path.exists(image_filename):
        return f"Error: Corresponding image file '{image_filename}' not found."

    # Load the YOLO model
    try:
        model = YOLO("models/InTheBinRGB1.pt", task="detect")
    except Exception as e:
        return f"Error loading YOLO model: {e}"

    # Perform inference on the image
    try:
        results = model(image_filename)
    except Exception as e:
        return f"Error during model inference: {e}"

    # Access detections
    detections = results[0]  # Access the first result
    if detections.boxes is None or len(detections.boxes) == 0:
        return "No object detected"

    # Extract boxes, confidence scores, and class IDs
    boxes = detections.boxes  # Bounding boxes with confidence and class IDs
    confidences = boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = boxes.cls.cpu().numpy()  # Class IDs
    names = detections.names  # Class names

    # Get the most confident detection
    best_index = confidences.argmax()
    best_label = names[int(class_ids[best_index])]

    return best_label

