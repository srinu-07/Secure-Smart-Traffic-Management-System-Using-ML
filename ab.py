import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('yolov8x.pt')

# Print class names to verify 'ambulance' is included
print("Class names:", model.names)

# Constants
CONF_THRESHOLD = 0.5  # Confidence threshold for detections

# Sample list of lane image paths
lane_file_paths = [
    'tr3.jpeg',
    'tr4.JPG',
    'ambu.jpg',  # Add your ambulance image here
]

def preprocess_image(img_path):
    """Load and preprocess the image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}. Check the path or file format.")
        return None  # Return None if image reading fails
    return img

def draw_detections(img, detections):
    """Draw bounding boxes and labels on the image."""
    results = []
    for box in detections:
        box_data = box.xyxy[0]
        if len(box_data) == 6:
            x1, y1, x2, y2, confidence, class_id = box_data
            label = results[0].names[int(class_id)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def process_images():
    """Process each image and count vehicles."""
    vehicle_counts = []
    ambulance_detected = False

    for i, img_path in enumerate(lane_file_paths):
        img = preprocess_image(img_path)
        if img is None:  # Skip processing if image loading failed
            continue

        results = model(img)
        detections = results[0].boxes

        small_count = 0
        heavy_count = 0

        for box in detections:
            box_data = box.xyxy[0]  # Get the bounding box data
            if len(box_data) == 6:  # Ensure we have enough values
                x1, y1, x2, y2, confidence, class_id = box_data
                
                if confidence > CONF_THRESHOLD:
                    label = results[0].names[int(class_id)]
                    
                    # Count vehicles based on their type
                    if label in ['car', 'motorcycle']:  # Small vehicles
                        small_count += 1
                    elif label in ['bus', 'truck']:  # Heavy vehicles
                        heavy_count += 1
                    elif label == 'ambulance':  # Check for ambulance
                        ambulance_detected = True
        
        vehicle_counts.append((small_count, heavy_count))
        print(f"Lane {i + 1} - Small vehicles: {small_count}, Heavy vehicles: {heavy_count}")

        # Optionally draw detections
        img_with_detections = draw_detections(img, detections)
        cv2.imshow(f"Detections Lane {i + 1}", img_with_detections)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return vehicle_counts, ambulance_detected

def start_process():
    """Start processing images and check for ambulance detection."""
    vehicle_counts, ambulance_detected = process_images()
    if ambulance_detected:
        print("Ambulance detected! Prioritizing green signal.")
    else:
        print("No ambulance detected.")

if __name__ == "__main__":
    start_process()
