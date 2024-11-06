import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8x.pt')  # Replace with the path to your model

def detect_ambulance(image):
    results = model(image)
    ambulance_detected = False

    # Iterate over detected boxes
    for result in results:  # Iterate through results if there are multiple
        for box in result.boxes:
            if box is not None:
                # Check if the expected attributes are present
                if hasattr(box, 'xyxy') and hasattr(box, 'confidence') and hasattr(box, 'cls'):
                    x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates
                    conf = box.confidence.item()  # Get confidence score
                    cls = box.cls.item()  # Get class index
                    label = results.names[int(cls)]  # Get class label

                    if conf > 0.5:  # Confidence threshold
                        if label == 'ambulance':
                            ambulance_detected = True
                            print(f"Ambulance detected at: {x1}, {y1}, {x2}, {y2}")

    return ambulance_detected

# Capture video or read an image
cap = cv2.VideoCapture('ambu.jpg')  # Replace with your video file or camera source

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detect_ambulance(frame):
        # Trigger traffic signal change or alert system
        print("Priority action: Change traffic signal for ambulance.")

    # Optionally display the frame
    cv2.imshow('Traffic Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
