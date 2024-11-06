import cv2
import torch
import numpy as np
import time
from emoji import emojize

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to preprocess the image
def preprocess_image(image):
    # Same preprocessing logic
    def resize_image(image, target_size=(640, 640)):
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def sharpen_image(image):
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        return cv2.filter2D(image, -1, kernel)

    def enhance_image(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def reduce_noise(image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    image_resized = resize_image(image)
    image_sharpened = sharpen_image(image_resized)
    image_enhanced = enhance_image(image_sharpened)
    return reduce_noise(image_enhanced)

# Function to detect vehicles and classify them
def detect_vehicles(image, confidence_threshold=0.3):
    image = preprocess_image(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    df = results.pandas().xyxy[0]
    df = df.rename(columns={'class': 'class_id'})
    df = df[df['confidence'] >= confidence_threshold]

    vehicle_classes = {
        'small': [2, 3],  # Car, motorcycle
        'heavy': [5, 7],  # Bus, truck
        'ambulance': [1]  # Ambulance
    }

    small_vehicles = df[df['class_id'].isin(vehicle_classes['small'])]
    heavy_vehicles = df[df['class_id'].isin(vehicle_classes['heavy'])]
    ambulances = df[df['class_id'].isin(vehicle_classes['ambulance'])]

    return small_vehicles, heavy_vehicles, ambulances

# Function to calculate vehicle counts and time allocation
def get_lane_priority(lane_images):
    vehicle_counts = {'small': [], 'heavy': [], 'ambulance': []}
    for image in lane_images:
        small_vehicles, heavy_vehicles, ambulances = detect_vehicles(image)
        vehicle_counts['small'].append(len(small_vehicles))
        vehicle_counts['heavy'].append(len(heavy_vehicles))
        vehicle_counts['ambulance'].append(len(ambulances))

    time_allocation = [1.5 * small + 2.5 * heavy + 2 * ambulance
                       for small, heavy, ambulance in zip(vehicle_counts['small'], vehicle_counts['heavy'], vehicle_counts['ambulance'])]

    return vehicle_counts, time_allocation

# Function to display the traffic signals
def display_traffic_signals(lane_images, active_lane, vehicle_counts, time_allocation):
    signal_radius = 20
    for i, image in enumerate(lane_images):
        red_pos, yellow_pos, green_pos = (50, 50), (50, 100), (50, 150)
        cv2.circle(image, red_pos, signal_radius, (0, 0, 255), -1)
        cv2.circle(image, yellow_pos, signal_radius, (0, 255, 255), -1)
        cv2.circle(image, green_pos, signal_radius, (0, 0, 0), -1)

        if i == active_lane:
            cv2.circle(image, green_pos, signal_radius, (0, 255, 0), -1)
            cv2.circle(image, red_pos, signal_radius, (0, 0, 0), -1)
            cv2.circle(image, yellow_pos, signal_radius, (0, 0, 0), -1)

        text = f'Small: {vehicle_counts["small"][i]}, Heavy: {vehicle_counts["heavy"][i]}, Time: {time_allocation[i]}s'
        cv2.putText(image, text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    combined_image = cv2.hconcat([cv2.resize(img, (300, 300)) for img in lane_images])
    cv2.imshow('Traffic Signals', combined_image)
    cv2.waitKey(1)

    for i in range(len(lane_images)):
        signal = emojize(":green_circle:") if i == active_lane else emojize(":red_circle:")
        print(f'Lane {i+1} {signal}: Small Vehicles: {vehicle_counts["small"][i]}, Heavy Vehicles: {vehicle_counts["heavy"][i]}, Time Allocated: {time_allocation[i]}s')

# Main function to manage the traffic system
def main():
    lane1 = cv2.imread('tr3.jpeg')
    lane2 = cv2.imread('tr4.JPG')
    lane3 = cv2.imread('tr7.jpeg')
    lane4 = cv2.imread('tr1.jpeg')
    lane_images = [lane1, lane2, lane3, lane4]

    current_lane = 0  # Start with the first lane

    while True:
        vehicle_counts, time_allocation = get_lane_priority(lane_images)

        # Check if there's an ambulance; if so, prioritize that lane
        priority_lane = next((i for i, ambulances in enumerate(vehicle_counts['ambulance']) if ambulances > 0), current_lane)

        # Display signals for all lanes and apply the green signal to the priority lane
        display_traffic_signals(lane_images, priority_lane, vehicle_counts, time_allocation)

        # Wait for the green signal time for the active lane
        green_signal_time = time_allocation[priority_lane]
        time.sleep(green_signal_time)

        # Move to the next lane in round-robin fashion
        current_lane = (priority_lane + 1) % len(lane_images)

if __name__ == "__main__":
    main()
