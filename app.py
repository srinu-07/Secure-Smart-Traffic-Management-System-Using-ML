import cv2
import torch
import numpy as np
import time
from emoji import emojize

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Ensure the model is pretrained

def preprocess_image(image):
    def resize_image(image, target_size=(640, 640)):
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized

    def sharpen_image(image):
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def enhance_image(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def reduce_noise(image):
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised

    # Apply preprocessing steps
    image_resized = resize_image(image)
    image_sharpened = sharpen_image(image_resized)
    image_enhanced = enhance_image(image_sharpened)
    image_preprocessed = reduce_noise(image_enhanced)
    return image_preprocessed

def detect_vehicles(image, confidence_threshold=0.3):
    image = preprocess_image(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for YOLO
    results = model(image_rgb)
    df = results.pandas().xyxy[0]
    df = df.rename(columns={'class': 'class_id'})
    
    # Filter detections based on confidence
    df = df[df['confidence'] >= confidence_threshold]
    
    # Define vehicle classes
    vehicle_classes = {
        'small': [2, 3],  # Car, motorcycle
        'heavy': [5, 7],  # Bus, truck
        'ambulance': [1]  # Adjust if necessary
    }

    small_vehicles = df[df['class_id'].isin(vehicle_classes['small'])]
    remaining_vehicles = df[~df.index.isin(small_vehicles.index)]
    heavy_vehicles = remaining_vehicles[remaining_vehicles['class_id'].isin(vehicle_classes['heavy'])]
    ambulances = df[df['class_id'].isin(vehicle_classes['ambulance'])]

    return small_vehicles, heavy_vehicles, ambulances

# Function to get vehicle counts and priority
def get_lane_priority(lane_images):
    vehicle_counts = {'small': [], 'heavy': [], 'ambulance': []}
    for image in lane_images:
        if image is not None:
            small_vehicles, heavy_vehicles, ambulances = detect_vehicles(image)
            vehicle_counts['small'].append(len(small_vehicles))
            vehicle_counts['heavy'].append(len(heavy_vehicles))
            vehicle_counts['ambulance'].append(len(ambulances))
        else:
            vehicle_counts['small'].append(0)
            vehicle_counts['heavy'].append(0)
            vehicle_counts['ambulance'].append(0)
    
    # Determine time allocation for each lane
    time_allocation = [1.5 * small + 2.5 * heavy + 2 * ambulance for small, heavy, ambulance in zip(vehicle_counts['small'], vehicle_counts['heavy'], vehicle_counts['ambulance'])]
    
    # Prioritize lanes with ambulances
    if any(vehicle_counts['ambulance']):
        priority_lane = vehicle_counts['ambulance'].index(max(vehicle_counts['ambulance']))
    else:
        priority_lane = np.argmin(time_allocation)
    
    return priority_lane, vehicle_counts, time_allocation

# Function to display traffic signals
def display_traffic_signals(lane_images, priority_lane, vehicle_counts, time_allocation):
    signal_radius = 20  # Signal size
    gap = 20  # Reduced gap between signals

    for i, image in enumerate(lane_images):
        if image is not None:
            # Define fixed signal positions with adjusted spacing
            red_pos = (50, 50)
            yellow_pos = (50, 100)
            green_pos = (50, 150)
            
            # Ensure positions are tuples of integers
            red_pos = (int(red_pos[0]), int(red_pos[1]))
            yellow_pos = (int(yellow_pos[0]), int(yellow_pos[1]))
            green_pos = (int(green_pos[0]), int(green_pos[1]))
            
            # Draw Red, Yellow, Green lights
            cv2.circle(image, red_pos, signal_radius, (0, 0, 255), -1)  # Red light
            cv2.circle(image, yellow_pos, signal_radius, (0, 255, 255), -1)  # Yellow light
            cv2.circle(image, green_pos, signal_radius, (0, 255, 0), -1)  # Green light
            
            # Set the green light for the priority lane
            if i == priority_lane:
                cv2.circle(image, green_pos, signal_radius, (0, 255, 0), -1)  # Green light for priority lane
                # Red and Yellow lights should be off
                cv2.circle(image, red_pos, signal_radius, (0, 0, 0), -1)  # Off red light
                cv2.circle(image, yellow_pos, signal_radius, (0, 0, 0), -1)  # Off yellow light
            else:
                # Red and Yellow lights should be on
                cv2.circle(image, red_pos, signal_radius, (0, 0, 255), -1)  # Red light
                cv2.circle(image, yellow_pos, signal_radius, (0, 255, 255), -1)  # Yellow light
                cv2.circle(image, green_pos, signal_radius, (0, 0, 0), -1)  # Off green light
            
            # Display vehicle counts and time allocation on the image
            text = f'Small: {vehicle_counts["small"][i]}, Heavy: {vehicle_counts["heavy"][i]}, Time: {time_allocation[i]}s'
            text_pos = (20, 200)
            text_pos = (int(text_pos[0]), int(text_pos[1]))  # Ensure coordinates are integers
            cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            # Create a placeholder image if the original image is None
            placeholder = np.zeros_like(lane_images[0])
            cv2.putText(placeholder, f'Lane {i+1}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(placeholder, 'No Image', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            lane_images[i] = placeholder
    
    # Resize images to ensure all four fit within the display window
    resized_images = [cv2.resize(img, (300, 300)) for img in lane_images]
    
    # Concatenate images horizontally and ensure they fit within the display window
    combined_image = cv2.hconcat(resized_images)

    # Display the combined image
    cv2.imshow('Traffic Signals', combined_image)
    cv2.waitKey(1)  # Use a short wait time to update the display

    # Print updated vehicle counts and time allocations to the terminal
    for i in range(len(lane_images)):
        signal = emojize(":green_circle:") if i == priority_lane else emojize(":red_circle:")
        print(f'Lane {i+1} {signal}: Small Vehicles: {vehicle_counts["small"][i]}, Heavy Vehicles: {vehicle_counts["heavy"][i]}, Time Allocated: {time_allocation[i]}s')

# Main function to process images and manage traffic signals
def main():
    # Load or capture images from four lanes
    lane1 = cv2.imread('tr3.jpeg')
    lane2 = cv2.imread('tr3.jpeg')
    lane3 = cv2.imread('tr3.jpeg')
    lane4 = cv2.imread('tr3.jpeg')
    lane_images = [lane1, lane2, lane3, lane4]

    # Ensure images are loaded correctly
    for i, img in enumerate(lane_images):
        if img is None:
            print(f"Warning: Lane {i+1} image could not be loaded.")
    
    # Get lane priority, vehicle counts, and time allocation
    priority_lane, vehicle_counts, time_allocation = get_lane_priority(lane_images)

    # Manage traffic signal change
    previous_lanes = []
    while True:
        # Display traffic signals based on priority
        display_traffic_signals(lane_images, priority_lane, vehicle_counts, time_allocation)
        
        if time_allocation[priority_lane] > 0:
            print(f'Lane {priority_lane + 1} will be green for {time_allocation[priority_lane]} seconds.')
            time.sleep(time_allocation[priority_lane])
            print('Switching to the next lane.')

        # Update previous lanes
        previous_lanes.append(priority_lane)
        if len(previous_lanes) > 3:
            previous_lanes.pop(0)
        
        # Determine the next lane to give the green signal
        remaining_time_allocation = [time if i not in previous_lanes else float('inf') for i, time in enumerate(time_allocation)]
        priority_candidates = np.where(np.array(remaining_time_allocation) == min(remaining_time_allocation))[0]
        if len(priority_candidates) > 0:
            priority_lane = priority_candidates[0]
        
        # Check for ambulance and give immediate priority
        if any(vehicle_counts['ambulance']):
            priority_lane = vehicle_counts['ambulance'].index(max(vehicle_counts['ambulance']))

if __name__ == '__main__':
    main()
