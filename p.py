import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time
from ultralytics import YOLO  # Import the YOLO class from the ultralytics library

# Load YOLOv8 model (ensure you have the model file yolov8x.pt)
model = YOLO("yolov8x.pt")  # Make sure this file is in your working directory or provide the correct path

# Vehicle time allocation (seconds)
vehicle_time = {'small': 1.5, 'heavy': 2.5, 'ambulance': 2.0}

# Confidence threshold for detections
CONF_THRESHOLD = 0.5

# Initialize GUI window
root = tk.Tk()
root.title("Smart Traffic Management System")

# Frame for displaying the images
image_frames = [tk.Label(root) for _ in range(4)]
for i, frame in enumerate(image_frames):
    frame.grid(row=0, column=i, padx=10, pady=10)

# Placeholder for lane images (as loaded from files)
lane_images = [None, None, None, None]
lane_file_paths = [
    'tr3.jpeg',  # Replace with actual image paths
    'tr4.JPG', 
    'tr3.jpeg', 
    'tr4.JPG'
]

# Image preprocessing: Enhance contrast or other improvements as needed
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_enhanced = cv2.equalizeHist(img_gray)  # Enhance contrast
    img_color = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for YOLO
    return img_color

# Load images for each lane and display in the GUI
def load_images():
    for i in range(4):
        img = cv2.imread(lane_file_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((200, 200))  # Resize for GUI display
        lane_images[i] = ImageTk.PhotoImage(img_pil)
        image_frames[i].config(image=lane_images[i])

# Process images using YOLOv8 to detect vehicles and ambulances
def process_images():
    vehicle_counts = []
    ambulance_detected = False

    for i, img_path in enumerate(lane_file_paths):
        img = preprocess_image(img_path)  # Preprocess image
        results = model(img)[0]  # Use the model to predict on the image
        
        # Filter out low-confidence detections (based on CONF_THRESHOLD)
        detections = results.pandas().xyxy[0]
        high_confidence_detections = detections[detections['confidence'] > CONF_THRESHOLD]

        small_count = 0
        heavy_count = 0

        for idx, row in high_confidence_detections.iterrows():
            label = row['name']
            if label in ['car', 'motorbike']:  # Considering small vehicles
                small_count += 1
            elif label in ['truck', 'bus']:  # Considering heavy vehicles
                heavy_count += 1
            elif label == 'ambulance':  # Ambulance detection
                ambulance_detected = True
        
        # Append vehicle counts as a tuple (small_count, heavy_count)
        vehicle_counts.append((small_count, heavy_count))

        # Output vehicle count for the current lane in the terminal
        print(f"Lane {i+1} - Small vehicles: {small_count}, Heavy vehicles: {heavy_count}")

    return vehicle_counts, ambulance_detected

# Calculate time for each lane based on vehicle count
def calculate_time(vehicle_counts):
    lane_times = []

    for count in vehicle_counts:
        small_count, heavy_count = count
        time = small_count * vehicle_time['small'] + heavy_count * vehicle_time['heavy']
        lane_times.append(time)
    
    return lane_times

# Display green signal for each lane and stop after one cycle
def give_green_signal(lane_times, ambulance_detected):
    if ambulance_detected:
        # Priority to the lane with an ambulance
        print("Ambulance detected, prioritizing lane 1.")
        display_signals(0, lane_times)  # For example, give green to lane 1
        return  # Stop after one cycle as requested

    # Sort lanes by calculated time in ascending order
    sorted_lanes = sorted(range(len(lane_times)), key=lambda k: lane_times[k])

    # Give green signal to each lane in sorted order
    for i in sorted_lanes:
        display_signals(i, lane_times)
        root.update()  # Update the GUI
        root.after(int(lane_times[i] * 1000))  # Delay for allocated time

    # No looping, end after one cycle

# Display the signals and print the current lane status in the terminal
def display_signals(green_index, lane_times):
    signal_colors = ['🔴', '🔴', '🔴', '🔴']  # Default all lanes to red
    signal_colors[green_index] = '🟢'  # Set the current green lane

    # Print the lane signals and time for each lane
    print(f"\nLane Signals: {signal_colors}")
    print(f"Lane Times: {['%.1f' % t for t in lane_times]} seconds")

    # Update the GUI for the signals
    for i, frame in enumerate(image_frames):
        frame.config(bg='green' if i == green_index else 'red')
        root.update()

    time.sleep(1)  # Small delay to simulate signal transition

# Main function to load images, process them, and give green signal
def start_process():
    load_images()
    vehicle_counts, ambulance_detected = process_images()
    lane_times = calculate_time(vehicle_counts)
    give_green_signal(lane_times, ambulance_detected)
    
    # Destroy the GUI window after processing all lanes
    root.after(1000, root.destroy)

# Start the process on window load
start_process()

# Run the GUI main loop
root.mainloop()
