# ------------------------- colorPick.py -------------------------
# Author: Lital H. Ben Baruch
# Date: 25th September 2023
# Contact: lital.h.ben.baruch@gmail.com
# Description: This code is designed for detecting and tracking specified colors in real-time using a webcam.
# It provides the functionality to adjust and save HSV (Hue, Saturation, Value) bounds for different colors and then
# tests the color detection using those bounds. The comments added aim to clarify the purpose of each function and the
# main logic of the script.


import cv2
import numpy as np
import json


# Save the color values to a JSON file
def save_values_to_file(values_dict, filename='saved_colors.json'):
    with open(filename, 'w') as f:
        json.dump(values_dict, f, indent=4)  # Using indent=4 for pretty printing


# Load the color values from a JSON file
def load_values_from_file(filename='saved_colors.json'):
    with open(filename, 'r') as f:
        return json.load(f)


# Placeholder function for OpenCV trackbars
def empty(x):
    pass


# Resize an image to fit within given width and height, maintaining the aspect ratio

def resize_image(image, width, height):
    aspect_ratio = image.shape[1] / float(image.shape[0])
    if width / height >= aspect_ratio:
        return cv2.resize(image, (int(height * aspect_ratio), height))
    else:
        return cv2.resize(image, (width, int(width / aspect_ratio)))


# Get initial color values based on color name
def get_initial_values(color):
    # Define initial color values for various colors in HSV format
    colors = {
        "sky_blue": [85, 130, 100, 255, 100, 255],
        "green": [35, 85, 60, 255, 40, 255],
        "light_green": [45, 75, 50, 255, 50, 255],
        "orange": [10, 30, 100, 255, 80, 255],
        "red": [0, 180, 50, 255, 50, 255],
        "yellow": [25, 35, 50, 255, 50, 255],
        "dark_blue": [100, 140, 50, 255, 50, 255],
        "purple": [140, 160, 50, 255, 50, 255],
        "pink": [150, 170, 50, 255, 50, 255],
        "brown": [5, 25, 50, 200, 20, 200],
        "gray": [0, 179, 0, 50, 10, 200],
        "black": [0, 179, 0, 255, 0, 40],
        "default": [0, 179, 0, 255, 0, 255]
    }
    return colors.get(color_name, colors["default"])


# Track and adjust color values in real-time
def color_tracker(source, color="default"):
    is_webcam = False
    if isinstance(source, int):  # If source is an integer, assume it's a webcam index
        is_webcam = True
        cap = cv2.VideoCapture(source)

    # Create a window with trackbars to adjust HSV values
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 340)

    hue_min, hue_max, sat_min, sat_max, val_min, val_max = get_initial_values(color_name)
    cv2.createTrackbar("Hue Min", "TrackBars", hue_min, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", hue_max, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", sat_min, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", sat_max, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", val_min, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", val_max, 255, empty)

    while True:
        if is_webcam:
            ret, img = cap.read()
            if not ret:
                break
        else:
            img = cv2.imread(source)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        hue_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        val_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        img_result = cv2.bitwise_and(img, img, mask=mask)

        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        top_row = np.hstack([img, img_hsv])
        bottom_row = np.hstack([mask_colored, img_result])
        combined_img = np.vstack([top_row, bottom_row])

        max_width = 800
        max_height = 800
        resized_combined = resize_image(combined_img, max_width, max_height)

        cv2.imshow("All Images", resized_combined)

        # Check if the "TrackBars" window was closed (or q was pressed)
        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty("TrackBars", 0) < 0:
            break

    # When the loop ends, release resources and return the color values
    if is_webcam:
        cap.release()

    cv2.destroyAllWindows()

    # Save and return the values when the loop breaks (window closed or 'q' pressed)
    color_values = {color_name: {
        'hue_min': hue_min,
        'hue_max': hue_max,
        'sat_min': sat_min,
        'sat_max': sat_max,
        'val_min': val_min,
        'val_max': val_max
    }}

    return color_values


# Check the detected colors in real-time using a webcam
def check_colors_with_webcam(color_values):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros_like(frame[:, :, 0])  # Initialize a black mask

        # Check each color and apply its mask
        for color_name, values in color_values.items():
            lower = np.array([values['hue_min'], values['sat_min'], values['val_min']])
            upper = np.array([values['hue_max'], values['sat_max'], values['val_max']])

            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Display the result
        result = cv2.bitwise_and(frame, frame, mask=combined_mask)
        cv2.imshow('Detected Colors', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    user_input = input("Do you want to initialize the color range for this project? (yes/no): ").strip().lower()

    results = {}

    if user_input == 'yes':
        source_image = 0  # Or your image path
        colors_to_process = ["yellow", "blue", "purple", "pink"]

        for color_name in colors_to_process:
            print(f"Tracking for {color_name}...")
            values = color_tracker(source_image, color_name)
            print(f"values selected for {color_name} : {', '.join(map(str, values[color_name].values()))}")

            results[color_name] = values[color_name]  # Flatten the nested dictionary

        # Save the results to a file
        save_values_to_file(results)

    # If not initializing, load the values from the saved file
    elif user_input == 'no':
        results = load_values_from_file()
        print("Loaded color values from file.")

    else:
        print("Invalid input. Exiting...")
        exit()

    # Check the identified colors using the webcam
    user_check = input("Do you want to check color identification? (yes/no): ").strip().lower()
    if user_check == 'yes':
        check_colors_with_webcam(results)
    elif user_check == 'no':
        print("Exiting...")
        exit()
    else:
        print("Invalid input. Exiting...")
        exit()
