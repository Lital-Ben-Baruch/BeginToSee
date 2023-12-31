"""
colorPick.py
Author: Lital H. Ben Baruch
Date: 25th September 2023
Contact: lital.h.ben.baruch@gmail.com
Description: Detects and tracks specified colors in real-time using a webcam.
Adjusts and saves HSV (Hue, Saturation, Value) bounds for different colors and tests the color detection.
"""
import os

import cv2
import numpy as np
import json
import warnings

MAX_WIDTH = 500
MAX_HEIGHT = 500

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


def save_values_to_file(values_dict: dict, filename: str = 'saved_colors.json') -> None:
    """Save the color values to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(values_dict, f, indent=4)


# def load_values_from_file(filename: str = 'saved_colors.json') -> dict:
#     """Load the color values from a JSON file."""
#     with open(filename, 'r') as f:
#         return json.load(f)


def empty(x: int) -> None:
    """Placeholder function for OpenCV trackbars."""
    pass


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to fit within given width and height, maintaining the aspect ratio."""
    aspect_ratio = image.shape[1] / float(image.shape[0])
    if width / height >= aspect_ratio:
        return cv2.resize(image, (int(height * aspect_ratio), height))
    else:
        return cv2.resize(image, (width, int(width / aspect_ratio)))


def get_initial_values(color_name: str) -> list:
    """Get initial color values based on color name."""
    return colors.get(color_name, colors["default"])


def get_color_choice():
    """Get user's choice of color."""

    # Print available color choices
    print("Available colors:")
    colors_list = list(colors.keys())
    for idx, color in enumerate(colors_list[:-1], 1):
        print(f"{idx}. {color}")

    # the last color
    last_color_idx = len(colors_list)
    print(f"{last_color_idx}. Other")
    # print(f"{len(colors) + 1}. Other")

    # Get user's choice as a comma-separated string
    choice_str = input("Choose the colors you want to process (separated by commas, e.g., 1, 3, 5): ").strip()

    # Split the user's input into a list of integers
    choice_list = [int(item) for item in choice_str.split(',')]

    # Convert the list of integers into the corresponding color names
    chosen_colors = [list(colors.keys())[i - 1] for i in choice_list]

    # If user chooses to add a different color
    if "default" in chosen_colors:
        default_indices = [i for i, color in enumerate(chosen_colors) if color == "default"]
        for index in default_indices:
            new_color_name = input(f"Enter name for the new {index + 1}-th 'default' color: ").strip()
            chosen_colors[index] = new_color_name

    return chosen_colors


def color_tracker(source, color_name: str = "default") -> dict:
    """Track and adjust color values in real-time."""
    is_webcam = False
    if isinstance(source, int):  # Check if source is an integer (webcam index)
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

        resized_combined = resize_image(combined_img, MAX_WIDTH, MAX_HEIGHT)

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


def create_colors_mask(frame, color_values):
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
    return result


def check_colors_with_source(source, color_values: dict) -> None:
    """Check the detected colors in real-time using a webcam or an image."""
    if isinstance(source, int):  # Check if source is an integer (webcam index)
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            mask_colors = create_colors_mask(frame, color_values)
            combined_img = np.hstack([mask_colors, frame])
            cv2.imshow('Original Live Feed and Detection', combined_img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):  # Check if 'q' key is pressed
                break

        cap.release()

    else:
        frame = cv2.imread(source)
        resized_frame = resize_image(frame, MAX_WIDTH, MAX_HEIGHT)
        mask_colors = create_colors_mask(frame, color_values)
        resized_mask = resize_image(mask_colors, MAX_WIDTH, MAX_HEIGHT)
        # cv2.imshow('Detected Colors', mask_colors)
        # cv2.imshow('Image', frame)
        combined_img = np.hstack([resized_frame, resized_mask])
        cv2.imshow('Original Image and Detection', combined_img)
        cv2.waitKey(0)


def load_values_from_file(filename: str = 'saved_colors.json') -> dict:
    """Load the color values from a JSON file."""
    while True:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Failed to open or read the file: {filename}")
        else:
            print(f"The file {filename} does not exist.")

        # Ask the user for a new file path
        new_filename = input("Please provide the path to the JSON file you want to use: ").strip()
        if new_filename:
            filename = new_filename


if __name__ == "__main__":
    src_input = input("Do you want to use a webcam? (yes/no) [yes]: ").strip().lower()
    if src_input == 'yes' or src_input == '':
        source_image = 0
        source_input = source_image
    else:
        path_input = input("Do you want the image ball_colors.jpg? (yes/no) [yes]:").strip().lower()
        if path_input == 'yes' or path_input == '':
            source_image = "Resources/ball_colors.jpg"
        else:
            source_image = input("Please provide the path to the image you want to use: ").strip().lower()
        source_input = source_image

    colorPref_input = input(
        "Choose an option:\n"
        "1. Recognize the colors yellow, pink, blue, and purple.\n"
        "2. Recognize the colors sky_blue, green, light_green, orange, red, yellow, dark_blue, purple, and pink.\n"
        "3. Define different colors.\n"
        "Enter your choice (1/2/3) [3] : "
    ).strip().lower() or '3'

    if colorPref_input == '1':
        colors_to_process = ["yellow", "dark_blue", "purple", "pink"]
    elif colorPref_input == '2':
        colors_to_process = ["sky_blue", "green", "light_green", "orange", "red", "yellow", "dark_blue", "purple",
                             "pink"]
    elif colorPref_input == '3':
        colors_to_process = get_color_choice()
        cv2.destroyAllWindows()

    user_input = input("Do you want to initialize the color range for this project? (yes/no) [no]: ").strip().lower()
    results = {}
    if user_input == 'yes':
        for color_name in colors_to_process:
            print(f"Tracking for {color_name}...")
            values = color_tracker(source_image, color_name)
            print(f"values selected for {color_name} : {', '.join(map(str, values[color_name].values()))}")
            results[color_name] = values[color_name]  # Flatten the nested dictionary

        save_values_to_file(results)

    elif user_input == 'no' or user_input == '':
        json_file = input(
            "Do you want to load the saved_colors.json file? (yes/no) [yes]: ").strip()
        if json_file == 'yes' or json_file == '':
            results = load_values_from_file()
            print("Loaded color values from saved_colors.json file.")
        else:
            json_file_name = input(
                "Please provide the path to the json file you want to use:").strip()
            results = load_values_from_file(json_file_name)
            print(f"Loaded color values from {json_file_name} file.")

    else:
        print("Invalid input. Exiting...")
        exit()

    user_check = input("Do you want to check color identification? (yes/no) [yes]: ").strip().lower()
    if user_check == 'yes' or user_check == '':
        print("Press q to exit")
        check_colors_with_source(source_input, results)

    elif user_check == 'no':
        print("Exiting...")
        exit()
    else:
        print("Invalid input. Exiting...")
        exit()
