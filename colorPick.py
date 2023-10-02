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

MAX_WIDTH = 500
MAX_HEIGHT = 500
BB_RADIUS_CENTER = 8
BB_AREA = 50
draw = False
eraser_resize = 1

my_color_value_dict = {  # BGR
    "sky_blue": [255, 255, 0],
    "green": [0, 204, 0],
    "light_green": [0, 255, 0],
    "orange": [0, 128, 255],
    "red": [0, 0, 255],
    "yellow": [0, 255, 255],
    "dark_blue": [255, 0, 0],
    "purple": [255, 0, 127],
    "pink": [255, 0, 255],
    "brown": [0, 51, 102],
    "gray": [128, 128, 128],
    "black": [0, 0, 0],
    "default": [255, 255, 255]
}

my_points = []  # [x, y, color]
my_points_del = []  # [x, y, color]

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


def draw_on_canvas(points, img_res):
    for point in points:
        cv2.circle(img_res, (point[0], point[1]), BB_RADIUS_CENTER, my_color_value_dict[point[2]], cv2.FILLED)


def delete_from_canvas(points, img_res, backup_image):
    for point in points:
        cv2.circle(img_res, (point[0], point[1]), point[3], (0, 0, 0),
                   cv2.FILLED)  # fill with black. for my original color my_color_value_dict[point[2]]

        # Create a mask for the erased area (white circle on a black background)
        circular_mask = np.zeros_like(img_res)
        x, y, r = point[0], point[1], point[3]  # Modify these values according to your needs
        cv2.circle(circular_mask, (x, y), r, (255, 255, 255), thickness=cv2.FILLED)

        # Restore the circular region from the backup_image
        circular_region = cv2.bitwise_and(backup_image, circular_mask)

        img_res = cv2.bitwise_or(img_res, circular_region)
    return img_res


def create_colors_mask(frame, color_values, source):
    frame_result = frame.copy()
    new_point = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros_like(frame[:, :, 0])  # Initialize a black mask
    counter = 0
    # Check each color and apply its mask
    for color_name, values in color_values.items():
        lower = np.array([values['hue_min'], values['sat_min'], values['val_min']])
        upper = np.array([values['hue_max'], values['sat_max'], values['val_max']])

        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        # Detecting contours, adding bounding boxes, and marking the center
        x_center, y_center = process_contours(mask, frame_result)
        if isinstance(source, int):  # Check if source is an integer (webcam index)
            # cv2.circle(frame_result, (x_center, y_center), BB_RADIUS_CENTER, my_color_value[counter], cv2.FILLED)
            cv2.circle(frame_result, (x_center, y_center), BB_RADIUS_CENTER, my_color_value_dict[color_name], cv2.FILLED)

        if x_center != 0 and y_center != 0:
            new_point.append([x_center, y_center, color_name, BB_RADIUS_CENTER])

        counter += 1

    mask_result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    return mask_result, frame_result, new_point


def check_colors_with_source(source, color_values, draw_flag):
    """Check the detected colors in real-time using a webcam or an image."""
    global my_points
    global my_points_del
    global eraser_resize
    clear_canvas = False

    if isinstance(source, int):  # Check if source is an integer (webcam index)
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            backup_image = frame.copy()

            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):  # Check if 'c' key is pressed
                clear_canvas = True
                my_points = []  # Clear the list of points
                my_points_del = []

            if key & 0xFF == ord('b'):  # Check if 'b' key is pressed
                eraser_resize += 1

            mask_colors, frame_res, color_points = create_colors_mask(frame, color_values, source)
            circular_mask = np.zeros_like(frame_res)
            # Clear the canvas if clear_canvas is True
            if clear_canvas:
                frame_res = frame.copy()
                my_points = []  # Clear the list of points
                my_points_del = []
                clear_canvas = False  # Reset the canvas clear flag
                circular_mask = np.zeros_like(frame_res)
            # read the points from the color_points and draw them
            if draw_flag:
                eraser_color = 'yellow'
                if color_points:
                    for point in color_points:
                        if point[2] != eraser_color:
                            my_points.append(point)
                        elif my_points:
                            print('point before', point)
                            point[3] *= eraser_resize
                            print('point after', point)
                            my_points_del.append(point)

                if len(my_points) != 0:
                    draw_on_canvas(my_points, frame_res)
                    if len(my_points_del) != 0:
                        frame_res = delete_from_canvas(my_points_del, frame_res, backup_image)

                    # Debug: Display the circular_mask
                    circular_mask = np.zeros_like(frame_res)
                    for point in my_points_del:
                        x, y, r = point[0], point[1], point[3]
                        cv2.circle(circular_mask, (x, y), r, (255, 255, 255), thickness=cv2.FILLED)
                    # cv2.imshow('Circular Mask', circular_mask)

            top_row = np.hstack([mask_colors, frame_res])
            bottom_row = np.hstack([circular_mask, circular_mask])
            combined_img = np.vstack([top_row, bottom_row])
            cv2.imshow('Original Live Feed and Detection', combined_img)
            # else:
            #     combined_img = np.hstack([mask_colors, frame_res])
            #     cv2.imshow('Original Live Feed and Detection', combined_img)
            if key & 0xFF == ord('q'):  # Check if 'q' key is pressed
                break

        cap.release()

    else:
        frame = cv2.imread(source)
        resized_frame = resize_image(frame, MAX_WIDTH, MAX_HEIGHT)
        mask_colors, frame_res, color_points = create_colors_mask(frame, color_values, source)
        resized_mask = resize_image(mask_colors, MAX_WIDTH, MAX_HEIGHT)
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


def process_contours(mask, frame_to_draw=None):
    cnt_color = (255, 0, 0)  # Blue color for drawing the detected contours.
    cnt_thick = 3  # Thickness of the contour lines.
    largest_area = 0  # To keep track of the largest contour's area.
    cx, cy = 0, 0  # Initialize the center coordinates.

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Processing Each Contour:

    for contour in contours:
        area = cv2.contourArea(contour)
        # The contour is considered relevant if its area is greater than 500 (to ignore noise) and if its area is larger
        # than the previous largest contour's area.
        if area > BB_AREA and area > largest_area:
            largest_area = area
            # Finding the Centroid:
            # Calculates moments, which are a set of scalar values that provide information about the image's shape.
            # The centroid (center) of the contour is calculated using these moments.
            M = cv2.moments(contour)
            if M["m00"] != 0:  # avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            # Visualization:
            # Draw contours if a 'frame_to_drawq' is provided
            if frame_to_draw is not None:
                cv2.drawContours(frame_to_draw, contour, -1, cnt_color, cnt_thick)

    # Returns the x and y coordinates of the largest contour's center.
    return cx, cy


if __name__ == "__main__":
    colors_to_process =[]
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
        "1. Recognize the colors yellow, dark_blue, purple, pink, and green.\n"
        "2. Recognize the colors sky_blue, green, light_green, orange, red, yellow, dark_blue, purple, and pink.\n"
        "3. Define different colors.\n"
        "Enter your choice (1/2/3) [3] : "
    ).strip().lower() or '3'

    if colorPref_input == '1':
        colors_to_process = ["yellow", "dark_blue", "purple", "pink", "green"]
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
            "Choose an option:\n"
            "1. load the saved_colors.json file.\n"
            "2. load the saved_colors_web.json file.\n"
            "3. load the saved_colors_img.json file.\n"
            "4. Define different file.\n"
            "Enter your choice (1/2/3/4) [1] : "
        ).strip().lower() or '1'

        if json_file == '1' or json_file == '':
            results = load_values_from_file()
            print("Loaded color values from saved_colors.json file.")

        if json_file == '2':
            json_file_name = 'saved_colors_web.json'
            results = load_values_from_file(json_file_name)
            print(f"Loaded color values from {json_file_name} file.")

        if json_file == '3':
            json_file_name = 'saved_colors_img.json'
            results = load_values_from_file(json_file_name)
            print(f"Loaded color values from {json_file_name} file.")

        if json_file == '4':
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
        check_colors_with_source(source_input, results, draw)
        cv2.destroyAllWindows()

    if isinstance(source_image, int):  # Check if source is an integer (webcam index)
        user_check = input("\n\n Do you want to start drawing? (yes/no) [yes]: ").strip().lower()
        if user_check == 'yes' or user_check == '':
            print("Press q to exit and c to clear the drawing")
            draw = True
            check_colors_with_source(source_input, results, draw)

        elif user_check == 'no':
            print("Exiting...")
            exit()
        else:
            print("Invalid input. Exiting...")
            exit()
