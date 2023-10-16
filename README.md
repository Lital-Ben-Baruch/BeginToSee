# BeginToSee

An introductory repository dedicated to foundational image processing and computer vision techniques using OpenCV. Ideal for beginners looking to grasp the core concepts and dive into the world of visual computing.

## Introduction

Welcome to BeginToSee, your gateway to the exciting world of image processing and computer vision. This repository provides a comprehensive introduction to essential techniques using OpenCV and Python.

## Repository Contents

- **camera_live_feed.py**: Basic Python script for capturing live video from a camera source and displaying it.
- **SmartDocScanner.py**: This is the main script for the SmartDocScanner project, which aims to scan and rectify documents with layout issues when captured using the camera.
- **colorPick.py**: A Python script for real-time color detection and tracking using a webcam or image source. Choose from predefined or custom colors, initialize color ranges, and visualize color detection.
- **select_points_in_image.py**: Python script for selecting points on an image interactively. Points are displayed on the image.
- **utils**: This directory contains utility files used by various scripts.

## Videos

Explore the capabilities of the `colorPick.py` script through a series of demonstration videos:

- **[Video 1](https://www.youtube.com/watch?v=nVIYtBjiknQ) - Color Detection with Webcam (Default Options)**
    - Use a webcam as the image source.
    - Detect predefined colors.
    - Load color values from an existing JSON file (`saved_colors_web.json`).
    - Observe real-time color detection.

- **[Video 2](https://www.youtube.com/watch?v=fknPJysUATI) - Color Detection with Webcam (New Colors)**
    - Use a webcam as the image source.
    - Choose custom colors to detect.
    - Initialize color ranges for new colors.
    - Experience real-time color detection.

- **[Video 3](https://youtu.be/v0LJaIhnoO0) - Color Detection with Image (Default Options)**
    - Select an image as the source.
    - Detect predefined colors.
    - Load color values from an existing JSON file (`saved_colors_img.json`).
    - View color detection results.

- **[Video 4](https://youtu.be/0PIJ31NGWLA) - Color Detection with Image (New Colors)**
    - Choose an image as the source.
    - Define custom colors for detection.
    - Initialize color ranges for new colors.
    - Visualize color detection outcomes.

> Note: For more advanced functionality and features, check out the `visual-painter` branch of this repository. The "visual-painter" branch contains additional videos (5 and 6) with enhancements and interactive features, such as eraser functionality, eraser size magnification, and the ability to clear the image entirely. These enhancements provide more control and interactivity when using the script. Be sure to explore the updated code and demonstration videos in that branch to see these features in action!


## Using Phone Camera Feed in SmartDocScanner.py

To use the camera feed from your phone, follow these steps:

1. Download the "IP Webcam" app from your phone's app store.
2. Connect both your computer and phone to the same Wi-Fi network.
3. Open the "IP Webcam" app on your phone and tap "Start server" to begin streaming.
4. Copy the first IP address shown on your phone's screen (e.g., `https://your_phone_ip:8080/`).
5. Paste this IP address into a web browser on your computer and press Enter.
6. A web page will open; choose "video renderer" and select "Full screen."
7. Right-click on the video, select "Inspect," and hover over the video element to find the "Current source" link.
8. Copy the URL of the video source and paste it into your Python code as follows:

   ``` python
   Capture = cv2.VideoCapture("http://your_phone_ip:8080/video")
   ```

9. Replace `"your_phone_ip"` with the actual IP address from step 4.


## Dependencies

This project relies on OpenCV, a popular computer vision library for Python. You can install it using the following command:

```bash
pip install opencv-python
```

## Important Note for SmartDocScanner.py

The `SmartDocScanner.py` script relies on utility functions stored in separate `utils` files to function correctly. These utility files should be present in the same directory as `SmartDocScanner.py` for the script to work as intended. Please ensure that you have the following `utils` files in your project directory:

- `image_utils.py`: Contains utility functions for image preprocessing and manipulation.
- `contour_utils.py`: Contains utility functions for working with contours and shapes in images.

Make sure to include these utility files when working with the `SmartDocScanner.py` script.

## License

This project is licensed under the [GNU General Public License (GPL)](LICENSE).

## Get Started

To get started with the code and explore the provided videos, follow these steps:
1. Clone this repository to your local machine.
2. Install the necessary dependencies (OpenCV).
3. Run the desired Python scripts.
4. Check out the demonstration videos to see the code in action.

## Contributing

Contributions to this repository are welcome! Whether you want to improve the code, add new features, or fix issues, please feel free to submit a pull request.

## Issues

If you encounter any problems or have suggestions, please open an issue

Enjoy your journey into the world of computer vision and image processing with BeginToSee!

For questions or inquiries, please contact [lital.h.ben.baruch@gmail.com](mailto:lital.h.ben.baruch@gmail.com).