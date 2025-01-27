import cv2

import numpy as np


# Emma Chetan PWP Circumference and Center Project Video
# Draw a circle on the outermost circumference of the top of a can. Draw its center point.

# For each frame, apply a mask to only detect the top of the can, and return the circumference and center drawn on the frame
def frame_processor(img):

    # Apply an hsv to only register the top of the can.
    # https://www.geeksforgeeks.org/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-with-cv-inrange-opencv/
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_silver = np.array([0, 0, 0])

    upper_silver = np.array([180, 20, 255])

    # The mask will only detect the top of the can. All circles will be detected within the mask.
    mask = cv2.inRange(hsv_image, lower_silver, upper_silver)

    hsvapplied = cv2.bitwise_and(img, img, mask=mask)

    # Convert to grayscale.
    gray = cv2.cvtColor(hsvapplied, cv2.COLOR_BGR2GRAY)

    # Blur using 5 x 5 kernel to reduce noise.
    gray_blurred = cv2.blur(gray, (5, 5))

    # Apply Hough transform to detect all possible circles, returning the center coordinates and radius of each circle.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=70, minRadius=50, maxRadius=1000)

    # https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
    # Convert the circle parameters a, b and r to integers.
    if detected_circles is not None:

        detected_circles = np.uint16(np.around(detected_circles))

        # pt is the first, optimal detected circle.
        pt = detected_circles[0][0]

        # Where x1 and x2 are the coordinates of the center point.
        x1, x2, radius = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(img, (x1, x2), radius, (0, 255, 0), 2)

        # Draw a small circle to show the center.
        cv2.circle(img, (x1, x2), 1, (0, 0, 255), 3)

    return img


# Open the video capture on the video of the can.
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
cap = cv2.VideoCapture('can2.mov')  # 0 for default camera, or provide a video file path

if cap.isOpened():

    while True:

        try:

            # Read the frame from the video capture.
            success, frame = cap.read()

            # If the frame was not read correctly, break the loop to end the program.
            if success == True:

                # Display the frame.
                cv2.imshow("Emma Chetan Circumference and Center Video PWP", frame_processor(frame))

                # Check if the user pressed 'q' to quit.
                if cv2.waitKey(25) == ord('q'):

                    break

            else:

                break

        except Exception:

            continue

# Release the video capture and close all windows.
cap.release()

cv2.destroyAllWindows()
