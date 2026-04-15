import cv2
import numpy as np

# -------- VIDEO FILE LOAD --------
cap = cv2.VideoCapture("traffic.mp4")

# Check if video opened
if not cap.isOpened():
    print("Error: Video not found or cannot open")
    exit()

while True:
    ret, frame = cap.read()

    # Video end check
    if not ret:
        print("Video ended")
        break

    # Resize frame
    frame = cv2.resize(frame, (600, 400))

    # -------- ROI (hidden use only) --------
    roi = frame[100:300, 200:400]

    # -------- Noise Reduction --------
    roi = cv2.GaussianBlur(roi, (5,5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # -------- RED --------
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # -------- YELLOW --------
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # -------- GREEN --------
    lower_green = np.array([36,50,70])
    upper_green = np.array([89,255,255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Pixel count
    red_pixels = np.sum(mask_red)
    yellow_pixels = np.sum(mask_yellow)
    green_pixels = np.sum(mask_green)

    # -------- Decision Logic --------
    if red_pixels > 8000:
        signal = "STOP (RED)"
        color = (0,0,255)
        cv2.circle(frame, (300,200), 40, (0,0,255), 4)

    elif yellow_pixels > 8000:
        signal = "READY (YELLOW)"
        color = (0,255,255)
        cv2.circle(frame, (300,200), 40, (0,255,255), 4)

    elif green_pixels > 8000:
        signal = "GO (GREEN)"
        color = (0,255,0)
        cv2.circle(frame, (300,200), 40, (0,255,0), 4)

    else:
        signal = "NO SIGNAL"
        color = (255,255,255)

    # -------- Display Text --------
    cv2.putText(frame, signal, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # -------- Show Final Output --------
    cv2.imshow("Traffic Light Detector", frame)

    # Exit on ESC
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()