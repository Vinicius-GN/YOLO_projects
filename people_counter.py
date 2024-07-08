from ultralytics import YOLO 
import cv2  
import cvzone 
import math  
from lib.sort import Sort  
import numpy as np  

# Initialize video capture from a video file
cap = cv2.VideoCapture("Images/people_counter/people.mp4")
# Load the YOLO model with pre-trained weights
model = YOLO('../Yolo_weights/yolov8n.pt')

# Define the classes that the model can detect
clss_objects =[
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
mask = cv2.imread("Images/people_counter/mask1.png")  # Load a mask image to focus detection on specific areas
count_up = 0  # Counter for objects moving upwards
count_down = 0  # Counter for objects moving downwards
ID_list = []  # List to keep track of counted object IDs
tracking = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # Initialize the SORT tracker

# Define coordinates for counting lines
Up = [103, 161, 296, 161]
Down = [527, 403, 725, 403]

while True:
    success, img = cap.read()  # Read frames from the video
    imgRegion = cv2.bitwise_and(img, mask)  # Apply the mask to isolate the region of interest
    result = model(imgRegion, stream=True)  # Perform object detection within the masked region

    setas = cv2.imread("Images/people_counter/graphics-1.png", cv2.IMREAD_UNCHANGED)  # Load overlay graphics
    img = cvzone.overlayPNG(img, setas, [650, 0])  # Overlay graphics onto the video frame

    detections = np.empty((0, 5))  # Initialize an empty array for detections

    # Process each detection result
    for r in result:
        boxes = r.boxes  # Get the bounding boxes of detected objects
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract and convert box coordinates to integers

            # Calculate the confidence score
            confidence = math.ceil((box.conf[0]*100))/100

            # Prepare tracking information
            tracking_info = np.array([x1, y1, x2, y2, confidence])
            detections = np.vstack((detections, tracking_info))  # Append detection info to the array

    results_trckr = tracking.update(detections)  # Update the tracker with new detections
    cv2.line(img, (Up[0], Up[1]), (Up[2], Up[3]), color=(0, 0, 255), thickness=4)  # Draw the upper counting line
    cv2.line(img, (Down[0], Down[1]), (Down[2], Down[3]), color=(0, 0, 255), thickness=4)  # Draw the lower counting line

    # Process tracking results
    for result in results_trckr:
        x1, y1, x2, y2, ID = map(int, result)
        w, h = (x2-x1), (y2-y1)

        cvzone.putTextRect(img, f'ID:{ID}', (max(0, x1), max(35, y1)), thickness=2, scale=2, offset=10)  # Display the ID on the frame
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))  # Highlight the tracked object with a rectangle

        # Define the center point of the bounding boxes
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Draw the center point

        # Counting logic for objects crossing the lines
        if (Up[0] < cx < Up[2]) and (Up[1] - 15 < cy < Up[1] + 15) and (ID not in ID_list):
            ID_list.append(ID)
            count_up += 1
            cv2.line(img, (Up[0], Up[1]), (Up[2], Up[3]), color=(0, 255, 0), thickness=6)  # Change line color when counting
        elif (Down[0] < cx < Down[2]) and (Down[1] - 15 < cy < Down[1] + 15) and (ID not in ID_list):
            ID_list.append(ID)
            count_down += 1
            cv2.line(img, (Down[0], Down[1]), (Down[2], Down[3]), color=(0, 255, 0), thickness=6)  # Change line color when counting
        
        # Display counters on the screen
        cv2.putText(img, f'{count_up}', (830, 85), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,0,0), thickness=3)
        cv2.putText(img, f'{count_down}', (1100, 85), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,0,0), thickness=3)

    # Display the video with overlays
    cv2.imshow("Image", img)
    cv2.imshow("Image mask", imgRegion)
    cv2.waitKey(1)  # Update the display and wait for a key press
