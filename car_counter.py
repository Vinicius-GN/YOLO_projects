from ultralytics import YOLO
import cv2
import cvzone
import math
from lib.sort import*
import numpy as np

# Initialize the video capture and load the YOLO model with pre-trained weights for object detection
cap = cv2.VideoCapture("Images/car_counter/cars.mp4")
model = YOLO('../Yolo_weights/yolov8n.pt')

# List of class objects that the model can detect
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
mask = cv2.imread("Images/car_counter/mask.png")  # Load a mask image to focus detection on specific regions
count = 0
ID_list = []

# Configure the tracker with specific parameters for aging and IOU thresholds
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Main processing loop
while(True):
    sucess, img = cap.read()  # Read a frame from the video
    imgRegion = cv2.bitwise_and(img, mask)  # Apply the mask to the frame to isolate the region of interest
    results = model(imgRegion, stream = True)  # Detect objects within the masked region

    cars = cv2.imread("Images/car_counter/graphics.png", cv2.IMREAD_UNCHANGED)  # Load an overlay image
    img = cvzone.overlayPNG(img, cars, [0,0])  # Overlay the graphics onto the original frame

    # Initialize an empty array for storing detections
    detections = np.empty((0, 5))

    # Define a line for counting crossing objects
    limits_x1 = 400
    limits_y1 = 297
    limits_x2 = 673
    limits_y2 = 297

    # Process each detected object
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]  # Extract the bounding box coordinates
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            confidence = math.ceil((box.conf[0]*100))/100  # Calculate the confidence level

            cls = int(box.cls[0])  # Get the class index
            if clss_objects[cls] in ['car', 'motorcycle', 'bus', 'bicycle', 'truck'] and confidence > 0.2:
                tracking_info = np.array([x1,y1,x2,y2,confidence])  # Prepare tracking info
                detections = np.vstack((detections, tracking_info))  # Add to detections

    results_tckr = tracker.update(detections)  # Update the tracker with new detections
    cv2.line(img, (limits_x1, limits_y1),(limits_x2, limits_y2), color=(0,0,255), thickness=5)  # Draw the counting line

    # Process each tracked object
    for result in results_tckr:
        x1, y1, x2, y2, ID = result
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        w, h = (x2-x1), (y2-y1)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))  # Draw a rectangle around the tracked object (bounding box representation)
        cx, cy = x1+w//2, y1+h//2  # Calculate the center of the bounding box
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Draw a center point

        # Counting logic
        if (cx >= limits_x1 and cx<=limits_x2) and (limits_y1 - 15 < cy < limits_y2 + 15) and (ID not in ID_list):
            count += 1  # Increment count
            ID_list.append(ID)  # Add ID to the list to avoid recounting
            cv2.line(img, (limits_x1, limits_y1),(limits_x2, limits_y2), color=(0,255,0), thickness=5)  # Change line color when counting

        cvzone.putTextRect(img, f'ID:{int(ID)}', (max(0, x1), max(35, y1)), thickness=3, scale=2, offset=10)  # Display the ID of the tracked object
        cv2.putText(img, f'{count}', (230, 100),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3, color=(0,0,0), thickness=8)  # Display the count

    # Display the images
    cv2.imshow("Image", img)
    cv2.imshow("Image mask", imgRegion)
    cv2.waitKey(1)  # Wait for key press
