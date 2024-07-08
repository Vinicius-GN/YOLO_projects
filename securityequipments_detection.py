from ultralytics import YOLO
import cv2
import cvzone
import math
from lib.sort import*
import numpy as np

#YOLOv8 model retrained using the Following Dataset (https://drive.google.com/drive/folders/1yMGJU0hL1WcQCua4TUgRPV9fIHcFdQdq?usp=drive_link_)

# Load a custom-trained YOLOv8 model for detecting specific safety equipment and vehicles.
model = YOLO('Yolo_weights/best.pt')

# Open a video file to analyze; ensure the path and filename are correct.
rec = cv2.VideoCapture("Images/sec_equipment/3.mp4")

# Verify that the video was successfully opened.
if not rec.isOpened():
    print("Error: Could not open video.")
    exit()

# List of specific objects the model is trained to recognize, focused on safety equipment and vehicles.
clss_objects = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 
                'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 
                'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 
                'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

# Main loop to process the video frame by frame.
while True:
    success, img = rec.read()  # Attempt to read the next frame from the video.

    # Check if the frame was successfully read.
    if not success:
        print("Error: Could not read frame.")
        break  # Exit the loop if no more frames are available or if there's a read error.

    result = model(img, stream=True)  # Apply the YOLO model to the frame for object detection.

    # Iterate over each detection in the frame.
    for r in result:
        boxes = r.boxes  # Retrieve bounding boxes from the results.
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates of the bounding box.
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)  # Convert coordinates to integers.
            w, h = (x2-x1), (y2-y1)  # Calculate width and height of the bounding box.
            cls = int(box.cls[0])  # Determine the class index of the detected object.
            accuracy = math.ceil(box.conf[0] * 100)/100  # Calculate and round the confidence level.

            # Use different colors to indicate different types of detected items based on their safety status.
            if accuracy > 0.3:  # Process only detections with confidence higher than 30%.
                if clss_objects[cls] in ['Hardhat', 'Mask', 'Safety Vest', 'Gloves']:
                    # Use green color for safety-compliant detections.
                    cvzone.cornerRect(img, [x1, y1, w, h], l=5, colorR=(0, 255, 0), rt=2)
                elif clss_objects[cls] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'NO-Gloves']:
                    # Use red color for safety-noncompliant detections.
                    cvzone.cornerRect(img, [x1, y1, w, h], l=5, colorR=(0, 0, 255), rt=2)
                else:
                    # Use blue color for other objects.
                    cvzone.cornerRect(img, [x1, y1, w, h], l=5, colorR=(255, 0, 0), rt=2)

                # Display the class and accuracy of the detection on the image.
                cvzone.putTextRect(img, f'{clss_objects[cls]} {accuracy}', (max(0, x1), max(20, y1)), scale=1)

    # Display the processed image in a window.
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the video stream.
        break

# Release the video capture object and close all OpenCV windows after the loop finishes.
rec.release()
cv2.destroyAllWindows()




