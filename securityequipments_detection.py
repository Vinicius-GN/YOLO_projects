from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*
import numpy as np

#Model trained using Google Colab with the Dataset (https://drive.google.com/drive/folders/1yMGJU0hL1WcQCua4TUgRPV9fIHcFdQdq?usp=drive_link_)

model = YOLO('best.pt')
rec = cv2.VideoCapture("Images/4.mp4")

clss_objects = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 
                'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 
                'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 
                'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

while (True):
    sucess, img = rec.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            w, h = (x2-x1), (y2-y1)
            cls = int(box.cls[0])
            accuracy = math.ceil(box.conf[0] * 100)/100

            if(accuracy > 0.3):
                if clss_objects[cls] == 'Hardhat' or clss_objects[cls] == 'Mask' or clss_objects[cls] == 'Safety Vest' or clss_objects[cls] == 'Gloves':
                    cvzone.cornerRect(img, [x1,y1, w, h], l=5, colorR=(0, 255, 0), rt = 2)
                elif clss_objects[cls] == 'NO-Hardhat' or clss_objects[cls] == 'NO-Mask' or clss_objects[cls] == 'NO-Safety Vest' or clss_objects[cls] == 'NO-Gloves':
                    cvzone.cornerRect(img, [x1,y1, w, h], l=5, colorR=(0, 0, 255), rt = 2)
                else:
                    cvzone.cornerRect(img, [x1,y1, w, h], l=5, colorR=(255, 0, 0), rt = 2)

                cvzone.putTextRect(img, f'{clss_objects[cls]} {accuracy}', (max(0, x1), max(20, y1)), scale=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
