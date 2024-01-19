from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*
import numpy as np

#For the counting accuracy we need to use a traking algoritm to save an ID to every single car that enters the image regionS

cap = cv2.VideoCapture("Images/cars.mp4")
model = YOLO('../Yolo_weights/yolov8n.pt')
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
mask = cv2.imread("Images/mask.png")
count = 0
ID_list = []
#Defining the Tracking constants (IOU -> quão bom é o overlaping)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while(True):
    sucess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream = True)

    cars = cv2.imread("Images/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, cars, [0,0])

    #Inicialize an empty array of five values for saving the detections in the region
    detections = np.empty((0, 5))

    #Line for counting
    limits_x1 = 400
    limits_y1 = 297
    limits_x2 = 673
    limits_y2 = 297

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Boxes
            x1,y1,x2,y2 = box.xyxy[0]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            #w, h = (x2-x1), (y2-y1)
            #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
            #Confidence
            confidence = math.ceil((box.conf[0]*100))/100

            #Classification
            cls = int(box.cls[0])
            if clss_objects[cls] == 'car' or clss_objects[cls] == 'motorcycle' or clss_objects[cls] == 'bus' or clss_objects[cls] == 'bycycle' or clss_objects[cls] == 'truck' and confidence>0.2:
                #cvzone.putTextRect(img, f'{clss_objects[cls]} {confidence}', (max(0, x1), max(20, y1)), thickness=1, scale=1)
                
                #Tracking information
                tracking_info = np.array([x1,y1,x2,y2,confidence])
                detections = np.vstack((detections, tracking_info))

    results_tckr = tracker.update(detections)
    cv2.line(img, (limits_x1, limits_y1),(limits_x2, limits_y2), color=(0,0,255), thickness=5)

    for result in results_tckr:
        x1, y1, x2, y2, ID = result
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        w, h = (x2-x1), (y2-y1)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        #Defininf the center of the bounding boxes
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        #Counting cars
        if (cx >= limits_x1 and cx<=limits_x2) and (limits_y1 - 15 < cy < limits_y2 + 15) and (ID not in ID_list):
            count += 1
            ID_list.append(ID)
            cv2.line(img, (limits_x1, limits_y1),(limits_x2, limits_y2), color=(0,255,0), thickness=5)

        cvzone.putTextRect(img, f'ID:{int(ID)}', (max(0, x1), max(35, y1)), thickness=3, scale=2, offset=10)
        cv2.putText(img, f'{count}', (230, 100),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3, color=(0,0,0), thickness=8)

    cv2.imshow("Image", img)
    cv2.imshow("Image mask", imgRegion)
    cv2.waitKey(1)
