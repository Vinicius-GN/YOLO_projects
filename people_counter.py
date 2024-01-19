from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*
import numpy as np

cap = cv2.VideoCapture("Images/people.mp4")
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
mask = cv2.imread("Images/mask1.png")
count_up = 0
count_down = 0
ID_list = []
tracking = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)

#Defining limis
Up = [103, 161, 296, 161]
Down = [527, 403, 725, 403]

while(True):
    sucess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    result = model(imgRegion, stream=True)

    setas = cv2.imread("Images/graphics-1.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, setas, [650,0])

    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            #Confidence
            confidence = math.ceil((box.conf[0]*100))/100

            #Classification
            cls = int(box.cls[0])
                
            #Tracking information
            tracking_info = np.array([x1,y1,x2,y2,confidence])
            detections = np.vstack((detections, tracking_info))

    results_trckr = tracking.update(detections)
    cv2.line(img, (Up[0], Up[1]), (Up[2], Up[3]), color=(0, 0, 255), thickness=4)
    cv2.line(img, (Down[0], Down[1]), (Down[2], Down[3]), color=(0, 0, 255), thickness=4)

    for result in results_trckr:
        x1, y1, x2, y2, ID = result
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        w, h = (x2-x1), (y2-y1)

        cvzone.putTextRect(img, f'ID:{int(ID)}', (max(0, x1), max(35, y1)), thickness=2, scale=2, offset=10)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

        #Defining the center of the bounding boxes
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        #Counting people
        if (Up[0] < cx < Up[2]) and (Up[1] - 15 < cy < Up[1] + 15) and (ID not in ID_list):
            ID_list.append(ID)
            count_up += 1
            cv2.line(img, (Up[0], Up[1]), (Up[2], Up[3]), color=(0, 255, 0), thickness=6)
        elif (Down[0] < cx < Down[2]) and (Down[1] - 15 < cy < Down[1] + 15) and (ID not in ID_list):
            ID_list.append(ID)
            count_down += 1
            cv2.line(img, (Down[0], Down[1]), (Down[2], Down[3]), color=(0, 255, 0), thickness=6)
        

        cv2.putText(img, f'{count_up}', (830, 85), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,0,0), thickness=3)
        cv2.putText(img, f'{count_down}', (1100, 85), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4, color=(0,0,0), thickness=3)

    cv2.imshow("Image", img)
    cv2.imshow("Image mask", imgRegion)
    cv2.waitKey(1)
