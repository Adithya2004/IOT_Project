import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
#cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('mall_people.mp4')
area = [(367,160),(367,220),(767,220),(767,160)]
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
area_in = set()
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 4 != 0:
        continue
    count=0
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    tracker = Tracker()
    l=[]
    #print(px)
    for index,row in px.iterrows():
        cl = int(row[5])
        if(cl==0):
            x1,y1,x2,y2,conf = int(row[0]),int(row[1]),int(row[2]),int(row[3]),row[4]
            if(conf>0.1):
                l.append([x1,y1,x2,y2])
    bbox_id = tracker.update(l)
    xq =0
    for bbox in bbox_id:
        x3,y3,x4,y4,i = bbox
        cx,cy = int(x3+x4)//2,int(y3+y4)//2
        results = cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if results >= 0:
            xq+=1
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),3)
            cv2.putText(frame,str(i),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
            area_in.add(i)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,0),3)
    count_in_area = len(area_in)
    cv2.putText(frame,str(xq),(60,120),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
