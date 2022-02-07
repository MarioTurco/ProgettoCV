from audioop import maxpp
import numpy as np
import csv 
import cv2

x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []

imagepath="Datasets/ICDAR2015/train/x/img_1000.jpg"
image = cv2.imread(imagepath)
new_image = image
new_image2 = image

#Usa un reader CSV per estrarre le singole coordinate dal ground truth
coordinates="Datasets/ICDAR2015/train/y/gt_img_1000.txt"
coords = open(coordinates, "r", encoding='utf-8-sig')
reader = csv.reader(coords)
for lines in reader:
    x1.append(int(lines[0]))
    y1.append(int(lines[1]))
    x2.append(int(lines[2]))
    y2.append(int(lines[3]))
    x3.append(int(lines[4]))
    y3.append(int(lines[5]))
    x4.append(int(lines[6]))
    y4.append(int(lines[7]))

for i in range(0, reader.line_num):
    points = np.array([[x1[i], y1[i]], [x2[i], y2[i]], [x3[i], y3[i]], [x4[i], y4[i]]], dtype=np.int32)
    new_image = cv2.polylines(new_image, [points], True, (0,256,0))       
    new_image2 = cv2.rectangle(new_image2, (min(x1[i],x2[i],x3[i],x4[i]), min(y1[i],y2[i],y3[i],y4[i])), (max(x1[i],x2[i],x3[i],x4[i]), max(y1[i],y2[i],y3[i],y4[i])), (0,0,205))    

cv2.imshow('Bounding Boxes', new_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()