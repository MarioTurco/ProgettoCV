import csv
from hashlib import new
import os

x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []

train_y_path="Datasets/ICDAR2015/train/y"
for filename in os.listdir(train_y_path):
    file = open(os.path.join(train_y_path, filename), 'r', encoding='utf-8-sig')
    file2 = open(os.path.join(train_y_path, "new"+filename), 'w', encoding='utf-8-sig', newline='')
    reader = csv.reader(file)
    writer = csv.writer(file2)
    new_line = []
    for lines in reader:
        x1 = (int(lines[0]))
        y1 = (int(lines[1]))
        x2 = (int(lines[2]))
        y2 = (int(lines[3]))
        x3 = (int(lines[4]))
        y3 = (int(lines[5]))
        x4 = (int(lines[6]))
        y4 = (int(lines[7]))
        new_x1 = min(x1,x2,x3,x4)
        new_y1 = min(y1,y2,y3,y4)
        new_x2 = max(x1,x2,x3,x4)
        new_y2 = max(y1,y2,y3,y4)
        new_line.append([new_x1, new_y1, new_x2, new_y2])
   
    writer.writerows(new_line)
    new_line=[]
    




        


