import cv2
import time
import numpy as np
import copy 
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

W_View_size = 480

filename = os.listdir('/home/jwsi/moonClassification/dataset')
filename.sort()

label = [1,3,1,2,1,1,1,1,1,1,
        1,5,5,1,1,1,1,1,5,5,
        4,5,3,3,3,2,3,3,3,2,
        4,2,2,4,2,2,2,2,4,2,
        2,3,4,4,4,4,4,4,4,2,
        2,2,2,2,3,3,3,3,5,5,
        5,5,5,3,4,4,4,4,5,5,
        5,5,3,3,3,4,4,4,2,2,
        4,2,3,4,4,1,1,1,1,5,
        5,5,5,5,3,3,5,5,2,1,1]

matrix = [[0,0,0,0,0]for i in range(5)]


def calculate(predict, i):
    global label
    global matrix
   
    ans = int(label[i])
  
    if predict == ans:
        matrix[ans-1][ans-1] += 1
    else:
        matrix[ans-1][predict-1] += 1
    
   
for i in range(len(filename)):
    img = cv2.imread('/home/jwsi/moonClassification/dataset/'+filename[i])
    height, width = img.shape[0], img.shape[1]
    rate = width / W_View_size
    print(filename[i])

    H_View_size = int(height / rate)
    
    img = cv2.resize(img, (W_View_size,H_View_size))
    cimg = copy.deepcopy(img)
    realimg = copy.deepcopy(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 1Stage

    img = cv2.Canny(img, 100, 150)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)) #(9, 9)
   
    img = cv2.dilate(img, kernel, iterations=6)
    img = cv2.erode(img, kernel,iterations=6)
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40,param1=30,param2=10,minRadius=0, maxRadius=140)
    circles = np.uint16(np.around(circles))
    
    maxRadius = 0
    cord = [0,0]

    for a in circles[0,:]:
        if maxRadius < a[2] :
            maxRadius = a[2]
            cord= [a[0], a[1]]

    left = cord[0] - int(maxRadius*1.5)
    top = cord[1] - int(maxRadius*1.5)
    right = cord[0] + int(maxRadius*1.5)
    bottom = cord[1] + int(maxRadius*1.5)

    if left < 0 : left = 0
    if top < 0 : top = 0
    if right > W_View_size : right = 480
    if bottom > H_View_size : bottom = H_View_size

    cimg = cimg[top:bottom, left:right]
    rimg = copy.deepcopy(cimg)
    rimg = cv2.resize(rimg, (480,480))

    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    cimg = cv2.resize(cimg, (480,480))

    ret3, th3 = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    # 2Stage

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #(9, 9)
   
    th3 = cv2.dilate(th3, kernel, iterations=2)
    th3 = cv2.erode(th3, kernel,iterations=2)

    circles = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT, 1, 30,param1=30,param2=10,minRadius=0, maxRadius=175)

    circles = np.uint16(np.around(circles))
    
    maxRadius = 0
    cord = [0,0]

    for a in circles[0,:]:
        if maxRadius < a[2] :
            maxRadius = a[2]
            cord= [a[0], a[1]]

    left = int(cord[0]) - maxRadius
    top = int(cord[1]) - maxRadius
    right = int(cord[0]) + maxRadius
    bottom = int(cord[1]) + maxRadius


    if left < 0  : left = 0
    if top < 0 : top = 0
    if right > 480 : right = 480
    if bottom > 480 : bottom = 480

    th3 = th3[top:bottom, left:right]

    height = 100
    width = 100
    radius = 50


    th3 = cv2.resize(th3, (height,width))
    
    result = np.zeros((height, width), np.uint8)

    circlearea = 0
    countarea = 0
    weight = [0, 0]

    for y in range(height):
        for x in range(width):
            if (x-radius)**2 + (y-radius)**2 <= radius**2:
                circlearea += 1
                if th3[y][x] > 0:
                    result[y][x] = 255
                    if x  < radius : weight[0] += 1
                    else : weight[1] += 1

                    countarea += 1

    rate = countarea/circlearea

    print(rate) 


    print(label[i], end=" ")
    if rate >= 0.805:
        cv2.putText(realimg,  "Full Moon", (20 ,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        print("보름달")
        calculate(3,i)
    elif rate >0.3 and rate < 0.805:
        if weight[0] > weight[1]:
            cv2.putText(realimg,  "Waning Moon", (20 ,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            print("하현달")
            calculate(4,i)
        else:
            cv2.putText(realimg,  "Waxing Moon", (20 ,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            print("상현달")
            calculate(2,i)
    else:
        if weight[0] > weight[1]:
            cv2.putText(realimg,  "Dark Moon", (20 ,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            print("그믐달")
            calculate(5,i)
        else: 
            cv2.putText(realimg,  "Cresent Moon", (20 ,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            print("초승달")
            calculate(1,i)

    print()
    #cv2.imshow('img', realimg)
    #cv2.imshow('result', result)
    

    if cv2.waitKey(0) & 0xFF == 27:
        continue

tp = 0
sumval = 0

for i in range(5):
    tp += matrix[i][i]
    sumval += sum(matrix[i])

acc = tp / sumval

print(acc)

classlabel = ["Cresent Moon","Waxing Moon","Full Moon","Waning Moon","Dark Moon"]

df_cm = DataFrame(matrix, index=[i for i in classlabel], columns = [i for i in classlabel])
plt.figure(figsize = (8,7))
sns.heatmap(df_cm, annot=True)
plt.show()