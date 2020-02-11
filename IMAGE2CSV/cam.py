# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:35:04 2019

@author: KRISHNAN
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        img = mpimg.imread(img_name)  
        gray = rgb2gray(img)    
        gray = cv2.resize(gray, (28,28), interpolation=cv2.INTER_CUBIC)
        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.show()
        g = np.array(gray)
        g4 = g.flatten()

cam.release()

cv2.destroyAllWindows()
img = mpimg.imread("RdEpj.png") 
gray = rgb2gray(img)    
gray = cv2.resize(gray, (28,28), interpolation=cv2.INTER_CUBIC)
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()
g = np.array(gray)
g4 = g.flatten()
data1 = pd.DataFrame(g1)
data2 = pd.DataFrame(g2)
data3 = pd.DataFrame(g3)
data4 = pd.DataFrame(g4)
data1.to_csv("data3.csv")  
data2.to_csv("data0.csv")
data3.to_csv("data8.csv")
data4.to_csv("data6.csv")