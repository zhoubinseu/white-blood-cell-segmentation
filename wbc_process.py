"""
对染色血细胞显微图像进行处理：
    1. 自动提取图像中的白细胞，并计数
    2. 统计图像中白细胞的平均核面积占比（白细胞核面积/白细胞总面积）

Processing microscopic images of stained blood cells:
    1. Automatically extract white blood cells from images and count
    2. Calculate the average nuclear area ratio of white blood cells in the image (white blood cell nuclear area / total white blood cell area)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

# load image in BGR format
image1 = cv2.imread("./images/WhiteBloodCell01.bmp", 1)
image2 = cv2.imread("./images/WhiteBloodCell02.bmp", 1)
image3 = cv2.imread("./images/WhiteBloodCell03.bmp", 1)
image4 = cv2.imread("./images/WhiteBloodCell04.bmp", 1)

# 显示图片直方图
def showHist(image):
    plt.hist(image.flatten(),bins = 256)
    plt.show()
# 增强对比度
def contrast(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a = 2
    output = image_gray*float(a)
    output[output>255] = 255
    output = np.round(output)
    output = output.astype(np.uint8)
    return output

# 去除面积较小的区域
def removeSmallRegion(image, size):
    _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    image, contours, hierarch = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < size:
            cv2.drawContours(image,[contours[i]],0,0,-1)
    return image

def process_img(image):

    # 预处理
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_s_channel = image_hsv[:,:,1]
    contrasted_image = contrast(image)

    cell_image = image_s_channel.copy()
    nucleus_image = contrasted_image.copy()

    # 图像分割
    # k-means 聚类
    # 分割细胞
    cell_data = cell_image.reshape((-1, 1))
    cell_data = np.float32(cell_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    cell_ret,cell_label,cell_center=cv2.kmeans(cell_data,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    cell_center = np.uint8(cell_center)
    cell_res = cell_center[cell_label.flatten()]
    cell = cell_res.reshape((cell_image.shape))
    max_v = np.max(cell)
    cell[cell==max_v] = 255
    cell[cell<max_v] = 0
    #分割细胞核
    nucleus_data = nucleus_image.reshape((-1, 1))
    nucleus_data = np.float32(nucleus_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    nucleus_ret,nucleus_label,nucleus_center=cv2.kmeans(nucleus_data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    nucleus_center = np.uint8(nucleus_center)
    nucleus_res = nucleus_center[nucleus_label.flatten()]
    nucleus = nucleus_res.reshape((nucleus_image.shape))
    min_v = np.min(nucleus)
    nucleus[nucleus>min_v] = 0
    nucleus[nucleus==min_v] = 255

    # 对分割结果进行处理
    kernel_cell = np.ones((5,5),np.uint8)
    processed_cell = removeSmallRegion(cell, 300)
    processed_cell = cv2.dilate(processed_cell,kernel_cell,iterations = 1)

    # 细胞核一定在细胞内，因此将细胞外部的像素去除
    temp_cell = processed_cell.copy()
    # 创建一个二值mask
    temp_cell[temp_cell>0] = 1
    processed_nucleus = temp_cell*nucleus
    kernel_nucleus = np.ones((2,2),np.uint8)
    processed_nucleus = cv2.dilate(processed_nucleus,kernel_nucleus,iterations = 1)

    # 计算白细胞个数和计算面积, 并且用矩形框标出细胞位置
    color = (255, 0, 0)
    _,cnts,hierarchy = cv2.findContours(processed_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_num = len(cnts)
    cell_area = 0
    for i in range(cell_num):
        cell_area += cv2.contourArea(cnts[i])
        cnt = cnts[i]
        x, y, w, h = cv2.boundingRect(cnt)
        image = cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    # 计算细胞核的面积
    _,cnts,hierarchy = cv2.findContours(processed_nucleus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nucleus_area = 0
    for i in range(len(cnts)):
        nucleus_area += cv2.contourArea(cnts[i])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, 'white cell number: '+str(cell_num), (0, 15), font, 0.5, color, 1)
    image = cv2.putText(image, 'nucleus_area/cell_area: '+str(round(nucleus_area/cell_area, 4)), (0, 35), font, 0.5, color, 1)

    print("white cell number: ", cell_num)
    print("nucleus_area/cell_area: ", round(nucleus_area/cell_area, 4))

    #显示最终的结果
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_img(image=image4)