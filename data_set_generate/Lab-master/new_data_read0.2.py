import numpy as np
import cv2
import dlib
import os
import math

def get_file_number(path):
    file_num = 0
    dir_num = 0
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            file_num += 1
        if os.path.isdir(sub_path):
            dir_num += 1
    return file_num, dir_num

def read_img(path):
    pos = (280, 220)
    w = 400
    h = 400
    ROI = []
    file_num, dir_num = get_file_number(path)
    for i in range(file_num):
        print(i)
        filename = path + '/' + str(i) + '.jpg'
        temp_img = cv2.imread(filename)
        temp_img = temp_img[pos[1]+20:pos[1]+400-10, pos[0]+10:pos[0]+400-30]
        ROI.append(temp_img)
    return ROI

def read_img(path, start):
    pos = (240, 230)
    w = 400
    h = 400
    ROI = []
    file_num, dir_num = get_file_number(path)
    for i in range(file_num):
        print(i+start)
        filename = path + '/' + str(i+start) + '.jpg'
        temp_img = cv2.imread(filename)
        #temp_img = temp_img[pos[1]+20:pos[1]+400-20, pos[0]+20:pos[0]+400-20]
        ROI.append(temp_img)
    return ROI

def viz_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

path = 'D:/blow/blow.mp4'
save_path = 'D:/blow/img'
save_path1 = 'D:/blow/flow/'
frame_list = read_img(save_path, 0)
for i in range(len(frame_list)-1):
    print('帧数:' + str(i))
    pre_gray = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame_list[i + 1], cv2.COLOR_BGR2GRAY)
    temp_flow = cv2.calcOpticalFlowFarneback(pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    bgr = viz_flow(temp_flow)
    cv2.imwrite(save_path1 + str(i) + '.jpg', bgr)
    print('写入' + save_path1 + str(i) + '.jpg')


'''
for i in range(total_frame):
    print('帧数:' + str(i))
    ret, temp_frame = cap.read()
    save = save_path + '/' + str(i) + '.jpg'
    cv2.imwrite(save, temp_frame)
    if i > 105:
        cv2.imshow('img', temp_frame)
        cv2.waitKey(300)
    if i > 120:
        break
print(total_frame)
'''