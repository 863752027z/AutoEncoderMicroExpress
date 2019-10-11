import cv2
import os
import numpy as np


def get_path(base_path1, base_path2, remain_dir):
    path1 = base_path1 + remain_dir + '/'
    path2 = base_path2 + remain_dir + '/'
    path_list1 = []
    path_list2 = []
    for root, dirs, files in os.walk(path1):
        for i in range(len(dirs)):
            temp_path = path1 + dirs[i]
            path_list1.append(temp_path)
            temp_path = path2 + dirs[i]
            path_list2.append(temp_path)
        break
    return path_list1, path_list2


def croppy(path1, path2):
    print('处理文件', path1)
    file_list1 = []
    file_list2 = []
    for root, dirs, files in os.walk(path1):
        for i in range(len(files)):
            temp_path = path1 + '/' + files[i]
            file_list1.append(temp_path)
            temp_path = path2 + '/' + files[i]
            file_list2.append(temp_path)
    for i in range(len(file_list1)):
        print('读', file_list1[i])
        temp_img = cv2.imread(file_list1[i])
        temp_img = temp_img[250:620, 280:-340, :]
        '''
        print(temp_img.shape)
        temp_img = cv2.resize(temp_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('img', temp_img)
        k = cv2.waitKey(30)&0xff
        if k ==27:
            cv2.destroyAllWindows()
        '''
        print('写入' + file_list2[i])
        cv2.imwrite(file_list2[i], temp_img)


base_path1 = 'D:/SAMM/'
base_path2 = 'F:/GenDataSet/SAMM/'
#remain_dir = ['006', '007', '008', '009', '010', '011', '012', '013', '014', '015']
remain_dir = ['006']
for i in range(len(remain_dir)):
    dir_path = base_path2 + remain_dir[i]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('创建' + dir_path)

    path1, path2 = get_path(base_path1, base_path2, remain_dir[i])
    temp_path = base_path1 + remain_dir[i]
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        print('创建' + temp_path)
    for j in range(len(path1)):
        if not os.path.exists(path2[j]):
            os.mkdir(path2[j])
            print('创建' + path2[j])
            croppy(path1[j], path2[j])

