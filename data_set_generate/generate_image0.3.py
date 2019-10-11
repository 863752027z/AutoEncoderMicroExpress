import dlib
import cv2
import os

detector_face_cut = cv2.CascadeClassifier('F:/data/haarcascade_frontalface_default.xml')


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


def face_cut(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector_face_cut.detectMultiScale(gray, 1.1, 5)
    print('人脸个数', len(faces))
    c = 1
    y = 1.1
    while len(faces) == 0 or faces[0][2]*faces[0][3] < 300*300:
        print('未检测到人脸')
        if c % 2 == 1:
            x = 5
        else:
            x = 3
        faces = detector_face_cut.detectMultiScale(gray, y, x)
        c += 1
        if c > 20:
            y = 1.1
        if c > 40:
            y = 1.2
        if c > 60:
            y = 1.3
        if c > 70:
            y = 1.4
        if c > 80:
            y = 1.5
    print('检测到人脸')

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    temp_img = img[y:y + h - 1, x:x + w - 1]
    return temp_img


def croppy(path1, path2, gen_face):
    print('处理文件', path1)
    file_list1 = []
    file_list2 = []
    for root, dirs, files in os.walk(path1):
        for i in range(len(files)):
            temp_path = path1 + '/' + files[i]
            file_list1.append(temp_path)
            temp_path = path2 + '/' + files[i]
            file_list2.append(temp_path)
    first_img = cv2.imread(file_list1[0])
    temp_img = face_cut(first_img)
    path = gen_face + file_list2[0][21:].replace('/', '_')
    print('path', path)
    cv2.imwrite(path, temp_img)
    #cv2.imshow('img', temp_img)
    #cv2.waitKey(1000)
    '''
    for i in range(len(file_list1)):
        print('读', file_list1[i])
        temp_img = cv2.imread(file_list1[i])
        
        print(temp_img.shape)
        temp_img = cv2.resize(temp_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    '''
    ''' 
        cv2.imshow('img', temp_img)
        k = cv2.waitKey(0)&0xff
        if k ==27:
            cv2.destroyAllWindows()
    '''
        #print('写入' + file_list2[i])
        #cv2.imwrite(file_list2[i], temp_img)


gen_face = 'F:/SAMM_FACE_CUT/gen_face/'
base_path1 = 'D:/SAMM/'
base_path2 = 'F:/SAMM_FACE_CUT/SAMM/'
#remain_dir = ['008', '009', '010', '011', '012', '013', '014', '015']
remain_dir = ['006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
              '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035',
              '036', '037']
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
            croppy(path1[j], path2[j], gen_face)

