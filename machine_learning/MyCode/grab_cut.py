import numpy as np
import cv2
import matplotlib.pyplot as plt


file_path = 'F:/suolong.jpg'
  # 读入图片
img = cv2.imread(file_path)
  # 创建一个和加载图像一样形状的 填充为0的掩膜
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
 # 定义一个矩形
rect = (100, 50, 421, 378)

i = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
plt.imshow(i)
plt.show()