import cv2
path = "/home/nhan/Desktop/New Folder/HR/_2_jpg.rf.1862bf8d42707d677b75a37396adb1d1_out.jpg"
img = cv2.imread(path)
print(img.shape)
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()