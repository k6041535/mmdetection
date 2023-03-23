import shutil
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ImageAllPath(pathFile):
    ImagePath_list = os.listdir(pathFile)
    print(len(ImagePath_list))
    return ImagePath_list
 
#CT
'''
def ImageFor():
    pathFile = "C:/Users/user/Desktop/train_data/HY/CXR"
    save_path = 'C:/Users/user/Desktop/train_data/HY_repeat/CXR'
    ImagePath_list = ImageAllPath(pathFile)
    n = 1
    num = 0
    for index in range(len(ImagePath_list)-1):
 
        path =pathFile + '/' + ImagePath_list[index]
        if os.path.exists(path):
        
            img1 = cv2.imread(path)
            #print('img1 = ',path)
            plt.imshow(img1)
            plt.show()
            path = pathFile+ '/' + ImagePath_list[index+1]
            img2 = cv2.imread(path)
            #print('img2 = ',path)
            
            img1_gray1 = img1[448:589,2575:2733]
            img2_gray2 = img2[448:589,2575:2733]
            ss = ssim(img1_gray1,img2_gray2)
            #plt.subplot(121),plt.imshow(img1_gray1)
            #plt.subplot(122),plt.imshow(img2_gray2)
            #plt.show()
            #print('ss = ',ss)

            img1_crop1 = img1[474:620,4125:4269]
            img2_crop2 = img2[474:620,4125:4269]
            ss2 = ssim(img1_crop1,img2_crop2)
            #plt.subplot(121),plt.imshow(img1_crop1)
            #plt.subplot(122),plt.imshow(img2_crop2)
            #plt.show()
            #print('ss2 = ',ss2)
            #print('n = ',n)
            n = n + 1 

            num = (ss + ss2) / 2 
            if num > 0.98:
                # 删除图片
                shutil.move(path, save_path)
                print("正在移除重复照片：", path)
 
''' 
def ImageFor():
    pathFile = "C:/Users/user/Desktop/train_data/CCK"
    save_path = 'C:/Users/user/Desktop/train_data/CCK_repeat'
    ImagePath_list = ImageAllPath(pathFile)
    n = 1
    num = 0
    for index in range(len(ImagePath_list)-1):
 
        path =pathFile + '/' + ImagePath_list[index]
        if os.path.exists(path):
        
            img1 = cv2.imread(path)
            #print('img1 = ',path)
            #plt.imshow(img1)
            #plt.show()
            path = pathFile+ '/' + ImagePath_list[index+1]
            img2 = cv2.imread(path)
            #print('img2 = ',path)
            
            img1_gray1 = img1[201:1975,1208:2704]
            img2_gray2 = img2[201:1975,1208:2704]
            ss = ssim(img1_gray1,img2_gray2)
            #plt.subplot(121),plt.imshow(img1_gray1)
            #plt.subplot(122),plt.imshow(img2_gray2)
            #plt.show()
            #print('ss = ',ss)

            img1_crop1 = img1[206:1958,2756:4142]
            img2_crop2 = img2[206:1958,2756:4142]
            ss2 = ssim(img1_crop1,img2_crop2)
            #plt.subplot(121),plt.imshow(img1_crop1)
            #plt.subplot(122),plt.imshow(img2_crop2)
            #plt.show()
            #print('ss2 = ',ss2)

            n = n + 1 

            num = (ss + ss2) / 2 
            if num > 0.98:
                # 删除图片
                shutil.move(path, save_path)
                print("正在移除重复照片：", path)
if __name__ == '__main__':
    
    ImageFor()
