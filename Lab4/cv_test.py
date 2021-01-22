import cv2
import numpy as np
from pca import pca,calc_psnr

imgs = np.empty((10, 250, 250))
new_imgs = np.empty((10, 250, 250))
for i in range(10):
    path = 'img/George_W_Bush_00' + str(i + 1).zfill(2) + '.jpg'
    img = cv2.imread(path, 0)
    # img = cv2.resize(img, (80, 80))
    cv2.imshow(str(i), img)
    # img = img.flatten().astype(np.float)
    img = img.astype(np.float)
    imgs[i] = img
    new_imgs[i], pc, mean = pca(img, 80)

new_imgs = new_imgs.astype(np.uint8)
for i in range(10):
    new_img = new_imgs[i]
    print('PSNR of picture ' +  str(i + 1) + ': ' + str(calc_psnr(imgs[i].flatten(), new_imgs[i].flatten())))
    # new_img = cv2.resize(new_img, (250, 250))
    cv2.imshow('new' + str(i), new_img)
    cv2.imwrite('img/all_00' + str(i + 1).zfill(2) + '.jpg', new_img)

cv2.waitKey(0)

