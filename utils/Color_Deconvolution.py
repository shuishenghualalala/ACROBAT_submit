import sys
import os
import cv2

import histomicstk as htk

import numpy as np
import scipy as sp
import cv2

import matplotlib.pyplot as plt

def deconv_macenko_pca(img):
    """generate images deconvoluted with pca, which includes H, E and DAB."""
    # inputImageFile = (img_path)  # H&E.png
    imInput = img
    I_0 = None
    im_sda = htk.preprocessing.color_conversion.rgb_to_sda(imInput, I_0)

    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imInput, I_0)

    # Perform color deconvolution
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0).Stains

    # print('Estimated stain colors (rows):', w_est.T[:3])

    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    stains = ['hematoxylin',  # nuclei stain
              'eosin',  # cytoplasm stain
              'dab']  # set to null if input contains only two stains
    # Unlike SNMF, we're not guaranteed the order of the different stains.
    # find_stain_index guesses which one we want
    channels = [htk.preprocessing.color_deconvolution.find_stain_index(
        stain_color_map[stain], w_est) for stain in stains]
    channels = np.array(channels)
    deconv_result = deconv_result[..., channels]

    #     return deconv_result, w_est
    return imInput, deconv_result


def deconvolution_morph(k, imDeconvolved, thr):
    '''
    H:0, E:1, DAB:2
    '''
    #     if type(imDeconvolved) is 'numpy.ndarray':
    #         img_sep = imDeconvolved[:, :, k]  # pca
    #     else:
    #         img_sep = imDeconvolved.Stains[:, :, k]  #单通道灰度图
    img_sep = imDeconvolved[:, :, k]  # pca
    mask_sep = np.ones(img_sep.shape, np.uint8) * (img_sep[:, :] < thr)
    kernel = np.ones((5, 5), np.uint8)
    # 先膨胀再腐蚀 填充小洞
    mask_sep = cv2.morphologyEx(mask_sep, cv2.MORPH_CLOSE, kernel)
    # 先腐蚀后膨胀 去噪
    mask_sep = cv2.morphologyEx(mask_sep, cv2.MORPH_OPEN, kernel)
    #     plt.figure(figsize=(5,5))
    #     plt.subplot(1,2,1)
    #     plt.imshow(img_sep)
    #     plt.subplot(1,2,2)
    #     plt.imshow(mask_sep)
    return img_sep, mask_sep


def deconvolution_fusion(imDeconvolved, thr):
    H_sep, H_mask = deconvolution_morph(0, imDeconvolved, thr)
    E_sep, E_mask = deconvolution_morph(1, imDeconvolved, thr)
    DAB_sep, DAB_mask = deconvolution_morph(2, imDeconvolved, thr)
    fusion_mask = H_mask + E_mask + DAB_mask
    fusion_mask = np.ones(fusion_mask.shape, np.uint8) * (fusion_mask[:, :] > 0)
    #     plt.figure(figsize=(5,5))
    #     plt.imshow(fusion_mask)
    return fusion_mask


def deconvolution_mask(imInput, mask):
    mask = cv2.dilate(mask, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=5)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    # 先膨胀再腐蚀 填充小洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 先腐蚀后膨胀 去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    max_contour=0
    return mask, max_contour

def get_HE_mask(img):
    imInput, imDeconvolved = deconv_macenko_pca(img)
    fusion_mask = deconvolution_fusion(imDeconvolved, 200)
    img_mask, max_contour = deconvolution_mask(imInput, fusion_mask)
    return img_mask

if __name__ == '__main__':
    dir = ''
    save_dir = ''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for p  in os.listdir(dir):
        pair_path = os.path.join(dir,p)
        for i in os.listdir(pair_path):
            if '_HE.' in i:
                img_path = os.path.join(pair_path,i)
                img = cv2.imread(img_path)
                # img = __gamma_adjust(img,1)
                imInput, imDeconvolved = deconv_macenko_pca(img)
                # print(imDeconvolved.shape)
                fusion_mask = deconvolution_fusion(imDeconvolved, 200) # 230 200
                cv2.imwrite(os.path.join(save_dir,i),fusion_mask)

                img_mask, max_contour = deconvolution_mask(imInput, fusion_mask)
                plt.imshow(img)
                plt.show()
                plt.imshow(img_mask)
                plt.show()


