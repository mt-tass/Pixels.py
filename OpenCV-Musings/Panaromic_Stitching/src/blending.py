import cv2
import numpy as np

def feather_blend(img1,mask1,img2,mask2):
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
    dist1_norm = dist1 / (dist1.max() + 1e-10)
    dist2_norm = dist2 / (dist2.max() + 1e-10)
    total_dist = dist1_norm + dist2_norm
    total_dist[total_dist == 0] = 1
    weight1 = dist1_norm / total_dist
    weight2 = dist2_norm / total_dist
    if len(img1.shape) == 3:
        weight1 = weight1[:, :, np.newaxis]
        weight2 = weight2[:, :, np.newaxis]
    blended = (img1.astype(np.float32) * weight1 + 
               img2.astype(np.float32) * weight2)
    blended = blended.astype(img1.dtype)
    return blended


def simple_blend(img1,mask1,img2,mask2):
    mask1_float = mask1.astype(float) / 255
    mask2_float = mask2.astype(float) / 255
    overlap = (mask1_float > 0) & (mask2_float > 0)
    only_1 = (mask1_float > 0) & (mask2_float == 0)
    only_2 = (mask1_float == 0) & (mask2_float > 0)
    blended = np.zeros_like(img1)
    blended[only_1] = img1[only_1]
    blended[only_2] = img2[only_2]
    if np.any(overlap):
        blended[overlap] = ((img1[overlap].astype(float) + 
                            img2[overlap].astype(float)) / 2).astype(img1.dtype)
    return blended