import cv2
import numpy as np

def detect_keypoints_sift(image , n_feats =5000):
    if len(image.shape) == 3:
        gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else :
        gray = image
    sift = cv2.SIFT_create(nfeatures = n_feats)
    keypoints , descriptors = sift.detectAndCompute(gray,None)
    return keypoints,descriptors

def detect_keypoints_orb(image , n_feats=5000):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    orb = cv2.ORB_create(nfeatures = n_feats)
    keypoints , descriptors = orb.detectAndCompute(gray,None)
    return keypoints,descriptors

def visualize_keypoints(image):
    kps , descs = detect_keypoints_sift(image)
    image_kps = cv2.drawKeypoints(image,kps,None,color=(0,255,0) , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_kps