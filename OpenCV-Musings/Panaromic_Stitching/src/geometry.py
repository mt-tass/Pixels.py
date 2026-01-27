import cv2
import numpy as np

def estimate_homography(pts0 , pts1 , ransac_thresh = 4.0 , confidence = 0.995):
    if len(pts0) < 4:
        return None , None
    H , mask = cv2.findHomography(pts0,pts1,method = cv2.RANSAC , ransacReprojThreshold=ransac_thresh , confidence=confidence , maxIters=2000)
    return H , mask

def compute_error_reprojection(pts0,pts1,H):
    pts0_h = np.array([pts0[0],pts0[1],1.0],dtype = np.float64)
    projection = H @ pts0_h
    if abs(projection[2]) < 1e-8:
        return float('inf')
    projection_xy = projection[:2]/projection[2]
    error = np.sqrt((pts1[0] - projection_xy[0])**2 + (pts1[1] - projection_xy[1])**2)
    return error

def get_inliers(matches , mask):
    mask = mask.ravel().astype(bool)
    inlier_matches = []
    for i in range(len(matches)):
        if mask[i]:
            inlier_matches.append(matches[i])
    return inlier_matches
