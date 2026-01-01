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

def match_features_bf(desc1,desc2):
    matcher = cv2.BFMatcher(cv2.NORM_L2 , crossCheck=False)
    raw_matches = matcher.knnMatch(desc1,desc2,k=2)
    return raw_matches
def apply_ratio_test(raw_matches,ratio=0.75):
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m,n = pair
            if m.distance < ratio*n.distance :
                good_matches.append(m)
    return good_matches
def extract_matched_pts(kp1,kp2,matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1,pts2
def visualize_matches(img1,kp1,img2,kp2,matches,max_display =50):
    if len(matches) > max_display :
        display_matches = matches[:max_display]
    else:
        display_matches = matches
    output = cv2.drawMatches(img1,kp1,img2,kp2,display_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output