import cv2
import numpy as np
from .features import (
    detect_keypoints_sift,
    detect_keypoints_orb,
    match_features_bf,
    apply_ratio_test,
    extract_matched_pts
)
from .geometry import estimate_homography
from .warping import get_canvas_size, warp_image, create_mask, place_image_on_canvas
from .blending import feather_blend

class PanoramaStitcher:
    def __init__(self, feature_type='sift', ratio_threshold=0.75, ransac_threshold=4.0):
        self.feature_type = feature_type
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
    def stitch_pair(self,img1,img2):
        if self.feature_type == 'sift':
            kp1, desc1 = detect_keypoints_sift(img1)
            kp2, desc2 = detect_keypoints_sift(img2)
        else:
            kp1, desc1 = detect_keypoints_orb(img1)
            kp2, desc2 = detect_keypoints_orb(img2)
        print(f"Image 1: {len(kp1)} keypoints")
        print(f"Image 2: {len(kp2)} keypoints")
        raw_matches = match_features_bf(desc1, desc2)
        good_matches = apply_ratio_test(raw_matches, ratio=self.ratio_threshold)
        print(f"    Good matches: {len(good_matches)}")
        if len(good_matches) < 4:
            print("error")
            return None
        pts1, pts2 = extract_matched_pts(kp1, kp2, good_matches)
        print("estimating homography...")
        H, mask = estimate_homography(pts1, pts2, ransac_thresh=self.ransac_threshold)
        if H is None:
            print("error")
            return None
        inliers = np.sum(mask)
        inlier_ratio = 100 * inliers / len(mask)
        print(f"Inliers: {inliers} / {len(mask)} ({inlier_ratio:.1f}%)")
        print("canvas size...")
        canvas_shape, offset = get_canvas_size(img1.shape, img2.shape, H)
        img1_warped = warp_image(img1, H, canvas_shape, offset)
        img2_canvas, mask2 = place_image_on_canvas(img2, canvas_shape, offset)
        mask1 = create_mask(img1_warped)
        result = feather_blend(img1_warped, mask1, img2_canvas, mask2)
        return result
    
    def stitch_multiple(self, images):
        if len(images) == 2:
            return self.stitch_pair(images[0], images[1])
        center_idx = len(images) // 2
        print(f"total images: {len(images)}")
        result_right = images[center_idx].copy()
        for i in range(center_idx + 1, len(images)):
            result_right = self.stitch_pair(result_right, images[i])
            if result_right is None:
                return None
        result_left = images[center_idx].copy()
        for i in range(center_idx - 1, -1, -1):
            result_left = self.stitch_pair(images[i], result_left)
            if result_left is None:
                return None
        final_result = self.stitch_pair(result_left, result_right)
        return final_result