import cv2
import numpy as np

def get_canvas_size(shape1,shape2,H):
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    corners1 = np.float32([[0, 0],[0, h1],[w1, h1],[w1, 0]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    corners2 = np.float32([[0, 0],[0, h2],[w2, h2],[w2, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners1_transformed, corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    offset = (-x_min, -y_min)
    return (canvas_height,canvas_width),offset

def warp_image(image,H,canvas_shape,offset):
    translation_matrix = np.array([[1, 0, offset[0]],[0, 1, offset[1]],[0, 0, 1]], dtype=np.float64)
    H_translated = translation_matrix @ H
    warped = cv2.warpPerspective(image,H_translated,(canvas_shape[1],canvas_shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return warped

def create_mask(image):
    if len(image.shape) == 3:
        mask = (image.sum(axis=2) > 0).astype(np.uint8) * 255
    else:
        mask = (image > 0).astype(np.uint8) * 255
    return mask

def place_image_on_canvas(image,canvas_shape,offset):
    h, w = image.shape[:2]
    canvas_h, canvas_w = canvas_shape
    if len(image.shape) == 3:
        canvas = np.zeros((canvas_h,canvas_w, 3),dtype=image.dtype)
    else:
        canvas = np.zeros((canvas_h,canvas_w),dtype=image.dtype)
    mask = np.zeros((canvas_h,canvas_w),dtype=np.uint8)
    x_start = int(offset[0])
    y_start = int(offset[1])
    x_end = x_start + w
    y_end = y_start + h
    x_start_clip = max(0, x_start)
    y_start_clip = max(0, y_start)
    x_end_clip = min(canvas_w, x_end)
    y_end_clip = min(canvas_h, y_end)
    src_x_start = x_start_clip - x_start
    src_y_start = y_start_clip - y_start
    src_x_end = src_x_start + (x_end_clip - x_start_clip)
    src_y_end = src_y_start + (y_end_clip - y_start_clip)
    canvas[y_start_clip:y_end_clip, x_start_clip:x_end_clip] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end]
    mask[y_start_clip:y_end_clip, x_start_clip:x_end_clip]=255
    return canvas, mask