import cv2
import glob
from src.stitcher import PanoramaStitcher

def main():
    image_paths = sorted(glob.glob("data/raw/img*.jpg"))
    if len(image_paths) == 0:
        print("image were not found , format of naming ('img1.jpg' for first image)")
        return
    print(f"found {len(image_paths)} images")
    images = []
    for i,path in enumerate(image_paths):
        image = cv2.imread(path)
        if image is None :
            print(f"failed to load image {i+1}")
        else :
            images.append(image)
    if len(images) < 2:
        print("atleast 2 images needed , load more !")
        return
    stitcher = PanoramaStitcher(feature_type='sift',ransac_threshold=4.0,ratio_threshold=0.75)
    result = stitcher.stitch_multiple(images)
    if result is None:
        print("panaromic stitching failed , try different threshold values")
    else:
        output_path = "data/results/final_panaroma.jpg"
        cv2.imwrite(output_path, result)
        print("panaromic image saved in results")
        cv2.imshow("panaroma",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
    