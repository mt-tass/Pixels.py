import cv2 as cv
import numpy as np
#video block
class getVideo:
    def __init__(self,camera_idx=0):
        self.capture = cv.VideoCapture(camera_idx)
        if not self.capture.isOpened():
            return None
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH,720)
    def getFrame(self):
        success , frame = self.capture.read()
        if not success :
            return None
        else:
            return frame
    def release(self):
        self.capture.release()

#light capture block
class lightDetector:
    def __init__(self,min_thresh=225,min_area=10,max_area=10000):
        self.min_val = min_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.lower_bound = np.array([0,0,self.min_val],dtype="uint8")
        self.upper_bound = np.array([180,255,255],dtype="uint8")
    def get_bp(self,frame):
        if frame is None:
            return None
        hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame,self.lower_bound,self.upper_bound)
        matrix = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        mask = cv.morphologyEx(mask,cv.MORPH_OPEN,matrix)
        mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,matrix)

        contours,heirarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        largest_area = 0
        for contour in contours:
            area = cv.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                if area > largest_area:
                    largest_area = area
                    largest_contour = contour
        if largest_contour is not None:
            Moments = cv.moments(largest_contour)
            if Moments["m00"] != 0:
                cx = int(Moments["m10"]/Moments["m00"])
                cy = int(Moments["m01"]/Moments["m00"])
                return (cx,cy)
        return None
#image overlay block
class imageOverlay:
    def put_image(self,background,foreground,position,scale=1.0):
        fgh,fgw = foreground.shape[:2]
        if scale != 1.0:
            fgw,fgh = int(fgw*scale) , int(fgh*scale)
            foreground = cv.resize(foreground,(fgw,fgh),interpolation=cv.INTER_AREA)
        cx,cy = position
        x1,y1 = cx-(fgw//2) , cy-(fgh//2)
        x2,y2 = x1+fgw,y1+fgh
        bgh,bgw = background.shape[:2]
        bx1,by1 = max(0,x1),max(0,y1)
        bx2,by2 = min(bgw,x2),min(bgh,y2)
        fx1,fy1 = max(0,-x1),max(0,-y1)
        fx2,fy2 = fgw-max(0,x2-bgw),fgh-max(0,y2-bgh)

        if bx1 >= bx2 or by1>=by2:
            return background
        bg_roi = background[by1:by2,bx1:bx2]
        fg_roi = foreground[fy1:fy2,fx1:fx2]
        if fg_roi.shape[2] == 4:
            rgb = fg_roi[:,:,:3].astype(float)
            alpha = fg_roi[:,:,3]/255.0
            alpha = np.stack([alpha]*3 , axis=2)
            blend = alpha*fg_roi + (1-alpha)*bg_roi.astype(float)
            result = background.copy()
            result[by1:by2,bx1:bx2] = blend.astype('uint8') 
            return result
        else:
            result = background.copy()
            result[by1:by2,bx1:bx2] = fg_roi
            return result

cam_idx = 0
hsv_thresh = 250
min_area = 100
max_area = 10000
scale = 0.7
cam = getVideo(cam_idx)
detect = lightDetector(hsv_thresh,min_area,max_area)
overlay = imageOverlay()
patronous = cv.imread("data/unicorn.png")
if patronous is not None:
    print("Your Patronous in ready!!")
while True:
    frame = cam.getFrame()
    if frame is None :
        break
    pos = detect.get_bp(frame)
    if pos is not None:
        frame = overlay.put_image(background=frame,foreground=patronous,position=pos,scale=scale)
    cv.imshow("Expecto Patronous",frame)
    if cv.waitKey(1) == ord('q'):
        break
cam.release()
cv.destroyAllWindows()