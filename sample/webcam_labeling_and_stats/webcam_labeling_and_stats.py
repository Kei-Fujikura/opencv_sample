import datetime
import numpy as np
import random
import copy
import cv2

percent_w = (20,60) # min,max (%)
percent_h = (20,60) # min,max (%)
base_img = "base.png"

class CVCapture:

    def __init__(self):
        self.cap = {}
        self.lastCaptured = None
    
    def Init(self,cameraNo="0",width=640,height=480,fps=10,exposure=-4,gain=0):
        cno = str(cameraNo)
        self.cap[cno] = cv2.VideoCapture(int(cno))
        c = self.cap[cno]

        c.set(cv2.CAP_PROP_EXPOSURE,exposure)
        c.set(cv2.CAP_PROP_GAIN,gain)
        c.set(6,cv2.VideoWriter_fourcc(*'MJPG'))
        c.set(cv2.CAP_PROP_FPS,fps)
        c.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    
    def Exit(self):
        for no in self.cap:
            self.cap[no].release()
        cv2.destroyAllWindows()

    def Capture(self,no="0",imgname="",color=None):
        r, img = self.cap[no].read()
        if color is not None:
            img = cv2.cvtColor(img,color)
        self.lastCaptured = img
        if imgname == "":
            imgname = ".\\tmp.png"
        cv2.imwrite(imgname,img)
        return r,img
    
    def Save(self,filename,image=None):
        if (image is None):
            if (self.lastCaptured.all() is not None):
                image = self.lastCaptured
        
        if image is not None:
            cv2.imwrite(filename,image)
        

# 撮像
colors = []
frame = None
cap = CVCapture()
cap.Init()

imbase = cv2.imread(base_img)

try:
    while True:
        # ラベリング
        r,c = cap.Capture()
        height, width, channels = c.shape[:3]
        dst = copy.copy(c)
        dst = cv2.absdiff(dst,imbase)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, th2 = cv2.threshold(gray, 20, 100, cv2.THRESH_BINARY |cv2.THRESH_OTSU) 
        cv2.imwrite("gray.png",th2)
        nLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(th2)
        print("labels: " + str(nLabels))

        if len(colors) < nLabels:
            for i in range(len(colors), nLabels + 1):
                rgb = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                colors.append(np.array(rgb))

        for y in range(0, height):
            for x in range(0, width):
                if labelImage[y, x] > 0:
                    dst[y, x] = colors[labelImage[y, x]]
                else:
                    dst[y, x] = [0, 0, 0]

        for cnt,x in enumerate(range(nLabels)):
            s = stats[cnt]
            x,y,w,h,area = s
            hlim = dst.shape[0]
            wlim = dst.shape[1]
            if (wlim*percent_w[0]/100.0 < w) and (w < wlim*percent_w[1]/1.2) and (hlim*percent_h[0]/100 < h) and (h < hlim*percent_h[1]/1.2):
                t = (int(centroids[cnt][0]),int(centroids[cnt][1]))
                cv2.circle(c, t, 3,(0,0,255),-1)
                cv2.rectangle(c, (x+3,y+3), (x+w-3,y+h-3), (0,0,255),3)
                print("%2d:%s:%s"%(cnt,t,s))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('frame1',c)
finally:
    cap.Exit()

