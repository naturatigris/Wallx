import cv2
import numpy as np
from FindingBalloons import *

def detectHit(img,bboxs):
    img1= cropImage(img,0.1)
    # Process cropped images...
    cropped_images=splitBalloons(img1,bboxs)
    #showBalloons(cropped_images)

    s,b=findBallInBalloon(cropped_images)
    #showBalloons(s)

    return b

def splitBalloons(img, bboxs):
    cropped_images = []
    for b in bboxs:
        x1, y1, x2, y2 = b[0],b[1],b[2],b[3]
        cropped_images.append(img[y1:y2, x1:x2])
    return cropped_images

def showBalloons(imgBalloonList):
    for x,im in enumerate(imgBalloonList):
        cv2.imshow("Balloon", im)
    
def findBallInBalloon(imgBalloonList):
    imgb=[]
    b=False
    for img in imgBalloonList:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 5,0)
        max=10
        circles=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=max)
        print(type(circles).__name__)
        if type(circles).__name__!='NoneType':
            circles=np.uint16(np.around(circles))
            if len(circles)!=0:
                for c in circles[0,:]:
                    print(c[2])
                    cv2.circle(img,(c[0],c[1]),c[2],(255,0,255),2)
                    imgb.append(img)
                    #cv2.imshow("img",img)
                    b=True
            
    return imgb,b


if __name__ == "__main__":
    # Test code or additional logic if needed
    pass
