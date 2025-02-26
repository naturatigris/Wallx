import cv2
import numpy as np

def findBalloons(img):
    img = cropImage(img,0.1)
    img = preprocess(img)
    return img

def cropImage(img,cropval):
    h,w,c= img.shape
    img = img[int(cropval*h):h, 0:w]
    return img

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 5,0)
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 50     # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    img = cv2.Canny(adjusted, 50, 100)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel,iterations=1)
    #cv2.imshow("preprocees",img)
    return img

def findContours(img):
    bboxs=[]
    h, w = img.shape[:2]  # Get height and width of the image
    imgContours = np.zeros((h, w, 3), np.uint8)  # Initialize a blank RGB image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 2500:  # Filter contours based on area
            cv2.drawContours(imgContours, [cnt], -1, (255, 0, 255), 2)  # Draw contour
            (x, y, w, h) = cv2.boundingRect(cnt)
            bboxs.append([x,y,x+w,y+h])
            cv2.rectangle(imgContours, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            #cv2.imshow("s",imgContours)
    return imgContours,bboxs


if __name__ == "__main__":
    # Test code or additional logic if needed
    pass
