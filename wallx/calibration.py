import cv2
import pickle
import numpy as np
def WarpImage(img, points):
    if len(points) != 4:
        print("Error: Four points required for perspective transform")
        return None

    # Define the four corners of the original image
    pts_src = np.array(points, dtype=np.float32)

    # Define the four corners of the desired output
    width = max(np.linalg.norm(pts_src[0] - pts_src[1]), np.linalg.norm(pts_src[2] - pts_src[3]))
    height = max(np.linalg.norm(pts_src[0] - pts_src[3]), np.linalg.norm(pts_src[1] - pts_src[2]))
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the perspective transformation
    warped_img = cv2.warpPerspective(img, matrix, (int(width), int(height)))

    return warped_img


def calibration():
    circles = []

    def mousePoints(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            circles.append((x, y))
            print(f"Circle added: {x}, {y}")

    droidcam_url = "http://192.168.142.16:4747/video"

    cap = cv2.VideoCapture(droidcam_url)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        for x in range(4):
            if len(circles) > x:
                cv2.circle(img, (circles[x][0], circles[x][1]), 10, (0, 255, 0), cv2.FILLED, 2)
        cv2.imshow("Img", img)
        cv2.setMouseCallback("Img", mousePoints)
        
        if len(circles) == 4:
            print("4 circles selected.")
            with open("calibration_data.pkl", 'wb') as file:
                pickle.dump(circles, file)
            break
            
        if cv2.waitKey(1) == 27:  # Press Esc key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibration()
