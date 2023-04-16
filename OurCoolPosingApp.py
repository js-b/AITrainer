import cv2 as cv
import mediapipe as mp
import time
import PoseEstimationModule as pm


cTime = 0
pTime = 0
cap = cv.VideoCapture('PoseVideos/1.mp4')
detector= pm.PoseEstimator()

while True:
    success, image = cap.read()
    image = detector.findPose(image)
    lmList = detector.findPosition(image)
    if len(lmList)!=0:
        print(lmList[14])
        #Tracking the elbow
    if len(lmList)!=0:
        cv.circle(image, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(image, str(int(fps)), (70, 100), cv.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 100), 3)
    cv.imshow("Image", image)
    cv.waitKey(1)


