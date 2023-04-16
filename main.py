import cv2 as cv
import mediapipe as mp
import time

class PoseEstimator():
    def __init__(self,mode=False, complexity=1, upBody= False, smooth=True, detectionCon= 0.5, trackingCon= 0.5):
        self.mode=mode
        self.complexity= complexity
        self.upBody=upBody
        self.smooth= smooth
        self.detectionCon=detectionCon
        self.trackingCon= trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.upBody,self.smooth, self.detectionCon,self.trackingCon)

    def findPose(self,image, draw= True):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return image

    def findPosition(self, image, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, land in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
                # print(id, land)
                cx, cy = int(land.x * w), int(land.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(image, (cx, cy), 3, (255, 0, 0), cv.FILLED)
        return lmList
#m
def main():
    cTime = 0
    pTime = 0
    cap = cv.VideoCapture(0)
    detector= PoseEstimator()

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


if __name__ == '__main__':
    main()