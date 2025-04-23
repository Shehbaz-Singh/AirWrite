import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)  # Ensure it's a float
        self.trackCon = float(trackCon)            # Ensure it's a float

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []  # Initialize lmList

        # You can define tipIds based on MediaPipe landmarks
        self.tipIds = [4, 8, 12, 16, 20]  # Indices for tips of the fingers

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Process the image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []  # Reset lmList at the beginning
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList
    
    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def isPalmOpen(self):
        # This function checks if the palm is open based on finger positions
        if len(self.lmList) == 21:  # Check if landmarks are detected
            # Check if the tips of the fingers are above the base of the fingers
            fingers_open = all(self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 1][2] for i in range(1, 5))
            thumb_open = self.lmList[self.tipIds[0]][2] < self.lmList[self.tipIds[0] - 1][2]  # Check the thumb
            return fingers_open and thumb_open
        return False
