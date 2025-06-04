import numpy as np
import cv2
import mediapipe as mp
import time
import pandas as pd
from joblib import load

mp_drawing = mp.solutions.drawing_utils
hand = mp.solutions.hands
mp_hands = mp.solutions.hands.Hands()

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
qda = load('qda_model.pkl')
minmax = load('minmax.joblib')

cap = cv2.VideoCapture(0)
pTime = 0
cap.set(3, 1280)
cap.set(4, 720)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands.process(image)
    image.flags.writeable = True
    rows, cols, _ = image.shape

    swap = []
    all_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, hand.HAND_CONNECTIONS)

            for landmark in hand_landmarks.landmark:
                y_landmark = int(landmark.y * rows)
                swap.append(y_landmark)

        if len(swap) == 21:  
            all_landmarks.append(swap)

        if all_landmarks: 
            all_landmarks = np.array(all_landmarks)

            all_landmarks = minmax.transform(all_landmarks)

            y_pred = qda.predict(all_landmarks)
            cv2.putText(image, str(classes[y_pred[0]]), (250, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    result.write(image)
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

result.release()
cap.release()
cv2.destroyAllWindows()
