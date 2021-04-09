import cv2
import copy
import numpy as np
from keras.models import load_model

model = load_model("best_val_loss.hdf5")

def Predict(Word_Characters):
    Predictions = []

    for Characters in Word_Characters:
        Prediction = []
        for img in Characters:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
            div = gray / morph
            gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)

            thresh = cv2.resize(thresh, (32,32), interpolation = cv2.INTER_AREA)

            x = np.array([thresh]).reshape(-1, 32, 32, 1) / 255.0
            Prediction.append(np.argmax(model.predict(x)))

        Predictions.append(copy.deepcopy(Prediction))
    
    return Predictions
