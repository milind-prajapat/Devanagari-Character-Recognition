import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical

model = load_model("best_val_loss.hdf5")

def Predict(Characters):
    Predictions = []

    for gray in Characters:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        blur = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 2)

        thresh = cv2.resize(thresh, (32,32), interpolation = cv2.INTER_AREA)

        x = np.array([thresh]).reshape(-1, 32, 32, 1) / 255.0
        Predictions.append(np.argmax(model.predict(x)))

    return Predictions