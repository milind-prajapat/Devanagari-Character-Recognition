import os
import cv2
import copy
import numpy as np

from scipy import stats
from keras.models import load_model

Models = np.array([load_model(os.path.join(Path, 'best_val_loss.hdf5')) for Path in ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5']])

def Predict(Word_Characters):
    Predictions = []

    for Characters in Word_Characters:
        Prediction = []
        for Character in Characters:
            gray = cv2.cvtColor(Character, cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            for i in range(morph.shape[0]):
                for j in range(morph.shape[1]):
                    if not morph[i][j]:
                        morph[i][j] = 1
            
            div = gray / morph
            gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

            thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_AREA)

            x = np.array([thresh]).reshape(-1, 32, 32, 1) / 255.0
            Prediction.append(stats.mode(np.array([np.argmax(model.predict(x), axis = 1) for model in Models]))[0][0][0])

        Predictions.append(copy.deepcopy(Prediction))
    
    return Predictions