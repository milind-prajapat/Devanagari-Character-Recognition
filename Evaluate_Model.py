import os
import cv2
import numpy as np
import pandas as pd

import Split_Words
import Split_Characters

from scipy import stats
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Loss_Models = np.array([load_model(os.path.join(Path, 'best_val_loss.hdf5')) for Path in ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5']])
Accuracy_Models = np.array([load_model(os.path.join(Path, 'best_val_acc.hdf5')) for Path in ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5']])

Label_Dict = {'क': 0, 'ख': 1, 'ग': 2, 'घ': 3, 'ङ': 4, 
              'च': 5, 'छ': 6, 'ज': 7, 'झ': 8, 'ञ': 9, 
              'ट': 10, 'ठ': 11, 'ड': 12, 'ढ': 13, 'ण': 14, 
              'त': 15, 'थ': 16, 'द': 17, 'ध': 18, 'न': 19, 
              'प': 20, 'फ': 21, 'ब': 22, 'भ': 23, 'म': 24, 
              'य': 25, 'र': 26, 'ल': 27, 'व': 28, 'श': 29, 
              'ष': 30, 'स': 31, 'ह': 32, 'क्ष': 33, 'त्र': 34, 'ज्ञ': 35, 
              'अ': 36, 'आ': 37, 'इ': 38, 'ई': 39, 'उ': 40, 'ऊ': 41, 'ऋ': 42, 'ए': 43, 'ऐ': 44, 'ओ': 45, 'औ': 46, 'अं': 47, 'अ:': 48}

x_test = []
y_test = [[['क', 'ल', 'म'], ['प', 'त', 'ल'], ['र', 'व', 'न']],
          [['क', 'म', 'ल'], ['फ', 'स', 'ल'], ['म', 'ह', 'ल'], ['च', 'म', 'क'], ['ल', 'प', 'क'], ['प', 'ट', 'क'], ['न', 'ह', 'र'], ['प', 'ह', 'र'], ['ल', 'ह', 'र']],
          [['आ', 'म'], ['ग', 'म'], ['औ', 'र'], ['अं', 'म'], ['अ:', 'म']],
          [['घ', 'ब', 'न'], ['क', 'च', 'ट'], ['च', 'र', 'म']],
          [['ज']],
          [['ह']],
          [['घ', 'ब', 'न']],
          [['क', 'च', 'ट']],
          [['ज्ञ', 'त', 'ह']],
          [['श', 'क्ष', 'थ']],
          [['च', 'र', 'म']],
          [['द', 'उ', 'ए', 'इ', 'क', 'स']],
          [['द', 'प', 'म', 'र', 'ल', 'न']]]

Path = 'Words'
Images = sorted(os.listdir(Path), key = lambda x: int(os.path.splitext(x)[0]))

for Image_Name in Images:
    Words = Split_Words.Split(cv2.imread(os.path.join(Path, Image_Name)))
    Word_Characters = Split_Characters.Split(Words)
    
    for Characters in Word_Characters:
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

            x_test.append(thresh)

x_test = np.array(x_test).reshape(-1, 32, 32, 1) / 255.0
y_test = [Label_Dict[Character] for Image in y_test for Word in Image for Character in Word]

Predictions = np.array([np.argmax(model.predict(x_test), axis = 1) for model in Loss_Models])

Classification_Report = []
for Prediction in Predictions:
    Classification_Report.append([accuracy_score(y_test, Prediction), 
                                  precision_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                                  recall_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                                  f1_score(y_test, Prediction, average = 'weighted', zero_division = 0)])

Prediction = stats.mode(Predictions)[0][0]
Classification_Report.append([accuracy_score(y_test, Prediction), 
                              precision_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                              recall_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                              f1_score(y_test, Prediction, average = 'weighted', zero_division = 0)])

print(pd.DataFrame(Classification_Report, index = ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5', 'Boosting'], columns = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']))

Predictions = np.array([np.argmax(model.predict(x_test), axis = 1) for model in Accuracy_Models])

Classification_Report = []
for Prediction in Predictions:
    Classification_Report.append([accuracy_score(y_test, Prediction), 
                                  precision_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                                  recall_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                                  f1_score(y_test, Prediction, average = 'weighted', zero_division = 0)])

Prediction = stats.mode(Predictions)[0][0]
Classification_Report.append([accuracy_score(y_test, Prediction), 
                              precision_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                              recall_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                              f1_score(y_test, Prediction, average = 'weighted', zero_division = 0)])

print(pd.DataFrame(Classification_Report, index = ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5', 'Boosting'], columns = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']))