import os
import cv2
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

Label_Dict = {0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
              5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
              10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
              15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
              20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
              25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
              30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
              36: 'अ', 37: 'आ', 38: 'इ', 39: 'ई', 40: 'उ', 41: 'ऊ', 42: 'ऋ', 43: 'ए', 44: 'ऐ', 45: 'ओ', 46: 'औ', 47: 'अं', 48: 'अ:'}

df = pd.read_csv(os.path.join("Splitted_Dataset", "Reference.csv"))

x_validation = []
y_validation = []
for class_id in df.loc[:,"class id"]:
    for Image_Name in os.listdir(os.path.join("Splitted_Dataset", "Validation", str(class_id))):
        x_validation.append(cv2.imread(os.path.join("Splitted_Dataset", "Validation", str(class_id), Image_Name), 0))
        y_validation.append(class_id)

x_validation = np.array(x_validation).reshape(-1, 32, 32, 1) / 255.0

Predictions = np.array([np.argmax(model.predict(x_validation), axis = 1) for model in Predict_Characters.Models])
Predictions = stats.mode(Predictions)[0][0]

acc = accuracy_score(y_validation, Predictions)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))