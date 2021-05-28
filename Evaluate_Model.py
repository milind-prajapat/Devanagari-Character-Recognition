import os
import cv2
import numpy as np
import pandas as pd

import Split_Words
import Split_Characters
import Predict_Characters

from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(os.path.join('Split Dataset', 'Reference.csv'))

x_validation = []
y_validation = []

for Class in df.loc[:, 'Class']:
    for Image_Name in os.listdir(os.path.join('Split Dataset', 'Validation', str(Class))):
        x_validation.append(cv2.imread(os.path.join('Split Dataset', 'Validation', str(Class), Image_Name), 0))
        y_validation.append(Class)

x_validation = np.array(x_validation).reshape(-1, 32, 32, 1) / 255.0
Predictions = [np.argmax(Model.predict(x_validation), axis = 1) for Model in Predict_Characters.Models]

Dict = {}
for Model_Number, Prediction in enumerate(Predictions):
    Dict[f'Model_{Model_Number + 1}'] = [accuracy_score(y_validation, Prediction), 
                                     precision_score(y_validation, Prediction, average = 'weighted', zero_division = 0), 
                                     recall_score(y_validation, Prediction, average = 'weighted', zero_division = 0), 
                                     f1_score(y_validation, Prediction, average = 'weighted', zero_division = 0)]

Prediction = stats.mode(Predictions)[0][0]
Dict['Boosting'] = [accuracy_score(y_validation, Prediction),
                    precision_score(y_validation, Prediction, average = 'weighted', zero_division = 0), 
                    recall_score(y_validation, Prediction, average = 'weighted', zero_division = 0), 
                    f1_score(y_validation, Prediction, average = 'weighted', zero_division = 0)]

Validation_Report = pd.DataFrame.from_dict(Dict, orient = 'index', columns = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']).round(4)

Expected_Outcomes = [[['क', 'ल', 'म'], ['प', 'त', 'ल'], ['र', 'व', 'न']],
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

Predictions = []
for Image_Name in Images:
    Words = Split_Words.Split(cv2.imread(os.path.join(Path, Image_Name)))
    Characters = Split_Characters.Split(Words)
    Predictions.append(Predict_Characters.Predict(Characters, Evaluate = True))
    
y_test = [Predict_Characters.Reversed_Label_Dict[Character] for Image in Expected_Outcomes for Word in Image for Character in Word]
Predictions = np.array([Character for Image in Predictions for Word in Image for Character in Word]).T.tolist()

Dict = {}
for Model_Number, Prediction in enumerate(Predictions):
    Dict[f'Model_{Model_Number + 1}'] = [accuracy_score(y_test, Prediction), 
                                     precision_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                                     recall_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                                     f1_score(y_test, Prediction, average = 'weighted', zero_division = 0)]

Prediction = stats.mode(Predictions)[0][0]
Dict['Boosting'] = [accuracy_score(y_test, Prediction),
                    precision_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                    recall_score(y_test, Prediction, average = 'weighted', zero_division = 0), 
                    f1_score(y_test, Prediction, average = 'weighted', zero_division = 0)]

Test_Report = pd.DataFrame.from_dict(Dict, orient = 'index', columns = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']).round(4)

print('Classification Report on Validation Data:')
print(Validation_Report)
print('')
print('Classification Report on Sample Words:')
print(Test_Report)