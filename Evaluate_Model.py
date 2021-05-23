import os
import cv2
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import accuracy_score
from keras.models import load_model

df = pd.read_csv(os.path.join('Split_Dataset', 'Reference.csv'))

x_validation = []
y_validation = []

for class_id in df.loc[:, 'class id']:
    for Image_Name in os.listdir(os.path.join('Split_Dataset', 'Validation', str(class_id))):
        x_validation.append(cv2.imread(os.path.join('Split_Dataset', 'Validation', str(class_id), Image_Name), 0))
        y_validation.append(class_id)

x_validation = np.array(x_validation).reshape(-1, 32, 32, 1) / 255.0

Models = np.array([load_model(os.path.join(Path, 'best_val_loss.hdf5')) for Path in ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5']])
Predictions = np.array([np.argmax(model.predict(x_validation), axis = 1) for model in Models])
Predictions = stats.mode(Predictions)[0][0]

acc = accuracy_score(y_validation, Predictions)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))