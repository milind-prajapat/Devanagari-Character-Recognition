import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical

Labels = ["ka", "kha", "ga", "gha", "kna",
          "cha", "chha", "ja", "jha", "yna",
          "taamatar", "thaa", "daa", "dhaa", "adna",
          "tabala", "tha", "da", "dha", "na",
          "pa", "pha", "ba", "bha", "ma",
          "yaw", "ra", "la", "waw", "motosaw",
          "petchiryakha", "patalosaw", "ha", "chhya", "tra", "gya",
          "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

x_test = []
y_test = []

df = pd.read_csv(r"Test/Reference.csv")

for i in df.index:
    img = cv2.imread(df.iloc[i,0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    x_test.append(thresh.copy())
    y_test.append(df.iloc[i,1])

x_test = np.array(x_test).reshape(-1, 32, 32, 1) / 255.0
y_test = to_categorical(y_test, 46)

model = load_model("best_val_loss.hdf5")
#loss, acc = model.evaluate(x_test, y_test)

print(df.iloc[:,1].values)
print([np.argmax(x) for x in model.predict(x_test)])

#print("Loss:", loss)
#print("Accuracy:", acc)