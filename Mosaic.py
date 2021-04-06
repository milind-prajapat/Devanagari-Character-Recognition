import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical

Label_dict={
1:"क", 2:"ख", 3:"ग", 4:"घ", 5:"ङ",
6:"च", 7:"छ", 8:"ज", 9:"झ", 10:"ञ",
11:"ट", 12:"ठ", 13:"ड", 14:"ढ", 15:"ण",
16:"त", 17:"थ", 18:"द", 19:"ध", 20:"न",
21:"प", 22:"फ", 23:"ब", 24:"भ", 25:"म",
26:"य", 27:"र", 28:"ल", 29:"व", 30:"श", 31:"ष",
32:"स", 33:"ह",34:"क्ष", 35:"त्र", 36:"ज्ञ", 37:"०", 38: "१", 39: "२", 40:"३", 41:"४", 42:"५", 43:"६", 44:"७", 45:"८", 46:"९"
47:"अ", 48:"आ", 49:"इ", 50:"ई", 51:"उ", 52:"ऊ", 53:"ऋ", 54:"ए", 55:"ऐ", 56:"ओ", 57:"औ",
58:"अं" , 59:"अ:"}

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
