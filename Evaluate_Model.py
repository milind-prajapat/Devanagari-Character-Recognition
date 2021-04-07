import os
import cv2
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical

x_test = []
y_test = []

df = pd.read_csv(os.path.join("Test", "Reference.csv"))

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
    y_test.append(df.iloc[i, 1])

x_test = np.array(x_test).reshape(-1, 32, 32, 1) / 255.0
y_test = to_categorical(y_test, 59)

model = load_model("best_val_acc.hdf5")
loss, acc = model.evaluate(x_test, y_test)

print("Best Accuracy Model:")
print("Loss on Test Data :", loss)
print("Accuracy on Test Data :", "{:.4%}".format(acc))

model = load_model("best_val_loss.hdf5")
loss, acc = model.evaluate(x_test, y_test)

print("Best Loss Model:")
print("Loss on Test Data :", loss)
print("Accuracy on Test Data :", "{:.4%}".format(acc))

y_pred = [Label_Dict[class_id] for class_id in np.argmax(model.predict(x_test), axis = 1)]
y_test = [Label_Dict[class_id] for class_id in np.argmax(y_test, axis = 1)]

Results = pd.DataFrame(list(zip(df.iloc[:, 0], y_test, y_pred)), columns = ["File_Name", "Actual", "Prediction"])
Results.to_csv(os.path.join("Test", "Final_Results.csv"), index = False)