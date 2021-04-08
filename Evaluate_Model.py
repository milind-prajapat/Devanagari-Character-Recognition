import os
import cv2
import Split_Words
import pandas as pd
import Split_Characters
import Predict_Characters

from sklearn.metrics import accuracy_score

os.system("cls")

Label_Dict = {0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
              5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
              10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
              15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
              20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
              25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
              30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
              36: '०', 37: '१', 38: '२', 39: '३', 40: '४', 41: '५', 42: '६', 43: '७', 44: '८', 45: '९',
              46: 'अ', 47: 'आ', 48: 'इ', 49: 'ई', 50: 'उ', 51: 'ऊ', 52: 'ऋ', 53: 'ए', 54: 'ऐ', 55: 'ओ', 56: 'औ', 57: 'अं', 58: 'अ:'}

df = pd.read_csv(os.path.join("Test", "Reference.csv"))

for Image_Path in df.iloc[:,0]:
    Words = Split_Words.Split(cv2.imread(Image_Path))
    Word_Characters = Split_Characters.Split(Words)
    Predictions = Predict_Characters.Predict(Word_Characters)

    for Prediction in Predictions:
        Word = ""
        for class_id in Prediction:
            Word += Label_Dict[class_id]
        print(Word, end = " ")
    print("")

##Characters = [cv2.imread(img) for img in df.iloc[:,0]]
##y_pred = Predict_Characters.Predict(Characters)
##
##y_true = df.iloc[:, 1].values.to_list()
##
##acc = accuracy_score(y_true, y_pred)
##print("Accuracy on Test Data :", "{:.4%}".format(acc))
##
##y_pred = [Label_Dict[class_id] for class_id in np.argmax(y_pred, axis = 1)]
##y_true = [Label_Dict[class_id] for class_id in np.argmax(y_true, axis = 1)]
##
##Results = pd.DataFrame(list(zip(df.iloc[:, 0], y_true, y_pred)), columns = ["File_Name", "Actual", "Prediction"])
##Results.to_csv(os.path.join("Test", "Final_Results.csv"), index = False)
##
