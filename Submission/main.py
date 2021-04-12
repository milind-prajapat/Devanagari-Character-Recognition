import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import Split_Words
import Split_Characters
import Predict_Characters

Label_Dict = {0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
              5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
              10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
              15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
              20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
              25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
              30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
              36: 'अ', 37: 'आ', 38: 'इ', 39: 'ई', 40: 'उ', 41: 'ऊ', 42: 'ऋ', 43: 'ए', 44: 'ऐ', 45: 'ओ', 46: 'औ', 47: 'अं', 48: 'अ:'}

Path = "Words"

'''A nested list containing list of each word on a page'''
answer = []

for Image_Name in os.listdir(Path):
    Words = Split_Words.Split(cv2.imread(os.path.join(Path, Image_Name)))
    Word_Characters = Split_Characters.Split(Words)
    Predictions = Predict_Characters.Predict(Word_Characters)

    for Prediction in Predictions:
        Word = ""
        list = []
        for class_id in Prediction:
            Word += Label_Dict[class_id]
            list.append(Label_Dict[class_id])
        answer.append(list)           
        print(Word, end = " ")
    print("")
    




