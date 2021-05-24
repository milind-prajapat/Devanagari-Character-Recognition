import os
import cv2

import Split_Words
import Split_Characters
import Predict_Characters

Path = 'Words'
Images = sorted(os.listdir(Path), key = lambda x: int(os.path.splitext(x)[0]))

for Image_Name in Images:
    Words = Split_Words.Split(cv2.imread(os.path.join(Path, Image_Name)))
    Characters = Split_Characters.Split(Words)
    Predictions = Predict_Characters.Predict(Characters)

    Words = []
    for Prediction in Predictions:
        Word = ''.join(Prediction)
        Words.append(Word)
    Words = ' '.join(Words)

    print(Words)