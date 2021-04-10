import cv2
import copy
import numpy as np

def Split(Words):
    Word_Characters = []
    for Word in Words:
        gray = cv2.cvtColor(Word, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)

        for row in range(int(thresh.shape[0] * 0.25)):
            thresh[row] = 0

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        bounding_rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 200:
                bounding_rects.append((x, y, w, h))

        Characters = []
        if len(bounding_rects):
            bounding_rects.sort(key = lambda x: x[0] + int(x[2] / 2))

            Length = len(bounding_rects)
            
            index = 0
            while index < (Length - 1):
                if bounding_rects[index + 1][3] / bounding_rects[index + 1][2] < 3:
                    x = bounding_rects[index][0]
                    y = bounding_rects[index][1]
                    w = bounding_rects[index][2]
                    h = bounding_rects[index][3]

                    x = max(0, x - 3)  
                    w = min(Word.shape[1] - x, w + 6)
                    h = min(Word.shape[0], h + y + 6)
                    y = 0

                    Character = np.zeros((max(w,h), max(w,h), 3), np.uint8)
                    Character.fill(255)

                    if w > h:
                        Character[int((w - h) / 2):int((w + h) / 2), :] = Word[y:y + h, x:x + w]
                    else:
                        Character[:, int((h - w) / 2):int((w + h) / 2)] = Word[y:y + h, x:x + w]

                    Characters.append(Character.copy())
                else:
                    x = min(bounding_rects[index][0], bounding_rects[index + 1][0])
                    w = max(bounding_rects[index][0] + bounding_rects[index][2], bounding_rects[index + 1][0] + bounding_rects[index + 1][2]) - x
                    y = min(bounding_rects[index][1], bounding_rects[index + 1][1])
                    h = max(bounding_rects[index][1] + bounding_rects[index][3], bounding_rects[index + 1][1] + bounding_rects[index + 1][3]) - y

                    bounding_rects[index] = (x, y, w, h)
                    del bounding_rects[index + 1]
                    index -= 1
                    Length -= 1
                index += 1

            x = bounding_rects[-1][0]
            y = bounding_rects[-1][1]
            w = bounding_rects[-1][2]
            h = bounding_rects[-1][3]

            x = max(0, x - 3)
            w = min(Word.shape[1] - x, w + 6)
            h = min(Word.shape[0], h + y + 6)
            y = 0

            Character = np.zeros((max(w,h), max(w,h), 3), np.uint8)
            Character.fill(255)

            if w > h:
                Character[int((w - h) / 2):int((w + h) / 2), :] = Word[y:y + h, x:x + w]
            else:
                Character[:, int((h - w) / 2):int((w + h) / 2)] = Word[y:y + h, x:x + w]

            Characters.append(Character.copy())

        Word_Characters.append(copy.deepcopy(Characters))

    return Word_Characters