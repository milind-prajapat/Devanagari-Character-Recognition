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
            if cv2.contourArea(contour) > 200:
                bounding_rects.append(cv2.boundingRect(contour))

        i = 0  
        Length = len(bounding_rects) 
        while i < Length:
            x, y, w, h = bounding_rects[i]
            j = 0
           
            while j < Length:

                if i != j and all([not any([all([bounding_rects[j][1] > y + h, bounding_rects[j][1] + bounding_rects[j][3] > y + h]), all([bounding_rects[j][1] < y, bounding_rects[j][1] + bounding_rects[j][3] < y])]),
                                   not any([all([bounding_rects[j][0] > x + w, bounding_rects[j][0] + bounding_rects[j][2] > x + w]), all([bounding_rects[j][0] < x, bounding_rects[j][0] + bounding_rects[j][2] < x])])]):

                    x = min(bounding_rects[i][0], bounding_rects[j][0])
                    w = max(bounding_rects[i][0] + bounding_rects[i][2], bounding_rects[j][0] + bounding_rects[j][2]) - x
                    y = min(bounding_rects[i][1], bounding_rects[j][1])
                    h = max(bounding_rects[i][1] + bounding_rects[i][3], bounding_rects[j][1] + bounding_rects[j][3]) - y

                    bounding_rects[i] = (x, y, w, h)

                    del bounding_rects[j]
                    i = -1
                    Length -= 1
                    break

                j += 1
            i += 1

        bounding_rects.sort(key = lambda x: x[0] + int(x[2] / 2))
        
        index = 0
        Length = len(bounding_rects)  
        while index < (Length - 1):
            x, y, w, h = bounding_rects[index]

            if bounding_rects[index + 1][3] / bounding_rects[index + 1][2] > 3.5:
                x = min(bounding_rects[index][0], bounding_rects[index + 1][0])
                w = max(bounding_rects[index][0] + bounding_rects[index][2], bounding_rects[index + 1][0] + bounding_rects[index + 1][2]) - x
                y = min(bounding_rects[index][1], bounding_rects[index + 1][1])
                h = max(bounding_rects[index][1] + bounding_rects[index][3], bounding_rects[index + 1][1] + bounding_rects[index + 1][3]) - y

                bounding_rects[index] = (x, y, w, h)
                del bounding_rects[index + 1]
                index -= 1
                Length -= 1

            index += 1
        
        Characters = []
        for x, y, w, h in bounding_rects:
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
