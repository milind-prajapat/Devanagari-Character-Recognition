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

        original_thresh = thresh.copy()

        h_proj = np.sum(thresh, axis = 1)
        Max = np.max(h_proj) / 2

        upper = None
        lower = None
        for i in range(h_proj.shape[0]):
            proj = h_proj[i]
            if proj > Max and upper == None:
                upper = i
            elif proj < Max and upper != None and lower == None:
                lower = i
                break

        if lower == None:
            lower = int(h_proj.shape[0] / 2)

        for row in range(lower + 3):
            thresh[row] = 0
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        bounding_rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 50:
                bounding_rects.append((x, y, w, h))
        
        bounding_rects.sort(key = lambda x: x[0] + int(x[2] / 2))

        index = 0
        Length = len(bounding_rects)  
        while index < (Length - 1):
            x, y, w, h = bounding_rects[index]

            x_left = max(x, bounding_rects[index + 1][0])
            y_top = max(y, bounding_rects[index + 1][1])
            x_right = min(x + w,  bounding_rects[index + 1][0] +  bounding_rects[index + 1][2])
            y_bottom = min(y + h, bounding_rects[index + 1][1] +  bounding_rects[index + 1][3])

            intersection_area = max(0, (x_right - x_left)) * max(0, (y_bottom - y_top))
            union = float((bounding_rects[index][2] * bounding_rects[index][3]) + (bounding_rects[index + 1][2] * bounding_rects[index + 1][3]) - intersection_area)

            area_ratio = (bounding_rects[index + 1][2] * bounding_rects[index + 1][3]) / union

            ratio = (bounding_rects[index + 1][2] * bounding_rects[index + 1][3]) / (w * h)

            if bounding_rects[index + 1][3] / bounding_rects[index + 1][2] > 3 or ratio <= 0.25 or (area_ratio > 0.9 and intersection_area != 0):
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
            x1 = max(0, x - 3)  
            w1 = min(Word.shape[1] - x, w + 6)
            h1 = min(Word.shape[0], h + y + 6)
            y1 = 0

            crop = original_thresh[y1:y1 + h1, x1:x1 + w1]
            
            h_proj = np.sum(crop, axis = 1)

            padding = None
            for i in range(h_proj.shape[0]):
                proj = h_proj[i]
                if proj != 0:
                    padding = y - i
                    break

            x = max(0, x - 3)
            y = max(0, y - padding)
            w = min(Word.shape[1] - x, w + 6)
            h = min(Word.shape[0] - y, h + 3 + padding)

            Character = np.zeros((max(w,h), max(w,h), 3), np.uint8)
            Character.fill(255)

            if w > h:
                Character[int((w - h) / 2):int((w + h) / 2), :] = Word[y:y + h, x:x + w]
            else:
                Character[:, int((h - w) / 2):int((w + h) / 2)] = Word[y:y + h, x:x + w]

            Characters.append(Character.copy())

        Word_Characters.append(copy.deepcopy(Characters))

    return Word_Characters
