import cv2
import copy
import numpy as np

def Sorting_Key(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx

def Sorting_Key_Line(contour):
    _, _, w, h = cv2.boundingRect(contour)
    
    return w / h

def Split(Words):
    Word_Characters = []
    for Word in Words:
        gray = cv2.cvtColor(Word, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        sobel = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        sobel = cv2.convertScaleAbs(sobel)

        blur = cv2.medianBlur(sobel, 5)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key = Sorting_Key_Line, reverse = True)

        _, _, w, h = cv2.boundingRect(contours[0])
        thickness = h
        cv2.drawContours(thresh, [contours[0]], 0, 255, -1)

        for contour in contours[1:]:
            cv2.drawContours(thresh, [contour], 0, 0, -1)

        thresh = cv2.bitwise_not(thresh)
        gray = cv2.bitwise_not(gray)
        new_img = cv2.bitwise_and(thresh, gray)

        blur = cv2.medianBlur(new_img, 5)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key = Sorting_Key)

        bounding_rects = []
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                bounding_rects.append(cv2.boundingRect(contour))

        Length = len(bounding_rects)
        avg = 0
        for index in range(Length - 1):
            avg += (bounding_rects[index + 1][0] - (bounding_rects[index][0] + bounding_rects[index][2]))

        avg /= Length
        avg *= 0.9
        
        Characters = []
        index = 0
        while index < (Length - 1):
            distance = bounding_rects[index + 1][0] - (bounding_rects[index][0] + bounding_rects[index][2])
            if distance >= avg:
                x = bounding_rects[index][0]
                y = bounding_rects[index][1]
                w = bounding_rects[index][2]
                h = bounding_rects[index][3]

                x = max(0, x - 3)
                y = max(0, y - 3 - thickness)
                w = min(Word.shape[1], w + 6)
                h = min(Word.shape[0], h + 6 + thickness)

                Character = np.zeros((max(w,h), max(w,h), 3), np.uint8)
                Character.fill(255)

                if w > h:
                    Character[int((w - h) / 2):int((w + h) / 2), :] = Word[y:y + h, x:x + w]
                else:
                    Character[:, int((h - w) / 2):int((w + h) / 2)] = Word[y:y + h, x:x + w]

                Characters.append(Character.copy())
            else:
                x = bounding_rects[index][0]
                w = bounding_rects[index + 1][0] - bounding_rects[index][0] + bounding_rects[index + 1][2]
                y = min(bounding_rects[index][1], bounding_rects[index + 1][1])
                h = max(bounding_rects[index][1] + bounding_rects[index][3], bounding_rects[index + 1][1] + bounding_rects[index + 1][3])

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
        y = max(0, y - 3 - thickness)
        w = min(Word.shape[1], w + 6)
        h = min(Word.shape[0], h + 6 + thickness)

        Character = np.zeros((max(w,h), max(w,h), 3), np.uint8)
        Character.fill(255)

        if w > h:
            Character[int((w - h) / 2):int((w + h) / 2), :] = Word[y:y + h, x:x + w]
        else:
            Character[:, int((h - w) / 2):int((w + h) / 2)] = Word[y:y + h, x:x + w]

        Characters.append(Character.copy())

        Word_Characters.append(copy.deepcopy(Characters))

    return Word_Characters