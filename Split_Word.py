import cv2
import numpy as np

def Split(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    sobel = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    sobel = cv2.convertScaleAbs(sobel)

    blur = cv2.medianBlur(sobel, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w / h > 5:
            cv2.drawContours(thresh, [contour], 0, 255, -1)
        else:
            cv2.drawContours(thresh, [contour], 0, 0, -1)

    thresh = cv2.bitwise_not(thresh)
    gray = cv2.bitwise_not(gray)
    new_img = cv2.bitwise_and(thresh, gray)

    blur = cv2.medianBlur(new_img, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key = lambda contour: contour[0][0][0])

    Characters = []
    for contour in contours:
        if cv2.contourArea(contour) > 200:
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x - 5)
            y = max(0, y - 15)
            w += 10
            h += 20

            Characters.append(img[y:y+h, x:x+h].copy())

    return Characters