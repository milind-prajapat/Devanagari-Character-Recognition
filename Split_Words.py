import cv2
import numpy as np

def Sorting_Key(contour):
    global Lines, Length, Size

    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    for i in range(Length):
        if cy > Lines[i][0] and cy < Lines[i][1]:
            return cx + ((i + 1) * Size)

def Split(img):
    global Lines, Length

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    blur = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 2)

    h_proj = np.sum(thresh, axis = 1)

    upper = None
    lower = None
    Lines = []
    for i in h_proj:
        if i != 0 and upper == None:
            upper = i
        elif i == 0 and upper != None and lower == None:
            lower = i
            Lines.append([upper, lower])
            upper = None
            lower = None

    Length = len(Lines)
    Size = thresh.shape[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key = Sorting_Key)

    Words = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        shape = (box[1][0] - box[0][0], box[3][1] - box[0][1])

        src = np.float32(box)
        dst = np.array([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]], np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img, M, shape)

        Words.append(warp.copy())

    return Words
