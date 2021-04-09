import cv2
import numpy as np

def Sorting_Key(rect):
    global Lines, Length, Size

    x, y, w, h = rect

    cx = x + int(w / 2)
    cy = y + int(h / 2)

    for i in range(Length):
        if cy >= Lines[i][0] and cy <= Lines[i][1]:
            return cx + ((i + 1) * Size)

def Split(img):
    global Lines, Length, Size

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)

    h_proj = np.sum(thresh, axis = 1)

    upper = None
    lower = None
    Lines = []
    for i in range(h_proj.shape[0]):
        proj = h_proj[i]
        if proj != 0 and upper == None:
            upper = i
        elif proj == 0 and upper != None and lower == None:
            lower = i
            Lines.append([upper, lower])
            upper = None
            lower = None

    Length = len(Lines)
    Size = thresh.shape[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 400:
            cy = y + int(h / 2)

            for i in range(Length):
                if cy >= Lines[i][0] and cy <= Lines[i][1]:
                    bounding_rects.append((x, y, w, h))
        else:
            cv2.drawContours(thresh, [contour], 0, 0, -1)

    i = 0  
    Length = len(bounding_rects) 
    while i < Length:
        x, y, w, h = bounding_rects[i]
        j = 0
        while j < Length:
            distancex = abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2]))
            distancey = abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3]))
            if i != j and any([all([bounding_rects[j][0] > x, bounding_rects[j][0] < x + w, bounding_rects[j][1] > y, bounding_rects[j][1] < y + h]), all([distancex <= 15, distancey <= max(bounding_rects[i][3], bounding_rects[j][3]) + 15]), all([distancex <= max(bounding_rects[i][2], bounding_rects[j][2]) + 15, distancey <= 15])]):
                y = min(bounding_rects[i][0], bounding_rects[j][0])
                w = max(bounding_rects[i][0] + bounding_rects[i][2], bounding_rects[j][0] + bounding_rects[j][2]) - x
                y = min(bounding_rects[i][1], bounding_rects[j][1])
                h = max(bounding_rects[i][1] + bounding_rects[i][3], bounding_rects[j][1] + bounding_rects[j][3]) - y

                bounding_rects[i] = (x, y, w, h)
                del bounding_rects[j]
                j = -1
                Length -= 1

                if j < i:
                    i -= 1
            j += 1
        i += 1

    Length = len(Lines)
    bounding_rects.sort(key = Sorting_Key)

    out = img.copy()
    for x,y,w,h in bounding_rects:
        cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 5)

    Words = []

    for x, y, w, h in bounding_rects:
        crop = img[y:y + h, x:x+ w]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = np.vstack(contours)

        rect = cv2.minAreaRect(contours)
        Box = cv2.boxPoints(rect)
        Box = np.int0(Box)

        index = np.argmin(np.sum(Box, axis = 1))

        box = []
        box.extend(Box[index:])
        box.extend(Box[0:index])
           
        shape = (box[1][0] - box[0][0], box[3][1] - box[0][1])

        src = np.float32(box)
        dst = np.array([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]], np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.bitwise_not(cv2.warpPerspective(cv2.bitwise_not(crop), M, shape))

        Words.append(warp.copy())

    return Words