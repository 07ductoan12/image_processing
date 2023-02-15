import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


def auto_brightness_contranst(image, clip_hist_precent=5):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = []
    accumulator.append(hist[0])
    for i in range(1, hist_size):
        accumulator.append(accumulator[i-1] + float(hist[i]))

    maximum = accumulator[-1]
    clip_hist_precent *= (maximum/100)
    clip_hist_precent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_precent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] > (maximum - clip_hist_precent):
        maximum_gray -= 1

    alpha = 255/(maximum_gray - minimum_gray)
    beta = -minimum_gray*alpha

    image = convertScale(image, alpha, beta)
    return image


def plotImage(image, caption=""):
    plt.figure(figsize=(6, 9))
    plt.title(caption)
    if (len(image.shape) == 3):
        plt.imshow(image)
    else:
        plt.imshow(image, cmap="gray")


def Line(line):
    x1, y1, x2, y2 = line[0]
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    if A > B and A != 0:
        return 1, B/A, -C/A
    if A < B and B != 0:
        return A/B, 1, -C/B
    return A, B, -C


def calDistanceP2L(line, point):
    x1, y1, x2, y2 = line[0]
    x, y = point
    a = y1-y2
    b = x2-x1
    c = -y1*(x2-x1) + x1*(y2-y1)
    distance = np.abs(a*x+b*y+c)/(np.sqrt(a**2 + b**2))
    return distance


def calDistanceP2P(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def getMinDistance(line1, line2):
    point1 = line1[0][:2]
    point2 = line1[0][2:]
    point3 = line2[0][:2]
    point4 = line2[0][2:]
    l1 = calDistanceP2P(point1, point3)
    l2 = calDistanceP2P(point1, point4)
    l3 = calDistanceP2P(point2, point3)
    l4 = calDistanceP2P(point2, point4)
    return min(l1, l2, l3, l4)


def sortPoint(point):
    return point[0] + point[1]


def calCos(line1, line2):
    x1, y1, x2, y2 = line1[0]
    u1 = [x2-x1, y2-y1]
    x1, y1, x2, y2 = line2[0]
    u2 = [x2-x1, y2-y1]
    cosa = (u1[0]*u2[0] + u1[1]*u2[1]) / \
        ((np.sqrt(u1[0]**2 + u1[1]**2))*(np.sqrt(u2[0]**2 + u2[1]**2)))
    return cosa


def calcAvgSectionColor(point, image):
    avg = 0
    count = 0
    x, y = point
    for i in range(-1, 1):
        for j in range(-1, 1):
            count += 1
            avg = image[y+j][x+i]
    avg /= count
    return avg


def checkAvgColorLine(line, image):
    x1, y1, x2, y2 = line[0]
    avg = 0
    count = 0
    dimensional = np.abs(x1-x2) - np.abs(y1-y2)
    if dimensional > 0 or np.abs(y1-y2) == 0:
        x_min, x_max = sorted([x1, x2])
        for i in range(x_max-x_min-1):
            count += 1
            x = x_min + i
            y = int(((x-x1)*(y2-y1))/(x2-x1) + y1)
            avg += calcAvgSectionColor([x, y], image)
    else:
        y_min, y_max = sorted([y1, y2])
        for i in range(y_max - y_min - 1):
            count += 1
            y = y_min + i
            x = int(((y-y1)*(x2-x1))/(y2-y1) + x1)
            avg += calcAvgSectionColor([x, y], image)
    if count == 0:
        return 255
    avg /= count
    return avg


def drawHoughLine(imgSrc, imgDst, imgGray, minLineLenght, maxLineGap, getLines):
    lines = cv.HoughLinesP(imgSrc, 5, np.pi/180, 127,
                           minLineLength=minLineLenght, maxLineGap=maxLineGap)
    resLines = []
    if lines is None:
        return imgDst, resLines
    if getLines:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            avgColor = checkAvgColorLine(line, imgGray)
            if avgColor <= 75:
                cv.line(imgDst, (x1, y1), (x2, y2), 255, 3)
                resLines.append(line)
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            avgColor = checkAvgColorLine(line, imgGray)
            if avgColor <= 75:
                cv.line(imgDst, (x1, y1), (x2, y2), 255, 3)
    return imgDst, resLines


def getLines(imgSrc, imgGray, minLineLenght, maxLineGap):
    lines = cv.HoughLinesP(imgSrc, 5, np.pi/180, 127,
                           minLineLength=minLineLenght, maxLineGap=maxLineGap)
    res_Line = []
    for line in lines:
        avgColor = checkAvgColorLine(line, imgGray)
        if avgColor <= 100:
            res_Line.append(line)
    return res_Line


def drawEdges(imgSrc, imgDst, imgGray, minLineLenght, maxLineGap, getLines=False):
    apertureSizes = [3, 5]
    resLines = []
    for size in apertureSizes:
        edges = cv.Canny(imgSrc, 30, 100, apertureSize=size)
        imgDst, lines = drawHoughLine(edges, imgDst, imgGray,
                                      minLineLenght, maxLineGap, getLines)
        if getLines:
            for line in lines:
                resLines.append(line)
    return imgDst, resLines


def getHoughLine(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    res = np.zeros_like(image, np.uint8)
    blurred = cv.GaussianBlur(image, (7, 7), 0)
    blurred = cv.bilateralFilter(blurred, 9, 7, 37)
    res, _ = drawEdges(blurred, res, image, 175, 5, False)
    kernelSize = [11, 17, 21]
    for kernel in kernelSize:
        img_thresh = cv.adaptiveThreshold(
            blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, kernel, 15)
        res, _ = drawEdges(img_thresh, res, image, 175, 10, False)
    return res


def calLenghLine(line):
    if len(line) == 1:
        x1, y1, x2, y2 = line[0]
    else:
        x1, y1, x2, y2 = line
    return np.sqrt(((x1-x2)**2) + ((y1-y2)**2))


def getFrame(original):
    image = getHoughLine(original)
    gray = cv.cvtColor(original, cv.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((7, 7), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    lines = getLines(image, gray, 125, 15)
    res = np.zeros(image.shape, np.uint8)
    linesSorted = sorted(lines, key=calLenghLine, reverse=True)[:50]
    for line in linesSorted:
        x1, y1, x2, y2 = line[0]
        cv.line(res, (x1, y1), (x2, y2), 255, 3)
        cv.line(original, (x1, y1), (x2, y2), 255, 3)
    return res, original, linesSorted


def checkPoint(point1, point2, point3):
    points = [point1, point2]
    points = sorted(points, key=sortPoint)
    x1, y1 = points[0]
    x2, y2 = points[1]
    x, y = point3
    if (x >= x1 and x <= x2) or (y >= y1 and y <= y2):
        return True
    return False


def mergeLines(lines, image):
    checkLine = np.zeros(len(lines))
    resLines = []
    for i in range(len(lines)):
        if checkLine[i] == 0:
            start = lines[i][0][:2]
            end = lines[i][0][2:]
            for j in range(i+1, len(lines)):
                if checkLine[j] == 0:
                    point1 = lines[j][0][:2]
                    point2 = lines[j][0][2:]
                    distance1 = calDistanceP2L(lines[i], point1)
                    distance2 = calDistanceP2L(lines[i], point2)
                    cosa = calCos(lines[i], lines[j])
                    if distance1 <= 13 and distance2 <= 13 and cosa >= 0.95:
                        point = [point1, point2, start, end]
                        point = sorted(point, key=sortPoint)
                        if (checkPoint(start, end, point1) == True or checkPoint(start, end, point2) == True):
                            start = point[0]
                            end = point[3]
                            checkLine[j] = 1
                        else:
                            minDistance = getMinDistance(lines[i], lines[j])
                            middleLine = [[point[1][0], point[1]
                                           [1], point[2][0], point[2][1]]]
                            avgML = checkAvgColorLine(middleLine, image)
                            if minDistance <= 720 and avgML <= 127:
                                start = point[0]
                                end = point[3]
                                checkLine[j] = 1

            line = [[start[0], start[1], end[0], end[1]]]
            resLines.append(line)
    return resLines
