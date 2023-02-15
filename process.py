import numpy as np
import cv2
import matplotlib.pyplot as plt


def plotImage(image, caption=""):
    plt.figure(figsize=(6, 10))
    plt.title(caption)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap="gray")


def calcAvgLineColor(line, image):
    x1, y1, x2, y2 = line
    pass


def sortLine(line):
    point1 = line[0][:2]
    point2 = line[0][2:]
    return calcDistanceP2P(point1, point2)


def sortPoint(point):
    return point[0] + point[1]


def calcDistanceP2P(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calcCos2Line(line1, line2):
    x1, y1, x2, y2 = line1[0]
    u1 = [x2-x1, y2-y1]
    x1, y1, x2, y2 = line2[0]
    u2 = [x2-x1, y2-y1]
    cosa = (u1[0]*u2[0] + u1[1]*u2[1]) / \
        ((np.sqrt(u1[0]**2 + u1[1]**2))*(np.sqrt(u2[0]**2 + u2[1]**2)))
    return np.abs(cosa)


def pointBetweenLine(line, point):
    point1 = line[0][:2]
    point2 = line[0][2:]
    p = [point1, point2]
    p = sorted(p, key=sortPoint)
    x1, y1 = p[0]
    x2, y2 = p[1]
    x, y = point
    if (x >= x1 and x <= x2) or (y >= y1 and y <= y2):
        return True
    return False


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


def calcDistanceP2L(line, point):
    x1, y1, x2, y2 = line[0]
    x, y = point
    a = y1-y2
    b = x2-x1
    c = -y1*(x2-x1) + x1*(y2-y1)
    distance = np.abs(a*x+b*y+c)/(np.sqrt(a**2 + b**2))
    return distance


def getMinDistance(line1, line2):
    point1 = line1[0][:2]
    point2 = line1[0][2:]
    point3 = line2[0][:2]
    point4 = line2[0][2:]
    l1 = calcDistanceP2P(point1, point3)
    l2 = calcDistanceP2P(point1, point4)
    l3 = calcDistanceP2P(point2, point3)
    l4 = calcDistanceP2P(point2, point4)
    return min(l1, l2, l3, l4)


def mergeLines(lines, image):
    checkLines = np.zeros(len(lines))
    resLines = []
    for i in range(len(lines)):
        if checkLines[i] != 0:
            continue
        start = lines[i][0][:2]
        end = lines[i][0][2:]
        point = [start, end]
        for j in range(i+1, len(lines)):
            if checkLines[j] != 0:
                continue
            point1 = lines[j][0][:2]
            point2 = lines[j][0][:2]
            cosa = calcCos2Line(lines[i], lines[j])
            D1 = calcDistanceP2L(lines[i], point1)
            D2 = calcDistanceP2L(lines[i], point2)
            if cosa >= 0.95 and D1 <= 13 and D2 <= 13:
                if (pointBetweenLine(lines[i], point1) == True) or (pointBetweenLine(lines[i], point2) == True):
                    checkLines[j] = 1
                else:
                    minDistance = getMinDistance(lines[i], lines[j])
                    if minDistance < 300:
                        checkLines[j] = 1
            if checkLines[j] == 1:
                point.append(point1)
                point.append(point2)
        point = sorted(point, key=sortPoint)
        start = point[0]
        end = point[-1]
        line = [[start[0], start[1], end[0], end[1]]]
        resLines.append(line)
    return resLines
