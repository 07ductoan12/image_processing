import numpy as np
import cv2
import matplotlib.pyplot as pl
from process import *


def convertScale(image, alpha, beta):
    new_img = image*alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


def autoBrightnessContrast(image, clip_hist_percent=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = []
    accumulator.append(float(hist[0]))
    for i in range(1, hist_size):
        accumulator.append(accumulator[i-1] + float(hist[i]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] <= clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255/(maximum_gray-minimum_gray)
    beta = -minimum_gray*alpha
    newimg = convertScale(image, alpha, beta)
    return newimg


def drawHoughLine(imgSrc, imgDst, minLineLengt, maxLineGap):
    lines = cv2.HoughLinesP(imgSrc, 1, np.pi/180, 127,
                            minLineLength=minLineLengt, maxLineGap=maxLineGap)
    if lines is None:
        return imgDst
    res_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(imgDst, (x1, y1), (x2, y2), 2)
    return imgDst


def drawEdges(imgSrc, imgDst, minLineLenght, maxLineGap):
    apertureSizes = [3, 5, 7]
    for size in apertureSizes:
        edges = cv2.Canny(imgSrc, 50, 150, apertureSize=size)
        imgDst = drawHoughLine(
            edges, imgDst, minLineLenght, maxLineGap)
    return imgDst


def getHoughLine(image):
    res = np.zeros(image.shape, np.uint8)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blurred = cv2.bilateralFilter(blurred, 9, 7, 37)
    res = drawEdges(blurred, res, 150, 5)
    kernelSizes = [17, 19, 21]
    for kernel in kernelSizes:
        img_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel, 15)
        res = drawEdges(img_thresh, res, 175, 10)
    return res


def getLines(imgSrc, minLineLenght, maxLineGap):
    lines = cv2.HoughLinesP(imgSrc, 1, np.pi/180, 127,
                            minLineLength=minLineLenght, maxLineGap=maxLineGap)
    lines = sorted(lines, key=sortLine, reverse=True)[:100]
    return lines


def getFrame(original_img):
    image = autoBrightnessContrast(original_img)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = getHoughLine(gray)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel)
    kernel = np.ones((7, 7), np.uint8)
    image = cv2.erode(image, kernel)
    lines = getLines(image, 100, 10)
    len_lines = len(lines)
    while (True):
        lines = mergeLines(lines, original_img)
        if len_lines == len(lines):
            break
        len_lines = len(lines)
    res = np.zeros(original_img.shape[:2], np.uint8)
    print(len_lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(original_img, (x1, y1), (x2, y2), 255, 2)
        cv2.line(res, (x1, y1), (x2, y2), 255, 2)
    return image, original_img, res, lines
