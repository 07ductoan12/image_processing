import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

im = cv.imread("page_test.jpg")


def showImg(window_name, image):
    cv.imshow(window_name, cv.resize(image, (720, 1280)))
    cv.waitKey(0)
    cv.destroyAllWindows()


def contour_detectiom(image):
    kernel = np.ones((9, 9), np.uint8)
    blur = cv.GaussianBlur(image.copy(), (7, 7), 0)
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
    dilate = cv.dilate(blur, rectKernel)
    img_morphology = cv.morphologyEx(
        dilate, cv.MORPH_CLOSE, kernel, iterations=3)

    mask = np.zeros(img_morphology.shape[:2], np.uint8)
    newmask = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgdModel, fgdModel = cv.grabCut(
        img_morphology, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_gc = image*mask[:, :, np.newaxis]

    showImg("", img_gc)


contour_detectiom(im)
