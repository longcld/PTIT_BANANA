import cv2
import numpy as np
# lower_blue = np.array([95, 50, 0])
# upper_blue = np.array([110, 135, 255])

lower_blue = np.array([95, 50, 0])
upper_blue = np.array([109, 255, 130])


# def preprocess(source_image, min_threshold=50, max_threshold=80):
#     gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)  # convert image into gray scale
#     gaussian = cv2.GaussianBlur(gray, (5, 5), 0)  # apply gaussian blur
#     canny = cv2.Canny(gaussian, min_threshold, max_threshold)
#     return canny, gaussian, gray
#
#
# def getRects(source_image, canny_image, min_points=5,
#                   axes_ratio=1.5, minor_axes_ratio=25, major_axes_ratio=15):
#     # declaring variables
#     i = 0
#     height, width, channels = source_image.shape
#     ellipse_list = []
#     rects = []
#
#     # find all the contours
#     contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     number_of_contours = len(contours)
#
#     # finding and filtering ellipses
#     while i < number_of_contours:
#         if len(contours[i]) >= min_points:
#             ellipse = cv2.fitEllipse(contours[i])
#             (x, y), (minor_axis, major_axis), angle = ellipse
#             if minor_axis != 0 and major_axis != 0 and major_axis / minor_axis <= axes_ratio:
#                 ellipse_min_ratio = width / minor_axis
#                 ellipse_maj_ratio = height / major_axis
#
#                 if minor_axes_ratio >= ellipse_min_ratio >= 1.5 and major_axes_ratio >= ellipse_maj_ratio >= 1.5:
#                     rect = cv2.boundingRect(contours[i])
#                     if (10 < rect[-1] < 40) and (10 < rect[-2] < 40) and (rect[0] > 120):
#                         rects.append(rect)
#
#         i += 1
#
#     return rects

def getRects(img):
    rects = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(conts)):
        (x, y, w, h) = cv2.boundingRect(conts[i])
        # if (70 > rect[-2] > 10 and 70 > rect[-1] > 10 and abs(rect[-2] - rect[-1]) < 10) and rect[0] > 140 and rect[1] > 50:
        if (10 < w < 40 and 10 < h < 40 and abs(w - h) < 10 and y > 30 and x > 120):
            # cv2.rectangle(img, (x, y), (x + w, y + h), [0, 255, 0], 1)
            rects.append((x, y, w, h))
    return rects
