import cv2 as cv
import numpy as np
import os
import pyautogui as gui
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import ImageFont, ImageDraw, Image

global ss

# set
display_text = False
display_chart = False



def detect(img, cascade):
    # detectMultiScale: Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def removeFaceAra(img, cascade):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # converts images from BGR to Gray
    gray = cv.equalizeHist(gray)  # stretches the histogram to either ends to improve the contrast of the image
    rects = detect(gray, cascade)

    height, width = img.shape[:2]

    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

    return img


def make_mask_image(img_bgr):
    img_hsv = cv.cvtColor(img_bgr,
                          cv.COLOR_BGR2HSV)  # to extract a colored object from BGD to HSV (Hue Saturation Value)

    # img_h,img_s,img_v = cv.split(img_hsv)

    low = (0, 30, 0)  # black in HSV
    high = (15, 255, 255)  # near red in HSV

    # # Threshold the HSV image to get colors from the range (low, high)
    img_mask = cv.inRange(img_hsv, low, high)
    return img_mask


def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def calculateAngle(A, B):
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
    return angle


def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv.contourArea(contour)

        x, y, w, h = cv.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    return max_area, max_contour


def getFingerPosition(max_contour, img_result, debug):
    points1 = []

    # STEP 6-1
    M = cv.moments(max_contour)
    # Center of contour is m10/m00, m01/m00
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # approxPolyDP approximate a curve or a polygon with another curve/polygon with less vertices so that the distance between them is less or equal to the specified precision
    max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
    hull = cv.convexHull(max_contour)  # find convex frame of a point set

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    if debug:
        cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

    # STEP 6-2
    hull = cv.convexHull(max_contour, returnPoints=False)
    defects = cv.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

    # STEP 6-3
    points = points1 + points2
    points = list(set(points))

    # STEP 6-4
    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)

    return 1, new_points


Flag = [0, 0, 0, 0, 0]
f = open("letter.txt", "w+", encoding="utf-8")
tt = []
global ss


def listToString(s):
    str1 = " "
    return (str1.join(s))


b, g, r, a = 255, 0, 0, 0
br, gr, rr, ar = 100, 200, 200, 0
fontpath = "fonts/gulim.ttc"
# fontpath = "fonts/batang.ttc"
font = ImageFont.truetype(fontpath, 35)


def process(img_bgr, debug):
    img_result = img_bgr.copy()

    # STEP 1
    img_bgr = removeFaceAra(img_bgr, cascade)

    # STEP 2
    img_binary = make_mask_image(img_bgr)

    # STEP 3: all the pixels near boundary will be discarded depending upon the size of kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Elliptical Kernel
    # MORPH_CLOSE: Dilation followed by Erosion; useful in closing small holes inside the foreground objects, or small black points on the object.
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    cv.imshow("Binary", img_binary)

    # STEP 4
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)  # finds contours in a binary image where each contour is stored as a vector of points

    if debug:
        for cnt in contours:
            cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)

            # STEP 5
    max_area, max_contour = findMaxArea(contours)
    # ss = listToString(tt)
    # cv.putText(img_result, ss, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    if max_area == -1:
        return img_result

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)

        # STEP 6
    ret, points = getFingerPosition(max_contour, img_result, debug)


# 손가락 반짝이는 부분
    if points != None:
        for point in points:
            for i in range(5):
                if keys[i].contains(Point(point)):
                    Flag[i] += 1
                    if Flag[i] % 10 == 9: ##############################################timer
                        if i ==15:
                            print('next')
                        print(menu[i])
                        f.write(menu[i] + " ")
                        tt.append(menu[i])
                        # gui.typewrite(F[i] + " ")
                        # for i in range(5):
                            # cv.circle(img_result, point, 20, [255, 255, 255], -1)
                else:
                    Flag[i] = 0
# 손가락 반짝이는 부분 종료


    # STEP 7
    # if ret > 0 and len(points) > 0:
        # for point in points:
            # cv.circle(img_result, point, 20, [255, 0, 255], 5)

    return img_result


current_file_path = os.path.dirname(os.path.realpath(__file__))
cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 20000)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 900)
# width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # width of video captured by the webcam
# height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # height of the video captured by the webcam
# max_keys_in_a_row = 10  # max number of keys in any row is 10 i.e the first row which contains qwertyuiop
# key_width = int(width / max_keys_in_a_row)  # width of one key. width is divided by 10 as the max number of keys in a single row is 10


x1 = 0
y1 = 0
x2 = 1280
y2 = 720

# 전체 가로
width = x2 - x1
# 전체 세로
height = y2 - y1

keys_points_list = []
div = 4

for j in range(div):
    for i in range(div):
        # x1, y1, x2, y2
        keys_points_list.append([width // div * i, height // div * j, width // div * (i + 1), height // div * (j + 1)])

# key 좌표
print(keys_points_list)

# keys = [Polygon([(key_x1[i], key_y1[i]), (key_x2[i], key_y1[i]), (key_x2[i], key_y2[i]), (key_x1[i], key_y2[i])]) for i
#         in range(div * div)]

# 손가락 위치에 따른 event 를 위한 position
keys = [Polygon([(p[0], p[1]), (p[2], p[1]), (p[2], p[3]), (p[0], p[3])]) for p in keys_points_list]
menu = list(str(i) for i in range(div * div))

##################################################################################################
menu[0]= '아메리카노'
menu[1]= '카페라떼'
menu[2]='카페모카'
menu[3]='카푸치노'
menu[4]='초코라떼'
menu[5]='고구마라떼'
menu[6]='녹차라떼'
menu[7]='홍차라떼'
menu[8]='요거트스무디'
menu[9]='딸기스무디'
menu[10]='레몬에이드'
menu[11]='자몽에이드'
menu[12]='<<'
menu[13]='초기화'
menu[14]='주문하기'
menu[15]='>>'
#############################################################################################################

while True:
    ret, img_bgr = cap.read()
    img_bgr = cv.flip(img_bgr, 1)  # 이미지 반전 처리리
    # row_keys_points = get_keys()
    # for key in row_keys_points:
    #     cv.putText(img_bgr, key[0], key[3], cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    #     cv.rectangle(img_bgr, key[1], key[2], (0, 255, 0), thickness=2)
    # if ret == False:
    #     break
#########################################################################################################################debug controller
    img_result = process(img_bgr, debug=False)
    # cv.rectangle(img_result, (x1, y1), (x2, y2), (255, 255, 0), 4)

    img_pil = Image.fromarray(img_result)
    draw = ImageDraw.Draw(img_pil)
    if display_text:
        po =0
        for p in keys_points_list:
            draw.text((p[0] + 10, p[1] +50), menu[po], font=font, fill=(b, g, r, a))
            po+=1

    img_result = np.array(img_pil)
    if display_chart:
        for p in keys_points_list:
            cv.rectangle(img_result, (p[0], p[1]), (p[2], p[3]), (255, 255, 255), 4) # ***********************이부분 고쳐서 색바꿈

    key = cv.waitKey(1)
    if key == 27:
        break
    img_text = np.zeros((500, 500, 3), np.uint8)
    img_pil = Image.fromarray(img_text)
    draw = ImageDraw.Draw(img_pil)
    ss = listToString(tt)
    draw.text((50, 50), ss, font=font, fill=(br, gr, rr, ar))
    img_text = np.array(img_pil)

    # ss = listToString(ss)
    # cv.putText(img_text, "{}".format(ss), (10, 30), cv.FONT_HERSHEY_DUPLEX, 1,
    # (100, 200, 200), 2)

    cv.imshow("Result", img_result)
    # cv.imshow("Text", img_text)

cap.release()
cv.destroyAllWindows()
