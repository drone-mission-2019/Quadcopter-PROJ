import os
from pyzbar.pyzbar import decode
from utils import *


#
# API to detect QR code in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   qr_codes: list
#       Each item of the list refers to a QR code in the original image. Each
#       item is a list of four tuple(int, int), which refer to the four vertexes
#       of this QR code rectangle(may be irregular).
#       For example, if there are two QR codes, the qr_codes should be:
#           [   [(0, 0), (10, 0), (10, 10), (0, 10)],
#               [(20, 20), (20, 30), (30, 30), (30, 20)]    ]
#       If qr_codes is empty, it means no QR code is detected from the image.
#
def get_qr_code(img):
    gray_img = get_gray(img)
    res = decode(gray_img)
    if len(res) == 0:
        return False, []
    # Currently only use the first detected QR code.
    qr_codes = []
    for tmp in res:
        points = []
        for p in tmp.polygon:
            points.append((p.x, p.y))
        qr_codes.append(points)
    return qr_codes


#
# API to detect target(T) and end(E) in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   results: list
#       A list of detection results. Each item is the following format:
#           ((min_x, min_y, max_x, max_y), letter)
#       For example:
#           [   ((0, 0, 10, 10), 'T'), ((20, 20, 40, 50), 'E')  ]
#       The above example results means that there's a 'T' and an 'E' in the
#       original image.
#       The letter may be '?', which means it's hard to determine whether it is
#       'T' or 'E'.
#
def get_E_or_T(img):
    gray_img = get_gray(img)
    _, th1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    tmp = cv2.blur(th1, (3, 3))
    contours, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    num = len(contours)
    hh = hierarchy[0]
    that = []
    for i in range(num):
        if hh[i][2] == -1:
            continue
        j = hh[i][2]
        if hh[j][0] != -1 or hh[j][1] != -1:
            continue
        if hh[j][2] == -1:
            continue
        j = hh[j][2]
        if hh[j][0] != -1 or hh[j][1] != -1 or hh[j][2] != -1:
            continue
        that.append(i)
    if len(that) != 1:
        for i in range(10):
            print('Error')
    cc = contours[that[0]]
    min_x = cc.min(axis=0)[0][0]
    min_y = cc.min(axis=0)[0][1]
    max_x = cc.max(axis=0)[0][0]
    max_y = cc.max(axis=0)[0][1]
    new_img = tmp[min_y: max_y + 1, min_x: max_x + 1]
    m, n = new_img.shape
    letter = new_img[m // 5: m - m // 5, n // 5: n - n // 5]
    letter2 = rotate(letter, 180)
    myletter = np.int64(letter) * np.int64(letter2)
    myletter[myletter > 0] = 255
    myletter = np.uint8(myletter)
    cc2, hh2 = cv2.findContours(myletter.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_TC89_L1)
    ans = '?'
    if len(cc2) == 2:
        ans = 'T'
    elif len(cc2) == 4:
        ans = 'E'
    return (min_x, min_y, max_x, max_y), ans
