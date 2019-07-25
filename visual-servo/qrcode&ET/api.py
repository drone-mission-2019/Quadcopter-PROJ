from utils import *
import qr_code


#
# API to detect QR code in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   flag: boolean
#       Whether the QR code is detected.
#   center: tuple(int, int)
#       The center coordinate of the QR code.
#
def get_qr_code(img):
    return qr_code.get_qr_code(img)


#
# API to detect cylinder im images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   flag: boolean
#       Whether the cylinder is detected.
#   pos: tuple(int, int)
#       The position of the cylinder.
#
def get_cylinder(img):
    if img.shape[0] < 20 or img.shape[1] < 20:
        return False, (0, 0)
    img2 = np.zeros((img.shape[0], img.shape[1]))
    t1 = img[:, :, 0] / 2 - img[:, :, 1]
    t2 = img[:, :, 0] / 2 - img[:, :, 2]
    img2[(t1 > 0) * (t2 > 0) * (img[:, :, 0] > 150)] = 255
    L = 10
    aa = img2.sum(axis=1)
    mm = -1e8
    that1 = (0, 0)
    for i in range(img.shape[0]):
        l = i
        r = i + L
        if r > img.shape[0]:
            break
        tmp = aa[l: r].sum()
        if tmp > mm:
            mm = tmp
            that1 = (l, r)
    bb = img2.sum(axis=0)
    mm = -1e8
    that2 = (0, 0)
    for i in range(img.shape[1]):
        l = i
        r = i + L
        if r > img.shape[1]:
            break
        tmp = bb[l: r].sum()
        if tmp > mm:
            mm = tmp
            that2 = (l, r)
    if that1 == (0, 0) or that2 == (0, 0):
        return False, (0, 0)
    x = int((that1[0] + that1[1]) / 2)
    y = int((that2[0] + that2[1]) / 2)
    return True, (y, x)


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


if __name__ == '__main__':
    from utils import *
    img = read_image('testcase/cylinder.jpg')
    print(img.shape)
    flag, pos = get_cylinder(img)
    print(flag, pos)
