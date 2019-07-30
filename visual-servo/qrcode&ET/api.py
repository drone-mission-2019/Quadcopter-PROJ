from utils import *
import qr_code
from queue import Queue


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
#   flag: boolean
#       Whether E or T is detected.
#   results: tuple(tuple(int, int, int, int), str)
#       The first tuple is (min_x, min_y, max_x, max_y).
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
    if len(that) < 0:
        return False, ((0, 0, 0, 0), '?')
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
    return True, ((min_x, min_y, max_x, max_y), ans)


#
# API to detect target(T) and end(E) in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#   eye: boolean
#       eye=True: left eye.
#       eye=False: right eye.
#
# Returns:
#   people: list of tuple(int, int)
#       Each tuple represent a coordinate of people.
#
def get_people(img, eye):
    fuck1 = np.array([172.87, 182.88, 188.63])
    fuck2 = np.array([189.49, 198.48, 204.345])
    img2 = img.copy()
    tmp = (img2 < 2).sum(axis=2)
    img2[tmp == 3] = [255, 255, 255]
    img2[np.abs(img2 - fuck1).max(axis=2) < 20] = [255, 255, 255]
    img2[np.abs(img2 - fuck2).max(axis=2) < 20] = [255, 255, 255]
    tmp = img2[:, :, 0].astype(np.int64) + img2[:, :, 1] + img2[:, :, 2]
    img2[tmp > 700] = [255, 255, 255]
    img2 = cv2.resize(img2, (320, 180))
    # print(img2.shape)
    if eye:
        img2[:30, 275:] = [255, 255, 255]
        img2[:60, :35] = [255, 255, 255]
    else:
        img2[:30, 125:170] = [255, 255, 255]
        img2[:30, 300:] = [255, 255, 255]
    show_image(img2)
    visit = np.zeros((img2.shape[0], img2.shape[1]))
    belong = np.zeros(visit.shape)
    cnt = 0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if not visit[i, j] and (img2[i, j] == 255).sum() < 3:
                Q = Queue()
                Q.put((i, j))
                visit[i, j] = 1
                cnt += 1
                belong[i, j] = cnt
                dx = [-1, 0, 0, 1]
                dy = [0, -1, 1, 0]
                while not Q.empty():
                    x, y = Q.get()
                    for k in range(4):
                        xx = x + dx[k]
                        yy = y + dy[k]
                        if xx < 0 or xx >= img2.shape[0] or yy < 0 or yy >= \
                                img2.shape[1]:
                            continue
                        if visit[xx, yy] == 1 or (
                                img2[xx, yy] == 255).sum() == 3:
                            continue
                        visit[xx, yy] = 1
                        belong[xx, yy] = cnt
                        Q.put((xx, yy))
    points = [[] for i in range(cnt + 1)]
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if belong[i, j] > 0:
                points[int(belong[i, j])].append((i, j))
    img3 = img2.copy()
    img3[:, :] = [255, 255, 255]
    threshold = 200
    new_points = []
    for pp in points:
        if len(pp) > threshold:
            new_points.append(pp)
            for p in pp:
                img3[p[0], p[1]] = img2[p[0], p[1]]
    show_image(img3)
    ans = []
    for pp in new_points:
        x = np.array([t[0] for t in pp]).mean()
        y = np.array([t[1] for t in pp]).mean()
        ans.append((int(y * 4 + 2), int(x * 4 + 2)))
    return ans


if __name__ == '__main__':
    img = read_image('testcase/zuoyan.jpeg')
    print(get_people(img, True))
