import numpy as np
import math
from utils import *
from pyzbar.pyzbar import decode


def get_dist(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def get_norm(A):
    return math.sqrt(A[0] ** 2 + A[1] ** 2)


def check_square(points):
    points = list(set(points))
    n = len(points)
    if n < 4:
        return False, ((0, 0), 0, [])
    # print(points)
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    x0 = (x.max() + x.min()) / 2
    y0 = (y.max() + y.min()) / 2
    if x.max() - x.min() < 10 or y.max() - y.min() < 10:
        return False, ((0, 0), 0, [])
    error = min(x.max() - x.min(), y.max() - y.min()) * 0.2
    dist = [(get_dist(point, (x0, y0)), point) for point in points]
    dist.sort(reverse=True)
    # print(dist)
    vertices = []
    for d, point in dist:
        flag = True
        for vertex in vertices:
            if get_dist(vertex, point) < error:
                flag = False
        if not flag:
            continue
        vertices.append(point)
        if len(vertices) == 4:
            break
    # print(vertices)
    new_vertices = [vertices[0]]
    while len(new_vertices) < 4:
        P = new_vertices[-1]
        that = 0
        mm = 1e9
        for vertex in vertices:
            dd = get_dist(vertex, P)
            if dd < mm and vertex not in new_vertices:
                that = vertex
                mm = dd
        new_vertices.append(that)
    vertices = new_vertices
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[0] if i == 3 else vertices[i + 1]
        v1 = (p1[0] - x0, p1[1] - y0)
        v2 = (p2[0] - x0, p2[1] - y0)
        t1 = math.sqrt(abs(v1[0] * v2[0] + v1[1] * v2[1]))
        if t1 > error * 2:
            # pass
            return False, ((0, 0), 0, [])
    # print(vertices)
    m = get_dist(vertices[0], vertices[1])
    n = get_dist(vertices[1], vertices[2])
    if m > n:
        m, n = n, m
    if m * 1.5 < n:
        return False, ((0, 0), 0, [])
    for point in points:
        flag = False
        for i in range(4):
            p = vertices[i]
            q = vertices[0] if i == 3 else vertices[i + 1]
            v = (p[0] - q[0], p[1] - q[1])
            w = (p[0] - point[0], p[1] - point[1])
            d = abs((v[0] * w[1] - v[1] * w[0]) / get_norm(v))
            if d < error:
                flag = True
        if not flag:
            return False, ((0, 0), 0, [])
    return True, ((x0, y0), (m + n) / 2.0, vertices)


def fuck(now):
    mm = -1e9
    that = 0
    for i in range(3):
        p1 = now[i]
        p2 = now[0] if i == 2 else now[i + 1]
        d = get_dist(p1, p2)
        if d > mm:
            mm = d
            that = (p1, p2)
    return (that[0][0] + that[1][0]) / 2, (that[0][1] + that[1][1]) / 2


def check_all(subimg):
    subimg = 255 - subimg
    aa = subimg.sum(axis=1)
    bb = subimg.sum(axis=0)
    aa_thresh = aa.min() + (aa.max() - aa.min()) * 0.2
    bb_thresh = bb.min() + (bb.max() - bb.min()) * 0.2
    aa[aa <= aa_thresh] = 0
    aa[aa > aa_thresh] = 1
    bb[bb <= bb_thresh] = 0
    bb[bb > bb_thresh] = 1
    aa = aa.reshape(-1, 1)
    subimg = subimg * aa
    subimg = subimg * bb
    aa = (subimg.sum(axis=1) > 0)
    min_x = 0
    while min_x < aa.shape[0] and not aa[min_x]:
        min_x += 1
    max_x = aa.shape[0] - 1
    while not aa[max_x] and max_x >= 0:
        max_x -= 1
    bb = (subimg.sum(axis=0) > 0)
    min_y = 0
    while min_y < bb.shape[0] and not bb[min_y]:
        min_y += 1
    max_y = bb.shape[0] - 1
    while not bb[max_y] and max_y >= 0:
        max_y -= 1
    if max_x - min_x > subimg.shape[0] * 0.8 and max_y - min_y > subimg.shape[1] * 0.8:
        return True
    return False


def in_img(P, img):
    if P[0] < 0 or P[1] >= img.shape[0]:
        return False
    if P[1] < 0 or P[1] >= img.shape[1]:
        return False
    return True


def in_qrcode(P, L, img):
    if not in_img(P, img):
        return False
    subimg = img[int(P[0] - L / 2): int(P[0] + L / 2), int(P[1] - L / 2): int(P[1] + L / 2)]
    tot = subimg.shape[0] * subimg.shape[1]
    black = (subimg == 0).sum()
    if tot == 0:
        return False
    rate = black / tot
    if 0.2 < rate < 0.8:
        return True
    return False


def fuck_all(now2, img):
    L = now2[0][0]
    A = now2[0][1]
    B = now2[1][1]
    mid = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
    v = (-(B[1] - A[1]), B[0] - A[0])
    v_norm = get_norm(v)
    v = (v[0] / v_norm, v[1] / v_norm)
    go_len = get_dist(A, B) / 2
    P = (mid[0] + v[0] * go_len, mid[1] + v[1] * go_len)
    Q = (mid[0] - v[0] * go_len, mid[1] - v[1] * go_len)
    # print(P, Q)
    if in_qrcode(P, L, img):
        return True, P
    if in_qrcode(Q, L, img):
        return True, Q
    if in_img(P, img):
        return True, Q
    if in_img(Q, img):
        return True, P
    return False, ('fuck', 'fuck')


def fuck_all_of_you(now3, img):
    rate = 4.7739
    L, center, vertices = now3[0]
    possible = []
    for vertex in vertices:
        v = (vertex[0] - center[0], vertex[1] - center[1])
        P = (center[0] + rate * v[0], center[1] + rate * v[1])
        possible.append(P)
    for P in possible:
        if in_qrcode(P, L, img):
            return True, P
    num = 0
    for P in possible:
        if in_img(P, img):
            num += 1
    if num == 3:
        for P in possible:
            if not in_img(P, img):
                return True, P
    return False, (0, 0)


def my_method(img):
    gray = get_gray(img)
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    tmp = cv2.blur(th1, (2, 2))
    contours, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    # show_image(tmp, gray=True)
    if check_all(tmp.copy()):
        return True, (img.shape[1] / 2, img.shape[0] / 2)
    possible = []
    new_contours = []
    for contour in contours:
        points = []
        for i in range(contour.shape[0]):
            points.append((contour[i][0][0], contour[i][0][1]))
        flag, (center, L, vertices) = check_square(points)
        if flag and L * 5 < min(img.shape[0], img.shape[1]):
            possible.append((L, center, vertices))
            new_contours.append(contour)
    # img2 = img.copy()
    # cv2.drawContours(img2, new_contours, -1, (0, 255, 0), 1)
    # show_image(img2, figsize=(20, 20))
    if len(possible) == 0:
        return False, (0, 0)
    possible.sort(reverse=True)
    mm = possible[0][0]
    # print(possible)
    now = []
    now2 = []
    now3 = []
    for i, (L, center, vertices) in enumerate(possible):
        if L < mm * 0.9:
            break
        now.append(center)
        now2.append((L, center))
        now3.append((L, center, vertices))
    # print(now3)
    if len(now) == 3:
        return True, fuck(now)
    if len(now) == 2:
        return fuck_all(now2, tmp.copy().T)
    if len(now) == 1:
        return fuck_all_of_you(now3, tmp.copy().T)
    return False, (0, 0)


def get_qr_code(img):
    gray_img = get_gray(img)
    res = decode(gray_img)
    if len(res) == 0:
        flag, center = my_method(img)
        return flag, (int(center[0]), int(center[1]))
    # Currently only use the first detected QR code.
    res = res[0]
    points = []
    for p in res.polygon:
        points.append((p.x, p.y))
    x = np.array([p[0] for p in points]).mean()
    y = np.array([p[1] for p in points]).mean()
    return True, (x, y)


if __name__ == '__main__':
    points = [(633, 381), (634, 380), (635, 380), (636, 380), (637, 380), (638, 380), (639, 380), (640, 380), (641, 380), (642, 380), (643, 380), (644, 380), (645, 380), (646, 380), (647, 380), (648, 380), (649, 380), (650, 380), (651, 380), (652, 380), (653, 380), (654, 380), (655, 381), (655, 382), (655, 383), (655, 384), (655, 385), (655, 386), (655, 387), (655, 388), (655, 389), (655, 390), (655, 391), (654, 392), (653, 392), (652, 392), (651, 392), (650, 392), (649, 392), (648, 392), (647, 392), (646, 392), (645, 392), (644, 392), (643, 392), (642, 392), (641, 392), (640, 392), (639, 392), (638, 392), (637, 392), (636, 392), (635, 392), (634, 391), (633, 390), (633, 389), (633, 388), (633, 387), (633, 386), (633, 385), (633, 384), (633, 383), (633, 382)]
    print(check_square(points))
