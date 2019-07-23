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
        return False, ((0, 0), 0)
    # print(points)
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    x0 = (x.max() + x.min()) / 2
    y0 = (y.max() + y.min()) / 2
    if x.max() - x.min() < 10 or y.max() - y.min() < 10:
        return False, ((0, 0), 0)
    error = min(x.max() - x.min(), y.max() - y.min()) * 0.1
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
    # print(vertices)
    m = get_dist(vertices[0], vertices[1])
    n = get_dist(vertices[1], vertices[2])
    if m > n:
        m, n = n, m
    if m * 1.5 < n:
        return False, ((0, 0), 0)
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
            return False, ((0, 0), 0)
    return True, ((x0, y0), (m + n) / 2.0)


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


def my_method(img):
    gray = get_gray(img)
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    tmp = cv2.blur(th1, (2, 2))
    contours, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    possible = []
    new_contours = []
    for contour in contours:
        points = []
        for i in range(contour.shape[0]):
            points.append((contour[i][0][0], contour[i][0][1]))
        flag, (center, L) = check_square(points)
        if flag:
            possible.append((L, center))
        if flag:
            new_contours.append(contour)
    img2 = img.copy()
    cv2.drawContours(img2, new_contours, -1, (0, 255, 0), 1)
    # show_image(img2, figsize=(20, 20))
    if len(possible) == 0:
        return False, (0, 0)
    possible.sort(reverse=True)
    mm = possible[0][0]
    # print(possible)
    now = []
    for i, (L, center) in enumerate(possible):
        if L < mm * 0.9:
            break
        now.append(center)
    # print(now)
    if len(now) == 1:
        return True, now[0]
    if len(now) == 2:
        return True, ((now[0][0] + now[1][0]) / 2, (now[0][1] + now[1][1]) / 2)
    if len(now) == 3:
        return True, fuck(now)
    xx = np.array([p[0] for p in now])
    x = int((xx.max() + xx.min()) / 2)
    yy = np.array([p[1] for p in now])
    y = int((yy.max() + yy.min()) / 2)
    return True, (x, y)


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
