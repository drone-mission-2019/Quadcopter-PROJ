import numpy as np
import cv2
import math
import vrep
from api import *

def left_transformation(dimension_array, transplant, x, y, z):
    alpha, beta, gama = dimension_array
    src_coordinates = np.array([x, y, z])

    Rx = np.array([[math.cos(alpha), math.sin(alpha), 0.],
                    [-math.sin(alpha), math.cos(alpha), 0],
                    [0., 0., 1.]])

    Ry = np.array([[math.cos(beta), 0., math.sin(beta)],
                    [0., 1., 0.],
                    [-math.sin(beta), 0., math.cos(beta)]])

    Rz = np.array([[1., 0., 0.],
                    [0., math.cos(gama), math.sin(gama)],
                    [0., -math.sin(gama), math.cos(gama)]])

    R = np.dot(Rx, Ry)
    R = np.dot(R, Rz)
    target_coordinates = np.dot(R, src_coordinates) + transplant
    return target_coordinates 

def trivial_transformation(clientID, x, y, z):
    relative_pos = [x, y, z]
    vrep.simx
    



def toRadians(args):
    return math.radians(args)

def zedDistance(clientID, zed1, zed0):
    """
    给出图像给定位置, 返回任务一中二维码的中心位置
    :param clientID: 远程控制 v-rep 的客户端ID
    :param zed1: 1号相机拍摄的画面
    :type  ndarrays
    :param zed0: 0号相机拍摄的画面
    :type  ndarrays

    :return: 目标位置，相对于世界坐标系
    :rtype: [x, y, z] with respect to the world
    """
    B = 0.12
    P_x = 1280.0
    P_y = 720.0
    alpha = 85.0
    beta = 54.0

    x_l = 0.0
    x_r = 0.0
    y_p = 0.0

    x0, y0 = get_qr_code(zed0)
    x1, y1 = get_qr_code(zed1)
    x_l = x1 - P_x / 2
    x_r = x0 - P_x / 2
    y_p = P_y / 2 - (y0+y1) / 2

    alpha_rad = toRadians(alpha)
    beta_rad = toRadians(beta)
    x = (B * x_l) / (x_l - x_r)
    y = (B * P_x * math.tan(beta_rad / 2) * y_p) / ((x_l - x_r) * P_y * math.tan(alpha_rad / 2))
    z = (B * P_x / 2) / ((x_l - x_r) * math.tan(alpha_rad / 2))
    print([x, y, z])


if __name__ == '__main__':
    img0 = cv2.imread('5zed0.jpg')
    img1 = cv2.imread('5zed1.jpg')
    
    zedDistance(0, img1, img0)