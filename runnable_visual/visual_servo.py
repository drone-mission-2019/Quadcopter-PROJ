# -*- coding=utf-8 -*-
import cv2
import numpy as np
import time
import math

def closeLoopCtrl(img1, img2, dimension_array, transplant):
    """闭环控制，给出图像给定位置（相对于摄像机连线中点坐标系），用于没有反馈控制的任务一
    :param img1, img2: 无人机摄像机双目视觉
    :type  img1, img2: ndarrays
    :param demension_array
    :type  [roll, pitch, yaw]
    :param transplant: 摄像机相对于世界坐标系原点平移
    :type  [X, Y, Z]

    :return: 目标位姿，目标位置
    :rtype: [x, y, z] with respect to the camera
    """
    pass

def openLoopCtrl(img1, img2, dimension_array1, transplant1, dimension_array2, transplant2):
    """开环控制，给出图像给定位置（相对于摄像机连线中点坐标系），用于没有反馈控制的任务一
    :param img1, img2: 无人机摄像机双目视觉
    :type  img1, img2: ndarrays
    :param demension_array
    :type  [roll, pitch, yaw]
    :param transplant: 摄像机相对于世界坐标系原点平移
    :type  [X, Y, Z]

    :return: 目标坐标
    :rtype: [x, y, z] with respect to the camera
    """
    # remained to be filled in
    left_camera_matrix = np.array() 
    left_distortion = np.array() 
    right_camera_matrix = np.array()
    right_distortion = np.array()

    om = np.array()
    R = cv2.Rodrigues(om)[0]
    T = np.array() 
    size = (1280, 720)

    #进行立体更正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, \
                                                                  right_camera_matrix, right_distortion, size, R,\
                                                                  T)
    # 计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

    # three dimension coordinate
    threeD = threeDReconstruction(img1, img2, left_map1, left_map2, right_map1, right_map2, Q)



def threeDReconstruction(img1, img2, left_map1, left_map2, right_map1, right_map2, Q):
    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(img1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, right_map1, right_map2, cv2.INTER_LINEAR)
    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # waiting for calibration TO-DO
    num = 3
    blockSize = 8

    stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    #disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)
    return threeD

def transformation_matirix(dimension_array1, transplant1, dimension_array2, transplant2):
    roll1, pitch1, yaw1 = dimension_array1
    X1, Y1, Z1 = transplant1
    roll2, pitch2, yaw2 = dimension_array2
    X2, Y2, Z2 = transplant2

    Rlx = np.array([[1., 0., 0.],
                    [0., math.cos(roll1), -math.sin(roll1)],
                    [0., math.sin(roll1), math.cos(roll1)]])  
    Rly = np.array([[math.cos(pitch1), 0., math.sin(pitch1)],
                    [0., 1., 0.],
                    [-math.sin(pitch1), 0., math.cos(pitch1)]])  
    Rlz = np.array([[math.cos(yaw1), -math.sin(yaw1), 0.],
                    [math.sin(yaw1), math.cos(yaw1), 0.],
                    [0., 0., 1.]]) 
    Rl = np.dot(Rlz, Rly)
    Rl = np.dot(Rl, Rlx)

    Rrx = np.array([[1., 0., 0.],
                    [0., math.cos(roll2), -math.sin(roll2)],
                    [0., math.sin(roll2), math.cos(roll2)]])  
    Rry = np.array([[math.cos(pitch2), 0., math.sin(pitch2)],
                    [0., 1., 0.],
                    [-math.sin(pitch2), 0., math.cos(pitch2)]])  
    Rrz = np.array([[math.cos(yaw2), -math.sin(yaw2), 0.],
                    [math.sin(yaw2), math.cos(yaw2), 0.],
                    [0., 0., 1.]]) 
    Rr = np.dot(Rrz, Rry)
    Rr = np.dot(Rr, Rrx)

    R = np.dot(Rr, Rl.T)
    T = transplant2 - np.dot(R, transplant1)
    return R, T