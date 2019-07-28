import cv2
import numpy as np
import dlib
import sys
import os

def display_image(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

def rotate_image(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(img, M, (nW, nH))
    return rotated


def edge_detector(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    display_image(lap_gray)

def face_detector_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(img,cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    return faces

def face_detector_dlib(img):
    detector = dlib.get_frontal_face_detector()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img, 1)
    return dets

def face_selector(file):
    path = '../pictures/images/' + file
    img = cv2.imread(path)
    checkin = False
    angles = -30
    while True:
        angles += 30
        processed_img = rotate_image(img, angles)
        faces = face_detector_dlib(processed_img)
        for index, face in enumerate(faces):
            checkin = True
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 255, 0), 3)
        if checkin == True or angles >= 360:
            break

    if checkin == True:
        print(file)
        print(faces)
        cv2.imwrite('../pictures/exists_face/' + file, processed_img)

if __name__ == '__main__':
    path = '../pictures/24zed1.jpg'
    img = cv2.imread(path)
    angles = -120
    processed_img = rotate_image(img, angles)
    faces = face_detector_dlib(processed_img)
    print(processed_img.shape)
    print(faces)
    # checkin = False
    # while True:
    #     angles -= 30
    #     processed_img = rotate_image(img, angles)
    #     faces = face_detector_dlib(processed_img)
    #     for index, face in enumerate(faces):
    #         checkin = True
    #         left = face.left()
    #         top = face.top()
    #         right = face.right()
    #         bottom = face.bottom()
    #         cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 255, 0), 3)
    #     if checkin == True or angles <= -360:
    #         break

    cv2.imwrite('example.png', processed_img)

