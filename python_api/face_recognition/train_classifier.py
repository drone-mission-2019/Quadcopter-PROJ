import cv2
import numpy as np
from scipy.spatial.distance import pdist
from skimage.measure import compare_ssim
import dlib
import sys
import os

person1 = cv2.imread('../pictures/face_1.png')
person2 = cv2.imread('../pictures/face_2.png')
person3 = cv2.imread('../pictures/face_3.png')
person4 = cv2.imread('../pictures/face_4.png')
person5 = cv2.imread('../pictures/face_5.png')
person6 = cv2.imread('../pictures/face_6.png')

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

def resize_image(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

def mahalanobis_distance(vec0, vec1):
    combo = np.array([vec0, vec1])
    return pdist(combo, 'mahalanobis')

def edge_detector(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    display_image(lap_gray)

def face_detector_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lap = cv2.Laplacian(img,cv2.CV_64F)#拉普拉斯边缘检测 
    # lap = np.uint8(np.absolute(lap))##对lap去绝对值
    # lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.08, 2)
    return faces

def face_detector_dlib(img):
    detector = dlib.get_frontal_face_detector()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dets = detector(img, 1)
    return dets

def face_selector(p, file):
    path = p + file
    img = cv2.imread(path)
    checkin = False
    angles = 30
    while True:
        angles -= 30
        processed_img = rotate_image(img, angles)
        # faces = face_detector_dlib(processed_img)
        # for index, face in enumerate(faces):
        #     checkin = True
        #     left = face.left()
        #     top = face.top()
        #     right = face.right()
        #     bottom = face.bottom()
        #     cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 255, 0), 3)
        faces = face_detector_opencv(processed_img)
        for (x, y, w, h) in faces:
            checkin = True
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0))
        if checkin == True or angles <= -360:
            break

    if checkin == True:
        print(file)
        print(faces)
        cv2.imwrite('../pictures/exists_face/' + file, processed_img)
        return faces
    else:
        print(file)
        return [[0, 0, 0, 0]]

def find_peopleID(img):
    pass

if __name__ == '__main__':
    img = cv2.imread('../pictures/face_2.png')
    people_2 = cv2.imread('../pictures/people2.jpeg')
    people_5 = cv2.imread('../pictures/people_5.jpeg')
    faces = face_selector('../pictures/', 'people2.jpeg')

    x, y, w, h = faces[0]
    face_2 = resize_image(img, w, h)
    people_2 = people_2[y:y+h, x:x+w, :]
    cv2.imwrite('../pictures/batch_2.jpg', people_2)
    ssim = compare_ssim(face_2, people_2, multichannel=True)

    faces = face_selector('../pictures/', 'people_5.jpeg')
    x, y, w, h = faces[0]
    face_2 = resize_image(img, w, h)
    people_5 = people_5[y:y+h, x:x+w, :]
    cv2.imwrite('../pictures/batch_5.jpg', people_5)
    ssim0 = compare_ssim(face_2, people_5, multichannel=True)
    
    print(ssim)
    print(ssim0)

