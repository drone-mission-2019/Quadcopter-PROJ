import cv2

img0 = cv2.imread('zed0.jpg')
img1 = cv2.imread('zed1.jpg')

cv2.namedWindow("Image") 
cv2.imshow("Image", img0) 
cv2.waitKey (0)
cv2.destroyAllWindows()