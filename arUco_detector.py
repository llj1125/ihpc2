# import the necessary packages
import argparse
import imutils
import cv2
import sys

image = cv2.imread("img.png")


h,w,c = image.shape
ratio = int(h) / int(w)

# let's downscale the image using new  width and height
down_width = 600
down_height = int(down_width * ratio)
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

# Detect the markers.
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(resized_down,aruco_dict,parameters=parameters)

out = cv2.aruco.drawDetectedMarkers(resized_down, corners, ids)

print(corners)

cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()