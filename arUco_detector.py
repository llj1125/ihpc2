# import the necessary packages
import argparse
import imutils
import cv2
import sys

image = cv2.imread("img_5.png")


h,w,c = image.shape
ratio = int(h) / int(w)

# let's downscale the image using new  width and height
down_width = 300
down_height = int(down_width * ratio)
down_points = (down_width, down_height)
image= cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

# Detect the markers.
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image,aruco_dict,parameters=parameters)

out = cv2.aruco.drawDetectedMarkers(image, corners, ids)

markerCentres = []

for marker in corners:
    centreX = (marker[0][0][0] + marker[0][2][0]) / 2
    centreY = (marker[0][0][1] + marker[0][2][1]) / 2
    markerTup = (centreX, centreY)
    markerCentres.append(markerTup)

print(markerCentres)

cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
