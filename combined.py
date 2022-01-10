import cv2
import numpy as np

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 700
params.maxArea = 1550

# Filter by Circularity
params.filterByCircularity = False

#Filter by Color
params.filterByColor = False

# Filter by Convexity
params.filterByConvexity = False

# Filter by Inertia
params.filterByInertia = False

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else :
	detector = cv2.SimpleBlobDetector_create(params)

# Read the image using imread function
image = cv2.imread('img.png')

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


h,w = img_gray.shape
ratio = int(h) / int(w)

# let's upscale the image using new  width and height
up_width = 600
up_height = int(up_width * ratio)
up_points = (up_width, up_height)
resized_up = cv2.resize(img_gray, up_points, interpolation=cv2.INTER_LINEAR)
cv2.waitKey(0)


h,w,c = image.shape
print("Original Height and Width:", h,"x", w)

# Detect blobs.
keypoints = detector.detect(resized_up)

for keypoint in keypoints:
	pixelX = int(keypoint.pt[0])
	pixelY = int(keypoint.pt[1])
	cmX = pixelX * 0.0264583333
	cmY = pixelY * 0.0264583333
	print("{:.2f}, {:.2f}".format(cmX, cmY))

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(resized_up, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)