import os
import re
import sys
import argparse
from os import listdir
from os.path import isfile, join
import time
import random
from PIL import Image
import pytesseract 
from pytesseract import Output
import cv2
import numpy as np
import pandas as pd
import imutils
import coverpy
import requests
import matplotlib.pyplot as plt

'''
from imutils.object_detection import non_max_suppression
import argparse

############################################################################################################
#
# This part of the program was trying to identify the relevant text region by using OpenCV text finder. 
# While the text finder worked, it was ultimately the ability to consitently classify the album-field as the 
# largest text area in a given topster that made the tesseract approach more successful. 
#
# Should it be interesting to pick up the OpenCV part of the project again, it currently requires a specified
# image as input, along with an LSTM model used for detection (in this case EAST), which currently sits in 
# the game's home-directory. It also takes two optional arguments. One sensitivity parameter for detection, 
# and dimension specifications (must be multiple of 32). These are also another reason for the preference of
# using tesseract. 
#
############################################################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

        # loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue
		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]
		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)
		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	# draw the bounding box on the image
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output image
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)

'''

def detectEdges(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)[1]
    imSblur = cv2.resize(blurred, (1080,720))
    #cv2.imshow("blurred",imSblur)
    plt.imshow(imSblur, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    #cv2.waitKey(0)
    imSthresh = cv2.resize(thresh, (1080,720))
    plt.imshow(imSthresh, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    #cv2.imshow("thresh",imSthresh)
    #cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Found",len(cnts),"contours")
    
    min_area = 5000
    max_area = 1500000
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+h]
            cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
            image_number += 1

    #cv2.destroyAllWindows()

    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # create a binary thresholded image on hue between red and yellow
    #lower = (0,240,160)
    #upper = (30,255,255)
    #thresh = cv2.inRange(hsv, lower, upper)

    # apply morphology
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    #clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    #clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # create a binary thresholded image on hue between red and yellow
    lower = (0,240,160)
    upper = (30,255,255)
    thresh = cv2.inRange(hsv, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # get external contours
    contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    result1 = img.copy()
    result2 = img.copy()
    print(len(contours))
    for c in contours:
        cv2.drawContours(result1,[c],0,(36,255,12),2)
        # get rotated rectangle from contour
        rot_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        # draw rotated rectangle on copy of img
        cv2.drawContours(result2,[box],0,(36,255,12),2)

    # save result
    cv2.imwrite("4cubes_thresh.jpg",thresh)
    cv2.imwrite("4cubes_clean.jpg",clean)
    cv2.imwrite("4cubes_result1.png",result1)
    cv2.imwrite("4cubes_result2.png",result2)

    # display result
    cv2.imshow("thresh", thresh)
    cv2.imshow("clean", clean)
    cv2.imshow("result1", result1)
    cv2.imshow("result2", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    print("found:",cnts)

    min_area = 100
    max_area = 1500
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+h]
            cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            image_number += 1

    cv2.imshow('sharpen', sharpen)
    cv2.waitKey()
    cv2.imshow('close', close)
    cv2.waitKey()
    cv2.imshow('thresh', thresh)
    cv2.waitKey()
    cv2.imshow('image', image)
    cv2.waitKey()
    '''
    '''
    # convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # perform edge detection
    edges = cv2.Canny(grayscale, 30, 100)
    
    # detect lines in the image using hough lines technique
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 3)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    '''

def findAlbumBox(df):
    for index, row in df.iterrows():
        if int(float(row["conf"])) > 5:
            return int(row["left"]),int(row["top"]),int(row["width"]),int(row["height"])

'''
This method takes text elements and strips them of specific elements of contamination
which may result from differently formatted topsters, such as numbered lists. It does
this by matching against several regex patterns. This could be done more elegantly if
a dictonary is used to hold regex patterns as keys as substituted token as value, but
this is problematic when keys are overlapping. 
'''


def multiple_re(text):
    pat1 = r'^\.\s'
    pat2 = r'^(\d+)\.\s'
    pat3 = r'^\:\s'
    pat4 = r'^\・\s'
    master_pat = r'|'.join((pat1, pat2, pat3, pat4))
    return re.sub(master_pat, '', text)

parser = argparse.ArgumentParser(description="option for running Randopsters with different data sets")
parser.add_argument('--path', dest="path", type=str, default="valid", help="Run Randopster with one of four datasets: valid, dev, test, or fail")

args = parser.parse_args()

print("Initiating random music inspiration game!\n")
path = "topsters/"+args.path+"_samples/" #directory .../test_samples/ or .../valid_samples/ or .../dev_samples/  depending on use
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

index = random.randint(0,len(onlyfiles)-1)
image = Image.open(path+onlyfiles[index])

cv_image = cv2.imread(path+onlyfiles[index])

rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
df = pd.DataFrame(results, columns=["left","top","width","height","text","conf"])
df["area"] = df["width"]*df["height"]
df = df.sort_values("area",ascending=False)

x,y,w,h = findAlbumBox(df)


#Use this to check where text is detected, everything right of the green box visualized here is considered text area

#cv2.namedWindow("Topster", cv2.WINDOW_NORMAL)
cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
imS = cv2.resize(cv_image, (1240,720))
plt.imshow(imS, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imshow("Topster", imS)
#cv2.waitKey(0)


width, height = image.size
#print(width, height)

left = x+w #crops at the end of the album-field to avoid confusion and restrict search area
top = 0
right = width
bottom = height

right_crop = image.crop((left,top,right,bottom))
right_crop.save("topsters/right_crop.png","PNG")
left_crop = image.crop((top,top,left,bottom))
left_crop.save("topsters/left_crop.png","PNG")
detectEdges(cv2.imread("topsters/left_crop.png"))

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

print("Scanning image for text.\n")

string=pytesseract.image_to_string("topsters/right_crop.png", lang = "eng+jpn")
#print(pytesseract.image_to_data("topsters/right_crop.png", lang = "eng"))
#print(pytesseract.image_to_data("topsters/right_crop.png", lang = "eng+jpn"))
string_list= string.split("\n")
fillist = list(filter(None,string_list))
#fillist.remove("\x0c")
#fillist[:] = [multiple_re(x) for x in fillist if x != " " and len(x)>3 and ('-' in x or '—' in x)]
fillist[:] = [multiple_re(x) for x in fillist if x != " " and len(x)>3 and ('-' in x or '—' in x or 'ー' in x)]

for x in fillist:
    print(x)

#When testing use these to check that the detected text makes sense and contains as many recommendations as expected

#print(fillist)
print("Detected", len(fillist),"albums for in topster.\n")

os.remove("topsters/right_crop.png")
os.remove("topsters/left_crop.png")

flag = True
coverpy = coverpy.CoverPy()
limit = 1
while(flag):
    index = random.randint(0,len(fillist)-1)
    recName = fillist[index]
    print("Your recommendation is:", recName)
    try:
        result = coverpy.get_cover(recName, limit)
        # Set a size for the artwork (first parameter) and get the result url.
        print(result.name)
        print(result.artwork(250))
        response = requests.get(result.artwork(500))
        file = open("topsters/download.jpg", "wb")
        file.write(response.content)
        file.close()
        
        rec = Image.open("topsters/download.jpg")
        rec.show()
        #rec = cv2.imread("/home/stautis/Documents/projects/music_inspo_game/download.jpg")
        #cv2.imshow("Recommendation",rec)
    except:
        e = sys.exc_info()[0]
        print("Could not display image for given recommendation")
    val = input("Are you satisfied with your recommendation? [y/n]: ")
    if val == "y":
        if 'rec' in locals():
            rec.close()
            os.remove("topsters/download.jpg")
        break
    else:
        continue

print("Thanks for using Randopster.")

