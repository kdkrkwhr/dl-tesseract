import numpy as np
import cv2
import datetime
from copy import deepcopy
from PIL import Image
import pytesseract as tess
import re

def preprocess(img):
	config = ('--psm 7')
	imgBlurred = cv2.GaussianBlur(img, (5, 5), 0)
	gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

	text = tess.image_to_string(img, lang='eng', config=config)
	canny = cv2.Canny(gray, 100, 100)
	ret2, threshold_img = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow("img", img)
	return threshold_img

def cleanPlate(plate): # Last 처리
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

	im1, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# cv2.imshow("threshold_img22", plate)

	if contours:
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)

		max_cnt = contours[max_index]
		max_cntArea = areas[max_index]

		x, y, w, h = cv2.boundingRect(max_cnt)
		# if not ratioCheck(max_cntArea, w, h):
		#
		# 	return plate, None

		cleaned_final = thresh[y: y+h, x: x+w]

		return cleaned_final, [x, y, w, h]
	else:
		return plate, None

def extract_contours(threshold_img):
	element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
	# cv2.imshow('threshold_img', threshold_img)

	morph_img_threshold = threshold_img.copy()
	cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
	
	im2, contours, hierarchy = cv2.findContours(morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
	return contours

def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272

	min = 15 * aspect * 15  # minimum area
	max = 125 * aspect * 125  # maximum area

	rmin = 3
	rmax = 6

	# if (area < min or area > max):
	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False

	return True

def isMaxWhite(plate):
	avg = np.mean(plate)

	if (avg >= 50):
		return True
	else:
		return False

def validateRotationAndRatio(rect): # 우선 처리
	(x, y), (width, height), rect_angle = rect

	if(width > height):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle

	if angle > 15:
		return False

	if height == 0 or width == 0:
		return False

	area = height * width
	if not ratioCheck(area, width, height):
		return False
	else:
		return True

def cleanAndRead(img, contours, cnt2):	# Main
	config = ('--psm 4')
	for i, cnt in enumerate(contours):

		min_rect = cv2.minAreaRect(cnt)
		# print('min_rect : ', min_rect)
		if validateRotationAndRatio(min_rect):  # 1차 검출 2개
			x, y, w, h = cv2.boundingRect(cnt)
			# x = x + 30
			# h = h - 10
			# w = w - 15

			plate_img = img[y: y + h, x: x + w]
			if (isMaxWhite(plate_img)):
				clean_plate, rect = cleanPlate(plate_img)  # 2차 검출
				if rect:
					if True:
						cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)

						cnt2 = cnt2 + 1
						# cv2.imshow("Cleaned Plate", plate_img)
						text = tess.image_to_string(clean_plate, lang='eng', config=config)
						text2 = re.sub('[-=.#/?:_$}]', '', text)
						cv2.putText(img, str(text2), (x, y - 10), cv2.FONT_ITALIC, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
						# cv2.imwrite("images/video/" + str(cnt2) + ".jpg", plate_img)

						# print("TEXT : ", text)
						print("TEXT : ", text2)
						cv2.imshow("img", img)

if __name__ == '__main__':
	# cap = cv2.VideoCapture('video/test1.mp4')
	cnt2 = 0
	img = cv2.imread("testData/success/1.jpg")  # 이미지 Input
	# img = cv2.imread("testData/0419/notdenmed.png")
	threshold_img = preprocess(img)
	contours = extract_contours(threshold_img)
	cleanAndRead(img, contours, cnt2)
	cv2.waitKey(0)