import numpy as np
import cv2
import os, sys
from fpdf import FPDF
from PIL import Image
pdf = FPDF()

if len(sys.argv) != 2:
	print("Usage: python convert.py <relative path of folder>")

base_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(base_path, sys.argv[1])

files = []
for filename in os.listdir(path):
	files.append(os.path.join(path, filename))

files.sort()
converted = []
for image in files:
	try:
		img = cv2.imread(image)
	except:
		print("File", image, "is not image format")
		break
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	white = cv2.inRange(hsv, (0,0,20), (360,50,255))
	white = cv2.GaussianBlur(white, (11,11),0)
	kernel = np.ones((9,9), np.uint8)
	dilated = cv2.dilate(white, kernel)
	eroded = cv2.erode(white, kernel)
	delta  = cv2.absdiff(dilated,eroded)
	kernel2 = np.ones((6,6), np.uint8)
	edges = cv2.erode(delta, kernel2)
	_, ctr, _ = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	hull_list = []
	drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
	for i in range(len(ctr)):
		hull = cv2.convexHull(ctr[i])
		hull_list.append(hull)
	max_hull = max(hull_list, key=cv2.contourArea)
	peri = cv2.arcLength(max_hull, True)
	approx = cv2.approxPolyDP(max_hull, 0.02*peri, True)
	req_cnt = approx
	pts = req_cnt.reshape(4, 2)
	rect = np.zeros((4, 2), dtype="float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	widthA = np.sqrt((rect[0][0] - rect[1][0])**2 + (rect[0][1] - rect[1][1])**2)
	widthB = np.sqrt((rect[2][0] - rect[3][0])**2 + (rect[2][1] - rect[3][1])**2)
	heightA = np.sqrt((rect[0][0] - rect[3][0])**2 + (rect[0][1] - rect[3][1])**2)
	heightB = np.sqrt((rect[1][0] - rect[2][0])**2 + (rect[1][1] - rect[2][1])**2)
	width = max(int(widthA), int(widthB))
	height = max(int(heightA), int(heightB))
	dest = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
	M = cv2.getPerspectiveTransform(rect, dest)
	warp = cv2.warpPerspective(img, M, (width, height))
	warp = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
	warp = cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 10)
	# warp = cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
	converted.append(warp.copy())

for image in converted:
	cv2.imwrite(os.path.join(base_path, "temp.jpg"), image)
	# img = Image.fromarray(image)
	pdf.add_page()
	pdf.image(os.path.join(base_path, "temp.jpg"), 0,0,210,297)
try:
	os.remove("temp.jpg")
except:
	pass
cv2.waitKey(0)
cv2.destroyAllWindows()
pdf.output(os.path.join(base_path, "images.pdf"), "F")