import cv2
import numpy as np

import CONFIG

class DataCompiler:
	def __init__(self):
		pass
	
	def Compile(self):
		
		for i in range(0, 182, 2):
			canvas = self.GenerateSample(i)
			cv2.imwrite("data/in/" + str(i).zfill(4) + ".jpg", canvas)
		
	def GenerateSample(self, theta):
		canvas = np.zeros((CONFIG.img_size[0], CONFIG.img_size[1], 3))
		theta = (theta * np.pi) / 180.0
		
		centroid = np.array([CONFIG.img_size[1] // 2, CONFIG.img_size[0] // 2])
		endpoint = centroid + (np.array([np.cos(theta), -np.sin(theta)]) * 48).astype(np.int32)
		
		cv2.line(canvas, centroid, endpoint, color = (0, 0, 255), thickness = 8)
		cv2.circle(canvas, center = centroid, radius = 8, color = (0, 255, 0), thickness = -1)
		cv2.circle(canvas, center = endpoint, radius = 8, color = (0, 255, 0), thickness = -1)
	
		return canvas
