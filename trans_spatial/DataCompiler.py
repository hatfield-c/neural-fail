import cv2
import numpy as np

import CONFIG

class DataCompiler:
	def __init__(self):
		pass
	
	def Compile(self):
		
		for i in range(24, 232):
			centroid = np.array([i, 32])
			canvas = self.GenerateSample(centroid)
			cv2.imwrite("data/in/" + str(i).zfill(4) + ".png", canvas)
		
	def GenerateSample(self, centroid):
		canvas = np.zeros((CONFIG.img_size[0], CONFIG.img_size[1], 3))
		poly = np.array([[0, -24], [6, -8], [24, -8], [9, 3], [14, 19], [0, 10], [-14, 19], [-9, 3], [-24, -8], [-6,-8]])
		
		polygon = poly + centroid
		
		cv2.fillPoly(canvas, pts = [polygon], color = (255, 0, 0))
		
		return canvas
