import cv2
import numpy as np

import CONFIG

class DataCompiler:
	def __init__(self):
		pass
	
	def Compile(self):
		
		out_path = "data/in/star/seq-01/"
		
		for i in range(24, 232):
			centroid = np.array([i, CONFIG.img_size[0] // 2])
			canvas = self.GenerateSample(centroid)
			
			img_path = out_path + "frame-" + str(i - 24).zfill(6) + ".color.png"
			
			cv2.imwrite(img_path, canvas)
		
	def GenerateSample(self, centroid):
		canvas = np.zeros((CONFIG.img_size[0], CONFIG.img_size[1], 3), dtype = np.uint8)
		poly = np.array([[0, -24], [6, -8], [24, -8], [9, 3], [14, 19], [0, 10], [-14, 19], [-9, 3], [-24, -8], [-6,-8]])
		
		polygon = poly + centroid
		
		cv2.fillPoly(canvas, pts = [polygon], color = (255, 0, 0))
		
		return canvas
