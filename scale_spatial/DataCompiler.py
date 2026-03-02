import cv2
import numpy as np

import CONFIG

class DataCompiler:
	def __init__(self):
		pass
	
	def Compile(self):
		
		for i in range(0, 100):
			canvas = self.GenerateSample(i)
			cv2.imwrite("data/in/" + str(i).zfill(4) + ".png", canvas)
		
	def GenerateSample(self, scale):
		thickness = 1
		scale = scale / 99
		scale = (scale + 0.1) * 4
		
		centroid = np.array([CONFIG.img_size[1] // 2, CONFIG.img_size[0] // 2])
		twidth, theight = cv2.getTextSize("@", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
		
		centroid[0] -= twidth // 2
		centroid[1] += theight // 2
		
		canvas = np.zeros((CONFIG.img_size[0], CONFIG.img_size[1], 3))
		cv2.putText(canvas, "@", centroid, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness, 2)
		
		return canvas
