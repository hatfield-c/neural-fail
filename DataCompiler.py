import cv2
import numpy as np

import CONFIG

class DataCompiler:
	def __init__(self):
		self.img_size = np.array([64, 256])
	
	def Compile(self):
		scene_id = "suburb"
		obj_id = "star"
		
		src_path = "data/in/" + scene_id + "_src.png"
		obj_path = "data/in/" + obj_id + ".png"
		out_path = "data/in/" + scene_id + "/"
		
		base = None
		if scene_id == "desert":
			base = np.array([112, 19])
		elif scene_id == "dock":
			base = np.array([515, 864])
		elif scene_id == "field":
			base = np.array([127, 26])
		elif scene_id == "google":
			base = np.array([51, 63])
		elif scene_id == "marsh":
			base = np.array([238, 282])
		elif scene_id == "river":
			base = np.array([254, 279])
		elif scene_id == "suburb":
			base = np.array([0, 459])
		elif scene_id == "star":
			base = np.array([0, 0])
		
		src_img = cv2.imread(src_path)
		obj_img = cv2.imread(obj_path)
		
		img_size = np.array((480, 640))
		scene_size = np.array([480, 1100], dtype = np.int32)
		src_img = cv2.resize(src_img, (scene_size[1], scene_size[0]))
		
		scene_img = src_img[base[0]:base[0] + img_size[0], base[1]:base[1] + img_size[1]]
		
		for i in range(0, img_size[1] - 64, 10):
			
			sample = scene_img.copy()
			
			obj = obj_img.copy()
			obj[obj[:, :, 0] == 0] = sample[240:240+64, i:i+64][obj[:, :, 0] == 0]
			
			sample[240:240+64, i:i+64] = obj
			
			sample_path = out_path + "frame-" + str(i).zfill(6) + ".color.png"
			pose_path = out_path + "frame-" + str(i).zfill(6) + ".pose.txt"
		
			pose = np.array([i])
			
			cv2.imshow("sample", sample)
			cv2.waitKey(0)
			
			cv2.imwrite(sample_path, sample)
			np.savetxt(pose_path, pose)
	
	def CompileStar(self):
		
		out_path = "data/in/star/seq-01/"
		
		for i in range(24, 232):
			centroid = np.array([i, self.img_size[0] // 2])
			canvas = self.GenerateSample(centroid)
			
			img_path = out_path + "frame-" + str(i - 24).zfill(6) + ".color.png"
			pose_path = out_path + "frame-" + str(i - 24).zfill(6) + ".pose.txt"
			
			pose = np.array([[0, 0, 0, (i - 24) / 207.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
			
			cv2.imwrite(img_path, canvas)
			np.savetxt(pose_path, pose)
		
	def GenerateSample(self, centroid):
		canvas = np.zeros((self.img_size[0], self.img_size[1], 3), dtype = np.uint8)
		poly = np.array([[0, -24], [6, -8], [24, -8], [9, 3], [14, 19], [0, 10], [-14, 19], [-9, 3], [-24, -8], [-6,-8]])
		
		polygon = poly + centroid
		
		cv2.fillPoly(canvas, pts = [polygon], color = (255, 0, 0))
		
		return canvas
