import cv2
import numpy as np

import CONFIG

class DataCompiler:
	def __init__(self):
		self.img_size = np.array([64, 256])
	
	def Compile(self):
		scene_id = "star"
		
		src_path = "data/in/" + scene_id + "_src.png"
		out_path = "data/in/" + scene_id + "/"
		
		base = None
		img_scale = 1
		if scene_id == "desert":
			base = np.array([112, 19])
			img_scale = 0.25
		elif scene_id == "dock":
			base = np.array([515, 864])
			img_scale = 1
		elif scene_id == "field":
			base = np.array([127, 26])
			img_scale = 0.25
		elif scene_id == "google":
			base = np.array([51, 63])
			img_scale = 0.25
		elif scene_id == "marsh":
			base = np.array([238, 282])
			img_scale = 0.5
		elif scene_id == "river":
			base = np.array([254, 279])
			img_scale = 0.5
		elif scene_id == "suburb":
			base = np.array([104, 42])
			img_scale = 0.25
		elif scene_id == "star":
			base = np.array([0, 0])
			img_scale = 1
		
		src_img = cv2.imread(src_path)
		src_size = np.array(src_img.shape) * img_scale
		src_size = src_size.astype(np.int32)
		
		src_img = cv2.resize(src_img, (src_size[1], src_size[0]))
		
		##
		bg = cv2.imread("data/in/suburb_src.png")
		
		for i in range(256):
			b = np.array([base[0], base[1] + i])
			
			sample = src_img[b[0]:b[0] + 64, b[1]:b[1] + 256]
			
			##
			sample = sample.copy()
			sample[sample[:, :, 0] == 0] = bg[:64, :256][sample[:, :, 0] == 0]
			
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
