import os
import cv2
import numpy as np
import torch

import CONFIG

class DataLoader:
	def __init__(self, scene_id):
		self.scene_id = scene_id
		
		print(f"[Loading dataset: {scene_id}]")
		
		in_path = "data/in/" + scene_id + "/"
		
		item_count = 256
		frame_offset = 0
		self.img_size = np.array([64, 256])
		if scene_id == "stairs":
			item_count = 500
			frame_offset = (item_count // 3)
		
		imgs = []
		poses = []
		
		for i in range(item_count):
			
			if i < 24 or i > item_count - 24:
				continue
			
			idx = i + frame_offset
			img_path = in_path + "frame-" + str(idx).zfill(6) + ".color.png"
			pose_path = in_path + "frame-" + str(i).zfill(6) + ".pose.txt"
			
			pose = np.loadtxt(pose_path)
			img = cv2.imread(img_path)
			#img = cv2.GaussianBlur(img, (17, 17), 0)
			#img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
			
			if False:
				cv2.namedWindow("img", flags = cv2.WINDOW_NORMAL)
				cv2.imshow("img", img)
				print(i, pose)
				cv2.waitKey(0)
			
			img = img / 256.0
			
			
			#pose = pose[[0, 2, 1]]
			#pose[2, 3] = -pose[2, 3]
			#pose = pose[:3, 3]
			
			pose = torch.FloatTensor(pose).float() / 256
			
			imgs.append(img)
			poses.append(pose)
			
		cv2.destroyAllWindows()
		
		imgs = np.stack(imgs)
		poses = np.stack(poses)
		
		axes_mags = np.max(poses, axis = 0) - np.min(poses, axis = 0)
		self.largest_axis = np.argmax(axes_mags)
		
		self.imgs = torch.FloatTensor(imgs)
		self.poses = torch.FloatTensor(poses)
		self.valid_indices = torch.arange(self.imgs.shape[0])
		self.invalid_indices = torch.zeros((0, 1))
		self.invalid_regions = []
		
		print(f"[{scene_id}]: {imgs.shape[0]} samples loaded")
		
	def DrawSamples(self, batch_size):
		indices = self.valid_indices
		
		#if batch_size > 0:
		#	if batch_size <= self.valid_indices.shape[0]:
				#indices = torch.randint(0, self.valid_indices.shape[0], (batch_size,))
		#		indices = np.random.choice(torch.arange(self.valid_indices.shape[0]), batch_size, replace = False)
		#		indices = self.valid_indices[indices]
		
		sample_count = indices.shape[0]
		
		imgs = self.imgs[indices].reshape(sample_count, -1)
		poses = self.poses[indices].reshape(sample_count)
		
		return imgs.cuda(), poses.cuda()
	