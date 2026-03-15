import os
import cv2
import numpy as np
import torch

import CONFIG

class DataLoader:
	def __init__(self, scene_id):
		self.scene_id = scene_id
		
		print(f"[Loading dataset: {scene_id}]")
		
		in_path = "data/in/" + scene_id + "/seq-01/"
		
		item_count = 1000
		if scene_id == "star":
			item_count = 208
		elif scene_id == "stairs":
			item_count = 500
		
		frame_offset = (item_count // 3)
		breakout = 207
		
		imgs = []
		poses = []
		
		for i in range(item_count):
			idx = i + frame_offset
			img_path = in_path + "frame-" + str(idx).zfill(6) + ".color.png"
			pose_path = in_path + "frame-" + str(i).zfill(6) + ".pose.txt"
			
			img = cv2.imread(img_path)
			img = cv2.resize(img, (CONFIG.img_size[0], CONFIG.img_size[1]))
			
			#cv2.imshow("img", img)
			#cv2.waitKey(0)
			
			pose = np.loadtxt(pose_path)
			
			pose = pose[[0, 2, 1]]
			pose[2, 3] = -pose[2, 3]
			pose = pose[:, 3]
			pose = torch.FloatTensor(pose).float()
			
			img = img / 256.0
			#pose = i / breakout
			
			imgs.append(img)
			poses.append(pose)
			
			if i == breakout:
				break
			
		imgs = np.stack(imgs)
		poses = np.stack(poses)
		
		self.imgs = torch.FloatTensor(imgs)
		self.poses = torch.FloatTensor(poses)
		self.valid_indices = torch.arange(self.imgs.shape[0])
		self.invalid_indices = torch.zeros((0, 1))
		
	def DrawSamples(self, batch_size):
		indices = self.valid_indices
		
		#if batch_size > 0:
		#	if batch_size <= self.valid_indices.shape[0]:
				#indices = torch.randint(0, self.valid_indices.shape[0], (batch_size,))
		#		indices = np.random.choice(torch.arange(self.valid_indices.shape[0]), batch_size, replace = False)
		#		indices = self.valid_indices[indices]
		
		sample_count = indices.shape[0]
		
		imgs = self.imgs[indices].reshape(sample_count, -1)
		poses = self.poses[indices].reshape(sample_count, -1)
		
		return imgs.cuda(), poses.cuda()
	