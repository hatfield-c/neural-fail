import os
import cv2
import numpy as np
import torch
import random

class DataLoader:
	def __init__(self):
		in_path = "data/in/"
		
		contents = os.listdir(in_path)
		
		imgs = []
		poses = []
		for i in range(len(contents)):
			img_name = contents[i]
			
			img = cv2.imread(in_path + img_name)
			p = int(img_name[:-4])
			
			img = img / 255.0
			p = p / 180.0
			
			imgs.append(img - 0.5)
			poses.append(p - 0.5)
			
		imgs = np.stack(imgs)
		poses = np.stack(poses)
		
		self.imgs = torch.FloatTensor(imgs)
		self.poses = torch.FloatTensor(poses)
		self.valid_indices = torch.arange(self.imgs.shape[0])
		
	def DrawSamples(self, sample_count):
		#sample_count = min(sample_count, len(self.valid_indices))
		
		#indices = torch.randint(0, self.valid_indices.shape[0], (sample_count,))
		#indices = random.sample(list(self.valid_indices), sample_count)
		#indices = self.valid_indices[indices]
		indices = self.valid_indices#[:10]
		
		#indices = torch.arange(0, 64)
		sample_count = len(indices)
		
		imgs = self.imgs[indices].reshape(sample_count, -1)
		poses = self.poses[indices].reshape(sample_count, 1)
		
		return imgs.cuda(), poses.cuda()
	