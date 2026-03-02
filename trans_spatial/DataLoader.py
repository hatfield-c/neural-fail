import os
import cv2
import numpy as np
import torch

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
			p = p / 255.0
			
			imgs.append(img)
			poses.append(p)
			
		imgs = np.stack(imgs)
		poses = np.stack(poses)
		
		self.imgs = torch.FloatTensor(imgs)
		self.poses = torch.FloatTensor(poses)
		self.valid_indices = torch.arange(self.imgs.shape[0])
		
	def DrawSamples(self, sample_count):
		#indices = torch.randint(0, self.valid_indices.shape[0], (sample_count,))
		#indices = self.valid_indices[indices]
		indices = self.valid_indices#[:46]
		
		#indices = torch.arange(0, 64)
		sample_count = indices.shape[0]
		
		imgs = self.imgs[indices].reshape(sample_count, -1)
		poses = self.poses[indices].reshape(sample_count, 1)
		
		return imgs.cuda(), poses.cuda()
	