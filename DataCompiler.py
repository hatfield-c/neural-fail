import time
import torch
import numpy as np
import math
import cv2
import random
import sklearn.cluster

import CONFIG

class DataCompiler:
	def __init__(self, group):
		self.group = group
		self.train_pipeline = CONFIG.train_pipelines[group][0]
		self.validation_pipeline = CONFIG.validation_pipelines[group][0]
		
		self.imgs = None
		self.poses = None
		
		self.img_size = torch.FloatTensor(CONFIG.img_size).cuda()
		self.img_grid = self.ImageGrid(CONFIG.img_size[0], CONFIG.img_size[1]).cuda() / (self.img_size - 1)
		
		self.unfold1 = torch.nn.Unfold((3, 3), padding = 1)
		self.unfold3 = torch.nn.Unfold((7, 7), padding = 3)
		self.unfold5 = torch.nn.Unfold((11, 11), padding = 5)
		self.unfold7 = torch.nn.Unfold((15, 15), padding = 7)
	
	def Compile(self):
		self.Load(self.train_pipeline)
		
		imgs_path = self.train_pipeline.data_base_path + "/train_imgs.float64"
		poses_path = self.train_pipeline.data_base_path + "/train_poses.float64"
		torch.save(self.imgs, imgs_path)
		torch.save(self.poses, poses_path)
		
		print("[train shape]")
		print("   ", self.imgs.shape)
		print("   ", self.poses.shape)
		print("    Max Position:", torch.max(self.poses[:, :3], dim = 0).values)
		print("    Min Position:", torch.min(self.poses[:, :3], dim = 0).values)
		
		self.Load(self.validation_pipeline)
		
		imgs_path = self.train_pipeline.data_base_path + "/validation_imgs.float64"
		poses_path = self.train_pipeline.data_base_path + "/validation_poses.float64"
		torch.save(self.imgs, imgs_path)
		torch.save(self.poses, poses_path)
		
		print("[validation shape]")
		print("   ", self.imgs.shape)
		print("   ", self.poses.shape)
		print("    Max Position:", torch.max(self.poses[:, :3], dim = 0).values)
		print("    Min Position:", torch.min(self.poses[:, :3], dim = 0).values)
		
	def GetImage(self, img_path, colors):
		raw_img = cv2.imread(img_path)
		
		raw_img = cv2.resize(raw_img, CONFIG.img_size)
		raw_img = cv2.GaussianBlur(raw_img, (3,3), 1)
		
		img = torch.FloatTensor(raw_img).cuda() / 255
		
		c_scores = img.reshape(CONFIG.img_size[0], CONFIG.img_size[1], 1, 3) - (colors.reshape(1, 1, 4, 3) / 255)
		c_scores = torch.linalg.norm(c_scores, dim = 3)
		c_scores = torch.exp(-50 * torch.square(c_scores))
		
		img_temp = torch.cat((img, c_scores), 2)
		img_temp = img_temp.view(1, CONFIG.img_size[0], CONFIG.img_size[1], -1)
		img_temp = torch.moveaxis(img_temp, 3, 1)
		
		k1 = self.GetSubImages(img_temp, self.unfold1, (3, 3))
		k3 = self.GetSubImages(img_temp, self.unfold3, (7, 7))
		k5 = self.GetSubImages(img_temp, self.unfold5, (11, 11))
		k7 = self.GetSubImages(img_temp, self.unfold7, (15, 15))
		
		k1 = k1.reshape(CONFIG.img_size[0], CONFIG.img_size[1], -1, 7)
		k3 = k3.reshape(CONFIG.img_size[0], CONFIG.img_size[1], -1, 7)
		k5 = k5.reshape(CONFIG.img_size[0], CONFIG.img_size[1], -1, 7)
		k7 = k7.reshape(CONFIG.img_size[0], CONFIG.img_size[1], -1, 7)
		
		m1 = torch.mean(k1, dim = 2)
		m3 = torch.mean(k3, dim = 2)
		m5 = torch.mean(k5, dim = 2)
		m7 = torch.mean(k7, dim = 2)
		
		boost_ordering = [
			0, 1, 2, # r, g, b [0:3]
			3, 4, 5, # mr1, mg1, mb1 [3:6]
			10, 11, 12, # mr3, mg3, mb3 [6:9]
			17, 18, 19, # mr5, mg5, mb5 [9:12]
			24, 25, 26, # mr7, mg7, mb7 [12:15]
			6, 7, 8, 9, # c01, c11, c21, c31 [15:19]
			13, 14, 15, 16, # c03, c13, c23, c33 [19:23]
			20, 21, 22, 23, # c05, c15, c25, c35 [23:27]
			27, 28, 29, 30, # c07, c17, c27, c37 [27:31]
			31, 32			# pos0, pos1
		]
		
		img_boosted = torch.cat((img, m1, m3, m5, m7, self.img_grid), dim = 2)
		img_boosted = img_boosted[:, :, boost_ordering]

		return img_boosted.cpu()
	
	def GetSubImages(self, img, unfolder, ksize):
		kernels = unfolder(img)
		kernels = torch.moveaxis(kernels, 1, 2)
		kernels = kernels.view(1, CONFIG.img_size[0], CONFIG.img_size[1], -1, ksize[0], ksize[1])
		kernels = torch.moveaxis(kernels, 3, 5)
		
		return kernels[0]
	
	def Load(self, pipeline):
		data_type = pipeline.data_type
		
		if data_type == "7scenes":
			self.Load7Scenes(pipeline)
		elif data_type == "cambridge":
			self.LoadCambridge(pipeline)
				
	def Load7Scenes(self, pipeline):
		root_paths = pipeline.data_paths
		data_counts = pipeline.data_counts
		colors = pipeline.colors
		colors = torch.FloatTensor(colors).cuda()
		
		imgs = []
		poses = []
		
		for root_index in range(len(root_paths)):
			root_path = root_paths[root_index]
			sample_count = data_counts[root_index]
			
			print("Loading root", root_path)
			
			for i in range(sample_count):

				if i % (sample_count // 10) == 0:
					print("   ", str(i // (sample_count // 10)) + "0%")
				
				img_path = pipeline.data_base_path + "/" + root_path + "frame-" + str(i).zfill(6) + ".color.png"
				pose_path = pipeline.data_base_path + "/" + root_path + "frame-" + str(i).zfill(6) + ".pose.txt"
				
				img = self.GetImage(img_path, colors).float()
				pose = np.loadtxt(pose_path)
				
				pose = pose[[0, 2, 1]]
				pose[2, 3] = -pose[2, 3]
				pose = pose[:, 3]
				pose = torch.FloatTensor(pose).float()
				
				imgs.append(img)
				poses.append(pose)
				
			print("")
		
		self.imgs = torch.stack(imgs)
		self.poses = torch.stack(poses)
    
	def LoadCambridge(self, pipeline):
		root_paths = pipeline.data_paths
		colors = pipeline.colors
		colors = torch.FloatTensor(colors).cuda()
		
		imgs = []
		poses = []
		
		print("Loading root", root_paths[0])
		
		with open(pipeline.data_base_path + "/" + root_paths[0]) as file:

			i = 0
			for line in file:
				
				if i < 3:
					i += 1
					continue
				
				buffer = line.split()
				
				pos = torch.FloatTensor([float(buffer[1]), float(buffer[2]), float(buffer[3])]).float()
				poses.append(pos)
				
				img_path = pipeline.data_base_path + "/" + buffer[0]
				img = self.GetImage(img_path, colors).float()
				
				imgs.append(img)
				
				i += 1
		
		self.imgs = torch.stack(imgs)
		self.poses = torch.stack(poses)
        
	def ImageGrid(self, height, width):
		return torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing = "ij"), dim = -1)