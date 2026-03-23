import torch
import cv2
import numpy as np

import ModelLinear
import ModelLinearNorm
import ModelVgg16
import ModelViT
import ModelViM
import ModelDeepSet
import ModelDspg
import ModelDsvgg

class Pipeline:
	def __init__(self, model_id, scene_id, learning_rate, epochs, print_every_epoch, ablations, color = "black"):
		self.model_id = model_id
		self.scene_id = scene_id
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.print_every_epoch = print_every_epoch
		self.color = color
		
		model_type = None
		losser_type = None
		
		if model_id == "linear":
			model_type = ModelLinear.ModelLinear
			losser_type = torch.nn.MSELoss
		elif model_id == "linear_norm":
			model_type = ModelLinearNorm.ModelLinearNorm
			losser_type = torch.nn.MSELoss	
		elif model_id == "vgg16":
			model_type = ModelVgg16.ModelVgg16
			losser_type = torch.nn.MSELoss	
		elif model_id == "vit":
			model_type = ModelViT.ModelViT
			losser_type = torch.nn.MSELoss
		elif model_id == "vim":
			model_type = ModelViM.ModelViM
			losser_type = torch.nn.MSELoss
		elif model_id == "deepset":
			model_type = ModelDeepSet.ModelDeepSet
			losser_type = torch.nn.MSELoss
		elif model_id == "dspg":
			model_type = ModelDspg.ModelDspg
			losser_type = torch.nn.MSELoss
		elif model_id == "dsvgg":
			model_type = ModelDsvgg.ModelDsvgg
			losser_type = torch.nn.MSELoss
		
		self.model_type = model_type
		self.losser_type = losser_type
		self.ablations = ablations
		self.ablation_count = len(ablations)
	
	def Ablate(self, index, data_loader):
		ablation = self.ablations[index]
		
		pos_count = ablation[0]
		neg_count = ablation[1]
		
		pc = 0
		nc = 0
		
		pos_indices = []
		neg_indices = []
		neg_regions = []
		beg = None
		
		for i in range(data_loader.imgs.shape[0]):
			if pc < pos_count:
				pos_indices.append(i)
				pc += 1
			elif nc < neg_count:
				if nc == 0:
					beg = i
					
				if i == data_loader.imgs.shape[0] - 1:
					 neg_regions.append((beg, i))
					
				neg_indices.append(i)
				nc += 1
			else:
				neg_regions.append((beg, i - 1))
				pos_indices.append(i)
				pc = 1
				nc = 0
		
		data_loader.valid_indices = torch.IntTensor(pos_indices).cuda()
		data_loader.invalid_indices = torch.IntTensor(neg_indices)
		data_loader.invalid_regions = neg_regions
		
		if False:
			for i in range(data_loader.valid_indices.shape[0]):
				img = data_loader.imgs[data_loader.valid_indices[i]]
				img = img.cpu().numpy()
				cv2.namedWindow("img", flags = cv2.WINDOW_NORMAL)
				cv2.imshow("img", img)
				print(data_loader.poses[data_loader.valid_indices[i]])
				cv2.waitKey(0)
			cv2.destroyAllWindows()
				
	def Reset(self, data_loader):
		data_loader.valid_indices = torch.arange(0, data_loader.imgs.shape[0])
		data_loader.invalid_indices = torch.zeros((0, 1))