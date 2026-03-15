import torch
import cv2

import ModelLinear
import ModelLinearNorm
import ModelVgg16
import ModelViT
import ModelViM
import ModelDeepSet

class Pipeline:
	def __init__(self, model_id, scene_id, learning_rate, epochs, print_every_epoch, ablations):
		self.model_id = model_id
		self.scene_id = scene_id
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.print_every_epoch = print_every_epoch
		
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
		
		self.model_type = model_type
		self.losser_type = losser_type
		self.ablations = ablations
		self.ablation_count = len(ablations)
	
	# todo: test star results with original and 7scenes ablation styles
	#    original: samples on side with large pool missing in center
	#    7scenes:  sample stride
	
	def Ablate(self, index, data_loader):
		ablation = self.ablations[index]
		
		lower = ablation[0]
		upper = ablation[1]
		
		indices = torch.cat((torch.arange(0, lower), torch.arange(upper, data_loader.imgs.shape[0])))
		
		data_loader.valid_indices = indices
