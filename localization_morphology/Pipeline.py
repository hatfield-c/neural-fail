import torch
import cv2

import ModelLinear

class Pipeline:
	def __init__(self, model_id, ablations):
		self.model_id = model_id
		
		model = None
		if model_id == "linear":
			model = ModelLinear.ModelLinear()
	
		self.model = model
		self.ablations = ablations
		self.ablation_count = len(ablations)
	
	def Ablate(self, index, data_loader):
		ablation = self.ablations[index]
		
		lower = ablation[0]
		upper = ablation[1]
		
		indices = torch.cat((torch.arange(0, lower), torch.arange(upper, data_loader.imgs.shape[0])))
		
		data_loader.valid_indices = indices
