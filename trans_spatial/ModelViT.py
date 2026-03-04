import torch
import torchvision
import numpy as np
import cv2

import CONFIG

class ModelViT(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		self.img_size = 128
		self.patch_size = 8
		self.depth = 4
		self.output_size = 1
		self.hidden_size = 32
		self.mlp_size = 32
		self.model = torchvision.models.vision_transformer.VisionTransformer(self.img_size, self.patch_size, self.depth, 16, self.hidden_size, self.mlp_size, num_classes = self.output_size)

	def forward(self, data, dumm0, dumm1):
		data = data.view(-1, 128, 128, 3)
		data = data.permute([0, 3, 1, 2])
		out = self.model(data)
		
		return out
	
	def Save(self, save_path):
		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, load_path):
		self.load_state_dict(torch.load(load_path))
		self.eval()

		print("\nLoaded model from", load_path)

		