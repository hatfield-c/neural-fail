import torch
import numpy as np
import cv2

import CONFIG
import external_src.vision_mamba

class ModelViM(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		self.img_size = 128
		self.patch_size = 8
		self.depth = 4
		self.output_size = 1
		self.hidden_size = 32
		self.mlp_size = 32

		self.model = external_src.vision_mamba.Vim(
			dim=64,
		    dt_rank=32,
		    dim_inner=64,
		    d_state=64,
		    num_classes=1,
		    image_size=self.img_size,
		    patch_size=16,
		    channels=3,
		    dropout=0.1,
		    depth=8,
		)

	def forward(self, data, dumm0, dumm1):
		data = data.view(-1, self.img_size, self.img_size, 3)
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

		