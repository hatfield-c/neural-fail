import torch
import numpy as np

import CONFIG

class ModelLinear(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.e00 = torch.nn.Linear(CONFIG.img_size[0] * CONFIG.img_size[1] * 3, 256).cuda()
		self.e01 = torch.nn.Linear(256, 256).cuda()
		self.e02 = torch.nn.Linear(256, 64).cuda()
		self.e03 = torch.nn.Linear(64, 64).cuda()
		
		self.out_layer = torch.nn.Linear(64, 3).cuda()
		
		self.activation = torch.nn.ReLU()

	def forward(self, data, dumm0, dumm1):
		
		out = self.e00(data)
		out = self.activation(out)
		
		out = self.e01(out)
		out = self.activation(out)
		
		out = self.e02(out)
		out = self.activation(out)
		
		out = self.e03(out)
		out = self.activation(out)
		
		out = self.out_layer(out)
		
		return out
	
	def Save(self, save_path):
		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, load_path):
		self.load_state_dict(torch.load(load_path))
		self.eval()

		print("\nLoaded model from", load_path)

		