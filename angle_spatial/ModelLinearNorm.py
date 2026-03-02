import torch
import numpy as np

import CONFIG

class ModelLinearNorm(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.e00 = torch.nn.Linear(CONFIG.img_size[0] * CONFIG.img_size[1] * 3, 256).cuda()
		self.e01 = torch.nn.Linear(256, 256).cuda()
		self.e02 = torch.nn.Linear(256, 64).cuda()
		self.e03 = torch.nn.Linear(64, 64).cuda()
		
		self.out_layer = torch.nn.Linear(64, 1).cuda()
		
		self.b00 = torch.nn.BatchNorm1d(256)
		self.b01 = torch.nn.BatchNorm1d(256)
		self.b02 = torch.nn.BatchNorm1d(64)
		self.b03 = torch.nn.BatchNorm1d(64)
		
		self.activation = torch.nn.ReLU()

	def forward(self, data, dumm0, dumm1):
		
		out = self.e00(data)
		out = self.b00(out)
		out = self.activation(out)
		
		out = self.e01(out)
		out = self.b01(out)
		out = self.activation(out)
		
		out = self.e02(out)
		out = self.b02(out)
		out = self.activation(out)
		
		out = self.e03(out)
		out = self.b03(out)
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

		