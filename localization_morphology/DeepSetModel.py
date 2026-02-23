import torch
import numpy as np

import CONFIG

class DeepSetModel(torch.nn.Module):
	def __init__(self, batch_size, set_size, embed_size, is_boosted):
		self.is_boosted = is_boosted
		self.channel_indices = torch.arange(0, 33)
		
		if not self.is_boosted:
			self.channel_indices = [ 0, 1, 2, 31, 32 ]
		
		super().__init__()

		self.normalizer = torch.nn.Parameter(torch.ones(3,), requires_grad = False)

		self.batch_size = batch_size
		self.set_size = set_size
		self.embed_size = embed_size

		self.e00 = torch.nn.Linear(len(self.channel_indices), 64).cuda()
		self.e01 = torch.nn.Linear(64, 64).cuda()
		self.e02 = torch.nn.Linear(64, 64).cuda()
		self.e03 = torch.nn.Linear(64, 128).cuda()
		self.e04 = torch.nn.Linear(128, self.embed_size).cuda()
		
		self.p0 = torch.nn.Linear(self.embed_size, 512).cuda()
		self.p1 = torch.nn.Linear(512, 256).cuda()
		self.out_layer = torch.nn.Linear(256, 3).cuda()
		
		self.b00 = torch.nn.BatchNorm1d(64)
		self.b01 = torch.nn.BatchNorm1d(64)
		self.b02 = torch.nn.BatchNorm1d(64)
		self.b03 = torch.nn.BatchNorm1d(128)
		self.b04 = torch.nn.BatchNorm1d(self.embed_size)
		
		self.b20 = torch.nn.BatchNorm1d(512)
		self.b21 = torch.nn.BatchNorm1d(256)
		
		self.activation = torch.nn.ReLU()

	def forward(self, data):
		data = data.reshape(-1, 33)[:, self.channel_indices]
		
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
		
		out = self.e04(out)
		out = self.b04(out)
		out = self.activation(out)
		out = out.view(self.batch_size, self.set_size, self.embed_size)
		out = torch.mean(out, dim = 1)
		#out = torch.max(out, dim = 1).values
		
		out = self.p0(out)
		out = self.b20(out)
		out = self.activation(out)
		out = self.p1(out)
		out = self.b21(out)
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

		