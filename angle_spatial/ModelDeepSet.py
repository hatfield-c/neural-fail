import torch
import numpy as np

import CONFIG

class ModelDeepSet(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.set_size = CONFIG.img_size[0] * CONFIG.img_size[1]
		self.embed_size = 16

		nodes = 16
		depth = 3
		entry_size = 5
		exit_size = self.embed_size

		self.encoder_linears = []
		self.encoder_norms = []
		for i in range(depth):
			insize = nodes
			outsize = nodes
			
			if i == 0:
				insize = entry_size
				
			if i == depth - 1:
				outsize = exit_size
				
			linear = torch.nn.Linear(insize, outsize).cuda()
			norm = torch.nn.BatchNorm1d(outsize)
			
			self.encoder_linears.append(linear)
			self.encoder_norms.append(norm)
			
		self.encoder_linears = torch.nn.ModuleList(self.encoder_linears)
		self.encoder_norms = torch.nn.ModuleList(self.encoder_norms)
		self.encoder_depth = depth

		entry_size = exit_size
		exit_size = 1
		nodes = 16
		depth = 5
		
		self.decoder_linears = []
		self.decoder_norms = []
		for i in range(depth):
			insize = nodes
			outsize = nodes
			
			if i == 0:
				insize = entry_size
				
			if i == depth - 1:
				outsize = exit_size
				
			linear = torch.nn.Linear(insize, outsize).cuda()
			norm = torch.nn.BatchNorm1d(outsize)
			
			self.decoder_linears.append(linear)
			self.decoder_norms.append(norm)
			
		self.decoder_linears = torch.nn.ModuleList(self.decoder_linears)
		self.decoder_norms = torch.nn.ModuleList(self.decoder_norms)
		self.decoder_depth = depth

		#self.activation = torch.nn.ReLU()
		self.activation = self.Radial
		
		self.img_grid = self.ImageGrid(CONFIG.img_size[0], CONFIG.img_size[1])
		self.grid_list = self.img_grid.reshape(-1, 2).cuda()

	def Radial(self, out):
		return torch.exp((-1) * torch.square(out - 1) / 2)

	def forward(self, data):
		batch_size = data.shape[0]
		out = data.view(batch_size * self.set_size, 3)
		out = torch.concatenate((out, self.grid_list.repeat(batch_size, 1)), dim = 1)

		for i in range(self.encoder_depth):
			linear = self.encoder_linears[i]
			norm = self.encoder_norms[i]
			
			out = linear(out)
			if i != self.encoder_depth - 1:
				#out = norm(out)
				out = self.activation(out)
		
		out = out.view(batch_size, self.set_size, self.embed_size)
		out = torch.mean(out, dim = 1)
		#out = torch.max(out, dim = 1).values
		
		for i in range(self.decoder_depth):
			linear = self.decoder_linears[i]
			norm = self.decoder_norms[i]
			
			out = linear(out)
			
			if i != self.decoder_depth - 1:
				#out = norm(out)
				out = self.activation(out)
			
		return out
	
	def Save(self, save_path):
		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, load_path):
		self.load_state_dict(torch.load(load_path))
		self.eval()

		print("\nLoaded model from", load_path)

	def ImageGrid(self, height, width):
		return torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing = "ij"), dim = -1)
		