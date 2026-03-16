import torch
import numpy as np

import CONFIG

class ModelDeepSet(torch.nn.Module):
	def __init__(self, img_size):
		super().__init__()

		self.img_size = img_size
		self.set_size = self.img_size[0] * self.img_size[1]
		self.embed_size = 16
		
		nodes = 16#self.embed_size
		depth = 2
		entry_size = 5
		exit_size = self.embed_size

		self.encoder_linears = []
		self.encoder_norms = []
		self.encoder_norm_means = []
		self.encoder_norm_vars = []
		self.encoder_norm_weights = []
		self.encoder_norm_bias = []
		for i in range(depth):
			insize = nodes
			outsize = nodes
			
			if i == 0:
				insize = entry_size
				
			if i == depth - 1:
				outsize = exit_size
				
			linear = torch.nn.Linear(insize, outsize).cuda()
			
			self.encoder_linears.append(linear)
			if i < depth - 1:
				norm = torch.nn.BatchNorm1d(outsize)
				norm_means = torch.zeros(outsize).cuda()
				norm_vars = torch.zeros(outsize).cuda()
				norm_weights = torch.zeros(outsize).cuda()
				norm_bias = torch.zeros(outsize).cuda()
				
				self.register_buffer("encoder_norm_means" + str(i), norm_means)
				self.register_buffer("encoder_norm_vars" + str(i), norm_vars)
				self.register_buffer("encoder_norm_weights" + str(i), norm_weights)
				self.register_buffer("encoder_norm_bias" + str(i), norm_bias)
				
				self.encoder_norms.append(norm)
				self.encoder_norm_means.append(norm_means)
				self.encoder_norm_vars.append(norm_vars)
				self.encoder_norm_weights.append(norm_weights)
				self.encoder_norm_bias.append(norm_bias)
			
		self.encoder_linears = torch.nn.ModuleList(self.encoder_linears)
		self.encoder_norms = torch.nn.ModuleList(self.encoder_norms)
		self.encoder_depth = depth

		entry_size = exit_size
		exit_size = 3
		nodes = 16
		depth = 2
		
		self.decoder_linears = []
		self.decoder_norms = []
		self.decoder_norms = []
		self.decoder_norm_means = []
		self.decoder_norm_vars = []
		self.decoder_norm_weights = []
		self.decoder_norm_bias = []
		for i in range(depth):
			insize = nodes
			outsize = nodes
			
			if i == 0:
				insize = entry_size
				
			if i == depth - 1:
				outsize = exit_size
				
			linear = torch.nn.Linear(insize, outsize).cuda()
			
			self.decoder_linears.append(linear)
			if i < depth - 1:
				norm = torch.nn.BatchNorm1d(outsize)
				norm_means = torch.zeros(outsize).cuda()
				norm_vars = torch.zeros(outsize).cuda()
				norm_weights = torch.zeros(outsize).cuda()
				norm_bias = torch.zeros(outsize).cuda()
				
				self.register_buffer("decoder_norm_means" + str(i), norm_means)
				self.register_buffer("decoder_norm_vars" + str(i), norm_vars)
				self.register_buffer("decoder_norm_weights" + str(i), norm_weights)
				self.register_buffer("decoder_norm_bias" + str(i), norm_bias)
				
				self.decoder_norms.append(norm)
				self.decoder_norm_means.append(norm_means)
				self.decoder_norm_vars.append(norm_vars)
				self.decoder_norm_weights.append(norm_weights)
				self.decoder_norm_bias.append(norm_bias)
			
		self.decoder_linears = torch.nn.ModuleList(self.decoder_linears)
		self.decoder_norms = torch.nn.ModuleList(self.decoder_norms)
		self.decoder_depth = depth

		#self.activation = torch.nn.ReLU()
		self.activation = self.Radial
		
		self.img_grid = self.ImageGrid(self.img_size[0], self.img_size[1])
		self.grid_list = self.img_grid.reshape(-1, 2).cuda()

	def Radial(self, out):
		return torch.exp((-0.5) * torch.square(out - 1))

	# PyTorch's batchnorm has unstable behavior when using either train or eval modes for inference on variable sized batches.
	# After training, we capture and store (i.e. bake) the correct batchnorm parameters in last epoch for use during inference.
	def forward(self, data, is_train = False, is_bake = False):
		batch_size = data.shape[0]
		out = data.view(batch_size * self.set_size, 3)
		out = torch.concatenate((out, self.grid_list.repeat(batch_size, 1)), dim = 1)

		
		#is_train = True
		#is_bake = False


		for i in range(self.encoder_depth):
			linear = self.encoder_linears[i]
			
			out = linear(out)
			if i != self.encoder_depth - 1:
				norm = self.encoder_norms[i]
				
				if is_train:
					if is_bake:
						norm_means = out.mean(0)
						norm_vars = out.var(0)
						norm_weights = norm.weight
						norm_bias = norm.bias
						
						self.encoder_norm_means[i] += norm_means[:].detach()
						self.encoder_norm_vars[i] += norm_vars[:].detach()
						self.encoder_norm_weights[i] += norm_weights[:]
						self.encoder_norm_bias[i] += norm_bias[:]
					
					out = norm(out)
				else:
					norm_means = self.encoder_norm_means[i]
					norm_vars = self.encoder_norm_vars[i]
					norm_weights = self.encoder_norm_weights[i]
					norm_bias = self.encoder_norm_bias[i]
					
					out = (out - norm_means.view(1, -1)) / torch.sqrt(norm_vars.view(1, -1) + norm.eps)
					out = (out * norm_weights.view(1, -1)) + norm_bias.view(1, -1)
				out = self.activation(out)
		
		out = out.view(batch_size, self.set_size, self.embed_size)
		out = torch.mean(out, dim = 1)
		#out = torch.max(out, dim = 1).values
		
		for i in range(self.decoder_depth):
			linear = self.decoder_linears[i]
			
			out = linear(out)
			
			if i != self.decoder_depth - 1:
				norm = self.decoder_norms[i]
				
				if is_train:
					if is_bake:
						norm_means = out.mean(0)
						norm_vars = out.var(0)
						norm_weights = norm.weight
						norm_bias = norm.bias
						
						self.decoder_norm_means[i] += norm_means[:].detach()
						self.decoder_norm_vars[i] += norm_vars[:].detach()
						self.decoder_norm_weights[i] += norm_weights[:]
						self.decoder_norm_bias[i] += norm_bias[:]
					
					out = norm(out)
				else:
					norm_means = self.decoder_norm_means[i]
					norm_vars = self.decoder_norm_vars[i]
					norm_weights = self.decoder_norm_weights[i]
					norm_bias = self.decoder_norm_bias[i]
					
					out = (out - norm_means.view(1, -1)) / torch.sqrt(norm_vars.view(1, -1) + norm.eps)
					out = (out * norm_weights.view(1, -1)) + norm_bias.view(1, -1)
				out = self.activation(out)
			
		return out
	
	def Save(self, save_path):
		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, load_path):
		self.load_state_dict(torch.load(load_path))

		print("\nLoaded model from", load_path)

	def ImageGrid(self, height, width):
		return torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing = "ij"), dim = -1)
		