import torch
import numpy as np

import CONFIG

class ModelVgg16(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.layer_in = torch.nn.Conv2d(3, 64, 3, padding = "same").cuda()
		self.layer_h0 = torch.nn.Conv2d(64, 64, 3, padding = "same").cuda()
		
		self.layer_h1 = torch.nn.Conv2d(64, 128, 3, padding = "same").cuda()
		self.layer_h2 = torch.nn.Conv2d(128, 128, 3, padding = "same").cuda()
		
		self.layer_h3 = torch.nn.Conv2d(128, 256, 3, padding = "same").cuda()
		self.layer_h4 = torch.nn.Conv2d(256, 256, 3, padding = "same").cuda()
		self.layer_h5 = torch.nn.Conv2d(256, 256, 3, padding = "same").cuda()
		
		self.layer_h6 = torch.nn.Conv2d(256, 512, 3, padding = "same").cuda()
		self.layer_h7 = torch.nn.Conv2d(512, 512, 3, padding = "same").cuda()
		self.layer_h8 = torch.nn.Conv2d(512, 512, 3, padding = "same").cuda()
		
		self.layer_h9 = torch.nn.Conv2d(512, 512, 3, padding = "same").cuda()
		self.layer_h10 = torch.nn.Conv2d(512, 512, 3, padding = "same").cuda()
		self.layer_h11 = torch.nn.Conv2d(512, 512, 3, padding = "same").cuda()
		
		self.layer_h12 = torch.nn.Linear(512 * int((CONFIG.img_size[0] / (2 ** 5)) * (CONFIG.img_size[1] / (2 ** 5))), 4096).cuda()
		self.layer_h13 = torch.nn.Linear(4096, 256).cuda()
		self.layer_out = torch.nn.Linear(256, 1).cuda()

		self.activation = torch.nn.ReLU()
		self.maxpool = torch.nn.MaxPool2d(2)

	def forward(self, data):
		data = data.reshape(-1, CONFIG.img_size[0], CONFIG.img_size[1], 3)
		data = torch.moveaxis(data, 3, 1)
		
		out = self.layer_in(data)
		out = self.activation(out)
		out = self.layer_h0(out)
		out = self.activation(out)
		
		out = self.maxpool(out)
		
		out = self.layer_h1(out)
		out = self.activation(out)
		out = self.layer_h2(out)
		out = self.activation(out)
		
		out = self.maxpool(out)
		
		out = self.layer_h3(out)
		out = self.activation(out)
		out = self.layer_h4(out)
		out = self.activation(out)
		out = self.layer_h5(out)
		out = self.activation(out)
		
		out = self.maxpool(out)
		
		out = self.layer_h6(out)
		out = self.activation(out)
		out = self.layer_h7(out)
		out = self.activation(out)
		out = self.layer_h8(out)
		out = self.activation(out)
		
		out = self.maxpool(out)
		
		out = self.layer_h9(out)
		out = self.activation(out)
		out = self.layer_h10(out)
		out = self.activation(out)
		out = self.layer_h11(out)
		out = self.activation(out)
		
		out = self.maxpool(out)
		
		out = out.reshape(-1, out.shape[1] * out.shape[2] * out.shape[3])
		out = self.layer_h12(out)
		out = self.activation(out)
		out = self.layer_h13(out)
		out = self.activation(out)
		
		out = self.layer_out(out)
		
		return out
	
	def Save(self, save_path):
		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, load_path):
		self.load_state_dict(torch.load(load_path))
		self.eval()

		print("\nLoaded model from", load_path)

		