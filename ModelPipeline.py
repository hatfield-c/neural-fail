import numpy as np

import CONFIG
import DataLoader
import Vgg16Model
import Vgg6Model
import LinearModel
import DeepSetModel

class ModelPipeline:
	def __init__(self, learning_rate, pipeline_id, model_id, data_id, data_type, model_base_path, data_base_path, data_paths, data_counts, is_boosted, colors = np.zeros((4, 3))):
		self.learning_rate = learning_rate
		self.pipeline_id = pipeline_id
		self.model_id = model_id
		self.data_id = data_id
		self.data_type = data_type
		self.model_base_path = model_base_path
		self.data_base_path = data_base_path
		self.data_paths = data_paths
		self.data_counts = data_counts
		self.is_boosted = is_boosted
		self.colors = np.array(colors)
	
	def Create(self):
		model = self.GetModel(self.model_id)
		loader = DataLoader.DataLoader(self.data_base_path)
		
		return model, loader
		
	def Load(self):
		pass
		
	def GetModel(self, model_id):
		model = None
		
		if model_id == "vgg16":
			model = Vgg16Model.Vgg16Model(self.is_boosted)
		
		if model_id == "vgg6":
			model = Vgg6Model.Vgg6Model(self.is_boosted)
		
		if model_id == "linear":
			model = LinearModel.LinearModel(self.is_boosted)
			
		if model_id == "deepset":
			model = DeepSetModel.DeepSetModel(CONFIG.batch_size, CONFIG.set_size, CONFIG.embed_size, self.is_boosted)
		
		if self.data_id == "chess":
			model.normalizer[0] = 4
			model.normalizer[1] = 4
			model.normalizer[2] = 4
		
		if self.data_id == "kitchen":
			model.normalizer[0] = 4
			model.normalizer[1] = 4
			model.normalizer[2] = 4
		
		if self.data_id == "stairs":
			model.normalizer[0] = 3
			model.normalizer[1] = 3
			model.normalizer[2] = 3
		
		if self.data_id == "church":
			model.normalizer[0] = 80
			model.normalizer[1] = 80
			model.normalizer[2] = 80
		
		if self.data_id == "street":
			model.normalizer[0] = 500
			model.normalizer[1] = 500
			model.normalizer[2] = 500
		
		if self.data_id == "hospital":
			model.normalizer[0] = 50
			model.normalizer[1] = 50
			model.normalizer[2] = 50
		
		return model
		