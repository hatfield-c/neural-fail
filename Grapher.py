import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import CONFIG
	
class Grapher:
	def __init__(self, group):
		self.group = group
	
	def Graph(self):
		
		graph_pipelines = CONFIG.validation_pipelines[self.group]
		
		self.GraphMedian(graph_pipelines)
		self.GraphMean(graph_pipelines)
		self.GraphMax(graph_pipelines)
		self.GraphLoss(graph_pipelines)
		self.GraphTime(graph_pipelines)
	
	def GraphTime(self, pipelines):
		
		x_axis = []
		y_axis = []
		
		for i in range(len(pipelines)):
			pipeline = pipelines[i]
			print("Graphing pipeline:", pipeline.pipeline_id)
			
			avg_time = np.loadtxt(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_avg_epoch_time.txt")
			
			x_axis.append(pipeline.pipeline_id)
			y_axis.append(avg_time)
			
		plt.bar(x_axis, y_axis)
		plt.grid(axis = 'y')
		plt.ylabel("avg. batch time (s)")
		#plt.show()
		plt.savefig("data/metrics/" + pipelines[0].data_id + "/" + pipelines[0].data_id + "_time_graph.png")
		plt.close()
	
	def GraphLoss(self, pipelines):
		c = [ "r", "b", "g", "c", "m", "y", "tab:orange", "tab:brown" ]
		
		min_val = 1e9
		max_val = -1e9
		for i in range(len(pipelines)):
			pipeline = pipelines[i]
			print("Graphing pipeline:", pipeline.pipeline_id)
			
			losses = np.loadtxt(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_loss_vals.txt")
			losses = losses[CONFIG.save_snapshots]
			
			plt.plot(CONFIG.save_snapshots, losses, c = c[i], label = pipeline.pipeline_id)
			
			min_val = min(min_val, losses[-1])
			max_val = max(max_val, losses[15])
			
		diff = max_val - min_val
		min_val = min_val - (diff * 0.05)
		max_val = max_val + (diff * 1)
		plt.ylim(bottom = min_val, top = max_val)
		
		plt.legend(loc = "upper right")
		plt.grid(axis = 'y')
		plt.xlabel("epochs")
		plt.ylabel("train loss (m)")
		#plt.show()
		plt.savefig("data/metrics/" + pipelines[0].data_id + "/" + pipelines[0].data_id + "_loss_graph.png")
		plt.close()
	
	def GraphMax(self, pipelines):
		c = [ "r", "b", "g", "c", "m", "y", "tab:orange", "tab:brown" ]
		
		min_val = 1e9
		max_val = -1e9
		for i in range(len(pipelines)):
			pipeline = pipelines[i]
			print("Graphing pipeline:", pipeline.pipeline_id)
			
			max_pos = np.loadtxt(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_max.txt")
			
			plt.plot(CONFIG.save_snapshots, max_pos, c = c[i], label = pipeline.pipeline_id)
			
			min_val = min(min_val, max_pos[5])
			max_val = max(max_val, max_pos[5])
			
		diff = max_val - min_val
		min_val = min_val - (diff * 0.3)
		max_val = max_val + (diff * 0.3)
		plt.ylim(bottom = min_val, top = max_val)
			
		_, loader = CONFIG.validation_pipelines[self.group][0].Create()
		loader.Load("validation")
		
		poses = loader.poses
		pos = poses[:, :3]
		pos_center = torch.mean(pos, dim = 0)
		pos_mag = torch.linalg.norm(torch.abs(pos - pos_center), dim = 1)
		pos_max = torch.max(pos_mag)
		plt.axhline(y = pos_max.item(), color = 'g')
			
		plt.legend(loc = "upper right")
		plt.grid(axis = 'y')
		plt.xlabel("epochs")
		plt.ylabel("max test radius (m)")
		#plt.show()
		plt.savefig("data/metrics/" + pipelines[0].data_id + "/" + pipelines[0].data_id + "_max_graph.png")
		plt.close()
	
	def GraphMean(self, pipelines):
		c = [ "r", "b", "g", "c", "m", "y", "tab:orange", "tab:brown" ]
		
		min_val = 1e9
		max_val = -1e9
		for i in range(len(pipelines)):
			pipeline = pipelines[i]
			print("Graphing pipeline:", pipeline.pipeline_id)
			
			mean_errors = np.loadtxt(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_mean.txt")
			
			plt.plot(CONFIG.save_snapshots, mean_errors, c = c[i], label = pipeline.pipeline_id)
			
			min_val = min(min_val, mean_errors[-1])
			max_val = max(max_val, mean_errors[15])
			
		diff = max_val - min_val
		min_val = min_val - (diff * 0.2)
		max_val = max_val + (diff * 1)
		plt.ylim(bottom = min_val, top = max_val)
			
		plt.legend(loc = "upper right")
		plt.grid(axis = 'y')
		plt.xlabel("epochs")
		plt.ylabel("mean test error (m)")
		#plt.show()
		plt.savefig("data/metrics/" + pipelines[0].data_id + "/" + pipelines[0].data_id + "_mean_graph.png")
		plt.close()
		
	def GraphMedian(self, pipelines):
		c = [ "r", "b", "g", "c", "m", "y", "tab:orange", "tab:brown" ]
		
		min_val = 1e9
		max_val = -1e9
		for i in range(len(pipelines)):
			pipeline = pipelines[i]
			print("Graphing pipeline:", pipeline.pipeline_id)
			
			median_errors = np.loadtxt(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_median.txt")
			
			plt.plot(CONFIG.save_snapshots, median_errors, c = c[i], label = pipeline.pipeline_id)
			
			min_val = min(min_val, median_errors[-1])
			max_val = max(max_val, median_errors[15])
			
		diff = max_val - min_val
		min_val = min_val - (diff * 0.2)
		max_val = max_val + (diff * 1)
		plt.ylim(bottom = min_val, top = max_val)
			
		plt.legend(loc = "upper right")
		plt.grid(axis = 'y')
		plt.xlabel("epochs")
		plt.ylabel("median test error (m)")
		#plt.show()
		plt.savefig("data/metrics/" + pipelines[0].data_id + "/" + pipelines[0].data_id + "_median_graph.png")
		plt.close()