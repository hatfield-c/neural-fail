import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import CONFIG
import DataLoader

class Tester:
	def __init__(self):
		pass
	
	def Test(self, pipe_id):
		tokens = pipe_id.split("_")
		root_scene = tokens[-1]
		
		ablations = CONFIG.pipelines[pipe_id].ablations
		loader = DataLoader.DataLoader(root_scene)
		
		for a in range(len(ablations)):
			
			if a != CONFIG.target_ablation and CONFIG.target_ablation is not None:
				continue
			
			ablation = ablations[a]	
			
			lower = ablation[0]
			upper = ablation[1]
			lower = loader.poses[lower]
			upper = loader.poses[upper]
			lower = (lower)#.int() * 255
			upper = (upper)#.int() * 255
			
			fig1 = plt.figure()
			ax1 = fig1.add_subplot()
			fig2 = plt.figure()
			ax2 = fig2.add_subplot(projection = "3d")
			fig, ax = plt.subplots()
			
			ax.set_xlim([-0.1, 1.1])
			ax.set_ylim([-0.1, 1.1])
			ax.set_xlabel("Path Progress")
			ax.set_ylabel("Prediction Error")
			ax.set_title("Translation Error")
			
			#ax1.axvspan(lower, upper, alpha=0.2)
			#ax1.set_xlim([-0.1, 1.1])
			#ax1.set_ylim([-0.1, 1.1])
			ax1.set_xlabel("X")
			ax1.set_ylabel("Y")
			ax1.set_title("Path - Top View")
			
			for pipe_id in CONFIG.pipelines:
				pipeline = CONFIG.pipelines[pipe_id]
				model_id = pipeline.model_id
				
				if pipeline.scene_id != root_scene:
					continue
				
				#if pipeline.model_id != "dspg":
				#	continue
				
				#if pipeline.model_id != "deepset":
				#	continue
			
				model_path = CONFIG.model_base_path + pipeline.model_id + "_" + pipeline.scene_id + "_a" + str(a) + ".pt"
				
				try:
					model = pipeline.model_type(loader.img_size).cuda()
					model.Load(model_path)
					
				except:
					print("Model not found:", model_path)
					continue
				
				a_imgs = loader.imgs[loader.valid_indices]
				a_poses = loader.poses[loader.valid_indices]
				
				poses = model(a_imgs.reshape(a_imgs.shape[0], -1).cuda(), False, False)
				
				poses = (poses.detach().cpu())#.int() * 255
				truth = (a_poses.detach().cpu())#.int() * 255
				
				print(poses[-1].detach().numpy(), truth[-1].numpy())
				
				#perror = torch.linalg.norm(poses - truth, dim = 1)
				perror = torch.abs(poses[:, 0] - truth)
				
				pipeline.Ablate(a, loader)
				valid_truth = truth[loader.valid_indices.cpu()]
				invalid_regions = loader.invalid_regions
				
				ax.axvspan(-0.1, 0, alpha = 0.2)
				ax.axvspan(1, 1.1, alpha = 0.2)
				for i in range(len(invalid_regions)):
					region = invalid_regions[i]
					ax.axvspan(region[0] / 207.0, (region[1] + 1) / 207.0, alpha = 0.2)
				
				pipeline.Reset(loader)
				
				ax.plot(np.arange(perror.shape[0]) / perror.shape[0], perror, label = model_id, c = pipeline.color)
				
				ax1.plot(truth[:], truth[:], linestyle = 'dashed', color = "black", alpha = 0.4)
				ax1.plot(truth[:], poses[:], label = model_id, c = pipeline.color)
				ax1.scatter(valid_truth[:], valid_truth[:], color = "green", alpha = 0.4)
				
				idx0 = 0#3
				idx1 = 1#7
				idx2 = 2#11
				
				#ax2.plot(truth[:, idx0], truth[:, idx1], truth[:, idx2], linestyle = 'dashed', color = "black", alpha = 0.4)
				#ax2.plot(poses[:, idx0], poses[:, idx1], poses[:, idx2], c = pipeline.color)
				#ax2.scatter(poses[:, idx0], poses[:, idx1], poses[:, idx2], color = "gray", alpha = 0.4)
				#ax2.scatter(valid_truth[:, idx0], valid_truth[:, idx1], valid_truth[:, idx2], color = "green", alpha = 0.4)
				
			ax.legend(loc = "upper left")
			ax1.legend(loc = "upper left")
			plt.show()
			fig.savefig("data/out/error_" + root_scene + "_" + str(a) + ".png")
			fig1.savefig("data/out/path_" + root_scene + "_" + str(a) + ".png")
			plt.close()