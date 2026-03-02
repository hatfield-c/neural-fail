import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import CONFIG
import DataLoader

class Tester:
	def __init__(self):
		pass
	
	def Test(self):
		loader = DataLoader.DataLoader()
		p_min = ((loader.poses[0] + 0.5) * 180)
		p_max = ((loader.poses[-1] + 0.5) * 180)
		
		for a in range(len(CONFIG.ablations)):
			
			if a != 2:
				continue
			
			ablation = CONFIG.ablations[a]
			
			lower = ablation[0]
			upper = ablation[1]
			lower = loader.poses[lower]
			upper = loader.poses[upper]
			lower = ((lower + 0.5) * 180).int()
			upper = ((upper + 0.5) * 180).int()
			
			fig, ax = plt.subplots()
			fig1, ax1 = plt.subplots()
			
			ax.axvspan(lower, upper, alpha=0.2)
			ax.set_xlim([p_min, p_max])
			ax.set_ylim([p_min, p_max])
			ax.set_xlabel("True Angle")
			ax.set_ylabel("Prediction Error")
			ax.set_title("Angle Error")
			
			ax1.axvspan(lower, upper, alpha=0.2)
			ax1.set_xlim([p_min, p_max])
			ax1.set_ylim([p_min, p_max])
			ax1.set_xlabel("True Angle")
			ax1.set_ylabel("Predicted Angle")
			ax1.set_title("Angle Path")
			
			for model_id in CONFIG.pipelines:
				pipeline = CONFIG.pipelines[model_id]
				
				if pipeline.model_id != "deepset":
					continue
				
				try:
					model = pipeline.model_type().cuda()
					model.Load(CONFIG.model_base_path + pipeline.model_id + "_a" + str(a) + ".pt")
					model.eval()
					
				except:
					print("Model not found:", CONFIG.model_base_path + pipeline.model_id + "_a" + str(a) + ".pt")
					continue
				
				#pipeline.Ablate(a, loader)
				a_imgs = loader.imgs[loader.valid_indices]
				a_poses = loader.poses[loader.valid_indices]
				
				poses = model(a_imgs.reshape(a_imgs.shape[0], -1).cuda(), False, False)
				
				print(poses[0].cpu().detach().numpy(), a_poses[0].cpu().numpy())

				poses = ((poses.detach().cpu() + 0.5) * 180).int()
				truth = ((a_poses.detach().cpu() + 0.5) * 180).int()
				poses = poses[:, 0]
				
				perror = torch.abs(poses - truth)
				
				ax.plot(truth, perror, label = model_id)
				ax1.plot(truth, poses, label = model_id)
				ax1.plot(truth, truth, linestyle = 'dashed', color = "black", alpha = 0.4)
				
			ax.legend(loc = "upper left")
			ax1.legend(loc = "upper left")
			plt.show()
			fig.savefig("data/out/error_" + str(a) + ".png")
			fig1.savefig("data/out/path_" + str(a) + ".png")
			plt.close()