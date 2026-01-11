import time
import math
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import CONFIG
import LossDirect

class Trainer:
	def __init__(self, group):
		self.group = group

	def Train(self, epochs):
		losser = LossDirect.LossDirect()
		
		_, loader = CONFIG.train_pipelines[self.group][0].Create()
		loader.Load("train")
		
		for pipeline in CONFIG.train_pipelines[self.group]:
			model, _ = pipeline.Create()
			model = model.cuda()
			
			optimizer = optim.Adam(
				model.parameters(),
				lr = pipeline.learning_rate,
			)
			
			print("\n\nTraining pipeline", pipeline.pipeline_id)
	
			loss_vals = []
			avg_time = 1
			start_total = time.time()
			save_snapshots = CONFIG.save_snapshots.copy()
	
			for e in range(epochs + 1):
				start_time = time.time()
	
				loss = losser.GetLoss(model, loader)
	
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				if e == 0:
					avg_time = (time.time() - start_time) / 7
				
				self.PrintUpdate(epochs, e, avg_time, loss)
				loss_vals.append(loss.cpu().detach().item())
	
				avg_time = (avg_time + (time.time() - start_time)) / 2
				
				if e == save_snapshots[0]:
					model.Save(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_e" + str(e) + ".pt")
					save_snapshots.pop(0)

			loss_vals = np.array(loss_vals)
			
			np.savetxt(pipeline.model_base_path + "/" + pipeline.pipeline_id +  "_avg_epoch_time.txt", np.array([avg_time]))
			np.savetxt(pipeline.model_base_path + "/" + pipeline.pipeline_id +  "_loss_vals.txt", loss_vals)

			print("\nCompleted in", int((time.time() - start_total) / 60), "minutes.")
			print("Final loss:", loss.item())
			
		return model

	def PrintUpdate(self, epochs, e, avg_time, loss):

		remaining_epochs = epochs - e

		if self.ShouldPrint_E(epochs, e):
			eta = avg_time * remaining_epochs
			eta = eta / 60
			eta = "{:.2f}".format(eta)

			completion = str(100 *(e / (epochs)))
			completion = completion[:4] + "%"

			print("   [", e, "/", epochs, ":", completion,  "]")
			print("    Loss	 :", loss.item())
			print("")
			print("    Batches left :", remaining_epochs)
			print("    Avg. Time    :", "{:.2f}".format(avg_time), "s")
			print("    ETA	      :", eta, "mins")
			print("\n")

	def ShouldPrint_E(self, epochs, e):
		if epochs < CONFIG.print_every_epoch:
			return True

		return e % CONFIG.print_every_epoch == 0 or e == epochs - 1
