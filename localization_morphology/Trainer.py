import torch
import time

import CONFIG
import DataLoader
import ModelLinear

class Trainer:
	def __init__(self):
		pass
	
	def Train(self, epochs):
		losser = torch.nn.MSELoss()
		model = ModelLinear.ModelLinear().cuda()
		loader = DataLoader.DataLoader()
		pipeline = CONFIG.linear_pipeline
		
		optimizer = torch.optim.Adam(
			model.parameters(),
			lr = CONFIG.learning_rate,
		)
		
		print("Training...")
		
		avg_time = 1
		start_total = time.time()
		
		for a in range(pipeline.ablation_count):
			pipeline.Ablate(a, loader)
			
			for e in range(epochs + 1):
				start_time = time.time()
		
				img, pose = loader.DrawSamples(CONFIG.batch_size)
				preds = model(img)
		
				loss = losser(preds, pose)
		
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				if e == 0:
					avg_time = (time.time() - start_time) / 7
				
				self.PrintUpdate(epochs, e, avg_time, loss)
		
				avg_time = (avg_time + (time.time() - start_time)) / 2
				
				#if e == save_snapshots[0]:
				#	model.Save(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_e" + str(e) + ".pt")
				#	save_snapshots.pop(0)
		
			model.Save(CONFIG.model_base_path + pipeline.model_id + "_a" + str(a) + ".pt")
		
			print("\nCompleted in", int((time.time() - start_total) / 60), "minutes.")
			print("Final loss:", loss.item())
		
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
