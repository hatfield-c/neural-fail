import torch
import time

import CONFIG
import DataLoader
import ModelLinear

class Trainer:
	def __init__(self):
		pass
	
	def Train(self, model_id):
		pipeline = CONFIG.pipelines[model_id]
		
		loader = DataLoader.DataLoader()
		
		print("Training...")
		
		avg_time = 1
		start_total = time.time()
		
		for a in range(pipeline.ablation_count):
			
			if a != 2:
				continue
			
			losser = pipeline.losser_type()
			model = pipeline.model_type().cuda()
		
			optimizer = torch.optim.Adam(
				model.parameters(),
				lr = pipeline.learning_rate,
			)	
		
			pipeline.Ablate(a, loader)
			
			for e in range(pipeline.epochs + 1):
				start_time = time.time()
		
				img, pose = loader.DrawSamples(CONFIG.batch_size)
				
				preds = model(img)
				
				loss = losser(preds, pose)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				if e == 0:
					avg_time = (time.time() - start_time) / 7
				
				self.PrintUpdate(pipeline.epochs, e, pipeline.print_every_epoch, avg_time, loss)
				
				if e % pipeline.print_every_epoch == 0:
					print(preds[0].cpu().detach().numpy(), pose[0].cpu().numpy())
				
				avg_time = (avg_time + (time.time() - start_time)) / 2
				
				#if e == save_snapshots[0]:
				#	model.Save(pipeline.model_base_path + "/" + pipeline.pipeline_id + "_e" + str(e) + ".pt")
				#	save_snapshots.pop(0)
		
			model.Save(CONFIG.model_base_path + pipeline.model_id + "_a" + str(a) + ".pt")
		
			print("\nCompleted in", int((time.time() - start_total) / 60), "minutes.")
			print("Final loss:", loss.item())
		
	def PrintUpdate(self, epochs, e, p, avg_time, loss):

		remaining_epochs = epochs - e

		if self.ShouldPrint_E(epochs, e, p):
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

	def ShouldPrint_E(self, epochs, e, p):
		if epochs < p:
			return True

		return e % p == 0 or e == epochs - 1
