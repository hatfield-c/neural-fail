import numpy as np

possible_actions = {
	"help": "help",
	"compile_data": "compile_data",
	"train": "train",
	"graph_results": "graph_results",
}
possible_actions_list = list(possible_actions.keys())

############################
#	NEURAL PARAMETERS
############################

#img_size = np.array([64, 64])

#epochs = 100000

#batch_size = 64
#set_size = (img_size[0] * img_size[1]) // 1
#embed_size = 1024

#snapshot_every = 1000
#print_every_epoch = 1000

#save_snapshots = [ 0, 200, 400, 600, 800 ]
#for i in range(snapshot_every, epochs + 1, snapshot_every):
#	save_snapshots.append(i)

############################
#	PIPELINES
############################

#import CONFIG_Pipelines

#train_pipelines = [
#	CONFIG_Pipelines.train_pipelines_chess,
#	CONFIG_Pipelines.train_pipelines_kitchen,
#	CONFIG_Pipelines.train_pipelines_stairs,
#	CONFIG_Pipelines.train_pipelines_church,
#	CONFIG_Pipelines.train_pipelines_street,
#	CONFIG_Pipelines.train_pipelines_hospital,
#]

#validation_pipelines = [
#	CONFIG_Pipelines.validation_pipelines_chess,
#	CONFIG_Pipelines.validation_pipelines_kitchen,
#	CONFIG_Pipelines.validation_pipelines_stairs,
#	CONFIG_Pipelines.validation_pipelines_church,
#	CONFIG_Pipelines.validation_pipelines_street,
#	CONFIG_Pipelines.validation_pipelines_hospital
#]
