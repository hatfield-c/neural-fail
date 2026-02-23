import numpy as np

import Pipeline

possible_actions = {
	"help": "help",
	"compile_data": "compile_data",
	"train": "train",
	"test": "test",
}
possible_actions_list = list(possible_actions.keys())

############################
#	NEURAL PARAMETERS
############################

model_base_path = "data/models/"

learning_rate = 1e-3

img_size = np.array([64, 256])

epochs = 1000
batch_size = 64

print_every_epoch = 1000

############################
#	PIPELINES
############################

linear_pipeline = Pipeline.Pipeline("linear", [[103, 103], [47, 156]])