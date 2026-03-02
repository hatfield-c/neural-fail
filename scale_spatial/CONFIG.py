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

img_size = np.array([128, 128])

batch_size = 64

############################
#	PIPELINES
############################

ablations = [[45, 45], [30, 60], [20, 70]]

linear_pipeline = Pipeline.Pipeline("linear", 1e-3, 10000, 1000, ablations)
linear_norm_pipeline = Pipeline.Pipeline("linear_norm", 1e-3, 10000, 1000, ablations)
vgg16_pipeline = Pipeline.Pipeline("vgg16", 1e-5, 1000, 100, ablations)
deepset_pipeline = Pipeline.Pipeline("deepset", 1e-3, 2000, 100, ablations)

pipelines = {
	linear_pipeline.model_id: linear_pipeline,
	linear_norm_pipeline.model_id: linear_norm_pipeline,
	vgg16_pipeline.model_id: vgg16_pipeline,
	deepset_pipeline.model_id: deepset_pipeline,
}