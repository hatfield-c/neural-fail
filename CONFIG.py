import numpy as np

import Pipeline

possible_actions = {
	"help": "help",
	"compile_star": "compile_star",
	"train": "train",
	"test": "test",
}
possible_actions_list = list(possible_actions.keys())

############################
#	NEURAL PARAMETERS
############################

model_base_path = "data/models/"

img_size = np.array([64, 64])

batch_size = 64

############################
#	PIPELINES
############################

#star_ablations = [[103, 103], [83, 123], [23, 183]]
star_ablations = [[103, 103], [83, 123], [83, 123]]
scenes_ablations = [[1, 2], [1, 4], [1, 8]]

pipelines = {
	### Star pipelines
	"linear_star": Pipeline.Pipeline("linear", "star", 1e-3, 1000, 100, star_ablations),
	"linear_norm_star": Pipeline.Pipeline("linear_norm", "star", 1e-3, 1000, 100, star_ablations),
	"vgg16_star": Pipeline.Pipeline("vgg16", "star", 1e-5, 1000, 10, star_ablations),
	"vit_star": Pipeline.Pipeline("vit", "star", 1e-3, 1000, 100, star_ablations),
	"vim_star": Pipeline.Pipeline("vim", "star", 1e-5, 1000, 10, star_ablations),
	"deepset_star": Pipeline.Pipeline("deepset", "star", 1e-3, 1000, 100, star_ablations),
	
	### Chess pipelines
	"linear_chess": Pipeline.Pipeline("linear", "chess", 1e-3, 3000, 100, scenes_ablations),
	"linear_norm_chess": Pipeline.Pipeline("linear_norm", "chess", 1e-3, 3000, 100, scenes_ablations),
	"vgg16_chess": Pipeline.Pipeline("vgg16", "chess", 1e-5, 3000, 100, scenes_ablations),
	"deepset_chess": Pipeline.Pipeline("deepset", "chess", 1e-3, 3000, 100, scenes_ablations),
}
