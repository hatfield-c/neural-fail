import numpy as np

import Pipeline

possible_actions = {
	"help": "help",
	"compile_star": "compile_star",
	"compile_data": "compile_data",
	"train": "train",
	"test": "test",
}
possible_actions_list = list(possible_actions.keys())

############################
#	NEURAL PARAMETERS
############################

model_base_path = "data/models/"

batch_size = 8

############################
#	PIPELINES
############################

star_ablations = [[83, 42], [43, 122], [23, 162]]
scenes_ablations = [[1, 2], [1, 4], [1, 8]]
fire_ablations = [[1, 2], [1, 4], [4, 16]]

target_ablation = 2
#target_ablation = None

vgg16_color = "blue"
deepset_color = "magenta"

pipelines = {
	### Star pipelines
	"linear_star": Pipeline.Pipeline("linear", "star", 1e-3, 1000, 100, star_ablations),
	"linear_norm_star": Pipeline.Pipeline("linear_norm", "star", 1e-3, 1000, 100, star_ablations),
	"vgg16_star": Pipeline.Pipeline("vgg16", "star", 1e-5, 1000, 10, star_ablations),
	"vit_star": Pipeline.Pipeline("vit", "star", 1e-3, 1000, 100, star_ablations),
	"vim_star": Pipeline.Pipeline("vim", "star", 1e-5, 1000, 10, star_ablations),
	"deepset_star": Pipeline.Pipeline("deepset", "star", 1e-3, 1000, 100, star_ablations),
	
	### Chess pipelines
	#"linear_chess": Pipeline.Pipeline("linear", "chess", 1e-3, 3000, 100, scenes_ablations),
	#"linear_norm_chess": Pipeline.Pipeline("linear_norm", "chess", 1e-3, 3000, 100, scenes_ablations),
	"vgg16_chess": Pipeline.Pipeline("vgg16", "chess", 1e-5, 3000, 100, scenes_ablations, vgg16_color),
	"deepset_chess": Pipeline.Pipeline("dspg", "chess", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
	
	### Fire pipelines
	"vgg16_fire": Pipeline.Pipeline("vgg16", "fire", 1e-5, 3000, 100, scenes_ablations),
	"deepset_fire": Pipeline.Pipeline("dspg", "fire", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
	
	### Heads pipelines
	"vgg16_heads": Pipeline.Pipeline("vgg16", "heads", 1e-5, 3000, 100, scenes_ablations),
	"deepset_heads": Pipeline.Pipeline("dspg", "heads", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
	
	### Kitchen pipelines
	"vgg16_kitchen": Pipeline.Pipeline("vgg16", "kitchen", 1e-5, 3000, 100, scenes_ablations),
	"deepset_kitchen": Pipeline.Pipeline("dspg", "kitchen", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
	
	### Office pipelines
	"vgg16_office": Pipeline.Pipeline("vgg16", "office", 1e-5, 3000, 100, scenes_ablations, vgg16_color),
	"deepset_office": Pipeline.Pipeline("dspg", "office", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
	#"deepset_office": Pipeline.Pipeline("deepset", "office", 1e-3, 10000, 1000, scenes_ablations, deepset_color),
	
	### Pumpkin pipelines
	"vgg16_pumpkin": Pipeline.Pipeline("vgg16", "pumpkin", 1e-5, 3000, 100, scenes_ablations),
	"deepset_pumpkin": Pipeline.Pipeline("dspg", "pumpkin", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
	
	### Stairs pipelines
	"vgg16_stairs": Pipeline.Pipeline("vgg16", "stairs", 1e-5, 3000, 100, scenes_ablations),
	"deepset_stairs": Pipeline.Pipeline("dspg", "stairs", 1e-3, 30000, 1000, scenes_ablations, deepset_color),
}
