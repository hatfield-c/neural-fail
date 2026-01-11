import ModelPipeline


############################
#	7SCENES - CHESS
############################

chess_train_counts = [ 1000, 1000, 1000, 1000 ]
chess_train = [ "seq-01/", "seq-02/", "seq-04/", "seq-06/" ]
chess_validate_counts = [ 1000, 1000 ]
chess_validate = [ "seq-03/", "seq-05/" ]
chess_colors = [(75, 68, 180), (116, 153, 179), (64, 54, 57), (207, 205, 201)]

train_pipelines_chess = [
	ModelPipeline.ModelPipeline(1e-3, "chess_deepset_boosted", "deepset", "chess", "7scenes", "data/models/chess_deepset_boosted", "data/samples/chess", chess_train, chess_train_counts, True, chess_colors),
	ModelPipeline.ModelPipeline(1e-3, "chess_deepset_noboost", "deepset", "chess", "7scenes", "data/models/chess_deepset_noboost", "data/samples/chess", chess_train, chess_train_counts, False, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg16_boosted", "vgg16", "chess", "7scenes", "data/models/chess_vgg16_boosted", "data/samples/chess", chess_train, chess_train_counts, True, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg16_noboost", "vgg16", "chess", "7scenes", "data/models/chess_vgg16_noboost", "data/samples/chess", chess_train, chess_train_counts, False, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg6_boosted", "vgg6", "chess", "7scenes", "data/models/chess_vgg6_boosted", "data/samples/chess", chess_train, chess_train_counts, True, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg6_noboost", "vgg6", "chess", "7scenes", "data/models/chess_vgg6_noboost", "data/samples/chess", chess_train, chess_train_counts, False, chess_colors),
	#ModelPipeline.ModelPipeline(1e-3, "chess_linear_boosted", "linear", "chess", "7scenes", "data/models/chess_linear_boosted", "data/samples/chess", chess_train, chess_train_counts, True, chess_colors),
	#ModelPipeline.ModelPipeline(1e-3, "chess_linear_noboost", "linear", "chess", "7scenes", "data/models/chess_linear_noboost", "data/samples/chess", chess_train, chess_train_counts, False, chess_colors),
]

validation_pipelines_chess = [
	ModelPipeline.ModelPipeline(1e-3, "chess_deepset_boosted", "deepset", "chess", "7scenes", "data/models/chess_deepset_boosted", "data/samples/chess", chess_validate, chess_validate_counts, True, chess_colors),
	ModelPipeline.ModelPipeline(1e-3, "chess_deepset_noboost", "deepset", "chess", "7scenes", "data/models/chess_deepset_noboost", "data/samples/chess", chess_validate, chess_validate_counts, False, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg16_boosted", "vgg16", "chess", "7scenes", "data/models/chess_vgg16_boosted", "data/samples/chess", chess_validate, chess_validate_counts, True, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg16_noboost", "vgg16", "chess", "7scenes", "data/models/chess_vgg16_noboost", "data/samples/chess", chess_validate, chess_validate_counts, False, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg6_boosted", "vgg6", "chess", "7scenes", "data/models/chess_vgg6_boosted", "data/samples/chess", chess_validate, chess_validate_counts, True, chess_colors),
	#ModelPipeline.ModelPipeline(1e-5, "chess_vgg6_noboost", "vgg6", "chess", "7scenes", "data/models/chess_vgg6_noboost", "data/samples/chess", chess_validate, chess_validate_counts, False, chess_colors),
	#ModelPipeline.ModelPipeline(1e-3, "chess_linear_boosted", "linear", "chess", "7scenes", "data/models/chess_linear_boosted", "data/samples/chess", chess_validate, chess_validate_counts, True, chess_colors),
	#ModelPipeline.ModelPipeline(1e-3, "chess_linear_noboost", "linear", "chess", "7scenes", "data/models/chess_linear_noboost", "data/samples/chess", chess_validate, chess_validate_counts, False, chess_colors),
]

############################
#	7SCENES - KITCHEN
############################

kitchen_train_counts = [ 1000, 1000, 1000, 1000, 1000, 1000, 1000 ]
kitchen_train = [ "seq-01/", "seq-02/", "seq-05/", "seq-07/", "seq-08/", "seq-11/", "seq-13/" ]
kitchen_validate_counts = [ 1000, 1000, 1000, 1000, 1000 ]
kitchen_validate = [ "seq-03/", "seq-04/", "seq-06/", "seq-12/", "seq-14/" ]
kitchen_colors = [ (225, 217, 211), (127, 118, 105), (38, 40, 110), (123, 149, 169) ]

train_pipelines_kitchen = [
	ModelPipeline.ModelPipeline(1e-3, "kitchen_deepset_boosted", "deepset", "kitchen", "7scenes", "data/models/kitchen_deepset_boosted", "data/samples/kitchen", kitchen_train, kitchen_train_counts, True, kitchen_colors),
	ModelPipeline.ModelPipeline(1e-3, "kitchen_deepset_noboost", "deepset", "kitchen", "7scenes", "data/models/kitchen_deepset_noboost", "data/samples/kitchen", kitchen_train, kitchen_train_counts, False, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg16_boosted", "vgg16", "kitchen", "7scenes", "data/models/kitchen_vgg16_boosted", "data/samples/kitchen", kitchen_train, kitchen_train_counts, True, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg16_noboost", "vgg16", "kitchen", "7scenes", "data/models/kitchen_vgg16_noboost", "data/samples/kitchen", kitchen_train, kitchen_train_counts, False, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg6_boosted", "vgg6", "kitchen", "7scenes", "data/models/kitchen_vgg6_boosted", "data/samples/kitchen", kitchen_train, kitchen_train_counts, True, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg6_noboost", "vgg6", "kitchen", "7scenes", "data/models/kitchen_vgg6_noboost", "data/samples/kitchen", kitchen_train, kitchen_train_counts, False, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-3, "kitchen_linear_boosted", "linear", "kitchen", "7scenes", "data/models/kitchen_linear_boosted", "data/samples/kitchen", kitchen_train, kitchen_train_counts, True, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-3, "kitchen_linear_noboost", "linear", "kitchen", "7scenes", "data/models/kitchen_linear_noboost", "data/samples/kitchen", kitchen_train, kitchen_train_counts, False, kitchen_colors),
]

validation_pipelines_kitchen = [
	ModelPipeline.ModelPipeline(1e-3, "kitchen_deepset_boosted", "deepset", "kitchen", "7scenes", "data/models/kitchen_deepset_boosted", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, True, kitchen_colors),
	ModelPipeline.ModelPipeline(1e-3, "kitchen_deepset_noboost", "deepset", "kitchen", "7scenes", "data/models/kitchen_deepset_noboost", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, False, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg16_boosted", "vgg16", "kitchen", "7scenes", "data/models/kitchen_vgg16_boosted", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, True, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg16_noboost", "vgg16", "kitchen", "7scenes", "data/models/kitchen_vgg16_noboost", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, False, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg6_boosted", "vgg6", "kitchen", "7scenes", "data/models/kitchen_vgg6_boosted", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, True, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-5, "kitchen_vgg6_noboost", "vgg6", "kitchen", "7scenes", "data/models/kitchen_vgg6_noboost", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, False, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-3, "kitchen_linear_boosted", "linear", "kitchen", "7scenes", "data/models/kitchen_linear_boosted", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, True, kitchen_colors),
	#ModelPipeline.ModelPipeline(1e-3, "kitchen_linear_noboost", "linear", "kitchen", "7scenes", "data/models/kitchen_linear_noboost", "data/samples/kitchen", kitchen_validate, kitchen_validate_counts, False, kitchen_colors),
]

############################
#	7SCENES - STAIRS
############################

stairs_train_counts = [ 500, 500, 500, 500 ]
stairs_train = [ "seq-02/", "seq-03/", "seq-05/", "seq-06/" ]
stairs_validate_counts = [ 500, 500 ]
stairs_validate = [ "seq-01/", "seq-04/" ]
stairs_colors = [ (124, 180, 200), (110, 100, 100), (40, 30, 30), (170, 160, 160) ]

train_pipelines_stairs = [
	ModelPipeline.ModelPipeline(1e-3, "stairs_deepset_boosted", "deepset", "stairs", "7scenes", "data/models/stairs_deepset_boosted", "data/samples/stairs", stairs_train, stairs_train_counts, True, stairs_colors),
	ModelPipeline.ModelPipeline(1e-3, "stairs_deepset_noboost", "deepset", "stairs", "7scenes", "data/models/stairs_deepset_noboost", "data/samples/stairs", stairs_train, stairs_train_counts, False, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg16_boosted", "vgg16", "stairs", "7scenes", "data/models/stairs_vgg16_boosted", "data/samples/stairs", stairs_train, stairs_train_counts, True, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg16_noboost", "vgg16", "stairs", "7scenes", "data/models/stairs_vgg16_noboost", "data/samples/stairs", stairs_train, stairs_train_counts, False, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg6_boosted", "vgg6", "stairs", "7scenes", "data/models/stairs_vgg6_boosted", "data/samples/stairs", stairs_train, stairs_train_counts, True, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg6_noboost", "vgg6", "stairs", "7scenes", "data/models/stairs_vgg6_noboost", "data/samples/stairs", stairs_train, stairs_train_counts, False, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-3, "stairs_linear_boosted", "linear", "stairs", "7scenes", "data/models/stairs_linear_boosted", "data/samples/stairs", stairs_train, stairs_train_counts, True, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-3, "stairs_linear_noboost", "linear", "stairs", "7scenes", "data/models/stairs_linear_noboost", "data/samples/stairs", stairs_train, stairs_train_counts, False, stairs_colors),
]

validation_pipelines_stairs = [
	ModelPipeline.ModelPipeline(1e-3, "stairs_deepset_boosted", "deepset", "stairs", "7scenes", "data/models/stairs_deepset_boosted", "data/samples/stairs", stairs_validate, stairs_validate_counts, True, stairs_colors),
	ModelPipeline.ModelPipeline(1e-3, "stairs_deepset_noboost", "deepset", "stairs", "7scenes", "data/models/stairs_deepset_noboost", "data/samples/stairs", stairs_validate, stairs_validate_counts, False, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg16_boosted", "vgg16", "stairs", "7scenes", "data/models/stairs_vgg16_boosted", "data/samples/stairs", stairs_validate, stairs_validate_counts, True, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg16_noboost", "vgg16", "stairs", "7scenes", "data/models/stairs_vgg16_noboost", "data/samples/stairs", stairs_validate, stairs_validate_counts, False, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg6_boosted", "vgg6", "stairs", "7scenes", "data/models/stairs_vgg6_boosted", "data/samples/stairs", stairs_validate, stairs_validate_counts, True, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-5, "stairs_vgg6_noboost", "vgg6", "stairs", "7scenes", "data/models/stairs_vgg6_noboost", "data/samples/stairs", stairs_validate, stairs_validate_counts, False, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-3, "stairs_linear_boosted", "linear", "stairs", "7scenes", "data/models/stairs_linear_boosted", "data/samples/stairs", stairs_validate, stairs_validate_counts, True, stairs_colors),
	#ModelPipeline.ModelPipeline(1e-3, "stairs_linear_noboost", "linear", "stairs", "7scenes", "data/models/stairs_linear_noboost", "data/samples/stairs", stairs_validate, stairs_validate_counts, False, stairs_colors),
]

############################
#	CAMBRIDGE - CHURCH
############################

church_train_counts = [ 0 ]
church_train = [ "dataset_train.txt" ]
church_validate_counts = [ 0 ]
church_validate = [ "dataset_test.txt" ]
church_colors = [ (242, 239, 230), (58, 80, 94), (52, 57, 46), (140, 165, 178) ]

train_pipelines_church = [
	ModelPipeline.ModelPipeline(1e-3, "church_deepset_boosted", "deepset", "church", "7scenes", "data/models/church_deepset_boosted", "data/samples/church", church_train, church_train_counts, True, church_colors),
	ModelPipeline.ModelPipeline(1e-3, "church_deepset_noboost", "deepset", "church", "7scenes", "data/models/church_deepset_noboost", "data/samples/church", church_train, church_train_counts, False, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg16_boosted", "vgg16", "church", "7scenes", "data/models/church_vgg16_boosted", "data/samples/church", church_train, church_train_counts, True, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg16_noboost", "vgg16", "church", "7scenes", "data/models/church_vgg16_noboost", "data/samples/church", church_train, church_train_counts, False, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg6_boosted", "vgg6", "church", "7scenes", "data/models/church_vgg6_boosted", "data/samples/church", church_train, church_train_counts, True, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg6_noboost", "vgg6", "church", "7scenes", "data/models/church_vgg6_noboost", "data/samples/church", church_train, church_train_counts, False, church_colors),
	#ModelPipeline.ModelPipeline(1e-3, "church_linear_boosted", "linear", "church", "7scenes", "data/models/church_linear_boosted", "data/samples/church", church_train, church_train_counts, True, church_colors),
	#ModelPipeline.ModelPipeline(1e-3, "church_linear_noboost", "linear", "church", "7scenes", "data/models/church_linear_noboost", "data/samples/church", church_train, church_train_counts, False, church_colors),
]

validation_pipelines_church = [
	ModelPipeline.ModelPipeline(1e-3, "church_deepset_boosted", "deepset", "church", "7scenes", "data/models/church_deepset_boosted", "data/samples/church", church_validate, church_validate_counts, True, church_colors),
	ModelPipeline.ModelPipeline(1e-3, "church_deepset_noboost", "deepset", "church", "7scenes", "data/models/church_deepset_noboost", "data/samples/church", church_validate, church_validate_counts, False, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg16_boosted", "vgg16", "church", "7scenes", "data/models/church_vgg16_boosted", "data/samples/church", church_validate, church_validate_counts, True, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg16_noboost", "vgg16", "church", "7scenes", "data/models/church_vgg16_noboost", "data/samples/church", church_validate, church_validate_counts, False, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg6_boosted", "vgg6", "church", "7scenes", "data/models/church_vgg6_boosted", "data/samples/church", church_validate, church_validate_counts, True, church_colors),
	#ModelPipeline.ModelPipeline(1e-5, "church_vgg6_noboost", "vgg6", "church", "7scenes", "data/models/church_vgg6_noboost", "data/samples/church", church_validate, church_validate_counts, False, church_colors),
	#ModelPipeline.ModelPipeline(1e-3, "church_linear_boosted", "linear", "church", "7scenes", "data/models/church_linear_boosted", "data/samples/church", church_validate, church_validate_counts, True, church_colors),
	#ModelPipeline.ModelPipeline(1e-3, "church_linear_noboost", "linear", "church", "7scenes", "data/models/church_linear_noboost", "data/samples/church", church_validate, church_validate_counts, False, church_colors),
]

############################
#	CAMBRIDGE - STREET
############################

street_train_counts = [ 0 ]
street_train = [ "dataset_train.txt" ]
street_validate_counts = [ 0 ]
street_validate = [ "dataset_test.txt" ]
street_colors = [ (228, 224, 215), (41, 36, 43), (141, 176, 195), (39, 54, 100) ]

train_pipelines_street = [
	ModelPipeline.ModelPipeline(1e-3, "street_deepset_boosted", "deepset", "street", "cambridge", "data/models/street_deepset_boosted", "data/samples/street", street_train, street_train_counts, True, street_colors),
	ModelPipeline.ModelPipeline(1e-3, "street_deepset_noboost", "deepset", "street", "cambridge", "data/models/street_deepset_noboost", "data/samples/street", street_train, street_train_counts, False, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg16_boosted", "vgg16", "street", "cambridge", "data/models/street_vgg16_boosted", "data/samples/street", street_train, street_train_counts, True, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg16_noboost", "vgg16", "street", "cambridge", "data/models/street_vgg16_noboost", "data/samples/street", street_train, street_train_counts, False, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg6_boosted", "vgg6", "street", "cambridge", "data/models/street_vgg6_boosted", "data/samples/street", street_train, street_train_counts, True, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg6_noboost", "vgg6", "street", "cambridge", "data/models/street_vgg6_noboost", "data/samples/street", street_train, street_train_counts, False, street_colors),
	#ModelPipeline.ModelPipeline(1e-3, "street_linear_boosted", "linear", "street", "cambridge", "data/models/street_linear_boosted", "data/samples/street", street_train, street_train_counts, True, street_colors),
	#ModelPipeline.ModelPipeline(1e-3, "street_linear_noboost", "linear", "street", "cambridge", "data/models/street_linear_noboost", "data/samples/street", street_train, street_train_counts, False, street_colors),
]

validation_pipelines_street = [
	ModelPipeline.ModelPipeline(1e-3, "street_deepset_boosted", "deepset", "street", "cambridge", "data/models/street_deepset_boosted", "data/samples/street", street_validate, street_validate_counts, True, street_colors),
	ModelPipeline.ModelPipeline(1e-3, "street_deepset_noboost", "deepset", "street", "cambridge", "data/models/street_deepset_noboost", "data/samples/street", street_validate, street_validate_counts, False, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg16_boosted", "vgg16", "street", "cambridge", "data/models/street_vgg16_boosted", "data/samples/street", street_validate, street_validate_counts, True, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg16_noboost", "vgg16", "street", "cambridge", "data/models/street_vgg16_noboost", "data/samples/street", street_validate, street_validate_counts, False, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg6_boosted", "vgg6", "street", "cambridge", "data/models/street_vgg6_boosted", "data/samples/street", street_validate, street_validate_counts, True, street_colors),
	#ModelPipeline.ModelPipeline(1e-5, "street_vgg6_noboost", "vgg6", "street", "cambridge", "data/models/street_vgg6_noboost", "data/samples/street", street_validate, street_validate_counts, False, street_colors),
	#ModelPipeline.ModelPipeline(1e-3, "street_linear_boosted", "linear", "street", "cambridge", "data/models/street_linear_boosted", "data/samples/street", street_validate, street_validate_counts, True, street_colors),
	#ModelPipeline.ModelPipeline(1e-3, "street_linear_noboost", "linear", "street", "cambridge", "data/models/street_linear_noboost", "data/samples/street", street_validate, street_validate_counts, False, street_colors),
]

############################
#	CAMBRIDGE - HOSPITAL
############################

hospital_train_counts = [ 0 ]
hospital_train = [ "dataset_train.txt" ]
hospital_validate_counts = [ 0 ]
hospital_validate = [ "dataset_test.txt" ]
hospital_colors = [(235, 218, 205), (144, 182, 202), (60, 81, 103), (31, 47, 115)]

train_pipelines_hospital = [
	ModelPipeline.ModelPipeline(1e-3, "hospital_deepset_boosted", "deepset", "hospital", "cambridge", "data/models/hospital_deepset_boosted", "data/samples/hospital", hospital_train, hospital_train_counts, True, hospital_colors),
	ModelPipeline.ModelPipeline(1e-3, "hospital_deepset_noboost", "deepset", "hospital", "cambridge", "data/models/hospital_deepset_noboost", "data/samples/hospital", hospital_train, hospital_train_counts, False, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg16_boosted", "vgg16", "hospital", "cambridge", "data/models/hospital_vgg16_boosted", "data/samples/hospital", hospital_train, hospital_train_counts, True, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg16_noboost", "vgg16", "hospital", "cambridge", "data/models/hospital_vgg16_noboost", "data/samples/hospital", hospital_train, hospital_train_counts, False, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg6_boosted", "vgg6", "hospital", "cambridge", "data/models/hospital_vgg6_boosted", "data/samples/hospital", hospital_train, hospital_train_counts, True, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg6_noboost", "vgg6", "hospital", "cambridge", "data/models/hospital_vgg6_noboost", "data/samples/hospital", hospital_train, hospital_train_counts, False, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-3, "hospital_linear_boosted", "linear", "hospital", "cambridge", "data/models/hospital_linear_boosted", "data/samples/hospital", hospital_train, hospital_train_counts, True, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-3, "hospital_linear_noboost", "linear", "hospital", "cambridge", "data/models/hospital_linear_noboost", "data/samples/hospital", hospital_train, hospital_train_counts, False, hospital_colors),
]

validation_pipelines_hospital = [
	ModelPipeline.ModelPipeline(1e-3, "hospital_deepset_boosted", "deepset", "hospital", "cambridge", "data/models/hospital_deepset_boosted", "data/samples/hospital", hospital_validate, hospital_validate_counts, True, hospital_colors),
	ModelPipeline.ModelPipeline(1e-3, "hospital_deepset_noboost", "deepset", "hospital", "cambridge", "data/models/hospital_deepset_noboost", "data/samples/hospital", hospital_validate, hospital_validate_counts, False, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg16_boosted", "vgg16", "hospital", "cambridge", "data/models/hospital_vgg16_boosted", "data/samples/hospital", hospital_validate, hospital_validate_counts, True, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg16_noboost", "vgg16", "hospital", "cambridge", "data/models/hospital_vgg16_noboost", "data/samples/hospital", hospital_validate, hospital_validate_counts, False, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg6_boosted", "vgg6", "hospital", "cambridge", "data/models/hospital_vgg6_boosted", "data/samples/hospital", hospital_validate, hospital_validate_counts, True, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-5, "hospital_vgg6_noboost", "vgg6", "hospital", "cambridge", "data/models/hospital_vgg6_noboost", "data/samples/hospital", hospital_validate, hospital_validate_counts, False, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-3, "hospital_linear_boosted", "linear", "hospital", "cambridge", "data/models/hospital_linear_boosted", "data/samples/hospital", hospital_validate, hospital_validate_counts, True, hospital_colors),
	#ModelPipeline.ModelPipeline(1e-3, "hospital_linear_noboost", "linear", "hospital", "cambridge", "data/models/hospital_linear_noboost", "data/samples/hospital", hospital_validate, hospital_validate_counts, False, hospital_colors),
]