import argparse
import time

import CONFIG

import DataCompiler
#import Trainer
#import Grapher

def GetCliAction():
	print("")
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("-a", "--action", type = str, help = "what action to take. must be one of the following: " + str(CONFIG.possible_actions_list))
	arg_parser.add_argument("-g", "--group", type = int, help = "what pipeline group to perform the action on. must be between 0 and " + str(len(CONFIG.train_pipelines)))

	args = arg_parser.parse_args()

	action = args.action
	group = args.group

	if action is None:
		print("	[Error]: You need to specify an action with the --action argument, i.e. --action [flag].") 
		print("	For more information, try: --help")
		exit()
		
	if action is None and (action == CONFIG.actions["train"] or action == CONFIG.actions["view_data"] or action == CONFIG.actions["validate_model"] or action == CONFIG.actions["graph_results"]):
		print("	[Error]: You need to specify a group with the --group argument, i.e. --group [flag].") 
		print("	For more information, try: --action help")
		exit()

	return action, group

def Main():
	print("Hello world!")
	exit()
	
	start_time = time.time()

	actions = CONFIG.possible_actions
	action, group = GetCliAction()

	if action not in actions:
		action = actions["help"]

	if action == actions["compile_data"]:
		compiler = DataCompiler.DataCompiler(group)
		compiler.Compile()

	#if action == actions["train"]:
	#	trainer = Trainer.Trainer(group)
	#	trainer.Train(CONFIG.epochs)
		
	#if action == actions["graph_results"]:
	#	grapher = Grapher.Grapher(group)
	#	grapher.Graph()

	runtime = time.time() - start_time
	runtime = "{:.2f}".format(runtime)

	print("\n[" + action + "]: Operation complete")
	print("    Total runtime: " + runtime + " sec\n")

Main()
