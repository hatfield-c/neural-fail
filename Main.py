import argparse
import time

import CONFIG

import DataCompiler
import Trainer
import Tester

def GetCliAction():
	print("")
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("-a", "--action", type = str, help = "what action to take. must be one of the following: " + str(CONFIG.possible_actions_list))
	arg_parser.add_argument("-p", "--pipe", type = str, help = "what model pipeline to use. must be one of the following: " + str([]))

	args = arg_parser.parse_args()

	action = args.action
	pipe_id = args.pipe

	if action is None:
		print("	[Error]: You need to specify an action with the --action argument, i.e. --action [flag].") 
		print("	For more information, try: --help")
		exit()
		
	#if action is None and (action == CONFIG.actions["train"] or action == CONFIG.actions["view_data"] or action == CONFIG.actions["validate_model"] or action == CONFIG.actions["graph_results"]):
	#	print("	[Error]: You need to specify a model with the --model argument, i.e. --model [string].") 
	#	print("	For more information, try: --action help")
	#	exit()

	return action, pipe_id

def Main():
	
	start_time = time.time()

	actions = CONFIG.possible_actions
	action, pipe_id = GetCliAction()

	if action not in actions:
		action = actions["help"]

	if action == actions["compile_star"]:
		compiler = DataCompiler.DataCompiler()
		compiler.CompileStar()

	if action == actions["compile_data"]:
		compiler = DataCompiler.DataCompiler()
		compiler.Compile()

	if action == actions["train"]:
		trainer = Trainer.Trainer()
		trainer.Train(pipe_id)
		
	if action == actions["test"]:
		tester = Tester.Tester()
		tester.Test(pipe_id)

	runtime = time.time() - start_time
	runtime = "{:.2f}".format(runtime)

	print("\n[" + action + "]: Operation complete")
	print("    Total runtime: " + runtime + " sec\n")

Main()
