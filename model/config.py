import torch
import torch.nn as nn
import torch.optim as optim

class RLModel:
	N_inputs = 1
	N_outputs = 5
	output_activation = nn.Sigmoid()
	loss = nn.BCELoss

class Scan:
	pass

class Storage:
	out_loc = 'store'
	paramset_prefix = 'PARAMS'

	model_lock_file = 'PARAMSWRITTEN.log'
	test_lock_file = 'TESTDONE.log'