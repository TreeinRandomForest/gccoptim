import torch
import torch.nn as nn
import torch.optim as optim
import os

class RLModel:
	N_inputs = 1
	N_outputs = 5
	output_activation = nn.Sigmoid()
	loss = nn.BCELoss

class Scan:
	pass

class Storage:
	out_loc = '/home/user/store'
	paramset_prefix = os.path.join(out_loc, 'PARAMS')

	model_lock_file = os.path.join(out_loc, 'PARAMSWRITTEN.log')
	test_lock_file = os.path.join(out_loc, 'TESTDONE.log')