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
	volume_loc = '/home/sanjay/GCCOptimization/gccoptim/store/dynamic'
	test_container_loc = '/home/user/store'

	test_script = '/home/sanjay/GCCOptimization/gccoptim/container-testsuite/run_experiments.sh'
	user_config = '/home/sanjay/GCCOptimization/gccoptim/store/user-config.xml'

class Containers:
	test_image = 'gcc_testsuite'