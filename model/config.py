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

class Podman:
	base_url = 'tcp:localhost:8080'

class Storage:
	volume_loc = '/home/sanjay/GCCOptimization/gccoptim/store/dynamic_O3' #all logs and results written here
	test_container_loc = '/home/user/store' #mount point in container namespace

	#both of these are copied to the container using a shared volume
	test_script = '/home/sanjay/GCCOptimization/gccoptim/container-testsuite/run_experiments.sh' #local script to run
	user_config = '/home/sanjay/GCCOptimization/gccoptim/store/user-config.xml' #local user config

class Containers:
	test_image = 'gcc_testsuite'