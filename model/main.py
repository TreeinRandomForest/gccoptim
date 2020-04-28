import numpy as np
import model, config
import time
import os

def read_options(filename):
	gcc_version, params = paramset.get_gcc_options()

	print(f'Using gcc {gcc_version}')

	return params

def check_test_suite_finished():
	return os.path.exists(config.Storage.test_lock_file)

def generate_param_set():
	return [{'asan-globals': 0},
	 		{'early-inlining-insns': 0}]

def write_params_to_file(params):
	for idx, paramset in enumerate(params):
		with open(f'{config.Storage.paramset_prefix}{idx}', 'w') as f:
			for elem in paramset:
				print(f'--param {elem}={paramset[elem]}', file=f)

def generate_model_lock_file(counter):
	with open(config.Storage.model_lock_file, 'w') as f:
		print(counter, time.time(), file=f)

def remove_test_lock_file():
	os.remove(config.Storage.test_lock_file)

def run():
	counter = 0
	while True:
		counter += 1
		while not check_test_suite_finished():
			time.sleep(1)
			print('Waiting')

		params_to_try = generate_param_set()

		write_params_to_file(params_to_try)

		generate_model_lock_file(counter)

		remove_test_lock_file()

if __name__=="__main__":
	run()
