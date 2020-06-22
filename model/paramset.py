import subprocess
import os

def get_gcc_params():
	'''Get list of available paramters, descriptions and ranges
	Hacky implementation
	'''
	gcc_version = subprocess.Popen(['gcc', '--version'], stdout=subprocess.PIPE)
	params_desc = subprocess.Popen(['gcc', '--help=params'], stdout=subprocess.PIPE)
	params_range = subprocess.Popen(['gcc', '--help=params', '-Q'], stdout=subprocess.PIPE)

	gcc_version = parse_version(gcc_version.stdout.read().decode())
	params_desc = params_desc.stdout.read().decode()
	params_range = params_range.stdout.read().decode()

	params = parse_params(params_desc, params_range)

	return gcc_version, params

def parse_version(gcc_version):
	return gcc_version.split(' ')[2]

def parse_params(params_desc, params_range):
	#get ranges
	params_range = [l.strip().split() for l in params_range.split(os.linesep) if len(l)!=0 and l.find('--param')==-1]
	params = dict([
			    	(opt[0], 
					 dict(zip(opt[1::2], [int(i) for i in opt[2::2]]))
					)
					for opt in params_range])

	#ignore descriptions for now

	return params