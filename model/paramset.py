import subprocess
import os
from utils import (check_test_suite_finished,
                  read_params_from_file, write_params_to_file,
                  run_test_suite)


def read_options():
    '''
    '''
    gcc_version, params = get_gcc_params()

    print(f'Using gcc {gcc_version}')

    return params

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

def test_success(container_name):
    return True

def search_range(metric_name, client, N_parallel=4):
    container_list, container_name_list = [], []

    counter = 0
    current_vals = 2**(np.arange(N_parallel) + 1)

    while True:
        value = current_vals[counter]
        params = {metric_name: value}

        container_name = generate_container_name()
        container_name = f'binarysearch_{metric_name}{value}_{container_name}'

        container = run_test_suite(container_name, params, client)

        container_list.append(container)
        container_name_list.append(container_name)

        counter += 1

        if len(container_list)==N_parallel:
            check_test_suite_finished(container_list)
                
            success_list = np.array([test_success(c) for c in container_name_list])
            if not all(success_list):
                low_idx = np.where(success_list==True)[0][-1]
                low = current_vals[low_idx]
                high = current_vals[low_idx+1]

                print(low)
                print(high)
                max_val = binary_search(metric_name, metric_low, metric_high, client)

            container_list = []
            counter = 0

def binary_search(metric_name, metric_low, metric_high, client):
    low, high = metric_low, metric_high

    while low != high:
        mid_point = int((metric_low + metric_high)/2.)

        params = {metric_name: mid_point}
        container_name = generate_container_name()
        container_name = f'binarysearch_{metric_name}{mid_point}_{container_name}'

        container = run_test_suite(container_name, params, client)        
        check_test_suite_finished([container])

        if test_success(container_name):
            low = mid_point
        else:
            high = mid_point

    return low