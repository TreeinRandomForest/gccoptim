import numpy as np
import model, config, paramset, results
import docker
import time
import shutil
import glob
import os
import numpy as np
from utils import (check_test_suite_finished,
                  read_params_from_file, write_params_to_file,
                  run_test_suite)

def stopping_criterion(counter):
    if counter==3:
        return False

    return True

def run(params):
    counter = 0

    client = docker.from_env()
    prune_freq = 100

    while stopping_criterion(counter):
        counter += 1

        #generate params
        params_to_try = generate_param_set(params, counter)

        #create dir for shared volume
        container_name = generate_container_name()

        container = run_test_suite(container_name, client)

        print(container)

        check_test_suite_finished([container])

        with open(os.path.join(storage_loc, 'logs'), 'w') as f:
            print(container.logs().decode(), file=f)

        if counter % prune_freq == 0:
            client.containers.prune()

def run_scan(params, 
             metric_name, 
             metric_min, 
             metric_max, 
             metric_step_size, 
             N_parallel=1):

    scanner = model.FullScan()

    clist = scanner.full_scan(metric_name,
                              metric_min,
                              metric_max,
                              metric_step_size,
                              'full_scan',
                              N_parallel=N_parallel)

    return scanner, clist

if __name__=="__main__":
    params = paramset.read_options()

    #run(params)
