import numpy as np
import model, config, paramset, results
import podman as docker
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
    '''Generic run function
    TODO: confirm this function not needed
    and logic can be now decoupled from container details
    '''
    counter = 0

    #client = docker.from_env()
    client = docker.PodmanClient(config.Podman.base_url)
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

def run_scan_one_metric(metric_name, 
                        metric_min, 
                        metric_max, 
                        metric_step_size, 
                        N_parallel=1):

    '''Run experiment for one metric in a certain range
    '''

    scanner = model.FullScan()

    clist = scanner.full_scan(metric_name,
                              metric_min,
                              metric_max,
                              metric_step_size,
                              'full_scan',
                              N_parallel=N_parallel)

    return scanner, clist

def run_full_scan(params):
  '''Run experiments over all parameters
  '''
  skipped = 0
  for p in params:
    p_min = params[p]['minimum']
    p_max = params[p]['maximum']

    if p_min==p_max:
      skipped += 1
      continue

    N_unique = (p_max-p_min+1)
    
    if N_unique > 50:
      p_step_size = int(N_unique / 50.)
    else: 
      p_step_size = 1

    print(p, p_min, p_max, p_step_size)

    scanner, clist = run_scan_one_metric(p, p_min, p_max, p_step_size, N_parallel=6)

  print(f"Skipped {skipped} params with p_max==p_min")


if __name__=="__main__":
    params = paramset.read_options()

    #run_full_scan(params) #will trigger experiments for every param for the full range
