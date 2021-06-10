import numpy as np
import model, config, paramset, results
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
