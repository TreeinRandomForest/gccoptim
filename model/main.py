import numpy as np
import model, config, paramset, results
import docker
import time
import shutil
import glob
import os
from utils import (read_options, check_test_suite_finished,
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
        container_name = ''.join([chr(i) for i in np.random.choice(np.concatenate([np.arange(65, 91), np.arange(97, 123)]), size=8)])

        container = run_test_suite(container_name, client)

        print(container)

        check_test_suite_finished([container])

        with open(os.path.join(storage_loc, 'logs'), 'w') as f:
            print(container.logs().decode(), file=f)

        if counter % prune_freq == 0:
            client.containers.prune()

if __name__=="__main__":
    params = read_options()

    #run(params)
