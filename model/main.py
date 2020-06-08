import numpy as np
import model, config, paramset
import docker
import time
import shutil
import os


def read_options():
    '''
    '''
    gcc_version, params = paramset.get_gcc_params()

    print(f'Using gcc {gcc_version}')

    return params

def check_test_suite_finished(container):
    container.wait()

def generate_param_set(params, counter):
    '''This is where the model goes
    '''
    
    #return [{'asan-globals': 0},
    #       {'early-inlining-insns': 0}]

    keys = list(params.keys())
    #key = key[10]
    key = 'unroll-jam-min-percent'

    minimum, maximum = params[key]['minimum'], params[key]['maximum']

    if minimum==maximum:
        raise ValueError(f"Minimum = Maximum for key = {key}")

    #return_val = [{key: value} for value in range(minimum, maximum, 1)]
    return_val = {key: 10}

    return return_val

def write_params_to_file(params, param_filename):
    with open(param_filename, 'w') as f:
        for elem in params:
            print(f'--param {elem}={params[elem]}', file=f)

def stopping_criterion(counter):
    if counter==3:
        return False

    return True

def run_test_suite(container_name, client):
    local_store = os.path.join(config.Storage.volume_loc, container_name)
    container_store = config.Storage.test_container_loc

    container = client.containers.run(config.Containers.test_image,
                                      'bash /home/user/store/run_experiments.sh',
                                      #'./run_experiments.sh',
                                      detach=True,
                                      name=container_name,
                                      volumes={local_store : {'bind': container_store, 'mode': 'rw'}}) 

    return container

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
        storage_loc = os.path.join(config.Storage.volume_loc, container_name)
        os.makedirs(storage_loc)

        param_filename = os.path.join(storage_loc, 'PARAMS')
        write_params_to_file(params_to_try, param_filename)

        #copy script to shared volume
        shutil.copy(config.Storage.user_config, storage_loc)
        shutil.copy(config.Storage.test_script, storage_loc)
        container = run_test_suite(container_name, client)
        print(container)

        check_test_suite_finished(container)

        with open(os.path.join(storage_loc, 'logs'), 'w') as f:
            print(container.logs().decode(), file=f)

        if counter % prune_freq == 0:
            client.containers.prune()

if __name__=="__main__":
    params = read_options()

    #run(params)
