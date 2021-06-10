import config
import os
import podman as docker
import shutil
import numpy as np
import subprocess

def get_client():
    '''Interface to the docker API
    '''
    #client = docker.from_env()
    client = docker.PodmanClient(base_url=config.Podman.base_url)

    return client

def generate_container_name():
    '''Want containers to have unique names
    '''

    return ''.join([chr(i) for i in np.random.choice(np.concatenate([np.arange(65, 91), np.arange(97, 123)]), size=8)])

def check_test_suite_finished(container_list):
    '''Wait till containers have finished running
    '''
    [container.wait() for container in container_list]

def read_params_from_file(params_filename):
    '''Parse compiler parameter values from file into dictionary
    '''
    params = {}
    with open(params_filename, 'r') as f:
        for line in f:
            metric_name, metric_val = line.rstrip('\n').lstrip('--').split('=')
            metric_name = metric_name.split()[1]
            metric_val = int(metric_val)

            params[metric_name] = metric_val

    return params

def write_params_to_file(params, param_filename):
    '''Write compiler parameter values from dictionary to file
    '''
    with open(param_filename, 'w') as f:
        for elem in params:
            print(f'--param {elem}={params[elem]}', file=f)

def run_test_suite(container_name, params, client):
    '''Run test suite in container with params
    '''

    local_store = os.path.join(config.Storage.volume_loc, container_name)
    container_store = config.Storage.test_container_loc

    if not os.path.exists(local_store): os.makedirs(local_store)

    #param file
    param_filename = os.path.join(local_store, 'PARAMS')
    write_params_to_file(params, param_filename)

    #user config and experiment script
    shutil.copy(config.Storage.user_config, local_store)
    shutil.copy(config.Storage.test_script, local_store)

    script_name = config.Storage.test_script.split('/')[-1]

    cmd = f"podman run --name {container_name} --volume {local_store}:{container_store}:Z \
           -d {config.Containers.test_image} bash {config.Storage.test_container_loc}/{script_name}"
    print(f"Running command:\n {cmd}")
    
    container = subprocess.Popen(['podman', 
                                 'run',
                                 '--name',
                                 container_name,
                                 '--volume',
                                 f'{local_store}:{container_store}:Z',
                                 '-d',
                                 config.Containers.test_image,
                                 'bash',
                                 f'{config.Storage.test_container_loc}/{script_name}'
                                ], stdout=subprocess.PIPE) 

    return container

def write_logs(container, outfile):
    '''Write out logs before stopping container
    '''
    with open(outfile, 'w') as f:
        printf(container.logs(), file=f)