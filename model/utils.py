import paramset
import config
import os
import docker

def get_client():
    client = docker.from_env()

    return client

def read_options():
    '''
    '''
    gcc_version, params = paramset.get_gcc_params()

    print(f'Using gcc {gcc_version}')

    return params

def check_test_suite_finished(container_list):
    [container.wait() for container in container_list]

def read_params_from_file(params_filename):
    params = {}
    with open(params_filename, 'r') as f:
        for line in f:
            metric_name, metric_val = line.rstrip('\n').lstrip('--').split('=')
            metric_name = metric_name.split()[1]
            metric_val = int(metric_val)

            params[metric_name] = metric_val

    return params

def write_params_to_file(params, param_filename):
    with open(param_filename, 'w') as f:
        for elem in params:
            print(f'--param {elem}={params[elem]}', file=f)

def run_test_suite(container_name, params, client):
    #useful locations
    local_store = os.path.join(config.Storage.volume_loc, container_name)
    container_store = config.Storage.test_container_loc

    if not os.path.exists(local_store): os.makedirs(local_store)

    #param file
    param_filename = os.path.join(local_store, 'PARAMS')
    write_params_to_file(params, param_filename)

    #user config and experiment script
    shutil.copy(config.Storage.user_config, local_store)
    shutil.copy(config.Storage.test_script, local_store)

    container = client.containers.run(config.Containers.test_image,
                                      'bash /home/user/store/run_experiments.sh',
                                      detach=True,
                                      name=container_name,
                                      volumes={local_store : {'bind': container_store, 'mode': 'rw'}}) 

    return container
