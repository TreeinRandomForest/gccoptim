import glob
import numpy as np
import config
import results
from utils import (check_test_suite_finished, generate_container_name,
                  read_params_from_file, write_params_to_file,
                  run_test_suite, get_client)

class FullScan:
    def __init__(self):
        self.client = get_client()

    def full_scan(
        self,
        metric_name, 
        metric_min, 
        metric_max, 
        metric_step_size, 
        container_tag, 
        N_parallel=3
        ):
        
        container_list = []
        for m in np.arange(metric_min, metric_max+1, metric_step_size):
            params = {metric_name: m}

            container_name = generate_container_name()
            container_name = f'{container_tag}_{metric_name}_{m}_{container_name}'

            print(f'Container Name: {container_name}')
            print(f'Params = {params}\n')

            container = run_test_suite(container_name, params, self.client)

            container_list.append(container)

            if N_parallel > 0 and len(container_list) == N_parallel:
                check_test_suite_finished(container_list)
                container_list = []

        check_test_suite_finished(container_list)

        return container_list

    def read_full_scan_results(self, container_tag):
        r = []
        for d in glob.glob(f'{config.Storage.volume_loc}/{container_tag}*'):
            res_loc = glob.glob(f'{d}/test-results/*')
            if len(res_loc) > 1:
                raise ValueError(f"Found Multiple Results: {res_loc}")
            res_loc = res_loc[0]

            param_file = f'{d}/PARAMS'
            
            current_params = read_params_from_file(param_file)
            current_res = results.read_results(res_loc)

            r.append((current_params, current_res))

        return r

class Annealing:
    def __init__(self, client, metric):
        self.client = client
        self.metric = metric

    def estimate_init_temperature(self, N_tries):
        for _ in range(N_tries):
            params = generate_init_params()
            container_name = generate_container_name()
            container = run_test_suite(container_name, params_current, client)
            check_test_suite_finished([container])
            res_current = self.read_result(container_name)

    def generate_init_params(self):
        pass

    def get_neighbor(self, params):
        pass

    def read_result(container_name):
        d = glob.glob(f'{config.Storage.volume_loc}/{container_name}/test-results/*')
        if len(d) > 1:
            raise ValueError("Found multiple folders with container name")
        d = d[0]

        current_res = results.read_results(d)

        return current_res

    def annealing(N_iter, 
                  T_decay=0.99, 
                  T_decay_iter=100,
                  prune_freq=100):
        T = estimate_init_temperature()

        params_current = generate_init_params()
        container_name = generate_container_name()
        container = run_test_suite(container_name, params_current, client)
        check_test_suite_finished([container])
        res_current = self.read_result(container_name)

        for i in range(N_iter):
            if i%T_decay_iter==0:
                T *= T_decay

            params_candidate = get_neighbor(params_current)
            container_name = generate_container_name()
            container = run_test_suite(container_name, params_candidate, client)
            check_test_suite_finished([container])
            res_candidate = self.read_result(container_name)

            prob_threshold = np.exp(-(res_current - res_candidate)/T)

            if np.random.random() < prob_threshold:
                params_current = params_candidate

            if counter % prune_freq == 0:
                client.containers.prune()
