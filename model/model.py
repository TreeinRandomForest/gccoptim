import glob
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
import config
import results
from utils import (check_test_suite_finished, generate_container_name,
                  read_params_from_file, write_params_to_file,
                  run_test_suite)
plt.ion()

class FullScan:
    def full_scan(
        self,
        metric_name, 
        metric_min, 
        metric_max, 
        metric_step_size, 
        container_tag, 
        N_parallel=1
        ):
        
        assert(N_parallel >= 0)
        
        container_list = []
        for m in np.arange(metric_min, metric_max+1, metric_step_size):
            params = {metric_name: m}

            container_name = generate_container_name()
            container_name = f'{container_tag}_{metric_name}_{m}_{container_name}'

            print(f'Container Name: {container_name}')
            print(f'Params = {params}\n')

            container = run_test_suite(container_name, params)

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
            if len(res_loc) == 0:
                print(f'Found No Results: {d}')
                continue

            res_loc = res_loc[0]

            param_file = f'{d}/PARAMS'
            
            try:
                current_params = read_params_from_file(param_file)
                current_res = results.read_results(res_loc)
            except:
                print(d)

            r.append((current_params, current_res))

        return r

    def convert_to_df(self, r):
        '''Only works for one metric being tuned
        Otherwise, should store param dict in param_name and write a function to flatten
        '''

        df = {'param_name': [],
              'param_val': [],
              'test_name': [],
              'test_version': [],
              'test_args': [],
              'test_desc': [],
              'metric_units': [],
              'metric_mean': [],
              'metric_data': [],
              }

        for entry in r:
            param, test_list = entry

            for test in test_list:
                df['param_name'].append(list(param.keys())[0])
                df['param_val'].append(list(param.values())[0])

                df['test_name'].append(test['Title'])
                df['test_version'].append(test['AppVersion'])
                df['test_args'].append(test['Arguments'])
                df['test_desc'].append(test['Description'])

                df['metric_units'].append(test['Scale'])
                df['metric_mean'].append(test['Data']['value'])
                df['metric_data'].append(test['Data']['raw_values'])

        return pd.DataFrame(df)

    def plot_stats(self, df, save_loc, params=None):
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        #max_over_min
        #default param, default val
        #error bars

        max_over_min = []
        for param_name in df['param_name'].unique(): #parameters scanned
            for test_name in df['test_name'].unique(): #Redis
                for test_desc in df['test_desc'].unique(): #SADD, LPOP etc.

                    d = df[(df['param_name']==param_name) & (df['test_name']==test_name) & (df['test_desc']==test_desc)].copy()

                    d.sort_values('param_val', ascending=True, inplace=True)

                    #default vals
                    if params is not None:
                        default_param_val = params[param_name]['default'] #actual default

                        #find closest match
                        row = d.loc[(d['param_val'] - default_param_val).abs().sort_values().index[0]]
                        default_param_val_closest = row['param_val']

                        default_metric_mean_closest = row['metric_mean']
                        max_over_default_val = d['metric_mean'].max() / default_metric_mean_closest

                    #metrics
                    max_over_min_val = d['metric_mean'].max()/d['metric_mean'].min()
                    if params is not None:
                        max_over_min_str = f"{param_name},{test_name},{test_desc},{max_over_min_val},{max_over_default_val}"
                    else:
                        max_over_min_str = f"{param_name},{test_name},{test_desc},{max_over_min_val}"
                    max_over_min.append(max_over_min_str)

                    #plotting
                    units = d['metric_units'].unique()
                    if len(units) > 1:
                        raise ValueError("Found multiple metrics")
                    units = units[0]

                    plt.clf()
                    
                    #plt.plot(d['param_val'], d['metric_mean'], 'p-')
                    e = d[d['metric_data'].apply(lambda x: x is None)]

                    d_min = d['metric_data'].apply(np.min)
                    d_max = d['metric_data'].apply(np.max)
                    err = np.array(pd.concat([d_min, d_max], axis=1).T)                    
                    plt.errorbar(d['param_val'], d['metric_mean'], yerr=yerr, fmt='p-')

                    plt.xlabel(param_name)
                    plt.ylabel(units)
                    if params is not None:
                        plt.plot([default_param_val_closest], [default_metric_mean_closest], 'p', c='orange')
                        plt.title(f'{test_name} : {test_desc} : max/min = {max_over_min_val:.3f}\n default={default_param_val} closest={default_param_val_closest} metric={default_metric_mean_closest} max/default={max_over_default_val:.3f}')
                    else:
                        plt.title(f'{test_name} : {test_desc} : max/min = {max_over_min_val:.3f}')

                    plt.savefig(f'{save_loc}/{param_name}_{test_name}_{test_desc.replace(": ", "_")}.png')


        with open(f'{save_loc}/max_over_min', 'w') as f:
            for m in max_over_min:
                print(m, file=f)

class Annealing:
    def __init__(self, metric):
        self.metric = metric

    def estimate_init_temperature(self, N_tries):
        for _ in range(N_tries):
            params = generate_init_params()
            container_name = generate_container_name()
            container = run_test_suite(container_name, params_current)
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
        container = run_test_suite(container_name, params_current)
        check_test_suite_finished([container])
        res_current = self.read_result(container_name)

        for i in range(N_iter):
            if i%T_decay_iter==0:
                T *= T_decay

            params_candidate = get_neighbor(params_current)
            container_name = generate_container_name()
            container = run_test_suite(container_name, params_candidate)
            check_test_suite_finished([container])
            res_candidate = self.read_result(container_name)

            prob_threshold = np.exp(-(res_current - res_candidate)/T)

            if np.random.random() < prob_threshold:
                params_current = params_candidate

            if counter % prune_freq == 0:
                client.containers.prune()

class BayesOpt:
    pass

class PolicyGradient:
    pass