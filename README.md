# gccoptim
Using reinforcement learning to do combinatorial optimization on gcc parameters

## Infrastructure Details

The simplest policy gradient implementation consists of multiple loops for the following steps:

* A simulated/real agent rolling out (state, action, reward, next state) tuples using a policy to collect trajectories.

* Update step on the policy model.

In our case, we want to use reinforcement learning to learn the update step for the parameter set for gcc. In addition, we want to use techniques like simulated annealing as a baseline.

There are two container images that can be built using the corresponding Dockerfiles:

* container-optimizer: image based on Fedora implementing the combinatorial optimization algorithm or reinforcement learning model.

* container-testsuite: image based on Fedora that includes the phoronix test suite (https://www.phoronix-test-suite.com/) that is used for measuring performance of the compiled binaries.

In the simplest setup, we repeat two steps indefinitely:

* container-optimizer: the model uses the current (and possibly) past configurations and performance measurements to suggest the next gcc parameter configuration(s) to try.

* container-testsuite: the parameter configuration(s) are used to compile the tests with the appropriate gcc parameters and run the benchmark.

### Parallelization:

* This branch (docker-sdk) alternates between updating the model and running tests with potentially k multiple parameter settings. All the k experiments need to finish before an update to the model.

### Future infrastructure extensions:

* Asynchronous updates from experiments.

* Use Beaker

## Notes:

__init__.py: regular imports

main.py:
    run_scan_one_metric -> loop over all unique values and run test-suite
    run_full_scan -> calls run_scan_one_metric for all metrics

config.py:
    #TODO: relative paths instead of absolute paths
    Global variables stored here
    Containers.test_image -> gcc_testsuite
    Storage.volume_loc -> store/dynamic
    Storage.test_container_loc -> /home/user/store
    Storage.test_script -> container-testsuite/run_experiments.sh
    Storage.user_config -> store/user-config.xml

paramset.py
    read_options -> gcc --version output
    get_gcc_params -> gcc --help=params and gcc --help=params -Q
    parse_version -> just cleaning version number
    parse_params -> return dictionary of params, descriptions and ranges
    test_success -> returns True for now
    search_range and binary_search -> don't need?

results.py:
    Should this be in utils?

    read_results -> read XML file (used in model.py)
    parse_xml -> parse XML file

utils.py:
    get_client: get docker client
    generate_container_name
    check_test_suite_finished
    read_params_from_file
    write_params_to_file
    run_test_suite
    write_logs

visualization.py:
    ignore for now

### 