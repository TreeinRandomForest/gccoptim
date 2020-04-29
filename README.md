# gccoptim
Using reinforcement learning to do combinatorial optimization on gcc parameters

# Infrastructure Details

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

To coordinate the two containers, we use a simple file-based locking system.

EXPLAIN LOCKING

