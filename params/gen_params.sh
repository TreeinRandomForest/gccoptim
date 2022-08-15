#!/bin/bash

set -ue

v=`gcc --version | head -1 | awk '{print $3}'`
echo $v

gcc -v --help=params > params"$v".opt
