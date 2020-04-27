#!/bin/bash

set -u
set -e

#FLAGFILE=$1
PARAMFILE=$1

#Setup for batch mode
cd /phoronix-test-suite && ./phoronix-test-suite list-all-tests
cp /store/user-config.xml /var/lib/phoronix-test-suite


#Compilation flags
export CFLAGS="-frecord-gcc-switches @${PARAMFILE}" #specific to redis

#Run benchmark
cd /phoronix-test-suite && ./phoronix-test-suite batch-benchmark redis #specific to redis

#Save results
HOSTNAME=$(cat /proc/sys/kernel/hostname)
cp -r /var/lib/phoronix-test-suite/test-results /store/$HOSTNAME
cp -r /var/lib/phoronix-test-suite/installed-tests /store/$HOSTNAME
