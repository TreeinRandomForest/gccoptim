#!/bin/bash

set -u
set -e

#FLAGFILE=$1
PARAMFILE=$1

#user

#Setup for batch mode
cd /phoronix-test-suite && ./phoronix-test-suite list-all-tests
#/home/sanjay/.phoronix-test-suite/user-config.xml
cp /store/user-config.xml /etc/phoronix-test-suite.xml

#Compilation flags
export CFLAGS="-frecord-gcc-switches @${PARAMFILE}" #specific to redis
echo $CFLAGS

#Run benchmark
cd /phoronix-test-suite && ./phoronix-test-suite batch-benchmark redis #specific to redis

#Save results
HOSTNAME=$(cat /proc/sys/kernel/hostname)
cp -r /var/lib/phoronix-test-suite/test-results /store/$HOSTNAME
cp -r /var/lib/phoronix-test-suite/installed-tests /store/$HOSTNAME
rm -rf /var/lib/phoronix-test-suite/installed-tests