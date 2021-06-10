#!/bin/bash

set -u
set -e

TEST_LOC=/home/user #phoronix-test-suite
STORE_LOC=/home/user/store #store (shared volume) specific to this container
RESULT_LOC=/var/lib

cp $STORE_LOC/user-config.xml /etc/phoronix-test-suite.xml
$TEST_LOC/phoronix-test-suite/phoronix-test-suite enterprise-setup

PARAMFILE=$STORE_LOC/PARAMS
export CFLAGS="-frecord-gcc-switches -O3 @${PARAMFILE}" #specific to redis
echo $PARAMFILE
echo $CFLAGS

#look at bash scripts for test to ensure gcc is being used
cd $TEST_LOC/phoronix-test-suite && ./phoronix-test-suite batch-benchmark redis >& /home/user/store/main_log #specific to redis

mv $RESULT_LOC/phoronix-test-suite/test-results $STORE_LOC #test results
mv $RESULT_LOC/phoronix-test-suite/installed-tests $STORE_LOC #test binaries
