#!/bin/bash

set -u
set -e

#setup batch mode
cp /store/user-config.xml /etc/phoronix-test-suite.xml

for file in /store/PARAMS*
do
	PARAMFILE=$file

	export CFLAGS="-frecord-gcc-switches @${PARAMFILE}" #specific to redis
	echo $CFLAGS

	cd /phoronix-test-suite && ./phoronix-test-suite batch-benchmark redis #specific to redis

	#Save results
	FOLDERNAME=$(echo $PARAMFILE | awk -F/ '{print $NF}')
	HOSTNAME=$(cat /proc/sys/kernel/hostname)
	
	mkdir -p /store/$HOSTNAME/$FOLDERNAME
	cp $file /store/$HOSTNAME/$FOLDERNAME/
	mv /var/lib/phoronix-test-suite/test-results /store/$HOSTNAME/$FOLDERNAME
	mv /var/lib/phoronix-test-suite/installed-tests /store/$HOSTNAME/$FOLDERNAME
done