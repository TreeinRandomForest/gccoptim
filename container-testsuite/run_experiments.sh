#!/bin/bash

set -u
set -e

TEST_LOC=/home/user
STORE_LOC=/home/user

#setup batch mode
cp $STORE_LOC/store/user-config.xml /etc/phoronix-test-suite.xml

while true
do
	while [ ! -f "$STORE_LOC/store/PARAMSWRITTEN.log" ] #check if params done 
	do
		sleep 1
		echo "Waiting for parameter generation"
	done

	#???# rm /store/PARAMSWRITTEN.log
	for file in $(find $STORE_LOC/store -regextype grep -regex "$STORE_LOC/store/PARAMS[0-9]*" -exec ls {} \;) #/store/PARAMS[0-9]*
	do
		PARAMFILE=$file

		export CFLAGS="-frecord-gcc-switches @${PARAMFILE}" #specific to redis
		echo $PARAMFILE
		echo $CFLAGS

		cd $TEST_LOC/phoronix-test-suite && ./phoronix-test-suite batch-benchmark redis #specific to redis

		#Save results
		FOLDERNAME=$(echo $PARAMFILE | awk -F/ '{print $NF}')
		HOSTNAME=$(cat /proc/sys/kernel/hostname)
		
		mkdir -p $STORE_LOC/store/$HOSTNAME/$FOLDERNAME #output folder
		mv $file $STORE_LOC/store/$HOSTNAME/$FOLDERNAME/ #param file
		mv $TEST_LOC/.phoronix-test-suite/test-results $STORE_LOC/store/$HOSTNAME/$FOLDERNAME #test results
		mv $TEST_LOC/.phoronix-test-suite/installed-tests $STORE_LOC/store/$HOSTNAME/$FOLDERNAME #test binaries
	done

	rm $STORE_LOC/store/PARAMSWRITTEN.log
	touch $STORE_LOC/store/TESTDONE.log
done
