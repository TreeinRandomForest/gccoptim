while [ ! -f "./PARAMSWRITTEN.log" ] #check if params done 
do
	sleep 1
	echo "Waiting for parameters"
done
