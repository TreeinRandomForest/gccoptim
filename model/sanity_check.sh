set -u
set -e

LOC=$1

HOME=$(pwd)

for dir in $LOC/*
do
	cd $dir
	
	#test dependent
	

	cd $HOME
done