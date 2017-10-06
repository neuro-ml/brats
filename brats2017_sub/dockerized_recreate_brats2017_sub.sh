#/bin/bash
BRATS_PATH=$1

docker build -t recreate_brats2017_sub . 
nvidia-docker run --rm \
	--volume $BRATS_PATH:/data/ \
	--volume $PWD:/main/ \
	--workdir /main/ \
	--user $UID \
	recreate_brats2017_sub \
	/bin/bash recreate_brats2017_sub.sh
