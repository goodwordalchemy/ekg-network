export INSTANCE_NAME="ekg-network"
export MACHINE_TYPE="n1-standard-8"

export ENABLE_CREATING_REMOTE_SERVER=true
export GPU_STARTUP_WAIT_TIME=30
export RUN_AT_STARTUP_SCRIPT="run-at-startup-on-remote-server.sh"

set -e 

if $ENABLE_CREATING_REMOTE_SERVER; then
	echo "Starting Google Cloud Compute instance that has a GPU attached..."
	gcloud compute instances create $INSTANCE_NAME \
		--machine-type $MACHINE_TYPE \
		--accelerator type=nvidia-tesla-k80,count=1 \
		--image-family ubuntu-1604-lts --image-project ubuntu-os-cloud \
		--maintenance-policy TERMINATE --restart-on-failure \
		--metadata startup-script='#!/bin/bash
		echo "Checking for CUDA and installing."
		# Check for CUDA and try to install.
		if ! dpkg-query -W cuda-9-0; then
		  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
		  dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
		  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
		  apt-get update
		  apt-get install cuda-9-0 -y
		fi'

	echo "Waiting $GPU_STARTUP_WAIT_TIME seconds for GPU to attach..."
	sleep $GPU_STARTUP_WAIT_TIME
fi

echo "Running startup script on remote server"
gcloud compute scp ./$RUN_AT_STARTUP_SCRIPT ekg-network:~/$RUN_AT_STARTUP_SCRIPT
echo "Finished copying startup script to remote server"

gcloud compute ssh $INSTANCE_NAME --command "source $RUN_AT_STARTUP_SCRIPT" -- -L 8888:localhost:8888
