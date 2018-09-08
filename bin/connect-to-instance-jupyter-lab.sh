export INSTANCE_NAME="ekg-network"
gcloud compute ssh $INSTANCE_NAME -- -L 8888:localhost:8888
