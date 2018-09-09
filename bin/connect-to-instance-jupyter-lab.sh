export INSTANCE_NAME="ekg-network"
gcloud compute ssh $INSTANCE_NAME -- -L 8080:localhost:8080
