export IMAGE_FAMILY="tf-latest-cu92"
export INSTANCE_NAME="ekg-network"
export ZONE="us-west1-b"

gcloud compute instances create $INSTANCE_NAME \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator='type=nvidia-tesla-p100,count=1' \
  --metadata='install-nvidia-driver=True' \
  --zone=$ZONE
