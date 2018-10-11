export INSTANCE_NAME="ekg-network"
export RUN_AT_STARTUP_SCRIPT="run-at-startup-on-remote-server.sh"

gcloud compute \
    instances create $INSTANCE_NAME \
    --machine-type "custom-10-61440" \
    --maintenance-policy "TERMINATE" \
    --restart-on-failure \
    --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
    --accelerator "type=nvidia-tesla-p100,count=1" \
    --min-cpu-platform "Intel Broadwell" \
    --image "nvidia-gpu-cloud-image-20180816" \
    --image-project "nvidia-ngc-public" \
    --boot-disk-size "200" \
    --boot-disk-type "pd-standard" 

gcloud compute scp ./$RUN_AT_STARTUP_SCRIPT ekg-network:~/$RUN_AT_STARTUP_SCRIPT
gcloud compute instances attach-disk $INSTANCE_NAME --disk ptbdb-data
