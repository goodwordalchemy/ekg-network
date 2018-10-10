sudo docker run \
    -it \
    --runtime=nvidia \
    -p 8888:8888 \
    -p 443:443 \
    --dns 8.8.8.8 \
    -u $(id -u):$(id -g) \
    -v $(pwd):/my-devel \
    -v /mnt/disks/ptbdb/:/mnt/disks/ptbdb/ \
        tf
