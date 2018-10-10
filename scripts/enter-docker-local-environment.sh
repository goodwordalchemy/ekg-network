docker run \
    -it \
    -p 8888:8888 \
    -p 443:443 \
    -u $(id -u):$(id -g) \
    -v $(pwd):/my-devel \
    -v $(pwd)/data/untracked:/mnt/disks/ptbdb/ \
    local-testing
