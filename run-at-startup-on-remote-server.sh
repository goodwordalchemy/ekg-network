sudo mkdir -p /mnt/disks/ptbdb
sudo mount -o discard,defaults /dev/sdb /mnt/disks/ptbdb
sudo chmod a+w /mnt/disks/ptbdb

sudo chown davidgoldberg .config

git clone https://github.com/goodwordalchemy/ekg-network.git
cp ~/.bashrc bashrc
sudo docker build -f ekg-network/dockerfiles/Nvidia.Dockerfile -t tf .
