export MINICONDA_PATH_COMMAND="export PATH=\"/home/davidgoldberg/miniconda3/bin:\$PATH\""

echo "Installing miniconda3"
wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b
echo $MINICONDA_PATH_COMMAND >> ~/.bashrc
export PATH="/home/davidgoldberg/miniconda3/bin:$PATH"

echo "Installing python-based data science tools"
yes | conda create -n tensorflow pip python=3.5

source activate tensorflow

pip install tensorflow keras ipython notebook matplotlib google-cloud-storage

echo "Installing ekg-network package."
sudo apt-get install git
git clone https://github.com/goodwordalchemy/ekg-network.git

cd ekg-network

echo "Preparing EKG data."
sudo mkdir -p /mnt/disks/ptbdb
sudo mount -o discard,defaults /dev/sdb /mnt/disks/ptbdb
sudo chmod a+w /mnt/disks/ptbdb
