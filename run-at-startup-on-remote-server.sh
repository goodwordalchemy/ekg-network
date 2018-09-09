export MINICONDA_PATH_COMMAND="export PATH=\"/home/davidgoldberg/miniconda3/bin:\$PATH\""
echo "Waiting for backgroud startup processes to stop."
wait 

echo "Installing miniconda3"
# wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b
echo $MINICONDA_PATH_COMMAND >> ~/.bashrc
export PATH="/home/davidgoldberg/miniconda3/bin:$PATH"


echo "Installing python-based data science tools"
conda create -n tensorflow pip python=3.5
wait

source activate tensorflow

pip install tensorflow keras ipython notebook matplotlib

echo "Installing ekg-network package."
sudo apt-get install git
git clone https://github.com/goodwordalchemy/ekg-network.git

cd ekg-network

echo "Preparing EKG data."

# mkdir -p data/cached_records

# gsutil -m rsync -r -d gs://ekg-network/ptdb/ data/cached_records

echo "Starting ipython notebook"
jupyter notebook
