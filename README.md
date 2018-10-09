# EKG Neural Network

I am working on training a neural network to automatically detect myocardial infarctions from EKG data.  This is a framework for testing out different neural network models and searching for optimal hyper parameters.  Note that when I first named this project, I was under the impression that ECGs were stilled called EKGs.

## Data
I have made the data that I am using publically available in the google storage object: `gs://ekg-network/truncated-samples/`.  To create this dataset, I downloaded all of the ekgs from the PTB Diagnostic ECG Database.  Then I cut each sample into 10-second segments."  

If you are going to work on this project, you will need to download this data.  There are many ways to do this, but the way I recommend is using `destination>`

## Configuring GPU environment on Google Cloud Instance
This is highly recommended, because there is too much data in this dataset to train a neural network of considerable size on a laptop.  The GPU gives me ~20x speedup.

1. Clone this repository: `https://github.com/goodwordalchemy/ekg-network`.
2. IF THIS IS THE FIRST TIME YOU ARE DOING THIS: you need to create a persistent disk containing ecg data.  In your terminal, run `gcloud compute disks create ptbdb-data --size 10 --type pd-ssd`.
3. Create a Google Cloud Compute instance with a GPU and most of the tools you need to use it with keras and tensorflow.  In your terminal, run `./scripts/configure-remote-server.sh`.  You will get an error about the ptbdb disk not existing, but your server will have started.  
4. ssh into the GCP instance: `./scripts/connect-to-instance.sh`.
5. IF THIS IS THE FIRST TIME YOU ARE DOING THIS: you need to load your persistent disk with ptbdb data. Run the following:
```bash
$ mkdir -p /mnt/disks/ptbdb/data/
$ mkdir -p /mnt/disks/ptbdb/results/
$ gsutil -m rsync -r -d gs://ekg-network/truncated-samples /mnt/disks/ptbdb/data
```
7. Run `./run-at-startup-on-remote-server.sh`.
8. To enter the Docker environment where you should be all set with the tools you need, run `source enter-docker-tf-gpu-environment.sh`
9. In the docker environment, run `cd my-devel/ekg-network`
10. To test that everything is all set, run `python3 -m select_model --config config_files/inception_grid_config.yaml`

## Quick Tour Of The Tool

Prospective neural network architectures are stored in the `models` directory.  Check out `models/simple_lstm.py` for an example.

Hyperparameter search scripts are stored in  the `select_model/search_methods/` directory.

There are some example configuration files in the `config_files` directory.

Once everything is configured, the way to use this tool is `python -m select_model --config <config-filename>`.
