I am working on training a neural network to automatically detect myocardial infarctions from EKG data.

This is a framework for testing out different neural network models and searching for optimal hyper parameters.

It contains tools for starting up a remote server on a google cloud compute instance and configuring it it so that it is ready to run model selection experiments.
* `./bin/configure-remote-server.sh` sets up a google cloud compute instance and transfers some keyfiles to it.
* `./run-at-startup-on-remote-server.sh` is to be run once the gcp instance is set up.  It downloads this same git repo, installs python, and installs tensorflow and keras, two of the libraries that this project is based on.

Prospective neural network architectures are stored in the `models` directory.  Check out `models/simple_lstm.py` for an example.

Hyperparameter search scripts are stored in  the `select_model/search_methods/` directory.

Once everything is configured, the way to use this tool is `python -m select_model`.
