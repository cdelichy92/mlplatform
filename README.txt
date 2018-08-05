ML Take-home

This package assumes Anaconda for python 3.6 is installed and the bash shell is
installed

To run:

# create the conda env from the requirements file
# (you can replace tensorflow by tensorflow-gpu in the requirements.yaml file
# if a gpu is available)
`$ conda create env -f requirements.yaml`

# activate the newly created conda env
`$ source activate mlplatform`

# to download the GloVe word vectors and a dataset:
`$ bin/download_prepare.sh`

# install the package
`$ pip install -e .`

# run the app
`$ run_app`

# or in the package folder:
`$ python app.py`

If there are problems with the dependencies I also exported my full conda env
to mlplatform.yaml

The app uses Flask built-in dev server, you can connect to it to: http://localhost:5000/
There are 2 URLS: - http://localhost:5000/train for the training phase
                  - http://localhost:5000/predict to compute predictions from
                  user text input (the first time the predict button is clicked on, it
                  takes some time for the service to load the graph etc...,
                  subsequent clicks are faster)
