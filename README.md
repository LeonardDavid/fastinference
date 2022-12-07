<h1> HEP-BNN </h1>

HEP-BNN is a framework for finding Low-Latency execution configurations of BNNs on Heterogeneous multiprocessor platforms.

# How to install

First set up a conda environment with all the necessary dependencies:

    git clone https://github.com/LeonardDavid/hep-bnn.git
    cd fastinference
    conda env create -f environment.yml
    conda activate fi

Please note that this environment also contains some larger packages such as PyTorch so the installation may take some time. 

Execute the following command before first time use, <b> and after any changes to any of the Jinja2 files, or any ```implement.py``` files </b>:

    python setup.py install

# How to use the framework

## Include the BNN model and data

Add your BNN model (in ```ONNX``` format) under 

    fastinference/implementations/neuralnet/cuda/automatic/model/
    
and the testing data (in ```csv``` format) under

    fastinference/implementations/neuralnet/cuda/automatic/data/
    
Also see the example files for the Fashion-MNIST and CIFAR10 models, used during development.

## Run the framework

Call the following command:

    fastinference/implementations/neuralnet/cuda/automatic/test_cuda.py --outpath tmp/fastinference/cuda_auto --dataset cifar
    
   
# Acknowledgements

Special thanks goes to [Sebastian Buschjäger](sebastian.buschjaeger@tu-dortmund.de/) for providing the original fastinference framework as a baseline.
