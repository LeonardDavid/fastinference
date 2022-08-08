#!/usr/bin/env python3

import copy
from datetime import datetime
from decimal import DecimalTuple
from genericpath import exists
from importlib.resources import path
import itertools
import sys
import os
import subprocess

from torch import batch_norm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier

from sklearn.tree import DecisionTreeClassifier

import fastinference.Loader

from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

import tempfile
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
import urllib.request

import operator

from fastinference.models.Ensemble import Ensemble

# not optimized for cuda/automatic
def download(url, filename, tmpdir = None):
    """Download the file under the given url and store it in the given tmpdir udner the given filename. If tmpdir is None, then `tempfile.gettmpdir()` will be used which is most likely /tmp on Linux systems.

    Args:
        url (str): The URL to the file which should be downloaded.
        filename (str): The name under which the downlaoded while should be stored.
        tmpdir (Str, optional): The directory in which the file should be stored. Defaults to None.

    Returns:
        str: Returns the full path under which the file is stored. 
    """
    if tmpdir is None:
        tmpdir = os.path.join(tempfile.gettempdir(), "data")

    os.makedirs(tmpdir, exist_ok=True)

    if not os.path.exists(os.path.join(tmpdir,filename)):
        print("{} not found. Downloading.".format(os.path.join(tmpdir,filename)))
        urllib.request.urlretrieve(url, os.path.join(tmpdir,filename))
    return os.path.join(tmpdir,filename)

# not optimized for cuda/automatic
def read_arff(path, class_name):
    """Loads the ARFF file under the given path and transforms it into a pandas dataframe. Each column which does not match class_name is copied into the pandas frame without changes. The column with the name `class_name` is renamed to `label` in the DataFrame. The behaviour of this method is undefined if the ARFF file already contains a `label` column and `class_name != 'label'`. 

    Args:
        path (str): The path to the ARFF file.
        class_name (str): The label column in the ARFF file

    Returns:
        pandas.DataFrame : A pandas dataframe containing the data from the ARFF file and an additional `label` column.
    """
    data, meta = loadarff(path)
    Xdict = {}
    for cname, ctype in zip(meta.names(), meta.types()):
        # Get the label attribute for the specific dataset:
        #   eeg: eyeDetection
        #   elec: class
        #   nomao: Class
        #   polish-bankruptcy: class
        if cname == class_name:
        #if cname in ["eyeDetection", "class",  "Class"]:
            enc = LabelEncoder()
            Xdict["label"] = enc.fit_transform(data[cname])
        else:
            Xdict[cname] = data[cname]
    return pd.DataFrame(Xdict)

# not optimized for cuda/automatic
def get_dataset(dataset, tmpdir = None, split = 0.3):
    """Returns XTrain, YTrain, XTest, YTest of the given dataset by name. If the dataset does not exist it will be automatically downloaded.

    Args:
        dataset (str): The name of the dataset to be returned (and downloaded if required.). Currently supports {magic, mnist, fashion, eeg}
        tmpdir (str, optional): The temporary folder to which the dataset is downloaded if it does not exist. If None then uses tempfile.gettempdir() to query for an appropriate temp folder. Defaults to None.
        split (float, optional): The applied train/test split. If the data-set comes with a pre-defined split (e.g. mnist) this value is ignored. Defaults to 0.3

    Raises:
        ValueError: Raises a ValueError if an unsupported dataset is passed as an argument

    Returns:
        XTrain, YTrain, XTest, YTest (2d np.array, np.array, 2d np.array, np.array): Returns the (N, d) train/test data and the (N, ) train/test labels where N is the number of data points and d is the number of features. 
    """

    if dataset == "magic":
        magic_path = download("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", "magic.csv", tmpdir)
        df = pd.read_csv(magic_path)
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1]
        Y = np.array([0 if y == 'g' else 1 for y in Y])
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=split, random_state=42)
    elif dataset == "fashion" or dataset == "mnist":
        def load_mnist(path, kind='train'):
            # Taken from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
            import os
            import gzip
            import numpy as np

            """Load MNIST data from `path`"""
            labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
            images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

            return images, labels

        if dataset == "fashion":
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "fashion")
            else:
                out_path = os.path.join(tmpdir, "data", "fashion")

            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)
        else:
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "mnist")
            else:
                out_path = os.path.join(tmpdir, "data", "mnist")

            train_path = download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)

        XTrain, YTrain = load_mnist(out_path, kind='train')
        XTest, YTest = load_mnist(out_path, kind='t10k')
    elif dataset == "eeg":
        eeg_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff", "eeg.arff", tmpdir)
        
        df = read_arff(eeg_path, "eyeDetection")
        df = pd.get_dummies(df)
        df.dropna(axis=1, inplace=True)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=split, random_state=42)
    else:
        raise ValueError("Unsupported dataset provided to get_dataset in test_utils.py: {}. Currently supported are {mnist, fashion eeg, magic}".format(dataset))
        # return None, None

    return XTrain, YTrain, XTest, YTest

# not optimized for cuda/automatic
def make_hash(o):
    """Generates a positive hash from the given object. Does also work for tuples / dicts and lists

    Args:
        o (The object to be hashed): A positive hash value
    """
    def freeze(o):
        # if isinstance(o, tuple):
        #     return frozenset( freeze(oi) for oi in o)

        if isinstance(o,dict):
            return frozenset({ k:freeze(v) for k,v in o.items()}.items())
        elif isinstance(o,(list,tuple,set)):
            return tuple([freeze(v) for v in o])
        else: 
            return hash(str(o))   
        # return o
    
    return str(hash(freeze(o)) + sys.maxsize + 1) 

# not optimized for cuda/automatic
def cfg_to_str(d):
    """A simple helper functions that formats a dictionary or lists of dictionaries into  readable string by removing large numpy arrays from them. 

    Args:
        d (dict or list of dict): The dictionary or list of dictionaries to be converted into a string

    Returns:
        str: The string
    """
    _d = copy.deepcopy(d)

    if isinstance(_d, list):
        return str([cfg_to_str(di) for di in _d])
    else:
        for k in list(_d.keys()):
            v = _d[k]
            if isinstance(v, np.ndarray) and (len(v.shape) > 2 or len(v) > 5):
                del _d[k]
                #_d[k] = "np.array"
        return str(_d)

# not optimized for cuda/automatic
def prepare_onnxmodel(onnx_path, out_path, name, benchmark_file, implementation_type, implementation_args = {}, base_optimizer = [], base_optimizer_args = [], ensemble_optimizer = [], ensemble_optimizer_args = []):
    # print("Loading testing data")
    # df = pd.read_csv(benchmark_file)
    # y_test = df["label"].to_numpy()
    # x_test = df.drop(columns=["label"]).to_numpy()
    # print("")
    # accuracy = accuracy_score(y_test, model.predict(x_test))*100.0

    fi_model = fastinference.Loader.NeuralNet(onnx_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print("Exporting {} with {} to {}".format(
        {"implementation_type":implementation_type, **implementation_args},{n:a for n, a in zip(base_optimizer, base_optimizer_args)}, name,out_path)
    )
    if len(base_optimizer) > 0 and base_optimizer[0] is not None:
        fi_model.optimize(base_optimizer, base_optimizer_args)
    fi_model.implement(out_path, "model", "cpp.{}".format(implementation_type), **implementation_args)
    
    prepare_and_compile = """
    cp ./main.cpp {outpath} && 
    cp {test_file} {outpath}/ && 
    cp ./CMakeLists.txt {outpath}
    """.replace("{outpath}", out_path).replace("{test_file}", benchmark_file)
    
    subprocess.call(prepare_and_compile, shell=True)

def run_experiment(out_path, name, feature_type, label_type, benchmark_file, batch_size, implem, nr_layers, n_repeat = 1):
    """Compiles and executes the cpp code in the given filename using the supplied benchmark file.

    Note 1: This code requires cmake for the compilation.
    Note 2: This call will likely only work on Linux / MAC as it utilizes cp to move some files around

    TODO: Make it platform independent. 

    Args:
        out_path (str): Folder in which all the cpp files are located.
        name (str): The name of the function that should be tested. In most cases this is the model name
        feature_type (str): int or double
        label_type (str): int or double
        benchmark_file (str): A *.csv file that contains the test data
        batch_size (int): size of batch
        implem (str): shorthand for implementation type (i.e. "xyz", "cpu")
        n_repeat (int, optional): The number of repetitions the experiment is repeated to get a more accurate estimation of the latency. Defaults to 5.

    Returns:
        dict: A dictionary that contains the output of the binary. It has the following fields: "accuracy", "diff accuracy", "latency [ms]", "size [Bytes]"
    """    
    
    prepare_and_compile = """
    cd {outpath} &&
    cmake . -DMODELNAME={name} -DLABEL_TYPE={label_type} -DFEATURE_TYPE={feature_type} -DBATCH_SIZE={batch_size} '-DIMPL=\"{implem}\"' '-DOUT_PATH=\"{outpath}\"' -DNR_LAYERS={nr_layers} &&
    make""".replace("{outpath}", out_path).replace("{name}", name).replace("{label_type}", label_type).replace("{feature_type}", feature_type).replace("{batch_size}", str(batch_size)).replace("{implem}",implem).replace("{nr_layers}",str(nr_layers))
    # Note: '-DIMPL="xyz"' needs to be exactly like that in order to have it as "xyz" in CMakeCache => std::string in c++ code
    # Note: same for '-DOUT_PATH="path/to/output"'
    
    print("\n")
    print("Calling {}".format(prepare_and_compile))
    subprocess.call(prepare_and_compile, shell=True)
    print("Running {} {} {}".format(os.path.join(out_path, "testCode"), benchmark_file, str(n_repeat)))
    
    output = subprocess.check_output([
        os.path.join(out_path, "testCode"),
        benchmark_file,
        str(n_repeat)
    ]).decode(sys.stdout.encoding).strip()

    print(output)
    print('\n')

    layers = []
    
    accuracy = float(output.split("\n")[13].split(" ")[1]) # first line "using CUDA profile (...)" is on position [3]
    diff = float(output.split("\n")[16].split(" ")[1])
    cpu_time = float(output.split("\n")[19].split(" ")[3])
    cpu_lat = float(output.split("\n")[19].split(" ")[7])
    gpu_time = float(output.split("\n")[20].split(" ")[3])
    gpu_lat = float(output.split("\n")[20].split(" ")[7])

    for l in range(nr_layers):
        layer_nr = int(output.split("\n")[22+l].split(" ")[1])
        cpu_time_l = float(output.split("\n")[22+l].split(" ")[3])
        cpu_ratio_l = float(output.split("\n")[22+l].split(" ")[6])
        gpu_time_l = float(output.split("\n")[22+l].split(" ")[10])
        gpu_ratio_l = float(output.split("\n")[22+l].split(" ")[13])
        total_time_l = cpu_time_l+gpu_time_l

        layer_row = []
        layer_row.append(batch_size)
        layer_row.append(layer_nr)
        layer_row.append(implem)
        layer_row.append(cpu_time_l)
        # layer_row.append(cpu_ratio_l)
        layer_row.append(gpu_time_l)
        # layer_row.append(gpu_ratio_l)
        layer_row.append(round(total_time_l,2))

        layers.append(layer_row)

    return {
        "accuracy": accuracy,
        #"diff accuracy": diff,
        "total_time [s]": cpu_time + gpu_time,
        "cpu_time [s]": cpu_time,
        "cpu_lat [ms]": cpu_lat,
        "gpu_time [s]": gpu_time, 
        "gpu_lat [ms]": gpu_lat,
        "layers": layers,
        "size [Bytes]": os.path.getsize(os.path.join(out_path, "testCode"))
    }

def prepare_fastinference(model_path, out_path, batch_size, impl_folder, implementation_type, implementation_args = {}, base_optimizer = [], base_optimizer_args = []):
    """Prepares all files for the given model and optimizations / implementations for the cpp backend.

    Note: This call will likely only work on Linux / MAC as it utilizes cp to move some files around

    TODO: Make it platform independent. 

    Args:
        model_path ([type]): The model to be generated
        out_path ([type]): The path in which all cpp files should be stored
        impl_folder (str): name of folder containing all different implementations
        implementation_type (str): The cpp implementation. 
        implementation_args (dict, optional): A dictionaries of additional parameters used during implementation. Defaults to {}.
        base_optimizer (list of string, optional): A list of optimizations that are applied before implementing the model. Defaults to [].
        base_optimizer_args (list of dict, optional): A list of parameters for each optimizer. Defaults to [].
        """    
    fi_model = fastinference.Loader.model_from_file(model_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print("Exporting {} using {}:{} with {}:{} to {}".format(
        fi_model.name, implementation_type, cfg_to_str(implementation_args), base_optimizer, cfg_to_str(base_optimizer_args), out_path
    ))

    if len(base_optimizer) > 0 and base_optimizer[0] is not None:
        fi_model.optimize(base_optimizer, base_optimizer_args)

    fi_model.implement(out_path, "model", "{}.{}".format(impl_folder, implementation_type), **implementation_args)
    #example: fastinference.implementations.neuralnet.cuda.xyz
    
    prepare_and_compile = """
    cp ./fastinference/implementations/neuralnet/cuda/automatic/main.cpp {outpath} && 
    cp ./fastinference/implementations/neuralnet/cuda/automatic/CMakeLists.txt {outpath}
    """.replace("{outpath}", out_path)
    
    print("\n")
    print("Calling {}".format(prepare_and_compile))
    subprocess.call(prepare_and_compile, shell=True)

    nr_layers = 0
    for layer in fi_model.layers:
        if layer.name != "batchnorm" and layer.name != "logsoftmax":
            # batchnorm is merged into activation
            # logsoftmax is not used (?) comparing to implement.py::to_implementation()
            # print(layer.name)
            nr_layers += 1

    return nr_layers # return number of layers to be used in main.cpp

def test_implementations(model, dataset, split, implementations, base_optimizers = [([None], [{}])], out_path = ".", model_name="Model", impl_folder = "cuda", n_repeat=1):
    print("Loading {}".format(dataset))
    #XTrain, YTrain, XTest, YTest = get_dataset(dataset,out_path,split)

    #print("Fitting model")
    #model.fit(XTrain, YTrain)

    #print("Storing model")
    #acc = accuracy_score(model.predict(XTest), YTest)*100.0
    # if isinstance(model, (DecisionTreeClassifier, RidgeClassifier, QuadraticDiscriminantAnalysis, RandomForestClassifier)):
    #     fimodel = fastinference.Loader.model_from_sklearn(model, name = model_name, accuracy = acc)
    #     path_to_model = fastinference.Loader.model_to_json(fimodel, os.path.join(out_path), file_name=model_name)
    #     print("SK ACC:", acc)
    #     print("MY ACC:", accuracy_score(fimodel.predict(XTest), YTest)*100.0)
    # else:
    #     path_to_model = model.store(out_path, acc, model_name)
        
    #print("Storing test data")
    #dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
    #path_to_testfile = os.path.join(out_path, "testing.csv")
    #dfTest.to_csv(path_to_testfile, header=True, index=False)

    path_to_testfile = os.path.join("fastinference/implementations/neuralnet/cuda/automatic/test_data/testing.csv")
    path_to_model = os.path.join("fastinference/implementations/neuralnet/cuda/automatic/test_model/cudatest.onnx")
    print(path_to_testfile)
    print(path_to_model)
    print('\n')

    performance = []
    print(implementations)
    print(base_optimizers)
    print('\n')

    now = datetime.now().strftime('%d-%m-%y_%H-%M-%S')

    # set the batch size lower and upper bound (aka the powers of 2)
    b_l = 1
    b_u = 10

    for impl, bopt in itertools.product(implementations, base_optimizers):
        for batch_size in (2**p for p in range(b_l, b_u)): # batch_size incrementing in powers of 2
            out_path_ext = os.path.join(out_path, now, model_name + "/" + impl[0] + "/" + str(batch_size))

            impl[1]['batch_size'] = batch_size

            nr_layers = prepare_fastinference(path_to_model, out_path_ext, batch_size, impl_folder, implementation_type = impl[0], implementation_args = impl[1], base_optimizer = bopt[0], base_optimizer_args = bopt[1])

            feature_type = impl[1].get("feature_type", "int")
            label_type = impl[1].get("label_type", "int")

            performance.append(
                {
                    "date_time":now,
                    "impl":impl[0],
                    "bch_sz": batch_size,
                    #"base_opt":bopt[0],
                    **run_experiment(out_path_ext, model_name, feature_type, label_type, path_to_testfile, batch_size, impl[0], nr_layers, n_repeat)
                }
            )

    layers = []
    
    for entry in range(len(performance)):
        for layer in range(len(performance[0].get("layers"))):
            layers.append(performance[entry].get("layers")[layer])

    # layers = [[4, 1, 'cpu', 0.0, 0.0, 0.0], [4, 2, 'cpu', 0.25, 0.0, 0.25], [4, 3, 'cpu', 0.34, 0.0, 0.34], [4, 4, 'cpu', 0.0, 0.0, 0.0], [4, 5, 'cpu', 0.33, 0.0, 0.33], [4, 6, 'cpu', 0.13, 0.0, 0.13], [4, 7, 'cpu', 0.0, 0.0, 0.0], [4, 8, 'cpu', 0.0, 0.0, 0.0], [4, 9, 'cpu', 0.01, 0.0, 0.01], [4, 10, 'cpu', 0.0, 0.0, 0.0], [4, 11, 'cpu', 0.0, 0.0, 0.0], [8, 1, 'cpu', 0.0, 0.0, 0.0], [8, 2, 'cpu', 0.22, 0.0, 0.22], [8, 3, 'cpu', 0.28, 0.0, 0.28], [8, 4, 'cpu', 0.0, 0.0, 0.0], [8, 5, 'cpu', 0.29, 0.0, 0.29], [8, 6, 'cpu', 0.12, 0.0, 0.12], [8, 7, 'cpu', 0.0, 0.0, 0.0], [8, 8, 'cpu', 0.0, 0.0, 0.0], [8, 9, 'cpu', 0.01, 0.0, 0.01], [8, 10, 'cpu', 0.0, 0.0, 0.0], [8, 11, 'cpu', 0.0, 0.0, 0.0], [4, 1, 'xyz', 0.0, 0.0, 0.0], [4, 2, 'xyz', 2.57, 0.85, 3.42], [4, 3, 'xyz', 0.4, 0.0, 0.4], [4, 4, 'xyz', 1.9, 0.13, 2.03], [4, 5, 'xyz', 1.91, 0.17, 2.08], [4, 6, 'xyz', 0.12, 0.0, 0.12], [4, 7, 'xyz', 1.64, 0.03, 1.67], [4, 8, 'xyz', 0.0, 0.0, 0.0], [4, 9, 'xyz', 1.85, 0.13, 1.98], [4, 10, 'xyz', 0.0, 0.0, 0.0], [4, 11, 'xyz', 1.76, 0.05, 1.81], [8, 1, 'xyz', 0.0, 0.0, 0.0], [8, 2, 'xyz', 1.57, 0.56, 2.13], [8, 3, 'xyz', 0.31, 0.0, 0.31], [8, 4, 'xyz', 0.97, 0.06, 1.03], [8, 5, 'xyz', 0.94, 0.1, 1.04], [8, 6, 'xyz', 0.12, 0.0, 0.12], [8, 7, 'xyz', 0.79, 0.06, 0.85], [8, 8, 'xyz', 0.0, 0.0, 0.0], [8, 9, 'xyz', 0.87, 0.06, 0.93], [8, 10, 'xyz', 0.0, 0.0, 0.0], [8, 11, 'xyz', 0.84, 0.03, 0.87]]
    # (no cpu) layers = [[2, 1, 'x', 0.0, 0.0, 0.0], [2, 2, 'x', 4.83, 3.6, 8.43], [2, 3, 'x', 0.33, 0.0, 0.33], [2, 4, 'x', 3.87, 0.29, 4.16], [2, 5, 'x', 4.13, 0.74, 4.87], [2, 6, 'x', 0.14, 0.0, 0.14], [2, 7, 'x', 3.51, 0.1, 3.61], [2, 8, 'x', 0.0, 0.0, 0.0], [2, 9, 'x', 4.19, 0.28, 4.47], [2, 10, 'x', 0.0, 0.0, 0.0], [2, 11, 'x', 3.95, 0.1, 4.05], [4, 1, 'x', 0.0, 0.0, 0.0], [4, 2, 'x', 2.57, 1.39, 3.96], [4, 3, 'x', 0.42, 0.0, 0.42], [4, 4, 'x', 1.87, 0.13, 2.0], [4, 5, 'x', 1.94, 0.28, 2.22], [4, 6, 'x', 0.12, 0.0, 0.12], [4, 7, 'x', 1.65, 0.03, 1.68], [4, 8, 'x', 0.0, 0.0, 0.0], [4, 9, 'x', 1.89, 0.13, 2.02], [4, 10, 'x', 0.0, 0.0, 0.0], [4, 11, 'x', 1.8, 0.04, 1.84], [8, 1, 'x', 0.0, 0.0, 0.0], [8, 2, 'x', 1.55, 0.56, 2.11], [8, 3, 'x', 0.31, 0.0, 0.31], [8, 4, 'x', 0.98, 0.06, 1.04], [8, 5, 'x', 0.94, 0.11, 1.05], [8, 6, 'x', 0.12, 0.0, 0.12], [8, 7, 'x', 0.79, 0.06, 0.85], [8, 8, 'x', 0.0, 0.0, 0.0], [8, 9, 'x', 0.87, 0.06, 0.93], [8, 10, 'x', 0.0, 0.0, 0.0], [8, 11, 'x', 0.84, 0.02, 0.86], [16, 1, 'x', 0.0, 0.0, 0.0], [16, 2, 'x', 1.32, 1.86, 3.18], [16, 3, 'x', 0.35, 0.0, 0.35], [16, 4, 'x', 0.77, 0.03, 0.8], [16, 5, 'x', 0.72, 0.08, 0.8], [16, 6, 'x', 0.13, 0.0, 0.13], [16, 7, 'x', 0.58, 0.04, 0.62], [16, 8, 'x', 0.0, 0.0, 0.0], [16, 9, 'x', 0.63, 0.04, 0.67], [16, 10, 'x', 0.0, 0.0, 0.0], [16, 11, 'x', 0.61, 0.03, 0.64], [32, 1, 'x', 0.0, 0.0, 0.0], [32, 2, 'x', 1.29, 1.85, 3.14], [32, 3, 'x', 0.36, 0.0, 0.36], [32, 4, 'x', 0.84, 0.02, 0.86], [32, 5, 'x', 0.49, 0.07, 0.56], [32, 6, 'x', 0.13, 0.0, 0.13], [32, 7, 'x', 0.36, 0.02, 0.38], [32, 8, 'x', 0.0, 0.0, 0.0], [32, 9, 'x', 0.38, 0.02, 0.4], [32, 10, 'x', 0.0, 0.0, 0.0], [32, 11, 'x', 0.34, 0.0, 0.34], [64, 1, 'x', 0.0, 0.0, 0.0], [64, 2, 'x', 0.95, 1.86, 2.81], [64, 3, 'x', 0.37, 0.0, 0.37], [64, 4, 'x', 0.55, 0.01, 0.56], [64, 5, 'x', 0.38, 0.13, 0.51], [64, 6, 'x', 0.13, 0.0, 0.13], [64, 7, 'x', 0.18, 0.01, 0.19], [64, 8, 'x', 0.0, 0.0, 0.0], [64, 9, 'x', 0.18, 0.01, 0.19], [64, 10, 'x', 0.0, 0.0, 0.0], [64, 11, 'x', 0.15, 0.0, 0.15], [128, 1, 'x', 0.0, 0.0, 0.0], [128, 2, 'x', 0.85, 1.87, 2.72], [128, 3, 'x', 0.33, 0.0, 0.33], [128, 4, 'x', 0.37, 0.01, 0.38], [128, 5, 'x', 0.25, 0.34, 0.59], [128, 6, 'x', 0.12, 0.0, 0.12], [128, 7, 'x', 0.21, 0.01, 0.22], [128, 8, 'x', 0.0, 0.0, 0.0], [128, 9, 'x', 0.11, 0.01, 0.12], [128, 10, 'x', 0.0, 0.0, 0.0], [128, 11, 'x', 0.09, 0.0, 0.09], [256, 1, 'x', 0.0, 0.0, 0.0], [256, 2, 'x', 0.78, 1.89, 2.67], [256, 3, 'x', 0.36, 0.0, 0.36], [256, 4, 'x', 0.29, 0.01, 0.3], [256, 5, 'x', 0.2, 0.39, 0.59], [256, 6, 'x', 0.13, 0.0, 0.13], [256, 7, 'x', 0.12, 0.0, 0.12], [256, 8, 'x', 0.0, 0.0, 0.0], [256, 9, 'x', 0.06, 0.0, 0.06], [256, 10, 'x', 0.0, 0.0, 0.0], [256, 11, 'x', 0.05, 0.0, 0.05], [512, 1, 'x', 0.0, 0.0, 0.0], [512, 2, 'x', 1.06, 1.88, 2.94], [512, 3, 'x', 0.34, 0.0, 0.34], [512, 4, 'x', 0.24, 0.01, 0.25], [512, 5, 'x', 0.13, 0.42, 0.55], [512, 6, 'x', 0.12, 0.0, 0.12], [512, 7, 'x', 0.08, 0.0, 0.08], [512, 8, 'x', 0.0, 0.0, 0.0], [512, 9, 'x', 0.03, 0.0, 0.03], [512, 10, 'x', 0.0, 0.0, 0.0], [512, 11, 'x', 0.02, 0.0, 0.02], [2, 1, 'y', 0.0, 0.0, 0.0], [2, 2, 'y', 4.9, 1.61, 6.51], [2, 3, 'y', 0.33, 0.0, 0.33], [2, 4, 'y', 4.08, 0.31, 4.39], [2, 5, 'y', 4.2, 0.7, 4.9], [2, 6, 'y', 0.13, 0.0, 0.13], [2, 7, 'y', 3.56, 0.11, 3.67], [2, 8, 'y', 0.0, 0.0, 0.0], [2, 9, 'y', 4.3, 0.33, 4.63], [2, 10, 'y', 0.0, 0.0, 0.0], [2, 11, 'y', 3.94, 0.13, 4.07], [4, 1, 'y', 0.0, 0.0, 0.0], [4, 2, 'y', 2.49, 0.86, 3.35], [4, 3, 'y', 0.4, 0.0, 0.4], [4, 4, 'y', 1.81, 0.14, 1.95], [4, 5, 'y', 1.87, 0.34, 2.21], [4, 6, 'y', 0.12, 0.0, 0.12], [4, 7, 'y', 1.57, 0.03, 1.6], [4, 8, 'y', 0.0, 0.0, 0.0], [4, 9, 'y', 1.82, 0.14, 1.96], [4, 10, 'y', 0.0, 0.0, 0.0], [4, 11, 'y', 1.73, 0.03, 1.76], [8, 1, 'y', 0.0, 0.0, 0.0], [8, 2, 'y', 1.56, 0.77, 2.33], [8, 3, 'y', 0.31, 0.0, 0.31], [8, 4, 'y', 0.98, 0.07, 1.05], [8, 5, 'y', 0.98, 0.25, 1.23], [8, 6, 'y', 0.12, 0.0, 0.12], [8, 7, 'y', 0.82, 0.06, 0.88], [8, 8, 'y', 0.0, 0.0, 0.0], [8, 9, 'y', 0.92, 0.07, 0.99], [8, 10, 'y', 0.0, 0.0, 0.0], [8, 11, 'y', 0.89, 0.02, 0.91], [16, 1, 'y', 0.0, 0.0, 0.0], [16, 2, 'y', 1.1, 0.99, 2.09], [16, 3, 'y', 0.36, 0.0, 0.36], [16, 4, 'y', 0.62, 0.04, 0.66], [16, 5, 'y', 0.56, 0.24, 0.8], [16, 6, 'y', 0.13, 0.0, 0.13], [16, 7, 'y', 0.45, 0.04, 0.49], [16, 8, 'y', 0.0, 0.0, 0.0], [16, 9, 'y', 0.5, 0.05, 0.55], [16, 10, 'y', 0.0, 0.0, 0.0], [16, 11, 'y', 0.47, 0.01, 0.48], [32, 1, 'y', 0.0, 0.0, 0.0], [32, 2, 'y', 1.33, 1.05, 2.38], [32, 3, 'y', 0.36, 0.0, 0.36], [32, 4, 'y', 0.81, 0.03, 0.84], [32, 5, 'y', 0.48, 0.28, 0.76], [32, 6, 'y', 0.13, 0.0, 0.13], [32, 7, 'y', 0.37, 0.02, 0.39], [32, 8, 'y', 0.0, 0.0, 0.0], [32, 9, 'y', 0.4, 0.05, 0.45], [32, 10, 'y', 0.0, 0.0, 0.0], [32, 11, 'y', 0.39, 0.03, 0.42], [64, 1, 'y', 0.0, 0.0, 0.0], [64, 2, 'y', 0.83, 0.89, 1.72], [64, 3, 'y', 0.37, 0.0, 0.37], [64, 4, 'y', 0.43, 0.02, 0.45], [64, 5, 'y', 0.33, 0.21, 0.54], [64, 6, 'y', 0.13, 0.0, 0.13], [64, 7, 'y', 0.17, 0.01, 0.18], [64, 8, 'y', 0.0, 0.0, 0.0], [64, 9, 'y', 0.17, 0.02, 0.19], [64, 10, 'y', 0.0, 0.0, 0.0], [64, 11, 'y', 0.14, 0.0, 0.14], [128, 1, 'y', 0.0, 0.0, 0.0], [128, 2, 'y', 0.85, 0.9, 1.75], [128, 3, 'y', 0.33, 0.0, 0.33], [128, 4, 'y', 0.36, 0.02, 0.38], [128, 5, 'y', 0.25, 0.23, 0.48], [128, 6, 'y', 0.12, 0.0, 0.12], [128, 7, 'y', 0.2, 0.02, 0.22], [128, 8, 'y', 0.0, 0.0, 0.0], [128, 9, 'y', 0.12, 0.02, 0.14], [128, 10, 'y', 0.0, 0.0, 0.0], [128, 11, 'y', 0.1, 0.0, 0.1], [256, 1, 'y', 0.0, 0.0, 0.0], [256, 2, 'y', 0.78, 0.9, 1.68], [256, 3, 'y', 0.34, 0.0, 0.34], [256, 4, 'y', 0.28, 0.02, 0.3], [256, 5, 'y', 0.2, 0.23, 0.43], [256, 6, 'y', 0.12, 0.0, 0.12], [256, 7, 'y', 0.12, 0.01, 0.13], [256, 8, 'y', 0.0, 0.0, 0.0], [256, 9, 'y', 0.06, 0.02, 0.08], [256, 10, 'y', 0.0, 0.0, 0.0], [256, 11, 'y', 0.05, 0.0, 0.05], [512, 1, 'y', 0.0, 0.0, 0.0], [512, 2, 'y', 1.13, 0.91, 2.04], [512, 3, 'y', 0.36, 0.0, 0.36], [512, 4, 'y', 0.25, 0.02, 0.27], [512, 5, 'y', 0.13, 0.23, 0.36], [512, 6, 'y', 0.12, 0.0, 0.12], [512, 7, 'y', 0.08, 0.01, 0.09], [512, 8, 'y', 0.0, 0.0, 0.0], [512, 9, 'y', 0.03, 0.02, 0.05], [512, 10, 'y', 0.0, 0.0, 0.0], [512, 11, 'y', 0.02, 0.0, 0.02], [2, 1, 'z', 0.0, 0.0, 0.0], [2, 2, 'z', 4.72, 1.57, 6.29], [2, 3, 'z', 0.33, 0.0, 0.33], [2, 4, 'z', 3.94, 0.4, 4.34], [2, 5, 'z', 4.05, 0.27, 4.32], [2, 6, 'z', 0.13, 0.0, 0.13], [2, 7, 'z', 3.45, 0.1, 3.55], [2, 8, 'z', 0.0, 0.0, 0.0], [2, 9, 'z', 4.16, 0.31, 4.47], [2, 10, 'z', 0.0, 0.0, 0.0], [2, 11, 'z', 3.82, 0.09, 3.91], [4, 1, 'z', 0.0, 0.0, 0.0], [4, 2, 'z', 2.54, 0.94, 3.48], [4, 3, 'z', 0.4, 0.0, 0.4], [4, 4, 'z', 1.86, 0.2, 2.06], [4, 5, 'z', 1.89, 0.17, 2.06], [4, 6, 'z', 0.12, 0.0, 0.12], [4, 7, 'z', 1.61, 0.04, 1.65], [4, 8, 'z', 0.0, 0.0, 0.0], [4, 9, 'z', 1.84, 0.15, 1.99], [4, 10, 'z', 0.0, 0.0, 0.0], [4, 11, 'z', 1.74, 0.04, 1.78], [8, 1, 'z', 0.0, 0.0, 0.0], [8, 2, 'z', 1.54, 0.7, 2.24], [8, 3, 'z', 0.31, 0.0, 0.31], [8, 4, 'z', 0.98, 0.14, 1.12], [8, 5, 'z', 0.93, 0.14, 1.07], [8, 6, 'z', 0.12, 0.0, 0.12], [8, 7, 'z', 0.79, 0.07, 0.86], [8, 8, 'z', 0.0, 0.0, 0.0], [8, 9, 'z', 0.87, 0.07, 0.94], [8, 10, 'z', 0.0, 0.0, 0.0], [8, 11, 'z', 0.83, 0.02, 0.85], [16, 1, 'z', 0.0, 0.0, 0.0], [16, 2, 'z', 1.19, 0.7, 1.89], [16, 3, 'z', 0.35, 0.0, 0.35], [16, 4, 'z', 0.65, 0.08, 0.73], [16, 5, 'z', 0.59, 0.13, 0.72], [16, 6, 'z', 0.13, 0.0, 0.13], [16, 7, 'z', 0.47, 0.04, 0.51], [16, 8, 'z', 0.0, 0.0, 0.0], [16, 9, 'z', 0.52, 0.05, 0.57], [16, 10, 'z', 0.0, 0.0, 0.0], [16, 11, 'z', 0.49, 0.01, 0.5], [32, 1, 'z', 0.0, 0.0, 0.0], [32, 2, 'z', 1.06, 0.57, 1.63], [32, 3, 'z', 0.35, 0.0, 0.35], [32, 4, 'z', 0.53, 0.05, 0.58], [32, 5, 'z', 0.31, 0.12, 0.43], [32, 6, 'z', 0.13, 0.0, 0.13], [32, 7, 'z', 0.23, 0.03, 0.26], [32, 8, 'z', 0.0, 0.0, 0.0], [32, 9, 'z', 0.24, 0.03, 0.27], [32, 10, 'z', 0.0, 0.0, 0.0], [32, 11, 'z', 0.22, 0.0, 0.22], [64, 1, 'z', 0.0, 0.0, 0.0], [64, 2, 'z', 0.93, 0.91, 1.84], [64, 3, 'z', 0.37, 0.0, 0.37], [64, 4, 'z', 0.5, 0.04, 0.54], [64, 5, 'z', 0.38, 0.1, 0.48], [64, 6, 'z', 0.13, 0.0, 0.13], [64, 7, 'z', 0.19, 0.02, 0.21], [64, 8, 'z', 0.0, 0.0, 0.0], [64, 9, 'z', 0.18, 0.02, 0.2], [64, 10, 'z', 0.0, 0.0, 0.0], [64, 11, 'z', 0.16, 0.0, 0.16], [128, 1, 'z', 0.0, 0.0, 0.0], [128, 2, 'z', 0.77, 1.97, 2.74], [128, 3, 'z', 0.33, 0.0, 0.33], [128, 4, 'z', 0.34, 0.04, 0.38], [128, 5, 'z', 0.21, 0.09, 0.3], [128, 6, 'z', 0.12, 0.0, 0.12], [128, 7, 'z', 0.16, 0.02, 0.18], [128, 8, 'z', 0.0, 0.0, 0.0], [128, 9, 'z', 0.1, 0.02, 0.12], [128, 10, 'z', 0.0, 0.0, 0.0], [128, 11, 'z', 0.08, 0.0, 0.08], [256, 1, 'z', 0.0, 0.0, 0.0], [256, 2, 'z', 0.66, 3.01, 3.67], [256, 3, 'z', 0.34, 0.0, 0.34], [256, 4, 'z', 0.27, 0.03, 0.3], [256, 5, 'z', 0.17, 0.09, 0.26], [256, 6, 'z', 0.12, 0.0, 0.12], [256, 7, 'z', 0.1, 0.02, 0.12], [256, 8, 'z', 0.0, 0.0, 0.0], [256, 9, 'z', 0.05, 0.02, 0.07], [256, 10, 'z', 0.0, 0.0, 0.0], [256, 11, 'z', 0.04, 0.0, 0.04], [512, 1, 'z', 0.0, 0.0, 0.0], [512, 2, 'z', 1.08, 3.69, 4.77], [512, 3, 'z', 0.34, 0.0, 0.34], [512, 4, 'z', 0.25, 0.04, 0.29], [512, 5, 'z', 0.13, 0.09, 0.22], [512, 6, 'z', 0.12, 0.0, 0.12], [512, 7, 'z', 0.08, 0.02, 0.1], [512, 8, 'z', 0.0, 0.0, 0.0], [512, 9, 'z', 0.04, 0.02, 0.06], [512, 10, 'z', 0.0, 0.0, 0.0], [512, 11, 'z', 0.03, 0.0, 0.03], [2, 1, 'xy', 0.0, 0.0, 0.0], [2, 2, 'xy', 4.97, 1.49, 6.46], [2, 3, 'xy', 0.33, 0.0, 0.33], [2, 4, 'xy', 4.13, 0.29, 4.42], [2, 5, 'xy', 4.33, 0.46, 4.79], [2, 6, 'xy', 0.13, 0.0, 0.13], [2, 7, 'xy', 3.68, 0.08, 3.76], [2, 8, 'xy', 0.0, 0.0, 0.0], [2, 9, 'xy', 4.36, 0.3, 4.66], [2, 10, 'xy', 0.0, 0.0, 0.0], [2, 11, 'xy', 4.07, 0.09, 4.16], [4, 1, 'xy', 0.0, 0.0, 0.0], [4, 2, 'xy', 2.74, 0.81, 3.55], [4, 3, 'xy', 0.4, 0.0, 0.4], [4, 4, 'xy', 2.05, 0.13, 2.18], [4, 5, 'xy', 2.05, 0.19, 2.24], [4, 6, 'xy', 0.12, 0.0, 0.12], [4, 7, 'xy', 1.75, 0.04, 1.79], [4, 8, 'xy', 0.0, 0.0, 0.0], [4, 9, 'xy', 1.98, 0.14, 2.12], [4, 10, 'xy', 0.0, 0.0, 0.0], [4, 11, 'xy', 1.89, 0.06, 1.95], [8, 1, 'xy', 0.0, 0.0, 0.0], [8, 2, 'xy', 1.53, 0.6, 2.13], [8, 3, 'xy', 0.31, 0.0, 0.31], [8, 4, 'xy', 0.95, 0.06, 1.01], [8, 5, 'xy', 0.92, 0.1, 1.02], [8, 6, 'xy', 0.12, 0.0, 0.12], [8, 7, 'xy', 0.79, 0.06, 0.85], [8, 8, 'xy', 0.0, 0.0, 0.0], [8, 9, 'xy', 0.87, 0.06, 0.93], [8, 10, 'xy', 0.0, 0.0, 0.0], [8, 11, 'xy', 0.84, 0.01, 0.85], [16, 1, 'xy', 0.0, 0.0, 0.0], [16, 2, 'xy', 1.22, 0.99, 2.21], [16, 3, 'xy', 0.35, 0.0, 0.35], [16, 4, 'xy', 0.69, 0.03, 0.72], [16, 5, 'xy', 0.63, 0.09, 0.72], [16, 6, 'xy', 0.13, 0.0, 0.13], [16, 7, 'xy', 0.51, 0.03, 0.54], [16, 8, 'xy', 0.0, 0.0, 0.0], [16, 9, 'xy', 0.56, 0.04, 0.6], [16, 10, 'xy', 0.0, 0.0, 0.0], [16, 11, 'xy', 0.53, 0.01, 0.54], [32, 1, 'xy', 0.0, 0.0, 0.0], [32, 2, 'xy', 1.1, 1.54, 2.64], [32, 3, 'xy', 0.35, 0.0, 0.35], [32, 4, 'xy', 0.67, 0.02, 0.69], [32, 5, 'xy', 0.36, 0.07, 0.43], [32, 6, 'xy', 0.13, 0.0, 0.13], [32, 7, 'xy', 0.27, 0.01, 0.28], [32, 8, 'xy', 0.0, 0.0, 0.0], [32, 9, 'xy', 0.27, 0.01, 0.28], [32, 10, 'xy', 0.0, 0.0, 0.0], [32, 11, 'xy', 0.25, 0.01, 0.26], [64, 1, 'xy', 0.0, 0.0, 0.0], [64, 2, 'xy', 0.99, 2.05, 3.04], [64, 3, 'xy', 0.37, 0.0, 0.37], [64, 4, 'xy', 0.55, 0.01, 0.56], [64, 5, 'xy', 0.39, 0.07, 0.46], [64, 6, 'xy', 0.12, 0.0, 0.12], [64, 7, 'xy', 0.18, 0.01, 0.19], [64, 8, 'xy', 0.0, 0.0, 0.0], [64, 9, 'xy', 0.17, 0.01, 0.18], [64, 10, 'xy', 0.0, 0.0, 0.0], [64, 11, 'xy', 0.15, 0.0, 0.15], [128, 1, 'xy', 0.0, 0.0, 0.0], [128, 2, 'xy', 0.83, 2.76, 3.59], [128, 3, 'xy', 0.33, 0.0, 0.33], [128, 4, 'xy', 0.36, 0.01, 0.37], [128, 5, 'xy', 0.24, 0.07, 0.31], [128, 6, 'xy', 0.11, 0.0, 0.11], [128, 7, 'xy', 0.19, 0.0, 0.19], [128, 8, 'xy', 0.0, 0.0, 0.0], [128, 9, 'xy', 0.1, 0.0, 0.1], [128, 10, 'xy', 0.0, 0.0, 0.0], [128, 11, 'xy', 0.08, 0.0, 0.08], [256, 1, 'xy', 0.0, 0.0, 0.0], [256, 2, 'xy', 0.77, 3.46, 4.23], [256, 3, 'xy', 0.34, 0.0, 0.34], [256, 4, 'xy', 0.3, 0.01, 0.31], [256, 5, 'xy', 0.19, 0.07, 0.26], [256, 6, 'xy', 0.12, 0.0, 0.12], [256, 7, 'xy', 0.12, 0.0, 0.12], [256, 8, 'xy', 0.0, 0.0, 0.0], [256, 9, 'xy', 0.06, 0.0, 0.06], [256, 10, 'xy', 0.0, 0.0, 0.0], [256, 11, 'xy', 0.05, 0.0, 0.05], [512, 1, 'xy', 0.0, 0.0, 0.0], [512, 2, 'xy', 1.09, 3.44, 4.53], [512, 3, 'xy', 0.34, 0.0, 0.34], [512, 4, 'xy', 0.25, 0.01, 0.26], [512, 5, 'xy', 0.13, 0.07, 0.2], [512, 6, 'xy', 0.12, 0.0, 0.12], [512, 7, 'xy', 0.08, 0.0, 0.08], [512, 8, 'xy', 0.0, 0.0, 0.0], [512, 9, 'xy', 0.04, 0.0, 0.04], [512, 10, 'xy', 0.0, 0.0, 0.0], [512, 11, 'xy', 0.02, 0.0, 0.02], [2, 1, 'xz', 0.0, 0.0, 0.0], [2, 2, 'xz', 4.91, 1.68, 6.59], [2, 3, 'xz', 0.33, 0.0, 0.33], [2, 4, 'xz', 4.04, 0.3, 4.34], [2, 5, 'xz', 4.22, 0.32, 4.54], [2, 6, 'xz', 0.13, 0.0, 0.13], [2, 7, 'xz', 3.6, 0.09, 3.69], [2, 8, 'xz', 0.0, 0.0, 0.0], [2, 9, 'xz', 4.27, 0.3, 4.57], [2, 10, 'xz', 0.0, 0.0, 0.0], [2, 11, 'xz', 4.0, 0.11, 4.11], [4, 1, 'xz', 0.0, 0.0, 0.0], [4, 2, 'xz', 2.76, 0.89, 3.65], [4, 3, 'xz', 0.4, 0.0, 0.4], [4, 4, 'xz', 2.06, 0.13, 2.19], [4, 5, 'xz', 2.08, 0.18, 2.26], [4, 6, 'xz', 0.12, 0.0, 0.12], [4, 7, 'xz', 1.77, 0.04, 1.81], [4, 8, 'xz', 0.0, 0.0, 0.0], [4, 9, 'xz', 2.02, 0.14, 2.16], [4, 10, 'xz', 0.0, 0.0, 0.0], [4, 11, 'xz', 1.93, 0.06, 1.99], [8, 1, 'xz', 0.0, 0.0, 0.0], [8, 2, 'xz', 1.6, 0.58, 2.18], [8, 3, 'xz', 0.31, 0.0, 0.31], [8, 4, 'xz', 0.99, 0.06, 1.05], [8, 5, 'xz', 0.99, 0.11, 1.1], [8, 6, 'xz', 0.12, 0.0, 0.12], [8, 7, 'xz', 0.83, 0.06, 0.89], [8, 8, 'xz', 0.0, 0.0, 0.0], [8, 9, 'xz', 0.93, 0.06, 0.99], [8, 10, 'xz', 0.0, 0.0, 0.0], [8, 11, 'xz', 0.9, 0.02, 0.92], [16, 1, 'xz', 0.0, 0.0, 0.0], [16, 2, 'xz', 1.16, 1.9, 3.06], [16, 3, 'xz', 0.35, 0.0, 0.35], [16, 4, 'xz', 0.63, 0.03, 0.66], [16, 5, 'xz', 0.56, 0.07, 0.63], [16, 6, 'xz', 0.13, 0.0, 0.13], [16, 7, 'xz', 0.46, 0.03, 0.49], [16, 8, 'xz', 0.0, 0.0, 0.0], [16, 9, 'xz', 0.51, 0.03, 0.54], [16, 10, 'xz', 0.0, 0.0, 0.0], [16, 11, 'xz', 0.49, 0.01, 0.5], [32, 1, 'xz', 0.0, 0.0, 0.0], [32, 2, 'xz', 1.18, 1.98, 3.16], [32, 3, 'xz', 0.35, 0.0, 0.35], [32, 4, 'xz', 0.7, 0.02, 0.72], [32, 5, 'xz', 0.39, 0.08, 0.47], [32, 6, 'xz', 0.13, 0.0, 0.13], [32, 7, 'xz', 0.29, 0.02, 0.31], [32, 8, 'xz', 0.0, 0.0, 0.0], [32, 9, 'xz', 0.31, 0.02, 0.33], [32, 10, 'xz', 0.0, 0.0, 0.0], [32, 11, 'xz', 0.29, 0.01, 0.3], [64, 1, 'xz', 0.0, 0.0, 0.0], [64, 2, 'xz', 0.98, 2.0, 2.98], [64, 3, 'xz', 0.37, 0.0, 0.37], [64, 4, 'xz', 0.56, 0.01, 0.57], [64, 5, 'xz', 0.4, 0.13, 0.53], [64, 6, 'xz', 0.12, 0.0, 0.12], [64, 7, 'xz', 0.2, 0.01, 0.21], [64, 8, 'xz', 0.0, 0.0, 0.0], [64, 9, 'xz', 0.19, 0.01, 0.2], [64, 10, 'xz', 0.0, 0.0, 0.0], [64, 11, 'xz', 0.16, 0.0, 0.16], [128, 1, 'xz', 0.0, 0.0, 0.0], [128, 2, 'xz', 0.8, 2.01, 2.81], [128, 3, 'xz', 0.33, 0.0, 0.33], [128, 4, 'xz', 0.35, 0.01, 0.36], [128, 5, 'xz', 0.24, 0.35, 0.59], [128, 6, 'xz', 0.12, 0.0, 0.12], [128, 7, 'xz', 0.2, 0.01, 0.21], [128, 8, 'xz', 0.0, 0.0, 0.0], [128, 9, 'xz', 0.1, 0.0, 0.1], [128, 10, 'xz', 0.0, 0.0, 0.0], [128, 11, 'xz', 0.08, 0.0, 0.08], [256, 1, 'xz', 0.0, 0.0, 0.0], [256, 2, 'xz', 0.77, 2.03, 2.8], [256, 3, 'xz', 0.34, 0.0, 0.34], [256, 4, 'xz', 0.28, 0.01, 0.29], [256, 5, 'xz', 0.19, 0.36, 0.55], [256, 6, 'xz', 0.12, 0.0, 0.12], [256, 7, 'xz', 0.13, 0.0, 0.13], [256, 8, 'xz', 0.0, 0.0, 0.0], [256, 9, 'xz', 0.06, 0.0, 0.06], [256, 10, 'xz', 0.0, 0.0, 0.0], [256, 11, 'xz', 0.04, 0.0, 0.04], [512, 1, 'xz', 0.0, 0.0, 0.0], [512, 2, 'xz', 0.93, 2.03, 2.96], [512, 3, 'xz', 0.34, 0.0, 0.34], [512, 4, 'xz', 0.24, 0.01, 0.25], [512, 5, 'xz', 0.13, 0.36, 0.49], [512, 6, 'xz', 0.12, 0.0, 0.12], [512, 7, 'xz', 0.08, 0.0, 0.08], [512, 8, 'xz', 0.0, 0.0, 0.0], [512, 9, 'xz', 0.04, 0.0, 0.04], [512, 10, 'xz', 0.0, 0.0, 0.0], [512, 11, 'xz', 0.02, 0.0, 0.02], [2, 1, 'yz', 0.0, 0.0, 0.0], [2, 2, 'yz', 4.83, 1.82, 6.65], [2, 3, 'yz', 0.32, 0.0, 0.32], [2, 4, 'yz', 3.98, 0.31, 4.29], [2, 5, 'yz', 4.15, 0.33, 4.48], [2, 6, 'yz', 0.13, 0.0, 0.13], [2, 7, 'yz', 3.53, 0.08, 3.61], [2, 8, 'yz', 0.0, 0.0, 0.0], [2, 9, 'yz', 4.18, 0.31, 4.49], [2, 10, 'yz', 0.0, 0.0, 0.0], [2, 11, 'yz', 3.92, 0.09, 4.01], [4, 1, 'yz', 0.0, 0.0, 0.0], [4, 2, 'yz', 2.65, 1.01, 3.66], [4, 3, 'yz', 0.4, 0.0, 0.4], [4, 4, 'yz', 1.97, 0.14, 2.11], [4, 5, 'yz', 2.0, 0.2, 2.2], [4, 6, 'yz', 0.12, 0.0, 0.12], [4, 7, 'yz', 1.7, 0.05, 1.75], [4, 8, 'yz', 0.0, 0.0, 0.0], [4, 9, 'yz', 1.93, 0.15, 2.08], [4, 10, 'yz', 0.0, 0.0, 0.0], [4, 11, 'yz', 1.86, 0.05, 1.91], [8, 1, 'yz', 0.0, 0.0, 0.0], [8, 2, 'yz', 1.66, 0.74, 2.4], [8, 3, 'yz', 0.31, 0.0, 0.31], [8, 4, 'yz', 1.0, 0.07, 1.07], [8, 5, 'yz', 1.03, 0.18, 1.21], [8, 6, 'yz', 0.12, 0.0, 0.12], [8, 7, 'yz', 0.84, 0.07, 0.91], [8, 8, 'yz', 0.0, 0.0, 0.0], [8, 9, 'yz', 0.98, 0.08, 1.06], [8, 10, 'yz', 0.0, 0.0, 0.0], [8, 11, 'yz', 0.95, 0.03, 0.98], [16, 1, 'yz', 0.0, 0.0, 0.0], [16, 2, 'yz', 1.18, 1.43, 2.61], [16, 3, 'yz', 0.35, 0.0, 0.35], [16, 4, 'yz', 0.63, 0.04, 0.67], [16, 5, 'yz', 0.57, 0.12, 0.69], [16, 6, 'yz', 0.13, 0.0, 0.13], [16, 7, 'yz', 0.46, 0.03, 0.49], [16, 8, 'yz', 0.0, 0.0, 0.0], [16, 9, 'yz', 0.52, 0.04, 0.56], [16, 10, 'yz', 0.0, 0.0, 0.0], [16, 11, 'yz', 0.49, 0.01, 0.5], [32, 1, 'yz', 0.0, 0.0, 0.0], [32, 2, 'yz', 1.27, 2.6, 3.87], [32, 3, 'yz', 0.36, 0.0, 0.36], [32, 4, 'yz', 0.76, 0.03, 0.79], [32, 5, 'yz', 0.41, 0.11, 0.52], [32, 6, 'yz', 0.13, 0.0, 0.13], [32, 7, 'yz', 0.33, 0.02, 0.35], [32, 8, 'yz', 0.0, 0.0, 0.0], [32, 9, 'yz', 0.32, 0.03, 0.35], [32, 10, 'yz', 0.0, 0.0, 0.0], [32, 11, 'yz', 0.28, 0.0, 0.28], [64, 1, 'yz', 0.0, 0.0, 0.0], [64, 2, 'yz', 0.99, 3.43, 4.42], [64, 3, 'yz', 0.37, 0.0, 0.37], [64, 4, 'yz', 0.57, 0.02, 0.59], [64, 5, 'yz', 0.41, 0.12, 0.53], [64, 6, 'yz', 0.13, 0.0, 0.13], [64, 7, 'yz', 0.19, 0.01, 0.2], [64, 8, 'yz', 0.0, 0.0, 0.0], [64, 9, 'yz', 0.18, 0.02, 0.2], [64, 10, 'yz', 0.0, 0.0, 0.0], [64, 11, 'yz', 0.15, 0.0, 0.15], [128, 1, 'yz', 0.0, 0.0, 0.0], [128, 2, 'yz', 0.83, 3.89, 4.72], [128, 3, 'yz', 0.34, 0.0, 0.34], [128, 4, 'yz', 0.36, 0.02, 0.38], [128, 5, 'yz', 0.24, 0.15, 0.39], [128, 6, 'yz', 0.11, 0.0, 0.11], [128, 7, 'yz', 0.19, 0.01, 0.2], [128, 8, 'yz', 0.0, 0.0, 0.0], [128, 9, 'yz', 0.11, 0.02, 0.13], [128, 10, 'yz', 0.0, 0.0, 0.0], [128, 11, 'yz', 0.09, 0.0, 0.09], [256, 1, 'yz', 0.0, 0.0, 0.0], [256, 2, 'yz', 0.76, 4.23, 4.99], [256, 3, 'yz', 0.33, 0.0, 0.33], [256, 4, 'yz', 0.28, 0.02, 0.3], [256, 5, 'yz', 0.19, 0.22, 0.41], [256, 6, 'yz', 0.12, 0.0, 0.12], [256, 7, 'yz', 0.12, 0.01, 0.13], [256, 8, 'yz', 0.0, 0.0, 0.0], [256, 9, 'yz', 0.06, 0.02, 0.08], [256, 10, 'yz', 0.0, 0.0, 0.0], [256, 11, 'yz', 0.04, 0.0, 0.04], [512, 1, 'yz', 0.0, 0.0, 0.0], [512, 2, 'yz', 0.94, 4.4, 5.34], [512, 3, 'yz', 0.34, 0.0, 0.34], [512, 4, 'yz', 0.24, 0.02, 0.26], [512, 5, 'yz', 0.12, 0.35, 0.47], [512, 6, 'yz', 0.12, 0.0, 0.12], [512, 7, 'yz', 0.07, 0.01, 0.08], [512, 8, 'yz', 0.0, 0.0, 0.0], [512, 9, 'yz', 0.03, 0.02, 0.05], [512, 10, 'yz', 0.0, 0.0, 0.0], [512, 11, 'yz', 0.02, 0.0, 0.02], [2, 1, 'xyz', 0.0, 0.0, 0.0], [2, 2, 'xyz', 5.07, 1.71, 6.78], [2, 3, 'xyz', 0.33, 0.0, 0.33], [2, 4, 'xyz', 4.17, 0.3, 4.47], [2, 5, 'xyz', 4.44, 0.33, 4.77], [2, 6, 'xyz', 0.13, 0.0, 0.13], [2, 7, 'xyz', 3.74, 0.1, 3.84], [2, 8, 'xyz', 0.0, 0.0, 0.0], [2, 9, 'xyz', 4.46, 0.3, 4.76], [2, 10, 'xyz', 0.0, 0.0, 0.0], [2, 11, 'xyz', 4.18, 0.09, 4.27], [4, 1, 'xyz', 0.0, 0.0, 0.0], [4, 2, 'xyz', 2.82, 0.99, 3.81], [4, 3, 'xyz', 0.4, 0.0, 0.4], [4, 4, 'xyz', 2.11, 0.13, 2.24], [4, 5, 'xyz', 2.15, 0.19, 2.34], [4, 6, 'xyz', 0.12, 0.0, 0.12], [4, 7, 'xyz', 1.77, 0.03, 1.8], [4, 8, 'xyz', 0.0, 0.0, 0.0], [4, 9, 'xyz', 2.07, 0.13, 2.2], [4, 10, 'xyz', 0.0, 0.0, 0.0], [4, 11, 'xyz', 1.96, 0.03, 1.99], [8, 1, 'xyz', 0.0, 0.0, 0.0], [8, 2, 'xyz', 1.69, 0.6, 2.29], [8, 3, 'xyz', 0.31, 0.0, 0.31], [8, 4, 'xyz', 1.02, 0.06, 1.08], [8, 5, 'xyz', 1.04, 0.11, 1.15], [8, 6, 'xyz', 0.12, 0.0, 0.12], [8, 7, 'xyz', 0.84, 0.07, 0.91], [8, 8, 'xyz', 0.0, 0.0, 0.0], [8, 9, 'xyz', 0.98, 0.07, 1.05], [8, 10, 'xyz', 0.0, 0.0, 0.0], [8, 11, 'xyz', 0.95, 0.03, 0.98], [16, 1, 'xyz', 0.0, 0.0, 0.0], [16, 2, 'xyz', 1.18, 1.31, 2.49], [16, 3, 'xyz', 0.35, 0.0, 0.35], [16, 4, 'xyz', 0.63, 0.03, 0.66], [16, 5, 'xyz', 0.57, 0.08, 0.65], [16, 6, 'xyz', 0.13, 0.0, 0.13], [16, 7, 'xyz', 0.46, 0.03, 0.49], [16, 8, 'xyz', 0.0, 0.0, 0.0], [16, 9, 'xyz', 0.51, 0.03, 0.54], [16, 10, 'xyz', 0.0, 0.0, 0.0], [16, 11, 'xyz', 0.49, 0.01, 0.5], [32, 1, 'xyz', 0.0, 0.0, 0.0], [32, 2, 'xyz', 1.14, 1.72, 2.86], [32, 3, 'xyz', 0.36, 0.0, 0.36], [32, 4, 'xyz', 0.68, 0.02, 0.7], [32, 5, 'xyz', 0.37, 0.07, 0.44], [32, 6, 'xyz', 0.13, 0.0, 0.13], [32, 7, 'xyz', 0.28, 0.01, 0.29], [32, 8, 'xyz', 0.0, 0.0, 0.0], [32, 9, 'xyz', 0.28, 0.01, 0.29], [32, 10, 'xyz', 0.0, 0.0, 0.0], [32, 11, 'xyz', 0.25, 0.0, 0.25], [64, 1, 'xyz', 0.0, 0.0, 0.0], [64, 2, 'xyz', 0.87, 2.1, 2.97], [64, 3, 'xyz', 0.37, 0.0, 0.37], [64, 4, 'xyz', 0.48, 0.01, 0.49], [64, 5, 'xyz', 0.35, 0.07, 0.42], [64, 6, 'xyz', 0.13, 0.0, 0.13], [64, 7, 'xyz', 0.17, 0.01, 0.18], [64, 8, 'xyz', 0.0, 0.0, 0.0], [64, 9, 'xyz', 0.17, 0.01, 0.18], [64, 10, 'xyz', 0.0, 0.0, 0.0], [64, 11, 'xyz', 0.14, 0.0, 0.14], [128, 1, 'xyz', 0.0, 0.0, 0.0], [128, 2, 'xyz', 0.82, 2.64, 3.46], [128, 3, 'xyz', 0.33, 0.0, 0.33], [128, 4, 'xyz', 0.36, 0.01, 0.37], [128, 5, 'xyz', 0.23, 0.07, 0.3], [128, 6, 'xyz', 0.12, 0.0, 0.12], [128, 7, 'xyz', 0.18, 0.0, 0.18], [128, 8, 'xyz', 0.0, 0.0, 0.0], [128, 9, 'xyz', 0.1, 0.0, 0.1], [128, 10, 'xyz', 0.0, 0.0, 0.0], [128, 11, 'xyz', 0.08, 0.0, 0.08], [256, 1, 'xyz', 0.0, 0.0, 0.0], [256, 2, 'xyz', 0.77, 3.15, 3.92], [256, 3, 'xyz', 0.34, 0.0, 0.34], [256, 4, 'xyz', 0.28, 0.01, 0.29], [256, 5, 'xyz', 0.19, 0.07, 0.26], [256, 6, 'xyz', 0.12, 0.0, 0.12], [256, 7, 'xyz', 0.11, 0.0, 0.11], [256, 8, 'xyz', 0.0, 0.0, 0.0], [256, 9, 'xyz', 0.06, 0.0, 0.06], [256, 10, 'xyz', 0.0, 0.0, 0.0], [256, 11, 'xyz', 0.04, 0.0, 0.04], [512, 1, 'xyz', 0.0, 0.0, 0.0], [512, 2, 'xyz', 1.07, 3.11, 4.18], [512, 3, 'xyz', 0.35, 0.0, 0.35], [512, 4, 'xyz', 0.25, 0.01, 0.26], [512, 5, 'xyz', 0.13, 0.07, 0.2], [512, 6, 'xyz', 0.12, 0.0, 0.12], [512, 7, 'xyz', 0.09, 0.0, 0.09], [512, 8, 'xyz', 0.0, 0.0, 0.0], [512, 9, 'xyz', 0.03, 0.0, 0.03], [512, 10, 'xyz', 0.0, 0.0, 0.0], [512, 11, 'xyz', 0.02, 0.0, 0.02]]

    out_path_ext = os.path.join(out_path, now, model_name + "/automatic/")
    if not os.path.exists(out_path_ext):
        os.makedirs(out_path_ext)

    with open(out_path_ext + "layers.txt", 'a') as f:
        f.write(str(layers))
    
    # print(layers)
    # print("\n")

    layers.sort(key = operator.itemgetter(0, 1, 2))

    # print(layers)
    # print("\n")

    # create dictionary in a way that is easy to read and to work with

    batch_layer = []

    batch_dict = {}
    batch_key = layers[0][0]

    for layer in range(len(layers)):
        # print(layers[layer])
        # if layer != len(layers)-1:
        #     if layers[layer][0] != layers[layer+1][0]:
        #         print("\n")
        #     if layers[layer][1] != layers[layer+1][1]:
        #         print("\n")

        if batch_key != layers[layer][0]:
            batch_dict[batch_key] = batch_layer
            batch_layer = []
            batch_key = layers[layer][0]
        
        batch_layer.append(layers[layer])

        if layer == len(layers)-1: # to make sure the last element is inserted
            batch_dict[batch_key] = batch_layer
            batch_layer = []        

    # print("\n")
    # print(batch_dict)
    # print("\n")

    batch_dict2 = {}

    for batch_key in batch_dict:
        # print(batch_dict[batch_key])
        
        batch_layer2 = []

        batch_dict2[batch_key] = {}
        layer_key = batch_dict[batch_key][0][1]
        
        for layer in range(len(batch_dict[batch_key])):
            # print(batch_dict[batch_key][layer])
            if layer_key != batch_dict[batch_key][layer][1]:
                batch_dict2[batch_key][layer_key] = batch_layer2
                batch_layer2 = []
                layer_key = batch_dict[batch_key][layer][1]
            
            batch_layer2.append(batch_dict[batch_key][layer])

            if layer == len(batch_dict[batch_key])-1:
                batch_dict2[batch_key][layer_key] = batch_layer2
                batch_layer2 = []

    # print("\n")
    # print(batch_dict2)
    # print("\n")

    # create optimal configuration using dictionary

    optimal_config = {}

    for batch_key in batch_dict2:
        for layer_key in batch_dict2[batch_key]:
            batch_dict2[batch_key][layer_key].sort(key = operator.itemgetter(5))
            for layer in range(len(batch_dict2[batch_key][layer_key])):
                print(batch_dict2[batch_key][layer_key][layer])
            print("\n")
        print("\n")
    print("\n")

    optimal_times = []

    for batch_key in batch_dict2:
        optimal_config[batch_key] = []
        total_cpu_time = 0
        total_gpu_time = 0
        total_time = 0
        for layer_key in batch_dict2[batch_key]:
            optimal_config[batch_key].append(batch_dict2[batch_key][layer_key][0])
            total_cpu_time += batch_dict2[batch_key][layer_key][0][3]
            total_gpu_time += batch_dict2[batch_key][layer_key][0][4]
            total_time += batch_dict2[batch_key][layer_key][0][5]
        optimal_times.append([batch_key, total_cpu_time, total_gpu_time, total_time])

    # print("optimal configurations:")
    # print(optimal_config)
    # print("\n")

    # for batch_key in optimal_config:
    #     print(optimal_config[batch_key])
    #     print("\n")

    # print("\n")

    # print("optimal times (unsorted)")
    # print(optimal_times)

    # for time in range(len(optimal_times)):
    #     print(optimal_times[time])
    # print("\n")

    optimal_times.sort(key = operator.itemgetter(3))
    
    print("optimal times for each batch size:")
    for time in range(len(optimal_times)):
        print("batch_size %d: %.2f + %.2f = %.2f" % (optimal_times[time][0], optimal_times[time][1], optimal_times[time][2], optimal_times[time][3]))
    print("\n")

    # print("optimal_time:")
    # print(optimal_times[0])
    # print("\n")

    optimal_batch_size = optimal_times[0][0]

    print("optimal batch_size: %d, cpu_time: %.2f, gpu_time: %.2f, total_time: %.2f" % (optimal_batch_size, optimal_times[0][1], optimal_times[0][2], optimal_times[0][3]))
    print("\n")

    implem = []

    print("optimal configuration for layer:")
    for layer in range(len(optimal_config[optimal_batch_size])):
        print("%d: implem: %s" % (layer+1, optimal_config[optimal_batch_size][layer][2]))
        implem.append(optimal_config[optimal_batch_size][layer][2])
    print("\n")

    out_path_ext = os.path.join(out_path, now, model_name + "/automatic/" + str(optimal_batch_size))
    impl_type = "automatic"
    impl_args = {}
    impl_args["label_type"] = "int"
    impl_args["batch_size"] = optimal_batch_size
    #impl_args["opt_impl"] = ['xyz', 'xyz', 'cpu', 'cpu', 'cpu', 'cpu', 'cpu', 'cpu', 'cpu', 'cpu', 'yz']
    impl_args["opt_impl"] = implem

    # print(impl_args["opt_impl"])
    
    print("Implementing optimal configuration:")
    print("\n")
    
    nr_layers = prepare_fastinference(path_to_model, out_path_ext, optimal_batch_size, impl_folder, implementation_type = impl_type, implementation_args = impl_args, base_optimizer = [None], base_optimizer_args = [{}])

    feature_type = "int"
    label_type = "int"

    performance.append(
        {
            "date_time":now,
            "impl":impl_type,
            "bch_sz": optimal_batch_size,
            **run_experiment(out_path_ext, model_name, feature_type, label_type, path_to_testfile, optimal_batch_size, impl_type, nr_layers, n_repeat)     
        }
    )

    return performance
