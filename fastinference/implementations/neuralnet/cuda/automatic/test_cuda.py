#!/usr/bin/env python3

import itertools
import os
import pandas as pd
import argparse

from torch import optim
from test_utils import get_dataset, prepare_fastinference, run_experiment, make_hash, test_implementations

import fastinference.Loader

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Benchmark various tree optimizations on the supplied dataset.')
    parser.add_argument('--outpath', required=True, help='Folder where data should written to.')
    parser.add_argument('--dataset','-d', required=True, help='Dataset to to be downloaded and used. Currently supported are {magic, mnist, fashion, eeg}.')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    parser.add_argument('--split','-s', required=False, default=0.2, type=float, help='Test/Train split.')
    args = parser.parse_args()
    
    performance = []
    model = []
    base_optimizers = [([None], [{}])]
    impl_folder = "cuda"

    # XTrain, YTrain, _, _ = get_dataset(args.dataset,split=args.split)

    implementations = [ 
        ("cpu",{"label_type":"int"},{"feature_type":"int"}), 
        ("x",{"label_type":"int"},{"feature_type":"int"}),
        ("y",{"label_type":"int"},{"feature_type":"int"}),
        ("z",{"label_type":"int"},{"feature_type":"int"}),
        ("xy",{"label_type":"int"},{"feature_type":"int"}),
        ("xz",{"label_type":"int"},{"feature_type":"int"}),
        ("yz",{"label_type":"int"},{"feature_type":"int"}),
        ("xyz",{"label_type":"int"},{"feature_type":"int"})
    ]

    # if args.nestimators <= 1:
    #     model = DecisionTreeClassifier(max_depth=args.maxdepth)
    #     base_optimizers = [
    #         ([None], [{}]),
    #         (["quantize"],[{"quantize_splits":"rounding", "quantize_leafs":1000}]),
    #         (["quantize"],[{"quantize_leafs":1000, "quantize_splits":1000}]),
    #     ]
    #     ensemble_optimizers = [
    #         ([None], [{}])
    #     ]
    # else:
    #     model = RandomForestClassifier(n_estimators=args.nestimators, max_depth=args.maxdepth, max_leaf_nodes=512)
    #     base_optimizers = [
    #         ([None], [{}]),
    #     ]

    #     ensemble_optimizers = [
    #         ([None], [{}]),
    #         #(["quantize"],[{"quantize_splits":"rounding", "quantize_leafs":1000}]),
    #         #(["leaf-refinement"], [{"X":XTrain, "Y":YTrain, "epochs":1, "optimizer":"adam", "verbose":True}]),
    #         #(["weight-refinement"], [{"X":XTrain, "Y":YTrain, "epochs":1, "optimizer":"sgd", "verbose":True}])
    #     ]

    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations, base_optimizers = base_optimizers, out_path = args.outpath, model_name = args.modelname, impl_folder = impl_folder)

    df = pd.DataFrame(performance)
    with pd.option_context('display.max_rows', None): 
        print(df)

if __name__ == '__main__':
    main()
    
