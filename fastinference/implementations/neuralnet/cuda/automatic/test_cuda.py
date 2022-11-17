#!/usr/bin/env python3

import itertools
import os
import pandas as pd
import argparse
import csv

from datetime import datetime
from timeit import default_timer as timer
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

    out_path = args.outpath
    now = datetime.now().strftime('%d-%m-%y_%H-%M-%S')

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

    start = timer()
    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations, now = now, base_optimizers = base_optimizers, out_path = out_path, model_name = args.modelname, impl_folder = impl_folder)
    end = timer()

    profile_path = os.path.abspath(os.path.join(out_path, "../../..", "profiles/results", now))
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
    
    df = pd.DataFrame(performance)
    with pd.option_context('display.max_rows', None): 
        print(df)
        with open(profile_path + "/all.txt", 'a') as f:
            f.write(str(df))
    
    print("\n")
    print("Total runtime including script: %.3fs" % (end-start))
    print("\n")

    for entry in range(len(performance)):
        date_time = performance[entry].get("date_time")
        #print(date_time)
        impl = performance[entry].get("impl")
        #print(impl)
        batch_size = performance[entry].get("bch_sz")
        #print(batch_size)
        accuracy = performance[entry].get("accuracy")
        #print(accuracy)
        cpu_time = performance[entry].get("cpu_time [s]")
        #print(cpu_time)
        cpu_lat = performance[entry].get("cpu_lat [ms]")
        #print(cpu_lat)
        gpu_time = performance[entry].get("gpu_time [s]")
        #print(gpu_time)
        gpu_lat = performance[entry].get("gpu_lat [ms]")
        #print(gpu_lat)

        path_folder_csv = "profiles/csv/" + str(date_time) + "/" + str(impl)
        if not os.path.exists(path_folder_csv):
            os.makedirs(path_folder_csv)
        
        path_csv = "profiles/csv/" + str(date_time) + "/" + str(impl) + "/timings_" + str(impl) + "_" + str(batch_size) + ".csv"

        # open the file in the write mode
        with open(path_csv, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)

            header = ['batch_size', 'layer_nr', 'implem', 'cpu time [s]', 'gpu time [s]', 'total time [s]']
            writer.writerow(header)

            # write a row to the csv file
            for i in range(len(performance[entry].get("layers"))):
                writer.writerow(performance[entry].get("layers")[i])


if __name__ == '__main__':
    main()
    
