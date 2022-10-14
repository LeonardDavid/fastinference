#!/usr/bin/env python3

from functools import cache
import sys
import os
import pandas as pd

import argparse

from sklearn.datasets import fetch_openml

def main():
    parser = argparse.ArgumentParser(description='Simple tool to generate datasets of varying difficulty / number of classes / number of features.')
    parser.add_argument('--out', required=True, help='Filename where data should written to.')
    parser.add_argument('--float', required=False, default=False, action='store_true', help='True if features are float, else they are integer' )

    args = parser.parse_args()

    print("Downloading CIFAR10")
    X,y = fetch_openml(data_id=40927,return_X_y=True,as_frame=False,data_home=args.out,cache=True)
    
    if not args.float:
        X = (X * 255).astype(int)

    XTrain, YTrain = X[:50000,:], y[:50000]
    XTest, YTest = X[50000:,:], y[50000:]

    out_name = os.path.splitext(args.out)[0]

    print("Exporting data")
    dfTrainLabels = pd.concat([pd.DataFrame(YTrain,columns=["label"])], axis=1)
    dfTrain = pd.concat([pd.DataFrame(XTrain, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTrain,columns=["label"])], axis=1)
    dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
    
    dfTrainLabels.to_csv(os.path.join(out_name, "labels.csv"), header=True, index=False)
    dfTrain.to_csv(os.path.join(out_name, "training.csv"), header=True, index=False)
    dfTest.to_csv(os.path.join(out_name, "testing.csv"), header=True, index=False)

if __name__ == '__main__':
    main()
