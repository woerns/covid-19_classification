import numpy as np

import random
import torch
import time
import datetime
import argparse

from data import load_dataset
from model import estimate, predict, crossvalidate


def place_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cv():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('-r', '--run_name', type=str, default='')
    parser.add_argument('-m', '--model_name', default='densenet169')
    parser.add_argument('--bs_heads', type=int, default=10)
    parser.add_argument('--std_threshold', type=float, default=0.0)

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--cv_folds', type=int, default=5)

    # Dataset
    parser.add_argument('--dataset', type=str, default='ucsd-ai4h')
    parser.add_argument('--data_root_dir', type=str, default='../datasets')
    # Note: Datasets must be stored under the data_root_dir
    # using dataset name, i.e. {data_root_dir}/{dataset}/{dataset_version}/
    # e.g. ../datasets/ucsd-ai4h/full/CT_COVID/img001.png
    # The images in the subdirectory are grouped by the label (CT_COVID in the given example).

    # Others
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if args.run_name == '':
        args.run_name = "_".join([args.model_name,
                             "bs{0:d}".format(args.bs_heads),
                             "stdth{0:.1f}".format(args.std_threshold),
                             datetime.datetime.now().strftime("%Y%m%d-%H%M")])
    start = time.time()

    # Set seeds
    place_seeds(args.seed)

    # Load data
    X, y, groups = load_dataset(args.dataset, args.data_root_dir)

    # Cross-validate model
    models = crossvalidate(X, y, groups, args)

    time_elapsed = time.time() - start
    print(f"Total train time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    print("done")


if __name__ == '__main__':
    run_cv()
