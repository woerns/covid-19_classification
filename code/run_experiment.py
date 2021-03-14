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
    parser.add_argument('-t', '--model_type', choices=['branching', 'ensemble'], default='branching')
    parser.add_argument('--heads', type=int, default=10)
    parser.add_argument('--conf_level', type=float, default=0.0)
    parser.add_argument('--swag', action='store_true', default=False)
    parser.add_argument('--branchout', action='store_true', default=False)

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr_halflife', type=int, default=5)   # halve learning rate every x epochs
    parser.add_argument('-swag_lr', '--swag_learning_rate', type=float, default=0.0001)
    parser.add_argument('--swag_momentum', type=float, default=0.9)
    parser.add_argument('--swag_rank', type=int, default=10)
    parser.add_argument('--swag_samples', type=int, default=10)
    parser.add_argument('--swag_start', type=float, default=0.8)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--bootstrap', action='store_true', default=False)
    parser.add_argument('--eval_interval', type=int, default=5)

    # Dataset
    parser.add_argument('--dataset', type=str, default='ucsd-ai4h')
    parser.add_argument('--data_root_dir', type=str, default='../datasets')
    # Note: Datasets must be stored under the data_root_dir
    # using dataset name, i.e. {data_root_dir}/{dataset}/{dataset_version}/
    # e.g. ../datasets/ucsd-ai4h/full/CT_COVID/img001.png
    # The images in the subdirectory are grouped by the label (CT_COVID in the given example).

    # Others
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', nargs='+', type=int, default=[42, 22, 719662, 945304, 139494, 386078, 307341, 328977, 323004, 795956])

    # Process input arguments
    args = parser.parse_args()
    if args.heads == 1:
        args.conf_level = 0.0
    if args.run_name == '':
        args.run_name = "_".join([args.model_name,
                                "{0}{1:d}".format(args.model_type, args.heads)])
        if args.swag:
            args.run_name += "_swag{0:d}".format(args.swag_samples)
        if args.bootstrap:
            args.run_name += "_bs"
        args.run_name = "_".join([args.run_name,
                                  args.dataset.lower(),
                                "cl{0:.2f}".format(args.conf_level),
                                datetime.datetime.now().strftime("%Y%m%d-%H%M")])
    if args.conf_level == 0.0:
        args.conf_level = None

    seed_list = args.seed
    # Load training and validation data
    X, y, groups = load_dataset(args.dataset, args.data_root_dir, dataset_version='full')
    # Load test data
    X_test, y_test, _ = load_dataset(args.dataset, args.data_root_dir, dataset_version='test')

    start = time.time()
    for seed in seed_list:
        # Set seeds
        args.seed = seed
        place_seeds(args.seed)
        print("Running experiment with seed %d..." % args.seed)
        # Cross-validate model
        cv_models, calibration_models = crossvalidate(X, y, groups, args, X_test=X_test, y_test=y_test)

    time_elapsed = time.time() - start
    print(f"Total train time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print("Done.")


    print("done")


if __name__ == '__main__':
    run_cv()
