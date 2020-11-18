import numpy as np

import random
import torch
import time

from utils import load_img_data
from model import estimate, predict, crossvalidate

def place_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_cv():
    place_seeds(42)

    start = time.time()

    # Load training data
    TRAIN_DATA_DIR = "../../dataset/full"
    X_train, y_train, groups = load_img_data(TRAIN_DATA_DIR)

    # Cross-validate model
    models = crossvalidate(X_train, y_train, groups, n_folds=5)

    time_elapsed = time.time() - start
    print(f"Total train time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    print("done")


def run_test():
    place_seeds(42)

    start = time.time()

    # Load training data
    TRAIN_DATA_DIR = "../../dataset/full"
    X_train, y_train, groups = load_img_data(TRAIN_DATA_DIR)

    # Train model on full training dataset
    model = estimate(X_train, y_train)

    time_elapsed = time.time() - start
    print(f"Total train time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Load model and perform inference
    model = torch.load('densenet169_nbootstrap10.pth', map_location=torch.device('cpu'))
    TEST_DATA_DIR = "../../dataset/val"
    X_test, y_test = load_img_data(TEST_DATA_DIR)

    y_pred = predict(X_test, model)

    print("done")


if __name__ == '__main__':
    run_test()
