import numpy as np

import random
import torch
import time

from utils import load_img_data
from model import estimate, predict


def run_test():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    start = time.time()

    # Load training data
    TRAIN_DATA_DIR = "../../dataset"
    X_train, y_train = load_img_data(TRAIN_DATA_DIR)

    # Train model
    model = estimate(X_train, y_train)

    time_elapsed = time.time() - start
    print(f"Total train time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Load model
    model = torch.load('densenet169_nbootstrap10.pth', map_location=torch.device('cpu'))
    TEST_DATA_DIR = "../../dataset/val"
    X_test, y_test = load_img_data(TEST_DATA_DIR)

    y_pred = predict(X_test, model)

    print("done")


if __name__ == '__main__':
    run_test()
