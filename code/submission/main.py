import numpy as np

import random
import torch
import time

from submission.utils import load_img_data
from submission.model import estimate


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

    # Save model
    # torch.save(model, 'resnet18_augv2.pickle')

    # Load model
    # model = torch.load('Model.pickle')

    # y_pred = predict(X_test, model)


if __name__ == '__main__':
    run_test()

    print("done.")