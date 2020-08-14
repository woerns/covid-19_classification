import time

from torchvision import datasets

from utils import load_model, load_transform, load_criterion
from models import ModelCrossValidation

import pickle


def run_experiment():
    MODEL_NAME_LIST = [
                               'cnn',
                               'resnet18',
                                'resnet18_bs10',
                                'resnet18_bs50'
    ]

    TRANSFORM_MAP = {'cnn': 'grayscale',
                     'resnet18': 'rgb',
                     'resnet18_bs10': 'rgb',
                     'resnet18_bs50': 'rgb'}

    CRITERION_MAP = {'cnn': 'binary_crossentropy',
                     'resnet18': 'binary_crossentropy',
                     'resnet18_bs10': 'binary_crossentropy',
                     'resnet18_bs50': 'binary_crossentropy'}

    DATASET_DIR = '../dataset'
    NUM_EPOCHS = 5  # tmp setting for testing, not optimized
    N_FOLDS = 5

    results = {}

    for model_name in MODEL_NAME_LIST:
        start = time.time()
        print("Testing model %s..." % model_name)
        model = load_model(model_name)
        data_transform = load_transform(TRANSFORM_MAP[model_name])
        criterion = load_criterion(CRITERION_MAP[model_name])

        ct_dataset = datasets.ImageFolder(root=DATASET_DIR,
                                          transform=data_transform)

        model_cv = ModelCrossValidation(model, criterion)
        model_cv.crossvalidate(ct_dataset, n_folds=N_FOLDS, n_epochs=NUM_EPOCHS)

        time_elapsed = time.time() - start
        print(f"Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        results[model_name] = model_cv


    with open('results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None


if __name__ == '__main__':

    run_experiment()

    print("done.")