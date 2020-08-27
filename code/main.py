import time

from torchvision import datasets

from utils import load_model, load_transform, load_criterion
from models import BSModelCrossValidation

import pickle


def run_experiment():
    MODEL_NAME_LIST = [
                               'cnn',
                               'resnet18',
                                'resnet18_bs10_full',
                                'resnet18_bs10_last',
                                'resnet18_bs50'
    ]

    TRANSFORM_MAP = {'cnn': 'grayscale',
                     'default': 'rgb'
                    }

    CRITERION_MAP = {'cnn': 'binary_crossentropy',
                     'default': 'binary_crossentropy'}

    DATASET_DIR = '../dataset'
    NUM_EPOCHS = 10
    N_FOLDS = 5

    results = {}

    for model_name in MODEL_NAME_LIST:
        start = time.time()
        print("Testing model %s..." % model_name)
        model = load_model(model_name)
        data_transform = load_transform(
            TRANSFORM_MAP[model_name] if model_name in TRANSFORM_MAP else TRANSFORM_MAP['default'])
        criterion = load_criterion(
            CRITERION_MAP[model_name] if model_name in CRITERION_MAP else CRITERION_MAP['default'])

        ct_dataset = datasets.ImageFolder(root=DATASET_DIR,
                                              transform=data_transform)

        model_cv = BSModelCrossValidation(model, criterion)
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