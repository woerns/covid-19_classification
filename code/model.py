import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from data import CTImageDataSet
from utils import load_data_transform, create_bs_train_loader, train, evaluate
from networks import create_model, create_branching_network

import sklearn
import sklearn.model_selection


def estimate(X_train, y_train, args):
    """
        Function to train model on input data.
    """
    N_BOOTSTRAP = args.bs_heads
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LEARNING_RATE = args.learning_rate
    MODEL_NAME = args.model_name
    RUN_NAME = args.run_name
    DEVICE = args.device

    # Create model
    model = create_branching_network(MODEL_NAME, n_heads=N_BOOTSTRAP)

    # Preprocess images and labels
    data_transform = load_data_transform(train=True)
    train_dataset = CTImageDataSet(X_train, y_train, transform=data_transform)
    # Create bootstrap datasets
    bs_train_loader = create_bs_train_loader(train_dataset, N_BOOTSTRAP, batch_size=BATCH_SIZE)

    # Train model
    print("Training model...")
    results = train(model, bs_train_loader, run_name=RUN_NAME, n_epochs=N_EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    print("Training completed.")

    return model


def crossvalidate(X, y, groups, args, X_test=None, y_test=None):
    """
        Function to cross-validate model on input data.
    """
    N_HEADS = args.heads
    CONFIDENCE_LEVEL = args.conf_level
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LEARNING_RATE = args.learning_rate
    MODEL_NAME = args.model_name
    MODEL_TYPE = args.model_type
    USE_BOOTSTRAP = args.bootstrap
    USE_SWAG = args.swag
    RUN_NAME = args.run_name
    DEVICE = args.device
    CV_FOLDS = args.cv_folds
    NULL_HYPOTHESIS = args.null_hypothesis

    print("Run configuration:")
    for k, v in args.__dict__.items():
        print("{0}: {1}".format(k, v))

    with open(os.path.join('./models', RUN_NAME + '.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if X_test is not None and y_test is not None:
        # Create test dataset
        test_data_transform = load_data_transform(train=False)
        test_dataset = CTImageDataSet(X_test, y_test, transform=test_data_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0)

    # Create cross-validation splits
    group_kfold = sklearn.model_selection.GroupKFold(n_splits=CV_FOLDS)
    models = []
    calibration_models = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
        print("Running fold %d..." % fold)

        X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
        X_val, y_val = [X[i] for i in val_idx], [y[i] for i in val_idx]
        print("Training samples: %d" % len(y_train))
        print("Validation samples: %d" % len(y_val))

        # Create train dataset
        train_data_transform = load_data_transform(train=True)
        train_dataset = CTImageDataSet(X_train, y_train, transform=train_data_transform)

        if USE_BOOTSTRAP:
            # Create bootstrap dataset
            train_loader = create_bs_train_loader(train_dataset, N_HEADS, batch_size=BATCH_SIZE)
        else:
            train_loader = [torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=0)]

        # Create validation datasets
        val_data_transform = load_data_transform(train=False)
        val_dataset = CTImageDataSet(X_val, y_val, transform=val_data_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0)

        if USE_SWAG:
            bn_update_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=0)
        else:
            bn_update_loader = None
        # Create model
        model = create_model(MODEL_NAME, MODEL_TYPE, N_HEADS, swag=USE_SWAG, swag_rank=args.swag_rank,
                             swag_samples=args.swag_samples, bn_update_loader=bn_update_loader)

        # Train model
        print("Training model...")
        results = train(model, train_loader, run_name=RUN_NAME, fold=fold, n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                        lr_hl=args.lr_halflife, swag=USE_SWAG, swag_lr=args.swag_learning_rate,
                        swag_start=args.swag_start, swag_momentum=args.swag_momentum, null_hypothesis=NULL_HYPOTHESIS,
                        confidence_level=CONFIDENCE_LEVEL, bootstrap=USE_BOOTSTRAP,
                        val_loader=val_loader, test_loader=test_loader, eval_interval=args.eval_interval, device=DEVICE)
        print("Training completed.")

        models.append(model)
        calibration_models.append(results['calibration_model'])

    print("Cross-validation completed.")

    return models, calibration_models


def predict(X_test, y_test, args, model=None, calibration_model=None):
    # import torch
    # model = torch.load(model+'/Model.pth', map_location=torch.device('cpu'))
    CONFIDENCE_LEVEL = args.conf_level
    RUN_NAME = args.run_name
    DEVICE = args.device
    NULL_HYPOTHESIS = args.null_hypothesis

    # Preprocess images
    data_transform = load_data_transform(train=False)
    test_dataset = CTImageDataSet(X_test, y_test, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0)

    log_dir = os.path.join("./runs", RUN_NAME, "test")
    writer = SummaryWriter(log_dir)

    test_results = evaluate(model, test_loader, writer, 0, 0, 0,
                            null_hypothesis=NULL_HYPOTHESIS,
                            confidence_level=CONFIDENCE_LEVEL,
                            calibration_model=calibration_model,
                            tag='test',
                            device=DEVICE)

    return test_results