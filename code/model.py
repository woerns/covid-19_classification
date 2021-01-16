import numpy as np
import torch

from data import CTImageDataSet
from utils import create_bs_network, load_data_transform, create_bs_train_loader, train

import sklearn
import sklearn.model_selection


def estimate(X_train, y_train, args):
    """
        Function to train model on input data.
    """
    N_BOOTSTRAP = args.bs_heads
    STD_THRESHOLD = args.std_threshold
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LEARNING_RATE = args.learning_rate
    MODEL_NAME = args.model_name
    RUN_NAME = args.run_name
    DEVICE = args.device

    # Create model
    model = create_bs_network(MODEL_NAME, output_dim=N_BOOTSTRAP)

    # Preprocess images and labels
    data_transform = load_data_transform(train=True)
    train_dataset = CTImageDataSet(X_train, y_train, transform=data_transform)
    # Create bootstrap datasets
    bs_train_loader = create_bs_train_loader(train_dataset, N_BOOTSTRAP, batch_size=BATCH_SIZE)

    # Train model
    print("Training model...")
    train(model, bs_train_loader, run_name=RUN_NAME, n_epochs=N_EPOCHS, lr=LEARNING_RATE,
          std_threshold=STD_THRESHOLD, device=DEVICE)
    print("Training completed.")

    return model


def crossvalidate(X, y, groups, args):
    """
        Function to cross-validate model on input data.
    """
    N_BOOTSTRAP = args.bs_heads
    STD_THRESHOLD = args.std_threshold
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.n_epochs
    LEARNING_RATE = args.learning_rate
    MODEL_NAME = args.model_name
    RUN_NAME = args.run_name
    DEVICE = args.device
    CV_FOLDS = args.cv_folds

    print("Run configuration:")
    for k, v in args.__dict__.items():
        print("{0}: {1}".format(k, v))

    # Create cross-validation splits
    group_kfold = sklearn.model_selection.GroupKFold(n_splits=CV_FOLDS)
    models = []

    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
        print("Running fold %d..." % fold)

        X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
        X_val, y_val = [X[i] for i in val_idx], [y[i] for i in val_idx]
        print("Training samples: %d" % len(y_train))
        print("Validation samples: %d" % len(y_val))

        # Create train datasets
        train_data_transform = load_data_transform(train=True)
        train_dataset = CTImageDataSet(X_train, y_train, transform=train_data_transform)

        # Create bootstrap datasets
        bs_train_loader = create_bs_train_loader(train_dataset, N_BOOTSTRAP, batch_size=BATCH_SIZE)

        # Create validation datasets
        val_data_transform = load_data_transform(train=False)
        val_dataset = CTImageDataSet(X_val, y_val, transform=val_data_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=2)
        # Create model
        model = create_bs_network(MODEL_NAME, output_dim=N_BOOTSTRAP)

        # Train model
        print("Training model...")
        train(model, bs_train_loader, run_name=RUN_NAME+"_fold{0:d}".format(fold), n_epochs=N_EPOCHS,
              lr=LEARNING_RATE, std_threshold=STD_THRESHOLD, val_loader=val_loader, device=DEVICE)
        print("Training completed.")

        models.append(model)

    print("Cross-validation completed.")

    return models


def predict(X_test, model=None, std_threshold=0.0):
    # import torch
    # model = torch.load(model+'/Model.pth', map_location=torch.device('cpu'))

    model.eval()

    # Preprocess images
    data_transform = load_data_transform(train=False)
    test_dataset = CTImageDataSet(X_test, [0]*len(X_test), transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2)

    y_pred = []
    dydX = []

    for data in test_loader:
        images, dummy_targets = data
        images.requires_grad = True
        outputs = model(images)
        input_grads = []
        bs_heads = len(outputs[0])
        for i in range(bs_heads):
            input_grad = torch.autograd.grad(outputs[0][i], images, retain_graph=(i<bs_heads))[0]
            input_grads.append(input_grad[0].mean(axis=0).numpy())  # Average across all RGB channels
        dydX.append(np.stack(input_grads))

        # need to average multiple predictions of bootstrap net
        mean_output = torch.sigmoid(outputs).data.mean(dim=-1)
        std_output = torch.sigmoid(outputs).data.std(dim=-1)
        predicted = (mean_output + std_threshold*std_output > 0.5).int()

        if predicted == 1:
            y_pred.append('COVID')
        else:
            y_pred.append('NonCOVID')

    return y_pred, dydX