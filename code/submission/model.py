import numpy as np

import torch
from torchvision import datasets

from submission.utils import CTImageDataSet
from submission.utils import create_bs_resnet, load_data_transform, create_bs_train_loader, train, load_img_data

ADD_MASK = False

def estimate(X_train, y_train):
    N_BOOTSTRAP = 10
    # Create model
    model = create_bs_resnet(output_dim=N_BOOTSTRAP, add_mask=ADD_MASK)

    # Preprocess images and create bootstrap datasets
    data_transform = load_data_transform(train=True, add_mask=ADD_MASK)
    train_dataset = CTImageDataSet(X_train, y_train, transform=data_transform, add_mask=ADD_MASK)
    bs_train_loader = create_bs_train_loader(train_dataset, N_BOOTSTRAP, batch_size=16)

    # Val loader
    data_transform = load_data_transform(train=False, add_mask=ADD_MASK)
    VAL_DATA_DIR = "../../val"
    X_val, y_val = load_img_data(VAL_DATA_DIR)
    val_dataset = CTImageDataSet(X_val, y_val, transform=data_transform, add_mask=ADD_MASK)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2)

    # Train model
    print("Training model...")
    train(model, bs_train_loader, n_epochs=20, val_loader=val_loader)
    print("Training completed.")

    return model


def predict(X_test, model):
    model.eval()
    # Preprocess images
    data_transform = load_data_transform(train=False, add_mask=ADD_MASK)
    test_dataset = CTImageDataSet(X_test, [None]*len(X_test), transform=data_transform, add_mask=ADD_MASK)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2)

    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, targets = data
            outputs = model(images)
            # need to average multiple predictions of bootstrap ResNet
            mean_output = outputs.data.mean(dim=-1)
            predicted = (mean_output > 0).int()
            y_pred.append(predicted)

    return y_pred