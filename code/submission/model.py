import torch

from submission.utils import CTImageDataSet
from submission.utils import create_bs_resnet, load_data_transform, create_bs_train_loader, train, load_img_data

ADD_MASK = False

def estimate(X_train, y_train):
    N_BOOTSTRAP = 10
    BATCH_SIZE = 32
    N_EPOCHS = 30
    LEARNING_RATE = 0.0001
    LABEL_MAP = {'COVID': 0, 'NonCOVID': 1}
    DEVICE = 'cuda'

    # Create model
    model = create_bs_resnet(output_dim=N_BOOTSTRAP, add_mask=ADD_MASK)

    # Preprocess images and labels
    data_transform = load_data_transform(train=True, add_mask=ADD_MASK)
    y_train = [LABEL_MAP[y] for y in y_train]
    train_dataset = CTImageDataSet(X_train, y_train, transform=data_transform, add_mask=ADD_MASK)
    # Create bootstrap datasets
    bs_train_loader = create_bs_train_loader(train_dataset, N_BOOTSTRAP, batch_size=BATCH_SIZE)

    # Val loader
    data_transform = load_data_transform(train=False, add_mask=ADD_MASK)
    VAL_DATA_DIR = "../../val"
    X_val, y_val = load_img_data(VAL_DATA_DIR)
    y_val = [LABEL_MAP[y] for y in y_val]
    val_dataset = CTImageDataSet(X_val, y_val, transform=data_transform, add_mask=ADD_MASK)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2)

    # Train model
    print("Training model...")
    train(model, bs_train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE, val_loader=val_loader, device=DEVICE)
    print("Training completed.")

    return model


def predict(X_test, model):
    model.eval()
    # Preprocess images
    data_transform = load_data_transform(train=False, add_mask=ADD_MASK)
    test_dataset = CTImageDataSet(X_test, [0]*len(X_test), transform=data_transform, add_mask=ADD_MASK)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2)

    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            images, dummy_targets = data
            outputs = model(images)
            # need to average multiple predictions of bootstrap ResNet
            mean_output = outputs.data.mean(dim=-1)
            predicted = (mean_output > 0).int()
            if predicted == 1:
                y_pred.append('NonCOVID')
            else:
                y_pred.append('COVID')

    return y_pred