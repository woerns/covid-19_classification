import torch

from utils import CTImageDataSet
from utils import create_bs_network, load_data_transform, create_bs_train_loader, train, load_img_data

def estimate(X_train, y_train):
    N_BOOTSTRAP = 10
    BATCH_SIZE = 32
    N_EPOCHS = 30
    LEARNING_RATE = 0.0001
    MODEL_NAME = 'densenet169'
    LABEL_MAP = {'COVID': 1, 'NonCOVID': 0}
    DEVICE = 'cpu'

    # Create model
    model = create_bs_network(MODEL_NAME, output_dim=N_BOOTSTRAP)

    # Preprocess images and labels
    data_transform = load_data_transform(train=True)
    y_train = [LABEL_MAP[y] for y in y_train]
    train_dataset = CTImageDataSet(X_train, y_train, transform=data_transform)
    # Create bootstrap datasets
    bs_train_loader = create_bs_train_loader(train_dataset, N_BOOTSTRAP, batch_size=BATCH_SIZE)

    # Val loader
    VAL_DATA_DIR = None  # Please specify directory where validation data is stored
    if VAL_DATA_DIR is not None:
        data_transform = load_data_transform(train=False)
        X_val, y_val = load_img_data(VAL_DATA_DIR)
        y_val = [LABEL_MAP[y] for y in y_val]
        val_dataset = CTImageDataSet(X_val, y_val, transform=data_transform)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=2)
    else:
        val_loader = None

    # Train model
    print("Training model...")
    train(model, bs_train_loader, model_name=MODEL_NAME, n_epochs=N_EPOCHS, lr=LEARNING_RATE, val_loader=val_loader, device=DEVICE)
    print("Training completed.")

    return model


def predict(X_test, model=None):
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

    with torch.no_grad():
        for data in test_loader:
            images, dummy_targets = data
            outputs = model(images)
            # need to average multiple predictions of bootstrap ResNet
            mean_output = torch.sigmoid(outputs).data.mean(dim=-1)
            std_output = torch.sigmoid(outputs).data.std(dim=-1)
            predicted = (mean_output + 2*std_output > 0.5).int()

            if predicted == 1:
                y_pred.append('COVID')
            else:
                y_pred.append('NonCOVID')

    return y_pred