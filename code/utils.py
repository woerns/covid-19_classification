import numpy as _np

import torch
from torchvision import transforms

from models import CNN, BootstrapLinear


def load_model(model_name):
    if model_name == 'cnn':
        # Simple CNN
        model = CNN(output_size=1)
    elif model_name == 'resnet18':
        # ResNet
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # Replace last layer
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
    elif model_name == 'resnet18_bs10':
        # ResNet
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # Replace last layer
        num_ftrs = model.fc.in_features
        model.fc = BootstrapLinear(num_ftrs, 10)
    elif model_name == 'resnet18_bs50':
        # ResNet
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # Replace last layer
        num_ftrs = model.fc.in_features
        model.fc = BootstrapLinear(num_ftrs, 50)
    else:
        raise ("Unknown model name.")

    return model


def load_transform(transform_name):
    if transform_name == 'grayscale':
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=1 / _np.sqrt(12))
        ])
    elif transform_name == 'rgb':
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif transform_name == 'grayscale_augmented':
        # To be changed
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=1 / _np.sqrt(12))
        ])
    elif transform_name == 'rgb_augmented':
        # To be changed
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ("Unknown transform.")

    return data_transform


def load_criterion(criterion_name):
    if criterion_name == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_name == 'binary_crossentropy':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ("Unknown transform.")

    return criterion
