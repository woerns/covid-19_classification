import random
import datetime

import numpy as np
import sklearn
import sklearn.metrics

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from skimage.feature import canny
from PIL import Image

class CTImageDataSet(torch.utils.data.Dataset):
    """CT Image dataset."""

    def __init__(self, X_train, y_train, transform=None, add_mask=False):

        if add_mask:
            self.x = [np.concatenate([x, np.expand_dims(canny(x.mean(axis=-1) / 255.) * 255, axis=2).astype(np.uint8)],
                            axis=2) for x in X_train]
        else:
            self.x = X_train
        # Convert numpy array to PILImage
        self.x = list(map(Image.fromarray, self.x))

        self.y = y_train
        self.n_samples = len(X_train)
        self.transform = transform


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x[idx]
        target = self.y[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def load_img_data(data_dir):
    dataset = datasets.ImageFolder(root=data_dir)

    X = []
    y = []
    for img, target in dataset:
        X.append(np.array(img))
        y.append(target)

    return X, y


def load_data_transform(train=False, add_mask=False):
    if add_mask:
        mean = [0.485, 0.456, 0.406, 0.406]
        std = [0.229, 0.224, 0.225, 0.225]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if train:
        data_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomRotation(degrees=(-10, 10)),
            # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        data_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return data_transform



def create_bs_resnet(output_dim=10, add_mask=False):
    # ResNet Full
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)

    if add_mask:
        with torch.no_grad():
            # Add additional input channel
            weight = model.conv1.weight.detach().clone()
            model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)  # here 4 indicates 4-channel input
            model.conv1.weight[:, :3] = weight
            model.conv1.weight[:, 3] = model.conv1.weight[:, 2]

    # Replace last layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, output_dim)


    # DenseNet
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    # # Replace last layer
    # num_ftrs = model.classifier.in_features
    # model.classifier = torch.nn.Linear(num_ftrs, output_dim)

    return model


def create_bs_train_loader(dataset, n_bootstrap, batch_size=16):
    bs_train_loader = []
    train_idx = list(range(len(dataset)))
    for _ in range(n_bootstrap):
        train_idx_bs = sklearn.utils.resample(train_idx, replace=True, n_samples=len(dataset))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx_bs)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=2)
        bs_train_loader.append(train_loader)

    return bs_train_loader


def train(model, bs_train_loader, n_epochs=10, val_loader=None):
    model.train()

    n_bootstrap = len(bs_train_loader)
    steps_per_epoch = len(bs_train_loader[0])
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    writer = SummaryWriter("./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    iterators = [iter(x) for x in bs_train_loader]

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i in range(steps_per_epoch):
            k = random.randint(0, n_bootstrap - 1)

            try:
                # get the next item
                inputs, labels = next(iterators[k])
            except StopIteration:
                # restart if we reached the end of iterator
                iterators[k] = iter(bs_train_loader[k])
                inputs, labels = next(iterators[k])

            # zero the parameter gradients
            # optimizer.zero_grad() # Note: this is slower
            for param in model.parameters():
                param.grad = None

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs[:, [k]], labels.view((-1, 1)).float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.float(), epoch * steps_per_epoch + i + 1)

            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %3d] train_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))

                running_loss = 0.0

                if val_loader is not None:
                    model.eval()
                    total = correct = 0.0
                    val_loss = 0.0
                    class_probs = []
                    y_test = []
                    y_pred = []

                    with torch.no_grad():
                        for data in val_loader:
                            images, labels = data
                            outputs = model(images)
                            # need to average multiple predictions of bootstrap net
                            mean_output = outputs.data.mean(dim=-1)
                            loss = criterion(mean_output, labels.float())
                            val_loss += loss.item()

                            # Compute accuracy
                            predicted = (mean_output > 0).int()
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                            # Compute class probabilities
                            class_probs.append(torch.sigmoid(mean_output).numpy())
                            y_pred.append(predicted.numpy())
                            y_test.append(labels.numpy())

                    y_test = np.concatenate(y_test)
                    y_pred = np.concatenate(y_pred)
                    class_probs = np.concatenate(class_probs)

                    print('[%d, %3d] val_loss: %.3f' %
                          (epoch + 1, i + 1, val_loss / len(val_loader)))
                    print('[%d, %3d] val_acc: %.1f %%' % (epoch + 1, i + 1, 100 * correct / total))
                    model.train()

                    # Write to tensorboard
                    writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch * steps_per_epoch + i + 1)
                    writer.add_scalar('Acc/val', correct / total, epoch * steps_per_epoch + i + 1)
                    writer.add_scalar('AUC/val', sklearn.metrics.roc_auc_score(y_test, class_probs), epoch * steps_per_epoch + i + 1)
                    writer.add_scalar('F1/val', sklearn.metrics.f1_score(y_test, y_pred), epoch * steps_per_epoch + i + 1)

        scheduler.step()

