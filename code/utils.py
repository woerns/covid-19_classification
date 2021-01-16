import os
import random

import numpy as np
import sklearn
import sklearn.metrics

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def load_data_transform(train=False, add_mask=False):
    if add_mask:
        mean = [0.485, 0.456, 0.406, 0.406]
        std = [0.229, 0.224, 0.225, 0.225]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if train:
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.5, 1.5)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return data_transform


def create_bs_network(model_name,output_dim=10, add_mask=False):

    if 'resnet' in model_name:
        # ResNet Full
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)

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
    elif 'densenet' in model_name:
        # DenseNet
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        # Replace last layer
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, output_dim)
    else:
        raise ValueError("Unknown model name %s." % model_name)

    return model


def create_bs_train_loader(dataset, n_bootstrap, batch_size=16):
    bs_train_loader = []
    train_idx = list(range(len(dataset)))

    if n_bootstrap == 1:
        # If only one head, apply default case and do not bootstrap
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        bs_train_loader.append(train_loader)
    else:
        for _ in range(n_bootstrap):
            train_idx_bs = sklearn.utils.resample(train_idx, replace=True, n_samples=len(dataset))
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx_bs)

            train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=2)
            bs_train_loader.append(train_loader)

    return bs_train_loader


def train(model, bs_train_loader, run_name, n_epochs=10, lr=0.0001, std_threshold=0, val_loader=None, device='cpu'):

    # push model to set device (CPU or GPU)
    model.to(device)

    model.train()

    n_bootstrap = len(bs_train_loader)
    steps_per_epoch = len(bs_train_loader[0])
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_prob = torch.nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    writer = SummaryWriter("./runs/" + run_name)

    iterators = [iter(x) for x in bs_train_loader]

    for epoch in range(n_epochs):  # loop over the datasets multiple times
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

            # push tensors to set device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

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
                print('[%d, %3d] train_loss: %.5f' %
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
                            # push tensors to set device (CPU or GPU)
                            images, labels = images.to(device), labels.to(device)

                            outputs = model(images)
                            # need to average multiple predictions of bootstrap net
                            mean_output = torch.sigmoid(outputs).data.mean(dim=-1)
                            if n_bootstrap > 1:
                                std_output = torch.sigmoid(outputs).data.std(dim=-1)
                            else:
                                std_output = 0

                            loss = criterion_prob(mean_output, labels.float())
                            val_loss += loss.item()

                            # Compute accuracy
                            predicted = (mean_output + std_threshold * std_output > 0.5).int()
                            total += labels.size(0)
                            correct += (predicted == labels.int()).sum().item()
                            acc = correct / total

                            # Compute class probabilities
                            class_probs.append(mean_output.cpu().numpy())
                            y_pred.append(predicted.cpu().numpy())
                            y_test.append(labels.cpu().numpy())

                    y_test = np.concatenate(y_test)
                    y_pred = np.concatenate(y_pred)
                    class_probs = np.concatenate(class_probs)

                    print('[%d, %3d] val_loss: %.3f' %
                          (epoch + 1, i + 1, val_loss / len(val_loader)))
                    print('[%d, %3d] val_acc: %.1f %%' % (epoch + 1, i + 1, 100 * acc))
                    model.train()

                    # Write to tensorboard
                    writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch * steps_per_epoch + i + 1)
                    writer.add_scalar('Acc/val', acc, epoch * steps_per_epoch + i + 1)
                    writer.add_scalar('AUC/val', sklearn.metrics.roc_auc_score(y_test, class_probs), epoch * steps_per_epoch + i + 1)
                    writer.add_scalar('F1/val', sklearn.metrics.f1_score(y_test, y_pred), epoch * steps_per_epoch + i + 1)

        scheduler.step()

    # Save model
    model_save_dir = "./models"
    torch.save(model, os.path.join(model_save_dir, run_name + '.pth'))