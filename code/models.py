import numpy as _np
import random

import copy

import datetime

import sklearn
import sklearn.model_selection

import torch
from torch.utils.tensorboard import SummaryWriter


### Simple CNN ###

class CNN(torch.nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        #         self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 16, 5)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.fc1 = torch.nn.Linear(32 * 53 * 53, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        #         x = self.bn1(x)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, (2, 2))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


### Bootstrap Linear Layer ###

class BootstrapLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(BootstrapLinear, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        self.output_size = output_size

    def forward(self, x):
        x = self.linear(x)

        if self.training:
            #             print(x.shape)
            output_idx = random.randint(0, self.output_size - 1)
            x = x[:, [output_idx]]
        # else:
        #     x = x.mean(dim=-1)

        return x


### Helper class for model cross validation ###

class ModelCrossValidation():
    def __init__(self, model, criterion):
        self.model = model
        self.cv_models = []
        self.criterion = criterion
        self.cv_metrics = None
        self.writer = None

    def train(self, model, train_loader, n_epochs=10, val_loader=None):
        model.train()

        steps_per_epoch = len(train_loader)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs, labels.view((-1, 1)).float())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                self.writer.add_scalar('Loss/train', loss.float(), epoch * steps_per_epoch + i + 1)
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %3d] train_loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))

                    running_loss = 0.0

                    if val_loader is not None:
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for data in val_loader:
                                images, labels = data
                                outputs = model(images)
                                # need to average multiple predictions of bootstrap net
                                mean_output = outputs.data.mean(dim=-1)
                                loss = self.criterion(mean_output, labels.float())
                                val_loss += loss.item()

                        self.writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch * steps_per_epoch + i + 1)
                        model.train()

    def evaluate(self, model, val_loader):
        model.eval()

        correct = 0
        total = 0
        #         running_loss = 0.0
        y_test = []
        y_pred = []
        class_probs = []
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = model(images)
                # need to average multiple predictions of bootstrap net
                mean_output = outputs.data.mean(dim=-1)
                #                 loss = self.criterion(outputs, labels)
                #                 running_loss += loss
                y_test.append(labels.numpy())
                # Compute predicted labels
                predicted = (mean_output > 0).int()  # For binary cross entropy loss output
                #                 _, predicted = torch.max(outputs.data, 1) # For cross entropy loss output
                y_pred.append(predicted.numpy())

                # Compute class probabilities
                class_probs.append(torch.sigmoid(mean_output).numpy())
                #                 class_probs.append(torch.softmax(outputs,1).numpy())  # For cross entropy loss output
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('val_acc: %.1f %%' % (100 * correct / total))

        y_test = _np.concatenate(y_test)
        y_pred = _np.concatenate(y_pred)
        class_probs = _np.concatenate(class_probs)

        performance_metrics = {'acc': correct / total,
                               #                                'auc': sklearn.metrics.roc_auc_score(y_test, class_probs[:,1]), # For cross entropy loss output
                               'auc': sklearn.metrics.roc_auc_score(y_test, class_probs),
                               'f1_score': sklearn.metrics.f1_score(y_test, y_pred),
                               'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, y_pred)}

        return performance_metrics

    def crossvalidate(self, dataset, n_folds=5, n_epochs=5):
        splits = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset)):
            print("Running fold %d..." % fold)

            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=16,
                                                       sampler=train_sampler,
                                                       #                                                shuffle=True,
                                                       num_workers=0)

            val_data = torch.utils.data.Subset(dataset, val_idx)
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=0)

            model = copy.deepcopy(self.model)

            self.writer = SummaryWriter('./runs/fold{0}'.format(fold))

            self.train(model, train_loader, n_epochs=n_epochs, val_loader=val_loader)
            self.cv_models.append(model)
            self.cv_metrics.append(self.evaluate(model, val_loader))

            self.writer.close()


class BSModelCrossValidation():
    def __init__(self, model, criterion):
        self.model = model
        self.n_bootstrap = list(model.parameters())[-1].size()[0]
        self.seed = 42
        self.cv_models = []
        self.criterion = criterion
        self.cv_metrics = None
        self.writer = None
        self.log_dir = './runs/'

    def train(self, model, train_loader_bs, n_epochs=10, val_loader=None):
        model.train()

        steps_per_epoch = len(train_loader_bs[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        iterators = [iter(x) for x in train_loader_bs]

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for i in range(steps_per_epoch):
                k = random.randint(0, self.n_bootstrap - 1)

                try:
                    # get the next item
                    inputs, labels = next(iterators[k])
                except StopIteration:
                    # restart if we reached the end of iterator
                    iterators[k] = iter(train_loader_bs[k])
                    inputs, labels = next(iterators[k])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs[:, [k]], labels.view((-1, 1)).float())
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                self.writer.add_scalar('Loss/train', loss.float(), epoch * steps_per_epoch + i + 1)
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %3d] train_loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))

                    running_loss = 0.0

                    if val_loader is not None:
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for data in val_loader:
                                images, labels = data
                                outputs = model(images)
                                # need to average multiple predictions of bootstrap net
                                mean_output = outputs.data.mean(dim=-1)
                                loss = self.criterion(mean_output, labels.float())
                                val_loss += loss.item()

                        model.train()

                        self.writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch * steps_per_epoch + i + 1)

            scheduler.step()

    def evaluate(self, model, val_loader):
        model.eval()

        correct = 0
        total = 0
        #         running_loss = 0.0
        y_test = []
        y_pred = []
        class_probs = []
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                outputs = model(images)
                # need to average multiple predictions of bootstrap net
                mean_output = outputs.data.mean(dim=-1)
                #                 loss = self.criterion(outputs, labels)
                #                 running_loss += loss
                y_test.append(labels.numpy())
                # Compute predicted labels
                predicted = (mean_output > 0).int()  # For binary cross entropy loss output
                #                 _, predicted = torch.max(outputs.data, 1) # For cross entropy loss output
                y_pred.append(predicted.numpy())

                # Compute class probabilities
                class_probs.append(torch.sigmoid(mean_output).numpy())
                #                 class_probs.append(torch.softmax(outputs,1).numpy())  # For cross entropy loss output
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('val_acc: %.1f %%' % (100 * correct / total))

        y_test = _np.concatenate(y_test)
        y_pred = _np.concatenate(y_pred)
        class_probs = _np.concatenate(class_probs)

        performance_metrics = {'acc': correct / total,
                               #                                'auc': sklearn.metrics.roc_auc_score(y_test, class_probs[:,1]), # For cross entropy loss output
                               'auc': sklearn.metrics.roc_auc_score(y_test, class_probs),
                               'f1_score': sklearn.metrics.f1_score(y_test, y_pred),
                               'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, y_pred)}

        return performance_metrics

    def crossvalidate(self, dataset, n_folds=5, n_epochs=5):
        # Set random seed
        _np.random.seed(self.seed)

        splits = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True)
        self.cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits.split(dataset)):
            print("Running fold %d..." % fold)

            # Create n_boostrap datasets
            train_loader_bs = []
            for _ in range(self.n_bootstrap):
                train_idx_bs = sklearn.utils.resample(train_idx, replace=True, n_samples=len(train_idx))
                train_sampler = torch.utils.data.SubsetRandomSampler(train_idx_bs)

                train_loader = torch.utils.data.DataLoader(dataset,
                                                           batch_size=16,
                                                           sampler=train_sampler,
                                                           num_workers=0)
                train_loader_bs.append(train_loader)

            val_data = torch.utils.data.Subset(dataset, val_idx)
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=0)

            model = copy.deepcopy(self.model)

            self.writer = SummaryWriter(
                self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M") + "/fold{0}".format(fold))

            self.train(model, train_loader_bs, n_epochs=n_epochs, val_loader=val_loader)
            self.cv_models.append(model)
            self.cv_metrics.append(self.evaluate(model, val_loader))

            self.writer.close()