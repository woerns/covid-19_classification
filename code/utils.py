import os
import random
import pickle

import numpy as np
import scipy as sp
import scipy.stats
import sklearn
import sklearn.metrics

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from networks import NetEnsemble
from calibration import compute_expected_calibration_error, compute_uncertainty_reliability, compute_posterior_wasserstein_dist, fit_calibration_model
from plots import plot_pred_reliability, plot_uncertainty_reliability


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


def create_sample_mask(model, branchout_layer_name):
    state_dict = model.state_dict()
    param_masks = []
    branch = False

    for param_name, _ in model.named_parameters():
        n = state_dict[param_name].numel()
        if branchout_layer_name in param_name:
            branch = True
        param_masks.append(np.zeros((n,)) + float(branch))

    sample_mask = np.concatenate(param_masks, axis=0)

    return sample_mask


def create_bs_train_loader(dataset, n_bootstrap, batch_size=16):
    bs_train_loader = []
    train_idx = list(range(len(dataset)))

    if n_bootstrap == 1:
        # If only one head, apply default case and do not bootstrap
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        bs_train_loader.append(train_loader)
    else:
        for _ in range(n_bootstrap):
            train_idx_bs = sklearn.utils.resample(train_idx, replace=True, n_samples=len(dataset))
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx_bs)

            train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=0)
            bs_train_loader.append(train_loader)

    return bs_train_loader


def fit_beta_distribution(pred_probs, dim=1):
    # Fit beta distribution using method-of-moments
    mean = pred_probs.mean(dim=dim)
    var = pred_probs.var(dim=dim)
    v = (mean*(1-mean)/var-1.0)
    alpha = mean*v
    beta = (1-mean)*v

    return alpha, beta


def evaluate(model, val_loader, writer, step_num, epoch,
             confidence_level=None, calibration_model=None, tag='val', device='cpu'):
    criterion = torch.nn.NLLLoss()
    eval_results = {}

    total = correct = 0.0
    avg_loss = 0.0
    class_probs = []
    y_true = []
    y_pred = []
    posterior_params = []

    model.eval()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data

            # push tensors to set device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)
    
            outputs = model(images)
            # outputs has shape (B, P, C) where B is batch size,
            # P is number of predictions and C is number of classes
            pred_probs = outputs.softmax(dim=-1)
            # need to average multiple predictions of all predictions
            mean_prob = pred_probs.mean(dim=-2)

            loss = criterion(mean_prob.log(), labels)
            avg_loss += loss.item()

            n_predictions = pred_probs.size(1)
            if n_predictions > 1:
                # To generalize confidence intervals for multiple classes, we fit a marginal distribution of the joint
                # Dirichlet distribution with parameters (alpha_0, alpha_1, ... alpha_C) for each class.
                # The marginal distribution for class j is a Beta distribution with parameters (alpha_j, sum_i,i!=j alpha_i).
                # We then take the class with the highest confidence quantile as the predicted class.
                alpha, beta = fit_beta_distribution(pred_probs, dim=-2)
                # Convert to numpy for subsequent scipy and sklearn functions
                alpha, beta = alpha.numpy(), beta.numpy()

                # Show warning if fitted Beta distribution is bimodal, i.e. alpha<1 and beta<1
                is_bimodal = (alpha < 1.) & (beta < 1.)
                if is_bimodal.any():
                    print("Warning: Fitted Beta distribution is bimodal.")

                if confidence_level is not None:
                    # Uncertainty-based decision rule
                    # Run left tail statistical test
                    quantile = sp.stats.beta.ppf(1. - confidence_level, alpha, beta)

                    if calibration_model is not None:
                        # Calibrate quantile based on calibration model
                        # Mirror quantiles of right-tailed distribution to left-tailed distribution for calibration model
                        quantile[alpha < beta] = 1. - quantile[alpha < beta]
                        B, C = quantile.shape  # B is batch size and C is number of classes
                        quantile = quantile.flatten()  # Note: calibration model only accepts 1D array
                        quantile = calibration_model.predict(quantile)
                        quantile = quantile.reshape(B, C)
                        # Map recalibrated quantile back to right-tailed distribution
                        quantile[alpha < beta] = 1. - quantile[alpha < beta]

                    # Convert back to PyTorch tensor and move to device
                    quantile = torch.from_numpy(quantile).to(device)
                    predicted = quantile.argmax(dim=-1)
                else:
                    predicted = mean_prob.argmax(dim=-1)

                posterior_params.append((alpha, beta))
            else:
                predicted = mean_prob.argmax(dim=-1)

            # Compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            class_probs.append(mean_prob.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    avg_loss /= len(val_loader)
    acc = correct / total

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    class_probs = np.concatenate(class_probs)
    n_classes = class_probs.shape[-1]
    posterior_params = tuple(map(np.concatenate, zip(*posterior_params)))  # Change from list of tuples into tuple of numpy arrays.

    print('[%d] {}_loss: %.3f'.format(tag) % (epoch, avg_loss))
    print('[%d] {}_acc: %.1f %%'.format(tag) % (epoch, 100 * acc))

    # Write to tensorboard
    writer.add_scalar('Epoch/{}'.format(tag), epoch, step_num)
    writer.add_scalar('Loss/{}'.format(tag), avg_loss, step_num)
    writer.add_scalar('Acc/{}'.format(tag), acc, step_num)
    writer.add_scalar('AUC/{}'.format(tag), sklearn.metrics.roc_auc_score(y_true, class_probs, labels=list(range(n_classes)), multi_class='ovo'), step_num)
    writer.add_scalar('F1/{}'.format(tag), sklearn.metrics.f1_score(y_true, y_pred, average='macro'), step_num)
    writer.add_scalar('ECE/{}'.format(tag), compute_expected_calibration_error(class_probs, y_true), step_num)
    writer.add_figure('Prediction reliability/{}'.format(tag), plot_pred_reliability(class_probs, y_true), step_num)

    if posterior_params:
        writer.add_scalar('Wasserstein dist/{}'.format(tag), compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true), step_num)
        writer.add_figure('Uncertainty reliability/{}'.format(tag), plot_uncertainty_reliability(class_probs, posterior_params, y_true, calibration_model=None), step_num)
        if calibration_model is not None:
            writer.add_scalar('Wasserstein dist (calibrated)/{}'.format(tag),
                              compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true,
                                                                 calibration_model=calibration_model), step_num)
            writer.add_figure('Uncertainty reliability (calibrated)/{}'.format(tag),
                                plot_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                                calibration_model=calibration_model), step_num)

        exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true)
        if exp_cdf.size > 0 and obs_cdf.size > 0:
            eval_results['uncertainty_calibration_data'] = (exp_cdf, obs_cdf)
        else:
            print("Warning: No calibration data. Could be caused if there are not enough samples per bin.")

    eval_results['y_pred'] = y_pred
    eval_results['class_probs'] = class_probs
    eval_results['y_true'] = y_true

    writer.flush()
    model.train()

    return eval_results


def train(model, train_loader, run_name, n_epochs=10, lr=0.0001, lr_hl=5, swag=True, swag_start=0.8, swag_lr=0.0001, swag_momentum=0.9, swag_branchout_layers=None,
          bootstrap=False, fold=None, confidence_level=None,
          val_loader=None, test_loader=None, eval_interval=5, log_dir=None, save=False, model_save_dir=None, device='cpu'):
    if bootstrap:
        assert isinstance(train_loader, list) and len(train_loader) > 0, \
            "Must pass in list of bootstrapped train loaders when applying bootstrap."
    else:
        if not isinstance(train_loader, list):
            train_loader = [train_loader]

    n_datasets = len(train_loader)
    steps_per_epoch = len(train_loader[0])

    if isinstance(swag_start, float) and swag_start < 1:
        swag_start_epoch = int(n_epochs*swag_start)
    else:
        swag_start_epoch = swag_start

    results = {}
    # push model to set device (CPU or GPU)
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    if swag:
        swag_optimizer = torch.optim.SGD(model.parameters(), lr=swag_lr, momentum=swag_momentum, weight_decay=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_hl, gamma=0.5)

    if fold is not None:
        log_dir = os.path.join(log_dir, "fold{0:d}".format(fold))

    writer = SummaryWriter(log_dir)

    iterators = [iter(x) for x in train_loader]

    global_step_num = 0

    for epoch in range(1, n_epochs+1):  # loop over the datasets multiple times
        running_loss = 0.0

        for i in range(steps_per_epoch):
            k = random.randint(0, n_datasets - 1)

            try:
                # get the next item
                inputs, labels = next(iterators[k])
            except StopIteration:
                # restart if we reached the end of iterator
                iterators[k] = iter(train_loader[k])
                inputs, labels = next(iterators[k])

            # push tensors to set device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            # optimizer.zero_grad() # Note: this is slower
            for param in model.parameters():
                param.grad = None

            outputs = model(inputs)
            # outputs has shape (B, P, C) where B is batch size,
            # P is number of predictions and C is number of classes

            if bootstrap:
                # NOTE: Forward and back prop for bootstrap ensemble can be optimized.
                #       Only need to compute it for the kth model.
                loss = criterion(outputs[:, k, :], labels)
            else:
                n_predictions = outputs.size(1)
                for j in range(n_predictions):
                    if j == 0:
                        loss = criterion(outputs[:, j, :], labels)
                    else:
                        loss += criterion(outputs[:, j, :], labels)

                loss /= n_predictions

            loss.backward()

            if swag and epoch >= swag_start_epoch:
                swag_optimizer.step()
                # DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    if bootstrap and isinstance(model.module, NetEnsemble):
                        model.module.update_swag(k)
                    else:
                        model.module.update_swag()
                else:
                    if bootstrap and isinstance(model, NetEnsemble):
                        model.update_swag(k) # Only update SWAG params for kth model
                    else:
                        model.update_swag() # Update SWAG params for all models
            else:
                optimizer.step()

            global_step_num += 1

            # print statistics
            running_loss += loss.item()

            if i % 10 == 0:  # print every 10 mini-batches
                if i > 0:
                    print('[%d, %3d] train_loss: %.5f' % (epoch, i, running_loss / 10))
                    running_loss = 0.0

                writer.add_scalar('Loss/train', loss.float(), global_step_num)
                if swag and epoch >= swag_start_epoch:
                    writer.add_scalar('Learning rate/swag/train', swag_optimizer.param_groups[0]['lr'], global_step_num)
                else:
                    writer.add_scalar('Learning rate/main/train', optimizer.param_groups[0]['lr'], global_step_num)

        if epoch % eval_interval == 0:
            if swag and swag_branchout_layers is not None and epoch >= swag_start_epoch:
                for layer_name in swag_branchout_layers:
                    print("Sampling SWAG models branching out at %s..." % layer_name)
                    # DataParallel
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.sample_mask = create_sample_mask(model.module.base_model, layer_name)
                        model.module.sample(device)
                    else:
                        model.sample_mask = create_sample_mask(model.base_model, layer_name)
                        model.sample(device)
                    swag_writer = SummaryWriter(os.path.join(log_dir, 'branchout_{}'.format(layer_name)))

                    if val_loader is not None:
                        tag = 'val'
                        # tag = 'val/branchout_{}'.format(layer_name)
                        print("Evaluating model on validation data...")
                        results[tag] = evaluate(model, val_loader, swag_writer, global_step_num, epoch,
                                confidence_level=confidence_level,
                                tag=tag,
                                device=device)

                        if 'uncertainty_calibration_data' in results[tag]:
                            print("Fitting calibration model...")
                            calibration_model = fit_calibration_model(results[tag]['uncertainty_calibration_data'])
                        else:
                            calibration_model = None

                        # if 'calibration_model' not in results:
                        #     results['calibration_model'] = {}
                        results['calibration_model/branchout_{}'.format(layer_name)] = calibration_model

                    if test_loader is not None:
                        tag = 'test'
                        print("Evaluating model on test data...")
                        results[tag] = evaluate(model, test_loader, swag_writer, global_step_num, epoch,
                                                   confidence_level=confidence_level,
                                                   calibration_model=calibration_model,
                                                   tag=tag,
                                                   device=device)
                    swag_writer.close()
            else:
                if swag and epoch >= swag_start_epoch:
                    print("Sampling SWAG models...")
                    # DataParallel
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.sample(device)
                    else:
                        model.sample(device)

                if val_loader is not None:
                    tag = 'val'
                    print("Evaluating model on validation data...")
                    results[tag] = evaluate(model, val_loader, writer, global_step_num, epoch,
                                            confidence_level=confidence_level,
                                            tag=tag,
                                            device=device)

                    if 'uncertainty_calibration_data' in results[tag]:
                        print("Fitting calibration model...")
                        calibration_model = fit_calibration_model(results[tag]['uncertainty_calibration_data'])
                    else:
                        calibration_model = None

                    results['calibration_model'] = calibration_model

                if test_loader is not None:
                    tag = 'test'
                    print("Evaluating model on test data...")
                    results[tag] = evaluate(model, test_loader, writer, global_step_num, epoch,
                                            confidence_level=confidence_level,
                                            calibration_model=calibration_model,
                                            tag=tag,
                                            device=device)

        scheduler.step()

    writer.close()

    if save:
        # Save models
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        print("Saving calibration model...")
        model_file_name = run_name + "_fold{0:d}_calibration.pkl".format(fold)
        with open(os.path.join(model_save_dir, model_file_name), 'wb') as file:
            pickle.dump(calibration_model, file)

        print("Saving model...")
        if fold is not None:
            model_file_name = run_name + "_fold{}.pth".format(fold)
        else:
            model_file_name = run_name + '.pth'
        torch.save(model, os.path.join(model_save_dir, model_file_name))

    print("Done.")

    return results
