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

from networks import NetEnsemble, BranchingNetwork
from calibration import compute_expected_calibration_error, compute_uncertainty_reliability, compute_posterior_wasserstein_dist, fit_calibration_model
from plots import plot_pred_reliability, plot_uncertainty_reliability
from swag import MultiSWAG
from logger import logger


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


def create_sample_mask(model, branchout_layer_name, branch_num=None):
    state_dict = model.state_dict()
    param_masks = []
    swag_branchout = False
    branchout_in_shared_network = False
    curr_branch = -1

    for param_name, _ in model.named_parameters():
        if param_name.split('.')[0] == 'branches' and int(param_name.split('.')[1]) > curr_branch:
            # Reset branchout flag because we are starting at a new branch.
            swag_branchout = False
            curr_branch += 1

        if branchout_in_shared_network and (branch_num is None or branch_num == curr_branch):
            swag_branchout = True
        elif branchout_layer_name in param_name:
            if curr_branch < 0:
                branchout_in_shared_network = True
                swag_branchout = True
            else:
                if branch_num is None or branch_num == curr_branch:
                    swag_branchout = True

        n = state_dict[param_name].numel()
        param_masks.append(np.zeros((n,)) + float(swag_branchout))

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

    if (mean*(1-mean) < var).any().item():
        logger.warning("Beta distribution parameters negative using unbiased variance. Using biased variance.")
        var_biased = pred_probs.var(dim=dim, unbiased=False)
        var = torch.where(mean*(1-mean)<var, var_biased, var)

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
            if n_predictions > 2:
                # To generalize confidence intervals for multiple classes, we fit a marginal distribution of the joint
                # Dirichlet distribution with parameters (alpha_0, alpha_1, ... alpha_C) for each class.
                # The marginal distribution for class j is a Beta distribution with parameters (alpha_j, sum_i,i!=j alpha_i).
                # We then take the class with the highest confidence quantile as the predicted class.
                alpha, beta = fit_beta_distribution(pred_probs, dim=-2)
                # Convert to numpy for subsequent scipy and sklearn functions
                alpha, beta = alpha.cpu().numpy(), beta.cpu().numpy()

                # Show warning if fitted Beta distribution is bimodal, i.e. alpha<1 and beta<1
                is_bimodal = (alpha < 1.) & (beta < 1.)
                if is_bimodal.any():
                    logger.warning("Fitted Beta distribution is bimodal.")

                if confidence_level is not None:
                    # Uncertainty-based decision rule
                    # Run left tail statistical test
                    quantile = sp.stats.beta.ppf(1. - confidence_level, alpha, beta)

                    if calibration_model is not None:
                        # Calibrate quantile based on calibration model
                        # Mirror quantiles of right-tailed distribution to left-tailed distribution for calibration model
                        quantile[alpha < beta] = 1. - quantile[alpha < beta]
                        B, C = quantile.shape  # B is batch size and C is number of classes

                        if 'unimodal' in calibration_model:
                            quantile_unimodal = quantile[~is_bimodal]
                            if quantile_unimodal.size > 0:
                                quantile_unimodal = calibration_model['unimodal'].predict(quantile_unimodal)
                                quantile[~is_bimodal] = quantile_unimodal

                        if 'bimodal' in calibration_model:
                            quantile_bimodal = quantile[is_bimodal]
                            if quantile_bimodal.size > 0:
                                quantile_bimodal = calibration_model['bimodal'].predict(quantile_bimodal)
                                quantile[is_bimodal] = quantile_bimodal

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

    logger.info('[%d] {}_loss: %.3f'.format(tag) % (epoch, avg_loss))
    logger.info('[%d] {}_acc: %.1f %%'.format(tag) % (epoch, 100 * acc))

    # Write to tensorboard
    writer.add_scalar('Epoch/{}'.format(tag), epoch, step_num)
    writer.add_scalar('Loss/{}'.format(tag), avg_loss, step_num)
    writer.add_scalar('Acc/{}'.format(tag), acc, step_num)
    writer.add_scalar('AUC/{}'.format(tag), sklearn.metrics.roc_auc_score(y_true, class_probs, labels=list(range(n_classes)), multi_class='ovo'), step_num)
    writer.add_scalar('F1/{}'.format(tag), sklearn.metrics.f1_score(y_true, y_pred, average='macro'), step_num)
    writer.add_scalar('ECE/{}'.format(tag), compute_expected_calibration_error(class_probs, y_true), step_num)
    writer.add_figure('Prediction reliability/{}'.format(tag), plot_pred_reliability(class_probs, y_true), step_num)

    if posterior_params:
        writer.add_scalar('Wasserstein dist (unimodal)/{}'.format(tag), compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true, dist_shape='unimodal'), step_num)
        writer.add_scalar('Wasserstein dist (bimodal)/{}'.format(tag),
                          compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true,
                                                             dist_shape='bimodal'), step_num)
        writer.add_figure('Uncertainty reliability (unimodal)/{}'.format(tag), plot_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape='unimodal', calibration_model=None), step_num)
        writer.add_figure('Uncertainty reliability (bimodal)/{}'.format(tag),
                          plot_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape='bimodal', calibration_model=None),
                          step_num)

        # fig_acc, fig_auc, fig_f1_score = plot_confidence_level_performance(class_probs, posterior_params, y_true, calibration_model=None)
        # writer.add_figure('Acc vs. confidence level/{}'.format(tag), fig_acc, step_num)
        # writer.add_figure('AUC vs. confidence level/{}'.format(tag), fig_auc, step_num)
        # writer.add_figure('F1-score vs. confidence level/{}'.format(tag), fig_f1_score, step_num)

        if calibration_model is not None:
            for name in calibration_model.keys():
                writer.add_scalar('Wasserstein dist ({}, calibrated)/{}'.format(name, tag),
                              compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true, dist_shape=name,
                                                                 calibration_model=calibration_model[name]), step_num)
                writer.add_figure('Uncertainty reliability ({}, calibrated)/{}'.format(name, tag),
                                  plot_uncertainty_reliability(class_probs, posterior_params, y_true, dist_shape=name,
                                                               calibration_model=calibration_model[name]),
                                  step_num)

            # fig_acc, fig_auc, fig_f1_score = plot_confidence_level_performance(class_probs, posterior_params, y_true,
            #                                                                    calibration_model=calibration_model)
            # writer.add_figure('Acc vs. confidence level (calibrated)/{}'.format(tag), fig_acc, step_num)
            # writer.add_figure('AUC vs. confidence level (calibrated)/{}'.format(tag), fig_auc, step_num)
            # writer.add_figure('F1-score vs. confidence level (calibrated)/{}'.format(tag), fig_f1_score, step_num)


        if 'uncertainty_calibration_data' not in eval_results:
            eval_results['uncertainty_calibration_data'] = {}

        exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                               dist_shape='unimodal')
        if exp_cdf.size > 0 and obs_cdf.size > 0:
            eval_results['uncertainty_calibration_data']['unimodal'] = (exp_cdf, obs_cdf)
        else:
            logger.warning("No calibration data. Could be caused if there are not enough samples per bin.")

        exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                           dist_shape='bimodal')
        if exp_cdf.size > 0 and obs_cdf.size > 0:
            eval_results['uncertainty_calibration_data']['bimodal'] = (exp_cdf, obs_cdf)
        else:
            logger.warning("No calibration data. Could be caused if there are not enough samples per bin.")

    eval_results['y_pred'] = y_pred
    eval_results['class_probs'] = class_probs
    eval_results['y_true'] = y_true

    writer.flush()
    model.train()

    return eval_results


def train(model, train_loader, run_name, n_epochs=10, lr=0.0001, lr_hl=5,
          swag=True, swag_samples=10, swag_start=0.8, swag_lr=0.0001, swag_momentum=0.9,
          swag_interval=10, swag_branchout_layers=None, swag_bn_data_ratio=1.0,
          bootstrap=False, fold=None, confidence_level=None,
          val_loader=None, test_loader=None, eval_interval=5, ckpt_interval=None,
          checkpoint=None, log_dir=None, save=False, model_save_dir=None, device='cpu'):
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

    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model

    if swag:
        if not isinstance(module, NetEnsemble) and isinstance(module.base_model, BranchingNetwork):
            swag_optimizer = torch.optim.SGD([
                {'params': module.trunk.parameters()},
                {'params': module.branches.parameters(), 'lr': swag_lr*module.n_branches}
            ], lr=swag_lr, momentum=swag_momentum, weight_decay=0.0)
        else:
            swag_optimizer = torch.optim.SGD(module.parameters(), lr=swag_lr, momentum=swag_momentum, weight_decay=0.0)

    if isinstance(module, BranchingNetwork) or (swag and isinstance(module.base_model, BranchingNetwork)):
        optimizer = torch.optim.Adam([
            {'params': module.trunk.parameters()},
            {'params': module.branches.parameters(), 'lr': lr*module.n_branches}
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_hl, gamma=0.5)

    if checkpoint is not None:
        if os.path.exists(checkpoint):
            logger.info(f"Loading checkpoint {checkpoint}...")
            ckpt = torch.load(checkpoint)
            init_epoch = ckpt['epoch']
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            random.setstate(ckpt['random_state']['random'])
            np.random.set_state(ckpt['random_state']['numpy'])
            torch.random.set_rng_state(ckpt['random_state']['torch'])
            if swag:
                swag_optimizer.load_state_dict(ckpt['swag_optimizer_state'])
        else:
            logger.warning("Provided checkpoint path does not exist. Starting training from scratch.")
            init_epoch = 0
    else:
        init_epoch = 0

    if fold is not None:
        log_dir = os.path.join(log_dir, "fold{0:d}".format(fold))

    writer = SummaryWriter(log_dir)

    iterators = [iter(x) for x in train_loader]

    global_step_num = 0

    for epoch in range(init_epoch+1, n_epochs+1):  # loop over the datasets multiple times
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

            if bootstrap:
                outputs = model(inputs, branch_num=k)
                # outputs has shape (B, P, C) where B is batch size,
                # P is number of predictions and C is number of classes
                loss = criterion(outputs[:, 0, :], labels)
            else:
                outputs = model(inputs)
                # outputs has shape (B, P, C) where B is batch size,
                # P is number of predictions and C is number of classes

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
                if i % swag_interval == 0:

                    if bootstrap:
                        if isinstance(module, NetEnsemble):
                            module.update_swag(k)  # Only update SWAG params for kth model
                        elif isinstance(module, MultiSWAG):
                            module.update_swag()  # Update SWAG params for trunk
                            module.update_swag(k)  # Update SWAG params for kth branch
                    else:
                        if isinstance(module, MultiSWAG):
                            module.update_swag()  # Update SWAG params for trunk
                            for j in range(module.n_branches):
                                module.update_swag(j)  # Update SWAG params for all branches
                        else:
                            module.update_swag()  # Update SWAG params for all models
            else:
                optimizer.step()

            global_step_num += 1

            # log statistics
            running_loss += loss.item()
            if i % 10 == 0:  # log every 10 mini-batches
                if i > 0:
                    logger.info('[%d, %3d] train_loss: %.5f' % (epoch, i, running_loss / 10))
                    writer.add_scalar('Loss/train', running_loss / 10, global_step_num)
                    running_loss = 0.0

                if swag and epoch >= swag_start_epoch:
                    writer.add_scalar('Learning rate/swag/train', swag_optimizer.param_groups[0]['lr'], global_step_num)
                else:
                    writer.add_scalar('Learning rate/main/train', optimizer.param_groups[0]['lr'], global_step_num)

        if ckpt_interval is not None and epoch % ckpt_interval == 0:
            logger.info("Saving checkpoint...")
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'random_state': {
                    'random': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.random.get_rng_state()
                }
            }
            if swag:
                ckpt['swag_optimizer_state'] = swag_optimizer.state_dict()

            if fold is not None:
                ckpt_file_name = f"{run_name}_fold{fold}_epoch{epoch}_ckpt.pth"
            else:
                ckpt_file_name = f"{run_name}_epoch{epoch}_ckpt.pth"

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            torch.save(ckpt, os.path.join(model_save_dir, ckpt_file_name))
            logger.info("Saving completed.")
            
        if epoch % eval_interval == 0:
            if swag and swag_branchout_layers is not None and epoch >= swag_start_epoch:
                for layer_name in swag_branchout_layers:
                    logger.info("Sampling SWAG models branching out at %s..." % layer_name)

                    # For SWAG model (depricated)
                    # model.sample_masks = [create_sample_mask(model.base_model, layer_name, branch_num=i) for i in range(model.base_model.n_heads)]
                    # model.sample(device)
                    # For Multi-SWAG model
                    module.sample(swag_samples, layer_name, swag_bn_data_ratio, device)

                    swag_writer = SummaryWriter(os.path.join(log_dir, 'branchout_{}'.format(layer_name)))

                    if val_loader is not None:
                        tag = 'val'
                        # tag = 'val/branchout_{}'.format(layer_name)
                        logger.info("Evaluating model on validation data...")
                        results[tag] = evaluate(model, val_loader, swag_writer, global_step_num, epoch,
                                confidence_level=confidence_level,
                                tag=tag,
                                device=device)

                        if 'uncertainty_calibration_data' in results[tag]:
                            logger.info("Fitting calibration model...")
                            calibration_model = {}
                            if 'unimodal' in results[tag]['uncertainty_calibration_data']:
                                calibration_model['unimodal'] = fit_calibration_model(results[tag]['uncertainty_calibration_data']['unimodal'])
                            if 'bimodal' in results[tag]['uncertainty_calibration_data']:
                                calibration_model['bimodal'] = fit_calibration_model(
                                    results[tag]['uncertainty_calibration_data']['bimodal'])
                        else:
                            calibration_model = None

                        results['calibration_model/branchout_{}'.format(layer_name)] = calibration_model

                    if test_loader is not None:
                        tag = 'test'
                        logger.info("Evaluating model on test data...")
                        results[tag] = evaluate(model, test_loader, swag_writer, global_step_num, epoch,
                                                   confidence_level=confidence_level,
                                                   calibration_model=calibration_model,
                                                   tag=tag,
                                                   device=device)
                    swag_writer.close()
            else:
                if swag and epoch >= swag_start_epoch:
                    logger.info("Sampling SWAG models...")
                    module.sample(swag_samples, None, swag_bn_data_ratio, device)

                if val_loader is not None:
                    tag = 'val'
                    logger.info("Evaluating model on validation data...")
                    results[tag] = evaluate(model, val_loader, writer, global_step_num, epoch,
                                            confidence_level=confidence_level,
                                            tag=tag,
                                            device=device)

                    if 'uncertainty_calibration_data' in results[tag]:
                        logger.info("Fitting calibration model...")
                        calibration_model = {}
                        if 'unimodal' in results[tag]['uncertainty_calibration_data']:
                            calibration_model['unimodal'] = fit_calibration_model(
                                results[tag]['uncertainty_calibration_data']['unimodal'])
                        if 'bimodal' in results[tag]['uncertainty_calibration_data']:
                            calibration_model['bimodal'] = fit_calibration_model(
                                results[tag]['uncertainty_calibration_data']['bimodal'])
                    else:
                        calibration_model = None

                    results['calibration_model'] = calibration_model

                if test_loader is not None:
                    tag = 'test'
                    logger.info("Evaluating model on test data...")
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

        logger.info("Saving calibration model...")
        model_file_name = run_name + "_fold{0:d}_calibration.pkl".format(fold)
        with open(os.path.join(model_save_dir, model_file_name), 'wb') as file:
            pickle.dump(calibration_model, file)

        logger.info("Saving model...")
        if fold is not None:
            model_file_name = run_name + "_fold{}.pth".format(fold)
        else:
            model_file_name = run_name + '.pth'
        torch.save(model, os.path.join(model_save_dir, model_file_name))

    logger.info("Done.")

    return results
