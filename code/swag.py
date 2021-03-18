import copy
from collections import deque

import numpy as np
import torch


class SWAG(torch.nn.Module):
    def __init__(self, base_model, n_rank=10, n_samples=10, bn_update_loader=None, sample_mask=None):
        super(SWAG, self).__init__()
        self.base_model = base_model
        self.n_rank = n_rank
        self.n_samples = n_samples
        self.n_iter = 0
        self.param_names = [name for name, _ in self.base_model.named_parameters()]
        self.param_mean = None
        self.param_second_mom = None
        self.param_buffer = deque()
        self.bn_update_loader = bn_update_loader
        self.sample_mask = sample_mask
        self.sampled_models = torch.nn.ModuleList()
        self.min_var = 1e-30

    def _get_params(self):
        params = []
        for p in self.base_model.parameters():
            params.append(p.detach().cpu().numpy().flatten())
        params = np.concatenate(params, axis=0)

        return params

    def _sample_params(self):
        z1 = np.random.normal(size=(len(self.param_mean),))
        z2 = np.random.normal(size=(len(self.param_buffer),))
        param_cov_diag = self.param_second_mom - self.param_mean ** 2
        param_cov_diag = np.clip(param_cov_diag, self.min_var, None)
        # NOTE: Different from Maddox et. al. (https://arxiv.org/abs/1902.02476)
        # Subtracting current mean rather than running mean.
        deviations_matrix = np.array(self.param_buffer) - self.param_mean
        sampled_deviations = (1. / np.sqrt(2) * np.sqrt(param_cov_diag) * z1 +
                              1. / np.sqrt(2 * (self.n_rank - 1)) * np.dot(deviations_matrix.T, z2))

        if self.sample_mask is not None:
            sampled_deviations *= self.sample_mask

        params = self.param_mean + sampled_deviations

        return params

    def _set_params(self, model, params, device):
        state_dict = copy.deepcopy(model.state_dict())
        idx = 0
        for p in self.param_names:
            n = state_dict[p].numel()
            state_dict[p].copy_(torch.from_numpy(params[idx:idx + n]).reshape(state_dict[p].shape).to(device))
            idx += n

        model.load_state_dict(state_dict, strict=True)

    def update_bn(self, model, device=None):
        assert self.bn_update_loader is not None, "Must provide bn_update_loader to call update BN layers."

        def _reset_bn(module):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)

        def _get_momenta(module, momenta):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                momenta[module] = module.momentum

        def _set_momenta(module, momenta):
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for inputs, labels in self.bn_update_loader:
            b = inputs.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                inputs = inputs.to(device)

            model(inputs)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)

    def update_swag(self):
        if len(self.param_buffer) >= self.n_rank:
            self.param_buffer.popleft()
        self.param_buffer.append(self._get_params())

        if self.n_iter == 0:
            self.param_mean = self.param_buffer[-1]
            self.param_second_mom = self.param_buffer[-1] ** 2
        else:
            self.param_mean = self.n_iter / (self.n_iter + 1) * self.param_mean + self.param_buffer[-1] / (
                        self.n_iter + 1)
            self.param_second_mom = self.n_iter / (self.n_iter + 1) * self.param_second_mom + self.param_buffer[
                -1] ** 2 / (self.n_iter + 1)

        self.n_iter += 1

        # neg_var_count = (self.param_second_mom - self.param_mean ** 2 < 0).sum()
        # if neg_var_count > 0:
        #     print("Warning: %d negative variances." % neg_var_count)

    def sample(self, device):
        # Sample from Gaussian posterior
        assert self.param_mean is not None and self.param_second_mom is not None, "First and second moment required. Call update_swag first."

        self.sampled_models = torch.nn.ModuleList()

        for i in range(self.n_samples):
            params = self._sample_params()
            model = copy.deepcopy(self.base_model)
            self._set_params(model, params, device)

            # DataParallel
            if torch.cuda.device_count() > 1:
                print(f'Sample {i} uses {torch.cuda.device_count()} GPUs!')
                model = torch.nn.DataParallel(model)

            self.update_bn(model, device)

            self.sampled_models.append(model)

    def forward(self, x):
        if self.training:
            return self.base_model(x)
        else:
            if self.sampled_models:
                sampled_outputs = [None] * self.n_samples

                with torch.no_grad():
                    for i in range(self.n_samples):
                        # Sample from Gaussian posterior
                        model = self.sampled_models[i]
                        sampled_outputs[i] = model(x)
                # Assuming outputs have shape (B, P, C) where B is batch size, P is number of predictions
                # and C is number of classes
                sampled_outputs = torch.cat(sampled_outputs, dim=1)

                return sampled_outputs
            else:
                return self.base_model(x)
