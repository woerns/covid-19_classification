import torch

from swag import SWAG


class NetEnsemble(torch.nn.Module):
    def __init__(self, models):
        super(NetEnsemble, self).__init__()
        self.ensemble = torch.nn.ModuleList(models)

    def update_swag(self, k=None):
        if k is None:
            for k in range(len(self.ensemble)):
                self.ensemble[k].update_swag()
        else:
            self.ensemble[k].update_swag()

    def sample(self, k=None):
        if k is None:
            for k in range(len(self.ensemble)):
                self.ensemble[k].sample()
        else:
            self.ensemble[k].sample()

    def forward(self, x):
        ensemble_outputs = []
        for model in self.ensemble:
            ensemble_outputs.append(model(x))

        # Concatenate K model outputs (B, P, ...) to (B, K*P, ...). B is batch size.
        ensemble_outputs = torch.cat(ensemble_outputs, dim=1)

        return ensemble_outputs


class BranchingNetwork(torch.nn.Module):
    def __init__(self, model, n_classes, n_heads):
        super(BranchingNetwork, self).__init__()
        self.model = model
        self._n_classes = n_classes
        self._n_heads = n_heads
        self._init_branchout()

    @property
    def n_heads(self):
        return self._n_heads

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_outputs(self):
        return self.n_heads * self.n_classes

    def _init_branchout(self):
        # Branchout last layer
        *_, (name, last_layer) = self.model.named_modules()
        self.model._modules[name] = torch.nn.Linear(last_layer.in_features, self.n_outputs)

    def forward(self, x):
        # Output has shape (B, P, C) where B is batch size, P is number of heads/predictions
        # and C is number of classes
        return self.model(x).view(-1, self.n_heads, self.n_classes)


def create_branching_network(model_name, n_classes, n_heads=10, add_mask=False):
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
    elif 'densenet' in model_name:
        # DenseNet
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
    else:
        raise ValueError("Unknown model name %s." % model_name)

    # Create branching network based on base model
    model = BranchingNetwork(model, n_classes, n_heads)

    return model


def get_swag_branchout_layers(model_name):
    if model_name == 'densenet169':
        branchout_layers = ['features.conv0',
                            'features.denseblock1',
                            'features.denseblock2',
                            'features.denseblock3',
                            'features.denseblock4',
                            'classifier']
    else:
        raise ValueError("Unknown model name %s." % model_name)

    return branchout_layers


def create_model(model_name, model_type, n_heads, n_classes, swag=False, swag_rank=10, swag_samples=10, bn_update_loader=None):
    if swag:
        assert bn_update_loader is not None, "Must provide training data loader for BN update when applying SWAG."

    if model_type == 'branching':
        model = create_branching_network(model_name, n_heads=n_heads, n_classes=n_classes)
        if swag:
            model = SWAG(model, n_rank=swag_rank, n_samples=swag_samples, bn_update_loader=bn_update_loader)
    elif model_type == 'ensemble':
        models = [create_branching_network(model_name, n_heads=1, n_classes=n_classes) for _ in range(n_heads)]
        if swag:
            models = [SWAG(x, n_rank=swag_rank, n_samples=swag_samples, bn_update_loader=bn_update_loader) for x in models]
        model = NetEnsemble(models)
    else:
        raise ValueError("Unknown model type %s." % model_type)

    return model
