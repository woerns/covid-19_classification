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

        # Stack K model outputs (B, ...) to (B, K, ...). B is batch size.
        ensemble_outputs = torch.stack(ensemble_outputs, dim=1)

        return ensemble_outputs


def create_branching_network(model_name, n_heads=10, add_mask=False):

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
        model.fc = torch.nn.Linear(num_ftrs, n_heads)
    elif 'densenet' in model_name:
        # DenseNet
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
        # Replace last layer
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, n_heads)
    else:
        raise ValueError("Unknown model name %s." % model_name)

    return model


def create_model(model_name, model_type, n_heads, swag=False, swag_rank=10, swag_samples=10, bn_update_loader=None):
    if swag:
        assert bn_update_loader is not None, "Must provide training data loader for BN update when applying SWAG."

    if model_type == 'branching':
        model = create_branching_network(model_name, n_heads=n_heads)
        if swag:
            model = SWAG(model, n_rank=swag_rank, n_samples=swag_samples, bn_update_loader=bn_update_loader)
    elif model_type == 'ensemble':
        models = [create_branching_network(model_name, n_heads=1) for _ in range(n_heads)]
        if swag:
            models = [SWAG(x, n_rank=swag_rank, n_samples=swag_samples, bn_update_loader=bn_update_loader) for x in models]
        model = NetEnsemble(models)
    else:
        raise ValueError("Unknown model type %s." % model_type)

    return model
