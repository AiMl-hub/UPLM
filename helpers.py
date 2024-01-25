import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import shutil


def get_num_classes(dataset_name):
    if dataset_name == "pacs":
        return 7
    elif dataset_name == "vlcs":
        return 5
    elif dataset_name == "office_home":
        return 65
    elif dataset_name == "terra":
        return 10
    else:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))


def create_model(args):
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    num_ftrs = model.fc.in_features
    num_classes = get_num_classes(args.dataset_name)
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model, num_classes


def get_monte_carlo_predictions(
    input_s, target_s, forward_passes, model, n_classes, n_samples
):
    """Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    tanh = nn.Tanh()

    # define fc layer with a dropout
    fc_layer = nn.Sequential(nn.Dropout(0.2), model.fc)

    # change device
    fc_layer.cuda()
    input_s = input_s.cuda()

    # take out the feature extractor, neglecting the final fc layer
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    # get the feature scores for input data
    y_hat = feature_extractor(input_s).squeeze().squeeze()

    all_predictions = []

    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        with torch.no_grad():
            output = fc_layer(y_hat)

        output = output.unsqueeze(1)
        all_predictions.append(output)

    dropout_predictions = torch.cat(all_predictions, dim=1)
    dropout_predictions = dropout_predictions.cpu()
    dropout_predictions = dropout_predictions.numpy()

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=1)  # shape (n_samples, n_classes)
    var = np.var(dropout_predictions, axis=1)  # shape (n_samples, n_classes)
    var = torch.from_numpy(var)
    var = 1 - tanh(var)

    return var


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def save_checkpoint(
    state, is_best, is_best_valid, checkpoint, filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))
    if is_best_valid:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best_valid.pth.tar"))
