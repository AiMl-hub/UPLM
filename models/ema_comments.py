from copy import deepcopy
import torch


class ModelEMA(object):
    # Initialize the EMA class by creating a deep copy of the model, setting it to evaluation mode,
    # and disabling gradients. Also store the decay factor, and whether the original model has a module.
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)  # deep copy of the model
        self.ema.to(
            args.device
        )  # move the copy to the device specified in the arguments
        self.ema.eval()  # set the copy to evaluation mode
        self.decay = decay  # store the decay factor
        self.ema_has_module = hasattr(
            self.ema, "module"
        )  # check if the original model has a module
        # get the keys of the parameters and buffers in the model's state dict
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        # set all parameters in the EMA to not require gradients
        for p in self.ema.parameters():
            p.requires_grad_(False)

    # Update the EMA by mixing the new values from the model with the old values in the EMA.
    def update(self, model):
        needs_module = (
            hasattr(model, "module") and not self.ema_has_module
        )  # check if the model has a module
        with torch.no_grad():  # temporarily set requires_grad to False
            msd = model.state_dict()  # get the state dict of the model
            esd = self.ema.state_dict()  # get the state dict of the EMA
            # update the parameters in the EMA
            for k in self.param_keys:
                if needs_module:
                    j = "module." + k  # use the module prefix if necessary
                else:
                    j = k
                model_v = msd[j].detach()  # get the value from the model
                ema_v = esd[k]  # get the value from the EMA
                esd[k].copy_(
                    ema_v * self.decay + (1.0 - self.decay) * model_v
                )  # update the value in the EMA

            # update the buffers in the EMA
            for k in self.buffer_keys:
                if needs_module:
                    j = "module." + k  # use the module prefix if necessary
                else:
                    j = k
                esd[k].copy_(msd[j])  # update the value in the EMA
