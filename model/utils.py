import os
import torch


def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state": state,  # state of training loop (was 'step')
        },
        path,
    )


def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict

        model_state_dict_fixed = OrderedDict()
        prefix = "module."
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith(prefix):
                k = k[len(prefix) :]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "state" in checkpoint:
        state = checkpoint["state"]
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {"step": checkpoint["step"]}
    return state


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
