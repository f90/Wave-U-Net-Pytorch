import os

import torch
import numpy as np
import librosa

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if mono:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr

def resample(audio, orig_sr, new_sr, mode="numpy"):
    if orig_sr == new_sr:
        return audio

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    out = librosa.resample(audio, orig_sr, new_sr)

    if mode == "pytorch":
        out = torch.tensor(out)
    return out

class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def save_model(model, optimizer, step, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return step

def load_latest_model_from(model, optimizer, location):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)
    return load_model(model, optimizer, newest_file)