import museval
from tqdm import tqdm

import utils

import numpy as np
import torch

from utils import compute_loss

def predict(audio, model):
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    outputs = {key: np.zeros(audio.shape, np.float32) for key in model.instruments}

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):

            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]] # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            for key, curr_targets in model(curr_input):
                outputs[key][:,target_start_pos:target_start_pos+model.shapes["output_frames"]] = curr_targets.squeeze(0).cpu().numpy()

    # Crop to expected length (since we padded to handle the frame shift)
    outputs = {key : outputs[key][:,:expected_outputs] for key in outputs.keys()}

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs

def predict_song(args, audio_path, model):
    model.eval()

    # Load mixture in original sampling rate
    mix_audio, mix_sr = utils.load(audio_path, sr=None, mono=False)
    mix_channels = mix_audio.shape[0]
    mix_len = mix_audio.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        mix_audio = np.mean(mix_audio, axis=0, keepdims=True)
    else:
        if mix_channels == 1: # Duplicate channels if input is mono but model is stereo
            mix_audio = np.tile(mix_audio, [args.channels, 1])
        else:
            assert(mix_channels == args.channels)

    # resample to model sampling rate
    mix_audio = utils.resample(mix_audio, mix_sr, args.sr)

    sources = predict(mix_audio, model)

    # Resample back to mixture sampling rate in case we had model on different sampling rate
    sources = {key : utils.resample(sources[key], args.sr, mix_sr) for key in sources.keys()}

    # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now
    for key in sources.keys():
        diff = sources[key].shape[1] - mix_len
        if diff > 0:
            print("WARNING: Cropping " + str(diff) + " samples")
            sources[key] = sources[key][:, :-diff]
        elif diff < 0:
            print("WARNING: Padding output by " + str(diff) + " samples")
            sources[key] = np.pad(sources[key], [(0,0), (0, -diff)], "constant", 0.0)

        # Adapt channels
        if mix_channels > args.channels:
            assert(args.channels == 1)
            # Duplicate mono predictions
            sources[key] = np.tile(sources[key], [mix_channels, 1])
        elif mix_channels < args.channels:
            assert(mix_channels == 1)
            # Reduce model output to mono
            sources[key] = np.mean(sources[key], axis=0, keepdims=True)

        sources[key] = np.asfortranarray(sources[key]) # So librosa does not complain if we want to save it

    return sources

def evaluate(args, dataset, model, instruments):
    perfs = list()
    model.eval()
    with torch.no_grad():
        for example in dataset:
            print("Evaluating " + example["mix"])

            # Load source references in their original sr and channel number
            target_sources = np.stack([utils.load(example[instrument], sr=None, mono=False)[0].T for instrument in instruments])

            # Predict using mixture
            pred_sources = predict_song(args, example["mix"], model)
            pred_sources = np.stack([pred_sources[key].T for key in instruments])

            # Evaluate
            SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)
            song = {}
            for idx, name in enumerate(instruments):
                song[name] = {"SDR" : SDR[idx], "ISR" : ISR[idx], "SIR" : SIR[idx], "SAR" : SAR[idx]}
            perfs.append(song)

    return perfs


def validate(args, model, criterion, test_data):
    # PREPARE DATA
    dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers)

    # VALIDATE
    model.eval()
    total_loss = 0.
    with tqdm(total=len(test_data) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, targets) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                for k in list(targets.keys()):
                    targets[k] = targets[k].cuda()

            _, avg_loss = compute_loss(model, x, targets, criterion)

            total_loss += (1. / float(example_num + 1)) * (avg_loss - total_loss)

            pbar.set_description("Current loss: " + str(total_loss))
            pbar.update(1)

    return total_loss