import argparse
import os
import time

import torch
import pickle
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import utils
from data import get_musdb_folds, SeparationDataset
from evaluate import evaluate
from waveunet import Waveunet

def compute_loss(model, inputs, targets, criterion, compute_grad=False):
    all_outputs = {}

    if model.separate:
        avg_loss = 0.0
        num_sources = 0
        for inst, output in model(inputs):
            loss = criterion(output, targets[inst])

            if compute_grad:
                loss.backward()

            avg_loss += loss.item()
            num_sources += 1

            all_outputs[inst] = output.detach().clone()

        avg_loss /= float(num_sources)
    else:
        loss = 0
        for inst, output in model(inputs):
            loss += criterion(output, targets[inst])
            all_outputs[inst] = output.detach().clone()

        if compute_grad:
            loss.backward()

        avg_loss = loss.item() / float(len(all_outputs))

    return all_outputs, avg_loss

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

## TRAIN PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of data loader worker threads (default: 1)')
parser.add_argument('--features', type=int, default=24,
                    help='# of feature channels per layer')
parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                    help='Folder to write logs into')
parser.add_argument('--dataset_dir', type=str, default="/mnt/musdb",
                    help='Dataset path')
parser.add_argument('--hdf_dir', type=str, default="hdf",
                    help='Dataset path')
parser.add_argument('--snapshot_dir', type=str, default='snapshots/waveunet',
                    help='Folder to write checkpoints into')
parser.add_argument('--load_model', type=str, default=None,
                    help='Reload a previously trained model (whole task model)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch size")
parser.add_argument('--levels', type=int, default=12,
                    help="Number DS/US blocks")
parser.add_argument('--depth', type=int, default=1,
                    help="Number of convs per block")
parser.add_argument('--sr', type=int, default=44100,
                    help="Sampling rate")
parser.add_argument('--channels', type=int, default=2,
                    help="Number of input audio channels")
parser.add_argument('--kernel_size', type=int, default=5,
                    help="Filter width of kernels. Has to be an odd number")
parser.add_argument('--output_size', type=float, default=2.0,
                    help="Output duration")
parser.add_argument('--strides', type=int, default=2,
                    help="Strides in Waveunet")
parser.add_argument('--patience', type=int, default=20,
                    help="Patience for early stopping on validation set")
parser.add_argument('--loss', type=str, default="L1",
                    help="L1 or L2")
parser.add_argument('--residual', type=str, default="normal", help="normal/bn/gn/he/wavenet")
parser.add_argument('--res', type=str, default="fixed", help="fixed/learned")
parser.add_argument('--separate', type=int, default=0, help="Train separate model for each source (1) or only one (0)")
parser.add_argument('--feature_growth', type=str, default="add",
                    help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

args = parser.parse_args()

INSTRUMENTS = ["bass", "drums", "other", "vocals"]
NUM_INSTRUMENTS = len(INSTRUMENTS)

#torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

# MODEL
num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
               [args.features*2**i for i in range(0, args.levels)]
target_outputs = int(args.output_size * args.sr)
model = Waveunet(args.channels, num_features, args.channels, INSTRUMENTS, kernel_size=args.kernel_size,
                 target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                 residual=args.residual, res=args.res, separate=args.separate)

if args.cuda:
    model = utils.DataParallel(model)
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

writer = SummaryWriter(args.log_dir)

### DATASET
musdb = get_musdb_folds(args.dataset_dir)
train_data = SeparationDataset(musdb, "train", INSTRUMENTS, args.sr, args.channels, model.shapes, True, args.hdf_dir)
val_data = SeparationDataset(musdb, "val", INSTRUMENTS, args.sr, args.channels, model.shapes, False, args.hdf_dir)
test_data = SeparationDataset(musdb, "test", INSTRUMENTS, args.sr, args.channels, model.shapes, False, args.hdf_dir)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

##### TRAINING ####

if args.loss == "L1":
    criterion = lambda x,y : torch.mean(torch.abs(x - y))
elif args.loss == "L2":
    criterion = lambda x,y : torch.mean((x-y)**2)
else:
    raise NotImplementedError("Couldn't find this loss!")

optimizer = Adam(params=model.parameters(), lr=args.lr)

state = {"step" : 0,
         "worse_epochs" : 0,
         "epochs" : 0,
         "best_loss" : np.Inf}

# LOAD MODEL CHECKPOINT IF DESIRED
if args.load_model is not None:
    print("Continuing training full model from checkpoint " + str(args.load_model))
    state = utils.load_model(model, optimizer, os.path.join(args.snapshot_dir, args.load_model))

print('TRAINING START')
while state["worse_epochs"] < args.patience:
    print("Training one epoch from iteration " + str(state["step"]))
    avg_time = 0.
    model.train()
    with tqdm(total=len(train_data) // args.batch_size) as pbar:
        np.random.seed()
        for example_num, (x, targets) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                for k in list(targets.keys()):
                    targets[k] = targets[k].cuda()

            t = time.time()

            # Compute loss for each instrument/model
            optimizer.zero_grad()
            outputs, avg_loss = compute_loss(model, x, targets, criterion, compute_grad=True)

            optimizer.step()

            state["step"] += 1

            t = time.time() - t
            avg_time += (1. / float(example_num + 1)) * (t - avg_time)

            writer.add_scalar("train_loss", avg_loss, state["step"])

            if example_num % 50 == 0:
                input_centre = torch.mean(x[0, :, model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], 0) # Stereo not supported for logs yet
                writer.add_audio("input", input_centre, state["step"], sample_rate=args.sr)

                for inst in outputs.keys():
                    writer.add_audio(inst + "_pred", torch.mean(outputs[inst][0], 0), state["step"], sample_rate=args.sr)
                    writer.add_audio(inst + "_target", torch.mean(targets[inst][0], 0), state["step"], sample_rate=args.sr)

            pbar.update(1)

    # VALIDATE
    val_loss = validate(args, model, criterion, val_data)
    print("VALIDATION FINISHED: LOSS: " + str(val_loss))
    writer.add_scalar("val_loss", val_loss, state["step"])

    # EARLY STOPPING CHECK
    checkpoint_path = os.path.join(args.snapshot_dir, "checkpoint_" + str(state["step"]))
    if val_loss >= state["best_loss"]:
        state["worse_epochs"] += 1
    else:
        print("MODEL IMPROVED ON VALIDATION SET!")
        state["worse_epochs"] = 0
        state["best_loss"] = val_loss
        state["best_checkpoint"] = checkpoint_path

    # SNAPSHOT
    print("Saving model...")
    utils.save_model(model, optimizer, state, checkpoint_path)

    state["epochs"] += 1

#### TESTING ####
# Test loss
print("TESTING")

# Load best model based on validation loss
state = utils.load_model(model, None, state["best_checkpoint"])
test_loss = validate(args, model, criterion, test_data)
print("TEST FINISHED: LOSS: " + str(test_loss))
writer.add_scalar("test_loss", test_loss, state["step"])

# Mir_eval metrics
test_metrics = evaluate(args, musdb["test"], model, INSTRUMENTS)

with open(os.path.join(args.snapshot_dir, "results.pkl", "wb")) as f:
    pickle.dump(test_metrics, f)

SDR = [np.mean([instrument["SDR"] for instrument in song.values()]) for song in test_metrics]
SIR = [np.mean([instrument["SIR"] for instrument in song.values()]) for song in test_metrics]

writer.add_scalar("test_SDR", SDR, state["step"])
writer.add_scalar("test_SIR", SIR, state["step"])
print("SDR: " + str(SDR))
print("SIR: " + str(SIR))

writer.close()