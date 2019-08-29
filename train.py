import argparse
import os
import time

import torch
import pickle
import numpy as np

from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import utils
from data import get_musdb_folds, SeparationDataset
from evaluate import evaluate
from waveunet import Waveunet

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
        for example_num, (x, target) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                target = target.cuda()

            output = model(x)

            # Compute loss by stacking sources together in channel dimension
            packed_output = torch.cat(list(output.values()), dim=1)
            loss = criterion(packed_output, target)

            total_loss += (1. / float(example_num + 1)) * (loss.item() - total_loss)

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
parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18",
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
parser.add_argument('--sr', type=int, default=22050,
                    help="Sampling rate")
parser.add_argument('--channels', type=int, default=1,
                    help="Number of input audio channels")
parser.add_argument('--output_size', type=float, default=2.0,
                    help="Output duration")
parser.add_argument('--strides', type=int, default=2,
                    help="Strides in Waveunet")
parser.add_argument('--patience', type=int, default=6,
                    help="Patience for early stopping on validation set")
parser.add_argument('--loss', type=str, default="L1",
                    help="L1 or L2")

args = parser.parse_args()

np.random.seed(1337)
INSTRUMENTS = ["bass", "drums", "other", "vocals"]
NUM_INSTRUMENTS = len(INSTRUMENTS)

torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

# MODEL
num_features = [args.features*i for i in range(1, args.levels+1)] # Double features every layer?
target_outputs = int(args.output_size * args.sr)
model = Waveunet(args.channels, num_features, args.channels, INSTRUMENTS, kernel_size=5, target_output_size=target_outputs, depth=args.depth, strides=args.strides)

if args.cuda:
    model = utils.DataParallel(model)
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

writer = SummaryWriter(args.log_dir)

### DATASET
musdb = get_musdb_folds(args.dataset_dir)
train_data = SeparationDataset(musdb, "train", INSTRUMENTS, args.sr, args.channels, model.shapes, random_hops=True)
val_data = SeparationDataset(musdb, "val", INSTRUMENTS, args.sr, args.channels, model.shapes, random_hops=False)
test_data = SeparationDataset(musdb, "test", INSTRUMENTS, args.sr, args.channels, model.shapes, random_hops=False)

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
        for example_num, (x, target) in enumerate(dataloader):
            if args.cuda:
                x = x.cuda()
                target = target.cuda()

            t = time.time()
            output = model(x)

            # Compute loss by stacking sources together in channel dimension
            packed_output = torch.cat(list(output.values()), dim=1)
            loss = criterion(packed_output, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            state["step"] += 1

            t = time.time() - t
            avg_time += (1. / float(example_num + 1)) * (t - avg_time)

            writer.add_scalar("train_loss", loss.item(), state["step"])

            if example_num % 50 == 0:
                writer.add_audio("input", x[0,:,model.shapes["output_start_frame"]:model.shapes["output_end_frame"]], state["step"], sample_rate=args.sr)
                for i in range(len(INSTRUMENTS)):
                    writer.add_audio(INSTRUMENTS[i] + "_pred", packed_output[0,i*args.channels:(i+1)*args.channels], state["step"], sample_rate=args.sr)
                    writer.add_audio(INSTRUMENTS[i] + "_target", target[0,i*args.channels:(i+1)*args.channels], state["step"], sample_rate=args.sr)

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