import argparse
import os
import time
from functools import partial

import torch
import pickle
import numpy as np

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import model.utils as model_utils
import utils
from data.dataset import SeparationDataset
from data.musdb import get_musdb_folds
from data.utils import crop_targets, random_amplify
from test import evaluate, validate
from model.waveunet import Waveunet

def main(args):
    #torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))

    writer = SummaryWriter(args.log_dir)

    ### DATASET
    musdb = get_musdb_folds(args.dataset_dir)
    # If not data augmentation, at least crop targets to fit model output shape
    crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training
    augment_func = partial(random_amplify, shapes=model.shapes, min=0.7, max=1.0)
    train_data = SeparationDataset(musdb, "train", args.instruments, args.sr, args.channels, model.shapes, True, args.hdf_dir, audio_transform=augment_func)
    val_data = SeparationDataset(musdb, "val", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)
    test_data = SeparationDataset(musdb, "test", args.instruments, args.sr, args.channels, model.shapes, False, args.hdf_dir, audio_transform=crop_func)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

    ##### TRAINING ####

    # Set up the loss function
    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Couldn't find this loss!")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

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

                # Set LR for this iteration
                utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)
                writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

                # Compute loss for each instrument/model
                optimizer.zero_grad()
                outputs, avg_loss = model_utils.compute_loss(model, x, targets, criterion, compute_grad=True)

                optimizer.step()

                state["step"] += 1

                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                writer.add_scalar("train_loss", avg_loss, state["step"])

                if example_num % args.example_freq == 0:
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
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path

        # CHECKPOINT
        print("Saving model...")
        model_utils.save_model(model, optimizer, state, checkpoint_path)

        state["epochs"] += 1

    #### TESTING ####
    # Test loss
    print("TESTING")

    # Load best model based on validation loss
    state = model_utils.load_model(model, None, state["best_checkpoint"], args.cuda)
    test_loss = validate(args, model, criterion, test_data)
    print("TEST FINISHED: LOSS: " + str(test_loss))
    writer.add_scalar("test_loss", test_loss, state["step"])

    # Mir_eval metrics
    test_metrics = evaluate(args, musdb["test"], model, args.instruments)

    # Dump all metrics results into pickle file for later analysis if needed
    with open(os.path.join(args.checkpoint_dir, "results.pkl"), "wb") as f:
        pickle.dump(test_metrics, f)

    # Write most important metrics into Tensorboard log
    avg_SDRs = {inst : np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics]) for inst in args.instruments}
    avg_SIRs = {inst : np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics]) for inst in args.instruments}
    for inst in args.instruments:
        writer.add_scalar("test_SDR_" + inst, avg_SDRs[inst], state["step"])
        writer.add_scalar("test_SIR_" + inst, avg_SIRs[inst], state["step"])
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

    writer.close()

if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
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
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    args = parser.parse_args()

    main(args)
