import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model.utils as model_utils
import utils
from data.dataset import SeparationDataset, center_crop_audio
from data.musdb import get_musdb_folds
from model.waveunet import Waveunet
from test import evaluate, validate


def main(args):
    # torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

    # MODEL
    num_features = (
        [args.features * i for i in range(1, args.levels + 1)]
        if args.feature_growth == "add"
        else [args.features * 2 ** i for i in range(0, args.levels)]
    )
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(
        args.channels,
        num_features,
        args.channels,
        args.instruments,
        kernel_size=args.kernel_size,
        target_output_size=target_outputs,
        depth=args.depth,
        strides=args.strides,
        conv_type=args.conv_type,
        res=args.res
    )

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print("model: ", model)
    print("parameter count: ", str(sum(p.numel() for p in model.parameters())))

    writer = SummaryWriter(args.log_dir)

    ### DATASET
    musdb = get_musdb_folds(args.dataset_dir)
    train_data = SeparationDataset(
        musdb["train"],
        args.instruments,
        args.sr,
        args.channels,
        model.input_size,
        model.output_size,
        args.num_chunks_per_audio,
        randomize=True,
    )
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer_size)

    val_data = SeparationDataset(
        musdb["val"],
        args.instruments,
        args.sr,
        args.channels,
        model.input_size,
        model.output_size,
        args.num_chunks_per_audio,
        randomize=False,
    )
    test_data = SeparationDataset(
        musdb["test"],
        args.instruments,
        args.sr,
        args.channels,
        model.input_size,
        model.output_size,
        args.num_chunks_per_audio,
        randomize=False,
    )

    # TODO add shuffle buffer somewhere for train!

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=utils.worker_init_fn,
    )

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
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=args.min_lr
    )

    # Set up training state dict that will also be saved into checkpoints
    state = {"step": 0, "worse_epochs": 0, "epochs": 0, "best_loss": np.Inf}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

    print("TRAINING START")
    while state["worse_epochs"] < args.patience:
        print(f"Training epoch {state['epochs']} starting from iteration {state['step']}")
        model.train()
        np.random.seed()

        with tqdm() as pbar:
            for example_num, (mix, targets) in enumerate(dataloader):
                if args.cuda:
                    mix = mix.cuda()
                    targets = targets.cuda()

                # Compute loss
                optimizer.zero_grad()
                outputs = model(mix)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                state["step"] += 1

                writer.add_scalar("train_loss", loss, state["step"])

                if example_num % args.example_freq == 0:
                    # Audio summaries
                    writer.add_audio(
                        "input",
                        torch.mean(center_crop_audio(mix[0], model.output_size), 0),
                        state["step"],
                        sample_rate=args.sr,
                    )

                    for i, inst in enumerate(args.instruments):
                        writer.add_audio(
                            inst + "_pred",
                            torch.mean(outputs[0,i*args.channels:(i+1)*args.channels], 0),
                            state["step"],
                            sample_rate=args.sr,
                        )
                        writer.add_audio(
                            inst + "_target",
                            torch.mean(targets[0,i*args.channels:(i+1)*args.channels], 0),
                            state["step"],
                            sample_rate=args.sr,
                        )

                pbar.update(1)

        # VALIDATE
        val_loss = validate(args, model, criterion, val_data)
        print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state["step"])

        # Adjust learning rate based on val loss
        scheduler.step(val_loss)
        writer.add_scalar("lr", utils.get_lr(optimizer), state["step"])

        # EARLY STOPPING CHECK
        checkpoint_path = os.path.join(
            args.checkpoint_dir, "checkpoint_" + str(state["step"])
        )
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
    avg_SDRs = {
        inst: np.mean([np.nanmean(song[inst]["SDR"]) for song in test_metrics])
        for inst in args.instruments
    }
    avg_SIRs = {
        inst: np.mean([np.nanmean(song[inst]["SIR"]) for song in test_metrics])
        for inst in args.instruments
    }
    for inst in args.instruments:
        writer.add_scalar("test_SDR_" + inst, avg_SDRs[inst], state["step"])
        writer.add_scalar("test_SIR_" + inst, avg_SIRs[inst], state["step"])
    overall_SDR = np.mean([v for v in avg_SDRs.values()])
    writer.add_scalar("test_SDR", overall_SDR)
    print("SDR: " + str(overall_SDR))

    writer.close()


if __name__ == "__main__":
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        default=["bass", "drums", "other", "vocals"],
        help='List of instruments to separate (default: "bass drums other vocals")',
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA (default: False)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of data loader worker threads (default: 1)",
    )
    parser.add_argument(
        "--features", type=int, default=32, help="Number of feature channels per layer"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/waveunet", help="Folder to write logs into"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/windaten/Datasets/MUSDB18HQ",
        help="Dataset path",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/waveunet",
        help="Folder to write checkpoints into",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Reload a previously trained model (whole task model)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for the LR scheduler"
    )
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--levels", type=int, default=6, help="Number of DS/US blocks")
    parser.add_argument(
        "--depth", type=int, default=1, help="Number of convs per block"
    )
    parser.add_argument("--sr", type=int, default=44100, help="Sampling rate")
    parser.add_argument(
        "--channels", type=int, default=2, help="Number of input audio channels"
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Filter width of kernels. Has to be an odd number",
    )
    parser.add_argument(
        "--output_size", type=float, default=2.0, help="Output duration"
    )
    parser.add_argument("--strides", type=int, default=4, help="Strides in Waveunet")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for early stopping on validation set",
    )
    parser.add_argument(
        "--example_freq",
        type=int,
        default=200,
        help="Write an audio summary into Tensorboard logs every X training iterations",
    )
    parser.add_argument("--loss", type=str, default="L1", help="L1 or L2")
    parser.add_argument(
        "--conv_type",
        type=str,
        default="gn",
        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn",
    )
    parser.add_argument(
        "--res",
        type=str,
        default="fixed",
        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned",
    )
    parser.add_argument(
        "--feature_growth",
        type=str,
        default="double",
        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)",
    )

    args = parser.parse_args()

    main(args)
