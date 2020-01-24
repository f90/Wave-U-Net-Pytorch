import argparse
import os
import utils

from test import predict_song
from waveunet import Waveunet

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--features', type=int, default=32,
                    help='# of feature channels per layer')
parser.add_argument('--load_model', type=str,
                    help='Reload a previously trained model')
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch size")
parser.add_argument('--levels', type=int, default=6,
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
parser.add_argument('--strides', type=int, default=4,
                    help="Strides in Waveunet")
parser.add_argument('--conv_type', type=str, default="gn",
                    help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
parser.add_argument('--res', type=str, default="fixed",
                    help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
parser.add_argument('--separate', type=int, default=1,
                    help="Train separate model for each source (1) or only one (0)")
parser.add_argument('--feature_growth', type=str, default="double",
                    help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

parser.add_argument('--input', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                    help="Path to input mixture to be separated")
parser.add_argument('--output', type=str, default=None, help="Output path (same folder as input path if not set)")

args = parser.parse_args()

INSTRUMENTS = ["bass", "drums", "other", "vocals"]
NUM_INSTRUMENTS = len(INSTRUMENTS)

# MODEL
num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
               [args.features*2**i for i in range(0, args.levels)]
target_outputs = int(args.output_size * args.sr)
model = Waveunet(args.channels, num_features, args.channels, INSTRUMENTS, kernel_size=args.kernel_size,
                 target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                 conv_type=args.conv_type, res=args.res, separate=args.separate)

if args.cuda:
    model = utils.DataParallel(model)
    print("move model to gpu")
    model.cuda()

print("Loading model from checkpoint " + str(args.load_model))
state = utils.load_model(model, None, args.load_model)

preds = predict_song(args, args.input, model)

output_folder = os.path.dirname(args.input) if args.output is None else args.output
for inst in preds.keys():
    utils.write_wav(os.path.join(output_folder, os.path.basename(args.input) + "_" + inst + ".wav"), preds[inst], args.sr)