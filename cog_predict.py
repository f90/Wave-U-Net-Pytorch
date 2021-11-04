import os
import cog
import tempfile
import zipfile
from pathlib import Path
import argparse
import data.utils
import model.utils as model_utils
from test import predict_song
from model.waveunet import Waveunet


class waveunetPredictor(cog.Predictor):
    def setup(self):
        """Init wave u net model"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--instruments",
            type=str,
            nargs="+",
            default=["bass", "drums", "other", "vocals"],
            help='List of instruments to separate (default: "bass drums other vocals")',
        )
        parser.add_argument(
            "--cuda", action="store_true", help="Use CUDA (default: False)"
        )
        parser.add_argument(
            "--features",
            type=int,
            default=32,
            help="Number of feature channels per layer",
        )
        parser.add_argument(
            "--load_model",
            type=str,
            default="checkpoints/waveunet/model",
            help="Reload a previously trained model",
        )
        parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        parser.add_argument(
            "--levels", type=int, default=6, help="Number of DS/US blocks"
        )
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
        parser.add_argument(
            "--strides", type=int, default=4, help="Strides in Waveunet"
        )
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
            "--separate",
            type=int,
            default=1,
            help="Train separate model for each source (1) or only one (0)",
        )
        parser.add_argument(
            "--feature_growth",
            type=str,
            default="double",
            help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)",
        )
        """
        parser.add_argument('--input', type=str, default=str(input),
                            help="Path to input mixture to be separated")
        parser.add_argument('--output', type=str, default=out_path, help="Output path (same folder as input path if not set)")
        """
        args = parser.parse_args([])
        self.args = args

        num_features = (
            [args.features * i for i in range(1, args.levels + 1)]
            if args.feature_growth == "add"
            else [args.features * 2 ** i for i in range(0, args.levels)]
        )
        target_outputs = int(args.output_size * args.sr)
        self.model = Waveunet(
            args.channels,
            num_features,
            args.channels,
            args.instruments,
            kernel_size=args.kernel_size,
            target_output_size=target_outputs,
            depth=args.depth,
            strides=args.strides,
            conv_type=args.conv_type,
            res=args.res,
            separate=args.separate,
        )

        if args.cuda:
            self.model = model_utils.DataParallel(model)
            print("move model to gpu")
            self.model.cuda()

        print("Loading model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(self.model, None, args.load_model, args.cuda)
        print("Step", state["step"])

    @cog.input("input", type=Path, help="audio mixture path")
    def predict(self, input):
        """Separate tracks from input mixture audio"""

        out_path = Path(tempfile.mkdtemp())
        zip_path = Path(tempfile.mkdtemp()) / "output.zip"

        preds = predict_song(self.args, input, self.model)

        out_names = []
        for inst in preds.keys():
            temp_n = os.path.join(
                str(out_path), os.path.basename(str(input)) + "_" + inst + ".wav"
            )
            data.utils.write_wav(temp_n, preds[inst], self.args.sr)
            out_names.append(temp_n)

        with zipfile.ZipFile(str(zip_path), "w") as zf:
            for i in out_names:
                zf.write(str(i))

        return zip_path
