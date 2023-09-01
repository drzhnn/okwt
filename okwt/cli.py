import argparse
import importlib.metadata as md


def get_cli():
    info = md.metadata("okwt")

    parser = argparse.ArgumentParser(
        prog=info["Name"],
        description=info["Summary"],
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {info['Version']}",
        help="show version and exit",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="show detailed error messages",
    )
    parser.add_argument(
        "--ffmpeg-path",
        required=False,
        metavar="PATH",
        default="ffmpeg",
        help="path to ffmpeg binary",
    )

    input_group = parser.add_argument_group("input options")
    input_group.add_argument(
        "-i",
        "--infile",
        required=True,
        metavar="FILE",
        help=(
            "input file. Supported formats: WAV (16/24/32-bit), WT"
            " (Surge, Bitwig Studio), WT (Dune 3), VitalTable, "
            "PNG, JPEG, TIFF. "
            "Any other data will be interpreted as raw bytes"
        ),
    )

    process_group = parser.add_argument_group("processing options")
    process_group.add_argument(
        "--reverse",
        default=False,
        action="store_true",
        help="reverse order of frames",
    )
    process_group.add_argument(
        "--flip",
        default=False,
        action="store_true",
        help="reverse data inside each frame",
    )
    process_group.add_argument(
        "--invert",
        default=False,
        action="store_true",
        help="invert phase",
    )
    process_group.add_argument(
        "--shuffle",
        required=False,
        nargs="?",
        const=0,
        type=int,
        metavar="NUM",
        action="store",
        help=(
            "randomize order of frames. "
            "If NUM is provided, the wavetable is divided into groups "
            "and these groups are shuffled instead of individual frames"
        ),
    )
    process_group.add_argument(
        "--seed",
        required=False,
        type=int,
        metavar="NUM",
        action="store",
        help="seed for random generator (default: 5336)",
    )
    process_group.add_argument(
        "--fade",
        required=False,
        type=str,
        nargs="+",
        default=None,
        action="store",
        metavar="SIZE",
        help=(
            "apply fade-in and fade-out to each frame. "
            "SIZE is a length of fades in samples"
        ),
    )
    process_group.add_argument(
        "--sort",
        action="store_true",
        required=False,
        help="sort frames within wavetable",
    )
    process_group.add_argument(
        "--resize",
        type=str,
        metavar="MODE",
        required=False,
        help=(
            "resize large file to fit into wavetable. Available modes: "
            "'truncate', 'linear', 'bicubic', 'geometric', 'percussive'"
        ),
    )
    process_group.add_argument(
        "--trim",
        nargs="?",
        const=0.005,
        type=str,
        required=False,
        action="store",
        metavar="THRESH",
        help=(
            "remove silence (values below threshold) "
            "from the beginning and end of the audio. "
            "THRESH is in range 0.0-1.0 (default: 0.005)"
        ),
    )
    process_group.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="apply peak normalization",
    )
    process_group.add_argument(
        "--maximize",
        default=False,
        action="store_true",
        help="apply peak normalization to each frame",
    )

    metadata_group = parser.add_argument_group("metadata options")
    metadata_group.add_argument(
        "--rate",
        metavar="SAMPLERATE",
        dest="samplerate",
        type=int,
        help=(
            "set samplerate for 'fmt' chunk. This will not resample audio data"
        ),
    )
    metadata_group.add_argument(
        "--add-srge",
        default=False,
        action="store_true",
        help="write 'srge' chunk (Surge)",
    )
    metadata_group.add_argument(
        "--add-uhwt",
        default=False,
        action="store_true",
        help="write 'uhWT' chunk (Hive)",
    )
    metadata_group.add_argument(
        "--comment",
        type=str,
        default="",
        help="add comment to 'uhWT' chunk",
    )

    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "-o",
        "--outfile",
        required=False,
        metavar="FILE",
        help="output wavetable file. Supported formats: WAV, WT",
    )
    output_group.add_argument(
        "--num-frames",
        type=int,
        help=(
            "force total number of frames. Most useful when used with --resize"
        ),
    )
    output_group.add_argument(
        "--frame-size",
        type=int,
        help="set size of a single frame in samples",
    )
    output_group.add_argument(
        "--new-frame-size",
        type=int,
        required=False,
        action="store",
        metavar="SIZE",
        help="resample audio data to new frame size",
    )
    output_group.add_argument(
        "--split",
        required=False,
        action="store_true",
        help=(
            "save frames as inviditual files. "
            "The fallback frame size is 2048 samples "
            "(use --frame-size to modify it)"
        ),
    )
    return parser.parse_args()
