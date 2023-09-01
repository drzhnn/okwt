import math
import sys
from pathlib import Path
from pprint import pprint

import numpy as np

from .cli import get_cli
from .constants import Constant
from .dsp import (
    fade,
    flip,
    interpolate,
    invert_phase,
    maximize,
    normalize,
    processing_log,
    resize,
    reverse,
    shuffle,
    sort,
    trim,
)
from .formats import InputFile
from .utils import (
    get_frame_size_from_hint,
    pad_audio_data,
    write_wav,
    write_wt,
)


def main() -> None:
    cli = get_cli()

    sys.tracebacklimit = 0 if not cli.debug else 1

    infile = cli.infile

    if not Path(infile).exists():
        raise FileNotFoundError(infile)

    infile_obj = InputFile(infile)
    content = infile_obj.recognize_type().parse()

    print("Input file details:")
    pprint(dict(content._asdict()), sort_dicts=False)

    frame_size_hint: int = get_frame_size_from_hint(infile_obj.name)
    if frame_size_hint:
        print("\nFrame size from file name hint:", frame_size_hint)

    if cli.outfile or cli.split:
        # Force adding uhWT chunk if comment is not empty
        cli.add_uhwt = True if cli.comment else cli.add_uhwt

        samplerate: int = cli.samplerate or Constant.DEFAULT_SAMPLERATE
        frame_size: int = (
            cli.frame_size
            or content.frame_size
            or frame_size_hint
            or Constant.DEFAULT_FRAME_SIZE
        )

        # Trim before anything else
        if cli.trim:
            audio_data = trim(content.audio_data, cli.trim)
        else:
            audio_data = content.audio_data

        # Round up num_frames and make target_num_frames a whole number
        # within allowed limits
        num_frames_rounded = math.ceil(len(audio_data) / frame_size)
        target_num_frames = min(num_frames_rounded, Constant.DEFAULT_NUM_FRAMES)
        target_array_size: int = frame_size * target_num_frames

        # Fit audio data into target_num_frames
        # by truncating, padding or resizing
        if not cli.resize:
            if len(audio_data) >= target_array_size:
                # Truncate
                audio_data = audio_data[:target_array_size]
            else:
                # Pad with zeros
                audio_data = pad_audio_data(
                    audio_data,
                    frame_size,
                    target_num_frames,
                )
        else:
            audio_data = pad_audio_data(
                audio_data,
                frame_size,
                num_frames_rounded,
            )
            target_num_frames = cli.num_frames or target_num_frames
            audio_data = resize(
                audio_data, target_num_frames, cli.resize, frame_size
            )

        if cli.new_frame_size:
            audio_data = interpolate(audio_data, frame_size, cli.new_frame_size)
            frame_size = cli.new_frame_size

        # Sanity check
        audio_data_length = len(audio_data)
        if audio_data_length != int(audio_data_length):
            raise ValueError(
                f"Something went wrong. Audio data length: {audio_data_length}"
            )

        # Set seed for all future random operations
        seed = cli.seed or Constant.DEFAULT_SEED

        if cli.flip:
            audio_data = flip(audio_data, frame_size)

        if cli.invert:
            audio_data = invert_phase(audio_data, frame_size)

        if cli.sort:
            audio_data = sort(audio_data, frame_size)

        if cli.shuffle or cli.shuffle == 0:
            audio_data = shuffle(audio_data, frame_size, cli.shuffle, seed)

        if cli.reverse:
            audio_data = reverse(audio_data, frame_size)

        if cli.maximize:
            audio_data = maximize(audio_data, frame_size, 1.0)

        # Apply fades after the effects which could cause clicks
        if cli.fade or cli.fade == []:
            audio_data = fade(audio_data, frame_size, cli.fade)

        if cli.normalize:
            audio_data = normalize(audio_data, 1.0)

        # Print processing details
        if processing_log:
            print("\nProcessing:")
            for i, msg in enumerate(processing_log, start=1):
                print(f"{i}.", msg)

        # Force truncate audio data if getting --num-frames from CLI
        out_num_frames = cli.num_frames or target_num_frames

        # Make sure audio_data is in frames
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, frame_size)

        # Truncate audio_data to final num_frames
        if cli.num_frames:
            audio_data = audio_data[:out_num_frames]

        if cli.split:
            out_dir = Path(cli.infile + ".d")
            Path.mkdir(out_dir, exist_ok=True)

            frames = np.array_split(audio_data[:out_num_frames], out_num_frames)
            for i, frame in enumerate(frames):
                filename = Path(out_dir / f"frame_{i+1:03}.wav")
                write_wav(
                    filename_out=filename.as_posix(),
                    audio_data=frame,
                    frame_size=frame_size,
                    num_frames=1,
                    samplerate=samplerate,
                    add_uhwt_chunk=cli.add_uhwt,
                    add_srge_chunk=cli.add_srge,
                    comment=cli.comment or f"Frame {i+1:03}",
                )
        else:
            outfile_extension = Path(cli.outfile).suffix
            if outfile_extension == ".wav":
                write_wav(
                    filename_out=cli.outfile,
                    audio_data=audio_data,
                    frame_size=frame_size,
                    num_frames=out_num_frames,
                    samplerate=samplerate,
                    add_uhwt_chunk=cli.add_uhwt,
                    add_srge_chunk=cli.add_srge,
                    comment=cli.comment,
                )
            elif outfile_extension == ".wt":
                write_wt(
                    filename_out=cli.outfile,
                    audio_data=audio_data,
                    frame_size=frame_size,
                    num_frames=out_num_frames,
                    flags=0,
                )
            else:
                raise NotImplementedError

            print("\nOutput file details:")
            outfile = InputFile(cli.outfile).recognize_type().parse()
            pprint(dict(outfile._asdict()), sort_dicts=False)


if __name__ == "__main__":
    main()
