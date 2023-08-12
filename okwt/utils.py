import base64
import re
import struct
import shlex
from hashlib import md5
import subprocess
import numpy as np


def bytes_to_array(audio_bytes: bytes, fmt_chunk):
    """Convert bytes to array"""
    bytes_per_sample = fmt_chunk.block_align // fmt_chunk.num_channels

    if fmt_chunk.codec_id == 1:
        if fmt_chunk.bitdepth in {16, 24, 32}:
            dtype = f"<i{bytes_per_sample}"
        else:
            raise ValueError("Unsupported bit depth:", fmt_chunk.bitdepth)
    elif fmt_chunk.codec_id == 3:
        if fmt_chunk.bitdepth in {32, 64}:
            dtype = f"<f{bytes_per_sample}"
        else:
            raise ValueError("Unsupported bit depth:", fmt_chunk.bitdepth)
    elif fmt_chunk.codec_id == 65534:
        if fmt_chunk.codec_id_hint == 1:
            dtype = f"<i{bytes_per_sample}"
        elif fmt_chunk.codec_id_hint == 3:
            dtype = f"<f{bytes_per_sample}"
        else:
            raise ValueError(
                "Unsupported codec id hint:", fmt_chunk.codec_id_hint
            )
    else:
        raise ValueError(
            "Unsupported format.",
            f"codec_id: {fmt_chunk.codec_id},",
            f"codec_id_hint: {fmt_chunk.codec_id_hint},",
            f"bitdepth: {fmt_chunk.bitdepth}",
        )

    audio_data = np.frombuffer(audio_bytes, dtype=dtype)

    if fmt_chunk.num_channels > 1:
        audio_data = audio_data.reshape(-1, fmt_chunk.num_channels)
    return to_float32(audio_data)


def get_md5(audio_data: bytes | np.ndarray) -> str:
    """Calculate MD5 hashsum of audio data."""
    if isinstance(audio_data, np.ndarray):
        audio_data = audio_data.tobytes()
    return md5(audio_data).hexdigest()


def to_mono(audio_data: np.ndarray):
    """Convert multichannel audio to mono"""
    if audio_data.ndim > 1:
        return audio_data[:, 0]
    else:
        return audio_data


def to_float32(audio_data: np.ndarray) -> np.ndarray[np.float32]:
    """Convert dtype to np.float32. Fit values into dtype range"""
    if audio_data.dtype != np.float32:
        audio_data = audio_data / np.iinfo(audio_data.dtype).max
    return audio_data.astype(np.float32)


def b64_to_array(base64_str, dtype=np.int16):
    """Convert base64 string into np.ndarray"""
    as_bytes = base64.b64decode(base64_str)
    return np.frombuffer(as_bytes, dtype)


def pad_audio_data(audio_data: np.ndarray, frame_size, target_num_frames):
    """Pad numpy array with zeros to match target size"""
    target_size = frame_size * target_num_frames

    if len(audio_data) < target_size:
        extra_zeros = int(max(target_size - len(audio_data), 0))
        audio_data_padded = np.pad(
            audio_data, (0, extra_zeros), mode="constant"
        )
        return audio_data_padded
    else:
        return audio_data


def get_frame_size_from_hint(filename: str) -> int:
    """Search for frame_size hints in file name ('WT512', 'wt-1024' etc.)"""
    pattern = r"(?:wt)(?:-?|_?|\s)(\d{2,})"
    results: list = re.findall(pattern, filename, re.IGNORECASE)
    if results == []:
        return 0
    else:
        hints = [int(r) for r in results]
        return max(hints)


def write_wav(
    filename_out: str,
    audio_data: np.ndarray,
    num_frames: int,
    frame_size: int,
    samplerate: int,
    comment: str = "",
    add_uhwt_chunk: bool = False,
    add_srge_chunk: bool = False,
) -> None:
    """Write data array as float32 .wav file with 16-bytes long 'fmt' chunk"""

    # Sanity check
    if audio_data.dtype != np.float32:
        raise TypeError(
            "Audio data should be in float32 format at this point. "
            f"Received {audio_data.dtype}"
        )

    RIFF_HEAD_LENGTH = 12
    FMT_CHUNK_LENGTH = 16
    FMT_NUM_CHANNELS = 1
    FMT_BITDEPTH = 32
    FMT_AVG_BYTERATE = FMT_NUM_CHANNELS * FMT_BITDEPTH * samplerate // 8
    FMT_CODEC_ID = 3
    FMT_BYTE_ALIGN = audio_data.itemsize
    UHWT_CHUNK_LENGTH = 272
    SRGE_CHUNK_LENGTH = 8
    SRGE_CHUNK_VERSION = 1

    CHUNK_HEADER = 8  # chunk_id + chunk_size
    riff_size: int = (
        RIFF_HEAD_LENGTH
        + (CHUNK_HEADER + FMT_CHUNK_LENGTH)
        + (CHUNK_HEADER + UHWT_CHUNK_LENGTH) * add_uhwt_chunk
        + (CHUNK_HEADER + SRGE_CHUNK_LENGTH) * add_srge_chunk
        + (4 * audio_data.size)
    )
    head = (b"RIFF", riff_size, b"WAVE")
    head_packed = struct.pack("<4s i 4s", *head)

    fmt_chunk = (
        b"fmt ",
        FMT_CHUNK_LENGTH,
        FMT_CODEC_ID,
        FMT_NUM_CHANNELS,
        samplerate,
        FMT_AVG_BYTERATE,
        FMT_BYTE_ALIGN,
        FMT_BITDEPTH,
    )
    fmt_packed = struct.pack("<4s i h h i i h h", *fmt_chunk)

    uhwt_packed = b""
    if add_uhwt_chunk:
        comment_encoded: bytes = comment[:256].encode("utf-8")
        comment_length: int = len(comment_encoded)
        uhwt_chunk = (
            b"uhWT",
            UHWT_CHUNK_LENGTH,
            num_frames,
            frame_size,
            comment_encoded,
        )
        uhwt_packed = struct.pack(
            f"<4s i 4x i i 4x {comment_length}s {256 - comment_length}x",
            *uhwt_chunk,
        )

    srge_packed = b""
    if add_srge_chunk:
        srge_chunk = (
            b"srge",
            SRGE_CHUNK_LENGTH,
            SRGE_CHUNK_VERSION,
            frame_size,
        )
        srge_packed = struct.pack("<4s i i i", *srge_chunk)

    data_header = (b"data", audio_data.size * 4)
    data_header_packed = struct.pack("<4s i", *data_header)

    with open(filename_out, "wb") as outfile:
        outfile.write(head_packed)
        outfile.write(fmt_packed)
        outfile.write(uhwt_packed)
        outfile.write(srge_packed)
        outfile.write(data_header_packed)
        audio_data.tofile(outfile)


def write_wt(
    filename_out: str,
    audio_data: np.ndarray,
    num_frames: int,
    frame_size: int,
    flags: int = 0,
) -> None:
    header = (b"vawt", frame_size, num_frames, flags)
    header_packed = struct.pack("<4s i H H", *header)

    with open(filename_out, "wb") as outfile:
        outfile.write(header_packed)
        audio_data.tofile(outfile)


def ffprobe_samplerate(filename_in: str, fallback_samplerate: int) -> int:
    """Probe media file (using 'ffprobe') and find its samplerate"""
    ffprobe_command = f"ffprobe -hide_banner {filename_in}"
    ffprobe_out = subprocess.check_output(
        shlex.split(ffprobe_command), stderr=subprocess.STDOUT
    )
    samplerate_match = re.search(r"(\d+) Hz", ffprobe_out.decode("utf-8"))

    if samplerate_match:
        return int(samplerate_match.group(1))
    else:
        return fallback_samplerate


def ffmpeg_read(filename_in: str, samplerate_in: int):
    read_as_float32 = f"ffmpeg -hide_banner -v error -i {filename_in} "
    read_as_float32 += f"-f wav -c:a pcm_f32le -ar {samplerate_in} -"

    with subprocess.Popen(
        shlex.split(read_as_float32),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        cache, error = process.communicate(timeout=3)

    return cache
