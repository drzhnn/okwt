import json
import struct
from collections import namedtuple
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .constants import Constant
from .utils import (
    b64_to_array,
    bytes_to_array,
    ffmpeg_read,
    ffprobe_samplerate,
    get_md5,
    to_float32,
    to_mono,
)


class InputFile:
    def __init__(self, infile):
        self.infile = Path(infile).resolve()

    @property
    def parent(self) -> Path:
        return self.infile.parent

    @property
    def name(self) -> str:
        return self.infile.name

    @property
    def extension(self) -> str:
        return self.infile.suffix.lower()

    @property
    def size_in_bytes(self) -> int:
        return self.infile.stat().st_size

    @cached_property
    def cache(self) -> bytes | dict:
        if self.extension in Picture.SUPPORTED_FORMATS:
            cache = Image.open(self.infile)
            return cache
        elif self.extension == ".vitaltable":
            with self.infile.open("rt") as f:
                cache = json.load(f)
            return cache
        else:
            with self.infile.open("rb") as f:
                cache = memoryview(f.read())
            return cache

    @property
    def fourcc(self) -> bytes:
        return self.cache[:4]

    def recognize_type(self):
        if self.extension == ".wav" and self.fourcc == b"RIFF":
            return Wav(self.infile)
        elif self.extension == ".wt":
            if self.fourcc == b"vawt":
                return Wt(self.infile)
            else:
                return Wt_Dune(self.infile)
        elif self.extension == ".vitaltable":
            return VitalTable(self.infile)
        elif self.extension in Picture.SUPPORTED_FORMATS:
            return Picture(self.infile)
        elif self.extension in Ffmpeg.SUPPORTED_FORMATS:
            return Ffmpeg(self.infile)
        else:
            return Raw(self.infile)


class Picture(InputFile):
    SUPPORTED_FORMATS = [".png", ".jpeg", ".jpg", ".tiff"]

    def parse(self):
        resized_image = self.cache.resize(
            (Constant.DEFAULT_FRAME_SIZE, Constant.DEFAULT_NUM_FRAMES),
            Image.Resampling.BICUBIC,
        )
        grayscale = resized_image.convert("L")
        filtered = grayscale.filter(ImageFilter.DETAIL)
        numpy_array = np.array(filtered)[::-1]
        normalized = (numpy_array / 255.0 - 0.5) * 1.9
        audio_data = normalized.reshape(normalized.size).astype(np.float32)
        md5 = get_md5(audio_data)
        info = namedtuple(
            "info",
            [
                "num_frames",
                "frame_size",
                "audio_data",
                "md5",
            ],
        )
        return info(
            Constant.DEFAULT_NUM_FRAMES,
            Constant.DEFAULT_FRAME_SIZE,
            audio_data,
            md5,
        )


class Wt(InputFile):
    KNOWN_FLAGS = {
        1: "is sample",
        2: "loop sample",
        4: "16-bit audio",
        8: "int16 data is in range 2^16",
    }

    def parse(self):
        chunk = namedtuple(
            "chunk",
            [
                "chunk_id",
                "frame_size",
                "num_frames",
                "flags",
                "audio_data",
                "md5",
            ],
        )
        header_format = "<4s i H H"
        header_length = struct.calcsize(header_format)
        data_length = self.size_in_bytes - header_length
        file_format = f"{header_format} {data_length}s"
        unpacked_file = struct.unpack_from(file_format, self.cache)
        chunk_id, frame_size, num_frames, flags, audio_bytes = unpacked_file

        used_flags = [flag for flag in Wt.KNOWN_FLAGS if (flags & flag)]
        dtype = np.int16 if 4 in used_flags else np.float32
        md5 = get_md5(audio_bytes)
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        audio_data = to_float32(audio_data)
        values = (
            chunk_id,
            frame_size,
            num_frames,
            used_flags,
            audio_data,
            md5,
        )
        return chunk(*values)


class Wt_Dune(InputFile):
    def parse(self):
        chunk = namedtuple(
            "chunk",
            [
                "frame_size",
                "num_frames",
                "audio_data",
                "md5",
            ],
        )
        dune_header = "<h 10x h h i i 5x h 320x"
        header_length = struct.calcsize(dune_header)
        data_length = self.size_in_bytes - header_length
        file_format = f"{dune_header} {data_length}s"
        unpacked_file = struct.unpack_from(file_format, self.cache, 0)
        _, frame_size, _, bitdepth, num_frames, _, audio_bytes = unpacked_file
        dtype = np.int16 if bitdepth == 1 else np.int32
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        md5 = get_md5(audio_bytes)
        values = (frame_size, num_frames, audio_data, md5)
        return chunk(*values)


class Wav(InputFile):
    RIFF_HEADER_LENGTH = 12
    SUPPORTED_CODECS = {
        1: np.int16,
        65534: np.int32,
        3: np.float32,
    }

    def parse(self):
        has_uhwt = has_srge = False

        riff = Chunk.riff(self.cache, 0)
        structure = [riff]

        chunk_header = "<4s i"
        chunk_header_length = struct.calcsize(chunk_header)

        starts_at = Wav.RIFF_HEADER_LENGTH
        while starts_at < len(self.cache):
            chunk_id, chunk_length = struct.unpack_from(
                chunk_header, self.cache, starts_at
            )
            if chunk_id == b"fmt ":
                fmt_chunk = Chunk.fmt(self.cache, starts_at)
                structure.append(fmt_chunk)
            elif chunk_id == b"uhWT":
                uhwt_chunk = Chunk.uhwt(self.cache, starts_at)
                structure.append(uhwt_chunk)
                has_uhwt = True
            elif chunk_id == b"srge":
                srge_chunk = Chunk.srge(self.cache, starts_at)
                structure.append(srge_chunk)
                has_srge = True
            elif chunk_id == b"data":
                data_chunk, audio_bytes = Chunk.data(
                    self.cache, starts_at, chunk_length
                )
                structure.append(data_chunk)
            else:
                unknown_chunk = Chunk.unknown(self.cache, starts_at)
                structure.append(unknown_chunk)
            starts_at += chunk_header_length + chunk_length
            starts_at += 1 if chunk_length % 2 else 0

        audio_data = bytes_to_array(audio_bytes, fmt_chunk)
        audio_data = to_mono(audio_data)

        if has_uhwt:
            frame_size = uhwt_chunk.frame_size
        elif has_srge:
            frame_size = srge_chunk.frame_size
        else:
            frame_size = Constant.DEFAULT_FRAME_SIZE

        num_frames = len(audio_data) / frame_size

        md5 = get_md5(audio_data)
        info = namedtuple(
            "info",
            [
                "chunks",
                "audio_data",
                "frame_size",
                "num_frames",
                "md5",
            ],
        )
        return info(
            structure,
            audio_data,
            frame_size,
            num_frames,
            md5,
        )


class Ffmpeg(InputFile):
    SUPPORTED_FORMATS = [
        ".flac",
        ".wv",
        ".alac",
        ".ape",
        ".tta",
        ".mp3",
        ".ogg",
        ".opus",
        ".aac",
        ".aif",
        ".aiff",
        ".aifc",
    ]

    import shutil
    import sys

    sys.tracebacklimit = 0

    for app in ["ffmpeg", "ffprobe"]:
        if not shutil.which(app):
            raise OSError(f"{app} binary not found")

    def __init__(self, infile):
        super().__init__(infile)

    @cached_property
    def cache(self):
        windows_safe_infile = f"'{self.infile}'"
        samplerate_in = ffprobe_samplerate(
            windows_safe_infile, Constant.DEFAULT_SAMPLERATE
        )

        cache = ffmpeg_read(windows_safe_infile, samplerate_in)
        return cache

    def parse(self):
        riff = Chunk.riff(self.cache, 0)
        structure = [riff]

        chunk_header = "<4s i"
        chunk_header_length = struct.calcsize(chunk_header)

        starts_at = Wav.RIFF_HEADER_LENGTH
        cache_size = len(self.cache)
        while starts_at < cache_size:
            chunk_id, chunk_length = struct.unpack_from(
                chunk_header, self.cache, starts_at
            )
            if chunk_id == b"fmt ":
                fmt_chunk = Chunk.fmt(self.cache, starts_at)
                structure.append(fmt_chunk)
            elif chunk_id == b"data":
                # When piping, ffmpeg sets the size of data chunk to -1.
                # So we assume that `data` is the last chunk in the file
                # and load everything after 'data' chunk's header as audio data
                if chunk_length == -1:
                    chunk_length = cache_size - starts_at - chunk_header_length

                data_chunk, audio_bytes = Chunk.data(
                    self.cache, starts_at, chunk_length
                )
                structure.append(data_chunk)
            else:
                unknown_chunk = Chunk.unknown(self.cache, starts_at)
                structure.append(unknown_chunk)
            starts_at += chunk_header_length + chunk_length
            starts_at += 1 if chunk_length % 2 else 0

        audio_data = bytes_to_array(audio_bytes, fmt_chunk)
        audio_data = to_mono(audio_data)

        frame_size = Constant.DEFAULT_FRAME_SIZE

        num_frames = len(audio_data) / frame_size

        md5 = get_md5(audio_data)

        info = namedtuple(
            "info",
            [
                "chunks",
                "audio_data",
                "frame_size",
                "num_frames",
                "md5",
            ],
        )
        return info(
            structure,
            audio_data,
            frame_size,
            num_frames,
            md5,
        )


class Raw(InputFile):
    def parse(self):
        info = namedtuple(
            "info",
            ["audio_data", "num_frames", "frame_size", "md5"],
        )
        # Make sure length of cache is an even number
        if len(self.cache) % 2:
            self.cache = self.cache[:-1]

        data_with_nans = np.frombuffer(self.cache, dtype=np.float32)
        audio_data = np.nan_to_num(data_with_nans)
        normalized_data = (
            audio_data.astype(np.float32) / np.finfo(audio_data.dtype).max
        )
        clipped_data = np.clip(normalized_data, -0.01, 0.01) * 99.9
        frame_size = 2048

        num_frames = len(clipped_data) / frame_size
        md5 = get_md5(clipped_data)
        return info(clipped_data, num_frames, frame_size, md5)


class Chunk(Wav):
    @classmethod
    def riff(cls, cache: bytes, offset: int):
        chunk = namedtuple(
            "chunk",
            ["chunk_id", "chunk_size", "chunk_format"],
        )
        riff_format = "<4s i 4s"
        riff_values = struct.unpack_from(riff_format, cache, offset)
        riff_header = chunk(*riff_values)
        return riff_header

    @classmethod
    def unknown(cls, cache: bytes, offset: int):
        chunk = namedtuple(
            "chunk",
            ["chunk_id", "chunk_size"],
        )
        chunk_format = "<4s i"
        chunk_values = struct.unpack_from(chunk_format, cache, offset)
        ch = chunk(*chunk_values)
        return ch

    @classmethod
    def fmt(cls, cache: bytes, offset: int):
        chunk = namedtuple(
            "chunk",
            [
                "chunk_id",
                "chunk_size",
                "codec_id",
                "num_channels",
                "samplerate",
                "avg_byterate",
                "block_align",
                "bitdepth",
                "extension_size",
                "valid_bit_per_sample",
                "channel_mask",
                "codec_id_hint",
            ],
        )
        fmt_head_format = "<4s i"
        chunk_id, chunk_size = struct.unpack_from(
            fmt_head_format, cache, offset
        )

        if chunk_size in {16, 18}:
            fmt_format = "<H H i i H H"
        elif chunk_size == 40:
            fmt_format = "<H H i i H H H H i H"
        else:
            raise ValueError("Unexpected size of 'fmt' chunk:", chunk_size)

        fmt_values = struct.unpack_from(
            fmt_format, cache, offset + struct.calcsize(fmt_head_format)
        )

        if len(fmt_values) == 6:
            fmt_values = *fmt_values, None, None, None, None

        fmt_chunk = chunk(chunk_id, chunk_size, *fmt_values)
        return fmt_chunk

    @classmethod
    def uhwt(cls, cache: bytes, offset: int):
        chunk = namedtuple(
            "chunk",
            ["chunk_id", "chunk_size", "num_frames", "frame_size", "comment"],
        )
        uhwt_format = "<4s i 4x i i 4x 256s"
        uhwt_values = struct.unpack_from(uhwt_format, cache, offset)
        uhwt_decoded = uhwt_values[:4] + (
            uhwt_values[4].replace(b"\x00", b"").decode(),
        )
        return chunk(*uhwt_decoded)

    @classmethod
    def clm(cls, cache: bytes, offset: int, length: int):
        chunk = namedtuple(
            "chunk",
            ["chunk_id", "chunk_size", "frame_size", "flags"],
        )
        clm_format = "<4s i 3x 4s x s s s s s s s s"
        clm_values = struct.unpack_from(clm_format, cache, offset)
        chunk_id, chunk_size, frame_size, *flags = clm_values
        clm_values_decoded = (int(c) for c in clm_values[2:])
        frame_size, *flags = clm_values_decoded
        clm_decoded = (
            chunk_id,
            chunk_size,
            frame_size,
            flags,
        )
        return chunk(*clm_decoded)

    @classmethod
    def srge(cls, cache: bytes, offset: int):
        chunk = namedtuple(
            "chunk",
            ["chunk_id", "chunk_size", "version", "frame_size"],
        )
        srge_format = "<4s i i i"
        srge_values = struct.unpack_from(srge_format, cache, offset)
        return chunk(*srge_values)

    @classmethod
    def data(cls, cache: bytes, offset: int, length: int) -> tuple:
        chunk = namedtuple(
            "chunk",
            ["chunk_id", "chunk_size"],
        )
        data_format = f"<4s i {length}s"
        data_values = struct.unpack_from(data_format, cache, offset)
        chunk_id, chunk_size, data_bytes = data_values
        return (chunk(chunk_id, chunk_size), data_bytes)


class VitalTable(InputFile):
    def parse_audio_sample(self) -> tuple:
        audio_data_b64 = self.cache["groups"][0]["components"][0]["audio_file"]
        audio_data = b64_to_array(audio_data_b64)
        audio_data = to_float32(audio_data)
        samplerate = int(
            self.cache["groups"][0]["components"][0]["audio_sample_rate"]
        )
        frame_size = int(
            self.cache["groups"][0]["components"][0]["keyframes"][0][
                "window_size"
            ]
        )
        num_frames = len(audio_data) / frame_size
        md5 = get_md5(audio_data)
        return frame_size, num_frames, samplerate, audio_data, md5

    def parse_separate_frames(self) -> tuple:
        keys = self.cache["groups"][0]["components"][0]["keyframes"]
        frames = []
        for key in keys:
            as_b64 = b64_to_array(key["wave_data"])
            frames.append(as_b64)
        num_frames = len(keys)
        frame_size = 2048
        samplerate = 48000
        audio_data = np.concatenate(frames)
        audio_data = to_float32(audio_data)
        md5 = get_md5(audio_data)
        return frame_size, num_frames, samplerate, audio_data, md5

    def parse(self):
        info = namedtuple(
            "info",
            ["frame_size", "num_frames", "samplerate", "audio_data", "md5"],
        )
        table_type = self.cache["groups"][0]["components"][0]["type"]
        if table_type == 0:
            values = self.parse_separate_frames()
        elif table_type == "Audio File Source":
            values = self.parse_audio_sample()
        else:
            raise ValueError("VitalTable: Uknown file structure.")
        return info(*values)
