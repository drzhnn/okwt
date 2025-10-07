import numpy as np
from PIL import Image

processing_log = []


def reverse(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Reverse order of frames"""
    processing_log.append("Reverse order of frames")
    return np.flip(audio_data.reshape(-1, frame_size), axis=0)


def flip(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Reverse data within each frame"""
    audio_data = audio_data.reshape(-1, frame_size)
    processing_log.append("Reverse data within each frame")
    return np.flip(audio_data, axis=1)


def invert_phase(
    audio_data: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Invert phase"""
    audio_data = audio_data.reshape(-1, frame_size)
    processing_log.append("Invert phase")
    return audio_data * -1


def shuffle(audio_data: np.ndarray, frame_size: int, num_groups: int, seed: int) -> np.ndarray:
    """Randomize order of frames"""
    mutable_copy = np.copy(audio_data).reshape(-1)

    max_frames = audio_data.size // frame_size
    if (num_groups == 0) or (num_groups >= max_frames):
        num_groups = audio_data.size // frame_size

    num_frames = mutable_copy.size // (frame_size * num_groups) * num_groups

    mutable_copy = mutable_copy[: num_frames * frame_size].reshape(num_groups, -1)

    bit_generator = np.random.PCG64(seed=seed)
    rng = np.random.Generator(bit_generator)

    rng.shuffle(mutable_copy)
    mutable_copy = mutable_copy.reshape(-1)

    processing_log.append(f"Shuffle frames ({num_groups} groups)")
    return mutable_copy


def fade(audio_data: np.ndarray, frame_size: int, fades: str | list[str]) -> np.ndarray:
    """Apply fade-in and fade-out to each frame"""

    audio_data_as_frames = audio_data.reshape(-1, frame_size)

    # Convert 'fades' values from str into int
    int_fades = []
    for value in fades:
        if "%" in value:
            int_fades.append(round(frame_size * int(value.strip("%")) / 100))
        else:
            int_fades.append(int(value))

    if len(int_fades) == 1:
        fade_in = fade_out = int_fades[0]
    elif len(int_fades) >= 2:
        fade_in, fade_out = int_fades[0], int_fades[1]

    mutable_copy = np.copy(audio_data_as_frames)

    for i in range(mutable_copy.shape[0]):
        if fade_in:
            mutable_copy[i, :fade_in] *= np.linspace(0.0, 1.0, fade_in)
        if fade_out:
            mutable_copy[i, -fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    processing_log.append(f"Apply fades to each frame: {fade_in}, {fade_out}")
    return mutable_copy


def sort(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Sort frames within wavetable"""
    sorted_frames = np.sort(audio_data.reshape(-1, frame_size), axis=0)
    processing_log.append("Sort frames")
    return sorted_frames


def trim(audio_data: np.ndarray, threshold) -> np.ndarray:
    """Remove values below threshold from the beginning and end of the array"""
    threshold = float(threshold)
    mask = np.abs(audio_data) < threshold
    start_idx = np.argmax(~mask)
    end_idx = len(audio_data) - np.argmax(~mask[::-1])

    processing_log.append(
        f"Remove silence from the beginning and end of audio (threshold: {threshold}, max value: {np.max(audio_data)})"
    )

    return audio_data[start_idx:end_idx]


def resize(
    audio_data: np.ndarray,
    target_num_frames: int,
    resize_mode: str,
    frame_size: int,
) -> np.ndarray:
    """Resize array to fit into target_num_frames"""

    try:
        audio_data = audio_data.reshape(-1, frame_size)
    except Exception as e:
        raise NotImplementedError(f"Can't reshape {audio_data.shape}", e)

    resize_mode = resize_mode.lower()

    if len(resize_mode) < 3:
        raise SyntaxError("Resize mode keyword must be more than 2 letters long")

    if resize_mode in "truncate":
        # Simply trim the end
        processing_log.append("Resize mode: truncate")
        return audio_data[:target_num_frames]
    elif resize_mode in "linear":
        # Sample input array at equal intervals
        indices = np.linspace(0, len(audio_data) - 1, target_num_frames, dtype=int)
        processing_log.append("Resize mode: linear")
        return audio_data[indices]
    elif resize_mode in "bicubic":
        # Use Pillow library
        as_pil = Image.fromarray(audio_data)
        pil_resized = as_pil.resize((frame_size, target_num_frames), Image.Resampling.BICUBIC)
        new_array = np.array(pil_resized, dtype=np.float32)
        reshaped = new_array.reshape(-1)
        processing_log.append("Resize mode: bicubic")
        return reshaped
    elif resize_mode in "geometric":
        # Sample more often at the beginning
        indices = np.geomspace(
            1,
            len(audio_data) - 1,
            target_num_frames,
            dtype=int,
        )
        processing_log.append("Resize mode: geometric")
        return audio_data.reshape(-1, frame_size)[indices]
    elif resize_mode in "percussive":
        # Sample all frames in first half of the sample, then sample linearly.
        first_half_size = target_num_frames // 2
        first_half = audio_data[:first_half_size]
        indices = np.linspace(
            0,
            len(audio_data[first_half_size:]) - 1,
            target_num_frames - first_half_size,
            dtype=int,
        )
        second_half = audio_data[first_half_size:][indices]
        new = np.concatenate((first_half, second_half), axis=0)
        processing_log.append("Resize mode: percussive")
        return new
    else:
        raise SyntaxError(f"Unknown keyword '{resize_mode}'")


def clip(audio_data: np.ndarray, clip_to: float):
    return np.clip(audio_data, -clip_to, clip_to)


def normalize(audio_data: np.ndarray, normalize_to: float) -> np.ndarray:
    """Apply peak normalization to a specified value (0.0 - 1.0)"""
    peak_value = np.abs(audio_data).max()
    ratio = normalize_to / peak_value
    normalized = audio_data * ratio
    processing_log.append("Apply peak normalization")
    return normalized


def maximize(audio_data: np.ndarray, frame_size: int, maximize_to: float) -> np.ndarray:
    """Normalize each frame"""
    audio_data = audio_data.reshape(-1, frame_size)

    for frame in range(audio_data.shape[0]):
        audio_data[frame] = normalize(audio_data[frame], maximize_to)

    processing_log.append("Apply peak normalization to each frame")
    return audio_data


def interpolate(audio_data: np.ndarray, in_frame_size: int, out_frame_size):
    """Resample audio data to new frame size"""
    audio_frames = audio_data.reshape(-1, in_frame_size)
    target_size = audio_frames.shape[0] * out_frame_size
    linspace = np.linspace(0, len(audio_data), target_size)

    # TODO: fix edge values
    interpolated = np.interp(linspace, np.arange(len(audio_data)), audio_data).astype(np.float32)

    peak_value = np.abs(interpolated).max()
    normalize_to = 1.0
    ratio = normalize_to / peak_value
    normalized = interpolated * ratio

    processing_log.append(f"Resample to new frame size: {in_frame_size} -> {out_frame_size}")
    return normalized.astype(np.float32)


def overlap(audio_data: np.ndarray, frame_size: int, overlap_size: float):
    """Resize by overlapping frames"""

    raise NotImplementedError

    from scipy.signal import hann

    audio_data = audio_data.reshape(-1, frame_size)

    num_overlap_samples = int(overlap_size * audio_data.shape[1])

    fade_length = int(num_overlap_samples / 4)
    fade_in_window = hann(fade_length * 2)[:fade_length]
    fade_out_window = hann(fade_length * 2)[fade_length:]
    faded_array = audio_data.copy()

    for i in range(faded_array.shape[0]):
        faded_array[i, :fade_length] *= fade_in_window
        faded_array[i, -fade_length:] *= fade_out_window

    overlapping_array = np.concatenate(
        [faded_array[i, -num_overlap_samples * 2 :] for i in range(0, faded_array.shape[0])],
        axis=0,
    )
    return overlapping_array


def splice(audio_data: np.ndarray, num_frames: int):
    raise NotImplementedError


def mix_with_reversed_copy(audio_data: np.ndarray, num_frames: int):
    """Mix audio data with reversed copy of itself"""
    raise NotImplementedError
