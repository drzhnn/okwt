# okwt

This tool allows you to convert any file into a wavetable, and perform useful operations on its frames.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Links](#links)

## Introduction

Let me start by stating some simple truths: there are many different wavetable synths in the world today. And this is beautiful. But what is not so beautiful is the fact that there are almost as many different wavetable file formats. And, while understandable, still, if you are a sound designer or producer and have a wavetable synth or two, then chances are, you are no stranger to a situation when you have a wavetable and you just can't load it into your synth. Even if it's a regular `.wav` file. So weird, right?

I feel you.

So I asked around, and it turned out `.wav` is just a filename extension and can actually represent many things. What really matters is the kind of data encapsulated within the file, and the way that data is described in special metadata chunks. If only there was a tool capable of reading such metadata from different file formats and utilizing it to transform an unreadable wavetable into a readable one...

Well, maybe okwt can help you with that.

The most important property of any wavetable is its __frame size__: the number of individual samples in a single __frame__ (one waveform cycle). If we know that number, we can divide the overall size of audio data by that amount and we'll get a __number of frames__ in that wavetable. 

By the way, wavetables work in frame rate (frames per second) domain, and the usual __sample rate__ (samples per second) property of the `.wav` file doesn't mean anything in the world of wavetables. Sample rate is just a value in 'fmt' metadata chunk of the `.wav` file, required by the PCM WAVE specification. Sample rate only matters when previewing wavetable files in a media player or audio editor. But I digress...

Let's say we have a wavetable `.wav` file which is 4096 samples long. Do you know how many frames it is? Is it 2 * 2048 or 4 * 1024 or else? How would a synth know? It won't. If there is no additional information in file's metadata (or if the synth can't read that kind of metadata), then it will use its __fallback frame size__. And every developer seems to have their own idea on what that value should be:

- Hive: 2048 samples
- Vital: 2048 samples
- Phaseplant: 2048 samples
- Dune: 2048 samples
- Serum: ?
- Surge: 1024 samples
- Ableton: 1024 samples
- Bitwig: 256 samples

Most of the synths listed above are capable of loading 256-frame long wavetables with 2048-sample long frames. And for the most of them 256x2048 is the upper limit. So, if we convert a wavetable into 256x2048 format, those synths should read it correctly, because they won't be able to load it as 128x4096 (frame's too big) or 512x1024 (too many frames). Looks like we've found the ideal format for a wavetable. And the size of such file would be just around 2MB, or 200MB for 100 wavetables. Not a big deal, unless you're a wavetable hoarding wonder. But even then, disk space shouldn't be your primary concern.

## Features

What okwt can do for you:

- read popular wavetable formats:  
    - `.wav`: 16-bit, 24-bit, 32-bit
    - `.wt`: Surge, Bitwig Studio
    - `.wt`: Dune3 (experimental support)
    - `.vitaltable`: Vital Audio (experimental support for sample based wavetables)
- extract useful information from wavetable chunks (`uhWT`, `srge`, `clm`)
- fix "unreadable" wavetables and resave them as correctly formatted `.wav`'s
- write wavetable metadata chunks (`uhWT`, `srge`)
- create wavetables from regular `.wav` files
- create wavetables from image files (`.png`, `.jpg`, `.tiff`)
- create wavetables from non-audio files
- apply effects to wavetable frames
- save wavetables as `.wav` or `.wt` files

## Installation

1. Using pip: `pip install okwt`  

2. From source:  
  - `git clone https://github.com/drzhnn/okwt.git`  
  - `cd okwt`  
  - `pip install .`  

3. From source using [poetry](https://python-poetry.org/docs/#installation):  
  - `git clone https://github.com/drzhnn/okwt.git`  
  - `cd okwt`  
  - `poetry install`  

## Usage

Show help: 

`okwt --help`  

Show audio file details: 

`okwt --infile audio.wav`

#### Convert any file to wavetable

Convert any file into the most universal wavetable format: 

- Extension: .wav
- Container: RIFF
- Format: WAVE
- Channels: 1
- Sample rate: 48000Hz
- Bit depth: 32-bit
- Codec ID: 3 (IEEE Float PCM)
- Chunks: 'fmt' and 'data'
- Size of 'fmt' chunk: 16 bytes
- Frame size: 2048 samples
- Number of frames: up to 256

`okwt --infile audio.wav --outfile wavetable.wav`

If your synth still refuses to load the resulting file, then try adding metadata chunks, or resample to 256-frame wavetable (see [Advanced examples](#advanced-examples))

#### Add chunks

Some synths require additional information in order to read wavetables correctly. This information (frame size, number of frames etc.) is usually stored in special metadata chunks inside .wav file. With okwt you can add 'uhWT' and 'srge' chunks:

`okwt --infile audio.wav --outfile wavetable.wav --add-uhwt`  
`okwt --infile audio.wav --outfile wavetable.wav --add-srge`  

#### Create wavetables from image files

Here's where real fun begins. You can create wavetables from pictures and photos. Internally, input image file will be resized to 2048x256 pixels and converted to black and white. After that, each horizontal line of the image will represent a frame of a wavetable, where each pixel is a single sample of a waveform, and its brightness is that sample's value, its amplitude:

`okwt --infile image.png --outfile wavetable.wav`

#### Resize large audio files

If input audio file is too big to fit into 256 frames, it will be truncated. But you can change this behavior by using `--resize` option:

`okwt --infile audio.wav --outfile wavetable.wav --resize linear`

Several resize modes are available:

- `truncate`: reads first 256 frames and skips the rest
- `linear`: reads the entire audio file and samples it at equal intervals
- `geometric`: samples the entire audio file in geometric progression
- `bicubic`: uses Pillow library to resize audio file as if it was an image
- `percussive`: tries to preserve more data from the beginning of the file, then samples linearly

Also, `--resize` can help stretching short audio files to fill larger wavetables.

#### Specify number of frames

Force resulting wavetable to have a specific number of frames instead of default 256. The following command will resize any file and fit it into 16 frames:

`okwt --infile audio.wav --outfile wavetable.wav --resize linear --num-frames 16`

#### Trim silence

Another useful feature of okwt is its ability to automatically skip the sampling of silence present at the beginning and end of the input audio file:

`okwt --infile audio.wav --outfile wavetable.wav --trim`

And you can even specify a threshold (in range 0.0-1.0):

`okwt --infile audio.wav --outfile wavetable.wav --trim 0.123`

#### Shuffle frames

It's possible to randomize the order of frames using `--shuffle`. By default it will randomize positions of all frames:

`okwt --infile audio.wav --outfile wavetable.wav --shuffle`

If this amount of chaos is too much for you, try shuffling a group of frames:

`okwt --infile audio.wav --outfile wavetable.wav --shuffle 4`

In this example, okwt will divide the wavetable into 4 groups and shuffle them around, while keeping the order of frames within each group.

If you want to change the seed of the internal random number generator, there's a handy `--seed` option available:

`okwt --infile audio.wav --outfile wavetable.wav --shuffle 4 --seed 4798235`

#### Fade-in, fade-out

Use `--fade` to remove clicks at the edges of frames. This command will apply 100 samples long fade-in and fade-out to each frame:

`okwt --infile audio.wav --outfile wavetable.wav --fade 100`

You can specify individual sizes for the fade-in and fade-out:

`okwt --infile audio.wav --outfile wavetable.wav --fade 100 700`

#### Normalize or maximize

Apply peak normalization:

`okwt --infile audio.wav --outfile wavetable.wav --normalize`

Or normalize each frame:

`okwt --infile audio.wav --outfile wavetable.wav --maximize`

#### Advanced examples

Convert a wavetable which has a just few frames into a full-fledged 256-frame long wavetable, with smooth interpolation between waveforms:

`okwt --infile audio.wav --outfile wavetable.wav --resize bicubic --num-frames 256`

Try this if you want to change the frame size, say from 1024 to 2048 samples:

`okwt --infile audio.wav --outfile wavetable.wav --frame-size 1024 --new-frame-size 2048 --add-srge`

## Dependencies

- numpy
- pillow

## Links

Other command line tools for working with wavetables:

1. [osc-gen](https://github.com/harveyormston/osc_gen)
2. [wt-tool](https://github.com/surge-synthesizer/surge/tree/main/scripts/wt-tool)

