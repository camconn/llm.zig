
# llm.zig
A project to implement LLMs in Zig.

This project implements inference for LLMs *from scratch* without third
party libraries[^1].

[^1]: The Zig standard library is used. 3rd party libraries
like [zig-clap](https://github.com/Hejsil/zig-clap) are used, but only for
non-AI code like parsing CLI arguments.

## Implemented
- GGML
    - Loading `FP32` and `Q8_0` quantizations are supported.
    - Loading `Q8_0` quantiation is supported for Llama 1 & 2 models.
- GPT-2
    - `FP32` GGUF models are supported
    - Currently not exposed through CLI.
- LLaMA 1/2
    - `FP32` and `Q8_0` models are supported
    - SentencePiece-like tokenization
    - Key-Value caching
    - Loading "V1" models exported from Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## Outside Scope
Things this library doesn't do:
- Full model loading support from `pytorch`, `xformers`, JAX, etc.
- Training or fine-tuning

## Future
In the future other architectures such as mingpt, nanogpt, LLaMA3, etc. may be added.

# Setup

A full set of options for running `llm.zig` can be viewed by running the executable with the
`--help` flag:

```
$ zig build run -- --help
```
or
```
$ zig build
$ ./zig-out/bin/llm_zig --help
```

To actually do inference, you will need to download models in the appropriate
format. Instructions for generating supported files are shown below.

## LLaMA
How to setup LLaMA 2 inference

### GGUF Weights
Obtain a set of `.gguf` weights from somewhere with the `float32`/`fp32` or `Q8_0` quantization.

Then, run the model like so:

```
$ zig build run -Doptimize=ReleaseSafe -- --model path/to/llama2/weights-7b-f32.ggml
```

Or, you can add the `--format` argument to manually specify ggml:
```
$ zig build run -Doptimize=ReleaseSafe -- --format ggml --model path/to/llama2/weights-7b-f32.ggml
```



### llama2.c weights
1. Download the LLaMA 2 (full fp32 weights) weights.
2. Export the weights using the [llama2.c](https://github.com/karpathy/llama2.c) `export.py` script for version 1:
```
$ python3 export.py --version 1 --meta-llama /path/to/llama-2-7b/ llama2-7b.bin
```
3. Download a copy of
[`tokenizer.bin`](https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin)
or export a copy yourself using the script from `llama2.c`:
```
$ python3 tokenizer.py --tokenizer-model=/path/to/llama2/tokenizer.model
```
4. Build the code and run with `zig build run -Doptimize=ReleaseSafe --`
```
$ zig build run -Doptimize=ReleaseSafe -- --format llama2.c --tokenizer
tokenizer.bin --prompt='Zebras are primarily grazers and can subsist on lower-quality vegetation. They are preyed on mainly by'
```

# Licensing
The code in `llm.zig` is licensed under the GNU Affero Public License Version 3 or any
later version at your choosing. A copy of this license is located in `LICENSE.txt`.

A copy of the Unicode Character Database (UCD) for Unicode Version 16.0.0 is included with this program for
unit testing. A copy of the Unicode license is located in `LICENSE-UNICODE.txt`.
The UCD file is located at `src/assets/unicode-16.0.0.txt`.

# TODO
- Nanogpt or mingpt
    - Probably going to do nano

# Notes
Various development notes about model loading and are located below. Most people
should disregard this section.
