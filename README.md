
# llm.zig
A project to implement LLMs in Zig.

This project implements inference for LLMs *from scratch* without third
party libraries[^1].

[^1]: The Zig standard library is used. 3rd party libraries
like [zig-clap](https://github.com/Hejsil/zig-clap) are used, but only for
non-AI code like parsing CLI arguments.

## Implemented
- GGML
    - Loading `FP32` and `Q8_0` quantizations are supported for all models with `.gguf` files.
    - Loading `Q8_0` quantiation is supported for
- LLaMA 2
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
The code in `llm.zig` is licensed under the GNU Public License Version 3 or any
later version at your choosing. A copy of this license is located in
`LICENSE.txt`.

# TODO
- Nanogpt or mingpt
    - Probably going to do nano

# Notes
Various development notes about model loading and are located below. Most people
should disregard this section.

## LLaMA
Notes about LLaMA exports from llama2.py

### Version 0
```
$ python3 export.py --version 0 --meta-llama /path/to/llama-2-7b/ llama2-7b-0.bin
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-05, 'vocab_size': -1}
layers len: 32
tok_embeddings shape: torch.Size([32000, 4096])
attention shape: torch.Size([4096])
wq shape: torch.Size([4096, 4096])
wk shape: torch.Size([4096, 4096])
wv shape: torch.Size([4096, 4096])
wo shape: torch.Size([4096, 4096])
ffn_norm shape: torch.Size([4096])
w1 shape: torch.Size([11008, 4096])
w2 shape: torch.Size([4096, 11008])
w3 shape: torch.Size([11008, 4096])
norm shape: torch.Size([4096])
max_seq_len: 2048
freqs_cos shape: torch.Size([2048, 64])
freqs_cos shape: torch.Size([2048, 64])
not shared_classifier shape (separate model.output): torch.Size([32000, 4096])
wrote llama2-7b-0.bin
```


## Version 1
Output from a modified `export.py` file from `llama2.c`:
```
$ python3 export.py --version 1 --meta-llama /path/to/llama-2-7b/ llama2-7b.bin
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-05, 'vocab_size': -1}
layers len: 32
attention shape: torch.Size([4096])
ffn_norm shape: torch.Size([4096])
norm shape: torch.Size([4096])
tok_embeddings shape: torch.Size([32000, 4096])
wq shape: torch.Size([4096, 4096])
wk shape: torch.Size([4096, 4096])
wv shape: torch.Size([4096, 4096])
wo shape: torch.Size([4096, 4096])
w1 shape: torch.Size([11008, 4096])
w2 shape: torch.Size([4096, 11008])
w3 shape: torch.Size([11008, 4096])
not shared_classifier (separate model.output) shape: torch.Size([32000, 4096])
wrote llama2-7b.bin
```

### SentencePiece Settings
Dumped `sentencepice` settings from `tokenizer.model` to better replicate llama
2's behavior.

```
>>> import sentencepiece.sentencepiece_model_pb2
>>> mp = sentencepiece.sentencepiece_model_pb2.ModelProto()
>>> mp.ParseFromString(open("/path/to/llama-2/tokenizer.model", 'rb').read())
499723
>>> print(mp.trainer_spec)
input: "/large_experiments/theorem/datasets/MERGED/all.test1.merged"
model_prefix: "spm_model_32k_200M_charcov099995_allowWSO__v2"
model_type: BPE
vocab_size: 32000
self_test_sample_size: 0
input_format: "text"
character_coverage: 0.99995
input_sentence_size: 200000000
seed_sentencepiece_size: 1000000
shrinking_factor: 0.75
num_threads: 80
num_sub_iterations: 2
max_sentence_length: 4192
shuffle_input_sentence: true
max_sentencepiece_length: 16
split_by_unicode_script: true
split_by_whitespace: true
split_by_number: true
treat_whitespace_as_suffix: false
split_digits: true
allow_whitespace_only_pieces: true
vocabulary_output_piece_score: true
hard_vocab_limit: true
use_all_vocab: false
byte_fallback: true
required_chars: ""
unk_id: 0
bos_id: 1
eos_id: 2
pad_id: -1
unk_surface: " ⁇ "
unk_piece: "<unk>"
bos_piece: "<s>"
eos_piece: "</s>"
pad_piece: "<pad>"
train_extremely_large_corpus: false
enable_differential_privacy: false
differential_privacy_noise_level: 0
differential_privacy_clipping_threshold: 0

>>> print(mp.normalizer_spec)
name: "identity"
precompiled_charsmap: ""
add_dummy_prefix: true
remove_extra_whitespaces: false
normalization_rule_tsv: ""
```
