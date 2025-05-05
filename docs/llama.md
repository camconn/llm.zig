
# LLaMA
Notes about LLaMA models.

## Version 0
Modified version 0 export of Llama2 7B model from Karpathy's `llama2.c`.
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
Modified version 1 export of Llama2 7B model from Karpathy's `llama2.c`.
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

## SentencePiece Settings
Dumped `sentencepiece` settings from `tokenizer.model` to better replicate llama
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
