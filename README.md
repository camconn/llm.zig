
# LLM.zig
A project to implement LLMs in Ziglang.

# Scope
Initially, this project will aim at implementing GPT or LLAMA 2.

# Details
Details about the model weights file.

## Version 0
```
$ python3 export.py --version 0 --meta-llama /run/user/1000/kio-fuse-iuNegU/NETWORK_SERVER/Shared/AI/llama-2/llama-2-7b/ llama2-7b-0.bin
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
$ python3 export.py --version 1 --meta-llama /run/user/1000/kio-fuse-iuNegU/NETWORK_SERVER/Shared/AI/llama-2/llama-2-7b/ llama2-7b.bin
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

# TODO
- [x] Load/parse the tokenizer + model
- [ ] Tokenizer
  - [ ] Encode string -> tokens
  - [ ] Decode tokens -> string
  - [ ] Add tests vs sentencepiece
- [ ] RoPE
- [ ] MatMul
- [ ] SwiGLU
- [ ] Attention
  - [ ] Caching (optional, may be helpful for iteration/speed)
- [ ] Feed Forward
- [ ] Softmax
- [ ] Logits/Sample


# Notes
Will probably be useful to use `std.math.Complex` for the RoPE stuff w/ complex
#s in a `@Vector`.
