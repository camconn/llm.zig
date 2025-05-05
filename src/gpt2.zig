// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! gpt2: This module contains an implementation of GPT-2 [1].
//! Code is based on the original open-sourced version of GPT-2 published by OpenAI on Github.
//!
//! [1]: https://github.com/openai/gpt-2

const std = @import("std");

const llm = @import("root.zig");
const ggml = llm.ggml;
const math = llm.math;
const sample = llm.sample;
const tkn = llm.token;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const Tokenizer = tkn.TikTokenizer;

const Weights = math.Weights;

pub const Error = error{
    /// Attempted to an unsupported model file.
    BadFile,
    /// The file has incorrect formatting or is corrupted.
    BadFormat,
};

pub const Gpt2Context = struct {
    const Self = @This();

    config: Config,
    transformer: Transformer,
    state: State,
    tokenizer: Tokenizer,
    file: ?ggml.GGUFFile,

    /// Attempt to initialize and and load a GPT-2 model from `file`.
    /// The returned `Gpt2Context` takes ownership of the opened file.
    pub fn init(file_name: []const u8, allocator: std.mem.Allocator) !Gpt2Context {
        var file = try ggml.GGUFFile.read_file(file_name, allocator);
        errdefer file.deinit();

        var tokenizer = try Tokenizer.init(file, allocator);
        errdefer tokenizer.deinit();
        var transformer = try Transformer.init(file, allocator);
        errdefer transformer.deinit();
        const config = transformer.config;
        var state = try State.init(config, allocator);
        errdefer state.deinit();

        return .{
            .config = config,
            .tokenizer = tokenizer,
            .transformer = transformer,
            .state = state,
            .file = file,
        };
    }

    /// De-initialize and free up any resources associated with this `Gpt2Context`.
    pub fn deinit(self: *Self) void {
        self.transformer.deinit();
        self.state.deinit();
        self.tokenizer.deinit();
        if (self.file) |*file| {
            file.deinit();
            self.file = null;
        }
    }
};

/// The loaded configuration for a GPT-2 model.
pub const Config = struct {
    const context_len_key = "gpt2.context_length";
    const block_count_key = "gpt2.block_count";
    const embedding_len_key = "gpt2.embedding_length";
    const ffn_len_key = "gpt2.feed_forward_length";
    const head_count_key = "gpt2.attention.head_count";
    const attn_epsilon_key = "gpt2.attention.layer_norm_epsilon";

    n_vocab: usize,
    n_ctx: usize,
    n_embed: usize,
    n_heads: usize,
    n_layers: usize,

    bos_id: ?Tokenizer.Token,

    /// Load a GPT-2 model config from the provided GGUF `file`.
    pub fn init(file: ggml.GGUFFile) !Config {
        if (file.getValueT(ggml.arch_key, .string)) |val| {
            if (!std.ascii.eqlIgnoreCase(val.string.str, "gpt2")) {
                std.debug.print("gpt2: Tried to load a file with the wrong architecture: {s}\n", .{val.string.str});
                return Error.BadFile;
            }
        } else {
            std.debug.print("gpt2: model is missing {s} type string metadata\n", .{ggml.arch_key});
            return Error.BadFormat;
        }

        const context_len = file.getValueT(context_len_key, .uint32) orelse return Error.BadFormat;
        const block_count = file.getValueT(block_count_key, .uint32) orelse return Error.BadFormat;
        const embed_len = file.getValueT(embedding_len_key, .uint32) orelse return Error.BadFormat;
        const head_count = file.getValueT(head_count_key, .uint32) orelse return Error.BadFormat;
        // TODO: Actually use epsilon
        const eps = file.getValueT(attn_epsilon_key, .float32) orelse return Error.BadFormat;
        std.debug.print("got model ε={d}\n", .{eps.float32});

        const vocab_array = file.getValueT("tokenizer.ggml.tokens", .array) orelse return Error.BadFormat;
        const n_vocab = vocab_array.array.len;

        const bos_id = file.getValueT("tokenizer.ggml.bos_token_id", .uint32) orelse return Error.BadFormat;

        return .{
            .n_vocab = @intCast(n_vocab),
            .n_ctx = context_len.uint32,
            .n_embed = embed_len.uint32,
            .n_heads = head_count.uint32,
            .n_layers = block_count.uint32,
            .bos_id = bos_id.uint32,
        };
    }
};

/// The working state of a GPT-2 transformer. Contains only the working memory and current
/// activations.
pub const State = struct {
    const Self = @This();

    arena: ArenaAllocator,

    input: []f32,

    work1: []f32,
    work2: []f32,
    fat_work: []f32,

    q: []f32,
    k: []f32,
    v: []f32,
    k_cache: []f32,
    v_cache: []f32,
    attention: []f32,

    layer_out: []f32,

    output: []f32,

    /// Initialize the state with the provided `config`.
    pub fn init(config: Config, allocator: std.mem.Allocator) !State {
        const c = config;

        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const layers = c.n_layers;
        const dim = c.n_embed;
        // # of layers * context len * size of heads
        const kv_cache_size = layers * c.n_ctx * dim;

        // Input buffer
        const input = try alloc.alloc(f32, dim);

        // Scratchpad buffers
        const work1 = try alloc.alloc(f32, dim);
        const work2 = try alloc.alloc(f32, dim);
        const fat_work = try alloc.alloc(f32, dim * 4);

        const q = try alloc.alloc(f32, dim);
        const k = try alloc.alloc(f32, dim);
        const v = try alloc.alloc(f32, dim);

        const attention = try alloc.alloc(f32, c.n_heads * c.n_ctx);

        // KV Cache
        const k_cache = try alloc.alloc(f32, kv_cache_size);
        const v_cache = try alloc.alloc(f32, kv_cache_size);

        // Layer output
        const layer_out = try alloc.alloc(f32, dim);

        // Final output
        const output = try alloc.alloc(f32, c.n_vocab);

        return .{
            .arena = arena,

            .input = input,

            .work1 = work1,
            .work2 = work2,
            .fat_work = fat_work,

            .q = q,
            .k = k,
            .v = v,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .attention = attention,

            .layer_out = layer_out,

            .output = output,
        };
    }

    /// Free up any invalidate any memory currently used by this `State` instance.
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }
};

/// One of the `block` layers of the GPT-2 transformer.
const Block = struct {
    // Pre-attention weights
    ln_1_w: Weights,
    ln_1_b: Weights,

    // Attention weights
    qkv_w: Weights,
    qkv_b: Weights,
    c_proj_w: Weights,
    c_proj_b: Weights,

    // Post-attention weights
    ln_2_w: Weights,
    ln_2_b: Weights,
    mlp_c_fc_w: Weights,
    mlp_c_fc_b: Weights,
    mlp_c_proj_w: Weights,
    mlp_c_proj_b: Weights,

    /// Load a `Block` number `n` from the GGUF file `file`.
    /// *Does not* take ownership of `file` and assumes its memory references will last the
    /// life of this block.
    pub fn init(file: ggml.GGUFFile, n: usize) !Block {
        return .{
            // ln_1 = attn_norm in gguf
            .ln_1_w = getTensor("attn_norm", "weight", file, n),
            .ln_1_b = getTensor("attn_norm", "bias", file, n),
            // c_attn = attn_qkv in gguf
            .qkv_w = getTensor("attn_qkv", "weight", file, n),
            .qkv_b = getTensor("attn_qkv", "bias", file, n),
            // [attn.]c_proj = attn_out in gguf
            .c_proj_w = getTensor("attn_output", "weight", file, n),
            .c_proj_b = getTensor("attn_output", "bias", file, n),

            // ln_2 = ffn_norm in gguf
            .ln_2_w = getTensor("ffn_norm", "weight", file, n),
            .ln_2_b = getTensor("ffn_norm", "bias", file, n),
            // [mlp.]c_fc = ffn_up in gguf
            .mlp_c_fc_w = getTensor("ffn_up", "weight", file, n),
            .mlp_c_fc_b = getTensor("ffn_up", "bias", file, n),
            // [mlp.]c_proj = ffn_down in gguf
            .mlp_c_proj_w = getTensor("ffn_down", "weight", file, n),
            .mlp_c_proj_b = getTensor("ffn_down", "bias", file, n),
        };
    }

    /// Get the contents of a tensor named `blk.<n>.<name>.<typ>` from the loaded GGUF `file`.
    /// Assumes that the backing tensor exists within the GGUF file.
    fn getTensor(name: []const u8, typ: []const u8, file: ggml.GGUFFile, n: usize) Weights {
        std.debug.assert(n < 32);
        var buf = [_]u8{0} ** 64;
        const full_name = std.fmt.bufPrint(&buf, "blk.{d}.{s}.{s}", .{ n, name, typ }) catch unreachable;
        if (file.getTensorWeights(full_name)) |tensor| {
            return tensor;
        }
        std.debug.print("Error: Tensor {s} does not exist in the GGUF file", .{full_name});
        @panic("Could not load tensor");
    }
};

/// Transformer weights for GPT-2.
pub const Transformer = struct {
    const Self = @This();

    arena: ArenaAllocator,

    config: Config,

    wte: Weights,
    wpe: Weights,
    layers: []Block,
    out_norm_w: Weights,
    out_norm_b: Weights,
    output_w: Weights,

    /// Initialize the the weights from a `file` into this `Transformer`.
    /// *Does not* take ownership of the provided file, but assumes its memory mapping stay valid.
    pub fn init(file: ggml.GGUFFile, allocator: std.mem.Allocator) !Transformer {
        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        const config = try Config.init(file);

        const n_layers = config.n_layers;
        var layers = try alloc.alloc(Block, n_layers);
        for (0..n_layers) |i| {
            layers[i] = try Block.init(file, i);
        }

        const tok_embedding = file.getTensorWeights("token_embd.weight").?;
        const pos_embedding = file.getTensorWeights("position_embd.weight").?;

        const output_norm_w = file.getTensorWeights("output_norm.weight").?;
        const output_norm_b = file.getTensorWeights("output_norm.bias").?;
        // GPT-2 allows you to share the token and output embedding matrices
        const output = file.getTensorWeights("output.weight") orelse tok_embedding;

        return .{
            .arena = arena,

            .config = config,
            .wte = tok_embedding,
            .wpe = pos_embedding,
            .layers = layers,
            .out_norm_w = output_norm_w,
            .out_norm_b = output_norm_b,
            .output_w = output,
        };
    }

    /// Free up the memory associated with this `Transformer`. This does not free up the
    /// `ggml.GGUFFile` used to load the current weights.
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Perform an inference pass for this transformer with `state` adding `token` at position `n_token`.
    /// Returns the probabilities of outputting each token.
    pub fn forward(self: Self, state: *State, token: Tokenizer.Token, n_token: usize) []f32 {
        //std.debug.print("Transformer.forward w/ token {d} at {d}\n", .{ token, n_token });
        const c = self.config;

        // corresponding lines of OpenAI' `gpt-2/src/model.py` are indicated next to their
        // implementation here.

        // x = wte(x) + wpe(x)
        self.applyEmbeddings(state.input, state.work1, token, n_token);

        for (0.., self.layers) |i, layer| {
            // h, present = block(h, 'h%d' % layer)
            // now inside def block(...)

            // Copy input into state.work1 so we have a copy
            @memcpy(state.work1, state.input);

            // a, present  = attn(norm(x, 'ln_1'), 'attn', nx)
            // now inside norm(x, 'ln_1')
            math.layerNorm(f32, state.work1, layer.ln_1_w.f32, layer.ln_1_b.f32, state.work2);
            // done with norm(x, 'ln_1')

            // state.work1 = ln_1(x)
            // state.input = x

            // now inside attn(...)
            self.attention(state, i, layer, n_token);
            // output of attention is in state.work2
            // now outside attn(...)
            // x = x + a
            math.add(f32, state.input, state.work2, state.input);

            // m = mlp(norm(x, 'ln_2'), 'mlp', ...)
            // now inside norm(...)
            // copy attention activations so we can add them back later
            @memcpy(state.layer_out, state.input);
            math.layerNorm(f32, state.layer_out, layer.ln_2_w.f32, layer.ln_2_b.f32, state.work1);
            // now outside norm(...), inside mlp(norm(...), ...)

            // now inside mlp(norm(...), ...)
            self.mlp(state, layer);
            // outside of mlp(norm(...), ...)
            // output of MLP is in state.layer_out

            // x = x + m
            math.add(f32, state.input, state.input, state.layer_out);
            // done with block

            // return x, present
            // layer or `block(...)` output is at state.input
        }

        // out_norm = ln_f
        // h = norm(h, 'ln_f')
        math.layerNorm(f32, state.input, self.out_norm_w.f32, self.out_norm_b.f32, state.output[0..c.n_embed]);

        // logits = tf.matmul(h_flat, wte)
        math.matrixMulVec(f32, state.output, self.output_w, .{ .f32 = state.input }, c.n_vocab, c.n_embed);

        // return results
        return state.output;
    }

    /// Apply input embedding for token `tok` at position `pos` to `out` while using temporary `scratch`.
    fn applyEmbeddings(self: Self, out: []f32, scratch: []f32, tok: Tokenizer.Token, pos: usize) void {
        self.applyTokenEmbedding(out, tok);
        self.applyPositionEmbedding(scratch, pos);
        math.add(f32, out, out, scratch);
    }

    /// Copy token embeddings for the current input `token` into the `out` vector.
    fn applyTokenEmbedding(self: Self, out: []f32, tok: Tokenizer.Token) void {
        const dim = self.config.n_embed;
        const token_offset: usize = @as(usize, @intCast(tok)) * dim;

        switch (self.wte) {
            .f32 => |floats| {
                const embeddings = floats[token_offset .. token_offset + dim];
                @memcpy(out, embeddings);
            },
            .q8_0 => |quantized| {
                const block_size = math.blockUnitLen(math.Block(.q8_0));
                const block_offset = token_offset / block_size;
                const n_blocks = dim / block_size;

                const embeddings = quantized[block_offset .. block_offset + n_blocks];
                math.dequantizeT(f32, .q8_0, embeddings, out) catch
                    @panic("Terrible situation in embeddings");
            },
            else => {
                std.debug.print("Unsupported token embedding format: {any}\n", .{
                    @as(math.WeightFormat, self.wte),
                });
                @panic("Unsupported token embedding quantization");
            },
        }
    }

    /// Copy learned position embeddings for position `pos` in the `out` vector.
    fn applyPositionEmbedding(self: Self, out: []f32, pos: usize) void {
        const dim = self.config.n_embed;
        const pos_offset: usize = pos * dim;

        switch (self.wpe) {
            .f32 => |floats| {
                const embeddings = floats[pos_offset .. pos_offset + dim];
                @memcpy(out, embeddings);
            },
            else => {
                std.debug.print("Unsupported token embedding format: {any}\n", .{
                    @as(math.WeightFormat, self.wpe),
                });
                @panic("Unsupported token embedding quantization");
            },
        }
    }

    /// Apply causal multi-head attention with key-value caching.
    fn attention(self: Self, state: *State, n: usize, layer: Block, n_token: usize) void {
        const c = self.config;

        // Pass input through q, k, and v, weights, which are the `c_attn` layer in the original
        // model.
        //
        // qkv_w = c_attn (weight)
        // qkv_b = c_attn (bias)
        // q, k, v = c_attn(x)
        const q_start = 0 * c.n_embed * c.n_embed;
        const k_start = 1 * c.n_embed * c.n_embed;
        const v_start = 2 * c.n_embed * c.n_embed;
        const end = 3 * c.n_embed * c.n_embed;

        const input: Weights = .{ .f32 = state.work1 };

        const wq = layer.qkv_w.f32[q_start..k_start];
        const bq = layer.qkv_b.f32[0..c.n_embed];
        math.matrixMulVec(f32, state.q, .{ .f32 = wq }, input, c.n_embed, c.n_embed);
        math.add(f32, state.q, bq, state.q);

        const wk = layer.qkv_w.f32[k_start..v_start];
        const bk = layer.qkv_b.f32[c.n_embed .. 2 * c.n_embed];
        math.matrixMulVec(f32, state.k, .{ .f32 = wk }, input, c.n_embed, c.n_embed);
        math.add(f32, state.k, bk, state.k);

        const wv = layer.qkv_w.f32[v_start..end];
        const bv = layer.qkv_b.f32[2 * c.n_embed .. 3 * c.n_embed];
        math.matrixMulVec(f32, state.v, .{ .f32 = wv }, input, c.n_embed, c.n_embed);
        math.add(f32, state.v, bv, state.v);

        // done with state.work1. It can safely be modified now
        // Finished w/ c_attn projection, now apply attention w/ softmax

        const layer_offset = n * c.n_ctx * c.n_embed;
        {
            // Update kv cache
            // Write through to KV cache since we are done calculating KV for the current token
            const cache_start = layer_offset + n_token * c.n_embed;
            const cache_key_vec = state.k_cache[cache_start .. cache_start + c.n_embed];
            const cache_val_vec = state.v_cache[cache_start .. cache_start + c.n_embed];
            @memcpy(cache_key_vec, state.k);
            @memcpy(cache_val_vec, state.v);
        }

        // Scaling factor which is equivalent to sqrt(d_k) where d_k is the dimensionality of
        // each head, or the size of each head.
        const head_size = c.n_embed / c.n_heads;
        const attention_scale = std.math.sqrt(@as(f32, @floatFromInt(head_size)));

        // a = multihead_attn(q, k ,v)
        //
        // Apply multi-head attention
        for (0..c.n_heads) |head| {
            // Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
            // Where d_k is the dimension (size) of each head.
            const query = state.q[head * head_size ..][0..head_size];
            const att = state.attention[head * c.n_ctx ..][0 .. n_token + 1];

            // Calculate QK^T / sqrt(d_k) for each token (including the next one)
            const base = layer_offset + head * head_size;
            for (0..n_token + 1) |tok| {
                const key = state.k_cache[base + tok * c.n_embed ..][0..head_size];

                const qk = math.dotProduct(query, key);
                const inner = qk / attention_scale;
                att[tok] = inner;
            }

            // Calculate softmax(QK^T/sqrt(d_k))
            math.softMax(att);

            // We now have the softmax(QK^T/sqrt(d_k)).
            // Need to multiply by value vector.
            var attn_tmp: []f32 = state.work1[head * head_size .. (head + 1) * head_size];
            @memset(attn_tmp, 0);

            // Multiply softmax(QK^T/sqrt(d_k)) by the values (V)
            // This gives us the attention for each QKV pairing
            for (0..n_token + 1) |tok| {
                const value = state.v_cache[base + tok * c.n_embed ..][0..head_size];
                const tok_attention = att[tok];

                for (0..head_size) |j| {
                    attn_tmp[j] += tok_attention * value[j];
                }
            }
        }

        // Put raw attention through output projection
        // a = conv1d(a, 'c_proj', n_state)
        const cproj_w = layer.c_proj_w;
        const cproj_b = layer.c_proj_b;
        const attn = state.work1;

        math.matrixMulVec(f32, state.work2, cproj_w, .{ .f32 = attn }, c.n_embed, c.n_embed);
        math.add(f32, state.work2, state.work2, cproj_b.f32);
    }

    /// Feed input through a multi-layer perceptron for a layer.
    fn mlp(self: Self, state: *State, layer: Block) void {
        // Current input is stored in `state.layer_out`
        const x = state.layer_out;

        const dim = self.config.n_embed;
        const dim4 = dim * 4;

        // h = gelu(conv1d(x, 'c_fc', n_state))
        // Go through `c_fc`
        // inside conv1d(x, 'c_fc')
        math.matrixMulVec(f32, state.fat_work, layer.mlp_c_fc_w, .{ .f32 = x }, dim4, dim);
        math.add(f32, state.fat_work, state.fat_work, layer.mlp_c_fc_b.f32);
        // done with conv1d(...)

        // Perform GELU Activation
        // gelu(...)
        math.geluApprox(f32, state.fat_work);

        // Go through `c_proj`
        // h2 = conv1d(h, 'c_proj', nx)
        // sic, (rows, cols) is reversed from earlier
        math.matrixMulVec(f32, state.layer_out, layer.mlp_c_proj_w, .{ .f32 = state.fat_work }, dim, dim4);
        math.add(f32, state.layer_out, state.layer_out, layer.mlp_c_proj_b.f32);
        // done with conv1d(...)

        // Current wave is in state.layer_out.
    }
};

pub fn main() !void {
    const path = "gpt2-f32.gguf";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("Loading GPT2 model from {s}\n", .{path});

    var context = try Gpt2Context.init(path, alloc);
    defer context.deinit();
    std.debug.print("GPT2 model loaded\n", .{});

    const c = context.config;
    std.debug.print("vocab: {d}; context len: {d}; embed: {d}; heads: {d}; layers: {d}\n", .{
        c.n_vocab,
        c.n_ctx,
        c.n_embed,
        c.n_heads,
        c.n_layers,
    });

    var sampler = llm.sample.Sampler.init(0.95, 0.90, context.config.n_vocab);

    const prompt_str = "Wikipedia the free online encyclopedia that";
    var generated = std.ArrayList(Tokenizer.Token).init(alloc);
    defer generated.deinit();
    try generated.append(context.config.bos_id.?);
    {
        const tks = try context.tokenizer.encode(prompt_str, alloc);
        defer alloc.free(tks);
        try generated.appendSlice(tks);
    }

    std.debug.print("Performing dummy inference\n", .{});
    var tok: tkn.TikTokenizer.Token = undefined;
    const seed_len = generated.items.len;
    for (0..120) |i| {
        if (i < seed_len) {
            tok = generated.items[i];
        }
        const input = tok;

        const out = context.transformer.forward(&context.state, tok, i);
        tok = @intCast(try sampler.sample(out, alloc));

        if (i + 1 >= seed_len) {
            try generated.append(tok);
        }
        const ser = try context.tokenizer.decode(generated.items, alloc);
        defer alloc.free(ser);
        std.debug.print("out {d: >2}: {d} → {d} -- {s}\n", .{ i, input, tok, ser });
    }
    std.debug.print("Done with dummy inference\n", .{});
}
