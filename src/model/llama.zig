// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! llama: This module contains an implementation of the LLaMA 2 LLM.

const std = @import("std");

const llm = @import("../root.zig");
const ggml = llm.ggml;
const math = llm.math;
const model = llm.model;
const tkn = llm.token;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const Tokenizer = tkn.SPTokenizer;
const Token = Tokenizer.Token;

const Weights = math.Weights;

pub const Error = error{
    /// You attempted to load a file that has an invalid format. Usually this means the file had
    /// a bad header or the wrong version. This could also mean that the file is truncated or has
    /// extra data at the end.
    BadFormat,
    /// You tried to load a file that has a correct format but is semantically incorrect. Usually
    /// this means you tried to load the wrong model.
    BadFile,
};

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    max_seq_length: usize,

    shared_classifier: bool,
    quantized: bool = false,

    /// Read the configuration from a GGUF file.
    pub fn readGGUF(file: ggml.GGUFFile) Config {
        // TODO: Do something more resilient to detect quantization.
        const quantized = file.fileType().? == .MOSTLY_Q8_0;
        return .{
            .dim = file.getValue("llama.embedding_length").?.uint32,
            .hidden_dim = file.getValue("llama.feed_forward_length").?.uint32,
            .n_layers = file.getValue("llama.block_count").?.uint32,
            .n_heads = file.getValue("llama.attention.head_count").?.uint32,
            .n_kv_heads = file.getValue("llama.attention.head_count_kv").?.uint32,
            .vocab_size = file.getValue("tokenizer.ggml.tokens").?.array.len,
            .max_seq_length = file.getValue("llama.context_length").?.uint32,
            .shared_classifier = file.getValue("output.weight") == null,
            .quantized = quantized,
        };
    }

    /// Temporarily open the checkpoint file to read only the header.
    /// Do not leave the file open because we will mmap it.
    pub fn read(model_path: []const u8) !Config {
        // First try to load the model with relative and then fallback to absolute path if that fails.
        const options: std.fs.File.OpenFlags = .{ .mode = .read_only };
        const cwd = std.fs.cwd();

        // zig fmt: off
        const file: ?std.fs.File = cwd.openFile(model_path, options)
            // Fall back to absolute path
            catch (std.fs.openFileAbsolute(model_path, options) catch null);
        // zig fmt: on

        if (file == null) {
            std.debug.print("Could open model file: {s}\nIs the path correct?\n", .{model_path});
            return std.fs.File.OpenError.FileNotFound;
        }
        defer file.?.close();

        var buffer = std.io.bufferedReader(file.?.reader());
        return try Config.read_inner(buffer.reader());
    }

    const magic_v1_str = "ak42";
    const magic_v1_num = std.mem.readInt(u32, magic_v1_str, .little);

    /// Read a "Version 1" `llama.bin` file as exported by `llama2.c/export.py`
    fn read_inner(reader: anytype) !Config {
        // Assume the machine which exports the v1 file is Little-Endian, which make parsing
        // easier.
        // First, we have the header
        const magic = try reader.readInt(u32, .little);
        if (magic != magic_v1_num) {
            return Error.BadFormat;
        }
        // Next, a signed integer for the export version
        const version = try reader.readInt(i32, .little);
        if (version != 1) {
            return error.BadFormat;
        }
        // The next 7 values are 32 bit signed numbers
        const dim = try reader.readInt(i32, .little);
        const hidden_dim = try reader.readInt(i32, .little);
        const n_layers = try reader.readInt(i32, .little);
        const n_heads = try reader.readInt(i32, .little);
        const n_kv_heads = try reader.readInt(i32, .little);
        const vocab_size = try reader.readInt(i32, .little);
        // There's a bug in the `export.py` method. It actually hard-codes the sequence length to
        // 2048 for V1 which is incorrect. That is at most half of the actual sequence length.
        const max_seq_length = try reader.readInt(i32, .little) * 2;

        // Next byte indicates if the token embeddings are shared between the last layer of the
        // model and the tokenizer embeddings.
        const shared_classifier = try reader.readByte() == 1;

        // And that's the end of the header.

        return Config{
            .dim = @intCast(dim),
            .hidden_dim = @intCast(hidden_dim),
            .n_layers = @intCast(n_layers),
            .n_heads = @intCast(n_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .vocab_size = @intCast(vocab_size),
            .max_seq_length = @intCast(max_seq_length),
            .shared_classifier = shared_classifier,
        };
    }
};

/// Represents the current state of a transformer.
pub const State = struct {
    arena: ArenaAllocator,

    // Precomputed coefficients
    sin: []f32,
    cos: []f32,

    // x
    input: []f32,
    // Working state
    //work: []f32,
    work: []f32,
    // Working state overflow
    work2: []f32,
    // hb
    hidden1: []f32,
    // hb2
    hidden2: []f32,

    // quantized weight vector in dimension `dim`
    quant_vec: Weights,
    quant_vec2: Weights,

    q: []f32,
    k: []f32,
    v: []f32,

    attention: []f32,
    // logits
    output: []f32,

    // caches
    // (layer, seq_len, dim)
    k_cache: []f32,
    v_cache: []f32,

    /// Initialize the working state.
    /// Adapted from Andrej Karpathy's `llama2.c`
    pub fn init(alloc: Allocator, config: Config) !State {
        var arena = ArenaAllocator.init(alloc);
        var a = arena.allocator();

        const kv_dim = config.dim * config.n_kv_heads / config.n_heads;

        const x = try a.alloc(f32, config.dim);
        const xb = try a.alloc(f32, config.dim);
        const xb2 = try a.alloc(f32, config.dim);
        const hb1 = try a.alloc(f32, config.hidden_dim);
        const hb2 = try a.alloc(f32, config.hidden_dim);

        const q = try a.alloc(f32, config.dim);
        const k = try a.alloc(f32, kv_dim);
        const v = try a.alloc(f32, kv_dim);

        const k_cache = try a.alloc(f32, config.n_layers * config.max_seq_length * kv_dim);
        const v_cache = try a.alloc(f32, config.n_layers * config.max_seq_length * kv_dim);

        const att = try a.alloc(f32, config.n_heads * config.max_seq_length);
        const logits = try a.alloc(f32, config.vocab_size);

        const sin, const cos = try precompute_frequencies(
            config.dim,
            config.n_heads,
            config.max_seq_length,
            a,
        );

        const Block = math.quant.Block(.q8_0);
        const quant_len = math.quant.blockUnitLen(Block);
        const quant_count = if (config.quantized) config.dim / quant_len else 0;
        const quant_vec = try a.alloc(Block, quant_count);
        const hidden_quant_count = if (config.quantized) config.hidden_dim / quant_len else 0;
        const quant_vec2 = try a.alloc(Block, hidden_quant_count);

        return .{
            // Internal memory lifetime
            .arena = arena,
            // Precomputed
            .sin = sin,
            .cos = cos,
            // layers and temporaries
            .input = x,
            .work = xb,
            .work2 = xb2,
            .hidden1 = hb1,
            .hidden2 = hb2,
            .q = q,
            .k = k,
            .v = v,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .attention = att,
            .output = logits,

            .quant_vec = .{ .q8_0 = quant_vec },
            .quant_vec2 = .{ .q8_0 = quant_vec2 },
        };
    }

    /// Deinitialize the state of this Transformer.
    pub fn deinit(self: *@This()) void {
        self.arena.deinit();
    }
};

/// Precompute frequencies for RoPE (Rotary Position Embeddings).
/// Caller owns the returned memory.
///
/// Returns two slices of coefficients corresponding to the `cos(m θ_n)` or `sin(m θ_n)` values
/// within the element-wise product of R^d_{Θ,m}x found in section 3.4.2 of the RoFormer paper [1].
/// Note that the implementation here uses a different definition of θ_i than the original paper.
///
/// Based on `precompute_freqs_cis` from Meta's LLaMA.
/// [1]: https://arxiv.org/abs/2104.09864v5
fn precompute_frequencies(
    dim: usize,
    n_heads: usize,
    len: usize,
    allocator: Allocator,
) !struct { []f32, []f32 } {
    const head_size = dim / n_heads;
    std.debug.assert(head_size & 1 == 0); // `head_size` must be divisible by 2.
    // We only compute half the number of coefficients in a head of `head_size` because RoPE
    // collapses a head of size d → d/2.
    const half_head = head_size / 2;
    const compute_len = len * half_head;

    var sin = try allocator.alloc(f32, compute_len);
    errdefer allocator.free(sin);
    var cos = try allocator.alloc(f32, compute_len);
    errdefer allocator.free(cos);

    //std.debug.print("dim: {d}; len: {d}; compute: {d}\n", .{ dim, len, compute_len });
    //std.debug.print("heads: {d}; head_size: {d}; prod {d}\n", .{ n_heads, head_size, n_heads * head_size });

    const head_size_f: f32 = @floatFromInt(head_size);

    for (0..len) |m| {
        const m_f: f32 = @floatFromInt(m);
        for (0..half_head) |hd| {
            // The cosine embedding for token #m at head position #h is a function of the position
            // on the head (h) and the token (m)
            // m_cos(m, h) = cos(m * theta)
            //             = cos(m * pow(10_000 * -1 * pow))
            //             = cos(m * pow(10_000 * -1 * (h / head_size)))
            // and likewise with m_sin(n, h)

            // Recover original head position by doubling since we are iterating over half of
            // the head size.
            const h_f: f32 = @floatFromInt(hd * 2);

            const pow = h_f / head_size_f;
            const theta = std.math.pow(f32, 10_000, -1 * pow);
            const m_theta = m_f * @as(f32, @floatCast(theta));

            const m_cos = std.math.cos(m_theta);
            const m_sin = std.math.sin(m_theta);

            const i = half_head * m + hd;

            sin[i] = m_sin;
            cos[i] = m_cos;
        }
    }
    return .{ sin, cos };
}

test "check precompute_frequencies static" {
    // Precomputed coefficients as exported into "v0" llama2.c model file for llama2 7b.
    const cos_file = @embedFile("../assets/cos.json");
    const sin_file = @embedFile("../assets/sin.json");

    const cos = try std.json.parseFromSlice([]f32, std.testing.allocator, cos_file, .{});
    defer cos.deinit();
    const sin = try std.json.parseFromSlice([]f32, std.testing.allocator, sin_file, .{});
    defer sin.deinit();

    const len = 2048;
    const sin_actual, const cos_actual = try precompute_frequencies(4096, 32, len, std.testing.allocator);
    defer std.testing.allocator.free(sin_actual);
    defer std.testing.allocator.free(cos_actual);

    try std.testing.expectEqual(sin.value.len, sin_actual.len);
    try std.testing.expectEqual(cos.value.len, cos_actual.len);

    const epsilon = 9e-3; // 1e-4 is too strict and fails
    for (0..cos.value.len, cos.value, cos_actual) |i, exp, actual| {
        if (!std.math.approxEqAbs(f32, exp, actual, epsilon)) {
            const diff = @abs(exp - actual);
            std.debug.print("Failure at cos index {d}: difference {d} exceeded {d}\n", .{ i, diff, epsilon });
            try std.testing.expectApproxEqAbs(exp, actual, epsilon);
        }
    }

    for (0..sin.value.len, sin.value, sin_actual) |i, exp, actual| {
        if (!std.math.approxEqAbs(f32, exp, actual, epsilon)) {
            const diff = @abs(exp - actual);
            std.debug.print("Failure at sin index {d}: difference {d} exceeded {d}\n", .{ i, diff, epsilon });
            try std.testing.expectApproxEqAbs(exp, actual, epsilon);
        }
    }
}

const Complex = std.math.Complex(f32);

/// Apply RoPE embeddings to a vector.
/// Accepts as argument the vector to apply embeddings to as well as the pre-computed slices of
/// sin(mθ_i) and cos(mθ_i) as provided by `precompute_frequencies`.
///
/// Expects to receive parameters about the transformer model such as the number of heads `n_heads`,
/// the size of the head to operate on `head_size`, and the current token position `n`.
fn apply_rope_embeddings(
    vector: []f32,
    sin: []const f32,
    cos: []const f32,
    n_heads: usize,
    head_size: usize,
    n: usize,
) void {
    // Base index of RoPE embedding coefficients
    const base = n * head_size / 2;

    for (0..n_heads) |hi| {
        var hd: usize = 0;
        while (hd < head_size) : (hd += 2) {
            const ii = base + (hd / 2);
            const vi = hi * head_size + hd;

            // Rotate each unit pair v ∈ vector by mθ_i
            //
            // Rotation is done by representing the point v = (v0, v1) to rotate in
            // complex form and then representing the angle to rotate mθ_i as a pre-computed
            // rotation matrix in complex form.

            // Find point (v0, v1) to rotate
            const v0 = vector[vi];
            const v1 = vector[vi + 1];
            const v = Complex{ .re = v0, .im = v1 };

            // The rotating Complex num by angle by mθ_i is equivalent to matrix multiplication:
            // [[cos mθ_i   -sin mθ_i]  *  [v0   v1]^T
            //  [sin mθ_i    cos mθ_i]]
            //  = [(v0*cos mθ_i - v1*sin mθ_i)  (v0*sin mθ_i  + v1*cos mθ_i)]^T
            //
            // Which is basically just multiplying the two complex values:
            //  (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
            //
            // So if a = cos mθ_i, b = sin mθ_i, c = v0, d = v1,
            //
            //  (cos mθ_i  + i sin(mθ_i)) * (v0 + i v1)
            //  = (v0*cos(mθ_i) - v1*sin mθ_i) + i(v1*cos mθ_i + v0*sin mθ_i)
            const m_sin = sin[ii];
            const m_cos = cos[ii];
            const rot = Complex{ .re = m_cos, .im = m_sin };

            const v_rot = v.mul(rot);

            // zig fmt: off
            vector[vi]     = v_rot.re;
            vector[vi + 1] = v_rot.im;
            // zig fmt: on
        }
    }
}

/// Represents a single `TransformerBlock` of the Llama V1/V2 transformer.
/// This method contains all the weights for a single "layer" of a model's layers.
pub const TransformerBlock = struct {
    attn_norm: Weights,
    wq: Weights,
    wk: Weights,
    wv: Weights,
    wo: Weights,

    ffn_norm: Weights,
    w1: Weights,
    w2: Weights,
    w3: Weights,

    /// Initialize a transformer block with the provided backing `file` `mmap(2)` using config
    /// if this is `n`-th layer.
    pub fn initGGUF(file: ggml.GGUFFile, _: Config, n: usize) TransformerBlock {
        // TODO: Assert that weights have correct sizes w/ passed-in config
        const attn_norm = getTensor("attn_norm", file, n);
        const wq = getTensor("attn_q", file, n);
        const wk = getTensor("attn_k", file, n);
        const wv = getTensor("attn_v", file, n);
        const wo = getTensor("attn_output", file, n);

        const ffn_norm = getTensor("ffn_norm", file, n);
        const w1 = getTensor("ffn_gate", file, n);
        const w2 = getTensor("ffn_down", file, n);
        const w3 = getTensor("ffn_up", file, n);

        return .{
            .attn_norm = attn_norm,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .ffn_norm = ffn_norm,
            .w1 = w1,
            .w2 = w2,
            .w3 = w3,
        };
    }

    /// Get the contents of a tensor named `blk.<n>.<name>.weight` from the loaded GGUF `file`.
    /// Assumes that the backing tensor exists within the GGUF file.
    fn getTensor(name: []const u8, file: ggml.GGUFFile, n: usize) Weights {
        std.debug.assert(n < 32);
        var buf = [_]u8{0} ** 64;
        const full_name = std.fmt.bufPrint(&buf, "blk.{d}.{s}.weight", .{ n, name }) catch unreachable;
        if (file.getTensorWeights(full_name)) |tensor| {
            return tensor;
        }
        std.debug.print("Error: Tensor {s} does not exist in the GGUF file", .{full_name});
        @panic("Could not load tensor");
    }
};

/// The weights of a Llama model - specifically the transformer bits.
///
/// This struct is immutable and represents the contents of a loaded model from a file on storage.
/// The file is read into memory with `mmap(2)`.
pub const TransformerV1 = struct {
    const Self = @This();
    const page_size_min = std.heap.page_size_min;
    const header_len = 256;

    // Internally managed resources
    fd: std.posix.fd_t,
    ptr: ?[]align(page_size_min) u8,
    arena: ArenaAllocator,

    // Config weights
    config: Config,

    // Embeddings
    token_embed: Weights,
    // Transformer layers
    layers: []TransformerBlock,
    // Output norms
    norm: Weights,
    // Only set whenever no shared classifier layer with tokenizer
    classifier: Weights,

    /// Initialize a transformer from a `.bin` file exported by the *V1 Export* of `llama2.c`'s
    /// [export.py script](https://github.com/karpathy/llama2.c/blob/master/export.py).
    ///
    /// Refer to this project's `README.md` for details on how to generate one.
    pub fn initV1(model_path: []const u8, config: Config, alloc: Allocator) !TransformerV1 {
        //std.debug.print("Opening Transformer model\n", .{});
        const fd = try std.posix.open(model_path, .{}, 0o440);
        errdefer std.posix.close(fd);

        // Here we mmap() the weights files because nobody wants to open up a 25 GB file raw!
        const stat = try std.posix.fstat(fd);
        const fsize: u64 = @intCast(stat.size);
        std.debug.print(
            "Model size: {d:.1} MiB\n",
            .{@as(f32, @floatFromInt(fsize)) / 1048576.0},
        );

        // See `std.os.linux.MAP` for more info.
        const mmap_type: std.posix.MAP = .{
            .TYPE = .SHARED,
            // Linux-only flags
            // Don't reserve swap pages
            .NORESERVE = true,
            // Try to populate all of the pages in memory (pre-read) if possible
            .POPULATE = true,
        };

        const ptr = try std.posix.mmap(null, fsize, std.posix.PROT.READ, mmap_type, fd, 0);
        errdefer std.posix.munmap(ptr);

        // Tell the kernel we expect to do sequential reads and that we will need the
        // mapped file in the near future.
        //
        // When measuring this it makes layer times more consistent, especially for the first
        // few iterations of `Transformer.forward()`
        const madvise_flags = std.posix.MADV.SEQUENTIAL | std.posix.MADV.WILLNEED;
        try std.posix.madvise(ptr.ptr, fsize, madvise_flags);

        // V1 Export Format from `llama2.c`
        // The first 256 bytes contain the header with trailing 0 padding.

        // TODO: Load the `Config` header here instead of in a separate discrete function.

        // We have the ptr. Time to handle it.
        const total_len: usize = @as(usize, @intCast(fsize)) - header_len;
        const total_elems = total_len / @sizeOf(f32);

        const raw_ptr: [*]align(4) f32 = @ptrFromInt(@intFromPtr(&ptr[header_len]));
        const weights = raw_ptr[0..total_elems];

        const vocab = config.vocab_size;
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const n_layers = config.n_layers;
        const n_heads = config.n_heads;
        const n_kv_heads = config.n_kv_heads;
        const head_size = dim / config.n_heads;

        // According to the output from the modified `export.py` file, we have these dimensions:
        // layers: 32 (known from config)
        var i: usize = 0;

        // normalization layers (7b)
        // attention:   [4096]f32
        // ffn_norm:    [4096]f32
        // norm:        [4096]f32
        const rms_attention = weights[i .. i + n_layers * dim];
        i += rms_attention.len;
        const ffn_norm = weights[i .. i + n_layers * dim];
        i += ffn_norm.len;
        const norms = weights[i .. i + dim];
        i += norms.len;

        // token embeddings (7b)
        // embed:       [vocab_size][4096]f32
        const token_embed = weights[i .. i + vocab * dim];
        i += token_embed.len;

        // attention layers (7b)
        // wq:          [4096][4096]f32
        // wk:          [4096][4096]f32
        // wv:          [4096][4096]f32
        // wo:          [4096][4096]f32

        const wq = weights[i .. i + n_layers * dim * (n_heads * head_size)];
        i += wq.len;
        const wk = weights[i .. i + n_layers * dim * (n_kv_heads * head_size)];
        i += wq.len;
        const wv = weights[i .. i + n_layers * dim * (n_kv_heads * head_size)];
        i += wv.len;
        const wo = weights[i .. i + n_layers * (n_heads * head_size) * dim];
        i += wo.len;

        // ff layers (7b)
        // w1:          [11008][4096]f32
        // w2:          [4096][11008]f32
        // w3:          [11008][4096]f32
        // output:      [vocab_size][4096]f32
        const w1 = weights[i .. i + n_layers * dim * hidden_dim];
        i += w1.len;
        const w2 = weights[i .. i + n_layers * hidden_dim * dim];
        i += w2.len;
        const w3 = weights[i .. i + n_layers * dim * hidden_dim];
        i += w3.len;

        var out_classifier: ?[]f32 = null;
        if (config.shared_classifier) {
            out_classifier = token_embed;
        } else {
            out_classifier = weights[i .. i + dim * vocab];
            i += out_classifier.?.len;
        }

        // make sure we read the whole file.
        const used_size = i * @sizeOf(f32) + header_len;
        if (fsize != used_size) {
            return Error.BadFile;
        }

        // Convert weight slices to `TransformerBlock`s.
        const dim2 = dim * dim;
        const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        const kv_len = dim * kv_dim;

        var arena = ArenaAllocator.init(alloc);
        errdefer arena.deinit();
        var allocator = arena.allocator();

        var layers = try allocator.alloc(TransformerBlock, config.n_layers);
        errdefer allocator.free(layers);
        for (0..config.n_layers) |n| {
            const l_attn_norm = rms_attention[n * dim .. (n + 1) * dim];

            const n_dim2 = n * dim2;
            const l_wq = wq[n_dim2 .. n_dim2 + dim2];
            const l_wk = wk[n * kv_len .. (n + 1) * kv_len];
            const l_wv = wv[n * kv_len .. (n + 1) * kv_len];
            const l_wo = wo[n * dim2 .. (n + 1) * dim2];

            const ffn_o = n * dim * config.hidden_dim;
            const ffn_e = ffn_o + dim * config.hidden_dim;

            const l_ffn_norm = ffn_norm[n * dim .. (n + 1) * dim];
            const l_w1 = w1[ffn_o..ffn_e];
            const l_w2 = w2[ffn_o..ffn_e];
            const l_w3 = w3[ffn_o..ffn_e];

            layers[n] = TransformerBlock{
                .attn_norm = .{ .f32 = l_attn_norm },
                .wq = .{ .f32 = l_wq },
                .wk = .{ .f32 = l_wk },
                .wv = .{ .f32 = l_wv },
                .wo = .{ .f32 = l_wo },

                .ffn_norm = .{ .f32 = l_ffn_norm },
                .w1 = .{ .f32 = l_w1 },
                .w2 = .{ .f32 = l_w2 },
                .w3 = .{ .f32 = l_w3 },
            };
        }

        return TransformerV1{
            .config = config,
            .fd = fd,
            .ptr = ptr,
            .arena = arena,

            .token_embed = .{ .f32 = token_embed },
            .layers = layers,
            .norm = .{ .f32 = norms },
            .classifier = .{ .f32 = out_classifier.? },
        };
    }

    /// Unmap any memory and free any resources used by this `Transformer`.
    pub fn deinit(self: *Self) void {
        std.debug.print("Transformer.deinit()\n", .{});
        if (self.ptr) |inner| std.posix.munmap(inner);
        if (self.fd >= 0) std.posix.close(self.fd);

        self.arena.deinit();
        self.ptr = null;
        self.fd = -1;
    }

    /// Calculate a forward pass of the transformer with the next token `token` at
    /// position `n_tok`.
    ///
    /// Returns a slice of logits from calculation. Caller **does not** down the returned
    /// slice and should not attempt to free it.
    pub fn forward(self: Self, state: *State, tok: Token, n_tok: usize, progress: ?std.Progress.Node) []f32 {
        // Get a considerable speedup for operations that occur within here.
        @setFloatMode(.optimized);

        const c = self.config;

        const dim = c.dim;
        const kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
        //const kv_mul = c.n_heads / c.n_kv_heads;
        const head_size = c.dim / c.n_heads;

        self.applyTokenEmbedding(state.input, tok);

        var layer_progress: ?std.Progress.Node = null;
        if (progress) |prog| {
            layer_progress = prog.start("Layer", c.n_layers);
        }
        defer if (layer_progress) |prog| prog.end();

        for (0..c.n_layers) |i| {
            // Layer offset for caching
            const layer_offset = i * c.max_seq_length * kv_dim;

            // Each "layer" is a TransformerBlock.

            // In TransformerBlock.__init__():
            //     attention = Attention(...)
            //     attention_norm = RMSNorm(...)
            //     ffn_norm = RMSNorm(...)
            //     feed_forward = FeedForward(dim, hidden_dim, ...)
            //
            // Each TransformerBlock.forward(x):
            //     h = x + attention.forward(attention_norm(input, freq_cs, freq_ss))
            //     out = h + feed_forward.forward(ffn_norm(h))
            //     return out
            //
            // In FeedForward.__init__():
            //     w1 = Linear(dim, hidden_dim, bias=False)
            //     w2 = Linear(hidden_dim, dim, bias=False)
            //     w3 = Linear(dim, hidden_dim, bias=False)
            // In Feedforward.forward(x):
            //     dropout(w1(F.silu(
            //                w1(x) * w3(x)
            //            )))
            // where F.silu = torch.nn.functional.silu
            // which is silu(x) = x (elementwise *) σ(x), where σ(x) is logistic sigmoid
            //
            // In Attention.__init__():
            //     # sets some constants known through `Config` already
            //     # NB: model_parallel_size = 1
            //     # NB: n_local_heads = n_heads
            //     n_rep = n_heads // n_kv_heads
            //     head_dim = dim // n_heads
            //     # NB: `bias=False` for all `Linear` layers here
            //     wq = Linear(dim, n_heads * head_dim)
            //     wk = Linear(dim, n_kv_heads * head_dim)
            //     wv = Linear(dim, n_kv_heads * head_dim)
            //     wo = Linear(n_heads * head_dim, dim)
            //     Ignore dropout
            //
            // In Attention.forward(x, freq_cs, freq_ss):
            //     bsz, seqlen, _ = x.shape
            //     xq, xk, xv = wq(x), wk(x), wv(x)
            //     xq = xq.view(bsz, seqlen, n_local_heads, head_dim)
            //     xk = xq.view(bsz, seqlen, n_local_kv_heads, head_dim)
            //     xv = xq.view(bsz, seqlen, n_local_kv_heads, head_dim)
            //
            //    # RoPE
            //    xq, xk = apply_rotary_emb(xq, xk, freq_cs, freq_ss)
            //
            //    # Batch MQ attention: expand out keys & values
            //    xk = repeat_kv(xk, n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            //    xv = repeat_kv(xv, n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            //
            //    xq = xq.transpose(1, 2)    # (bs, n_local_heads, seqlen, head_dim)
            //    xk = xk.transpose(1, 2)
            //    xv = xv.transpose(1, 2)
            //
            //    # Then flash attention or manual impl.
            //    # Manual attention:
            //    scores = matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
            //    scores = scores + self.mask[:, :, :seqlen, :seqlen]
            //    scores = F.softmax(scores.float, dim=-1).type_as(xq)
            //    output = matmul(scores, xv)
            //
            //    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            //
            //    return wo(output)

            const layer = self.layers[i];

            // TransformerBlock.forward()
            // first handle the `attention_norm` call.
            math.rmsNorm(state.work, state.input, layer.attn_norm.f32);

            // Now handle attention matrix multiplies
            if (c.quantized) {
                _ = math.quantize(.q8_0, state.work, state.quant_vec.q8_0) catch
                    @panic("Quantization error");
            }
            const wts: Weights = if (c.quantized)
                state.quant_vec
            else
                Weights{ .f32 = state.work };
            // xq = wq(x)
            math.matrixMulVec(f32, state.q, layer.wq, wts, dim, dim);
            // xk = wk(x)
            math.matrixMulVec(f32, state.k, layer.wk, wts, kv_dim, dim);
            // xv = wv(x)
            math.matrixMulVec(f32, state.v, layer.wv, wts, kv_dim, dim);

            // RoPE
            //     xq, xk = apply_rotary_emb(xq, xk, freq_cs, freq_ss)
            //
            // Apply RoPE embeddings on `q` over `dim` and `k` over `kv_dim`.
            apply_rope_embeddings(state.q, state.sin, state.cos, c.n_heads, head_size, n_tok);
            apply_rope_embeddings(state.k, state.sin, state.cos, c.n_kv_heads, head_size, n_tok);

            {
                // Update kv cache
                std.debug.assert(dim == kv_dim);
                const cache_start = layer_offset + n_tok * kv_dim;
                const cache_key_vec = state.k_cache[cache_start .. cache_start + kv_dim];
                const cache_val_vec = state.v_cache[cache_start .. cache_start + kv_dim];
                @memcpy(cache_key_vec, state.k);
                @memcpy(cache_val_vec, state.v);
            }

            // Inputs are state.q, state.k, state.v
            // Output is in state.work
            // Everything else can be mangled
            self.attention(state, i, n_tok);
            // Output of multi-head attention is now in `state.work`.

            // Almost done w/ Attention.forward(x), we just need to calculate the return
            // statement:
            //     return wo(output)
            if (c.quantized) {
                _ = math.quantize(.q8_0, state.work, state.quant_vec.q8_0) catch
                    @panic("Issue quantizing after attention");
            }
            math.matrixMulVec(f32, state.work2[0..dim], layer.wo, wts, dim, dim);
            // End of Attention.forward(x);

            // We are back in TransformerBlock.forward(x, freq_cs, freq_ss). We just need to add
            // the vector we stored in `state.work2` and add it to `x` then do a feed forward pass:
            //
            // def forward(self, x, few_cs, freq_ss):
            //     h = x + self.attention.forward(self.attention_norm(x), freqs_cs, freqs_ss)
            //     out = h + self.feed_forward.forward(self.ffn_norm(h))

            math.add(f32, state.input, state.input, state.work2);
            // We are no longer using the input `x` and can now use it as `h`.

            // Calculate ff = self.ffn_norm(h) = RMSNorm(h, feed_forward)
            math.rmsNorm(state.work, state.input, layer.ffn_norm.f32);

            // Calculate FeedForward.forward(x):
            //     return w2( silu(w1(x)) * w3(x) )

            if (c.quantized) {
                _ = math.quantize(.q8_0, state.work, state.quant_vec.q8_0) catch
                    @panic("Issue quantizing before hidden layers");
            }

            // hid1 = w1(x)
            math.matrixMulVec(f32, state.hidden1, layer.w1, wts, c.hidden_dim, dim);
            // hid2 = w3(x)
            math.matrixMulVec(f32, state.hidden2, layer.w3, wts, c.hidden_dim, dim);

            // Calculate SwiGLU
            math.swiglu(state.hidden1);
            math.elementProduct(state.hidden1, state.hidden1, state.hidden2);

            // w2 * (swiglu(w1(x)) * w3(x))
            if (c.quantized) {
                _ = math.quantize(.q8_0, state.hidden1, state.quant_vec2.q8_0) catch
                    @panic("Error during quantization");
            }
            const wts_hidden: Weights = if (c.quantized) state.quant_vec2 else Weights{ .f32 = state.hidden1 };
            math.matrixMulVec(f32, state.work, layer.w2, wts_hidden, dim, c.hidden_dim);
            // Done with FeedForward.forward(x)

            // Add back `h` to result of FeedForward.forward(x)
            //     out = h + feed_foward.forward(ffn_norm(h))
            //     return out
            math.add(f32, state.input, state.input, state.work);

            // Done with TransformerBlock.forward();
            //std.debug.print("Done with layer {d}/{d} with {d} at {d}\n", .{ i, c.n_layers, token, n_token });

            if (layer_progress) |prog| {
                prog.completeOne();
            }
        }

        // Done with layers
        //     h = self.norm(h)
        math.rmsNorm(state.input, state.input, self.norm.f32);

        // We are doing inference only, so no calculation of cross-entropy is needed
        // Logits are found by feeding `h` (state.input) through a linear layer.
        if (c.quantized) {
            _ = math.quantize(.q8_0, state.input, state.quant_vec.q8_0) catch
                @panic("Error during quantization");
        }
        const norm_output: Weights = if (c.quantized) state.quant_vec else Weights{ .f32 = state.input };
        math.matrixMulVec(f32, state.output, self.classifier, norm_output, c.vocab_size, dim);

        if (progress) |prog| {
            prog.completeOne();
        }

        return state.output;
    }

    /// Apply and copy token embeddings for the current input `token` into the `out` vector.
    fn applyTokenEmbedding(self: Self, out: []f32, tok: Token) void {
        const dim = self.config.dim;
        const token_offset: usize = @as(usize, @intCast(tok)) * dim;

        switch (self.token_embed) {
            .f32 => |floats| {
                const embeddings = floats[token_offset .. token_offset + dim];
                @memcpy(out, embeddings);
            },
            .q8_0 => |quantized| {
                const block_size = math.quant.blockUnitLen(math.quant.Block(.q8_0));
                const block_offset = token_offset / block_size;
                const n_blocks = dim / block_size;

                const embeddings = quantized[block_offset .. block_offset + n_blocks];
                math.dequantize(.q8_0, embeddings, out) catch
                    @panic("Terrible situation in embeddings");
            },
            else => {
                std.debug.print("Unsupported token embedding format: {any}\n", .{
                    @as(math.WeightFormat, self.token_embed),
                });
                @panic("Unsupported token embedding quantization");
            },
        }
    }

    /// Perform attention the current forward layer iteration of the Transformer.
    /// This calculates Attention(state.q, state.k, state.v) and writes the output
    /// into `state.work`.
    fn attention(self: Self, state: *State, layer: usize, n_token: usize) void {
        const c = self.config;

        const head_size = c.dim / c.n_heads;
        const kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
        const kv_mul = c.n_heads / c.n_kv_heads;
        const layer_offset = layer * c.max_seq_length * kv_dim;
        const attention_scale = std.math.sqrt(@as(f32, @floatFromInt(head_size)));

        // Perform multi-head attention over all heads
        for (0..c.n_heads) |head| {
            // Attention is defined in "Attention is All You Need"
            // The formula is:
            //   Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
            // Where d_k is the dimension of the queries and keys, and the values
            // are of dimensions d_v.
            const query = state.q[head * head_size ..][0..head_size];
            const att = state.attention[head * c.max_seq_length ..][0 .. n_token + 1];

            // Calculate QK^T / sqrt(d_k) for each token (including the next one)
            const base = layer_offset + (head / kv_mul) * head_size;
            for (0..n_token + 1) |tok| {
                const key = state.k_cache[base + tok * kv_dim ..][0..head_size];

                const qk = math.dotProduct(query, key);
                const inner = qk / attention_scale;
                att[tok] = inner;
            }

            // Calculate softmax(QK^T/sqrt(d_k))
            math.softMax(att);

            // We now have the softmax(QK^T/sqrt(d_k)).
            // Need to multiply by value vector.
            var attn_tmp: []f32 = state.work[head * head_size .. (head + 1) * head_size];
            @memset(attn_tmp, 0);

            // Multiply softmax(QK^T/sqrt(d_k)) by the values (V)
            // This gives us the attention for each QKV pairing
            for (0..n_token + 1) |tok| {
                const value = state.v_cache[base + tok * kv_dim ..][0..head_size];
                const tok_attention = att[tok];

                for (0..head_size) |j| {
                    attn_tmp[j] += tok_attention * value[j];
                }
            }
            // now has the attention almost ready
            // We calculate the output by multiplying it by `wo`
        }
    }
};

pub const LlamaContext = struct {
    const Self = @This();

    pub const vtable = model.VTable{
        .init = initGeneric,
        .tokenize = tokenize,
        .detokenize = detokenize,
        .to_string = toString,
        .forward = forward,
        .get_info = getInfo,
        .deinit = deinitVirt,
    };

    // Only used for `initGeneric`
    a: ?Allocator,

    config: Config,
    transformer: TransformerV1,
    state: State,
    tokenizer: Tokenizer,
    file: ?ggml.GGUFFile = null,

    /// Initialize and read a new Llama V1 or V2 inference context from a provided GGUF file.
    /// Returns an `Error.BadFile` whenever the file is an unsupported model or quantization
    /// format.
    pub fn initGeneric(file: ggml.GGUFFile, alloc: std.mem.Allocator) llm.model.LoadError!*anyopaque {
        try ensureCorrectModel(file);

        const ret = try alloc.create(LlamaContext);
        errdefer alloc.destroy(ret);

        const config = Config.readGGUF(file);

        // Load Tokenizer
        var tokenizer = try Tokenizer.initGGUF(file, alloc);
        errdefer tokenizer.deinit();

        var state = try State.init(alloc, config);
        errdefer state.deinit();

        // Load Transformer Weights
        const token_embed = try loadWeights(file, "token_embd.weight"); // sic
        const output_norm = try loadWeights(file, "output_norm.weight");
        const output_weight = try loadWeights(file, "output.weight");

        var arena = ArenaAllocator.init(alloc);
        errdefer arena.deinit();

        var arena_alloc = arena.allocator();
        var layers = try arena_alloc.alloc(TransformerBlock, config.n_layers);
        for (0..config.n_layers) |i| {
            layers[i] = TransformerBlock.initGGUF(file, config, i);
        }

        const transformer = TransformerV1{
            .fd = -1,
            .ptr = null,
            .arena = arena,

            .config = config,

            .token_embed = token_embed,
            .layers = layers,
            .norm = output_norm,
            .classifier = output_weight,
        };

        ret.* = .{
            .a = alloc,
            .config = config,
            .transformer = transformer,
            .state = state,
            .tokenizer = tokenizer,
            .file = file,
        };
        return ret;
    }

    /// Initialize and read a new Llama V1 or V2 inference context from a provided GGUF file.
    /// Returns an `Error.BadFile` whenever the file is an unsupported model or quantization
    /// format.
    pub fn init(file_name: []const u8, alloc: std.mem.Allocator) !LlamaContext {
        var file = try ggml.GGUFFile.read_file(file_name, alloc);
        errdefer file.deinit();

        try ensureCorrectModel(file);

        const config = Config.readGGUF(file);

        // Load Tokenizer
        var tokenizer = try Tokenizer.initGGUF(file, alloc);
        errdefer tokenizer.deinit();

        var state = try State.init(alloc, config);
        errdefer state.deinit();

        // Load Transformer Weights
        const token_embed = try loadWeights(file, "token_embd.weight"); // sic
        const output_norm = try loadWeights(file, "output_norm.weight");
        const output_weight = try loadWeights(file, "output.weight");

        var arena = ArenaAllocator.init(alloc);
        errdefer arena.deinit();

        var arena_alloc = arena.allocator();
        var layers = try arena_alloc.alloc(TransformerBlock, config.n_layers);
        for (0..config.n_layers) |i| {
            layers[i] = TransformerBlock.initGGUF(file, config, i);
        }

        const transformer = TransformerV1{
            .fd = -1,
            .ptr = null,
            .arena = arena,

            .config = config,

            .token_embed = token_embed,
            .layers = layers,
            .norm = output_norm,
            .classifier = output_weight,
        };

        return LlamaContext{
            .a = null,
            .config = config,
            .transformer = transformer,
            .state = state,
            .tokenizer = tokenizer,
            .file = file,
        };
    }

    /// Ensure the provided GGUF `file` is correct and that we can actually load it.
    /// Ensures that the loaded file is correct and is in a quantization format we actually
    /// support.
    fn ensureCorrectModel(file: ggml.GGUFFile) !void {
        // No longer require a specific name, just print out and make a best effort attempt to make
        // sure the name has "Llama" in it.
        if (file.getValue(ggml.name_key)) |name| {
            const inner = name.string.str;
            if (std.ascii.indexOfIgnoreCase(inner, "llama") == null) {
                std.debug.print("Model does not report as llama. Found name: {s}\n", .{inner});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Could not find GGML model name: {s}\n", .{ggml.name_key});
            return Error.BadFile;
        }

        if (file.getValue(ggml.arch_key)) |arch| {
            const inner = arch.string.str;
            if (!std.mem.eql(u8, inner, "llama")) {
                std.debug.print("{s} is wrong\n", .{inner});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Could not find model architecture: {s}\n", .{ggml.arch_key});
            return Error.BadFile;
        }

        const tokenizer_key = "tokenizer.ggml.model";
        if (file.getValue(tokenizer_key)) |tok_model| {
            const model_name = tok_model.string.str;
            if (!std.mem.eql(u8, model_name, "llama")) {
                std.debug.print("Model reports to be {s} and not \"llama\"!\n", .{model_name});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Missing tokenizer model {s}\n", .{tokenizer_key});
            return Error.BadFile;
        }

        if (file.fileType()) |ft| {
            if (ft != .ALL_F32 and ft != .MOSTLY_Q8_0) {
                std.debug.print("llama: Unsupported quantization: {}\n", .{ft});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Could not load file type\n", .{});
            return Error.BadFile;
        }
    }

    pub fn forward(ptr: *anyopaque, token: tkn.Token, n_token: usize) []f32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.transformer.forward(&self.state, token, n_token, null);
    }

    pub fn toString(ptr: *anyopaque, token: tkn.Token) ?[]const u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.tokenizer.getTokenChars(token);
    }

    pub fn tokenize(ptr: *anyopaque, str: []const u8, option: tkn.EncodingOption, allocator: std.mem.Allocator) model.RunError![]const tkn.Token {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const result = try self.tokenizer.encode(str, option, allocator);
        return Tokenizer.toGenericTokens(result);
    }

    pub fn detokenize(ptr: *anyopaque, tokens: []const tkn.Token, allocator: std.mem.Allocator) model.RunError![]u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        const as_sp_tokens = Tokenizer.fromGenericTokens(tokens);
        var ret = std.ArrayList(u8).init(allocator);
        errdefer ret.deinit();

        for (as_sp_tokens) |tok| {
            const chars = self.tokenizer.getTokenChars(tok).?;
            try ret.appendSlice(chars);
        }
        return ret.toOwnedSlice();
    }

    pub fn getInfo(ptr: *anyopaque) model.Info {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return .{
            .vocab_size = self.config.vocab_size,
            .context_len = self.config.max_seq_length,
            .add_start = true,
            .start_token = Tokenizer.BOS,
            .add_end = false,
            .end_token = Tokenizer.EOS,
        };
    }

    pub fn contextLen(ptr: *anyopaque) usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.config.max_seq_length;
    }

    pub fn deinit(self: *Self) void {
        self.transformer.deinit();
        self.state.deinit();
        self.tokenizer.deinit();
        if (self.file) |*inner| {
            inner.deinit();
        }
        self.file = null;
        if (self.a) |alloc| {
            alloc.destroy(self);
        }
    }

    pub fn deinitVirt(ptr: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.deinit();
    }
};

fn loadWeights(file: ggml.GGUFFile, name: []const u8) !Weights {
    if (file.getTensorWeights(name)) |wts| {
        return wts;
    }
    std.debug.print("Missing tensor weights in file: {s}\n", .{name});
    return Error.BadFile;
}
