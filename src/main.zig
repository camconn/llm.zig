// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

const std = @import("std");
const Allocator = std.mem.Allocator;

const clap = @import("clap");

const GPA = std.heap.GeneralPurposeAllocator(.{});
const ArenaAllocator = std.heap.ArenaAllocator;
const Complex = std.math.Complex;

pub fn main() !void {
    var gpa = GPA{};
    defer _ = gpa.deinit();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Display this help and exit.
        \\-m, --model <str>      Path to the model to use
        \\
    );

    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = gpa.allocator(),
    }) catch |err| {
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        return clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});
    }

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);

    // TODO: Handle prompt
    const stdout = bw.writer();
    try stdout.print("Loading model config\n", .{});
    try bw.flush();

    var alloc = gpa.allocator();

    const model_path = "/home/cam/dev/llama2.c/llama2-7b.bin";
    const model_path_dupe = try alloc.dupeZ(u8, model_path);
    defer alloc.free(model_path_dupe);
    const config = try read_config(model_path_dupe);
    try stdout.print("loaded config\n", .{});
    // zig fmt: off
    try stdout.print("dim: {d}, hidden: {d}, n_layers: {d}, n_heads: {d}, n_kv: {d}, vocab: {d}, max_seq: {d}, shared_classifier: {}\n", .{
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.max_seq_length,
        config.shared_classifier,
    });
    // zig fmt: on
    try bw.flush();

    const tokenizer_path = "tokenizer.bin";
    const tokenizer_path_dupe = try alloc.dupeZ(u8, tokenizer_path);
    defer alloc.free(tokenizer_path_dupe);
    var tokenizer = try load_tokenizer(tokenizer_path_dupe, alloc, config.vocab_size);
    defer tokenizer.deinit();
    try stdout.print("loaded tokenizer\nmax tokenizer length: {d}\n", .{tokenizer.max_len});
    try bw.flush();

    // TODO: Load weights
    var transformer = try TransformerV1.init(model_path_dupe);
    defer transformer.deinit();
    try stdout.print("Done loading model...\n", .{});
    try bw.flush();

    try stdout.print("Done with work.\n", .{});
    try stdout.print("\n\n", .{});
    try bw.flush();

    try stdout.print("Cleaning up\n", .{});
    try bw.flush();
}

const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    max_seq_length: usize,

    shared_classifier: bool,

    /// Read a "Version 1" `llama.bin` file as exported by `llama2.c/export.py`
    fn read(reader: anytype) !Config {
        // Assume the machine which exports the v1 file is Little-Endian, which make parsing
        // easier.
        // First, we have the header
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x616b3432) { // ak42
            return error.BadMagic;
        }
        // Next, a signed integer for the export version
        const version = try reader.readInt(i32, .little);
        if (version != 1) {
            return error.Version;
        }
        // The next 7 values are 32 bit signed numbers
        const dim = try reader.readInt(i32, .little);
        const hidden_dim = try reader.readInt(i32, .little);
        const n_layers = try reader.readInt(i32, .little);
        const n_heads = try reader.readInt(i32, .little);
        const n_kv_heads = try reader.readInt(i32, .little);
        const vocab_size = try reader.readInt(i32, .little);
        const max_seq_length = try reader.readInt(i32, .little);

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

/// Temporarily open the checkpoint file to read only the header.
/// Do not leave the file open because we ill mmap it.
fn read_config(model_path: []u8) !Config {
    const model_file = try std.fs.openFileAbsolute(model_path, .{ .mode = .read_only });
    // TODO: Use relative path support
    //const cwd = std.fs.cwd();
    //const model_file = try cwd.openFile(model_path, .{ .mode = .read_only });
    defer model_file.close();

    var buffer = std.io.bufferedReader(model_file.reader());
    return try Config.read(buffer.reader());
}

const Tokenizer = struct {
    const Self = @This();

    tokens: [][]u8,
    scores: []f32,
    max_len: usize,
    arena: ArenaAllocator,

    /// Read an exported Tokenizer model as exported by `llama2.c/tokenizer.py`
    fn read(reader: anytype, allocator: Allocator, vocab_size: usize) !Tokenizer {
        // Read a tokenizer file as written from Karpathy's `llama2.c`
        // According to `tokenizer.py`, the format is:
        //
        // A. 32 bits for the unsigned `max_token_length`
        // B. 32 bits for a float `score`
        // C. 32 bits for the unsigned length of bytes
        // D. The bytes for the token.
        // E. EOF or go to B again.
        //
        // The tokenizer file was likely created on a Little-Endian machine, so we
        // will assume these values are Little-Endian

        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        var ret: Tokenizer = undefined;

        ret.max_len = try reader.readInt(u32, .little);

        var alloc = arena.allocator();
        ret.scores = try alloc.alloc(f32, vocab_size);

        ret.tokens = try alloc.alloc([]u8, vocab_size);

        // Nothing we can really do here if there's an error
        for (0..vocab_size) |i| {
            const score_bits = try reader.readInt(u32, .little);
            const score: f32 = @bitCast(score_bits);
            ret.scores[i] = score;
            const token_len = try reader.readInt(u32, .little);
            const token = try alloc.alloc(u8, token_len);
            ret.tokens[i] = token;
            try reader.readNoEof(token[0..]);
        }

        // Copy the Area after all of the copies have been done, otherwise there will be
        // a leak from the Arena's allocations.
        ret.arena = arena;

        return ret;
    }

    fn deinit(self: *Self) void {
        self.arena.deinit();
    }
};

fn load_tokenizer(tokenizer_file: []u8, alloc: Allocator, vocab_size: usize) !Tokenizer {
    // open the file with fallback, then hand off to the tokenizer
    const cwd = std.fs.cwd();
    const tokenizer = try cwd.openFile(tokenizer_file, .{ .mode = .read_only });
    defer tokenizer.close();

    var buffer = std.io.bufferedReader(tokenizer.reader());

    return try Tokenizer.read(buffer.reader(), alloc, vocab_size);
}

/// Struct which contains the weights for the transformer.
/// This is loaded from a V1 export from `llama2.c`.
///
/// This struct is for calculation purposes immutable and represents
/// the contents of the loaded model, backed by `mmap(2)`.
const TransformerV1 = struct {
    pub const page_size_min = std.heap.page_size_min;

    const Self = @This();

    fd: std.posix.fd_t,
    ptr: ?[]align(page_size_min) u8,

    /// Initialize the Transformer weight pointers with the provided file.
    fn init(model_path: []u8) !TransformerV1 {
        std.debug.print("Opening Transformer model\n", .{});
        const fd = try std.posix.open(model_path, .{}, 0o440);

        // Here we mmap() the weights files because nobody wants to open up a 25 GB file raw!
        const stat = try std.posix.fstat(fd);
        // zig fmt: off
        std.debug.print(
            "Model size: {d:.1} MiB\n",
            .{ @as(f32, @floatFromInt(stat.size)) / 1048576.0 }
        );
        // zig fmt: on

        const fsize: u64 = @intCast(stat.size);

        // See `std.os.linux.MAP` for more info.
        //std.os.linux.MAP
        const mmap_type: std.posix.MAP = .{
            .TYPE = .PRIVATE,
        };

        const ptr = try std.posix.mmap(null, fsize, std.posix.PROT.READ, mmap_type, fd, 0);

        // V1 Export Format from `llama2.c`
        // The first 256 bytes contain the header with trailing 0 padding.
        //

        // We have the ptr. Time to handle it.
        //const _weight_start = @intFromPtr(ptr) + 256;

        // According to the output from the modified `export.py` file, we have these dimensions:
        // layers: 32 (known from config)

        // normalization layers:
        // attention:   [4096]f32
        // ffn_norm:    [4096]f32
        // norm:        [4096]f32

        // token embeddings:
        // embed:       [vocab_size][4096]f32

        // attention layers:
        // wq:          [4096][4096]f32
        // wk:          [4096][4096]f32
        // wv:          [4096][4096]f32
        // wo:          [4096][4096]f32

        // ff layers:
        // w1:          [11008][4096]f32
        // w2:          [4096][11008]f32
        // w3:          [11008][4096]f32
        // output:      [vocab_size][4096]f32

        return TransformerV1{
            .fd = fd,
            .ptr = ptr,
        };
    }

    fn deinit(self: *Self) void {
        std.debug.print("Transformer.deinit()\n", .{});
        std.posix.munmap(self.ptr.?);
        self.ptr = null;
        self.fd = -1;
    }
};
