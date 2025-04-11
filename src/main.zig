// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

const std = @import("std");
const Allocator = std.mem.Allocator;

const clap = @import("clap");

const GPA = std.heap.GeneralPurposeAllocator(.{});

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
    try stdout.print("dim: {d}, hidden: {d}, n_layers: {d}, n_heads: {d}, n_kv: {d}, vocab: {d}, max_seq: {d}\n", .{ config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size, config.max_seq_length });
    try bw.flush();

    const tokenizer_path = "tokenizer.bin";
    const tokenizer_path_dupe = try alloc.dupeZ(u8, tokenizer_path);
    defer alloc.free(tokenizer_path_dupe);
    var tokenizer = try load_tokenizer(tokenizer_path_dupe, gpa.allocator(), config.vocab_size);
    defer tokenizer.deinit();
    try stdout.print("loaded tokenizer\nmax tokenizer length: {d}\n", .{tokenizer.max_len});
    try bw.flush();

    // TODO: Load weights

    try stdout.print("Done\n", .{});
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

        return Config{
            .dim = @intCast(dim),
            .hidden_dim = @intCast(hidden_dim),
            .n_layers = @intCast(n_layers),
            .n_heads = @intCast(n_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .vocab_size = @intCast(vocab_size),
            .max_seq_length = @intCast(max_seq_length),
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
    alloc: Allocator,

    /// Read an exported Tokenizer model as exported by `llama2.c/tokenizer.py`
    fn read(reader: anytype, alloc: Allocator, vocab_size: usize) !Tokenizer {
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

        var ret: Tokenizer = undefined;

        ret.alloc = alloc;

        ret.max_len = try reader.readInt(u32, .little);

        ret.scores = try alloc.alloc(f32, vocab_size);
        errdefer alloc.free(ret.scores);

        ret.tokens = try alloc.alloc([]u8, vocab_size);
        errdefer {
            for (ret.tokens) |tok| {
                if (tok.len != 0) {
                    alloc.free(tok);
                }
            }
        }
        errdefer alloc.free(ret.tokens);

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

        return ret;
    }

    fn deinit(self: *Self) void {
        for (self.tokens) |token| {
            self.alloc.free(token);
        }
        self.alloc.free(self.tokens);
        self.alloc.free(self.scores);
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
