// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

const std = @import("std");

const clap = @import("clap");

const ggml = @import("ggml.zig");
const llama = @import("llama.zig");

const print_perf = false;

const Config = llama.Config;
const TransformerV1 = llama.TransformerV1;
const Tokenizer = llama.Tokenizer;
const Params = llama.Params;
const State = llama.State;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Display this help and exit.
        \\-m, --model <str>      Path to the model to use
        \\-t, --tokenizer <str>  Path to the tokenizer to use
        \\-p, --prompt <str>     Prompt to use
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

    const stdout = bw.writer();
    try stdout.print("Loading model config\n", .{});
    try bw.flush();

    var alloc = gpa.allocator();

    const model_path: []const u8 = res.args.model orelse "llama2-7b.bin";
    const config = try Config.read(model_path);
    try stdout.print("loaded config\n", .{});
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

    if (res.args.prompt == null) {
        try stdout.print("Falling back to default prompt\n", .{});
    }
    const prompt: []const u8 = res.args.prompt orelse "Wikipedia the free online encyclopedia that";
    try bw.flush();

    const tokenizer_path = res.args.tokenizer orelse "tokenizer.bin";
    var tokenizer = try Tokenizer.init(tokenizer_path, alloc, config.vocab_size);
    defer tokenizer.deinit();
    try stdout.print("Loaded tokenizer; max length: {d}\n", .{tokenizer.max_len});

    try stdout.print("Loading model weights... ", .{});
    try bw.flush();
    var transformer = try TransformerV1.init(model_path, config);
    defer transformer.deinit();
    try stdout.print("Done loading model...\n", .{});
    try bw.flush();

    const tokens = try tokenizer.encode(prompt, alloc);
    defer alloc.free(tokens);

    try stdout.print("Got {d} encoded tokens\n", .{tokens.len});
    const chars_list = tokenizer.tokens.items(.chars);
    for (0.., tokens) |i, tok| {
        const idx = tokenizer.findIndexByTokenId(tok).?;
        const chars = chars_list[idx];
        try stdout.print("Token #{d} = {d: >8}; <<{s}>>\n", .{ i, tok, chars });
    }
    try bw.flush();

    var state = try State.init(alloc, config);
    defer state.deinit();

    var picker = Params.init(0.95, 0.9, config.vocab_size);

    var n: usize = 0;
    var token: Tokenizer.Token = undefined;
    var progress = std.Progress.start(.{ .root_name = "Predicting" });
    defer progress.end();

    const start_time = try std.time.Instant.now();

    // TODO: Implement shifting whenever running for longer than `max_seq_length`
    while (n < config.max_seq_length) : (n += 1) {
        progress.setCompletedItems(n);
        if (n < tokens.len) {
            // Feed next token in prompt
            token = tokens[n];
        }

        const idx_in = tokenizer.findIndexByTokenId(token).?;
        const chars_in = tokenizer.tokens.items(.chars)[idx_in];

        const out = transformer.forward(&state, token, n, progress);
        const decoded = try picker.sample(out, alloc);

        if (comptime print_perf) {
            const now = try std.time.Instant.now();
            const elapsed_millis = now.since(start_time) / 1_000_000;
            const elapsed_secs = @as(f32, @floatFromInt(elapsed_millis)) / 1000;
            const per_token = @as(f32, @floatFromInt(elapsed_millis)) / (@as(f32, @floatFromInt(n + 1)));
            try stdout.print("{d} ms since start. cum: {d:.2} ms per token\n", .{
                elapsed_secs, per_token,
            });
        }

        const idx = tokenizer.findIndexByTokenId(decoded).?;
        const chars = tokenizer.tokens.items(.chars)[idx];

        try stdout.print("In: {d} <<{s}>>; Out: {d} <<{s}>>\n", .{ token, chars_in, decoded, chars });
        try bw.flush();
        token = decoded;
    }

    try stdout.print("Done\nCleaning up\n", .{});
    try bw.flush();
}
