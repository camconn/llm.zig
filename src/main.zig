// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

const std = @import("std");

const clap = @import("clap");

const llm = @import("llm");
const llama = llm.llama;

const print_perf = false;

const Config = llama.Config;
const TransformerV1 = llama.TransformerV1;
const Tokenizer = llm.token.Tokenizer;
const Sampler = llm.sample.Sampler;
const State = llama.State;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Display this help and exit.
        \\-f, --format <str>     Format of model to load ("ggml", "llama2.c")
        \\-m, --model <str>      Path to the model to use
        \\-t, --tokenizer <str>  Path to the tokenizer to use ("llama2.c" format only)
        \\-p, --prompt <str>     Prompt to use
        \\-d, --debug            Flag to enable debug mode (default off)
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

    const debug_mode = res.args.debug != 0;

    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();

    const model_format: []const u8 = res.args.format orelse "ggml";
    if (!(std.mem.eql(u8, model_format, "ggml") or std.mem.eql(u8, model_format, "llama2.c"))) {
        return clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});
    }

    try stdout.print("Loading model config\n", .{});

    var alloc = gpa.allocator();
    //var alloc = std.heap.page_allocator;

    const model_path: []const u8 = res.args.model orelse "llama2-7b.bin";
    const tokenizer_path = res.args.tokenizer orelse "tokenizer.bin";

    var context = if (std.mem.eql(u8, model_format, "ggml"))
        try llama.LlamaContext.init(model_path, alloc)
    else
        try load_llama2(tokenizer_path, model_path, alloc);

    defer context.deinit();

    var state = context.state;
    const transformer = context.transformer;
    var tokenizer = context.tokenizer;
    const config = context.config;

    if (debug_mode) {
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
    }

    if (res.args.prompt == null) {
        try stdout.print("Falling back to default prompt\n", .{});
    }
    const prompt: []const u8 = res.args.prompt orelse "Wikipedia the free online encyclopedia that";

    const tokens = try tokenizer.encode(prompt, alloc);
    defer alloc.free(tokens);

    if (debug_mode) try stdout.print("Got {d} encoded tokens\n", .{tokens.len});

    if (debug_mode) {
        for (0.., tokens) |i, tok| {
            const chars = tokenizer.getTokenChars(tok).?;
            try stdout.print("Token #{d} = {d: >8}; <<{s}>>\n", .{ i, tok, chars });
        }
    }

    try run_inference(transformer, tokens, &state, tokenizer, alloc, debug_mode);

    try stdout.print("Done\nCleaning up\n", .{});
}

fn run_inference(
    transformer: TransformerV1,
    prompt: []Tokenizer.Token,
    state: *State,
    tokenizer: Tokenizer,
    allocator: std.mem.Allocator,
    debug_mode: bool,
) !void {
    const stdout = std.io.getStdOut().writer();
    const progress: ?std.Progress.Node = if (debug_mode)
        std.Progress.start(.{ .root_name = "Predicting" })
    else
        null;
    defer if (progress) |prog| prog.end();

    var n: usize = 0;
    var tok: Tokenizer.Token = undefined;
    const config = transformer.config;
    var picker = Sampler.init(0.95, 0.9, config.vocab_size);

    const start_time = try std.time.Instant.now();
    // TODO: Implement shifting whenever running for longer than `max_seq_length`
    while (n < config.max_seq_length) : (n += 1) {
        if (progress) |prog| prog.setCompletedItems(n);
        const in_prompt = n < prompt.len;
        if (in_prompt) {
            // Feed next token in prompt
            tok = prompt[n];
        }

        const input = tokenizer.getTokenChars(tok).?;
        const out = transformer.forward(state, tok, n, progress);
        const decoded = try picker.sample(out, allocator);
        const predicted = tokenizer.getTokenChars(decoded).?;

        if (comptime print_perf) {
            const now = try std.time.Instant.now();
            const elapsed_millis = now.since(start_time) / 1_000_000;
            const elapsed_secs = @as(f32, @floatFromInt(elapsed_millis)) / 1000;
            const per_token = @as(f32, @floatFromInt(elapsed_millis)) / (@as(f32, @floatFromInt(n + 1)));
            try stdout.print("[{d:.1} sec total, {d:.2} ms per token]\n", .{
                elapsed_secs, per_token,
            });
        }

        if (debug_mode) {
            try stdout.print("In: {d} <<{s}>>; Out: {d} <<{s}>>\n", .{ tok, input, decoded, predicted });
        } else if (in_prompt) {
            // Don't print BOS at start of output.
            if (n != 0 and tok != Tokenizer.BOS) {
                try stdout.print("{s}", .{input});
            }
            // If the next token is a prediction, go ahead and print out the prediction.
            if (n + 1 == prompt.len) {
                try stdout.print("{s}", .{predicted});
            }
        } else {
            if (decoded != Tokenizer.EOS) {
                try stdout.print("{s}", .{predicted});
            }
        }
        tok = decoded;

        // End of output
        if (tok == Tokenizer.EOS) break;
    }
}

fn load_llama2(tokenizer_path: []const u8, model_path: []const u8, alloc: std.mem.Allocator) !llama.LlamaContext {
    const config = try Config.read(model_path);

    const stdout_file = std.io.getStdOut();
    const stdout = stdout_file.writer();

    try stdout.print("loaded config\n", .{});

    var tokenizer = try Tokenizer.initV1(tokenizer_path, config.vocab_size, alloc);
    errdefer tokenizer.deinit();
    try stdout.print("Loaded tokenizer; max length: {d}\n", .{tokenizer.max_len});

    try stdout.print("Loading model weights... ", .{});
    var transformer = try TransformerV1.initV1(model_path, config, alloc);
    errdefer transformer.deinit();
    try stdout.print("Done loading model...\n", .{});

    var state = try State.init(alloc, config);
    errdefer state.deinit();

    return .{
        .config = config,
        .transformer = transformer,
        .tokenizer = tokenizer,
        .state = state,
    };
}
