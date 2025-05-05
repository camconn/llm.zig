// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

const std = @import("std");

const clap = @import("clap");

const llm = @import("llm");

const Model = llm.model.Model;
const Token = llm.token.Token;

const print_perf = false;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Display this help and exit.
        \\-m, --model <str>      Path to the model to use
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

    try stdout.print("Loading model config\n", .{});

    var alloc = gpa.allocator();

    const model_path: []const u8 = res.args.model orelse "llama2-7b.bin";

    var model = Model.init(model_path, alloc) catch |err| {
        std.debug.print("Issue loading model from {s}: {any}\n", .{ model_path, err });
        return;
    };
    defer model.deinit();

    if (res.args.prompt == null) {
        try stdout.print("Falling back to default prompt\n", .{});
    }
    const prompt: []const u8 = res.args.prompt orelse "Wikipedia the free online encyclopedia that";

    const tokens = try model.tokenize(prompt, .start, alloc);
    defer alloc.free(tokens);

    if (debug_mode) try stdout.print("Got {d} encoded tokens\n", .{tokens.len});

    if (debug_mode) {
        for (0.., tokens) |i, tok| {
            const chars = model.toString(tok).?;
            try stdout.print("Token #{d} = {d: >8}; <<{s}>>\n", .{ i, tok, chars });
        }
    }

    try run_inference(&model, tokens, alloc, debug_mode);

    try stdout.print("Done\nCleaning up\n", .{});
}

fn run_inference(
    model: *Model,
    prompt: []const Token,
    allocator: std.mem.Allocator,
    debug_mode: bool,
) !void {
    const stdout = std.io.getStdOut().writer();

    const progress: ?std.Progress.Node = if (debug_mode)
        std.Progress.start(.{ .root_name = "Predicting" })
    else
        null;
    defer if (progress) |prog| prog.end();

    const model_info = model.getInfo();
    var picker = llm.sample.Sampler.init(0.95, 0.9, model_info.vocab_size);

    var n: usize = 0;
    var tok: Token = undefined;

    const start_time = try std.time.Instant.now();
    // TODO: Implement shifting whenever running for longer than context window
    while (n < model_info.context_len) : (n += 1) {
        if (progress) |prog| prog.setCompletedItems(n);
        const in_prompt = n < prompt.len;
        if (in_prompt) {
            // Feed next token in prompt
            tok = prompt[n];
        }

        const input = model.toString(tok).?;
        const out = model.forward(tok, n);
        const next_token = try picker.sample(out, allocator);
        const predicted = model.toString(next_token).?;

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
            try stdout.print("In: {d} <<{s}>>; Out: {d} <<{s}>>\n", .{ tok, input, next_token, predicted });
        } else if (in_prompt) {
            // Special handling for start of input token
            if (model_info.start_token) |start| {
                if (tok != start) {
                    try stdout.print("{s}", .{input});
                }
            } else {
                try stdout.print("{s}", .{input});
            }

            // If the next token is a prediction, go ahead and print out the prediction.
            if (n + 1 == prompt.len) {
                try stdout.print("{s}", .{predicted});
            }
        } else {
            if (model_info.end_token) |stop| {
                if (next_token != stop) {
                    try stdout.print("{s}", .{predicted});
                }
            } else {
                try stdout.print("{s}", .{predicted});
            }
        }
        tok = next_token;

        // End of output
        if (model_info.end_token) |stop| {
            if (tok == stop) break;
        }
    }
}
