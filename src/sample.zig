// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! sample: Sampling algorithms for LLMs and other stochastic models
//! This module contains functionality for sampling stochastic models with various algorithms.

const std = @import("std");

const llm = @import("root.zig");
const token = llm.token;
const math = llm.math;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const Tokenizer = token.Tokenizer;
const Token = Tokenizer.Token;

pub const Sampler = struct {
    const Self = @This();

    temperature: f32,
    top_p: f32,
    rng: std.Random.Xoroshiro128,
    vocab_size: usize,

    const Pair = struct {
        t: Tokenizer.Token,
        f: f32,

        fn desc(_: void, lhs: Pair, rhs: Pair) bool {
            const fun = std.sort.desc(f32);
            return @call(.auto, fun, .{ {}, lhs.f, rhs.f });
        }
    };

    pub fn init(temperature: f32, top_p: f32, vocab_size: usize) Sampler {
        const now = std.time.milliTimestamp();
        const rng = std.Random.Xoroshiro128.init(@bitCast(now));
        return .{
            .temperature = temperature,
            .top_p = top_p,
            .rng = rng,
            .vocab_size = vocab_size,
        };
    }

    /// Greedily or stochastically sample the next token given a set of token probabilities
    /// (or *logits*) depending on how this sampler was called when `init()`.
    /// If `temperature == 0` then greedily sample, otherwise do nucleus sampling.
    pub fn sample(self: *Self, probs: []f32, allocator: Allocator) !Tokenizer.Token {
        var next: Tokenizer.Token = undefined;
        if (self.temperature == 0) {
            const idx = std.sort.argMax(f32, probs, {}, std.sort.asc(f32)).?;
            next = @intCast(idx);
        } else {
            // Apply temperature
            for (0..self.vocab_size) |q| {
                probs[q] /= self.temperature;
            }
            math.softMax(probs);

            const random = self.rng.random().float(f32);
            if (self.top_p <= 0 or self.top_p >= 1) {
                @panic("Unimplemented");
            } else {
                var arena = ArenaAllocator.init(allocator);
                defer arena.deinit();
                next = try self.sample_nucleus(probs, random, arena.allocator());
            }
        }
        return next;
    }

    /// Implement Nucleus Sampling as described in The Curious Case of Neural Text Degeneration [1].
    /// [1]: https://arxiv.org/abs/1904.09751
    fn sample_nucleus(self: *Self, probs: []f32, random: f32, alloc: Allocator) !Tokenizer.Token {
        var tokens = try alloc.alloc(Pair, self.vocab_size);
        for (0..self.vocab_size, probs) |i, p| {
            tokens[i] = Pair{ .t = @intCast(i), .f = p };
        }

        // Sort from highest to lowest
        std.mem.sortUnstable(Pair, tokens, {}, Pair.desc);

        // Find the first `last_idx` tokens which have a sum >= `self.top_p`
        var sum: f32 = 0;
        var last_idx = self.vocab_size;
        for (0..last_idx, tokens) |i, tok| {
            sum += tok.f;
            if (sum >= self.top_p) {
                last_idx = i + 1;
                break;
            }
        }

        // random ∈ [0, 1)
        // and
        //    sum ∈ [0, ~top_p]
        // so
        //      r ∈ [0, ~top_p]
        const r = random * sum;
        var cdf: f32 = 0;
        for (tokens[0..last_idx]) |tok| {
            cdf += tok.f;
            if (r < cdf) {
                return tok.t;
            }
        }

        // Prevent OOB ref
        if (last_idx == self.vocab_size) {
            last_idx = self.vocab_size - 1;
        }
        // Did not find a token. Just return the last one
        return tokens[last_idx].t;
    }
};
