// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! bench: Benchmarking suite
//! This module contains benchmarks for measuring the performance of various math kernel
//! implementations and variations.

const std = @import("std");

const llm = @import("llm");
const math = llm.math;
const quant = llm.math.quant;

const Weights = math.Weights;

const warmup_iterations = 1000;
const iterations = 2_000;

pub fn main() !void {
    try setAffinity();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const seed = 0x13371337_13371337;

    std.debug.print("iterations: {d}, warmup iterations: {d}\n", .{ iterations, warmup_iterations });
    std.debug.print("===================================================\n", .{});

    try bench_f32(seed, alloc);
    try bench_q80(seed, alloc);
}

fn setAffinity() !void {
    const pid = std.os.linux.getpid();
    std.debug.print("my pid is {d}\n", .{pid});

    var cpu_set = std.mem.zeroes(std.os.linux.cpu_set_t);
    // Try to schedule on cpu #0
    cpu_set[0] |= 0x01;
    try std.os.linux.sched_setaffinity(pid, &cpu_set);

    const affinity_ret = std.os.linux.sched_getaffinity(pid, @sizeOf(std.os.linux.cpu_set_t), &cpu_set);
    if (affinity_ret != 0) {
        std.debug.print("sched_getaffinity returned {d}\n", .{affinity_ret});
        return error.affinity;
    }
    if (cpu_set[0] != 1) {
        std.debug.print("sched_getaffinity returned mask {x}\n", .{cpu_set});
        return error.affinity;
    }
}

fn fill(rand: std.Random, x: []f32) void {
    for (0..x.len) |i| {
        x[i] = rand.float(f32);
    }
}

/// Run benchmark for the method
fn benchmark(
    name: []const u8,
    func: *const fn (args: anytype) void,
    args: anytype,
) !void {
    // Do warm-up exactly as we will do the final run
    std.debug.print("warming up for {s}... ", .{name});
    const start_warmup = try std.time.Instant.now();
    for (0..warmup_iterations) |_| {
        std.mem.doNotOptimizeAway(func(args));
    }
    const finished_warmup = try std.time.Instant.now();
    const elapsed_warmup_ns = finished_warmup.since(start_warmup);
    std.mem.doNotOptimizeAway(elapsed_warmup_ns);
    std.debug.print("done\n", .{});

    // Actually test run
    const start = try std.time.Instant.now();
    for (0..iterations) |_| {
        std.mem.doNotOptimizeAway(func(args));
    }
    const end = try std.time.Instant.now();
    const elapsed_nanos = end.since(start);
    const millis = elapsed_nanos / 1_000_000;
    std.debug.print("{s: <40} | {d: >8.0} ms | {d: >8.4} nanos/iter\n", .{ name, millis, elapsed_nanos / iterations });
}

fn bench_f32(seed: comptime_int, alloc: std.mem.Allocator) !void {
    var rand = std.Random.Xoroshiro128.init(seed);
    const random = rand.random();

    const a = try alloc.alloc(f32, 4096);
    defer alloc.free(a);
    const b = try alloc.alloc(f32, 4096 * 4096);
    defer alloc.free(b);
    const c = try alloc.alloc(f32, 4096);
    defer alloc.free(c);

    std.mem.doNotOptimizeAway(fill(random, b));
    std.mem.doNotOptimizeAway(fill(random, c));

    const func = struct {
        fn matmul(args: anytype) void {
            const out, const m, const x = args;
            llm.math.matrixMulVec(f32, out, m, x, 4096, 4096);
        }
    }.matmul;

    try benchmark("fp32 matrixMulVec (4096x4096)x4096", func, .{ a, Weights{ .f32 = b }, Weights{ .f32 = c } });
}

fn bench_q80(seed: comptime_int, alloc: std.mem.Allocator) !void {
    const Q80Block = quant.Q80Block;
    var rand = std.Random.Xoroshiro128.init(seed);
    const random = rand.random();

    const a = try alloc.alloc(f32, 4096);
    defer alloc.free(a);
    const b = try alloc.alloc(Q80Block, 4096 * 4096 / quant.blockUnitLen(Q80Block));
    defer alloc.free(b);
    const c = try alloc.alloc(Q80Block, 4096 / quant.blockUnitLen(Q80Block));
    defer alloc.free(c);

    const trash_b = try alloc.alloc(f32, 4096 * 4096);
    defer alloc.free(trash_b);
    const trash_c = try alloc.alloc(f32, 4096);
    defer alloc.free(trash_c);

    std.mem.doNotOptimizeAway(fill(random, trash_b));
    std.mem.doNotOptimizeAway(fill(random, trash_c));

    _ = try quant.quantize(.q8_0, trash_b, b);
    _ = try quant.quantize(.q8_0, trash_c, c);

    const func = struct {
        fn matmul(args: anytype) void {
            const out, const m, const x = args;
            llm.math.matrixMulVec(f32, out, m, x, 4096, 4096);
        }
    }.matmul;

    try benchmark("Q8_0 matrixMulVec (4096x4096)x4096", func, .{ a, Weights{ .q8_0 = b }, Weights{ .q8_0 = c } });
}
