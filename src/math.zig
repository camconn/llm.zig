// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! Math: This module contains math and other helper functions for neural networks.
//!

const std = @import("std");

/// Vector item for non-quantized Neural Networks.
pub const Elem = f32;
/// Default vector length for non-quantized networks.
pub const vector_len = std.simd.suggestVectorLength(Elem) orelse 8;
/// Floating point Vector type for non-quantized math.
pub const Vec = @Vector(vector_len, Elem);

/// Perform RMS Normalization on `x` with the weights `y` and store the result in `out`.
/// Requires all inputs to have the same length.
///
/// This method mirrors the implementation of `RMSNorm` in Meta's `model.py`.
/// It implements the method described in [1], plus adds a small epsilon factor for numeric
/// stability.
/// [1]: https://arxiv.org/abs/1910.07467
pub fn rmsNorm(out: []f32, x: []const f32, y: []const f32) void {
    if (!(x.len == y.len and y.len == out.len)) {
        std.debug.print("lengths: out={d}, x={d}, y={d}\n", .{ out.len, x.len, y.len });
        std.debug.assert(out.len == x.len);
        std.debug.assert(x.len == y.len);
        @panic("Mismatched lengths");
    }

    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    var sum: f32 = 0;
    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const square = xs * xs;
        const chunk_sum = @reduce(.Add, square);
        sum += chunk_sum;
    }

    for (leftover_offset..x.len) |i| {
        const xs = x[i];
        const square = xs * xs;
        sum += square;
    }

    const epsilon = 1e-9;
    const x_len: f32 = @floatFromInt(x.len);
    const mean = sum / x_len + epsilon;
    const rms = std.math.sqrt(mean);

    // Now perform division + multiply by weights.
    const divisor: Vec = @splat(rms);
    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const ys: Vec = y[idx .. idx + vector_len][0..vector_len].*;

        const result = (xs * ys) / divisor;
        out[idx .. idx + vector_len][0..vector_len].* = result;
    }

    for (leftover_offset..x.len) |i| {
        const xs = x[i];
        const ys = y[i];

        const result = (xs * ys) / rms;
        out[i] = result;
    }
}

test "RMSNorm" {
    const xs = [_]f32{
        0.3941330, 0.8046976, 0.2116031, 0.8854799, 0.2516429, 0.4804138,
        0.5427210, 0.5138140, 0.4892414, 0.1699823, 0.2136165, 0.5613836,
    };
    const ys = [_]f32{
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
    };
    var out = [_]f32{0} ** 12;

    const expected = [_]f32{
        0.7737724, 1.5798038, 0.4154249, 1.7383978, 0.4940321, 0.9431612,
        1.0654845, 1.0087335, 0.9604918, 0.3337137, 0.4193776, 1.1021234,
    };
    rmsNorm(&out, &xs, &ys);

    const eps = 0.00005;
    for (0..out.len, out) |i, x| {
        try std.testing.expectApproxEqAbs(expected[i], x, eps);
    }
}

/// Calculate the safe softmax [1] of `x` and the result back inside of it.
/// Caller must ensure that `x` is non-empty and has length > 0.
///
/// [1]: https://en.wikipedia.org/wiki/Softmax_function#Numerical_algorithms
pub fn softMax(x: []f32) void {
    std.debug.assert(x.len != 0); // required for `std.sort.max`

    // TODO: Make this SIMD w/ vectors. This is embarrassingly parallelizable

    // Find maximum value for safe softmax
    const max_val = std.sort.max(f32, x, {}, std.sort.asc(f32)).?;

    var sum: f32 = 0;
    for (0..x.len) |i| {
        const val = @exp(x[i] - max_val);
        sum += val;
        x[i] = val;
    }

    // normalize
    for (0..x.len) |i| {
        x[i] /= sum;
    }
}

test "softMax" {
    var x = [_]f32{ 0.2712, 0.4690, 0.6202, 0.5626, 0.9199, 0.7668 };
    softMax(&x);

    const expected = [_]f32{
        0.1172398, 0.1428910, 0.1662066,
        0.1569152, 0.2242854, 0.1924621,
    };
    const eps = 0.00005;
    for (0.., x) |i, xv| {
        try std.testing.expectApproxEqAbs(expected[i], xv, eps);
    }
}

/// Calculate swiglu(x) and store that back into x.
pub fn swiglu(x: []f32) void {
    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    // Calculate σ(x) = x / (1 + e^{-x})

    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;

        const exp: Vec = std.math.exp(-xs);
        const ones: Vec = @splat(1);
        const denom = ones + exp;

        const out = xs / denom;

        x[idx .. idx + vector_len][0..vector_len].* = out;
    }

    for (leftover_offset..x.len) |i| {
        const xs = x[i];
        const exp = std.math.exp(-xs);

        x[i] = xs / (1 + exp);
    }
}

test "swiglu" {
    var input = [_]f32{
        0.2712, 0.4690, 0.6202,
        0.5626, 0.9199, 0.7668,
    };
    swiglu(&input);
    const expected = [_]f32{
        0.1538460, 0.2885145, 0.4032652,
        0.3584355, 0.6577047, 0.5236192,
    };

    const eps = 0.00005;
    for (0.., input) |i, inp| {
        try std.testing.expectApproxEqAbs(expected[i], inp, eps);
    }

    var input2 = [_]f32{
        0.3105714, 0.5104846, 0.4118931, 0.8377581, 0.9840687, 0.0081603,
        0.9676210, 0.3114376, 0.1081485, 0.0694769, 0.1797512, 0.7861544,
    };
    const expected2 = [_]f32{
        0.1792074, 0.3190120, 0.2477709, 0.5847492, 0.7163181, 0.0040968,
        0.7011818, 0.1797730, 0.0569954, 0.0359447, 0.0979315, 0.5400920,
    };
    swiglu(&input2);
    for (0.., input2) |i, inp| {
        try std.testing.expectApproxEqAbs(expected2[i], inp, eps);
    }
}

/// Add `x` and `y` then store into `out`.
/// Caller is responsible for ensuring the lengths of `x`, `y`, and `out` are the same.
pub fn add(out: []f32, x: []const f32, y: []const f32) void {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len == out.len);
    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const ys: Vec = y[idx .. idx + vector_len][0..vector_len].*;
        const sum = xs + ys;

        out[idx .. idx + vector_len][0..vector_len].* = sum;
    }

    for (leftover_offset..x.len) |i| {
        const xs = x[i];
        const ys = y[i];

        out[i] = xs + ys;
    }
}

test "add" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    var out = [_]f32{ 0, 0, 0 };

    add(&out, &a, &b);
    try std.testing.expectEqualDeep([_]f32{ 5, 7, 9 }, out);
}

/// Calculate the element-wise product of `x` and `y` then store the result in `out`.
/// Caller is responsible for ensuring the lengths of `x`, `y`, and `out` are the same.
pub fn elementProduct(out: []f32, x: []const f32, y: []const f32) void {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len == out.len);
    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const ys: Vec = y[idx .. idx + vector_len][0..vector_len].*;
        const prod = xs * ys;

        out[idx .. idx + vector_len][0..vector_len].* = prod;
    }

    for (leftover_offset..x.len) |i| {
        const xs = x[i];
        const ys = y[i];

        out[i] = xs * ys;
    }
}

/// Multiply a matrix `m` of (`rows`, `cols`) by a vector `x` of (`cols`) and store in `out`.
pub fn matrixMul(out: []f32, m: []const f32, x: []const f32, rows: usize, cols: usize) void {
    // zig fmt: off
    //std.debug.print("out: {d}, m: {d}, x: {d}, rows: {d}, cols: {d}\n",
    //                .{ out.len, m.len, x.len, rows, cols });
    // zig fmt: on
    std.debug.assert(out.len == rows);
    std.debug.assert(x.len == cols);
    std.debug.assert(m.len == rows * cols);

    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    for (0..rows) |row| {
        const m_off = row * cols;

        var sum: f32 = 0;
        for (0..chunks) |chunk| {
            const idx = chunk * vector_len;
            const m_idx = m_off + idx;
            const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
            const ms: Vec = m[m_idx .. m_idx + vector_len][0..vector_len].*;

            const prod = xs * ms;
            const chunk_sum = @reduce(.Add, prod);
            sum += chunk_sum;
        }

        for (leftover_offset..cols) |i| {
            const xs = x[i];
            const ms = m[m_off + i];
            sum += xs * ms;
        }

        out[row] = sum;
    }
}

test "matrixMul" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 2, 5 };
    var out2 = [_]f32{0} ** 2;

    matrixMul(&out2, &a, &b, 2, 3);
    try std.testing.expectEqualDeep([_]f32{ 20, 44 }, out2);

    const d = [_]f32{ 4, -1 };
    var out3 = [_]f32{0} ** 3;
    matrixMul(&out3, &a, &d, 3, 2);
    try std.testing.expectEqualDeep([_]f32{ 2, 8, 14 }, out3);

    const m = [_]f32{
        81, 11, 41, 97, 22,
        5,  13, 10, 8,  45,
        70, 42, 87, 4,  27,
        86, 90, 85, 37, 19,
        93, 89, 9,  97, 93,
        21, 35, 61, 37, 35,
        71, 82, 20, 82, 17,
        77, 99, 37, 59, 40,
        2,  88, 55, 44, 27,
        92, 27, 25, 36, 98,
        12, 44, 66, 69, 38,
        43, 13, 29, 79, 21,
    };
    const x = [_]f32{ 6, 18, 19, 4, 5, 15, 8, 7, 6, 3, 6, 1 };
    const z = [_]f32{ 9, 15, 7, 16, 10 };

    var out5 = [_]f32{0} ** 5;
    const expect5 = [_]f32{
        2855, 4581, 4242, 5244, 5014,
    };
    matrixMul(&out5, &m, &x, 5, 12);
    try std.testing.expectEqualDeep(expect5, out5);

    var out12 = [_]f32{0} ** 12;
    const expect12 = [_]f32{
        2953, 888, 2203, 3501, 4717, 2083, 3491, 3781, 2697, 2964, 2714, 2259,
    };
    matrixMul(&out12, &m, &z, 12, 5);
    try std.testing.expectEqualDeep(expect12, out12);
}

/// Find the dot product of two vectors `x` and `y`.
/// Caller is responsible for ensuring that both vectors have equal lengths.
pub fn dotProduct(x: []const f32, y: []const f32) f32 {
    if (x.len != y.len) {
        std.debug.print("x.len {d} != y.len {d}\n", .{ x.len, y.len });
    }
    std.debug.assert(x.len == y.len);

    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    var sum: f32 = 0;
    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const ys: Vec = y[idx .. idx + vector_len][0..vector_len].*;

        const chunk_prod = xs * ys;
        const chunk_sum = @reduce(.Add, chunk_prod);
        sum += chunk_sum;
    }

    for (leftover_offset..x.len) |i| {
        sum += x[i] * y[i];
    }
    return sum;
}

test "dotProduct" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 5, 7, 7 };
    try std.testing.expectEqual(40, dotProduct(&a, &b));

    const x = [_]f32{ 9, 13, 13, 7, 8, 4, 11, 8, 18, 15, 14, 14 };
    const y = [_]f32{ 14, 16, 10, 15, 16, 16, 1, 10, 13, 14, 13, 15 };
    try std.testing.expectEqual(1688, dotProduct(&x, &y));
}

/// Quantization types for AI model weights.
pub const WeightFormat = enum {
    /// Uncompressed weights in `f32` form.
    /// This is not actually a quantization, but a pseudo-type for internal purposes.
    Float32,
    /// Represents 8 bit weights scaled by a single precision float `d` for every 32 weights.
    /// The original weight `w = i * d` where `i` is the quantized integer and `d` is the scale.
    Q8_0,
};

/// Get the type of a Quantized of weights for `format`.
pub fn Block(format: WeightFormat) type {
    return switch (format) {
        .Float32 => f32,
        // QK8_0 = 32 in the ggml reference
        .Q8_0 => struct { f32, [32]i8 },
    };
}

test "Block() size" {
    const F32Block = Block(.Float32);
    try std.testing.expectEqual(@sizeOf(f32), @sizeOf(F32Block));

    // Ensure Q8_0 compatibility when reading from a GGUF file.
    const Q8_0Block = Block(.Q8_0);
    try std.testing.expectEqual(@sizeOf(f32) + 32 * @sizeOf(i8), @sizeOf(Q8_0Block));
}

/// Quantize weights `ws` from full-size `f32` to their quantized forms.
/// Caller is responsible for freeing the returned Quantized block array.
pub fn quantize(
    comptime format: WeightFormat,
    ws: []const f32,
    allocator: std.mem.Allocator,
) !struct { []const Block(format), f32 } {
    if (ws.len == 0) {
        return error.Empty;
    }

    // No conversion necessary
    if (format == .Float32) {
        return .{ ws, 0 };
    }

    const BlockType = Block(format);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .@"1").type).array;
    const block_size = array_info.len;
    std.debug.assert(ws.len % block_size == 0);

    const n_blocks = ws.len / block_size;
    const blocks = try allocator.alloc(BlockType, n_blocks);
    errdefer allocator.free(blocks);

    const err = switch (format) {
        .Q8_0 => quantize_q8_0(ws, blocks),
        else => @compileError("quantize method is unimplemented for " ++ format),
    };

    return .{
        blocks,
        err,
    };
}

const max_magnitude = struct {
    pub fn lessThan(_: void, lhs: f32, rhs: f32) bool {
        const a = @abs(lhs);
        const b = @abs(rhs);
        return a < b;
    }
};

// TODO: Vectorize this
fn quantize_q8_0(in: []const f32, blocks: []Block(.Q8_0)) f32 {
    // Calculate max error
    var err: f32 = 0;
    const n_blocks = blocks.len;
    const BlockType = Block(.Q8_0);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .@"1").type).array;
    const ArrayElem = array_info.child;
    const block_size = array_info.len;

    for (0..n_blocks) |i| {
        var blk: BlockType = undefined;
        var items = blk[1];
        const idx = i * block_size;

        const elems: []const f32 = in[idx .. idx + block_size];

        const max: f32 = std.sort.max(f32, elems, {}, max_magnitude.lessThan) orelse 0;
        const scale = max / 127.0;
        blk[0] = scale;
        // In the GGML reference the scale is inverted so that a cheaper multiply op can be
        // used element-wise.
        const inv_scale = if (scale == 0) 1 else 1 / scale; // ternary because divide-by-zero

        for (0..block_size) |j| {
            const old = elems[j];

            const scaled = std.math.round(old * inv_scale);
            std.debug.assert(scaled <= 127.0);
            std.debug.assert(scaled >= -127.0);
            // Ensure we are within 8 bits in a symmetric range.
            const rounded = std.math.clamp(scaled, -127.0, 127.0);

            // Save quantized weight
            const quantized: ArrayElem = @intFromFloat(rounded);
            items[j] = quantized;

            // Calculate maximum error for diagnostics
            const elem_err = @abs(@as(f32, @floatFromInt(quantized)) * inv_scale - old);
            err = @max(err, elem_err);
        }
        blk[1] = items;
        blocks[i] = blk;
    }
    return err;
}

test "quantize weights" {
    const empty: [0]f32 = [0]f32{};
    try std.testing.expectError(error.Empty, quantize(.Float32, &empty, std.testing.allocator));

    const f32s = [_]f32{ 1, 2, 3, 4, 5 };
    const f32_quant = try quantize(.Float32, &f32s, std.testing.allocator);
    try std.testing.expectEqualSlices(f32, &f32s, f32_quant[0]);
    try std.testing.expectEqual(0, f32_quant[1]);

    const no_scale = [_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };

    const q80_no_scale = try quantize(.Q8_0, &no_scale, std.testing.allocator);
    defer std.testing.allocator.free(q80_no_scale[0]);
    try std.testing.expectEqual(q80_no_scale[0].len, 1);

    const no_scale_exp = [32]i8{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };
    try std.testing.expectEqualSlices(i8, &no_scale_exp, &q80_no_scale[0][0][1]);
    try std.testing.expectEqual(0, q80_no_scale[1]);
}

/// De-quantize weights
pub fn dequantize(
    comptime format: WeightFormat,
    weights: []const Block(format),
    allocator: std.mem.Allocator,
) ![]const f32 {
    if (weights.len == 0) {
        return error.Empty;
    }

    if (format == .Float32) {
        return weights;
    }

    const BlockType = Block(format);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .@"1").type).array;
    const block_size = array_info.len;

    const n_out = weights.len * block_size;
    const out = try allocator.alloc(f32, n_out);
    errdefer allocator.free(out);

    switch (format) {
        .Q8_0 => dequantize_q8_0(weights, out),
        else => @compileError("dequantize method is unimplemented for " ++ format),
    }
    return out;
}

// TODO: Vectorize this
fn dequantize_q8_0(in: []const Block(.Q8_0), out: []f32) void {
    const BlockType = Block(.Q8_0);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .@"1").type).array;
    const block_size = array_info.len;

    for (0..in.len) |i| {
        const block = in[i];

        const idx = block_size * i;

        const scale = block[0];
        for (0..block_size) |j| {
            const elem = block[1][j];
            const elem_f: f32 = @floatFromInt(elem);
            out[idx + j] = elem_f * scale;
        }
    }
}

test "dequantize weights" {
    const empty = [0]f32{};
    try std.testing.expectError(error.Empty, dequantize(.Float32, &empty, std.testing.allocator));
    const empty_q80 = [0]Block(.Q8_0){};
    try std.testing.expectError(error.Empty, dequantize(.Q8_0, &empty_q80, std.testing.allocator));

    const no_scale = [_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };

    const q80_no_scale = try quantize(.Q8_0, &no_scale, std.testing.allocator);
    defer std.testing.allocator.free(q80_no_scale[0]);
    const out_dequantized = try dequantize(.Q8_0, q80_no_scale[0], std.testing.allocator);
    defer std.testing.allocator.free(out_dequantized);
}
