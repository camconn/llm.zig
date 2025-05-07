// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! Math: This module contains math and other helper functions for neural networks.

const std = @import("std");

pub const quant = @import("math/quant.zig");

const Q80Block = quant.Q80Block;
const Q6KBlock = quant.Q6KBlock;
const Q4KBlock = quant.Q4KBlock;

const Block = quant.Block;
const blockUnitLen = quant.blockUnitLen;

// TODO: Remove these pub declarizations
pub const quantize = quant.quantize;
pub const dequantize = quant.dequantize;

/// Helper struct for referencing quantized weights.
/// Contains a slice of `Block`s in `WeightFormat` to use when performing calculations.
pub const Weights = union(WeightFormat) {
    f32: []f32,
    f16: []f16,
    q8_0: []quant.Q80Block,
    q6_k: []quant.Q6KBlock,
    q4_k: []quant.Q4KBlock,

    pub fn len(self: @This()) usize {
        return switch (self) {
            .f32 => |f| f.len,
            .f16 => |f| f.len,
            .q8_0 => |q| q.len * comptime blockUnitLen(@typeInfo(@TypeOf(q)).pointer.child),
            .q6_k => |q| q.len * comptime blockUnitLen(@typeInfo(@TypeOf(q)).pointer.child),
            .q4_k => |q| q.len * comptime blockUnitLen(@typeInfo(@TypeOf(q)).pointer.child),
        };
    }
};

/// Quantization types for model weights.
pub const WeightFormat = enum {
    /// Uncompressed weights in `f32` form.
    /// This is not actually a quantization, but a pseudo-type for internal purposes.
    /// Blocks of `f32` are just the individual `f32` weights.
    /// Effectively 32 bits per weight.
    f32,
    /// Half-precision weights in `f16` form.
    /// This may or may not be a quantization depending on the native format of the weights.
    /// This is not treated as a quantization format, but just like an `f32`.
    /// Effectively 16 bits per weight.
    f16,
    /// Represents 8 bit weights scaled by a single precision float `d` for every 32 weights.
    /// The original weight `w = i * d` where `i` is the quantized integer and `d` is the scale.
    /// Effectively 8.5 bits per weight.
    q8_0,
    /// Represents 6 bit weights split between bottom 4 bits and the top 2 bits for each weight.
    /// Each weight is then scaled by a block scale and then a super-block scale.
    /// Effectively 6.5625 bits per weight.
    q6_k,
    /// Represents 4 bit weights in blocks of 32 where each block is in a super-block containing
    /// 8 blocks. Scales and min weights are quantized with 6 bits.
    /// Effectively 4.5 bits per weight.
    q4_k,
};

/// Compare `lhs` and `rhs` to see if they have the same inner quantization format.
inline fn same(lhs: Weights, rhs: Weights) bool {
    const a: WeightFormat = lhs;
    const b: WeightFormat = rhs;
    return a == b;
}

/// Create a SIMD vector for arbitrary type `T`.
fn Vect(comptime T: type) type {
    const len = comptime std.simd.suggestVectorLength(T) orelse 8;
    return @Vector(len, T);
}

/// Helper method to get the length of a vector from `Vect`
fn vectLen(comptime T: type) usize {
    if (@typeInfo(T) != .vector) {
        @compileError("Only implemented on `Vector()` and @Vector(.., ..) types");
    }
    return @typeInfo(T).vector.len;
}

/// Helper method to get the child elements of a vector type from `Vect`.
fn VectElem(comptime T: type) type {
    if (@typeInfo(T) != .vector) {
        @compileError("Only implemented on `Vector()` and @Vector(.., ..) types");
    }
    return @typeInfo(T).vector.child;
}

test "Vect setup" {
    // TODO: This is is kind of brittle and may break when compiling for different architectures
    //       other than x86_64.

    // We don't necessarily know the vector size on the target machine, so just find a basic
    // vector length and then extrapolate based on relative size of the element.
    const m = std.simd.suggestVectorLength(f32) orelse 8;
    const args = [_]struct { type, type, usize, type }{
        .{ f32, @Vector(m, f32), m, f32 },
        .{ f16, @Vector(m * 2, f16), m * 2, f16 },
        .{ i32, @Vector(m, i32), m, i32 },
        .{ i16, @Vector(m * 2, i16), m * 2, i16 },
        .{ i8, @Vector(m * 4, i8), m * 4, i8 },
    };

    // use `comptime` to force unroll because we are calculating types
    comptime for (args) |a| {
        const T, const Expected, const len_expected, const child_expected = a;
        const Vec = Vect(T);
        const len = vectLen(Vec);
        const child = VectElem(Vec);

        try std.testing.expectEqual(Expected, Vec);
        try std.testing.expectEqual(len_expected, len);
        try std.testing.expectEqual(child_expected, child);
    };
}

/// Compile-time guard to prevent use of floating-point types.
pub fn floatOnly(T: type) void {
    switch (@typeInfo(T)) {
        .float => {},
        else => @compileError("Only float types supported for this function"),
    }
}

/// Perform RMS Normalization on `x` with the weights `y` and store the result in `out`.
/// Requires all inputs to have the same length.
///
/// This method mirrors the implementation of `RMSNorm` in Meta's `model.py`.
/// It implements the method described in [1], plus adds a small epsilon factor for numeric
/// stability.
/// [1]: https://arxiv.org/abs/1910.07467
pub fn rmsNorm(out: []f32, x: []const f32, y: []const f32) void {
    rmsNormT(f32, out, x, y);
}

/// Perform RMS Normalization on `x` with the weights `y` and store in `out` with type `T`.
/// Requires all inputs to have the same length.
/// Requires that `T` is either `f32` or `f16`.
///
/// This method mirrors the implementation of `RMSNorm` in Meta's `model.py`.
/// It implements the method described in [1], plus adds a small epsilon factor for numeric
/// stability.
/// [1]: https://arxiv.org/abs/1910.07467
pub fn rmsNormT(T: type, out: []T, x: []const T, y: []const T) void {
    floatOnly(T);
    if (!(x.len == y.len and y.len == out.len)) {
        std.debug.print("lengths: out={d}, x={d}, y={d}\n", .{ out.len, x.len, y.len });
        std.debug.assert(out.len == x.len);
        std.debug.assert(x.len == y.len);
        @panic("Mismatched lengths");
    }

    const Vec = comptime Vect(T);
    const vector_len = comptime vectLen(Vec);

    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    var sum: T = 0;
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
    const x_len: T = @floatFromInt(x.len);
    const avg = sum / x_len + epsilon;
    const rms = std.math.sqrt(avg);

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

    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

    // Find maximum value for safe softmax
    const max_val = std.sort.max(f32, x, {}, std.sort.asc(f32)).?;

    const shift_r: u6 = @intCast(std.math.log2_int(usize, vector_len));
    const chunks = x.len >> shift_r;
    const leftover_offset = chunks * vector_len;

    // Create total sum and replace every element in `x` with exp(x[i] - max).
    var sum: f32 = 0;
    const max: Vec = @splat(max_val);
    for (0..chunks) |i| {
        const vals: Vec = x[i * vector_len ..][0..vector_len].*;
        const exp = @exp(vals - max);
        const chunk_sum = @reduce(.Add, exp);
        sum += chunk_sum;

        x[i * vector_len ..][0..vector_len].* = exp;
    }

    // Handle leftovers from SIMD
    for (leftover_offset..x.len) |i| {
        const val = @exp(x[i] - max_val);
        sum += val;
        x[i] = val;
    }

    // Now normalize with the divisor
    const summ: Vec = @splat(sum);
    for (0..chunks) |i| {
        const vals: Vec = x[i * vector_len ..][0..vector_len].*;
        const normalized = vals / summ;
        x[i * vector_len ..][0..vector_len].* = normalized;
    }

    // Handle leftovers from SIMD
    for (leftover_offset..x.len) |i| {
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

/// Perform approximate GELU activation [1] on slice `x` with type `T`.
/// [1]: https://arxiv.org/abs/1606.08415
pub fn geluApprox(T: type, x: []T) void {
    floatOnly(T);
    // Calculate 0.5x(1 + tanh([sqrt(2/pi) * (x + 0.044715 x^3)])) for each element in X.
    const factor = 0.044715;
    const root_2_pi = comptime std.math.sqrt(2.0 / std.math.pi);

    for (0.., x) |i, x_orig| {
        const x_cubed = std.math.pow(T, x_orig, 3);

        const inner_tanh = root_2_pi * (x_orig + factor * x_cubed);
        const tanh = std.math.tanh(inner_tanh);
        const out = x_orig * (1 + tanh) / 2.0;
        x[i] = out;
    }
}

/// Perform layer normalization [1] on `x` with weights `w` and bias `b`, with scratchpad `scratch`.
/// Requires all input slices have the same length.
/// Obliterates any data currently stored on `scratch`.
/// [1]: https://arxiv.org/abs/1607.06450
pub fn layerNorm(T: type, x: []T, w: []const T, b: []const T, scratch: []T) void {
    floatOnly(T);
    std.debug.assert(x.len == w.len);
    std.debug.assert(w.len == b.len);
    std.debug.assert(b.len == scratch.len);

    const Vec = comptime Vect(T);
    const vector_len = comptime vectLen(Vec);

    const len = x.len;
    const chunks = len / vector_len;
    const leftover_idx = chunks * vector_len;

    const avg = mean(T, x);
    const varnce = variance(T, x, scratch);

    // mu is the average
    const mu: Vec = @splat(@as(T, @floatCast(avg)));
    const eps = 0.00001;
    const denom = std.math.sqrt(varnce + eps);
    const denominator: Vec = @splat(@as(T, @floatCast(denom)));
    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const gamma: Vec = w[idx .. idx + vector_len][0..vector_len].*;
        const beta: Vec = b[idx .. idx + vector_len][0..vector_len].*;

        const numerator = (xs - mu) * gamma;
        const out = numerator / denominator + beta;
        std.mem.doNotOptimizeAway(out);

        x[idx .. idx + vector_len][0..vector_len].* = out;
    }

    for (leftover_idx..x.len) |i| {
        const numerator = (x[i] - @as(T, @floatCast(avg))) * w[i];
        x[i] = numerator / denom + b[i];
    }
}

test "layerNorm spot check" {
    var layer = [_]f32{
        2.0198,  -0.5469, -0.4136, -0.6555, -0.3068, 1.9992, -0.1188, -0.4410,
        0.0985,  0.2089,  -3.4739, 0.6712,  0.3281,  0.0761, 0.4307,  -0.6597,
        -0.6867, 0.5409,  0.3695,  0.0837,  -0.6385, 1.8382, 1.6295,  0.3715,
        0.4312,  0.4318,  0.7225,
    };
    var tmp = [_]f32{0} ** layer.len;

    const weights = [_]f32{
        -1.0709, 0.5777,  -2.0966, -0.7667, -0.3898, 0.7914,  0.2813,  0.2937,
        -1.9542, -0.4639, -1.6314, 0.6094,  0.5912,  -0.2582, -0.1030, 1.2118,
        -0.6067, -1.9866, 1.2060,  2.6261,  1.9976,  -0.1223, 0.8021,  -1.1589,
        1.7606,  1.0414,  1.9002,
    };
    const bias = [_]f32{
        -2.0098, -0.6924, 1.0087, 0.0361, 1.0716,  -1.4818, 1.3093,  0.1560,
        -0.7848, 1.0438,  1.1721, 0.3620, -0.4533, -0.7018, 1.6045,  -1.7152,
        -0.0646, 0.1947,  0.5007, 0.2433, -0.1326, -0.5058, -0.0215, -0.6167,
        0.1258,  0.6529,  2.2678,
    };
    const exp = [_]f32{
        -3.8909, -1.0778, 2.1436, 0.6262, 1.2433,  -0.1070, 1.2353, -0.0106,
        -0.6720, 1.0222,  6.7697, 0.6563, -0.3593, -0.6814, 1.5781, -2.6528,
        0.4203,  -0.5206, 0.7397, 0.0551, -1.6382, -0.6996, 1.0918, -0.8485,
        0.5774,  0.9206,  3.2778,
    };
    layerNorm(f32, layer[0..], &weights, &bias, &tmp);
    for (0..layer.len) |i| {
        const val = layer[i];
        try std.testing.expectApproxEqAbs(exp[i], val, 0.0001);
    }
}

/// Calculate the population variance of `x` using scratchpad space `scratch`.
/// Requires all input slices have the same length.
/// Overwrites any data currently on `scratch`.
pub fn variance(T: type, x: []const T, scratch: []T) f32 {
    floatOnly(T);
    const Vec = comptime Vect(T);
    const vector_len = comptime vectLen(Vec);

    const len = x.len;
    const chunks = len / vector_len;
    const leftover_idx = chunks * vector_len;

    const avg = mean(T, x);
    const mu: Vec = @splat(@as(T, @floatCast(avg)));
    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs = x[idx .. idx + vector_len][0..vector_len].*;

        const diff = xs - mu;
        const square = diff * diff;

        scratch[idx .. idx + vector_len][0..vector_len].* = square;
    }

    for (leftover_idx..len) |i| {
        const diff = x[i] - avg;
        scratch[i] = diff * diff;
    }

    return mean(T, scratch);
}

test "population variance" {
    const xs1 = [_]f32{
        1.5000, -0.5000, 0.2500, -0.3300, 0.5000, -0.2800,
    };
    var scratch1 = [_]f32{0} ** 6;
    try std.testing.expectApproxEqAbs(0.463867, variance(f32, &xs1, &scratch1), 0.0001);

    const xs2 = [_]f32{
        0.782705,    0.0185241,  -1.3987473,  2.558082,    1.0357028,
        -0.26164263, -1.0256457, 0.31200355,  -0.00951786, -0.58689433,
        0.6750471,   0.29773685, -0.46356103, -0.9778625,  -0.83633673,
        -0.14195803, -1.2546383, -0.53481174, -0.745961,   -1.1854576,
    };
    var scratch2 = [_]f32{0} ** 20;
    try std.testing.expectApproxEqAbs(0.8652293, variance(f32, &xs2, &scratch2), 0.0001);
}

/// Calculate the arithmetic mean of `x` with type `T`.
pub fn mean(T: type, x: []const T) f32 {
    floatOnly(T);
    const Vec = comptime Vect(T);
    const vector_len = comptime vectLen(Vec);

    const len = x.len;
    const chunks = len / vector_len;

    const leftover_idx = chunks * vector_len;
    var sum: f32 = 0;
    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
        const xs_sum = @reduce(.Add, xs);
        sum += @floatCast(xs_sum);
    }

    for (leftover_idx..len) |i| {
        sum += @floatCast(x[i]);
    }

    return sum / @as(f32, @floatFromInt(len));
}

test "mean spot check" {
    const xs1 = [_]f32{
        1.5000, -0.5000, 0.2500, -0.3300, 0.5000, -0.2800,
    };
    try std.testing.expectApproxEqAbs(0.19, mean(f32, &xs1), 0.0001);
}

/// Calculate swiglu(x) and store that back into x.
pub fn swiglu(x: []f32) void {
    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

    const chunks = x.len / vector_len;
    const leftover_offset = chunks * vector_len;

    // Calculate σ(x) = x / (1 + e^{-x})

    for (0..chunks) |i| {
        const idx = i * vector_len;
        const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;

        //const exp: Vec = std.math.exp(-xs);
        const exp: Vec = @exp(-xs);
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
pub fn add(comptime T: type, out: []T, x: []const T, y: []const T) void {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len == out.len);

    const Vec = comptime Vect(T);
    const vector_len = comptime vectLen(Vec);

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

    add(f32, &out, &a, &b);
    try std.testing.expectEqualDeep([_]f32{ 5, 7, 9 }, out);
}

/// Add `x` and `y` then store into a slice `out` of `T`.
/// Assumes `x` and `y` to have the same `WeightFormat` or quantization.
/// Assumes `x`, `y`, and `out` to have the same unit lengths.
pub fn addWeights(T: type, out: []T, x: Weights, y: Weights) void {
    same(x, y);
    const x_len = x.len();
    const y_len = y.len();
    std.debug.assert(x_len == y_len);
    std.debug.assert(x_len == out.len);

    switch (x) {
        .f32 => |xx| add(f32, out, xx, y.f32),
        .f16 => |xx| add(f16, out, xx, y.f16),
        .q8_0 => @panic("q8_0"),
        .q6_k => @panic("q6_k"),
        .q4_k => @panic("q4_k"),
    }
}

/// Calculate the element-wise product of `x` and `y` then store the result in `out`.
/// Caller is responsible for ensuring the lengths of `x`, `y`, and `out` are the same.
pub fn elementProduct(out: []f32, x: []const f32, y: []const f32) void {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len == out.len);

    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

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
/// Requires that the quantization format of `m` and `x` are the same. You can check this with
/// `same(m, x)`.
pub fn matrixMulVec(T: type, out: []T, m: Weights, x: Weights, rows: usize, cols: usize) void {
    const x_len = x.len();
    const m_len = m.len();

    std.debug.assert(out.len == rows);
    std.debug.assert(x_len == cols);
    std.debug.assert(m_len == rows * cols);
    std.debug.assert(same(m, x));

    switch (m) {
        .f32 => |mm| matrixMulVec_f32(T, out, mm, x.f32, rows, cols),
        .f16 => @panic("f16"),
        .q8_0 => |mm| matrixMulVec_q8_0(T, out, mm, x.q8_0, rows, cols),
        .q6_k => @panic("q6_k"),
        .q4_k => @panic("q4_k"),
    }
}

/// Matrix multiply for raw f32 weights.
fn matrixMulVec_f32(T: type, out: []T, m: []const f32, x: []const f32, rows: usize, cols: usize) void {
    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

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

        out[row] = @floatCast(sum);
    }
}

/// Matrix multiply for Q8_0 quantized weights.
fn matrixMulVec_q8_0(T: type, out: []T, m: []const Q80Block, x: []const Q80Block, rows: usize, cols: usize) void {
    const Vec = @Vector(32, i32);

    const block_size = comptime blockUnitLen(Q80Block);
    const block_log2 = comptime std.math.log2(block_size);
    std.debug.assert(block_log2 == 5);

    // We can't have leftover weights because both `m` and `x` have chunks of 32 weights because
    // they are slices of `Q80Block`s
    const col_blocks = cols >> block_log2;

    for (0..rows) |row| {
        const m_off = row * col_blocks;

        // For two dequantized number blocks `m` and `n` in Q8_0:
        //     m_a = s_m * x_a
        //     n_a = s_n * y_a
        // Where a ∈ [1, 32].
        // So m is a vector of elements [m1, m2, ..., m32]
        //                     And n is [n1, n2, ..., n32]
        //
        // Then product of the element pair `a` of `m` and `n` is
        //     m_a * n_a = s_m * x_a * s_n * y_a = (s_m*s_n) * (x_a*y_a)
        //
        // So to find the sum of the element-wise product of `m` and `n` that would be:
        // (m_1*n_1) + (m_2*n_2) + ...
        //      = (s_m*x_1 * s_n*y_1) + (s_m*x_2 + s_n*y_2) + ...
        //      = (s_m*s_n * x_1*y_1) + (s_m*s_n + x_2*y_2) + ...
        //      = (s_m*s_n) * ((x_1 * y_1) + (x_2 + y_2) + ...)
        // So we can do an element-wise product then sum the element-wise products to get a partial
        // results which we multiply by the scaling factors s_m and s_n to get the final sum
        // of the scaled products.
        var sum: f32 = 0;
        for (0.., x) |n, block| {
            // for each block in xs, gather the xs.
            const xs: Vec = block.weights;

            const m_idx = m_off + n;
            const m_block = m[m_idx];
            const ms: Vec = m_block.weights;

            const x_scale: f32 = @as(f16, @bitCast(block.scale));
            const m_scale: f32 = @as(f16, @bitCast(m_block.scale));
            const scale: f32 = x_scale * m_scale;

            const prod = xs * ms;
            const pre_sum = @reduce(.Add, prod);
            const prod_sum: f32 = @floatFromInt(pre_sum);
            const final_sum = prod_sum * scale;

            sum += final_sum;
        }
        out[row] = @floatCast(sum);
    }
}

test "matrixMul f32" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 5 };
    var out2 = [_]f32{0} ** 2;

    matrixMulVec(f32, &out2, .{ .f32 = &a }, .{ .f32 = &b }, 2, 3);
    try std.testing.expectEqualDeep([_]f32{ 20, 44 }, out2);

    var d = [_]f32{ 4, -1 };
    var out3 = [_]f32{0} ** 3;
    matrixMulVec(f32, &out3, .{ .f32 = &a }, .{ .f32 = &d }, 3, 2);
    try std.testing.expectEqualDeep([_]f32{ 2, 8, 14 }, out3);

    var m = [_]f32{
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
    var x = [_]f32{ 6, 18, 19, 4, 5, 15, 8, 7, 6, 3, 6, 1 };
    var z = [_]f32{ 9, 15, 7, 16, 10 };

    var out5 = [_]f32{0} ** 5;
    const expect5 = [_]f32{
        2855, 4581, 4242, 5244, 5014,
    };
    matrixMulVec(f32, &out5, .{ .f32 = &m }, .{ .f32 = &x }, 5, 12);
    try std.testing.expectEqualDeep(expect5, out5);

    var out12 = [_]f32{0} ** 12;
    const expect12 = [_]f32{
        2953, 888, 2203, 3501, 4717, 2083, 3491, 3781, 2697, 2964, 2714, 2259,
    };
    matrixMulVec(f32, &out12, .{ .f32 = &m }, .{ .f32 = &z }, 12, 5);
    try std.testing.expectEqualDeep(expect12, out12);
}

/// Find the dot product of two vectors `x` and `y`.
/// Caller is responsible for ensuring that both vectors have equal lengths.
pub fn dotProduct(x: []const f32, y: []const f32) f32 {
    if (x.len != y.len) {
        std.debug.print("x.len {d} != y.len {d}\n", .{ x.len, y.len });
    }
    std.debug.assert(x.len == y.len);

    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

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
