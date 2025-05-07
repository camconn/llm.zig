// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! Math: This module contains math and other helper functions for neural networks.

const std = @import("std");

/// Helper struct for referencing quantized weights.
/// Contains a slice of `Block`s in `WeightFormat` to use when performing calculations.
pub const Weights = union(WeightFormat) {
    f32: []f32,
    f16: []f16,
    q8_0: []Q80Block,
    q6_k: []Q6KBlock,
    q4_k: []Q4KBlock,

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

/// Structure of the `Q8_0` block in GGML. See [1] for more info.
/// A block has 32 weights, and the weight is derived as `w = g * d`.
/// [1]: https://github.com/ggml-org/llama.cpp/blob/2d451c80590b9ac250322769ac13d3b4870dbcf7/ggml/src/ggml-common.h#L213
pub const Q80Block = extern struct {
    /// Scale is an `f16` stored within a `u16`. Conversion to/from the native value is done with
    /// `@bitCast` and expanding to a `f32`.
    scale: u16,
    // QK8_0 = 32 in the ggml reference
    weights: [32]i8,
};

// refer to ggml-common.h for these values
/// Super block size for QK quantization schemes.
const QK_K = 256;
const K_SCALE_SIZE = 12;

/// Structure of a `Q6_K` super-block in GGML. See [1] for definition.
/// A `Q6_K` super-block is 16 sub-blocks of 16 elements each.
/// Each weight is derived as `w = g * a * d`
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-common.h#L320
pub const Q6KBlock = extern struct {
    /// High 4 bits for quantized weights.
    weights_lo: [@divExact(QK_K, 2)]u8,
    /// Low 4 bits for quantized weights.
    weights_hi: [@divExact(QK_K, 4)]u8,
    /// Quantized 8-bit scale.
    scales: [@divExact(QK_K, 16)]i8,
    /// Super-block scale.
    scale: u16,
};

/// Structure of a `Q4_K` super-block in GGML. See [1] for definition.
/// A `Q4_K` super-block is 8 sub-blocks of 32 elements each.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-common.h#L285
pub const Q4KBlock = extern struct {
    /// Superblock aggregate scale
    agg: extern struct {
        d: u16,
        dmin: u16,
    },
    scale: [K_SCALE_SIZE]u8,
    /// Quantized weights. 8 * 32 = 256 weights. Weights are 4-bits wide, so shove to 4 bit weights
    /// into a single byte to get 256 / 2 = 128 blocks.
    weights: [@divExact(QK_K, 2)]u8,
};

/// Get the block format for a given `format` in memory on on disk.
/// For non-f32 type weight blocks, the returned struct must have a `scale` and `weights` field
/// for compatibility with compiled code.
pub fn Block(format: WeightFormat) type {
    return comptime switch (format) {
        .f32 => f32,
        .f16 => f16,
        .q8_0 => Q80Block,
        .q6_k => Q6KBlock,
        .q4_k => Q4KBlock,
    };
}

test "Block() size" {
    const F32Block = Block(.f32);
    try std.testing.expectEqual(@sizeOf(f32), @sizeOf(F32Block));

    const F16Block = Block(.f16);
    try std.testing.expectEqual(@sizeOf(f16), @sizeOf(F16Block));

    // Ensure quantization block sizes match expected values.
    const Q8_0Block = Block(.q8_0);
    try std.testing.expectEqual(@sizeOf(i16) + 32 * @sizeOf(i8), @sizeOf(Q8_0Block));
    try std.testing.expectEqual(34, @sizeOf(Q8_0Block));

    const Q6_KBlock = Block(.q6_k);
    try std.testing.expectEqual(QK_K / 2 + QK_K / 4 + QK_K / 16 + @sizeOf(u16), @sizeOf(Q6_KBlock));
    try std.testing.expectEqual(210, @sizeOf(Q6_KBlock));

    const Q4_KBlock = Block(.q4_k);
    try std.testing.expectEqual(2 * @sizeOf(u16) + K_SCALE_SIZE * @sizeOf(u8) + @sizeOf(i8) * (QK_K / 2), @sizeOf(Q4_KBlock));
    try std.testing.expectEqual(144, @sizeOf(Q4_KBlock));
}

test "Block() element type" {
    inline for (@typeInfo(Weights).@"union".fields) |weight_field| {
        const exp_name = weight_field.name;
        const weights_type = weight_field.type;
        const exp_type = @typeInfo(weights_type).pointer.child;

        const tag_int = comptime for (@typeInfo(WeightFormat).@"enum".fields) |f| {
            if (std.mem.eql(u8, f.name, exp_name)) {
                break f.value;
            }
        };
        const tag: WeightFormat = @enumFromInt(tag_int);

        const actual = Block(tag);
        try std.testing.expectEqual(exp_type, actual);
    }
}

/// Get the unit length of a `Block()`.
pub fn blockUnitLen(BlockType: type) comptime_int {
    if (BlockType == f32 or BlockType == f16) {
        return 1;
    }

    return switch (BlockType) {
        Q80Block => {
            const struct_info = @FieldType(BlockType, "weights");
            const array_type = @typeInfo(struct_info).array;
            return array_type.len;
        },
        Q6KBlock, Q4KBlock => QK_K,
        else => @compileError("blockUnitLen unimplemented for " ++ @typeName(BlockType)),
    };
}

test "block unit length" {
    try std.testing.expectEqual(1, blockUnitLen(Block(.f32)));
    try std.testing.expectEqual(1, blockUnitLen(Block(.f16)));
    try std.testing.expectEqual(32, blockUnitLen(Block(.q8_0)));
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

fn floatOnly(T: type) void {
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

    // Poor man's dynamic dispatch
    switch (m) {
        .f32 => matrixMulVec_f32(T, out, m.f32, x.f32, rows, cols),
        .f16 => @panic("f16"),
        .q8_0 => matrixMulVec_q8_0(T, out, m.q8_0, x.q8_0, rows, cols),
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

/// Quantize weights `ws` from full-size `f32` to their quantized forms.
/// Caller is responsible for freeing the returned Quantized block array.
pub fn quantize(
    comptime format: WeightFormat,
    ws: []const f32,
    out: []Block(format),
) !f32 {
    if (ws.len == 0) {
        return error.Empty;
    }

    // No conversion necessary
    if (format == .f32 or format == .f16) {
        @memcpy(out, ws);
        return 0;
    }

    //const BlockType = Block(format);
    //const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .weights).type).array;
    //const block_size = array_info.len;
    //std.debug.assert(ws.len % block_size == 0);

    const err = switch (format) {
        .q8_0 => quantize_q8_0(ws, out),
        .q6_k => quantize_q6_k(ws, out),
        .q4_k => quantize_q4_k(ws, out),
        else => @compileError("quantize method is unimplemented for " ++ @tagName(format)),
    };

    // TODO: Do something with the quantization error.
    return err;
}

const max_magnitude = struct {
    pub fn lessThan(_: void, lhs: f32, rhs: f32) bool {
        const a = @abs(lhs);
        const b = @abs(rhs);
        return a < b;
    }
};

test "max_magnitude lessThan" {
    const elems = [_]f32{ -5, -10, 3, 7 };
    try std.testing.expectEqual(-10, std.sort.max(f32, &elems, {}, max_magnitude.lessThan).?);
    const elems2 = [_]f32{ -5, -29.9999, 3, 7, 30 };
    try std.testing.expectEqual(30, std.sort.max(f32, &elems2, {}, max_magnitude.lessThan).?);
}

// TODO: Vectorize this
/// Quantize `f32` values into `blocks` with the `Q8_0` quantization format.
/// Refer to `quantize_row_q8_0_ref` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/489716ba99ecd51164f79e8c6fec0b5bf634eac9/src/ggml-quants.c#L194
fn quantize_q8_0(in: []const f32, blocks: []Q80Block) f32 {
    const n_blocks = blocks.len;
    const BlockType = Block(.q8_0);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .weights).type).array;
    const Elem = array_info.child;
    const block_size = array_info.len;

    std.debug.assert(@mod(in.len, block_size) == 0);
    std.debug.assert(@divExact(in.len, block_size) == blocks.len);

    // Calculate max error
    var err: f32 = 0;

    //const min_val: @Vector(32, i8) = @splat(-Q80_MAX);
    //const max_val: @Vector(32, i8) = @splat(Q80_MAX);

    for (0..n_blocks) |i| {
        const idx = i * block_size;

        const elems: []const f32 = in[idx .. idx + block_size];

        const max: f32 = std.sort.max(f32, elems, {}, max_magnitude.lessThan) orelse 0;
        const scale = @abs(max / 127);
        std.debug.assert(scale >= 0);

        const scale_shrank: f16 = @floatCast(scale);
        blocks[i].scale = @bitCast(scale_shrank);
        // In the GGML reference the scale is inverted so that a cheaper multiply op can be
        // used element-wise.
        const inv_scale = if (scale == 0) 0 else 1 / scale; // ternary because divide-by-zero

        for (0..block_size, elems) |j, old| {
            const scaled = old * inv_scale;
            // Ensure we are within 8 bits in a symmetric range.
            const rounded = std.math.round(std.math.clamp(scaled, -128, 127));

            // Save quantized weight
            const quantized: Elem = @intFromFloat(rounded);
            blocks[i].weights[j] = quantized;

            // Calculate maximum error for diagnostics
            const elem_err = @abs(@as(f32, @floatFromInt(quantized)) * inv_scale - old);
            err = @max(err, elem_err);
        }
    }
    return err;
}

const group_max_eps: comptime_float = 1e-15;

// TODO: vectorize this
/// Quantize `f32` values into `blocks` with the `Q6_K` quantization format.
/// Refer to `quantize_row_q6_K_ref` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1620
fn quantize_q6_k(in: []const f32, blocks: []Q6KBlock) f32 {
    std.debug.assert(@mod(in.len, QK_K) == 0);
    std.debug.assert(@divExact(in.len, QK_K) == blocks.len);

    const n_blocks = blocks.len; // `nb` in ggml reference
    const n_subblocks = comptime @divExact(QK_K, 16); // number of subblocks within a block
    const BlockType = Q6KBlock;

    var limit = [_]i8{0} ** QK_K;
    var scales = [_]f32{0} ** n_subblocks;

    // Iterate superblocks
    super: for (0..n_blocks) |i| {
        const orig = in[i * QK_K .. (i + 1) * QK_K][0..QK_K];
        // Iterate inner blocks
        var max_scale: f32 = 0;
        var max_scale_mag: f32 = 0;

        for (0..n_subblocks) |j| {
            const orig_subblock = orig[j * 16 .. (j + 1) * 16];
            var out_subblock = limit[j * 16 .. (j + 1) * 16];

            const scale = make_qx_quants(32, orig_subblock, out_subblock[0..16], 1, null);
            scales[j] = scale;

            const scale_mag = @abs(scale);
            if (scale_mag > max_scale_mag) {
                max_scale = scale;
                max_scale_mag = scale_mag;
            }
        }

        if (max_scale_mag < group_max_eps) {
            blocks[i] = std.mem.zeroes(BlockType);
            continue :super;
        }

        const iscale: f32 = -128 / max_scale;
        blocks[i].scale = @bitCast(@as(f16, @floatCast(1 / iscale)));
        for (0..n_subblocks) |j| {
            const rounded: i32 = @intFromFloat(std.math.round(iscale * scales[j]));
            blocks[i].scales[j] = @intCast(@min(127, rounded));
        }

        inner: for (0..n_subblocks) |j| {
            const d = @as(f32, @as(f16, @bitCast(blocks[i].scale))) * @as(f32, @floatFromInt(blocks[i].scales[j]));
            if (d == 0) {
                continue :inner;
            }

            for (0..16) |k| {
                const x = orig[16 * j + k];
                var l: i32 = @intFromFloat(std.math.round(x / d));
                l = @max(-32, @min(31, l));
                limit[16 * j + k] = @intCast(l + 32);
            }
        }

        const ql = &blocks[i].weights_lo;
        const qh = &blocks[i].weights_hi;
        for (0..@divExact(QK_K, 128)) |ii| {
            const j = ii * 128;

            const off_lo = ii * 64;
            const off_hi = ii * 32;

            for (0..32) |l| {
                // Unambiguously safe, since we are masking with the lower nibble, the upper
                // nibble (and the sign bit) are always unset, making this positive.
                const q1: u8 = @bitCast(limit[j + l + 0] & 0x0f);
                const q2: u8 = @bitCast(limit[j + l + 32] & 0x0f);
                const q3: u8 = @bitCast(limit[j + l + 64] & 0x0f);
                const q4: u8 = @bitCast(limit[j + l + 96] & 0x0f);

                ql.*[off_lo + l + 0] = q1 | (q3 << 4);
                ql.*[off_lo + l + 32] = q2 | (q4 << 4);

                // Safe as above, because we are shifting within a single byte.
                const l0: u8 = @bitCast(limit[j + l] >> 4);
                const l1: u8 = @bitCast(limit[j + l + 32] >> 4);
                const l2: u8 = @bitCast(limit[j + l + 64] >> 4);
                const l3: u8 = @bitCast(limit[j + l + 96] >> 4);
                const tmp = (l0 << 0) | (l1 << 2) | (l2 << 4) | (l3 << 6);
                qh.*[off_hi + l] = @intCast(tmp);
            }
        }
        // TODO: Calculate error from each reconstructed element and report max error
    }
    return 0;
}

// reimplementation of `make_qx_quants` in `ggml-quants.c`
// differs from orig because `n` is implicit.
/// Quantize a sub-block `in` to the range `[-n_max, nmax-1]` and write to `out`.
fn make_qx_quants(n_max: i32, in: []const f32, out: []i8, rmse_type: i8, qw: ?[]const f32) f32 {
    const n = in.len;
    var max: f32 = 0;
    var amax: f32 = 0;
    var rmse = rmse_type;

    for (in) |x| {
        const ax = @abs(x);
        if (ax > amax) {
            amax = ax;
            max = x;
        }
    }

    if (amax < group_max_eps) {
        @memset(out[0..n], 0);
        return 0;
    }

    var iscale = @as(f32, @floatFromInt(-n_max)) / max;
    if (rmse == 0) {
        for (0..n, in) |i, x| {
            const l: i32 = @intFromFloat(@round(iscale * x));
            out[i] = @intCast(n_max + std.math.clamp(l, -n_max, n_max - 1));
        }
        return 1 / iscale;
    }

    var return_early: bool = false;
    if (rmse < 0) {
        rmse = -rmse;
        return_early = true;
    }

    var sumlx: f32 = 0;
    var suml2: f32 = 0;
    for (0..n, in) |i, x| {
        var l: i32 = @intFromFloat(@round(iscale * x));
        // TODO: This doesn't seem right. The orig code clamps to [-128, 127] then immediately adds
        //       back 128. Perhaps this okay because of the runtime domain of possible values?
        l = std.math.clamp(l, -n_max, n_max - 1);
        out[i] = @intCast(l + n_max);

        // zig fmt: off
        const w: f32 = if (qw) |inner| inner[i] else
            if (rmse == 1) x * x else
            if (rmse == 2) 1 else
            if (rmse == 3) @abs(x) else
            @sqrt(@abs(x));
        // zig fmt: on
        sumlx += w * @as(f32, @floatFromInt(l)) * x;
        suml2 += w * @as(f32, @floatFromInt(l * l));
    }

    var scale: f32 = if (suml2 == 0) 0 else sumlx / suml2;
    if (return_early) {
        if (suml2 > 0) {
            return 0.5 * (scale + 1 / iscale);
        } else {
            return 1 / iscale;
        }
    }

    var best: f32 = scale * sumlx;
    // orig is `for (int is = -9; is <= 9; ++is)`
    // below replicates that by iteration from [0..18] then subtracting iterand by 9 to be [-9..9]
    for (0..19) |s| {
        const is = @as(isize, @intCast(s)) - 9;
        if (is == 0) continue;

        const is_f: f32 = @floatFromInt(is);

        iscale = -(@as(f32, @floatFromInt(n_max)) + 0.1 * is_f) / max;
        sumlx = 0;
        suml2 = 0;

        for (0..n, in) |i, x| {
            var l: i32 = @intFromFloat(@round(iscale * x));
            l = std.math.clamp(l, -n_max, n_max - 1);
            // zig fmt: off
            const w = if (qw) |inner| inner[i] else
                if (rmse == 1) x * x else
                if (rmse == 2) 1 else
                if (rmse == 3) @abs(x) else
                @sqrt(@abs(x));
            // zig fmt: on
            sumlx += w * x * @as(f32, @floatFromInt(l));
            suml2 += w * @as(f32, @floatFromInt(l * l));
        }

        if (suml2 > 0 and sumlx * sumlx > best * suml2) {
            for (0..n, in) |i, x| {
                const l: i32 = @intFromFloat(std.math.round(iscale * x));
                const l_clamp = std.math.clamp(l, -n_max, n_max - 1);
                out[i] = @intCast(n_max + l_clamp);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// TODO: vectorize this
/// Quantize `f32` values into `blocks` with the `Q4_K` quantization format.
/// Refer to `quantize_row_q4_K_ref` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1208
fn quantize_q4_k(in: []const f32, blocks: []Q4KBlock) f32 {
    std.debug.assert(@mod(in.len, QK_K) == 0);
    std.debug.assert(@divExact(in.len, QK_K) == blocks.len);

    const n_blocks = blocks.len;

    var limit = [_]u8{0} ** QK_K;
    var limit_aux = [_]u8{0} ** 32;
    var weights = [_]f32{0} ** 32;
    var mins = [_]f32{0} ** @divExact(QK_K, 32);
    var scales = [_]f32{0} ** @divExact(QK_K, 32);

    for (0..n_blocks, blocks) |i, *out_block| {
        const block = in[i * QK_K ..][0..QK_K];

        var max_scale: f32 = 0;
        var max_min: f32 = 0;

        for (0..@divExact(QK_K, 32)) |j| {
            var sum_x2: f32 = 0;
            for (0..32) |l| {
                const x = block[32 * j + l];
                sum_x2 += x * x;
            }
            const root_avg = @sqrt(sum_x2 / 32);
            for (0..32) |l| {
                weights[l] = root_avg + @abs(block[j * 32 + l]);
            }
            scales[j] = make_qkx2_quants(15, block[j * 32 ..][0..32], &weights, limit[j * 32 ..][0..32], &mins[j], &limit_aux, -1, 0.1, 20, false);

            const scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            const min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        // zig fmt: off
        const inv_scale = if (max_scale > 0) 63 / max_scale else 0;
        const inv_min =   if (max_min > 0)   63 / max_min   else 0;
        // zig fmt: on
        for (0..@divExact(QK_K, 32)) |j| {
            var ls: u8 = @intFromFloat(std.math.round(inv_scale * scales[j]));
            var lm: u8 = @intFromFloat(std.math.round(inv_min * mins[j]));
            ls = @min(63, ls);
            lm = @min(63, lm);

            if (j < 4) {
                out_block.scale[j] = ls;
                out_block.scale[j + 4] = lm;
            } else {
                out_block.scale[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                out_block.scale[j - 4] |= ((ls >> 4) << 6);
                out_block.scale[j - 0] |= ((lm >> 4) << 6);
            }
        }

        // zig fmt: off
        out_block.agg.d    = @bitCast(@as(f16, @floatCast(max_scale / 63)));
        out_block.agg.dmin = @bitCast(@as(f16, @floatCast(max_min   / 63)));
        // zig fmt: on

        var sc: u8 = undefined;
        var m: u8 = undefined;
        inner: for (0..@divExact(QK_K, 32)) |j| {
            get_scale_min_k4(j, &out_block.scale, &sc, &m);
            const d = @as(f32, @floatCast(@as(f16, @bitCast(out_block.agg.d)))) * @as(f32, @floatFromInt(sc));

            if (d == 0) continue :inner;

            const dm = @as(f32, @floatCast(@as(f16, @bitCast(out_block.agg.dmin)))) * @as(f32, @floatFromInt(m));

            for (0..32) |k| {
                var l: i32 = @intFromFloat(std.math.round((block[32 * j + k] + dm) / d));
                l = std.math.clamp(l, 0, 15);
                limit[32 * j + k] = @intCast(l);
            }
        }

        for (0..@divExact(QK_K, 64)) |jj| {
            const j = jj * 64;
            var q = out_block.weights[jj * 32 .. (jj + 1) * 32];

            for (0..32) |l| {
                q[l] = limit[j + l] | (limit[j + l + 32] << 4);
            }
        }
    }

    // TODO: Calculate quantization error
    return 0;
}

/// Port of implementation of `get_scale_min_k4` from ggml [1].
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L631
inline fn get_scale_min_k4(j: usize, q: []const u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        // zig fmt: off
        d.* = (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4);
        m.* = (q[j+4] >>   4) | ((q[j-0] >> 6) << 4);
        // zig fmt: on
    }
}

/// This is a straightforward port of `make_qkx2_quants` from ggml [1].
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L550
fn make_qkx2_quants(
    n_max: i64,
    x: []const f32,
    weights: []f32,
    limit: []u8,
    the_min: *f32,
    limit_aux: []u8,
    r_min: f32,
    r_delta: f32,
    n_step: isize,
    use_mad: bool,
) f32 {
    const n = x.len;
    var min = x[0];
    var max = x[0];
    var sum_w = weights[0];
    var sum_x = sum_w * x[0];

    for (1..n, x[1..]) |i, xx| {
        if (xx < min) {
            min = xx;
        }
        if (xx > max) {
            max = xx;
        }

        const w = weights[i];
        sum_w += w;
        sum_x += w * xx;
    }

    min = @min(min, 0);
    if (max == min) {
        for (0..n) |i| {
            limit[i] = 0;
        }

        the_min.* = -min;
        return 0;
    }

    var iscale = @as(f32, @floatFromInt(n_max)) / (max - min);
    var scale = 1 / iscale;
    var best_mad: f32 = 0;
    for (0..n, x) |i, xx| {
        const l: i64 = @intFromFloat(std.math.round(iscale * (xx - min)));
        limit[i] = @intCast(std.math.clamp(l, 0, n_max));
        var diff = scale * @as(f32, @floatFromInt(limit[i])) + min - xx;
        diff = if (use_mad) @abs(diff) else diff * diff;
        const w = weights[i];
        best_mad += w * diff;
    }
    if (n_step < 1) {
        the_min.* = -min;
        return scale;
    }

    const un_step: usize = @intCast(n_step);
    for (0..(un_step + 1)) |is| {
        iscale = (r_min + r_delta * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(n_max))) / (max - min);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;

        for (0..n, x) |i, xx| {
            var l: i64 = @intFromFloat(std.math.round(iscale * (xx - min)));
            l = std.math.clamp(l, 0, n_max);

            limit_aux[i] = @intCast(l);
            const w = weights[i];
            sum_l += w * @as(f32, @floatFromInt(l));
            sum_l2 += w * @as(f32, @floatFromInt(l * l));
            sum_xl += w * @as(f32, @floatFromInt(l)) * xx;
        }

        const D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            // zig fmt: off
            var this_scale = (sum_w  * sum_xl - sum_x * sum_l ) / D;
            var this_min =   (sum_l2 * sum_x  - sum_l * sum_xl) / D;
            // zig fmt: on
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }

            var mad: f32 = 0;
            for (0..n, x) |i, xx| {
                var diff = this_scale * @as(f32, @floatFromInt(limit[i])) + this_min - xx;
                diff = if (use_mad) @abs(diff) else diff * diff;
                const w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (0..n) |i| {
                    limit[i] = limit_aux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    the_min.* = -min;
    return scale;
}

test "quantize weights" {
    const empty: [0]f32 = [0]f32{};
    try std.testing.expectError(error.Empty, quantize(.f32, &empty, &empty));

    var f32_out = [_]f32{0} ** 5;
    const f32s = [_]f32{ 1, 2, 3, 4, 5 };
    const f32_quant = try quantize(.f32, &f32s, &f32_out);
    try std.testing.expectEqualSlices(f32, &f32s, &f32_out);
    try std.testing.expectEqual(0, f32_quant);

    const no_scale = [_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };
    const no_scale_exp = [32]i8{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };
    var out_q80 = [_]Q80Block{std.mem.zeroes(Q80Block)};

    const q80_no_scale = try quantize(.q8_0, &no_scale, &out_q80);
    try std.testing.expectEqual(0, q80_no_scale);
    try std.testing.expectEqualSlices(i8, &no_scale_exp, &out_q80[0].weights);
}

/// De-quantize weights
pub fn dequantize(
    comptime format: WeightFormat,
    weights: []const Block(format),
    out: []f32,
) !void {
    try dequantizeT(f32, format, weights, out);
}

/// De-quantize weights to an array of type `T`.
pub fn dequantizeT(
    T: type,
    comptime format: WeightFormat,
    weights: []const Block(format),
    out: []T,
) !void {
    floatOnly(T);
    if (weights.len == 0) {
        return error.Empty;
    }

    if (format == .f32 or format == .f16) {
        return;
    }

    switch (format) {
        .q8_0 => dequantize_q8_0(T, weights, out),
        .q6_k => dequantize_q6_k(T, weights, out),
        .q4_k => dequantize_q4_k(T, weights, out),
        else => @compileError("dequantize method is unimplemented for " ++ @typeName(format)),
    }
}

// TODO: Vectorize this
/// Dequantize `in` from the `Q8_0` quantization format into `T` float values.
/// Refer to `dequantize_row_q8_0` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/489716ba99ecd51164f79e8c6fec0b5bf634eac9/src/ggml-quants.c#L349
fn dequantize_q8_0(T: type, in: []const Q80Block, out: []T) void {
    const BlockType = Block(.q8_0);
    const block_size = blockUnitLen(BlockType);

    for (0..in.len) |i| {
        const block = in[i];

        const idx = block_size * i;

        const scale: f32 = @as(f16, @bitCast(block.scale));
        std.debug.assert(scale >= 0);
        for (0..block_size) |j| {
            const elem = block.weights[j];
            const elem_f: f32 = @floatFromInt(@as(i32, elem));
            out[idx + j] = @floatCast(elem_f * scale);
        }
    }
}

// TODO: Vectorize this
/// Dequantize `in` from the `Q6_K` quantization format into `T` float values.
/// Refer to `dequantize_row_q6_K` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1690
fn dequantize_q6_k(T: type, in: []const Q6KBlock, out: []T) void {
    const BlockType = Q6KBlock;
    const block_size = blockUnitLen(BlockType);

    std.debug.assert(@mod(out.len, block_size) == 0);
    std.debug.assert(@divExact(out.len, block_size) == in.len);

    for (0.., in) |i, block| {
        const offset = i * block_size;

        const superblock_scale: f32 = @as(f16, @bitCast(block.scale));

        const w_lo = block.weights_lo;
        const w_hi = block.weights_hi;
        const scales = block.scales;

        for (0..@divExact(QK_K, 128)) |j| {
            const n = j * 128;
            const y = &out[offset + n ..][0..128];

            const lo = w_lo[j * 64 .. (j + 1) * 64];
            const hi = w_hi[j * 32 .. (j + 1) * 32];

            const sc = scales[j * 8 .. (j + 1) * 8];

            for (0..32) |l| {
                const is = l / 16;

                // zig fmt: off
                const q1: i8 = @intCast((lo[l +  0] & 0x0f) | (((hi[l] >> 0) & 3) << 4));
                const q2: i8 = @intCast((lo[l + 32] & 0x0f) | (((hi[l] >> 2) & 3) << 4));
                const q3: i8 = @intCast((lo[l +  0]   >> 4) | (((hi[l] >> 4) & 3) << 4));
                const q4: i8 = @intCast((lo[l + 32]   >> 4) | (((hi[l] >> 6) & 3) << 4));
                // zig fmt: on

                const sc1: f32 = @floatFromInt(sc[is + 0]);
                const sc2: f32 = @floatFromInt(sc[is + 2]);
                const sc3: f32 = @floatFromInt(sc[is + 4]);
                const sc4: f32 = @floatFromInt(sc[is + 6]);

                const y0 = superblock_scale * sc1 * @as(f32, @floatFromInt(q1 -% 32));
                const y1 = superblock_scale * sc2 * @as(f32, @floatFromInt(q2 -% 32));
                const y2 = superblock_scale * sc3 * @as(f32, @floatFromInt(q3 -% 32));
                const y3 = superblock_scale * sc4 * @as(f32, @floatFromInt(q4 -% 32));

                // zig fmt: off
                y.*[l +  0] = @floatCast(y0);
                y.*[l + 32] = @floatCast(y1);
                y.*[l + 64] = @floatCast(y2);
                y.*[l + 96] = @floatCast(y3);
                // zig fmt: on
            }
        }
    }
}

// TODO: Vectorize this
/// Dequantize `in` from the `Q4_K` quantization format into `T` float values.
/// Refer to `dequantize_row_q4_K` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1280
fn dequantize_q4_k(T: type, in: []const Q4KBlock, out: []T) void {
    const BlockType = Q4KBlock;
    const block_size = blockUnitLen(BlockType);

    std.debug.assert(@mod(out.len, block_size) == 0);
    std.debug.assert(@divExact(out.len, block_size) == in.len);

    for (0.., in) |i, *block| {
        var out_block = out[i * QK_K ..][0..QK_K];

        const weights = block.weights;

        const d: f32 = @as(f16, @bitCast(block.agg.d));
        const min: f32 = @as(f16, @bitCast(block.agg.dmin));

        var is: usize = 0;
        var sc: u8 = undefined;
        var m: u8 = undefined;
        for (0..@divExact(QK_K, 64)) |jj| {
            const j = jj * 64;

            const q = weights[jj * 32 .. (jj + 1) * 32][0..32];

            // zig fmt: off
            get_scale_min_k4(is + 0, &block.scale, &sc, &m);
            const d1 = d   * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));
            get_scale_min_k4(is + 1, &block.scale, &sc, &m);
            const d2 = d   * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));
            // zig fmt: on

            for (0..32) |l| {
                const result = d1 * @as(f32, @floatFromInt(q[l] & 0x0f)) - m1;
                out_block[j + l] = @floatCast(result);
            }
            for (0..32) |l| {
                const result = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
                out_block[j + l + 32] = @floatCast(result);
            }
            is += 2;
        }
    }
}

test "dequantize weights" {
    var out = [_]f32{0} ** 32;
    const empty = [0]f32{};
    try std.testing.expectError(error.Empty, dequantize(.f32, &empty, &out));
    const empty_q80 = [0]Block(.q8_0){};
    try std.testing.expectError(error.Empty, dequantize(.q8_0, &empty_q80, &out));

    const no_scale = [_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };

    var out_quant = [_]Block(.q8_0){std.mem.zeroes(Q80Block)};
    const q80_no_scale = try quantize(.q8_0, &no_scale, &out_quant);
    try std.testing.expectEqual(0, q80_no_scale);
    try dequantize(.q8_0, &out_quant, &out);
    try std.testing.expectEqualSlices(f32, &no_scale, &out);
}

test "dequantize(quantize(input)) q6_k and q4_k" {
    const input = [_]f32{
        0.26818057, 0.70117421, 0.80591067, 0.23019514, 0.14670639,
        0.33391678, 0.52973484, 0.45318381, 0.64022398, 0.28994352,
        0.57440085, 0.42841502, 0.00551542, 0.40409434, 0.84484653,
        0.44737177, 0.75096818, 0.81511407, 0.07341571, 0.82394374,
        0.29827766, 0.87487497, 0.13257837, 0.60109425, 0.95626913,
        0.99961994, 0.42491646, 0.73456629, 0.91688578, 0.96441928,
        0.98533557, 0.29554028, 0.01003327, 0.48162975, 0.29975987,
        0.91495502, 0.06700391, 0.68019735, 0.73848713, 0.90355828,
        0.63185128, 0.57677257, 0.4724628,  0.22348524, 0.40015188,
        0.01129494, 0.2232572,  0.63723492, 0.44318981, 0.80601651,
        0.76156629, 0.04124412, 0.35788252, 0.28793633, 0.19582641,
        0.89765805, 0.36163506, 0.94240101, 0.71120837, 0.67068579,
        0.30092527, 0.35680559, 0.45040693, 0.13315413, 0.29537174,
        0.81972883, 0.59423547, 0.87172982, 0.33067438, 0.49248542,
        0.57059288, 0.65961179, 0.89193526, 0.40634174, 0.20658933,
        0.22735729, 0.09656759, 0.78862714, 0.48417093, 0.65754907,
        0.27845261, 0.98663917, 0.59444545, 0.97350248, 0.47614487,
        0.14190687, 0.60782982, 0.6795271,  0.02671968, 0.34116549,
        0.85415479, 0.29213184, 0.41548042, 0.270123,   0.96003289,
        0.6622112,  0.98637266, 0.26190177, 0.48180395, 0.55795777,
        0.68291425, 0.98164342, 0.45446764, 0.17571396, 0.23874598,
        0.86202598, 0.48569754, 0.50599026, 0.16404926, 0.52368819,
        0.94225083, 0.43053954, 0.72541556, 0.70928258, 0.66261289,
        0.30840945, 0.35910923, 0.84805918, 0.49627614, 0.89978007,
        0.44581439, 0.35289054, 0.680217,   0.15880314, 0.40756339,
        0.39369585, 0.51360553, 0.29971385, 0.0489201,  0.09508946,
        0.19150976, 0.7781581,  0.21683124, 0.24743186, 0.82869239,
        0.77427872, 0.60696236, 0.52352254, 0.71968511, 0.41509592,
        0.75652485, 0.00477548, 0.44399738, 0.0083799,  0.06323526,
        0.9399441,  0.50391666, 0.83552948, 0.86409024, 0.21936696,
        0.73748613, 0.15461877, 0.75467339, 0.25872051, 0.95171161,
        0.49709519, 0.45294659, 0.05595667, 0.23729637, 0.3410649,
        0.81303617, 0.45054132, 0.83502449, 0.79948996, 0.43303826,
        0.24817904, 0.12250606, 0.48479977, 0.49849821, 0.55752783,
        0.08789795, 0.3937892,  0.45096781, 0.9832385,  0.69866774,
        0.27155947, 0.20381776, 0.2952266,  0.79640404, 0.5835283,
        0.76363749, 0.35736668, 0.02409709, 0.1042234,  0.25845312,
        0.88851827, 0.98197504, 0.41783902, 0.97035122, 0.34065692,
        0.47798785, 0.49641774, 0.60678651, 0.89854009, 0.09507555,
        0.94297451, 0.29527169, 0.31350958, 0.06369393, 0.79253947,
        0.43072594, 0.40610705, 0.26861318, 0.86756347, 0.70624882,
        0.640259,   0.6834766,  0.78367339, 0.34724362, 0.7311883,
        0.07728095, 0.78759387, 0.46740117, 0.01972796, 0.94352436,
        0.30881627, 0.70998402, 0.98331121, 0.72531005, 0.64998247,
        0.43503533, 0.89006498, 0.38304409, 0.05804393, 0.21076824,
        0.00841365, 0.83931816, 0.76405042, 0.26303586, 0.58260129,
        0.43186932, 0.34278714, 0.98516938, 0.11636042, 0.60443262,
        0.74488765, 0.9179661,  0.70137441, 0.32111254, 0.92630706,
        0.34251309, 0.97879491, 0.71861436, 0.63012254, 0.63813117,
        0.52341741, 0.3441749,  0.57462094, 0.24646284, 0.57768118,
        0.78210324, 0.21352484, 0.27498262, 0.63415368, 0.95555718,
        0.11141753,
    };

    // Check that implementations are correct by quantizing and then dequantizing, then check
    // the root mean square error.
    // check q6_k
    var q6_out = [_]Q6KBlock{std.mem.zeroes(Q6KBlock)} ** 1;
    _ = try quantize(.q6_k, &input, q6_out[0..]);
    var dequantized_q6 = [_]f32{0} ** 256;
    _ = try dequantizeT(f32, .q6_k, &q6_out, dequantized_q6[0..]);
    var sum_err2_q6: f32 = 0;
    for (0.., input) |i, expected| {
        const diff = expected - dequantized_q6[i];
        try std.testing.expectApproxEqAbs(expected, dequantized_q6[i], 0.1);
        sum_err2_q6 += diff * diff;
    }
    const rmse_q6 = @sqrt(sum_err2_q6 / input.len);
    try std.testing.expect(rmse_q6 < 0.01);

    // check q4_k
    var out_q4 = [_]Q4KBlock{std.mem.zeroes(Q4KBlock)};
    _ = try quantize(.q4_k, &input, &out_q4);
    var dequantized_q4 = [_]f32{0} ** QK_K;
    try dequantize(.q4_k, &out_q4, &dequantized_q4);

    var sum_err2_q4: f32 = 0;
    for (0.., input) |i, value| {
        const diff = @abs(value - dequantized_q4[i]);
        try std.testing.expect(diff < 0.15);
        sum_err2_q4 += diff * diff;
    }
    const rmse_q4 = @sqrt(sum_err2_q4 / input.len);
    try std.testing.expect(rmse_q4 < 0.02);
}
