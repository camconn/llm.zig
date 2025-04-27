// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! Math: This module contains math and other helper functions for neural networks.

const std = @import("std");

/// Helper struct for referencing quantized weights.
/// Contains a slice of `Block`s in `WeightFormat` to use when performing calculations.
pub const Weights = union(WeightFormat) {
    f32: []Block(.f32),
    q8_0: []Block(.q8_0),
};

/// Quantization types for model weights.
pub const WeightFormat = enum {
    /// Uncompressed weights in `f32` form.
    /// This is not actually a quantization, but a pseudo-type for internal purposes.
    /// Blocks of `f32` are just the individual `f32` weights.
    f32,
    /// Represents 8 bit weights scaled by a single precision float `d` for every 32 weights.
    /// The original weight `w = i * d` where `i` is the quantized integer and `d` is the scale.
    q8_0,
};

/// Compare `lhs` and `rhs` to see if they have the same inner quantization format.
inline fn same(lhs: Weights, rhs: Weights) bool {
    const a: WeightFormat = lhs;
    const b: WeightFormat = rhs;
    return a == b;
}

/// On-disk structure of the serialized `Q8_0` format in GGML. See [1] for more info.
/// [1]: https://github.com/ggml-org/llama.cpp/blob/2d451c80590b9ac250322769ac13d3b4870dbcf7/ggml/src/ggml-common.h#L213
const Q80Block = extern struct {
    scale: f32,
    // QK8_0 = 32 in the ggml reference
    weights: [32]i8,
};

/// Get the block format for a given `format` in memory on on disk.
/// For non-f32 type weight blocks, the returned struct must have a `scale` and `weights` field
/// for compatibility with compiled code.
pub fn Block(format: WeightFormat) type {
    return switch (format) {
        .f32 => f32,
        .q8_0 => Q80Block,
    };
}

test "Block() size" {
    const F32Block = Block(.f32);
    try std.testing.expectEqual(@sizeOf(f32), @sizeOf(F32Block));

    // Ensure Q8_0 compatibility when reading from a GGUF file.
    const Q8_0Block = Block(.q8_0);
    try std.testing.expectEqual(@sizeOf(f32) + 32 * @sizeOf(i8), @sizeOf(Q8_0Block));
}

/// Get the unit length of a `Block()`.
pub fn blockUnitLen(BlockType: type) comptime_int {
    if (BlockType == f32) {
        return 1;
    }

    const struct_info = @FieldType(BlockType, "weights");
    const array_type = @typeInfo(struct_info).array;
    return array_type.len;
}

test "block unit length" {
    try std.testing.expectEqual(1, blockUnitLen(Block(.f32)));
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

    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

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
    const Vec = comptime Vect(f32);
    const vector_len = comptime vectLen(Vec);

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
pub fn matrixMul(out: []f32, m: Weights, x: Weights, rows: usize, cols: usize) void {
    const x_len = switch (x) {
        .f32 => |f| f.len,
        .q8_0 => |q| q.len * blockUnitLen(Block(.q8_0)),
    };
    const m_len = switch (m) {
        .f32 => |f| f.len,
        .q8_0 => |q| q.len * blockUnitLen(Block(.q8_0)),
    };

    std.debug.assert(out.len == rows);
    std.debug.assert(x_len == cols);
    std.debug.assert(m_len == rows * cols);
    std.debug.assert(same(m, x));

    // Poor man's dynamic dispatch
    switch (m) {
        .f32 => matrixMul_f32(out, m.f32, x.f32, rows, cols),
        .q8_0 => matrixMul_q8_0(out, m.q8_0, x.q8_0, rows, cols),
    }
}

/// Matrix multiply for raw f32 weights.
pub fn matrixMul_f32(out: []f32, m: []const f32, x: []const f32, rows: usize, cols: usize) void {
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

        out[row] = sum;
    }
}

/// Matrix multiply for Q8_0 quantized weights.
fn matrixMul_q8_0(out: []f32, m: []const Q80Block, x: []const Q80Block, rows: usize, cols: usize) void {
    const Vec = @Vector(32, i32);

    const block_size = comptime blockUnitLen(Q80Block);
    const block_log2 = comptime std.math.log2(block_size);

    // We can't have leftover weights because both `m` and `x` have chunks of 32 weights because
    // they are slices of `Q80Block`s

    for (0..rows) |row| {
        const m_off = row * cols;
        const m_start_block = m_off >> block_log2;

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

            const m_idx = m_start_block + n;
            const m_block = m[m_idx];
            const ms: Vec = m_block.weights;

            const x_scale = block.scale;
            const m_scale = m_block.scale;

            const prod = xs * ms;
            const pre_sum = @reduce(.Add, prod);
            const prod_f: f32 = @floatFromInt(pre_sum);
            const final_sum = prod_f * x_scale * m_scale;

            sum += final_sum;
        }
        out[row] = sum;
    }
}

test "matrixMul f32" {
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 5 };
    var out2 = [_]f32{0} ** 2;

    matrixMul(&out2, .{ .f32 = &a }, .{ .f32 = &b }, 2, 3);
    try std.testing.expectEqualDeep([_]f32{ 20, 44 }, out2);

    var d = [_]f32{ 4, -1 };
    var out3 = [_]f32{0} ** 3;
    matrixMul(&out3, .{ .f32 = &a }, .{ .f32 = &d }, 3, 2);
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
    matrixMul(&out5, .{ .f32 = &m }, .{ .f32 = &x }, 5, 12);
    try std.testing.expectEqualDeep(expect5, out5);

    var out12 = [_]f32{0} ** 12;
    const expect12 = [_]f32{
        2953, 888, 2203, 3501, 4717, 2083, 3491, 3781, 2697, 2964, 2714, 2259,
    };
    matrixMul(&out12, .{ .f32 = &m }, .{ .f32 = &z }, 12, 5);
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
    allocator: std.mem.Allocator,
) !struct { []const Block(format), f32 } {
    if (ws.len == 0) {
        return error.Empty;
    }

    // No conversion necessary
    if (format == .f32) {
        return .{ ws, 0 };
    }

    const BlockType = Block(format);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .weights).type).array;
    const block_size = array_info.len;
    std.debug.assert(ws.len % block_size == 0);

    const n_blocks = ws.len / block_size;
    const blocks = try allocator.alloc(BlockType, n_blocks);
    errdefer allocator.free(blocks);

    const err = switch (format) {
        .q8_0 => quantize_q8_0(ws, blocks),
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
fn quantize_q8_0(in: []const f32, blocks: []Block(.q8_0)) f32 {
    // Calculate max error
    var err: f32 = 0;
    const n_blocks = blocks.len;
    const BlockType = Block(.q8_0);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .weights).type).array;
    const ArrayElem = array_info.child;
    const block_size = array_info.len;

    for (0..n_blocks) |i| {
        var blk: BlockType = undefined;
        var items = blk.weights;
        const idx = i * block_size;

        const elems: []const f32 = in[idx .. idx + block_size];

        const max: f32 = std.sort.max(f32, elems, {}, max_magnitude.lessThan) orelse 0;
        const scale = max / 127.0;
        blk.scale = scale;
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
        blk.weights = items;
        blocks[i] = blk;
    }
    return err;
}

test "quantize weights" {
    const empty: [0]f32 = [0]f32{};
    try std.testing.expectError(error.Empty, quantize(.f32, &empty, std.testing.allocator));

    const f32s = [_]f32{ 1, 2, 3, 4, 5 };
    const f32_quant = try quantize(.f32, &f32s, std.testing.allocator);
    try std.testing.expectEqualSlices(f32, &f32s, f32_quant[0]);
    try std.testing.expectEqual(0, f32_quant[1]);

    const no_scale = [_]f32{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };

    const q80_no_scale = try quantize(.q8_0, &no_scale, std.testing.allocator);
    defer std.testing.allocator.free(q80_no_scale[0]);
    try std.testing.expectEqual(q80_no_scale[0].len, 1);

    const no_scale_exp = [32]i8{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,  16,
        17, 31, 32, 33, 60, 63, 64, 65, 69, 70, 71, 72, 124, 125, 126, 127,
    };
    try std.testing.expectEqualSlices(i8, &no_scale_exp, &q80_no_scale[0][0].weights);
    try std.testing.expectEqual(0, q80_no_scale[1]);
}

/// De-quantize weights
pub fn dequantize(
    comptime format: WeightFormat,
    weights: []const Block(format),
    out: []f32,
) !void {
    if (weights.len == 0) {
        return error.Empty;
    }

    if (format == .f32) {
        return;
    }

    switch (format) {
        .q8_0 => dequantize_q8_0(weights, out),
        else => @compileError("dequantize method is unimplemented for " ++ format),
    }
    return;
}

// TODO: Vectorize this
fn dequantize_q8_0(in: []const Block(.q8_0), out: []f32) void {
    const BlockType = Block(.q8_0);
    const block_size = blockUnitLen(BlockType);

    for (0..in.len) |i| {
        const block = in[i];

        const idx = block_size * i;

        const scale = block.scale;
        for (0..block_size) |j| {
            const elem = block.weights[j];
            const elem_f: f32 = @floatFromInt(elem);
            out[idx + j] = elem_f * scale;
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

    const q80_no_scale = try quantize(.q8_0, &no_scale, std.testing.allocator);
    defer std.testing.allocator.free(q80_no_scale[0]);
    try dequantize(.q8_0, q80_no_scale[0], &out);
    try std.testing.expectEqualSlices(f32, &no_scale, &out);
}
