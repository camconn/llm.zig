// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! q80: `Q8_0` kernels

const std = @import("std");

const quant = @import("quant.zig");
const Block = quant.Block;
const blockUnitLen = quant.blockUnitLen;

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
pub fn quantize_q8_0(in: []const f32, blocks: []quant.Q80Block) f32 {
    const n_blocks = blocks.len;
    const BlockType = Block(.q8_0);
    const array_info = @typeInfo(std.meta.fieldInfo(BlockType, .weights).type).array;
    const Elem = array_info.child;
    const block_size = blockUnitLen(BlockType);

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

// TODO: Vectorize this
/// Dequantize `in` from the `Q8_0` quantization format into `T` float values.
/// Refer to `dequantize_row_q8_0` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/489716ba99ecd51164f79e8c6fec0b5bf634eac9/src/ggml-quants.c#L349
pub fn dequantize_q8_0(T: type, in: []const Q80Block, out: []T) void {
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

// ========================================
// Kernels
// ========================================

/// Matrix multiply for Q8_0 quantized weights.
pub fn matrixMulVec_q8_0(T: type, out: []T, m: []const Q80Block, x: []const Q80Block, rows: usize, cols: usize) void {
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
