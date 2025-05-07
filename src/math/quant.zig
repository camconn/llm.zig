// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! quant: Quantizing and de-quantizing utilities
//! This module contains code used for quantizing and de-quantizing tensors.
//! It includes the definitions for available quantization formats.

const std = @import("std");

const math = @import("../root.zig").math;

const Weights = math.Weights;

pub const q80 = @import("q80.zig");
pub const q6k = @import("q6k.zig");
pub const q4k = @import("q4k.zig");

// ========================================
// Type definitions and helper functions
// ========================================

/// Get the block format for a given `format` in memory on on disk.
pub fn Block(format: math.WeightFormat) type {
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

        const tag_int = comptime for (@typeInfo(math.WeightFormat).@"enum".fields) |f| {
            if (std.mem.eql(u8, f.name, exp_name)) {
                break f.value;
            }
        };
        const tag: math.WeightFormat = @enumFromInt(tag_int);

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
    try std.testing.expectEqual(QK_K, blockUnitLen(Block(.q6_k)));
    try std.testing.expectEqual(QK_K, blockUnitLen(Block(.q4_k)));
}

// refer to ggml-common.h for these values
/// Super block size for QK quantization schemes.
pub const QK_K = 256;
/// How many scales are inside a K-Quantization superblock
pub const K_SCALE_SIZE = 12;

pub const Q80Block = q80.Q80Block;
pub const Q6KBlock = q6k.Q6KBlock;
pub const Q4KBlock = q4k.Q4KBlock;

// ========================================
// quantization & de-quantization
// ========================================

/// A potential error which can occur during quantization or dequantization.
pub const QuantError = error{
    /// The input or output slice for the operation is empty (length 0).
    Empty,
    /// There is not enough space in the output slice OR the output slice's
    /// length does not exactly match the length of quantized or de-quantized
    /// output for the specified `WeightFormat`.
    NotEnoughSpace,
};

/// Quantize weights `ws` from full-size `f32` to their quantized forms.
/// Caller is responsible for freeing the returned Quantized block array.
pub fn quantize(
    comptime format: math.WeightFormat,
    ws: []const f32,
    out: []Block(format),
) !f32 {
    if (ws.len == 0) {
        return QuantError.Empty;
    }

    // No conversion necessary
    if (format == .f32 or format == .f16) {
        @memcpy(out, ws);
        return 0;
    }

    const BlockType = Block(format);
    const block_size = blockUnitLen(BlockType);
    std.debug.assert(@mod(ws.len, block_size) == 0);
    std.debug.assert(@mod(ws.len, block_size) == 0);
    std.debug.assert(@divExact(ws.len, block_size) == out.len);

    const err = switch (format) {
        .q8_0 => q80.quantize_q8_0(ws, out),
        .q6_k => q6k.quantize_q6_k(ws, out),
        .q4_k => q4k.quantize_q4_k(ws, out),
        else => @compileError("quantize method is unimplemented for " ++ @tagName(format)),
    };

    // TODO: Do something with the quantization error.
    return err;
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
    comptime format: math.WeightFormat,
    weights: []const Block(format),
    out: []f32,
) !void {
    try dequantizeT(f32, format, weights, out);
}

/// De-quantize weights to an array of type `T`.
/// Requires that `out` exactly matches the size of the de-quantized output of `weights` for `format`.
pub fn dequantizeT(
    T: type,
    comptime format: math.WeightFormat,
    weights: []const Block(format),
    out: []T,
) !void {
    math.floatOnly(T);
    if (weights.len == 0) {
        return QuantError.Empty;
    }

    const BlockType = @typeInfo(@TypeOf(weights)).pointer.child;
    const block_size = blockUnitLen(BlockType);
    const weights_out_len = block_size * weights.len;
    if (out.len != weights_out_len) {
        return QuantError.NotEnoughSpace;
    }

    // Superfluous with above, but included just in case.
    std.debug.assert(@mod(out.len, block_size) == 0);
    std.debug.assert(@divExact(out.len, block_size) == weights.len);

    if (format == .f32 or format == .f16) {
        // Check that the pointers are different so we don't violate `noalias`.
        if (@intFromPtr(out.ptr) != @intFromPtr(weights.ptr)) {
            std.mem.doNotOptimizeAway(@memcpy(out, weights));
        }
        return;
    }

    switch (format) {
        .q8_0 => q80.dequantize_q8_0(T, weights, out),
        .q6_k => q6k.dequantize_q6_k(T, weights, out),
        .q4_k => q4k.dequantize_q4_k(T, weights, out),
        else => @compileError("dequantize method is unimplemented for " ++ @tagName(format)),
    }
}

/// Rounds `value` to the nearest 32-bit integer.
/// Port of `nearest_int` from `ggml-quants.c`.
pub inline fn nearest_int(value: f32) i32 {
    return @intFromFloat(std.math.round(value));
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

test "dequantize(quantize(input)) q8_0, q6_k, and q4_k" {
    const file = @embedFile("../assets/q6_k.json");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const case = try std.json.parseFromSliceLeaky(@This().QuantizationTestCase, alloc, file, .{});
    const input = case.input;
    const input_len: f32 = @floatFromInt(input.len);
    const out_len = 512;

    // Check that implementations are correct by quantizing and then dequantizing, then check
    // the root mean square error.
    // check q6_k
    var q8_out = [_]Q80Block{std.mem.zeroes(Q80Block)} ** (512 / blockUnitLen(Q80Block));
    _ = try quantize(.q8_0, input, q8_out[0..]);
    var dequantized_q8 = [_]f32{0} ** out_len;
    _ = try dequantizeT(f32, .q8_0, &q8_out, dequantized_q8[0..]);
    var sum_err2_q8: f32 = 0;
    for (0.., input) |i, expected| {
        const diff = expected - dequantized_q8[i];
        try std.testing.expectApproxEqAbs(expected, dequantized_q8[i], 0.1);
        sum_err2_q8 += diff * diff;
    }
    const rmse_q8 = @sqrt(sum_err2_q8 / input_len);
    try std.testing.expect(rmse_q8 < 0.01);

    // check q6_k
    var q6_out = [_]Q6KBlock{std.mem.zeroes(Q6KBlock)} ** 2;
    _ = try quantize(.q6_k, input, q6_out[0..]);
    var dequantized_q6 = [_]f32{0} ** out_len;
    _ = try dequantizeT(f32, .q6_k, &q6_out, dequantized_q6[0..]);
    var sum_err2_q6: f32 = 0;
    for (0.., input) |i, expected| {
        const diff = expected - dequantized_q6[i];
        try std.testing.expectApproxEqAbs(expected, dequantized_q6[i], 0.1);
        sum_err2_q6 += diff * diff;
    }
    const rmse_q6 = @sqrt(sum_err2_q6 / input_len);
    try std.testing.expect(rmse_q6 < 0.01);

    // check q4_k
    var out_q4 = [_]Q4KBlock{std.mem.zeroes(Q4KBlock)} ** 2;
    _ = try quantize(.q4_k, input, &out_q4);
    var dequantized_q4 = [_]f32{0} ** out_len;
    try dequantize(.q4_k, &out_q4, &dequantized_q4);

    var sum_err2_q4: f32 = 0;
    for (0.., input) |i, value| {
        const diff = @abs(value - dequantized_q4[i]);
        try std.testing.expect(diff < 0.15);
        sum_err2_q4 += diff * diff;
    }
    const rmse_q4 = @sqrt(sum_err2_q4 / input_len);
    try std.testing.expect(rmse_q4 < 0.02);
}

// LCOV_EXCL_START
pub usingnamespace if (@import("builtin").is_test) struct {
    /// Helper for reading quantization test case data.
    const QuantizationTestCase = struct {
        /// Input original floating point tensor.
        input: []f32,
        /// Expected result of quantization.
        quant_ref: []u8,
        /// Expected result after de-quantizing the quantization.
        dequant_ref: []f32,
    };
} else struct {};
// LCOV_EXCL_STOP

test "Q4_K quantization reference JSON" {
    const file = @embedFile("../assets/q4_k.json");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const case = try std.json.parseFromSliceLeaky(@This().QuantizationTestCase, alloc, file, .{});

    const input = case.input;
    const quantization_exp_bs: []const Q4KBlock = @alignCast(std.mem.bytesAsSlice(Q4KBlock, case.quant_ref));
    var quantized_actual = [_]Q4KBlock{std.mem.zeroes(Q4KBlock)} ** 2;
    _ = try quantize(.q4_k, input, &quantized_actual);
    try std.testing.expectEqualSlices(Q4KBlock, quantization_exp_bs, &quantized_actual);

    var dequantized_actual = [_]f32{0} ** (QK_K * 2);
    try dequantize(.q4_k, &quantized_actual, &dequantized_actual);
    try std.testing.expectEqualSlices(f32, case.dequant_ref, &dequantized_actual);
}

test "Q6_K quantization reference JSON" {
    const file = @embedFile("../assets/q6_k.json");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const case = try std.json.parseFromSliceLeaky(@This().QuantizationTestCase, alloc, file, .{});

    const input = case.input;
    const quantization_exp_bs: []const Q6KBlock = @alignCast(std.mem.bytesAsSlice(Q6KBlock, case.quant_ref));
    var quantized_actual = [_]Q6KBlock{std.mem.zeroes(Q6KBlock)} ** 2;
    _ = try quantize(.q6_k, input, &quantized_actual);
    try std.testing.expectEqualSlices(Q6KBlock, quantization_exp_bs, &quantized_actual);

    var dequantized_actual = [_]f32{0} ** (QK_K * 2);
    try dequantize(.q6_k, &quantized_actual, &dequantized_actual);
    try std.testing.expectEqualSlices(f32, case.dequant_ref, &dequantized_actual);
}

test "Q8_0 quantization reference JSON" {
    const file = @embedFile("../assets/q8_0.json");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const case = try std.json.parseFromSliceLeaky(@This().QuantizationTestCase, alloc, file, .{});

    const input = case.input;
    const quantization_exp_bs: []const Q80Block = @alignCast(std.mem.bytesAsSlice(Q80Block, case.quant_ref));
    var quantized_actual = [_]Q80Block{std.mem.zeroes(Q80Block)} ** (QK_K * 2 / blockUnitLen(Q80Block));
    _ = try quantize(.q8_0, input, &quantized_actual);
    try std.testing.expectEqualSlices(Q80Block, quantization_exp_bs, &quantized_actual);

    var dequantized_actual = [_]f32{0} ** (QK_K * 2);
    try dequantize(.q8_0, &quantized_actual, &dequantized_actual);
    try std.testing.expectEqualSlices(f32, case.dequant_ref, &dequantized_actual);
}
