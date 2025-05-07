// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! quant: Quantizing and de-quantizing utilities
//! This module contains code used for quantizing and de-quantizing tensors.

const std = @import("std");

const math = @import("../root.zig").math;

const Weights = math.Weights;

pub const q80 = @import("q80.zig");
pub const q6k = @import("q6k.zig");
pub const q4k = @import("q4k.zig");

/// Get the block format for a given `format` in memory on on disk.
/// For non-f32 type weight blocks, the returned struct must have a `scale` and `weights` field
/// for compatibility with compiled code.
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
pub const QK_K = 256;
/// How many scales are inside a K-Quantization superblock
pub const K_SCALE_SIZE = 12;

/// Structure of a `Q6_K` super-block in GGML. See [1] for definition.
/// A `Q6_K` super-block is 16 sub-blocks of 16 elements each.
/// Each weight is derived as `w = g * a * d` where `a` is the block scale.
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
/// Weights are calculated as `w = q * g + m` where `g` is the weight scale and `m` is the block
/// minimum.
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
            const rounded: i32 = nearest_int(iscale * scales[j]);
            blocks[i].scales[j] = @intCast(@min(127, rounded));
        }

        inner: for (0..n_subblocks) |j| {
            const d = @as(f32, @as(f16, @bitCast(blocks[i].scale))) * @as(f32, @floatFromInt(blocks[i].scales[j]));
            if (d == 0) {
                continue :inner;
            }

            for (0..16) |k| {
                const x = orig[16 * j + k];
                var l: i32 = nearest_int(x / d);
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
                const l: i32 = nearest_int(iscale * x);
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
            const av_x = @sqrt(sum_x2 / 32);
            for (0..32) |l| {
                weights[l] = av_x + @abs(block[j * 32 + l]);
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
            var ls: u8 = @intCast(nearest_int(inv_scale * scales[j]));
            var lm: u8 = @intCast(nearest_int(inv_min * mins[j]));
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

        inner: for (0..@divExact(QK_K, 32)) |j| {
            const sc, const m = get_scale_min_k4(j, &out_block.scale);
            const d = @as(f32, @floatCast(@as(f16, @bitCast(out_block.agg.d)))) * @as(f32, @floatFromInt(sc));

            if (d == 0) continue :inner;

            const dm = @as(f32, @floatCast(@as(f16, @bitCast(out_block.agg.dmin)))) * @as(f32, @floatFromInt(m));

            for (0..32) |k| {
                var l: i32 = nearest_int((block[32 * j + k] + dm) / d);
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
inline fn get_scale_min_k4(j: usize, q: []const u8) struct { u8, u8 } {
    if (j < 4) {
        return .{
            // zig fmt: off
            q[j]   & 0x3f,
            q[j+4] & 0x3f,
            // zig fmt: on
        };
    } else {
        return .{
            // zig fmt: off
            (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4),
            (q[j+4] >>   4) | ((q[j-0] >> 6) << 4),
            // zig fmt: on
        };
    }
}

/// Port of `nearest_int` from `ggml-quants.c`.
inline fn nearest_int(value: f32) i32 {
    return @intFromFloat(std.math.round(value));
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
        if (xx < min) min = xx;
        if (xx > max) max = xx;

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
        const l: i64 = nearest_int(iscale * (xx - min));
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
            var l: i64 = nearest_int(iscale * (xx - min));
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
                var diff = this_scale * @as(f32, @floatFromInt(limit_aux[i])) + this_min - xx;
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
        .q8_0 => dequantize_q8_0(T, weights, out),
        .q6_k => dequantize_q6_k(T, weights, out),
        .q4_k => dequantize_q4_k(T, weights, out),
        else => @compileError("dequantize method is unimplemented for " ++ @tagName(format)),
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
    const block_size = blockUnitLen(Q6KBlock);

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
    const block_size = blockUnitLen(Q4KBlock);

    for (0.., in) |i, *block| {
        var out_block = out[i * block_size ..][0..block_size];

        const weights = block.weights;

        const d: f32 = @as(f16, @bitCast(block.agg.d));
        const min: f32 = @as(f16, @bitCast(block.agg.dmin));

        var is: usize = 0;
        for (0..@divExact(QK_K, 64)) |jj| {
            const j = jj * 64;

            const q = weights[jj * 32 .. (jj + 1) * 32][0..32];

            // zig fmt: off
            const sc1, const m1 = get_scale_min_k4(is + 0, &block.scale);
            const scale1 = d   * @as(f32, @floatFromInt(sc1));
            const min1   = min * @as(f32, @floatFromInt(m1));
            const sc2, const m2 = get_scale_min_k4(is + 1, &block.scale);
            const scale2 = d   * @as(f32, @floatFromInt(sc2));
            const min2   = min * @as(f32, @floatFromInt(m2));
            // zig fmt: on

            for (0..32) |l| {
                const result = scale1 * @as(f32, @floatFromInt(q[l] & 0x0f)) - min1;
                out_block[j + l] = @floatCast(result);
            }
            for (0..32) |l| {
                const result = scale2 * @as(f32, @floatFromInt(q[l] >> 4)) - min2;
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
