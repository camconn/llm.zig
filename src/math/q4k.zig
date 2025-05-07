// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! q4k: `Q4_K` kernels

const std = @import("std");

const quant = @import("quant.zig");
const Block = quant.Block;
const blockUnitLen = quant.blockUnitLen;

const QK_K = quant.QK_K;
const K_SCALE_SIZE = quant.K_SCALE_SIZE;

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

// TODO: vectorize this
/// Quantize `f32` values into `blocks` with the `Q4_K` quantization format.
/// Refer to `quantize_row_q4_K_ref` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1208
pub fn quantize_q4_k(in: []const f32, blocks: []Q4KBlock) f32 {
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
            var ls: u8 = @intCast(quant.nearest_int(inv_scale * scales[j]));
            var lm: u8 = @intCast(quant.nearest_int(inv_min * mins[j]));
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
                var l: i32 = quant.nearest_int((block[32 * j + k] + dm) / d);
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
        const l: i64 = quant.nearest_int(iscale * (xx - min));
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
            var l: i64 = quant.nearest_int(iscale * (xx - min));
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

// TODO: Vectorize this
/// Dequantize `in` from the `Q4_K` quantization format into `T` float values.
/// Refer to `dequantize_row_q4_K` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1280
pub fn dequantize_q4_k(T: type, in: []const Q4KBlock, out: []T) void {
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
