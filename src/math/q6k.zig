// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! q6k: `Q6_K` kernels

const std = @import("std");

const quant = @import("quant.zig");
const Block = quant.Block;
const blockUnitLen = quant.blockUnitLen;

const QK_K = quant.QK_K;

/// Structure of a `Q6_K` super-block in GGML. See [1] for definition.
/// A `Q6_K` super-block is 16 sub-blocks of 16 elements each.
/// Each weight is derived as `w = g * a * d` where `a` is the block scale.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-common.h#L320
pub const Q6KBlock = extern struct {
    /// High 4 bits for quantized weights.
    weights_lo: [@divExact(QK_K, 2)]u8,
    /// Upper 2 bits for quantized weights.
    weights_hi: [@divExact(QK_K, 4)]u8,
    /// Quantized 8-bit scale.
    scales: [@divExact(QK_K, 16)]i8,
    /// Super-block scale.
    scale: u16,
};

/// Dequantize `in` from the `Q6_K` quantization format into `T` float values.
/// This is a vectorized form of the straightforward linear implementation in
/// `dequantize_q6_ks`. If you are trying to diagnose or improve this version, study that method.
///
/// Refer to `dequantize_row_q6_K` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1690
pub fn dequantize_q6_k(T: type, in: []const Q6KBlock, out: []T) void {
    const block_size = blockUnitLen(Q6KBlock);
    // blocks are processed in groups of 16 elements
    const group_size = 16;

    const SVec = @Vector(group_size, u3);
    const BVec = @Vector(group_size, u8);
    const IVec = @Vector(group_size, i8);
    const FVec = @Vector(group_size, f32);

    for (0.., in) |i, block| {
        const offset = i * block_size;

        const superblock_scale: f32 = @as(f16, @bitCast(block.scale));

        const w_lo = block.weights_lo;
        const w_hi = block.weights_hi;
        const scales = block.scales;

        const groups = block_size / group_size;

        // In-lining this group-wise loop empirically cuts execution time so that this
        // implementation is competitive with the reference `dequantize_q6_k_ref` one from GGML.
        // This was measured to cut execution time on a matrix of (4096x4096) ona Zen 3 5900X
        // with DDR4 ECC memory at 3600 MHz from 3548 μs/iter to 2996 μs/iter, scaling better
        // as the matrix sizes grows.
        // This method under-performs the `dequantize_q6_k_ref` implementation at all sizes below
        // 4096x2048. At that size and above, this method has an advantage in execution time.
        inline for (0..groups) |g| {
            const j = g * group_size;
            const half: usize = @intFromBool(g >= 8);
            const lo_idx = @mod(j, 64) + half * 64;
            const hi_idx = @mod(j, 32) + half * 32;

            const lo_shift: u3 = @intCast(@mod(j / 64, 2) * 4);
            const hi_shift: u3 = @intCast(@mod((j / 32) * 2, 8));

            var lo: BVec = w_lo[lo_idx..][0..group_size].*;
            var hi: BVec = w_hi[hi_idx..][0..group_size].*;

            const lo_shr: SVec = @splat(lo_shift);
            lo >>= lo_shr;
            lo &= @as(BVec, @splat(0x0f));

            const hi_shr: SVec = @splat(hi_shift);
            hi = hi >> hi_shr;
            hi &= @as(BVec, @splat(0x03));
            hi <<= @as(BVec, @splat(4));

            const wt_u = lo | hi;
            var wt: IVec = @intCast(wt_u);
            wt -%= @splat(32);
            const wt_f: FVec = @floatFromInt(wt);

            // Cast scales from i8 to f32
            const sc = @as(f32, @floatFromInt(scales[g]));
            const scale: FVec = @splat(superblock_scale * sc);

            out[offset + g * group_size ..][0..group_size].* = scale * wt_f;
        }
    }
}

/// **DO NOT USE** this method. It is slower than the equivalent `dequantize_q6_k` and
/// `dequantize_q6_k_ref` implementations. It is only included for test purposes.
///
/// Dequantize `in` from the `Q6_K` quantization format into `T` float values.
/// This is the straight-forward implementation with no swizzling or vectorization.
/// Refer to `dequantize_row_q6_K` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1690
pub fn dequantize_q6_ks(T: type, in: []const Q6KBlock, out: []T) void {
    const block_size = blockUnitLen(Q6KBlock);

    for (0.., in) |i, block| {
        const offset = i * block_size;

        const superblock_scale: f32 = @as(f16, @bitCast(block.scale));

        const w_lo = block.weights_lo;
        const w_hi = block.weights_hi;
        const scales = block.scales;

        for (0..QK_K) |j| {
            const half = j / 128;

            const lo_idx = @mod(j, 64) + half * 64;
            const hi_idx = @mod(j, 32) + half * 32;
            const scale_idx = j / 16;

            const lo_shift: u3 = @truncate(@mod(j / 64, 2) * 4);
            const hi_shift: u3 = @truncate(@mod((j / 32) * 2, 8));

            const lo = (w_lo[lo_idx] >> lo_shift) & 0x0f;
            const hi = (w_hi[hi_idx] >> hi_shift) & 0x03;

            const sc: f32 = @floatFromInt(scales[scale_idx]);

            const w: i8 = @bitCast((hi << 4) | lo); //always safe, top 2 bits always `0`
            const w_f: f32 = @floatFromInt(w -% 32);

            const weight: f32 = @floatCast(superblock_scale * sc * w_f);
            out[offset + j] = weight;
        }
    }
}

/// Dequantize `in` from the `Q6_K` quantization format into `T` float values.
/// Refer to `dequantize_row_q6_K` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1690
pub fn dequantize_q6_k_ref(T: type, in: []const Q6KBlock, out: []T) void {
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

                const l0 = l + 0;
                const l1 = l + 32;

                // zig fmt: off
                const q1: i8 = @intCast((lo[l0] & 0x0f) | (((hi[l] >> 0) & 3) << 4));
                const q2: i8 = @intCast((lo[l1] & 0x0f) | (((hi[l] >> 2) & 3) << 4));
                const q3: i8 = @intCast((lo[l0]   >> 4) | (((hi[l] >> 4) & 3) << 4));
                const q4: i8 = @intCast((lo[l1]   >> 4) | (((hi[l] >> 6) & 3) << 4));
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

test "equivalence for dequantize_q6_k*" {
    // ensure that the dequantize kernels for q6_k are correct by running them on the same input
    // and checking the resulting output is the same

    var rng = std.Random.Xoroshiro128.init(std.testing.random_seed);
    const random = rng.random();

    var orig = [_]f32{0} ** (QK_K * 3);
    for (0..orig.len) |i| {
        orig[i] = random.float(f32) * 1000;
    }
    var input = [_]Q6KBlock{std.mem.zeroes(Q6KBlock)} ** 3;
    _ = quantize_q6_k(&orig, &input);

    var out_vec = [_]f32{0} ** (QK_K * 3);
    var out_straight = [_]f32{0} ** (QK_K * 3);
    var out_ref = [_]f32{0} ** (QK_K * 3);

    dequantize_q6_k(f32, &input, out_vec[0..]);
    dequantize_q6_ks(f32, &input, out_straight[0..]);
    dequantize_q6_k_ref(f32, &input, out_ref[0..]);

    try std.testing.expectEqualSlices(f32, &out_ref, &out_vec);
    try std.testing.expectEqualSlices(f32, &out_ref, &out_straight);
}

const group_max_eps: comptime_float = 1e-15;

// TODO: vectorize this
/// Quantize `f32` values into `blocks` with the `Q6_K` quantization format.
/// Refer to `quantize_row_q6_K_ref` in GGML [1] for the reference implementation.
/// [1]: https://github.com/ggml-org/ggml/blob/17733de6a7854b9696be7a563711c9aa4a34b2d3/src/ggml-quants.c#L1620
pub fn quantize_q6_k(in: []const f32, blocks: []Q6KBlock) f32 {
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
            const rounded: i32 = quant.nearest_int(iscale * scales[j]);
            blocks[i].scales[j] = @intCast(@min(127, rounded));
        }

        inner: for (0..n_subblocks) |j| {
            const d = @as(f32, @as(f16, @bitCast(blocks[i].scale))) * @as(f32, @floatFromInt(blocks[i].scales[j]));
            if (d == 0) {
                continue :inner;
            }

            for (0..16) |k| {
                const x = orig[16 * j + k];
                var l: i32 = quant.nearest_int(x / d);
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
                const l: i32 = quant.nearest_int(iscale * x);
                const l_clamp = std.math.clamp(l, -n_max, n_max - 1);
                out[i] = @intCast(n_max + l_clamp);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}
