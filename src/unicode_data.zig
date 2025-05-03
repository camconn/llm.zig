// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! unicode_data: Helper functions for handling unicode data.
//! This module contains methods and structs for reading from and handling the Unicode Character
//! Database (UCD).
//! This module is **only** intended for testing and generating code for character selection.
//! importing this module in the main library outside of a test scenario will cause the entire
//! UCD text file to be embedded in the compiled library/executable.
//!
//! This module also contains methods for generating `switch` arm ranges automatically for certain
//! character class selections. See `printRanges` for more info.

const std = @import("std");

// This entire file should not be checked for coverage since it's only used for testing.
// LCOV_EXCL_START

/// Unicode codepoint `General_Category` from the General_Category Value table listed at [1].
/// [1]: https://www.unicode.org/reports/tr44/#General_Category_Values
const UnicodeCategory = enum(u8) {
    // zig fmt: off
    // L = Lu | Ll | Lt | Lm | Lo
    // LC = Lu | Ll | Lt
    Lu, Ll, Lt,
    Lm, Lo,
    // M = Mn | Mc | Me
    Mn, Mc, Me,
    // N = Nd | Nl | No
    Nd, Nl, No,
    // P = Pc | Pd | Ps | Pe | Pi | Pf | Po
    Pc, Pd, Ps, Pe, Pi, Pf, Po,
    // S = Sm | Sc | Sk | So
    Sm, Sc, Sk, So,
    // Z = Zs | Zl | Zp
    Zs, Zl, Zp,
    // C = Cc | Cf | Cs | Co | Cn
    Cc, Cf, Cs, Co, Cn,
    // zig fmt: on
};

/// Category lookup table of string in UCD to `UnicodeCategory`
const category_table = cc: {
    const start = @intFromEnum(UnicodeCategory.Lu);
    const end = @intFromEnum(UnicodeCategory.Cn);
    const size = end - start + 1;
    var table: [size]struct { []const u8, UnicodeCategory } = undefined;
    for (start..end + 1) |i| {
        const index = i - start;
        const category: UnicodeCategory = @enumFromInt(i);
        //table[index][0] = std.fmt.comptimePrint("{any}", .{category});
        table[index][0] = @tagName(category);
        table[index][1] = category;
    }
    break :cc table;
};

/// Unicode codepoint entry within the Unicode Character Database (UCD).
const UnicodeEntry = packed struct {
    codepoint: u21,
    category: UnicodeCategory,
};

/// Read the UCD's `unicodedata.txt` file according to the format listed in its table [1] in the
/// Unicode Technical Report.
/// This is not used when actually running the program, just whenever testing or generating
/// arms `switch` for the character class methods.
/// [1]: https://www.unicode.org/reports/tr44/#UnicodeData.txt
pub fn readUnicodeEntries(src: []const u8, allocator: std.mem.Allocator) !std.ArrayList(UnicodeEntry) {
    var file_buffer = std.io.fixedBufferStream(src);
    var reader = file_buffer.reader();

    var ret = std.ArrayList(UnicodeEntry).init(allocator);
    errdefer ret.deinit();

    var buf = [_]u8{0} ** 256;

    while (try reader.readUntilDelimiterOrEof(buf[0..], '\n')) |line| {
        var iter = std.mem.splitSequence(u8, line, ";");
        const code = iter.next().?;
        // name
        _ = iter.next().?;
        // General category value
        const category_name = iter.next().?;
        std.debug.assert(category_name.len > 0);
        // canonical_comb_class
        _ = iter.next().?;
        // bidi
        _ = iter.next().?;
        // decomp
        _ = iter.next().?;
        // numeric_1
        _ = iter.next().?;
        // numeric_2
        _ = iter.next().?;
        // numeric_3
        _ = iter.next().?;
        // bidi_mirrored
        _ = iter.next().?;
        // unicode1_name
        _ = iter.next().?;
        // iso_comment
        _ = iter.next().?;
        // simple_uppercase
        _ = iter.next().?;
        // simple_lowercase
        _ = iter.next().?;
        // simple_titlecase
        _ = iter.next().?;
        std.debug.assert(iter.next() == null);

        const codepoint_int = try std.fmt.parseInt(u32, code, 16);
        const cat: UnicodeCategory = c: {
            for (0..30) |i| {
                const entry = category_table[i];
                if (std.mem.eql(u8, entry[0], category_name)) {
                    break :c entry[1];
                }
            }
            @panic("Uh oh");
        };

        try ret.append(UnicodeEntry{
            .codepoint = @intCast(codepoint_int),
            .category = cat,
        });
    }

    return ret;
}

/// Helper struct define checks for if a `UnicodeCategory` is in certain group.
pub const UnicodeCheck = struct {
    pub fn isL(c: UnicodeCategory) bool {
        return switch (c) {
            .Lu, .Ll, .Lt, .Lm, .Lo => true,
            else => false,
        };
    }
    pub fn isN(c: UnicodeCategory) bool {
        return switch (c) {
            .Nd, .Nl, .No => true,
            else => false,
        };
    }
    pub fn isCc(c: UnicodeCategory) bool {
        return c == .Cc;
    }
};

// Change this to `true` and do `zig run unicode.zig` to print out the range tables
const generate_ranges = false;

// `pub usingnamespace` here is to prevent compiling the code when doing a library.
// If for some reason you update the unicode data file, change this to true and then
// run to generate the ranges for `isLetter` and `isNumber`.
pub usingnamespace if (generate_ranges) struct {
    // Helper function to generate code for match statement ranges for various character classes.
    fn printRanges(
        codepoints: []UnicodeEntry,
        criteria: fn (UnicodeCategory) bool,
    ) void {
        var start: ?u21 = null;
        var last: ?u21 = null;

        for (codepoints) |cp| {
            if (!criteria(cp.category)) continue;

            const c = cp.codepoint;
            // Handle start of search
            if (start == null) {
                start = c;
            }
            if (last == null) {
                last = c;
            }

            const contiguous = (c - last.?) <= 1;

            if (!contiguous) {
                const length = last.? - start.?;
                if (length >= 1) {
                    // Print range
                    std.debug.print("0x{X:0>4}...0x{X:0>4} => true,\n", .{ start.?, last.? });
                } else {
                    // Print single codepoint
                    std.debug.print("0x{X:0>4} => true,\n", .{last.?});
                }

                // Non-contiguous, start a new span.
                start = c;
            }

            last = c;
        }

        // Handle last characters in set
        const single = last.? == start.?;
        if (single) {
            // Print single codepoint
            std.debug.print("0x{X:0>4} => true,\n", .{last.?});
        } else {
            // Print range
            std.debug.print("0x{X:0>4}...0x{X:0>4} => true,\n", .{ start.?, last.? });
        }
    }

    // Main method to print out `switch` matching blocks to generate implementations of `isNumber`
    // and `isLetter`.
    pub fn main() !void {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const alloc = gpa.allocator();

        const unicode_file = @embedFile("assets/unicode-16.0.0.txt");
        const entries = try readUnicodeEntries(unicode_file, alloc);
        defer entries.deinit();

        std.debug.print("========================================\n", .{});
        std.debug.print("Ranges for isNumber:\n", .{});
        printRanges(entries.items, UnicodeCheck.isN);
        std.debug.print("========================================\n", .{});

        std.debug.print("========================================\n", .{});
        std.debug.print("Ranges for isLetter:\n", .{});
        printRanges(entries.items, UnicodeCheck.isL);
        std.debug.print("========================================\n", .{});
    }
} else struct {};

// LCOV_EXCL_STOP
