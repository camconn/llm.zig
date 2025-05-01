// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! regex: Regular expressions for tokenization
//! This module contains implementations for running regular expressions on input data, in
//! particular during tokenization. Some tokenizers such as `tiktoken` use regular expressions
//! as part of tokenization, so this module aims to implement that.

const std = @import("std");

// Matches the literal strings
// 's
// 't
// 're
// 've
// 'm
// 'll
// 'd
// optional SPACE followed by
//   [\p{L}]+          - 1 or more unicode class L: Matches any letter from any language
//   [\p{N}]+          - 1 or more unicode class N: Matches any number in any script
//   [^\s\p{L}\p{N}]+  - 1 or more of (NOT whitespace and NOT unicode class L and NOT unicode class N)
// \s+(?!\S)           - 1 or more whitespace followed by look-ahead NOT whitespace
// \s+                 - 1 or more whitespace
//
// Note \s is equivalent to \p{White_Space}

/// GPT-2 TikToken regex for GPT-2
const tiktoken_gpt =
    \\'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
;

const space = ' ';
const apostrophe = '\'';

/// Helper struct to tokenize strings into groups that can be correctly tokenized by the BPE
/// `sample.TikTokenizer` implementation.
///
/// This is a implemented as a hand-rolled FSM implementing the GPT-2 tokenization
/// regular expression.
pub const Gpt2Pattern = struct {
    const Self = @This();

    const State = enum {
        /// Starting state
        start,
        /// Literal space character for first position
        space,
        /// Whitespace character
        whitespace,
        /// Literal apostrophe character
        apostrophe,
        /// Apostrophe followed by `r` or `v`.
        apostrophe_rv,
        /// Apostrophe followed by `l`.
        apostrophe_l,
        /// Last input character was a letter
        letters,
        /// Last input character was a number
        numbers,
        /// Last input character was not a number, letter, or whitespace
        not_space_letter_numeric,
        /// Finished consuming, ready for next match
        end,
    };

    cursor: usize,
    buf: []const u8,
    state: State,

    /// Initialize a new `Gpt2Pattern` matching items from the `src` string.
    /// Assumes that `src` is a valid UTF-8 string.
    pub fn init(src: []const u8) Gpt2Pattern {
        return .{
            .cursor = 0,
            .buf = src,
            .state = .start,
        };
    }

    /// Find and return the next matching group or return if there are no more.
    /// This is intended to be used in a `while` loop.
    pub fn next(self: *Self) ?[]const u8 {
        const start = self.*.cursor;
        var ret: ?[]const u8 = null;

        if (self.*.cursor >= self.*.buf.len and self.*.state == .start) {
            // Early return. We're done here.
            return null;
        }

        while (true) {
            //std.debug.print("state: {any}; cursor: {d}\n", .{ self.*.state, self.*.cursor });
            // Check if we're at the end... If we are, go ahead and emit the match
            if (self.*.state == .end) {
                ret = self.*.buf[start..self.*.cursor];
                self.*.state = .start;
                break;
            }

            // Handle end of buffer
            if (self.cursor >= self.buf.len) {
                switch (self.state) {
                    .start => unreachable,
                    // Go back around to start and emit match
                    .end => continue,
                    // Terminal group. Emit final match then quit
                    .whitespace, .space, .letters, .numbers, .not_space_letter_numeric => {
                        self.*.state = .end;
                        continue;
                    },
                    // We received an apostrophe and are waiting for the next character, but
                    // we've already reached the end of the buffer, so go ahead and just emit
                    // the single apostrophe as if we
                    .apostrophe => {
                        self.*.state = .end;
                        continue;
                    },
                    // Backtrack required
                    .apostrophe_rv, .apostrophe_l => {
                        // Emit single apostrophe
                        // Backtrack and emit the trailing character after the apostrophe as
                        // its own match.
                        self.backtrack(start, 1);
                        self.*.state = .end;
                        continue;
                    },
                }
            }

            const b = self.buf[self.cursor..];
            const chr_len = calcFirstCodepointUnicodeLen(b) catch 0;
            if (chr_len == 0) {
                @panic("next codepoint failed to calculate length");
            }
            const chr = decodeUtf8First(self.buf[self.cursor..]) catch 0;
            if (chr == 0) {
                @panic("Invalid decode");
            }

            //std.debug.print("got character {d} at {d}\n", .{ chr, self.cursor });

            // After consuming character, move forward cursor
            self.*.cursor += 1;

            // Heart of the regular expression matcher. This switches on the current state and
            // operates on the last byte consumed.
            switch (self.*.state) {
                .start => {
                    if (chr == space) {
                        self.*.state = .space;
                    } else if (isWhitespace(chr)) {
                        self.*.state = .whitespace;
                    } else if (chr == apostrophe) {
                        self.*.state = .apostrophe;
                    } else if (isNumber(chr)) {
                        self.*.state = .numbers;
                    } else if (isLetter(chr)) {
                        self.*.state = .letters;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                    }
                },
                .space => {
                    if (isWhitespace(chr)) {
                        self.*.state = .whitespace;
                    } else if (isLetter(chr)) {
                        self.*.state = .letters;
                    } else if (isNumber(chr)) {
                        self.*.state = .numbers;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                    }
                },
                .whitespace => {
                    if (isWhitespace(chr)) {
                        // keep going

                        // Lookahead past next token and see if following token is whitespace
                        // or not
                        if (self.cursor + 2 < self.buf.len) {
                            const peek1 = self.buf[self.cursor];
                            const peek2 = self.buf[self.cursor + 1];

                            if (isWhitespace(peek1) and !isWhitespace(peek2)) {
                                self.*.state = .end;
                            }
                        }
                    } else {
                        // backtrack by one and then emit match
                        self.backtrack(start, 1);
                    }
                },
                .apostrophe => {
                    if (chr == 's' or chr == 't' or chr == 'm' or chr == 'd') {
                        self.*.state = .end;
                    } else if (chr == 'r' or chr == 'v') {
                        self.*.state = .apostrophe_rv;
                    } else if (chr == 'l') {
                        self.*.state = .apostrophe_l;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                        self.backtrack(start, 1);
                    }
                },
                .apostrophe_rv => {
                    if (chr == 'e') {
                        self.*.state = .end;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                        self.backtrack(start, 2);
                    }
                },
                .apostrophe_l => {
                    if (chr == 'l') {
                        self.*.state = .end;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                        self.backtrack(start, 2);
                    }
                },
                .letters => {
                    if (isLetter(chr)) {
                        // keep going;
                    } else {
                        self.*.state = .end;
                        self.backtrack(start, 1);
                    }
                },
                .numbers => {
                    if (isNumber(chr)) {
                        // keep going
                    } else {
                        self.*.state = .end;
                        self.backtrack(start, 1);
                    }
                },
                .not_space_letter_numeric => {
                    if (isLetter(chr) or isNumber(chr) or isWhitespace(chr)) {
                        self.*.state = .end;
                        self.backtrack(start, 1);
                    } else {
                        // Keep going
                    }
                },
                .end => @panic("reached .end, but should have been caught earlier"),
            }
        }

        return ret;
    }

    /// Backtrack the current cursor by `amount`.
    /// Requires `start + amount ≤ cursor`
    fn backtrack(self: *Self, start: usize, amount: usize) void {
        // Can't backtrack cursor whenever
        std.debug.assert(start + amount <= self.*.cursor);
        self.*.cursor -= amount;
    }
};

test "match GPT2 tokenization groups" {
    const verify = struct {
        fn check(src: []const u8, exp: []const []const u8) !void {
            var iter = Gpt2Pattern.init(src);
            for (exp) |group| {
                const next = iter.next();
                try std.testing.expect(next != null);
                try std.testing.expectEqualSlices(u8, group, next.?);
            }
            try std.testing.expectEqual(null, iter.next());
        }
    }.check;

    const sample = "a bird in the hand is worth 2 in the bush.";
    const sample_exp = [_][]const u8{
        "a",   " bird", " in",   " the", " hand", " is", " worth", " 2",
        " in", " the",  " bush", ".",
    };
    try verify(sample, &sample_exp);

    const back_exp = [_][]const u8{ "back", "2", "back", "." };
    try verify("back2back.", &back_exp);

    const end_apos1 = "everybody knows it's";
    const end_apos1_exp = [_][]const u8{ "everybody", " knows", " it", "'s" };
    try verify(end_apos1, &end_apos1_exp);

    const end_apos2 = "you'v";
    const end_apos2_exp = [_][]const u8{
        "you",
        "'",
        "v",
    };
    try verify(end_apos2, &end_apos2_exp);

    const end_apos3 = "she'll be coming around the mtn when she comes. She'l";
    const end_apos3_exp = [_][]const u8{
        "she",  "'ll",  " be",   " coming", " around",
        " the", " mtn", " when", " she",    " comes",
        ".",    " She", "'",     "l",
    };
    try verify(end_apos3, &end_apos3_exp);

    const partial_shel = "she'l be coming";
    const partial_shel_exp = [_][]const u8{
        "she", "'", "l", " be", " coming",
    };
    try verify(partial_shel, &partial_shel_exp);

    const no_contraction = "break' my code";
    const no_contraction_exp = [_][]const u8{
        "break", "'", " my", " code",
    };
    try verify(no_contraction, &no_contraction_exp);

    const func_call = "func(1, 2a, foo_bar, 'baz')";
    const func_call_exp = [_][]const u8{
        "func", "(", "1", ",", " 2", "a", ",", " foo", "_", "bar", ",", " '", "baz", "')",
    };
    try verify(func_call, &func_call_exp);

    const surround_ws = "    a    ";
    const surround_ws_exp = [_][]const u8{ "   ", " a", "    " };
    try verify(surround_ws, &surround_ws_exp);

    const ws_only = "    \r\n\t ";
    const ws_only_exp = [_][]const u8{"    \r\n\t "};
    try verify(ws_only, &ws_only_exp);

    const hello_ex = "Hello, friend";
    const hello_exp = [_][]const u8{ "Hello", ",", " friend" };
    try verify(hello_ex, &hello_exp);

    const mixed = "I got your number it's 867-5309!";
    const mixed_exp = [_][]const u8{ "I", " got", " your", " number", " it", "'s", " 867", "-", "5309", "!" };
    try verify(mixed, &mixed_exp);
}

// implementations of various helper functions for matching against unicode groups

/// List of Unicode characters obtained from Wikipedia [1].
/// [1]: https://en.wikipedia.org/wiki/Whitespace_character#Unicode
const ws_chars = [_]u21{
    0x0009,
    0x000A,
    0x000B,
    0x000C,
    0x000D,
    0x0020,
    0x0085,
    0x00A0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
};

// TODO: Benchmark this, it may be too slow.
/// Determine if `src` is a whitespace Unicode codepoint.
fn isWhitespace(src: u21) bool {
    // Don't binary search, just use linear search because we expect to see the lower value
    // chars (space, newline, carriage return, tab) most of the time.
    if (std.mem.indexOfScalar(u21, &ws_chars, src)) |_| {
        return true;
    }
    return false;
}

test "isWhitespace" {
    try std.testing.expectEqual(true, isWhitespace(' '));
    try std.testing.expectEqual(true, isWhitespace('\r'));
    try std.testing.expectEqual(true, isWhitespace('\n'));
    try std.testing.expectEqual(true, isWhitespace('\t'));
    try std.testing.expectEqual(false, isWhitespace('A'));
    try std.testing.expectEqual(false, isWhitespace(0));
}

/// Determine if `src` is a letter Unicode codepoint.
fn isLetter(src: u21) bool {
    // TODO: Find if in class `L` which is `Lu` | `Ll` | `Lt` | `Lm` | `Lo`
    // https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-4/#G134153

    const a: u21 = 'a';
    const z: u21 = 'z';
    const A: u21 = 'A';
    const Z: u21 = 'Z';

    const lower = a <= src and src <= z;
    const upper = A <= src and src <= Z;

    return lower or upper;
}

test "isLetter" {
    try std.testing.expectEqual(true, isLetter('a'));
    try std.testing.expectEqual(true, isLetter('z'));
    try std.testing.expectEqual(true, isLetter('A'));
    try std.testing.expectEqual(true, isLetter('Z'));
    try std.testing.expectEqual(true, isLetter('o'));
    try std.testing.expectEqual(true, isLetter('h'));
    try std.testing.expectEqual(true, isLetter('a'));
    try std.testing.expectEqual(true, isLetter('i'));

    try std.testing.expectEqual(false, isLetter('0'));
    try std.testing.expectEqual(false, isLetter('\n'));
    try std.testing.expectEqual(false, isLetter(0));
}

/// Determine if `src` is a numeric Unicode codepoint.
fn isNumber(src: u21) bool {
    // TODO: Find if in class `N` which is `Nd` | `Nl` | `No`
    // https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-4/#G134153
    const zero: u21 = '0';
    const nine: u21 = '9';
    const numeric = zero <= src and src <= nine;

    // TODO: Add other qualifiers

    return numeric;
}

/// Calculate the length of the first unicode codepoint in the provided string.
/// Assumes the string is a valid UTF-8 string and does not end prematurely.
fn calcFirstCodepointUnicodeLen(str: []const u8) !usize {
    const top_bit = 0x80;
    const leading_bits = 0xC0;

    // Trivial case first for single-byte
    if (str.len < 1) {
        return error.Empty;
    }
    if (str[0] & 0x80 == 0) {
        return 1;
    }

    // Decode length of codepoint from the bitmask for leading byte.
    var size: usize = 0;
    // bits to check
    const masks = [_]u8{
        0b1110_0000,
        0b1111_0000,
        0b1111_1000,
    };
    // what we expect when performing bitwise AND against leading byte
    const good_mask = [_]u8{
        0b1100_0000,
        0b1110_0000,
        0b1111_0000,
    };
    const leading = str[0];
    for (0.., masks) |i, mask| {
        if (leading & mask == good_mask[i]) {
            size = i + 2;
        }
    }

    // Check that we found a matching bitmask
    if (size == 0) {
        // Invalid unicode
        return error.Unicode;
    }

    for (1..size) |i| {
        if (str[i] & leading_bits != top_bit) {
            std.debug.print("Invalid byte at index {d} in {d}\n", .{ i, str });
            return error.Unicode;
        }
    }

    return size;
}

test "calcFirstCodepointUnicodeLen" {
    try std.testing.expectError(error.Empty, calcFirstCodepointUnicodeLen(""));
    try std.testing.expectEqual(1, try calcFirstCodepointUnicodeLen("asdf"));
    try std.testing.expectEqual(1, try calcFirstCodepointUnicodeLen("Hello, world"));
    try std.testing.expectEqual(2, try calcFirstCodepointUnicodeLen("Ġ"));
    try std.testing.expectEqual(2, try calcFirstCodepointUnicodeLen("Ġhello"));
    try std.testing.expectEqual(3, try calcFirstCodepointUnicodeLen("你好"));
    try std.testing.expectEqual(4, try calcFirstCodepointUnicodeLen("🇺"));
    try std.testing.expectEqual(4, try calcFirstCodepointUnicodeLen("🇸"));
    try std.testing.expectEqual(4, try calcFirstCodepointUnicodeLen("🇺🇸"));
}

fn decodeUtf8First(str: []const u8) !u21 {
    const len = try calcFirstCodepointUnicodeLen(str);

    return switch (len) {
        1 => str[0],
        2 => std.unicode.utf8Decode2(str[0..2].*),
        3 => std.unicode.utf8Decode3(str[0..3].*),
        4 => std.unicode.utf8Decode4(str[0..4].*),
        else => @panic("Invalid calculated length"),
    };
}
