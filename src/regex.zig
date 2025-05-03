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

const llm = @import("root.zig");
const unicode = llm.unicode;
const isLetter = unicode.isLetter;
const isNumber = unicode.isNumber;
const isWhitespace = unicode.isWhitespace;

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

        var iters: usize = 0;
        // Cap the iteration limit at n*4 to account for backtracking.
        const iter_limit = (self.buf.len - self.cursor) * 4;
        while (true) {
            iters += 1;
            if (iters >= iter_limit) {
                @panic("Iteration limit exceeded. A hanging match may be occurring, this is probably a library bug.");
            }

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
                        //
                        // Backtrack and emit the trailing character after the apostrophe as
                        // its own match.
                        //
                        // We can assume the last character was 1 byte long because we are
                        // currently matching on the partial string 'r, 'v, or 'l.
                        self.backtrack(start, 1);
                        self.*.state = .end;
                        continue;
                    },
                }
            }

            const b = self.buf[self.cursor..];
            const chr_len = unicode.calcFirstCodepointUnicodeLen(b) catch 0;
            if (chr_len == 0) {
                @panic("next codepoint failed to calculate length");
            }
            const chr = unicode.decodeUtf8First(b) catch 0;
            if (chr == 0) {
                std.debug.print("buf: {d}, <<{s}>>\n", .{ b, b });
                @panic("Invalid decode");
            }

            //std.debug.print("got character {d} at {d}\n", .{ chr, self.cursor });

            // After consuming character, move forward cursor
            self.*.cursor += chr_len;

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
                        // Lookahead to see if next char is whitespace and if the following token
                        // is not whitespace. If so, then emit the current match.
                        // Otherwise, keep going
                        if (self.peekOffset(self.cursor)) |result1| {
                            const peek1 = result1[0];
                            const index = result1[1];

                            if (!isWhitespace(peek1)) {
                                // This is only possible whenever we start matching on a buffer
                                // like "SWN" where "S" is SPACE, W is any whitespace character,
                                // and "N" is a non-whitespace character.
                                // To reach this point we are currently on `chr` == `W`.
                                //
                                // Backtrack 1 code point and emit the group "S".
                                self.backtrack(start, chr_len);
                                self.*.state = .end;
                                continue;
                            }

                            if (self.peekOffset(index)) |result2| {
                                const peek2 = result2[0];

                                if (isWhitespace(peek1) and !isWhitespace(peek2)) {
                                    self.*.state = .end;
                                }
                            }
                        }
                    } else {
                        // backtrack by one to exclude non-whitespace char and then emit match
                        // NB: This is only possible whenever a whitespace char != 0x20 precedes
                        //     a non-whitespace char.
                        self.backtrack(start, chr_len);
                        self.*.state = .end;
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
                        self.backtrack(start, chr_len);
                    }
                },
                .apostrophe_rv => {
                    if (chr == 'e') {
                        self.*.state = .end;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                        // backtrack current non-matching char + r/v
                        self.backtrack(start, chr_len + 1);
                    }
                },
                .apostrophe_l => {
                    if (chr == 'l') {
                        self.*.state = .end;
                    } else {
                        self.*.state = .not_space_letter_numeric;
                        // backtrack current non-matching char + l
                        self.backtrack(start, chr_len + 1);
                    }
                },
                .letters => {
                    if (isLetter(chr)) {
                        // keep going;
                    } else {
                        self.*.state = .end;
                        self.backtrack(start, chr_len);
                    }
                },
                .numbers => {
                    if (isNumber(chr)) {
                        // keep going
                    } else {
                        self.*.state = .end;
                        self.backtrack(start, chr_len);
                    }
                },
                .not_space_letter_numeric => {
                    if (isLetter(chr) or isNumber(chr) or isWhitespace(chr)) {
                        self.*.state = .end;
                        self.backtrack(start, chr_len);
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

    /// Peek at the next Unicode character in the buffer starting at `offset`.
    /// Returns the codepoint and ending offset in the buffer string otherwise.
    /// Returns `null` if a read would extend past the end of the buffer, and
    /// Assumes the buffer string is valid UTF-8.
    fn peekOffset(self: Self, offset: usize) ?struct { u21, usize } {
        if (offset >= self.buf.len) {
            return null;
        }

        // Check length of next codepoint
        const len = unicode.calcFirstCodepointUnicodeLen(self.buf[offset..]) catch 0;
        if (len == 0) {
            // TODO(error): We know we have invalid UTF-8 at this point.
            return null;
        }

        // Check that we have room to check the next codepoint
        const ending = offset + len;
        if (ending > self.buf.len) {
            // TODO(error): We know we have invalid UTF-8 at this point.
            return null;
        }

        // Check the codepoint and ensure we don't have a decode error.
        const char = unicode.decodeUtf8First(self.buf[offset..ending]) catch 0;
        if (char == 0) {
            // TODO(error): We know we have invalid UTF-8 at this point.
            return null;
        }

        return .{ char, ending };
    }
};

test "match GPT2 tokenization groups" {
    const verify = struct {
        fn check(src: []const u8, exp: []const []const u8) !void {
            //std.debug.print("Verifying <<{s}>>\n", .{src});
            var iter = Gpt2Pattern.init(src);
            for (exp) |group| {
                const next = iter.next();
                try std.testing.expect(next != null);
                //std.debug.print("got match <<{s}>> (len {d})\n", .{ next.?, next.?.len });
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

    const multi_char = "The great wall 🧱 is in 🇨🇳";
    const multi_char_exp = [_][]const u8{ "The", " great", " wall", " 🧱", " is", " in", " 🇨🇳" };
    try verify(multi_char, &multi_char_exp);

    const varied_ws = "This string has \u{2000}\u{2008}multichar whitespaces";
    const varied_ws_exp = [_][]const u8{ "This", " string", " has", " \u{2000}", "\u{2008}", "multichar", " whitespaces" };
    try verify(varied_ws, &varied_ws_exp);

    const varied_ws2 = "This also has \u{2000} multichar whitespaces";
    const varied_ws2_exp = [_][]const u8{ "This", " also", " has", " \u{2000}", " multichar", " whitespaces" };
    try verify(varied_ws2, &varied_ws2_exp);

    const varied_ws3 = "This also has\u{2000}\u{2008} multichar.";
    const varied_ws3_exp = [_][]const u8{ "This", " also", " has", "\u{2000}\u{2008}", " multichar", "." };
    try verify(varied_ws3, &varied_ws3_exp);

    const varied_ws4 = "This also has \u{2000}\u{2008} multichar.";
    const varied_ws4_exp = [_][]const u8{ "This", " also", " has", " \u{2000}\u{2008}", " multichar", "." };
    try verify(varied_ws4, &varied_ws4_exp);

    const v_weird = "V՟";
    const v_weird_exp = [_][]const u8{ "V", "՟" };
    try verify(v_weird, &v_weird_exp);
}

test "Gpt2Pattern fuzz" {
    const fuzzer = struct {
        fn fuzz(_: void, buf: []const u8) !void {
            // Only accept valid UTF-8
            const cleaned = if (std.mem.indexOf(u8, buf, "\u{0000}")) |i| buf[0..i] else buf;
            if (!std.unicode.utf8ValidateSlice(cleaned)) return;

            var matcher = Gpt2Pattern.init(cleaned);
            var sum: usize = 0;

            var matches = std.ArrayList([]const u8).init(std.testing.allocator);
            defer matches.deinit();

            while (matcher.next()) |group| {
                sum += group.len;
                try matches.append(group);
            }

            if (matcher.next() != null) {
                std.debug.print("Changed ending: {s}\n", .{cleaned});
                return error.ChangedEnding;
            }

            if (sum != cleaned.len) {
                std.debug.print("Missing groups:\n", .{});
                std.debug.print("Cleaned: <<{s}>>\n", .{cleaned});
                std.debug.print("Cleaned: <<{d}>>\n", .{cleaned});
                for (matches.items) |group| {
                    std.debug.print("group: <<{s}>>\n", .{group});
                }
                return error.Missing;
            }
        }
    }.fuzz;

    // Currently broken in zig 0.14.0
    //try std.testing.fuzz({}, fuzzer, .{ .corpus = &.{
    //    "",
    //    "Hello, world!",
    //    "This is a test",
    //    "This also has\u{2000}\u{2008} multichars.",
    //    "The great wall 🧱 is in 🇨🇳",
    //} });
    try std.testing.fuzz({}, fuzzer, .{});
}
