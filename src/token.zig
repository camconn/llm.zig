// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! token: Tokenization algorithms and utilities
//! This module contains tools for converting text to and from tokens that models understand.

const std = @import("std");

const ggml = @import("root.zig").ggml;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

/// A single Token's ID. Represents one of the possible vocabulary tokens for a number of models.
/// Note that invalid/padding tokens are < 0. Additionally, Tokens are not necessarily comparable
/// between models.
pub const Token = i32;

/// Implementation of the SentencePiece tokenizer.
pub const SPTokenizer = struct {
    const Self = @This();

    /// A single Token's ID. Represents one of 32000 vocab value or a sentinel token.
    /// Note that a padding token == -1, hence the signed-ness.
    pub const TokenEntry = struct {
        score: f32,
        id: Token,
        chars: []u8,
    };
    const Storage = std.MultiArrayList(TokenEntry);

    tokens: Storage,
    max_len: usize,
    arena: ArenaAllocator,
    token_to_idx: []usize,

    // Surrogate whitespace character used by Sentencepiece for spaces.
    const whitespace_string = "▁"; // "\xe2\x96\x81";

    pub fn initV1(file_path: []const u8, vocab_size: usize, alloc: Allocator) !SPTokenizer {
        // First try to load the model with relative and then fallback to absolute path if that fails.
        const options: std.fs.File.OpenFlags = .{ .mode = .read_only };
        const cwd = std.fs.cwd();

        // zig fmt: off
        const file: ?std.fs.File = cwd.openFile(file_path, options)
            // Fall back to absolute path
            catch (std.fs.openFileAbsolute(file_path, options) catch null);
        // zig fmt: on

        if (file == null) {
            std.debug.print("Could not open tokenizer file: {s}\nIs the path correct?\n", .{file_path});
            return std.fs.File.OpenError.FileNotFound;
        }
        defer file.?.close();

        var buffer = std.io.bufferedReader(file.?.reader());
        return try SPTokenizer.read(buffer.reader(), vocab_size, alloc);
    }

    const Sorter = struct {
        const S = @This();
        toks: *Storage,
        pub fn lessThan(self: S, left_index: usize, right_index: usize) bool {
            const lhs = self.toks.items(.chars)[left_index];
            const rhs = self.toks.items(.chars)[right_index];

            // TODO: Use a real Unicode collator and ensure source tokens are normalized.
            return std.mem.order(u8, lhs, rhs) == .lt;
        }
    };

    /// Read an exported Tokenizer model as exported by `llama2.c/tokenizer.py`
    /// Any scores and bytes read will be allocated with the supplied `Allocator` and only
    /// be freed after calling `deinit()`.
    fn read(reader: anytype, vocab_size: usize, allocator: Allocator) !SPTokenizer {
        // Read a tokenizer file as written from Karpathy's `llama2.c`
        // According to `tokenizer.py`, the format is:
        //
        // A. 32 bits for the unsigned `max_token_length`
        // B. 32 bits for a float `score`
        // C. 32 bits for the unsigned length of bytes
        // D. The bytes for the token.
        // E. EOF or go to B again.
        //
        // The tokenizer file was likely created on a Little-Endian machine, so we
        // will assume these values are Little-Endian

        const max_len: usize = @intCast(try reader.readInt(u32, .little));

        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        var alloc = arena.allocator();

        var tokens = Storage{};
        try tokens.ensureTotalCapacity(alloc, vocab_size + 10); // add 10 size for safety.

        // Nothing we can really do here if there's an error
        for (0..vocab_size) |i| {
            const score_bits = try reader.readInt(u32, .little);
            const score: f32 = @bitCast(score_bits);
            const token_len = try reader.readInt(u32, .little);
            var token = try alloc.alloc(u8, token_len);

            try reader.readNoEof(token[0..]);

            // NB: Entries of index [3,256+3) are encoded in some special fashion for their raw
            // bytes. These values are encoded like `<0xZZ>` where `ZZ` is the token's character
            // value in hexadecimal.
            const entry = TokenEntry{
                .score = score,
                .id = @intCast(i),
                .chars = token,
            };

            try tokens.append(alloc, entry);
        }

        // Sort the token so that the `chars` elements are in order
        //dumpTokens(tokens);
        tokens.sortUnstable(Sorter{ .toks = &tokens });

        // `tokens` is now sorted by the source bytes/characters of each token entry in
        // semi-alphabetical order.

        // Build mapping of Token ID to index in `tokens`
        var token_to_idx = try alloc.alloc(usize, vocab_size);
        const ids = tokens.items(.id);
        for (0..vocab_size) |i| {
            const token = ids[i];
            token_to_idx[@intCast(token)] = i;
        }

        // Copy the Area after all of the copies have been done, otherwise there will be
        // a leak from the Arena's allocations.
        // Same with Token Storage
        return .{
            .max_len = max_len,
            .tokens = tokens,
            .arena = arena,
            .token_to_idx = token_to_idx,
        };
    }

    pub fn initGGUF(file: ggml.GGUFFile, allocator: Allocator) !SPTokenizer {
        const context_len = file.getValue("llama.context_length").?.uint32;

        const token_chars = file.getValue("tokenizer.ggml.tokens").?.array;
        const token_scores = file.getValue("tokenizer.ggml.scores").?.array;
        // The vocabulary size is equivalent to the length of the tokens array.
        const vocab_size = token_chars.len;

        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        var alloc = arena.allocator();

        var tokens = Storage{};
        try tokens.ensureTotalCapacity(alloc, vocab_size + 10); // add 10 size for safety.

        // Nothing we can really do here if there's an error
        for (0..vocab_size) |i| {
            const chars = token_chars.array[i].string.str;
            const score = token_scores.array[i].float32;
            // Don't care about the type of token
            //const token_type = token_ids.array[i].int32;

            // Replace the sentinel whitespace character with an actual space character because
            // sentencepiece uses the bold underline like lunatics for whitespace.
            var to_use: []u8 = chars;
            if (std.mem.containsAtLeast(u8, to_use, 1, whitespace_string)) {
                const cs = try alloc.alloc(u8, to_use.len);
                const count = std.mem.replace(u8, chars, whitespace_string, " ", cs);
                const new_len = cs.len - (count * 2);
                to_use = cs[0..new_len];
                //std.debug.print("Replaced underscore from <<{s}>> to <<{s}>>\n", .{ chars, to_use });
            }

            // NB: Entries of index [3,256+3) are encoded in some special fashion for their raw
            // bytes. These values are encoded like `<0xZZ>` where `ZZ` is the token's character
            // value in hexadecimal.
            const token = TokenEntry{
                .score = score,
                .chars = to_use,
                .id = @intCast(i),
            };
            try tokens.append(alloc, token);
        }

        // Sort the token so that the `chars` elements are in order
        //dumpTokens(tokens);
        tokens.sortUnstable(Sorter{ .toks = &tokens });

        // `tokens` is now sorted by the source bytes/characters of each token entry in
        // semi-alphabetical order.

        // Build mapping of Token ID to index in `tokens`
        var token_to_idx = try alloc.alloc(usize, vocab_size);
        const ids = tokens.items(.id);
        for (0..vocab_size) |i| {
            const token = ids[i];
            token_to_idx[@intCast(token)] = i;
        }

        // Copy the Area after all of the copies have been done, otherwise there will be
        // a leak from the Arena's allocations.
        // Same with Token Storage
        return .{
            .max_len = context_len,
            .tokens = tokens,
            .arena = arena,
            .token_to_idx = token_to_idx,
        };
    }

    fn dumpTokens(tokens: Storage) void {
        const slice = tokens.slice();

        const id = slice.items(.id);
        const score = slice.items(.score);
        const chars = slice.items(.chars);

        for (0..32000) |i| {
            const fmt = std.fmt.fmtSliceHexUpper(chars[i]);

            std.debug.print("id: {d}; score={d:.3} chars={X}\n", .{ id[i], score[i], fmt });
        }
    }

    /// Free up any resources associated with this `Tokenizer` and invalidate any pointers to
    /// token strings or token entries.
    pub fn deinit(self: *Self) void {
        self.tokens.deinit(self.arena.allocator());
        self.arena.deinit();
    }

    // TODO: Derive these from the read tokenizer file instead of statically defining them.
    const UNK = 0;
    pub const BOS = 1;
    pub const EOS = 2;
    const PAD = -1;

    const UNK_PIECE = "<unk>";
    const BOS_PIECE = "<s>";
    const EOS_PIECE = "</s>";
    const PAD_PIECE = "<pad>";
    const SPACE_PIECE = " ";

    /// Encode text into a series of tokens.
    /// Any slice returned will be allocated with `token_alloc`. The allocator used for
    /// calling `init()` will not be used.
    /// The caller is responsible for freeing the returned tokens from `token_alloc`.
    pub fn encode(self: Self, text: []const u8, alloc: Allocator) ![]Token {
        // TODO: Handle un-encodable symbols with `<UNK>` or `UNK` token.

        // Optimistically assume we will output the entire text character
        // as a token.

        // Make a heuristic guess about how long our tokenized sequence will be
        var output_final = std.ArrayList(Token).init(alloc);
        try output_final.ensureTotalCapacity(text.len >> 2);
        try output_final.append(BOS);

        const slice = self.tokens.slice();
        const chars = slice.items(.chars);
        const ids = slice.items(.id);
        const scores = slice.items(.score);

        // SentencePiece automatically appends whitespace whenever the `add_dummy_prefix=True`
        // in the Normalizer.
        // Llama 2's tokenizer happens to use this setting, so go ahead and add a single space
        // in front of all tokens we search for.
        // This will be merged in the merge step so that [" ", "hello"] becomes [" hello"]
        const space = lookupIndex(chars, SPACE_PIECE).?; // TODO: This fails when loading GGUF
        // space's token ID is 35, but don't hard-code that for now
        const space_id = ids[space];

        // Use a SinglyLinkedList because we are going to be repeatedly removing nodes later
        // and linked lists have the best behavior for that.
        const Out = std.SinglyLinkedList(Token);
        const Node = Out.Node;
        var space_node = Node{ .data = space_id };
        const tokens = Out{ .first = &space_node };
        var last: *Node = &space_node;

        // Force using an arena allocator because the zig compiler freaks out whenever
        // calling `alloc.free(*Node)`
        var arena = ArenaAllocator.init(alloc);
        defer arena.deinit();
        var token_alloc = arena.allocator();

        var idx: usize = 0;
        // BPE-like Search algorithm
        // Start with the longest possible string to match then slowly trim
        // trailing characters until we get a matching character.
        search: while (true) {
            const end = @min(idx + 1, text.len);

            // TODO: Support UTF-8 strings.
            const candidate = text[idx..end];

            var match: ?TokenEntry = null;
            if (lookupIndex(chars, candidate)) |found| {
                match = self.tokens.get(found);
            }

            if (match) |found| {
                var next = try token_alloc.create(Node);
                next.data = found.id;
                last.insertAfter(next);
                last = next;

                //std.debug.print("old idx={d}, out_i={d}\n", .{ idx, output.items.len });
                idx += found.chars.len;
                //std.debug.print("new idx: {d}\n", .{idx});
                continue :search;
            } else if (idx == text.len) {
                //std.debug.print("Done searching\n", .{});
                break :search;
            } else {
                // Fall back and just use the single char as the token value.
                // TODO: Figure out how to handle Unicode here.
                const token_id: Token = @intCast(text[idx] + 3);

                var next = try token_alloc.create(Node);
                next.data = token_id;
                last.insertAfter(next);
                last = next;

                idx += 1;
                continue :search;
            }

            // Done searching
            break;
        }

        // Do merge algorithm
        //std.debug.print("Performing merge starting with {d}\n", .{output.items.len});
        var buf: [64]u8 = undefined;
        while (true) {
            var best: f32 = -1e10;
            var best_id: ?Token = null;
            var best_idx: ?*Node = null;

            var node = tokens.first;

            while (node.?.next) |next| {
                const left_idx = self.findIndexByTokenId(node.?.data).?;
                const right_idx = self.findIndexByTokenId(next.data).?;

                const left_chars = chars[left_idx];
                const right_chars = chars[right_idx];

                const combined = try std.fmt.bufPrint(&buf, "{s}{s}", .{ left_chars, right_chars });
                if (lookupIndex(chars, combined)) |index| {
                    if (scores[index] > best) {
                        best = scores[index];
                        best_idx = node;
                        best_id = ids[index];
                    }
                }

                node = next;
            }

            if (best_idx) |best_node| {
                //const left_id = best_node.data;
                //const right_id = best_node.next.?.data;
                //std.debug.print("Merging token {d} with {d}\n", .{ left_id, right_id });

                // Replace value of the lefthand side with the merged value.
                // Delete the next node.
                best_node.data = best_id.?;
                _ = best_node.removeNext(); // don't free, used arena
            } else {
                // found nothing to merge, that means we're done
                break;
            }
        }

        // Iterate built list and add it to arraylist for final export
        var node = tokens.first;
        while (node) |inner| {
            try output_final.append(inner.data);
            node = inner.next;
            // don't free, used arena
        }

        // Shrink output to final size
        return output_final.toOwnedSlice();
    }

    const Searcher = struct {
        fn compare(ctx: []const u8, item: []const u8) std.math.Order {
            return std.mem.order(u8, ctx, item);
        }
    };

    /// Find the index for a needle in the haystack.
    /// Returns an index if the exact match is found and `null` otherwise.
    fn lookupIndex(haystack: [][]u8, needle: []const u8) ?usize {
        if (std.sort.binarySearch([]u8, haystack, needle, Searcher.compare)) |index| {
            return index;
        } else {
            return null;
        }
    }

    /// Find a token by its index in `tokens`.
    fn findIndexByTokenId(self: Self, token: Token) ?usize {
        // TODO: find a better path to make this private
        return self.token_to_idx[@intCast(token)];
    }

    /// Get the token entry for the given token.
    pub fn getTokenChars(self: Self, tok: Token) ?[]const u8 {
        if (self.findIndexByTokenId(tok)) |idx| {
            return self.tokens.items(.chars)[idx];
        }
        return null;
    }
};

test "SPTokenizer.encode" {
    const tok_path = try std.testing.allocator.dupe(u8, "tokenizer.bin");
    defer std.testing.allocator.free(tok_path);
    var tokenizer = try SPTokenizer.initV1(tok_path, 32000, std.testing.allocator);
    defer tokenizer.deinit();

    {
        const sample = "Hello, world! How are you today?";
        const ids = [_]Token{ 1, 15043, 29892, 3186, 29991, 1128, 526, 366, 9826, 29973 };

        const tokens = try tokenizer.encode(sample, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Token, &ids, tokens);
    }

    {
        const sample = "Hello\nworld";
        const ids = [_]Token{ 1, 15043, 13, 11526 };

        const tokens = try tokenizer.encode(sample, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Token, &ids, tokens);
    }

    {
        const sample = "Byte pair encoding[1][2] (also known as BPE, or digram";
        // zig fmt: off
        const ids = [_]Token{
            1,
            19831, 5101, 8025, 29961, 29896, 3816, 29906, 29962,
            313, 15189, 2998, 408, 350, 4162, 29892, 470,
            4697, 2572
        };
        // zig fmt: on

        const tokens = try tokenizer.encode(sample[0..], std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Token, &ids, tokens);
    }

    {
        const sample = @embedFile("assets/bpe_sample.txt");
        const out_ids = @embedFile("assets/bpe_sample_expected.json");

        var ids_json = try std.json.parseFromSlice([]Token, std.testing.allocator, out_ids, .{});
        defer ids_json.deinit();
        const ids = ids_json.value;

        const tokens = try tokenizer.encode(sample[0..], std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Token, ids, tokens);
    }
}
