// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! token: Tokenization algorithms and utilities
//! This module contains tools for converting text to and from tokens that models understand.

const std = @import("std");

const llm = @import("root.zig");
const ggml = llm.ggml;
const regex = llm.regex;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const tokenizer_key = "tokenizer.ggml.model";

pub const TokenizerError = error{
    /// Tried to load the wrong tokenizer type for a model.
    WrongTokenizer,
    /// Tried to load invalid tokenizer settings.
    Tokenizer,
    /// Tokenizer has invalid format or contents.
    BadFormat,
};

/// An arbitrary token for any model.
/// Intended to represent a token in any Language Model.
/// Not necessarily comparable between models.
pub const Token = i64;
/// Disambiguation alias to `Token`.
pub const TokenizerToken = Token;

/// Encoding options when using a tokenizer.
pub const EncodingOption = enum {
    /// Indicates this is the start of tokenization, so the model or tokenizer should add a
    /// token indicating the beginning of input.
    start,
    /// Don't use any special encoding options. Use this for when the start of text is before
    /// the start of the context window.
    none,
};

/// Implementation of the `sentencepiece` tokenizer.
pub const SPTokenizer = struct {
    const Self = @This();

    /// A single Token's ID. Represents one of the possible vocabulary tokens for a number of models.
    /// Note that invalid/padding tokens are < 0.
    /// Tokens are not necessarily compatible between models.
    pub const Token = i64;

    /// A SentencePiece Token Entry.
    pub const TokenEntry = struct {
        score: f32,
        id: Self.Token,
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

    /// Encode `text` into a series of tokens with option to add a start `BOS` to returned slice.
    /// Any slice returned will be allocated with `token_alloc`. The allocator used for
    /// calling `init()` will not be used.
    /// The caller is responsible for freeing the returned tokens from `token_alloc`.
    pub fn encode(self: Self, text: []const u8, option: EncodingOption, alloc: Allocator) ![]Self.Token {
        // TODO: Handle un-encodable symbols with `<UNK>` or `UNK` token.

        // Optimistically assume we will output the entire text character
        // as a token.

        // Make a heuristic guess about how long our tokenized sequence will be
        var output_final = std.ArrayList(Self.Token).init(alloc);
        try output_final.ensureTotalCapacity(text.len >> 2);
        if (option == .start) {
            try output_final.append(BOS);
        }

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
        const Out = std.SinglyLinkedList(Self.Token);
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
                const token_id: Self.Token = @intCast(text[idx] + 3);

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
            var best_id: ?Self.Token = null;
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
    fn findIndexByTokenId(self: Self, token: Self.Token) ?usize {
        // TODO: find a better path to make this private
        return self.token_to_idx[@intCast(token)];
    }

    /// Get the token entry for the given token.
    pub fn getTokenChars(self: Self, tok: Self.Token) ?[]const u8 {
        if (self.findIndexByTokenId(tok)) |idx| {
            return self.tokens.items(.chars)[idx];
        }
        return null;
    }

    /// Cast a slice of this vocabulary's `tokens` into a generic slice of `TokenizerToken`.
    pub fn toGenericTokens(tokens: []const Self.Token) []const TokenizerToken {
        const as_bytes = std.mem.sliceAsBytes(tokens);
        return std.mem.bytesAsSlice(TokenizerToken, as_bytes);
    }

    /// Cast a slice of generic `tokens` into a slice of this vocabulary's tokens.
    pub fn fromGenericTokens(tokens: []const TokenizerToken) []const Self.Token {
        const as_bytes = std.mem.sliceAsBytes(tokens);
        return std.mem.bytesAsSlice(Self.Token, as_bytes);
    }
};

test "SPTokenizer.encode" {
    const tok_path = try std.testing.allocator.dupe(u8, "tokenizer.bin");
    defer std.testing.allocator.free(tok_path);
    var tokenizer = try SPTokenizer.initV1(tok_path, 32000, std.testing.allocator);
    defer tokenizer.deinit();

    {
        const sample = "Hello, world! How are you today?";
        const ids = [_]SPTokenizer.Token{ 1, 15043, 29892, 3186, 29991, 1128, 526, 366, 9826, 29973 };

        const tokens = try tokenizer.encode(sample, .start, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(SPTokenizer.Token, &ids, tokens);
    }

    {
        const sample = "Hello\nworld";
        const ids = [_]SPTokenizer.Token{ 1, 15043, 13, 11526 };

        const tokens = try tokenizer.encode(sample, .start, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(SPTokenizer.Token, &ids, tokens);
    }

    {
        const sample = "Byte pair encoding[1][2] (also known as BPE, or digram";
        // zig fmt: off
        const ids = [_]SPTokenizer.Token{
            1,
            19831, 5101, 8025, 29961, 29896, 3816, 29906, 29962,
            313, 15189, 2998, 408, 350, 4162, 29892, 470,
            4697, 2572
        };
        // zig fmt: on

        const tokens = try tokenizer.encode(sample[0..], .start, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(SPTokenizer.Token, &ids, tokens);
    }

    {
        const sample = @embedFile("assets/bpe_sample.txt");
        const out_ids = @embedFile("assets/bpe_sample_expected.json");

        var ids_json = try std.json.parseFromSlice([]SPTokenizer.Token, std.testing.allocator, out_ids, .{});
        defer ids_json.deinit();
        const ids = ids_json.value;

        const tokens = try tokenizer.encode(sample[0..], .start, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(SPTokenizer.Token, ids, tokens);
    }
}

/// Helper reference for codepoints in the base BPE setup.
const codepoints: [1024]u8 = cc: {
    var buf: [1024]u8 = undefined;
    for (0..512) |i| {
        buf[2 * i] = i / 256;
        buf[2 * i + 1] = @truncate(i);
    }
    break :cc buf;
};

/// Map a u16 codepoint to string.
fn get_codepoint_str(value: u16) []const u8 {
    std.debug.assert(value < 512);
    if (value < 256) {
        const idx = 2 * value + 1;
        return codepoints[idx .. idx + 1];
    }

    const start = 2 * value;
    const stop = start + 2;
    return codepoints[start..stop];
}

test "codepoint helper" {
    for (0..512) |i| {
        const actual = get_codepoint_str(@intCast(i));
        var expanded: usize = 0;
        if (i < 256) {
            try std.testing.expectEqual(actual.len, 1);
            expanded = actual[0];
        } else {
            try std.testing.expectEqual(actual.len, 2);
            const fst: usize = actual[0];
            const snd: usize = actual[1];

            expanded = fst * 256 + snd;
        }

        try std.testing.expectEqual(i, expanded);
    }
}

// Compare to `data_gym_to_mergeable_bpe_ranks` in `load.py`
/// Temporary helper used for constructing base `tiktoken` mappings.
const rank_to_intbytes = [_]u8{
    33,  34,  35,  36,  37,  38,  39,  40,
    41,  42,  43,  44,  45,  46,  47,  48,
    49,  50,  51,  52,  53,  54,  55,  56,
    57,  58,  59,  60,  61,  62,  63,  64,
    65,  66,  67,  68,  69,  70,  71,  72,
    73,  74,  75,  76,  77,  78,  79,  80,
    81,  82,  83,  84,  85,  86,  87,  88,
    89,  90,  91,  92,  93,  94,  95,  96,
    97,  98,  99,  100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 161, 162,
    163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187,
    188, 189, 190, 191, 192, 193, 194, 195,
    196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 206, 207, 208, 209, 210, 211,
    212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 235,
    236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 255, 0,   1,   2,   3,
    4,   5,   6,   7,   8,   9,   10,  11,
    12,  13,  14,  15,  16,  17,  18,  19,
    20,  21,  22,  23,  24,  25,  26,  27,
    28,  29,  30,  31,  32,  127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137,
    138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 173,
};

/// Mapping of codepoint (index) to the actual rank for a byte.
/// Equivalent-ish version of lookup table for `data_gym_byte_to_byte` in tiktoken.
const byte_to_bytes = cc: {
    var buf = [_]u8{0} ** 324;
    // flattened: list(sorted([(ord(x), y) for (x, y) in data_gym_byte_to_byte.items()]))
    const pairs = [_]u16{
        33,  33,  34,  34,  35,  35,  36,  36,  37,  37,  38,  38,  39,  39,  40,  40,
        41,  41,  42,  42,  43,  43,  44,  44,  45,  45,  46,  46,  47,  47,  48,  48,
        49,  49,  50,  50,  51,  51,  52,  52,  53,  53,  54,  54,  55,  55,  56,  56,
        57,  57,  58,  58,  59,  59,  60,  60,  61,  61,  62,  62,  63,  63,  64,  64,
        65,  65,  66,  66,  67,  67,  68,  68,  69,  69,  70,  70,  71,  71,  72,  72,
        73,  73,  74,  74,  75,  75,  76,  76,  77,  77,  78,  78,  79,  79,  80,  80,
        81,  81,  82,  82,  83,  83,  84,  84,  85,  85,  86,  86,  87,  87,  88,  88,
        89,  89,  90,  90,  91,  91,  92,  92,  93,  93,  94,  94,  95,  95,  96,  96,
        97,  97,  98,  98,  99,  99,  100, 100, 101, 101, 102, 102, 103, 103, 104, 104,
        105, 105, 106, 106, 107, 107, 108, 108, 109, 109, 110, 110, 111, 111, 112, 112,
        113, 113, 114, 114, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119, 120, 120,
        121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 161, 161, 162, 162,
        163, 163, 164, 164, 165, 165, 166, 166, 167, 167, 168, 168, 169, 169, 170, 170,
        171, 171, 172, 172, 174, 174, 175, 175, 176, 176, 177, 177, 178, 178, 179, 179,
        180, 180, 181, 181, 182, 182, 183, 183, 184, 184, 185, 185, 186, 186, 187, 187,
        188, 188, 189, 189, 190, 190, 191, 191, 192, 192, 193, 193, 194, 194, 195, 195,
        196, 196, 197, 197, 198, 198, 199, 199, 200, 200, 201, 201, 202, 202, 203, 203,
        204, 204, 205, 205, 206, 206, 207, 207, 208, 208, 209, 209, 210, 210, 211, 211,
        212, 212, 213, 213, 214, 214, 215, 215, 216, 216, 217, 217, 218, 218, 219, 219,
        220, 220, 221, 221, 222, 222, 223, 223, 224, 224, 225, 225, 226, 226, 227, 227,
        228, 228, 229, 229, 230, 230, 231, 231, 232, 232, 233, 233, 234, 234, 235, 235,
        236, 236, 237, 237, 238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243,
        244, 244, 245, 245, 246, 246, 247, 247, 248, 248, 249, 249, 250, 250, 251, 251,
        252, 252, 253, 253, 254, 254, 255, 255, 256, 0,   257, 1,   258, 2,   259, 3,
        260, 4,   261, 5,   262, 6,   263, 7,   264, 8,   265, 9,   266, 10,  267, 11,
        268, 12,  269, 13,  270, 14,  271, 15,  272, 16,  273, 17,  274, 18,  275, 19,
        276, 20,  277, 21,  278, 22,  279, 23,  280, 24,  281, 25,  282, 26,  283, 27,
        284, 28,  285, 29,  286, 30,  287, 31,  288, 32,  289, 127, 290, 128, 291, 129,
        292, 130, 293, 131, 294, 132, 295, 133, 296, 134, 297, 135, 298, 136, 299, 137,
        300, 138, 301, 139, 302, 140, 303, 141, 304, 142, 305, 143, 306, 144, 307, 145,
        308, 146, 309, 147, 310, 148, 311, 149, 312, 150, 313, 151, 314, 152, 315, 153,
        316, 154, 317, 155, 318, 156, 319, 157, 320, 158, 321, 159, 322, 160, 323, 173,
    };
    var i: usize = 0;
    while (i < pairs.len) : (i += 2) {
        const key = pairs[i];
        const val = pairs[i + 1];

        buf[@intCast(key)] = @intCast(val);
    }
    break :cc buf;
};

/// Analogous to the inline function `decode_data_gym` in tiktoken's `load.py`
/// This takes the padded character values present within the BPE vocabulary files used for
/// `tiktoken` and translates them to their native representations.
fn decode_data_gym(str: []const u8, out: []u8) []u8 {
    var iter = std.unicode.Utf8Iterator{
        .bytes = str,
        .i = 0,
    };

    var i: usize = 0;
    while (iter.nextCodepoint()) |point| {
        const trunc: usize = @intCast(point);
        std.debug.assert(trunc < byte_to_bytes.len);

        const mapped = byte_to_bytes[trunc];

        out[i] = mapped;
        i += 1;
    }

    return out[0..i];
}

test "tiktoken decode_data_gym" {
    var buf = [_]u8{0} ** 20;
    const decoded_hello = decode_data_gym("hello", &buf);
    const expected_hello = "hello";
    try std.testing.expectEqualSlices(u8, expected_hello, decoded_hello);

    const decoded_world = decode_data_gym("Ġworld", &buf);
    const expected_world = " world";
    try std.testing.expectEqualSlices(u8, expected_world, decoded_world);

    const decoded_zig = decode_data_gym("lang.zig123456", &buf);
    const expected_zig = "lang.zig123456";
    try std.testing.expectEqualSlices(u8, expected_zig, decoded_zig);
}

/// Implementation of the `tiktoken` BPE Tokenizer.
pub const TikTokenizer = struct {
    const Self = @This();

    const context_len_keys: [2][]const u8 = .{
        "gpt2.context_length",
        "qwen3.context_length",
    };

    /// A single Token's ID. Represents one of the possible vocabulary tokens for a number of models.
    /// Tokens are not necessarily compatible between models.
    pub const Token = i64;
    //pub const Token = u64;

    const Rank = Self.Token;
    const Ranks = std.StringHashMap(Rank);
    const Specials = std.StringArrayHashMap(Self.Token);
    const TokenList = std.ArrayList(Rank);
    const Reverse = std.AutoHashMap(Rank, []const u8);

    tokens: Ranks,
    rank_to_token: Reverse,
    special_tokens: Specials,

    /// Optional BOS token setting
    bos: ?Self.Token,
    /// Optional EOS token setting
    eos: ?Self.Token,
    add_bos: bool = false,

    arena: ArenaAllocator,

    /// Read and initialize a `TikTokenizer` instance from the provided `file` using the given
    /// `allocator`.
    pub fn init(file: ggml.GGUFFile, allocator: Allocator) !TikTokenizer {
        const tokenizer_model = file.getValueT(tokenizer_key, .string).?.string;
        // TODO: Swap regex pattern based on support tokenizer_model string.
        if (!std.mem.eql(u8, tokenizer_model.str, "gpt2")) {
            std.debug.print("Found non-GPT tokenizer model: {s}\n", .{tokenizer_model.str});
            return TokenizerError.WrongTokenizer;
        }

        var found_context_len: bool = false;
        var context_len: u32 = undefined;
        for (context_len_keys) |key| {
            if (file.getValueT(key, .uint32)) |len| {
                found_context_len = true;
                context_len = len.uint32;
            }
        }
        if (!found_context_len) {
            std.debug.print("Could not find context length for Tokenizer\n", .{});
            return TokenizerError.Tokenizer;
        }

        const token_chars = file.getValue("tokenizer.ggml.tokens").?.array;
        //const token_types = file.getValue("tokenizer.ggml.token_type").?.array;
        const token_merges = file.getValue("tokenizer.ggml.merges").?.array;
        // The vocabulary size is equivalent to the length of the tokens array.
        const vocab_size: usize = @intCast(token_chars.len);
        std.debug.print("vocab size: {d}\n", .{vocab_size});

        //std.debug.print("types: {d}; merges {d}\n", .{ token_types.len, token_merges.len });

        const bos = file.getValueT("tokenizer.ggml.bos_token_id", .uint32).?.uint32;
        const eos = file.getValueT("tokenizer.ggml.eos_token_id", .uint32).?.uint32;
        //const padding = file.getValue("tokenizer.ggml.padding_token_id").?.uint32;
        var add_bos: bool = false;
        if (file.getValueT("tokenizer.ggml.add_bos_token", .boolean)) |val| {
            // TODO: Figure out where this goes.
            add_bos = val.boolean;
        }
        // TODO: We assume that EOS and BOS tokens are always at the end of the vocabulary.
        //       That may not always be true, so we should fix that.
        const n_specials: usize = if (bos == eos) 1 else 2;
        const normal_vocab: usize = vocab_size - n_specials;

        // Ensure vocabulary in file is consisten
        const non_merged_vocab = @as(usize, std.math.maxInt(u8)) + 1 + n_specials;
        const num_merges = vocab_size - non_merged_vocab;
        if (num_merges != token_merges.array.len) {
            std.debug.print("Mismatch in vocab; num_merges={d} but non-byte vocab is {d}\n", .{
                num_merges,
                token_merges.array.len,
            });
            return TokenizerError.BadFormat;
        }

        //std.debug.print("bos {d} eos {d}\n", .{ bos, eos });

        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        var tokens = Ranks.init(alloc);
        errdefer tokens.deinit();
        try tokens.ensureTotalCapacity(@intCast(normal_vocab));

        for (0..normal_vocab) |i| {
            const rank: Rank = @intCast(i);
            const tok = token_chars.array[i].string.str;

            // It's necessary to pre-process tokens because the may still have things like the
            // space placeholder "Ġ" present.
            var cleaned = try alloc.alloc(u8, tok.len);
            errdefer alloc.free(cleaned);
            cleaned = decode_data_gym(tok, cleaned);
            try tokens.put(cleaned, rank);
        }

        var special = Specials.init(alloc);
        errdefer special.deinit();
        try special.ensureTotalCapacity(@intCast(n_specials));

        if (n_specials >= 1) {
            for (0..n_specials) |i| {
                const rank: Rank = @as(Rank, @intCast(normal_vocab)) + @as(Rank, @intCast(i));
                const tok = token_chars.array[i].string.str;
                try special.put(tok, rank);
            }
        }

        // Setup reverse mappings
        var reverse = Reverse.init(alloc);
        errdefer reverse.deinit();
        try reverse.ensureTotalCapacity(@intCast(vocab_size));

        var norm_iter = tokens.iterator();
        while (norm_iter.next()) |next| {
            try reverse.put(next.value_ptr.*, next.key_ptr.*);
        }
        var special_iter = special.iterator();
        while (special_iter.next()) |next| {
            try reverse.put(next.value_ptr.*, next.key_ptr.*);
        }

        // Copy the Area after all of the copies have been done, otherwise there will be
        // a leak from the Arena's allocations.
        // Same with Token Storage
        return .{
            .add_bos = add_bos,
            .bos = bos,
            .eos = eos,
            .tokens = tokens,
            .special_tokens = special,
            .rank_to_token = reverse,
            .arena = arena,
        };
    }

    /// Load a TikToken BPE file and encoder JSON file into an actual set of ranks.
    fn loadDataToBPERanks(
        bpe_reader: anytype,
        encoder_reader: anytype,
        special_tokens: Ranks,
        allocator: std.mem.Allocator,
    ) !TikTokenizer {
        var arena = ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const alloc = arena.allocator();

        // Parse `vocab.bpe` or similar file.
        // 128 is too small apparently
        const line_buf = try alloc.alloc(u8, 512);
        defer alloc.free(line_buf);

        // skip version line
        // TODO: Handle version line differences
        _ = try bpe_reader.readUntilDelimiter(line_buf, '\n');

        var ranks = Ranks.init(alloc);
        errdefer ranks.deinit();

        // {bytes([b]): i for i, b in enumerate(rank_to_intbyte)}
        for (0.., rank_to_intbytes) |i, b| {
            const key = get_codepoint_str(b);
            try ranks.put(key, @intCast(i));
        }

        // Parse the space-separated entry on each line of the BPE vocabulary file, map them to
        // their native representations, and then store them in a ranks mapping.
        var n: usize = ranks.count();
        while (true) {
            const line = bpe_reader.readUntilDelimiterOrEof(line_buf, '\n') catch line_buf[0..0];
            if (line) |ln| {
                if (ln.len == 0) {
                    std.debug.print("Somehow got a zero-length line. That's an error\n", .{});
                    @panic("Read zero-length line");
                }

                var spliterator = std.mem.splitSequence(u8, ln, " ");
                const lhs = spliterator.next().?;

                const rhs = spliterator.next().?;
                // Ensure there are only two entries per line
                std.debug.assert(spliterator.next() == null);

                var combined = try alloc.alloc(u8, lhs.len + rhs.len);
                std.mem.copyForwards(u8, combined, lhs);
                std.mem.copyForwards(u8, combined[lhs.len..], rhs);

                const cleaned = decode_data_gym(combined, combined);
                // Shrink away unused bytes if possible
                if (cleaned.len < combined.len) {
                    _ = alloc.resize(combined, cleaned.len);
                }
                try ranks.put(cleaned, @intCast(n));

                n += 1;
            } else {
                //std.debug.print("think we reached the end of the file\n", .{});
                break;
            }
        }
        //std.debug.print("Read {d} entries from BPE file and set {d}\n", .{ lines, n });

        // Add special tokens
        var specials = Specials.init(alloc);
        errdefer specials.deinit();
        var special_iter = special_tokens.iterator();
        while (special_iter.next()) |next| {
            try specials.put(next.key_ptr.*, next.value_ptr.*);
        }

        // Create a reversed mapping of Token to String
        var reversed = std.AutoHashMap(Rank, []const u8).init(alloc);
        errdefer reversed.deinit();
        try reversed.ensureTotalCapacity(ranks.count());

        var token_iter = ranks.iterator();
        while (token_iter.next()) |next| {
            const token = next.key_ptr;
            const rank = next.value_ptr;
            try reversed.put(rank.*, token.*);
        }
        // Add special tokens to reverse id to string mapping
        special_iter.index = 0; // reset specials iterator
        while (special_iter.next()) |next| {
            try reversed.put(next.value_ptr.*, next.key_ptr.*);
        }

        var merge_arena = ArenaAllocator.init(alloc);
        defer merge_arena.deinit();
        const merge_alloc = merge_arena.allocator();

        var merges = try readJsonMerges(encoder_reader, merge_alloc);
        _ = merges.remove("<|endoftext|>");
        _ = merges.remove("<|startoftext|>");

        //std.debug.print("Decoded {d} merges from encoder.json\n", .{merges.count()});
        //std.debug.print("Decoded {d} ranks from vocab.bpe\n", .{ranks.count()});
        std.debug.assert(merges.count() == ranks.count());

        return .{
            .tokens = ranks,
            .arena = arena,
            .rank_to_token = reversed,
            .special_tokens = specials,
            // TODO: Figure out if we ever need to pass in BOS or EOS IDs.
            //       this method is only used for testing, so maybe we don't need it.
            .bos = null,
            .eos = null,
        };
    }

    /// Parse encoder.json
    fn readJsonMerges(encoder_reader: anytype, alloc: std.mem.Allocator) !std.StringHashMap(usize) {
        var json_reader = std.json.reader(alloc, encoder_reader);
        defer json_reader.deinit();

        const Merges = std.StringHashMap(usize);
        var merges = Merges.init(alloc);
        errdefer merges.deinit();

        var last_key: ?[]const u8 = null;
        while (true) {
            const tok = try json_reader.nextAlloc(alloc, .alloc_always);
            switch (tok) {
                .object_begin, .object_end => {},
                .end_of_document => {
                    //std.debug.print("end_of_document\n", .{});
                    break;
                },
                .allocated_string => |str| {
                    if (last_key) |other| {
                        std.debug.print("got string {s} but overwriting old {s}\n", .{ str, other });
                        alloc.free(other);
                    }

                    last_key = str;
                },
                .allocated_number => |num| {
                    defer alloc.free(num);
                    if (last_key) |key| {
                        const number = try std.fmt.parseInt(usize, num, 10);
                        try merges.put(key, number);
                        last_key = null;
                    } else {
                        std.debug.print("Got number {s} but there's no previous key\n", .{num});
                    }
                },
                // zig fmt: off
                .partial_number, .partial_string,
                .partial_string_escaped_1, .partial_string_escaped_2,
                .partial_string_escaped_3, .partial_string_escaped_4 => {
                    std.debug.print("Got partial variant: {any}\n", .{tok});
                    @panic("partial should never happen with .alloc_always");
                },
                // zig fmt: on
                else => {
                    std.debug.print("Got unknown token variant: {any}\n", .{tok});
                    @panic("Unknown variant");
                },
            }
        }
        return merges;
    }

    /// Free up any resources associated with this `Tokenizer` and invalidate any pointers to
    /// token strings or token entries.
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Encode `text` into a series of tokens.
    /// Any slice returned will be allocated with `token_alloc`. The allocator used for
    /// calling `init()` will not be used.
    /// The caller is responsible for freeing the slice of returned tokens with `token_alloc`.
    pub fn encode(self: Self, text: []const u8, option: EncodingOption, token_alloc: Allocator) ![]Self.Token {
        var tokens = TokenList.init(token_alloc);
        defer tokens.deinit();
        // Best guesstimate
        try tokens.ensureTotalCapacity(text.len / 2);

        if (option == .start and self.add_bos) {
            if (self.bos) |bos| {
                try tokens.append(bos);
            }
        }

        // Encode the source `text` by sliding a working window through the text and tokenizing
        // in parts.
        //
        // Pay attention and do extra handling for special tokens just like the `tiktoken`
        // reference implementation.
        var working = text[0..];
        const specials = self.special_tokens.keys()[0..self.special_tokens.count()];
        while (working.len != 0) {
            var special_id: ?TokenizerToken = null;
            var next_idx = working.len;
            var window = working[0..];

            // If we have any special tokens, get the span between start of working buffer and
            // the special token location. Then set that "clean" span to the current window.
            if (findFirstMultiple(working, specials)) |match| {
                const idx = match[0];
                const special_match = match[1];
                special_id = self.special_tokens.get(special_match);
                std.debug.assert(special_id != null);

                // Get portion of window up to special token
                window = working[0..idx];
                // How far to advance forward
                next_idx = idx + special_match.len;
            }

            // Tokenize the current window
            var iter = regex.Gpt2Pattern.init(window);
            while (iter.next()) |group| {
                if (self.tokens.get(group)) |token| {
                    // If we have an elementary token, append that
                    try tokens.append(@intCast(token));
                } else {
                    // Otherwise, break down group into individual tokens
                    try self.encode_piece(group, &tokens, token_alloc);
                }
            }

            // Append special token where they belong in original text, after the processed window
            if (special_id) |id| {
                try tokens.append(id);
            }
            // Move sliding window forward
            working = working[next_idx..];
        }

        return tokens.toOwnedSlice();
    }

    /// Encode a `piece` within a greater input string into BPE-form and append to `tokens` list.
    /// Assumes that the `piece` is representable in the BPE form for this `TikTokenizer`.
    /// Requires the provided piece is non-empty.
    fn encode_piece(self: Self, piece: []const u8, tokens: *TokenList, allocator: Allocator) !void {
        // Base case, the piece is just 1 token that we can directly encode.
        if (self.tokens.get(piece)) |val| {
            try tokens.append(val);
            return;
        }

        var arena = ArenaAllocator.init(allocator);
        defer arena.deinit();
        const alloc = arena.allocator();

        // We have a composite piece which can be decomposed. Go ahead and perform Byte-Pair
        // Encoding and then merge up.
        //
        // Below is a simplified reproduction of `_byte_pair_merge` within `tiktoken/src/lib.rs`
        const Pairs = std.ArrayList(struct { usize, Rank });
        var parts = Pairs.init(alloc);
        defer parts.deinit();
        try parts.ensureTotalCapacity(piece.len + 1);

        const maximum = std.math.maxInt(Rank);
        var min_rank: struct { Rank, usize } = .{ maximum, std.math.maxInt(usize) };

        // Populate initial `parts` list with baseline BPE ranks for merging.
        for (0..piece.len - 1) |i| {
            const tmp = piece[i .. i + 2];
            const r = self.tokens.get(tmp) orelse maximum;
            if (r < min_rank[0]) {
                min_rank[0] = r;
                min_rank[1] = i;
            }
            try parts.append(.{ i, r });
        }

        try parts.append(.{ piece.len - 1, maximum });
        try parts.append(.{ piece.len, maximum });

        // Merge together ranks in the `parts` list and try to elide the middle element if possible.
        const getRank = struct {
            // Merge together the entries `[i, i+1, i+2]` and return the lower-scoring form
            // if it exists, or return the maximum score value otherwise.
            fn getRank(pc_: []const u8, rs_: Ranks, parts_: Pairs, i: usize) Rank {
                if (i + 3 < parts_.items.len) {
                    const tmp = pc_[parts_.items[i][0]..parts_.items[i + 3][0]];

                    return rs_.get(tmp) orelse maximum;
                } else {
                    return maximum;
                }
            }
        }.getRank;

        // While not at the end, merge pairs together, going from the lower rank pair to the highest
        // rank pair.
        while (min_rank[0] != maximum) {
            //std.debug.print("piece: {s}; {any}\n", .{ piece, parts.items });
            const i = min_rank[1];

            if (i > 0) {
                parts.items[i - 1][1] = getRank(piece, self.tokens, parts, i - 1);
            }

            parts.items[i][1] = getRank(piece, self.tokens, parts, i);
            _ = parts.orderedRemove(i + 1);

            min_rank = .{ maximum, std.math.maxInt(usize) };
            for (0.., parts.items[0 .. parts.items.len - 1]) |j, pair| {
                const rank = pair[1];

                if (rank < min_rank[0]) {
                    min_rank = .{ rank, j };
                }
            }
        }

        // Now we're back in `byte_pair_encode`, so iterate over `parts` in pairs of 2
        // and find the rank for each BPE and then add that back to tokens.
        var i: usize = 0;
        while (i < parts.items.len - 1) : (i += 1) {
            const lhs = parts.items[i][0];
            const rhs = parts.items[i + 1][0];
            const str = piece[lhs..rhs];
            //std.debug.print("Adding token for str <<{s}>>\n", .{str});
            if (self.tokens.get(str)) |bpe| {
                try tokens.append(bpe);
            } else {
                std.debug.print("Issue finding BPE encoding for {s} ({d}-{d})\n", .{ str, lhs, rhs });
                @panic("Unable to BPE");
            }
        }
    }

    /// Decode a list of `tokens` into the string representation. Caller is responsible for freeing
    /// the returned memory.
    pub fn decode(self: Self, tokens: []const Self.Token, allocator: std.mem.Allocator) ![]u8 {
        var ret = std.ArrayList(u8).init(allocator);
        try ret.ensureTotalCapacity(tokens.len * 4);
        defer ret.deinit();

        for (tokens) |token| {
            if (self.rank_to_token.get(token)) |chars| {
                try ret.appendSlice(chars);
            } else {
                std.debug.print("Could not find matching char string for BPE token {d}\n", .{token});
                @panic("Invalid token");
            }
        }

        return try ret.toOwnedSlice();
    }

    /// Cast a slice of this vocabulary's `tokens` into a generic slice of `TokenizerToken`.
    pub fn toGenericTokens(tokens: []const Self.Token) []const TokenizerToken {
        const as_bytes = std.mem.sliceAsBytes(tokens);
        return std.mem.bytesAsSlice(TokenizerToken, as_bytes);
    }

    /// Cast a slice of generic `tokens` into a slice of this vocabulary's tokens.
    pub fn fromGenericTokens(tokens: []const TokenizerToken) []const Self.Token {
        const as_bytes = std.mem.sliceAsBytes(tokens);
        return std.mem.bytesAsSlice(Self.Token, as_bytes);
    }
};

test "GPT-2 Tokenizer" {
    var vocab_stream = std.io.fixedBufferStream(@embedFile("assets/gpt2-vocab.bpe"));
    const vocab = vocab_stream.reader();
    var encoder_stream = std.io.fixedBufferStream(@embedFile("assets/gpt2-encoder.json"));
    const encoder = encoder_stream.reader();

    const alloc = std.testing.allocator;

    var special_tokens = TikTokenizer.Ranks.init(alloc);
    defer special_tokens.deinit();
    try special_tokens.put("<|endoftext|>", 50256);

    var tokenizer = try TikTokenizer.loadDataToBPERanks(vocab, encoder, special_tokens, alloc);
    defer tokenizer.deinit();

    // Find these values from [1] or derive them manually with the `tiktoken` Python library
    // [1]: https://tiktokenizer.vercel.app/?model=gpt2
    const T = TikTokenizer.Token;
    const h1 = [_]T{ 15496, 11, 1545 };
    const h2 = [_]T{ 15496, 11, 995, 0 };
    const h3 = [_]T{ 40, 1392, 534, 1271, 340, 338, 807, 3134, 12, 20, 26895, 0 };
    const h4 = [_]T{ 1639, 389, 7062, 50256 };
    const h5 = [_]T{ 464, 1049, 3355, 12520, 100, 109, 318, 287, 12520, 229, 101, 8582, 229, 111 };
    const h6 = [_]T{ 1639, 389, 7062, 50256, 271, 340, 1107, 996, 50256, 3549, 2420, 994 };
    const h7 = [_]T{ 50256, 1662, 262, 886, 996, 0, 50256 };
    const h8 = [_]T{
        464,   1049, 3355,  12520, 100, 109, 318, 287,  12520, 229, 101,  8582,
        229,   111,  50256, 40,    561, 588, 284, 3187, 340,   530, 1110, 13,
        50256,
    };
    const cases = [_]struct { []const u8, []const T }{
        .{ "Hello, friend", &h1 },
        .{ "Hello, world!", &h2 },
        .{ "I got your number it's 867-5309!", &h3 },
        .{ "You are welcome<|endoftext|>", &h4 },
        .{ "The great wall 🧱 is in 🇨🇳", &h5 },
        .{ "You are welcome<|endoftext|>is it really though<|endoftext|>more text here", &h6 },
        .{ "<|endoftext|>not the end though!<|endoftext|>", &h7 },
        .{ "The great wall 🧱 is in 🇨🇳<|endoftext|>I would like to visit it one day.<|endoftext|>", &h8 },
    };

    for (cases) |case| {
        const input = case[0];
        const expected = case[1];
        const actual = try tokenizer.encode(input, .none, alloc);
        defer alloc.free(actual);
        try std.testing.expectEqualSlices(TikTokenizer.Token, expected, actual);

        const decoded = try tokenizer.decode(actual, alloc);
        defer alloc.free(decoded);
        try std.testing.expectEqualSlices(u8, input, decoded);
    }
}

/// Search for the first index and match of an element of `needles` within `haystack` using a naïve
/// brute force approach.
/// If no match is found, return `null`.
/// If two or more elements in `needles` are identical, matches against the first are returned.
///
/// Requires that `needles` is a non-empty slice and that each element (or *needle*) is non-empty.
/// Assumes that `haystack` is a valid UTF-8 sequence.
fn findFirstMultiple(haystack: []const u8, needles: []const []const u8) ?struct { usize, []const u8 } {
    // TODO: Use Knuth-Morris-Pratt Algorithm [1] or better to speed this up.
    // [1]: https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm

    // Trivial cases
    if (haystack.len == 0) return null;
    if (needles.len == 0) return null;

    // Find maximum and minimum needle length
    var min_needle_len: usize = std.math.maxInt(usize);
    var max_needle_len: usize = 0;
    for (needles) |needle| {
        min_needle_len = @min(min_needle_len, needle.len);
        max_needle_len = @max(max_needle_len, needle.len);
    }

    std.debug.assert(min_needle_len != 0);

    // Handle more base cases
    if (min_needle_len > haystack.len) {
        return null;
    }

    // We checked for overflow above, so this is safe now.
    const end_index = haystack.len - min_needle_len + 1;

    var i: usize = 0;
    // Iterate each UTF-8 codepoint because we don't want to match in the middle of a codepoint
    // for equality because that's incorrect.
    var iter = std.unicode.Utf8Iterator{
        .bytes = haystack,
        .i = 0,
    };
    while (i <= end_index) {
        const current = haystack[i..];
        for (needles) |needle| {
            if (std.mem.startsWith(u8, current, needle)) {
                return .{ i, needle };
            }
        }

        if (iter.nextCodepoint()) |point| {
            // Advance forward the proper amount
            const to_advance = std.unicode.utf8CodepointSequenceLength(point) catch @panic("invalid utf-8 sequence");
            i += to_advance;
        } else {
            // Or advance forward a little bit
            i += @max(iter.peek(1).len, 1);
        }
        if (i >= end_index or i >= haystack.len) break;
    }

    return null;
}

test "findFirstMultiple cases" {
    const no_needles = "© There is no spoon 　";
    const missing_needles = [_][]const u8{"<|tag|>"};
    const no_match = findFirstMultiple(no_needles, &missing_needles);
    try std.testing.expect(no_match == null);

    const needles = [_][]const u8{ "red", "green", "blue" };
    const haystack = "RGB stands for red-green-blue";

    const match_red = findFirstMultiple(haystack, &needles);
    try std.testing.expect(match_red != null);
    try std.testing.expectEqualSlices(u8, "red", match_red.?[1]);

    const real_world = "You are welcome<|endoftext|>";
    const real_specials = [_][]const u8{"<|endoftext|>"};
    const real_match = findFirstMultiple(real_world, &real_specials);
    try std.testing.expect(real_match != null);
    try std.testing.expectEqualSlices(u8, real_match.?[1], real_specials[0]);
    try std.testing.expectEqual(15, real_match.?[0]);

    const uni_world = "© 2025 　<|tag|>";
    const uni_specials = [_][]const u8{"<|tag|>"};
    const uni_match = findFirstMultiple(uni_world, &uni_specials);
    try std.testing.expect(uni_match != null);
    try std.testing.expectEqualSlices(u8, uni_match.?[1], uni_specials[0]);
    try std.testing.expectEqual(11, uni_match.?[0]);
}
