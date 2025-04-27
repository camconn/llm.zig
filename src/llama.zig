// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! llama: This module contains an implementation of the LLaMA 2 LLM.
//!

const std = @import("std");

const llm = @import("root.zig");
const ggml = llm.ggml;
const math = llm.math;

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

pub const Error = error{
    /// You attempted to load a file that has an invalid format. Usually this means the file had
    /// a bad header or the wrong version. This could also mean that the file is truncated or has
    /// extra data at the end.
    BadFormat,
    /// You tried to load a file that has a correct format but is semantically incorrect. Usually
    /// this means you tried to load the wrong model.
    BadFile,
};

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    max_seq_length: usize,

    shared_classifier: bool,

    /// Read the configuration from a GGUF file.
    pub fn readGGUF(file: ggml.GGUFFile) Config {
        return .{
            .dim = file.getValue("llama.embedding_length").?.uint32,
            .hidden_dim = file.getValue("llama.feed_forward_length").?.uint32,
            .n_layers = file.getValue("llama.block_count").?.uint32,
            .n_heads = file.getValue("llama.attention.head_count").?.uint32,
            .n_kv_heads = file.getValue("llama.attention.head_count_kv").?.uint32,
            .vocab_size = file.getValue("tokenizer.ggml.tokens").?.array.len,
            .max_seq_length = file.getValue("llama.context_length").?.uint32,
            .shared_classifier = file.getValue("output.weight") == null,
        };
    }

    /// Temporarily open the checkpoint file to read only the header.
    /// Do not leave the file open because we will mmap it.
    pub fn read(model_path: []const u8) !Config {
        // First try to load the model with relative and then fallback to absolute path if that fails.
        const options: std.fs.File.OpenFlags = .{ .mode = .read_only };
        const cwd = std.fs.cwd();

        // zig fmt: off
        const file: ?std.fs.File = cwd.openFile(model_path, options)
            // Fall back to absolute path
            catch (std.fs.openFileAbsolute(model_path, options) catch null);
        // zig fmt: on

        if (file == null) {
            std.debug.print("Could open model file: {s}\nIs the path correct?\n", .{model_path});
            return std.fs.File.OpenError.FileNotFound;
        }
        defer file.?.close();

        var buffer = std.io.bufferedReader(file.?.reader());
        return try Config.read_inner(buffer.reader());
    }

    const magic_v1_str = "ak42";
    const magic_v1_num = std.mem.readInt(u32, magic_v1_str, .little);

    /// Read a "Version 1" `llama.bin` file as exported by `llama2.c/export.py`
    fn read_inner(reader: anytype) !Config {
        // Assume the machine which exports the v1 file is Little-Endian, which make parsing
        // easier.
        // First, we have the header
        const magic = try reader.readInt(u32, .little);
        if (magic != magic_v1_num) {
            return Error.BadFormat;
        }
        // Next, a signed integer for the export version
        const version = try reader.readInt(i32, .little);
        if (version != 1) {
            return error.BadFormat;
        }
        // The next 7 values are 32 bit signed numbers
        const dim = try reader.readInt(i32, .little);
        const hidden_dim = try reader.readInt(i32, .little);
        const n_layers = try reader.readInt(i32, .little);
        const n_heads = try reader.readInt(i32, .little);
        const n_kv_heads = try reader.readInt(i32, .little);
        const vocab_size = try reader.readInt(i32, .little);
        // There's a bug in the `export.py` method. It actually hard-codes the sequence length to
        // 2048 for V1 which is incorrect. That is at most half of the actual sequence length.
        const max_seq_length = try reader.readInt(i32, .little) * 2;

        // Next byte indicates if the token embeddings are shared between the last layer of the
        // model and the tokenizer embeddings.
        const shared_classifier = try reader.readByte() == 1;

        // And that's the end of the header.

        return Config{
            .dim = @intCast(dim),
            .hidden_dim = @intCast(hidden_dim),
            .n_layers = @intCast(n_layers),
            .n_heads = @intCast(n_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .vocab_size = @intCast(vocab_size),
            .max_seq_length = @intCast(max_seq_length),
            .shared_classifier = shared_classifier,
        };
    }
};

pub const Tokenizer = struct {
    // TODO: Read the original `sentencepiece.sentencepice_model_pb.ModelProto()` file instead
    //       to read original/custom `.model` files.
    const Self = @This();

    /// A single Token's ID. Represents one of 32000 vocab value or a sentinel token.
    /// Note that a padding token == -1, hence the signed-ness.
    pub const Token = i16;
    const TokenEntry = struct {
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

    pub fn init(file_path: []const u8, alloc: Allocator, vocab_size: usize) !Tokenizer {
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
        return try Tokenizer.read(buffer.reader(), alloc, vocab_size);
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
    fn read(reader: anytype, allocator: Allocator, vocab_size: usize) !Tokenizer {
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

    fn initGGUF(file: ggml.GGUFFile, allocator: Allocator) !Tokenizer {
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
            const token = Tokenizer.TokenEntry{
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

    pub fn deinit(self: *Self) void {
        self.tokens.deinit(self.arena.allocator());
        self.arena.deinit();
    }

    const UNK = 0;
    const BOS = 1;
    const EOS = 2;
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
    pub fn findIndexByTokenId(self: Self, token: Token) ?usize {
        // TODO: find a better path to make this private
        return self.token_to_idx[@intCast(token)];
    }
};

test "Tokenizer.encode" {
    const tok_path = try std.testing.allocator.dupe(u8, "tokenizer.bin");
    defer std.testing.allocator.free(tok_path);
    var tokenizer = try Tokenizer.init(tok_path, std.testing.allocator, 32000);
    defer tokenizer.deinit();

    {
        const sample = "Hello, world! How are you today?";
        const ids = [_]Tokenizer.Token{ 15043, 29892, 3186, 29991, 1128, 526, 366, 9826, 29973 };

        const tokens = try tokenizer.encode(sample, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Tokenizer.Token, &ids, tokens);
    }

    {
        const sample = "Hello\nworld"; // TODO: Handle issue w/ leading spaces
        const ids = [_]Tokenizer.Token{ 15043, 13, 11526 };

        const tokens = try tokenizer.encode(sample, std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Tokenizer.Token, &ids, tokens);
    }

    {
        const sample = "Byte pair encoding[1][2] (also known as BPE, or digram";
        // zig fmt: off
        const ids = [_]Tokenizer.Token{
            19831, 5101, 8025, 29961, 29896, 3816, 29906, 29962,
            313, 15189, 2998, 408, 350, 4162, 29892, 470,
            4697, 2572
        };
        // zig fmt: on

        const tokens = try tokenizer.encode(sample[0..], std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Tokenizer.Token, &ids, tokens);
    }

    {
        const sample = @embedFile("assets/bpe_sample.txt");
        const out_ids = @embedFile("assets/bpe_sample_expected.json");

        var ids_json = try std.json.parseFromSlice([]Tokenizer.Token, std.testing.allocator, out_ids, .{});
        defer ids_json.deinit();
        const ids = ids_json.value;

        const tokens = try tokenizer.encode(sample[0..], std.testing.allocator);
        defer std.testing.allocator.free(tokens);

        try std.testing.expectEqualSlices(Tokenizer.Token, ids, tokens);
    }
}

/// Represents the current state of a transformer.
pub const State = struct {
    arena: ArenaAllocator,

    // Precomputed coefficients
    sin: []f32,
    cos: []f32,

    // x
    input: []f32,
    // Working state
    work: []f32,
    // Working state overflow
    work2: []f32,
    // hb
    hidden1: []f32,
    // hb2
    hidden2: []f32,

    q: []f32,
    k: []f32,
    v: []f32,

    attention: []f32,
    // logits
    output: []f32,

    // caches
    // (layer, seq_len, dim)
    k_cache: []f32,
    v_cache: []f32,

    /// Initialize the working state.
    /// Adapted from Andrej Karpathy's `llama2.c`
    pub fn init(alloc: Allocator, config: Config) !State {
        var arena = ArenaAllocator.init(alloc);
        var a = arena.allocator();

        const kv_dim = config.dim * config.n_kv_heads / config.n_heads;

        const x = try a.alloc(f32, config.dim);
        const xb = try a.alloc(f32, config.dim);
        const xb2 = try a.alloc(f32, config.dim);
        const hb1 = try a.alloc(f32, config.hidden_dim);
        const hb2 = try a.alloc(f32, config.hidden_dim);

        const q = try a.alloc(f32, config.dim);
        const k = try a.alloc(f32, kv_dim);
        const v = try a.alloc(f32, kv_dim);

        const k_cache = try a.alloc(f32, config.n_layers * config.max_seq_length * kv_dim);
        const v_cache = try a.alloc(f32, config.n_layers * config.max_seq_length * kv_dim);

        const att = try a.alloc(f32, config.n_heads * config.max_seq_length);
        const logits = try a.alloc(f32, config.vocab_size);

        const sin, const cos = try precompute_frequencies(
            config.dim,
            config.n_heads,
            config.max_seq_length,
            a,
        );

        return .{
            // Internal memory lifetime
            .arena = arena,
            // Precomputed
            .sin = sin,
            .cos = cos,
            // layers and temporaries
            .input = x,
            .work = xb,
            .work2 = xb2,
            .hidden1 = hb1,
            .hidden2 = hb2,
            .q = q,
            .k = k,
            .v = v,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .attention = att,
            .output = logits,
        };
    }

    /// Deinitialize the state of this Transformer.
    pub fn deinit(self: *@This()) void {
        self.arena.deinit();
    }
};

/// Precompute frequencies for RoPE (Rotary Position Embeddings).
/// Caller owns the returned memory.
///
/// Returns two slices of coefficients corresponding to the `cos(m θ_n)` or `sin(m θ_n)` values
/// within the element-wise product of R^d_{Θ,m}x found in section 3.4.2 of the RoFormer paper [1].
/// Note that the implementation here uses a different definition of θ_i than the original paper.
///
/// Based on `precompute_freqs_cis` from Meta's LLaMA.
/// [1]: https://arxiv.org/abs/2104.09864v5
fn precompute_frequencies(
    dim: usize,
    n_heads: usize,
    len: usize,
    allocator: Allocator,
) !struct { []f32, []f32 } {
    const head_size = dim / n_heads;
    std.debug.assert(head_size & 1 == 0); // `head_size` must be divisible by 2.
    // We only compute half the number of coefficients in a head of `head_size` because RoPE
    // collapses a head of size d → d/2.
    const half_head = head_size / 2;
    const compute_len = len * half_head;

    var sin = try allocator.alloc(f32, compute_len);
    errdefer allocator.free(sin);
    var cos = try allocator.alloc(f32, compute_len);
    errdefer allocator.free(cos);

    //std.debug.print("dim: {d}; len: {d}; compute: {d}\n", .{ dim, len, compute_len });
    //std.debug.print("heads: {d}; head_size: {d}; prod {d}\n", .{ n_heads, head_size, n_heads * head_size });

    const head_size_f: f32 = @floatFromInt(head_size);

    for (0..len) |m| {
        const m_f: f32 = @floatFromInt(m);
        for (0..half_head) |hd| {
            // The cosine embedding for token #m at head position #h is a function of the position
            // on the head (h) and the token (m)
            // m_cos(m, h) = cos(m * theta)
            //             = cos(m * pow(10_000 * -1 * pow))
            //             = cos(m * pow(10_000 * -1 * (h / head_size)))
            // and likewise with m_sin(n, h)

            // Recover original head position by doubling since we are iterating over half of
            // the head size.
            const h_f: f32 = @floatFromInt(hd * 2);

            const pow = h_f / head_size_f;
            const theta = std.math.pow(f32, 10_000, -1 * pow);
            const m_theta = m_f * @as(f32, @floatCast(theta));

            const m_cos = std.math.cos(m_theta);
            const m_sin = std.math.sin(m_theta);

            const i = half_head * m + hd;

            sin[i] = m_sin;
            cos[i] = m_cos;
        }
    }
    return .{ sin, cos };
}

test "check precompute_frequencies static" {
    // Precomputed coefficients as exported into "v0" llama2.c model file for llama2 7b.
    const cos_file = @embedFile("assets/cos.json");
    const sin_file = @embedFile("assets/sin.json");

    const cos = try std.json.parseFromSlice([]f32, std.testing.allocator, cos_file, .{});
    defer cos.deinit();
    const sin = try std.json.parseFromSlice([]f32, std.testing.allocator, sin_file, .{});
    defer sin.deinit();

    const len = 2048;
    const sin_actual, const cos_actual = try precompute_frequencies(4096, 32, len, std.testing.allocator);
    defer std.testing.allocator.free(sin_actual);
    defer std.testing.allocator.free(cos_actual);

    try std.testing.expectEqual(sin.value.len, sin_actual.len);
    try std.testing.expectEqual(cos.value.len, cos_actual.len);

    const epsilon = 9e-3; // 1e-4 is too strict and fails
    for (0..cos.value.len, cos.value, cos_actual) |i, exp, actual| {
        if (!std.math.approxEqAbs(f32, exp, actual, epsilon)) {
            const diff = @abs(exp - actual);
            std.debug.print("Failure at cos index {d}: difference {d} exceeded {d}\n", .{ i, diff, epsilon });
            try std.testing.expectApproxEqAbs(exp, actual, epsilon);
        }
    }

    for (0..sin.value.len, sin.value, sin_actual) |i, exp, actual| {
        if (!std.math.approxEqAbs(f32, exp, actual, epsilon)) {
            const diff = @abs(exp - actual);
            std.debug.print("Failure at sin index {d}: difference {d} exceeded {d}\n", .{ i, diff, epsilon });
            try std.testing.expectApproxEqAbs(exp, actual, epsilon);
        }
    }
}

const Complex = std.math.Complex(f32);

/// Apply RoPE embeddings to a vector.
/// Accepts as argument the vector to apply embeddings to as well as the pre-computed slices of
/// sin(mθ_i) and cos(mθ_i) as provided by `precompute_frequencies`.
///
/// Expects to receive parameters about the transformer model such as the number of heads `n_heads`,
/// the size of the head to operate on `head_size`, and the current token position `n`.
fn apply_rope_embeddings(
    vector: []f32,
    sin: []const f32,
    cos: []const f32,
    n_heads: usize,
    head_size: usize,
    n: usize,
) void {
    // Base index of RoPE embedding coefficients
    const base = n * head_size / 2;

    for (0..n_heads) |hi| {
        var hd: usize = 0;
        while (hd < head_size) : (hd += 2) {
            const ii = base + (hd / 2);
            const vi = hi * head_size + hd;

            // Rotate each unit pair v ∈ vector by mθ_i
            //
            // Rotation is done by representing the point v = (v0, v1) to rotate in
            // complex form and then representing the angle to rotate mθ_i as a pre-computed
            // rotation matrix in complex form.

            // Find point (v0, v1) to rotate
            const v0 = vector[vi];
            const v1 = vector[vi + 1];
            const v = Complex{ .re = v0, .im = v1 };

            // The rotating Complex num by angle by mθ_i is equivalent to matrix multiplication:
            // [[cos mθ_i   -sin mθ_i]  *  [v0   v1]^T
            //  [sin mθ_i    cos mθ_i]]
            //  = [(v0*cos mθ_i - v1*sin mθ_i)  (v0*sin mθ_i  + v1*cos mθ_i)]^T
            //
            // Which is basically just multiplying the two complex values:
            //  (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
            //
            // So if a = cos mθ_i, b = sin mθ_i, c = v0, d = v1,
            //
            //  (cos mθ_i  + i sin(mθ_i)) * (v0 + i v1)
            //  = (v0*cos(mθ_i) - v1*sin mθ_i) + i(v1*cos mθ_i + v0*sin mθ_i)
            const m_sin = sin[ii];
            const m_cos = cos[ii];
            const rot = Complex{ .re = m_cos, .im = m_sin };

            const v_rot = v.mul(rot);

            // zig fmt: off
            vector[vi]     = v_rot.re;
            vector[vi + 1] = v_rot.im;
            // zig fmt: on
        }
    }
}

/// Represents the flattened weights of a neural network.
/// The best guess we have is that the alignment is *4* based on the size of an f32.
const Weights = []align(4) const f32;

/// Represents a single `TransformerBlock` of the Llama V1/V2 transformer.
/// This method contains all the weights for a single "layer" of a model's layers.
pub const TransformerBlock = struct {
    attn_norm: math.Weights,
    wq: math.Weights,
    wk: math.Weights,
    wv: math.Weights,
    wo: math.Weights,

    ffn_norm: math.Weights,
    w1: math.Weights,
    w2: math.Weights,
    w3: math.Weights,

    /// Initialize a transformer block with the provided backing `file` `mmap(2)` using config
    /// if this is `n`-th layer.
    fn initGGML(file: ggml.GGUFFile, _: Config, n: usize) TransformerBlock {
        // TODO: Assert that weights have correct sizes w/ passed-in config
        const attn_norm = getTensor("attn_norm", file, n);
        const wq = getTensor("attn_q", file, n);
        const wk = getTensor("attn_k", file, n);
        const wv = getTensor("attn_v", file, n);
        const wo = getTensor("attn_output", file, n);

        const ffn_norm = getTensor("ffn_norm", file, n);
        const w1 = getTensor("ffn_gate", file, n);
        const w2 = getTensor("ffn_down", file, n);
        const w3 = getTensor("ffn_up", file, n);

        return .{
            .attn_norm = attn_norm,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .ffn_norm = ffn_norm,
            .w1 = w1,
            .w2 = w2,
            .w3 = w3,
        };
    }

    /// Get the contents of a tensor named `blk.<n>.<name>.weight` from the loaded GGUF `file`.
    /// Assumes that the backing tensor exists within the GGUF file.
    fn getTensor(name: []const u8, file: ggml.GGUFFile, n: usize) math.Weights {
        std.debug.assert(n < 32);
        var buf = [_]u8{0} ** 64;
        const full_name = std.fmt.bufPrint(&buf, "blk.{d}.{s}.weight", .{ n, name }) catch unreachable;
        if (file.getTensorInfo(full_name)) |tensor| {
            return tensor.getElems(file.tensor_data_offset);
        }
        std.debug.print("Error: Tensor {s} does not exist in the GGUF file", .{full_name});
        @panic("Could not load tensor");
    }
};

/// The weights of a Llama model - specifically the transformer bits.
///
/// This struct is immutable and represents the contents of a loaded model from a file on storage.
/// The file is read into memory with `mmap(2)`.
pub const TransformerV1 = struct {
    const Self = @This();
    const page_size_min = std.heap.page_size_min;
    const header_len = 256;

    // Internally managed resources
    fd: std.posix.fd_t,
    ptr: ?[]align(page_size_min) u8,
    arena: ArenaAllocator,

    // Config weights
    config: Config,

    // Embeddings
    token_embed: math.Weights,
    // Transformer layers
    layers: []TransformerBlock,
    // Output norms
    norm: math.Weights,
    // Only set whenever no shared classifier layer with tokenizer
    classifier: math.Weights,

    /// Initialize a transformer from a `.bin` file exported by the *V1 Export* of `llama2.c`'s
    /// [export.py script](https://github.com/karpathy/llama2.c/blob/master/export.py).
    ///
    /// Refer to this project's `README.md` for details on how to generate one.
    pub fn initV1(model_path: []const u8, config: Config, alloc: Allocator) !TransformerV1 {
        //std.debug.print("Opening Transformer model\n", .{});
        const fd = try std.posix.open(model_path, .{}, 0o440);
        errdefer std.posix.close(fd);

        // Here we mmap() the weights files because nobody wants to open up a 25 GB file raw!
        const stat = try std.posix.fstat(fd);
        const fsize: u64 = @intCast(stat.size);
        std.debug.print(
            "Model size: {d:.1} MiB\n",
            .{@as(f32, @floatFromInt(fsize)) / 1048576.0},
        );

        // See `std.os.linux.MAP` for more info.
        const mmap_type: std.posix.MAP = .{
            .TYPE = .SHARED,
            // Linux-only flags
            // Don't reserve swap pages
            .NORESERVE = true,
            // Try to populate all of the pages in memory (pre-read) if possible
            .POPULATE = true,
        };

        const ptr = try std.posix.mmap(null, fsize, std.posix.PROT.READ, mmap_type, fd, 0);
        errdefer std.posix.munmap(ptr);

        // Tell the kernel we expect to do sequential reads and that we will need the
        // mapped file in the near future.
        //
        // When measuring this it makes layer times more consistent, especially for the first
        // few iterations of `Transformer.forward()`
        const madvise_flags = std.posix.MADV.SEQUENTIAL | std.posix.MADV.WILLNEED;
        try std.posix.madvise(ptr.ptr, fsize, madvise_flags);

        // V1 Export Format from `llama2.c`
        // The first 256 bytes contain the header with trailing 0 padding.

        // TODO: Load the `Config` header here instead of in a separate discrete function.

        // We have the ptr. Time to handle it.
        const total_len: usize = @as(usize, @intCast(fsize)) - header_len;
        const total_elems = total_len / @sizeOf(f32);

        const raw_ptr: [*]align(4) f32 = @ptrFromInt(@intFromPtr(&ptr[header_len]));
        const weights = raw_ptr[0..total_elems];

        const vocab = config.vocab_size;
        const dim = config.dim;
        const hidden_dim = config.hidden_dim;
        const n_layers = config.n_layers;
        const n_heads = config.n_heads;
        const n_kv_heads = config.n_kv_heads;
        const head_size = dim / config.n_heads;

        // According to the output from the modified `export.py` file, we have these dimensions:
        // layers: 32 (known from config)
        var i: usize = 0;

        // normalization layers (7b)
        // attention:   [4096]f32
        // ffn_norm:    [4096]f32
        // norm:        [4096]f32
        const rms_attention: Weights = weights[i .. i + n_layers * dim];
        i += rms_attention.len;
        const ffn_norm: Weights = weights[i .. i + n_layers * dim];
        i += ffn_norm.len;
        const norms: Weights = weights[i .. i + dim];
        i += norms.len;

        // token embeddings (7b)
        // embed:       [vocab_size][4096]f32
        const token_embed: Weights = weights[i .. i + vocab * dim];
        i += token_embed.len;

        // attention layers (7b)
        // wq:          [4096][4096]f32
        // wk:          [4096][4096]f32
        // wv:          [4096][4096]f32
        // wo:          [4096][4096]f32

        const wq: Weights = weights[i .. i + n_layers * dim * (n_heads * head_size)];
        i += wq.len;
        const wk: Weights = weights[i .. i + n_layers * dim * (n_kv_heads * head_size)];
        i += wq.len;
        const wv: Weights = weights[i .. i + n_layers * dim * (n_kv_heads * head_size)];
        i += wv.len;
        const wo: Weights = weights[i .. i + n_layers * (n_heads * head_size) * dim];
        i += wo.len;

        // ff layers (7b)
        // w1:          [11008][4096]f32
        // w2:          [4096][11008]f32
        // w3:          [11008][4096]f32
        // output:      [vocab_size][4096]f32
        const w1: Weights = weights[i .. i + n_layers * dim * hidden_dim];
        i += w1.len;
        const w2: Weights = weights[i .. i + n_layers * hidden_dim * dim];
        i += w2.len;
        const w3: Weights = weights[i .. i + n_layers * dim * hidden_dim];
        i += w3.len;

        var out_classifier: ?Weights = null;
        if (config.shared_classifier) {
            out_classifier = token_embed;
        } else {
            out_classifier = weights[i .. i + dim * vocab];
            i += out_classifier.?.len;
        }

        // make sure we read the whole file.
        const used_size = i * @sizeOf(f32) + header_len;
        if (fsize != used_size) {
            return Error.BadFile;
        }

        // Convert weight slices to `TransformerBlock`s.
        const dim2 = dim * dim;
        const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        const kv_len = dim * kv_dim;

        var arena = ArenaAllocator.init(alloc);
        errdefer arena.deinit();
        var allocator = arena.allocator();

        var layers = try allocator.alloc(TransformerBlock, config.n_layers);
        errdefer allocator.free(layers);
        for (0..config.n_layers) |n| {
            const l_attn_norm = rms_attention[n * dim .. (n + 1) * dim];

            const n_dim2 = n * dim2;
            const l_wq = wq[n_dim2 .. n_dim2 + dim2];
            const l_wk = wk[n * kv_len .. (n + 1) * kv_len];
            const l_wv = wv[n * kv_len .. (n + 1) * kv_len];
            const l_wo = wo[n * dim2 .. (n + 1) * dim2];

            const ffn_o = n * dim * config.hidden_dim;
            const ffn_e = ffn_o + dim * config.hidden_dim;

            const l_ffn_norm = ffn_norm[n * dim .. (n + 1) * dim];
            const l_w1 = w1[ffn_o..ffn_e];
            const l_w2 = w2[ffn_o..ffn_e];
            const l_w3 = w3[ffn_o..ffn_e];

            layers[n] = TransformerBlock{
                .attn_norm = .{ .f32 = l_attn_norm },
                .wq = .{ .f32 = l_wq },
                .wk = .{ .f32 = l_wk },
                .wv = .{ .f32 = l_wv },
                .wo = .{ .f32 = l_wo },

                .ffn_norm = .{ .f32 = l_ffn_norm },
                .w1 = .{ .f32 = l_w1 },
                .w2 = .{ .f32 = l_w2 },
                .w3 = .{ .f32 = l_w3 },
            };
        }

        return TransformerV1{
            .config = config,
            .fd = fd,
            .ptr = ptr,
            .arena = arena,

            .token_embed = .{ .f32 = token_embed },
            .layers = layers,
            .norm = .{ .f32 = norms },
            .classifier = .{ .f32 = out_classifier.? },
        };
    }

    /// Unmap any memory and free any resources used by this `Transformer`.
    pub fn deinit(self: *Self) void {
        std.debug.print("Transformer.deinit()\n", .{});
        if (self.ptr) |inner| std.posix.munmap(inner);
        if (self.fd >= 0) std.posix.close(self.fd);

        self.arena.deinit();
        self.ptr = null;
        self.fd = -1;
    }

    /// Calculate a forward pass of the transformer with the next token `token` at
    /// position `n_token`.
    ///
    /// Returns a slice of logits from calculation. Caller **does not** down the returned
    /// slice and should not attempt to free it.
    pub fn forward(self: Self, state: *State, token: Tokenizer.Token, n_token: usize, progress: std.Progress.Node) []f32 {
        // Get a considerable speedup for operations that occur within here.
        @setFloatMode(.optimized);

        const c = self.config;

        const dim = c.dim;
        const kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
        //const kv_mul = c.n_heads / c.n_kv_heads;
        const head_size = c.dim / c.n_heads;

        const token_offset: usize = @as(usize, @intCast(token)) * dim;
        const embeddings = self.token_embed.f32[token_offset .. token_offset + dim];
        @memcpy(state.input, embeddings);

        const layer_progress = progress.start("Layer", c.n_layers);

        for (0..c.n_layers) |i| {
            // Layer offset for caching
            const layer_offset = i * c.max_seq_length * kv_dim;

            // Each "layer" is a TransformerBlock.

            // In TransformerBlock.__init__():
            //     attention = Attention(...)
            //     attention_norm = RMSNorm(...)
            //     ffn_norm = RMSNorm(...)
            //     feed_forward = FeedForward(dim, hidden_dim, ...)
            //
            // Each TransformerBlock.forward(x):
            //     h = x + attention.forward(attention_norm(input, freq_cs, freq_ss))
            //     out = h + feed_forward.forward(ffn_norm(h))
            //     return out
            //
            // In FeedForward.__init__():
            //     w1 = Linear(dim, hidden_dim, bias=False)
            //     w2 = Linear(hidden_dim, dim, bias=False)
            //     w3 = Linear(dim, hidden_dim, bias=False)
            // In Feedforward.forward(x):
            //     dropout(w1(F.silu(
            //                w1(x) * w3(x)
            //            )))
            // where F.silu = torch.nn.functional.silu
            // which is silu(x) = x (elementwise *) σ(x), where σ(x) is logistic sigmoid
            //
            // In Attention.__init__():
            //     # sets some constants known through `Config` already
            //     # NB: model_parallel_size = 1
            //     # NB: n_local_heads = n_heads
            //     n_rep = n_heads // n_kv_heads
            //     head_dim = dim // n_heads
            //     # NB: `bias=False` for all `Linear` layers here
            //     wq = Linear(dim, n_heads * head_dim)
            //     wk = Linear(dim, n_kv_heads * head_dim)
            //     wv = Linear(dim, n_kv_heads * head_dim)
            //     wo = Linear(n_heads * head_dim, dim)
            //     Ignore dropout
            //
            // In Attention.forward(x, freq_cs, freq_ss):
            //     bsz, seqlen, _ = x.shape
            //     xq, xk, xv = wq(x), wk(x), wv(x)
            //     xq = xq.view(bsz, seqlen, n_local_heads, head_dim)
            //     xk = xq.view(bsz, seqlen, n_local_kv_heads, head_dim)
            //     xv = xq.view(bsz, seqlen, n_local_kv_heads, head_dim)
            //
            //    # RoPE
            //    xq, xk = apply_rotary_emb(xq, xk, freq_cs, freq_ss)
            //
            //    # Batch MQ attention: expand out keys & values
            //    xk = repeat_kv(xk, n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            //    xv = repeat_kv(xv, n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            //
            //    xq = xq.transpose(1, 2)    # (bs, n_local_heads, seqlen, head_dim)
            //    xk = xk.transpose(1, 2)
            //    xv = xv.transpose(1, 2)
            //
            //    # Then flash attention or manual impl.
            //    # Manual attention:
            //    scores = matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
            //    scores = scores + self.mask[:, :, :seqlen, :seqlen]
            //    scores = F.softmax(scores.float, dim=-1).type_as(xq)
            //    output = matmul(scores, xv)
            //
            //    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            //
            //    return wo(output)

            const layer = self.layers[i];

            // TransformerBlock.forward()
            // first handle the `attention_norm` call.
            math.rmsNorm(state.work, state.input, layer.attn_norm.f32);

            // Now handle attention matrix multiplies
            // xq = wq(x)
            math.matrixMul(f32, state.q, layer.wq.f32, state.work, dim, dim);
            // xk = wk(x)
            math.matrixMul(f32, state.k, layer.wk.f32, state.work, kv_dim, dim);
            // xv = wv(x)
            math.matrixMul(f32, state.v, layer.wv.f32, state.work, kv_dim, dim);

            // RoPE
            //     xq, xk = apply_rotary_emb(xq, xk, freq_cs, freq_ss)
            //
            // Apply RoPE embeddings on `q` over `dim` and `k` over `kv_dim`.
            apply_rope_embeddings(state.q, state.sin, state.cos, c.n_heads, head_size, n_token);
            apply_rope_embeddings(state.k, state.sin, state.cos, c.n_kv_heads, head_size, n_token);

            {
                // Update kv cache
                std.debug.assert(dim == kv_dim);
                const cache_start = layer_offset + n_token * kv_dim;
                const cache_key_vec = state.k_cache[cache_start .. cache_start + kv_dim];
                const cache_val_vec = state.v_cache[cache_start .. cache_start + kv_dim];
                @memcpy(cache_key_vec, state.k);
                @memcpy(cache_val_vec, state.v);
            }

            // Inputs are state.q, state.k, state.v
            // Output is in state.work
            // Everything else can be mangled
            self.attention(state, i, n_token);

            // Almost done w/ Attention.forward(x), we just need to calculate the return
            // statement:
            //     return wo(output)
            const attention_preout = state.work[0..];

            math.matrixMul(f32, state.work2[0..dim], layer.wo.f32, attention_preout, dim, dim);
            // End of Attention.forward(x);

            // We are back in TransformerBlock.forward(x, freq_cs, freq_ss). We just need to add
            // the vector we stored in `state.work2` and add it to `x` then do a feed forward pass:
            //
            // def forward(self, x, few_cs, freq_ss):
            //     h = x + self.attention.forward(self.attention_norm(x), freqs_cs, freqs_ss)
            //     out = h + self.feed_forward.forward(self.ffn_norm(h))

            math.add(f32, state.input, state.input, state.work2);
            // We are no longer using the input `x` and can now use it as `h`.

            // Calculate ff = self.ffn_norm(h) = RMSNorm(h, feed_forward)
            math.rmsNorm(state.work, state.input, layer.ffn_norm.f32);

            // Calculate FeedForward.forward(x):
            //     return w2( silu(w1(x)) * w3(x) )

            // hid1 = w1(x)
            math.matrixMul(f32, state.hidden1, layer.w1.f32, state.work, c.hidden_dim, dim);
            // hid2 = w3(x)
            math.matrixMul(f32, state.hidden2, layer.w3.f32, state.work, c.hidden_dim, dim);

            // Calculate SwiGLU
            math.swiglu(state.hidden1);
            math.elementProduct(state.hidden1, state.hidden1, state.hidden2);

            // w2 * (swiglu(w2(x)) * w3(x))
            math.matrixMul(f32, state.work, layer.w2.f32, state.hidden1, dim, c.hidden_dim);
            // Done with FeedForward.forward(x)

            // Add back `h` to result of FeedForward.forward(x)
            //     out = h + feed_foward.forward(ffn_norm(h))
            //     return out
            math.add(f32, state.input, state.input, state.work);

            // Done with TransformerBlock.forward();
            //std.debug.print("Done with layer {d}/{d} with {d} at {d}\n", .{ i, c.n_layers, token, n_token });

            layer_progress.completeOne();
        }
        layer_progress.end();

        var ending = progress.start("Token Output", 2);

        // Done with layers
        //     h = self.norm(h)
        math.rmsNorm(state.input, state.input, self.norm.f32);

        ending.completeOne();

        // We are doing inference only, so no calculation of cross-entropy is needed
        // Logits are found by feeding `h` (state.input) through a linear layer.
        math.matrixMul(f32, state.output, self.classifier.f32, state.input, c.vocab_size, dim);

        ending.completeOne();
        ending.end();

        progress.completeOne();

        return state.output;
    }

    /// Perform attention the current forward layer iteration of the Transformer.
    /// This calculates Attention(state.q, state.k, state.v) and writes the output
    /// into `state.work`.
    fn attention(self: Self, state: *State, layer: usize, n_token: usize) void {
        const c = self.config;

        const head_size = c.dim / c.n_heads;
        const kv_dim = (c.dim * c.n_kv_heads) / c.n_heads;
        const kv_mul = c.n_heads / c.n_kv_heads;
        const layer_offset = layer * c.max_seq_length * kv_dim;

        // Perform multi-head attention over all heads
        for (0..c.n_heads) |head| {
            // Attention is defined in "Attention is All You Need"
            // The formula is:
            //   Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
            // Where d_k is the dimension of the queries and keys, and the values
            // are of dimensions d_v.
            const query = state.q[head * head_size ..][0..head_size];
            const att = state.attention[head * c.max_seq_length ..][0 .. n_token + 1];

            // Calculate QK^T / sqrt(d_k) for each token (including the next one)
            const base = layer_offset + (head / kv_mul) * head_size;
            for (0..n_token + 1) |tok| {
                const key = state.k_cache[base + tok * kv_dim ..][0..head_size];

                const qk = math.dotProduct(query, key);
                const inner = qk / std.math.sqrt(@as(f32, @floatFromInt(head_size)));
                att[tok] = inner;
            }

            // Calculate softmax(QK^T/sqrt(d_k)
            math.softMax(att);

            // We now have the softmax(QK^T/sqrt(d_k)).
            // Need to multiply by value vector.
            var attn_tmp: []f32 = state.work[head * head_size .. (head + 1) * head_size];
            @memset(attn_tmp, 0);

            // Multiply softmax(QK^T/sqrt(d_k)) by the values (V)
            // This gives us the attention for each QKV pairing
            for (0..n_token + 1) |tok| {
                const value = state.v_cache[base + tok * kv_dim ..][0..head_size];
                const tok_attention = att[tok];

                for (0..head_size) |j| {
                    attn_tmp[j] += tok_attention * value[j];
                }
            }
            // now has the attention almost ready
            // We calculate the output by multiplying it by `wo`
        }
    }
};

pub const Params = struct {
    const Self = @This();

    temperature: f32,
    top_p: f32,
    random: std.Random,
    vocab_size: usize,

    const Pair = struct {
        t: Tokenizer.Token,
        f: f32,

        fn desc(_: void, lhs: Pair, rhs: Pair) bool {
            const fun = std.sort.desc(f32);
            return @call(.auto, fun, .{ {}, lhs.f, rhs.f });
        }
    };

    pub fn init(temperature: f32, top_p: f32, vocab_size: usize) Params {
        const now = std.time.milliTimestamp();
        var rng = std.Random.Xoroshiro128.init(@bitCast(now));
        return .{
            .temperature = temperature,
            .top_p = top_p,
            .random = rng.random(),
            .vocab_size = vocab_size,
        };
    }

    pub fn sample(self: *Self, probs: []f32, allocator: Allocator) !Tokenizer.Token {
        var next: Tokenizer.Token = undefined;
        if (self.temperature == 0) {
            const idx = std.sort.argMax(f32, probs, {}, std.sort.asc(f32)).?;
            next = @intCast(idx);
        } else {
            // Apply temperature
            for (0..self.vocab_size) |q| {
                probs[q] /= self.temperature;
            }
            math.softMax(probs);

            const random = self.random.float(f32);
            if (self.top_p <= 0 or self.top_p >= 1) {
                @panic("Unimplemented");
            } else {
                var arena = ArenaAllocator.init(allocator);
                defer arena.deinit();
                next = try self.sample_nucleus(probs, random, arena.allocator());
            }
        }
        return next;
    }

    fn sample_nucleus(self: *Self, probs: []f32, random: f32, alloc: Allocator) !Tokenizer.Token {
        var tokens = try alloc.alloc(Pair, self.vocab_size);
        for (0..self.vocab_size, probs) |i, p| {
            tokens[i] = Pair{ .t = @intCast(i), .f = p };
        }

        // Sort from highest to lowest
        std.mem.sortUnstable(Pair, tokens, {}, Pair.desc);

        // Find the first `last_idx` tokens which have a sum >= `self.top_p`
        var sum: f32 = 0;
        var last_idx = self.vocab_size;
        for (0..last_idx, tokens) |i, token| {
            sum += token.f;
            if (sum >= self.top_p) {
                last_idx = i + 1;
                break;
            }
        }

        // random ∈ [0, 1)
        // and
        //    sum ∈ [0, ~top_p]
        // so
        //      r ∈ [0, ~top_p]
        const r = random * sum;
        var cdf: f32 = 0;
        for (tokens[0..last_idx]) |token| {
            cdf += token.f;
            if (r < cdf) {
                return token.t;
            }
        }

        // Prevent OOB ref
        if (last_idx == self.vocab_size) {
            last_idx = self.vocab_size - 1;
        }
        // Did not find a token. Just return the last one
        return tokens[last_idx].t;
    }
};

pub const LlamaContext = struct {
    config: Config,
    transformer: TransformerV1,
    state: State,
    tokenizer: Tokenizer,
    file: ?ggml.GGUFFile = null,

    /// Initialize and read a new Llama V1 or V2 inference context from a provided GGUF file.
    /// Returns an `Error.BadFile` whenever the file is an unsupported model or quantization
    /// format.
    pub fn init(file_name: []const u8, alloc: std.mem.Allocator) !LlamaContext {
        var file = try ggml.GGUFFile.read_file(file_name, alloc);
        errdefer file.deinit();

        try ensureCorrectModel(file);

        const config = Config.readGGUF(file);

        // Load Tokenizer
        var tokenizer = try Tokenizer.initGGUF(file, alloc);
        errdefer tokenizer.deinit();

        var state = try State.init(alloc, config);
        errdefer state.deinit();

        // Load Transformer Weights
        const token_embed = try loadWeights(file, "token_embd.weight"); // sic
        const output_norm = try loadWeights(file, "output_norm.weight");
        const output_weight = try loadWeights(file, "output.weight");

        var arena = ArenaAllocator.init(alloc);
        errdefer arena.deinit();

        var arena_alloc = arena.allocator();
        var layers = try arena_alloc.alloc(TransformerBlock, config.n_layers);
        for (0..config.n_layers) |i| {
            layers[i] = TransformerBlock.initGGML(file, config, i);
        }

        const transformer = TransformerV1{
            .fd = -1,
            .ptr = null,
            .arena = arena,

            .config = config,

            .token_embed = token_embed,
            .layers = layers,
            .norm = output_norm,
            .classifier = output_weight,
        };

        return LlamaContext{
            .config = config,
            .transformer = transformer,
            .state = state,
            .tokenizer = tokenizer,
            .file = file,
        };
    }

    /// Ensure the provided GGUF `file` is correct and that we can actually load it.
    /// Ensures that the loaded file is correct and is in a quantization format we actually
    /// support.
    fn ensureCorrectModel(file: ggml.GGUFFile) !void {
        // No longer require a specific name, just print out and make a best effort attempt to make
        // sure the name has "Llama" in it.
        if (file.getValue(ggml.name_key)) |name| {
            const inner = name.string.str;
            if (std.ascii.indexOfIgnoreCase(inner, "llama") == null) {
                std.debug.print("Model does not report as llama. Found name: {s}\n", .{inner});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Could not find GGML model name: {s}\n", .{ggml.name_key});
            return Error.BadFile;
        }

        if (file.getValue(ggml.arch_key)) |arch| {
            const inner = arch.string.str;
            if (!std.mem.eql(u8, inner, "llama")) {
                std.debug.print("{s} is wrong\n", .{inner});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Could not find model architecture: {s}\n", .{ggml.arch_key});
            return Error.BadFile;
        }

        const tokenizer_key = "tokenizer.ggml.model";
        if (file.getValue(tokenizer_key)) |tok_model| {
            const model_name = tok_model.string.str;
            if (!std.mem.eql(u8, model_name, "llama")) {
                std.debug.print("Model reports to be {s} and not \"llama\"!\n", .{model_name});
                return Error.BadFile;
            }
        } else {
            std.debug.print("llama: Missing tokenizer model {s}\n", .{tokenizer_key});
            return error.BadFile;
        }

        if (file.fileType()) |ft| {
            if (ft != .ALL_F32 and ft != .MOSTLY_Q8_0) {
                std.debug.print("llama: Unsupported quantization: {}\n", .{ft});
                return error.BadFile;
            }
        } else {
            std.debug.print("llama: Could not load file type\n", .{});
            return error.BadFile;
        }
    }

    pub fn deinit(self: *@This()) void {
        self.transformer.deinit();
        self.state.deinit();
        self.tokenizer.deinit();
        if (self.file) |*inner| {
            inner.deinit();
        }
        self.file = null;
    }
};

fn loadWeights(file: ggml.GGUFFile, name: []const u8) !math.Weights {
    if (file.getTensorInfo(name)) |tensor| {
        return tensor.getElems(file.tensor_data_offset);
    }
    std.debug.print("Missing tensor weights in file: {s}\n", .{name});
    return Error.BadFile;
}
