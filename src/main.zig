// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

const std = @import("std");
const Allocator = std.mem.Allocator;

const clap = @import("clap");

const GPA = std.heap.GeneralPurposeAllocator(.{});
const ArenaAllocator = std.heap.ArenaAllocator;
const Complex = std.math.Complex;

pub fn main() !void {
    var gpa = GPA{};
    defer _ = gpa.deinit();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Display this help and exit.
        \\-m, --model <str>      Path to the model to use
        \\
    );

    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = gpa.allocator(),
    }) catch |err| {
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        return clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});
    }

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);

    // TODO: Handle prompt
    const stdout = bw.writer();
    try stdout.print("Loading model config\n", .{});
    try bw.flush();

    var alloc = gpa.allocator();

    const model_path = "/home/cam/dev/llama2.c/llama2-7b.bin";
    const model_path_dupe = try alloc.dupeZ(u8, model_path);
    defer alloc.free(model_path_dupe);
    const config = try read_config(model_path_dupe);
    try stdout.print("loaded config\n", .{});
    // zig fmt: off
    try stdout.print("dim: {d}, hidden: {d}, n_layers: {d}, n_heads: {d}, n_kv: {d}, vocab: {d}, max_seq: {d}, shared_classifier: {}\n", .{
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.max_seq_length,
        config.shared_classifier,
    });
    // zig fmt: on
    try bw.flush();

    const tokenizer_path = "tokenizer.bin";
    const tokenizer_path_dupe = try alloc.dupeZ(u8, tokenizer_path);
    defer alloc.free(tokenizer_path_dupe);
    var tokenizer = try load_tokenizer(tokenizer_path_dupe, alloc, config.vocab_size);
    defer tokenizer.deinit();
    try stdout.print("loaded tokenizer\nmax tokenizer length: {d}\n", .{tokenizer.max_len});
    try bw.flush();

    var transformer = try TransformerV1.init(model_path_dupe, config);
    defer transformer.deinit();
    try stdout.print("Done loading model...\n", .{});
    try bw.flush();

    {
        const prompt_example = "Hello, world! How are you today?";
        const prompt_dupe = try alloc.dupe(u8, prompt_example);
        defer alloc.free(prompt_dupe);

        const tokens = try tokenizer.encode(prompt_dupe, alloc);
        defer alloc.free(tokens);

        std.debug.print("Got {d} encoded tokens\n", .{tokens.len});
        for (0.., tokens) |i, tok| {
            std.debug.print("Token #{d} = {d}\n", .{ i, tok });
        }
    }

    try stdout.print("Done with work.\n", .{});
    try stdout.print("\n\n", .{});
    try bw.flush();

    try stdout.print("Cleaning up\n", .{});
    try bw.flush();
}

const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    max_seq_length: usize,

    shared_classifier: bool,

    /// Read a "Version 1" `llama.bin` file as exported by `llama2.c/export.py`
    fn read(reader: anytype) !Config {
        // Assume the machine which exports the v1 file is Little-Endian, which make parsing
        // easier.
        // First, we have the header
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x616b3432) { // ak42
            return error.BadMagic;
        }
        // Next, a signed integer for the export version
        const version = try reader.readInt(i32, .little);
        if (version != 1) {
            return error.Version;
        }
        // The next 7 values are 32 bit signed numbers
        const dim = try reader.readInt(i32, .little);
        const hidden_dim = try reader.readInt(i32, .little);
        const n_layers = try reader.readInt(i32, .little);
        const n_heads = try reader.readInt(i32, .little);
        const n_kv_heads = try reader.readInt(i32, .little);
        const vocab_size = try reader.readInt(i32, .little);
        const max_seq_length = try reader.readInt(i32, .little);

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

/// Temporarily open the checkpoint file to read only the header.
/// Do not leave the file open because we ill mmap it.
fn read_config(model_path: []u8) !Config {
    const model_file = try std.fs.openFileAbsolute(model_path, .{ .mode = .read_only });
    // TODO: Use relative path support
    //const cwd = std.fs.cwd();
    //const model_file = try cwd.openFile(model_path, .{ .mode = .read_only });
    defer model_file.close();

    var buffer = std.io.bufferedReader(model_file.reader());
    return try Config.read(buffer.reader());
}

const Tokenizer = struct {
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
            // That seems to not be required for decoding, so we don't worry about it.

            const entry = TokenEntry{
                .score = score,
                .id = @intCast(i),
                .chars = token,
            };

            try tokens.append(alloc, entry);
        }

        // Sort the token so that the `chars` elements are in order
        const Sorter = struct {
            const Sorter = @This();
            toks: *Storage,
            pub fn lessThan(self: Sorter, left_index: usize, right_index: usize) bool {
                const lhs = self.toks.items(.chars)[left_index];
                const rhs = self.toks.items(.chars)[right_index];

                // TODO: Use a real Unicode collator and ensure source tokens are normalized.
                return std.mem.order(u8, lhs, rhs) == .lt;
            }
        };

        //dumpTokens(tokens);

        tokens.sortUnstable(Sorter{ .toks = &tokens });
        //tokens.sort(Sorter{ .toks = &tokens });
        // `tokens` is now sorted by the source bytes/characters of each token entry in
        // semi-alphabetical order.

        // Copy the Area after all of the copies have been done, otherwise there will be
        // a leak from the Arena's allocations.
        // Same with Token Storage
        return .{
            .max_len = max_len,
            .tokens = tokens,
            .arena = arena,
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

    fn deinit(self: *Self) void {
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
    fn encode(self: Self, text: []const u8, token_alloc: Allocator) ![]Token {
        // TODO: Handle un-encodable symbols with `<UNK>` or `UNK` token.

        // Optimistically assume we will output the entire text character
        // as a token.
        var output = std.ArrayList(Token).init(token_alloc);
        defer output.deinit();

        // Make a heuristic guess about how long our tokenized sequence will be
        try output.ensureTotalCapacity(text.len >> 2);

        const slice = self.tokens.slice();
        const chars = slice.items(.chars);
        const ids = slice.items(.id);
        const scores = slice.items(.score);

        // SentencePiece automatically appends whitespace whenever the `add_dummy_prefix=True`
        // in the Normalizer.
        // Llama 2's tokenizer happens to use this setting, so go ahead and add a single space
        // in front of all tokens we search for.
        // This will be merged in the merge step so that [" ", "hello"] becomes [" hello"]
        const space = lookupIndex(self.tokens.items(.chars), SPACE_PIECE).?;
        // space's token ID is 35, but don't hard-code that for now
        try output.append(ids[space]);

        var idx: usize = 0;
        // BPE-like Search algorithm
        // Start with the longest possible string to match then slowly trim
        // trailing characters until we get a matching character.
        search: while (true) {
            const rem = @min(text.len - idx, self.max_len);
            var end = @min(idx + rem, text.len);

            // Candidate is an empty slice first.
            var candidate: []const u8 = undefined;

            var match: ?TokenEntry = null;
            trim: while (end > idx) {
                candidate = text[idx..end];

                if (lookupIndex(chars, candidate)) |found| {
                    match = self.tokens.get(found);
                    //std.debug.print("found match (id={d}): <<{s}>>\n", .{ ids[found], chars[found] });
                    break :trim;
                }

                //std.debug.print("no match for (len {d}) <<{s}>>\n", .{ candidate.len, candidate });
                end -= 1;
            }

            if (match) |found| {
                try output.append(found.id);
                //std.debug.print("old idx={d}, out_i={d}\n", .{ idx, output.items.len });
                idx += found.chars.len;
                //std.debug.print("new idx: {d}\n", .{idx});
                continue :search;
            } else if (idx == text.len) {
                //std.debug.print("Done searching\n", .{});
                break :search;
            } else {
                std.debug.print("Failed to find match at out_i={d}, idx={d}, max_len={d}, text.len={d}\n", .{ output.items.len, idx, self.max_len, text.len });
                //std.debug.print("idx={d}, max_len={d}, end={d}; other={s}\n", .{ idx, self.max_len, end, text[idx .. idx + self.max_len] });
                //std.debug.print("No match found for <<{s}>>\n", .{text[idx..end]});
                //std.debug.print("out_i={d}, idx={d}, text.len={d}\n", .{ output.items.len, idx, text.len });
                //std.debug.print("output {any}\n", .{output});
                @panic("Failed to tokenize candidate");
            }

            // Done searching
            break;
        }

        //std.debug.print("Performing merge starting with {d}\n", .{output.items.len});

        // Do merge algorithm
        var buf: [64]u8 = undefined;
        var i: usize = 0;
        while (i < output.items.len - 1) {
            //std.debug.print("merge with i={d}\n", .{i});
            var best: f32 = -999_999_999;
            var best_merge: ?TokenEntry = null;

            const left_idx = findTokenById(ids, output.items[i]).?;
            const right_idx = findTokenById(ids, output.items[i + 1]).?;

            const left_chars = chars[left_idx];
            const right_chars = chars[right_idx];

            const combined = try std.fmt.bufPrint(&buf, "{s}{s}", .{ left_chars, right_chars });
            //std.debug.print("Trying to merge <<{s}>>\n", .{combined});
            if (lookupIndex(chars, combined)) |index| {
                if (scores[index] > best) {
                    best = scores[index];
                    best_merge = self.tokens.get(index);
                }
            }

            if (best_merge) |have_better| {
                // zig fmt: off
                //const left_id = output.items[i];
                //const right_id = output.items[i + 1];
                //std.debug.print("Merged tokens (id={d}) <<{s}>> and (id={d}) <<{s}>> into token {d} <<{s}>>\n",
                //                .{ left_id, left_chars, right_id, right_chars, have_better.id, have_better.chars });
                // zig fmt: on
                _ = output.orderedRemove(i);
                output.items[i] = have_better.id;

                // Don't increment `i` and try to merge again.
            }
            //} else {
            i += 1;
            //}
        }

        // Shrink output to final size
        return output.toOwnedSlice();
    }

    const Searcher = struct {
        fn compare(ctx: []const u8, item: []const u8) std.math.Order {
            return std.mem.order(u8, ctx, item);
        }
    };

    /// Find the index for a needle in the haystack.
    /// Returns an index if the exact match is found and `null` otherwise.
    fn lookupIndex(haystack: [][]u8, needle: []const u8) ?usize {
        //if (std.sort.binarySearch([]u8, self.tokens.field(.chars), str, Searcher.compare)) |index| {
        if (std.sort.binarySearch([]u8, haystack, needle, Searcher.compare)) |index| {
            return index;
        } else {
            return null;
        }
    }

    /// Find a token by its token id.
    fn findTokenById(ids: []Token, token: Token) ?usize {
        for (0.., ids) |i, id| {
            if (id == token) {
                return i;
            }
        }

        return null;
    }
};

fn load_tokenizer(tokenizer_file: []u8, alloc: Allocator, vocab_size: usize) !Tokenizer {
    // open the file with fallback, then hand off to the tokenizer
    const cwd = std.fs.cwd();
    const tokenizer = try cwd.openFile(tokenizer_file, .{ .mode = .read_only });
    defer tokenizer.close();

    var buffer = std.io.bufferedReader(tokenizer.reader());

    return try Tokenizer.read(buffer.reader(), alloc, vocab_size);
}

test "Tokenizer.encode" {
    const tok_path = try std.testing.allocator.dupe(u8, "tokenizer.bin");
    defer std.testing.allocator.free(tok_path);
    var tokenizer = try load_tokenizer(tok_path, std.testing.allocator, 32000);
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

/// Struct which contains the weights for the transformer.
/// This is loaded from a V1 export from `llama2.c`.
///
/// This struct is for calculation purposes immutable and represents
/// the contents of the loaded model, backed by `mmap(2)`.
const TransformerV1 = struct {
    const Self = @This();
    const page_size_min = std.heap.page_size_min;
    const header_len = 256;

    // Best we can guess is the alignment is `4` based on the file export.
    const Weights = []align(4) f32;

    fd: std.posix.fd_t,
    ptr: ?[]align(page_size_min) u8,

    // Embeddings
    token_embed: Weights,
    // Attention
    rms_attention: Weights,
    // Attention Layers
    w_q: Weights,
    w_k: Weights,
    w_v: Weights,
    w_o: Weights,
    // Feed forward
    feed_forward_norm: Weights,
    // Unknown
    w1: Weights,
    w2: Weights,
    w3: Weights,
    // Output norms
    norm: Weights,
    // Only set whenever no shared classifier layer with tokenizer
    classifier: Weights,

    /// Initialize the Transformer weight pointers with the provided file.
    fn init(model_path: []u8, config: Config) !TransformerV1 {
        std.debug.print("Opening Transformer model\n", .{});
        const fd = try std.posix.open(model_path, .{}, 0o440);
        errdefer std.posix.close(fd);

        // Here we mmap() the weights files because nobody wants to open up a 25 GB file raw!
        const stat = try std.posix.fstat(fd);
        // zig fmt: off
        std.debug.print(
            "Model size: {d:.1} MiB\n",
            .{ @as(f32, @floatFromInt(stat.size)) / 1048576.0 }
        );
        // zig fmt: on

        const fsize: u64 = @intCast(stat.size);

        // See `std.os.linux.MAP` for more info.
        //std.os.linux.MAP
        const mmap_type: std.posix.MAP = .{
            .TYPE = .PRIVATE,
        };

        const ptr = try std.posix.mmap(null, fsize, std.posix.PROT.READ, mmap_type, fd, 0);
        errdefer std.posix.munmap(ptr);

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
        const layers = config.n_layers;
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
        const rms_attention: Weights = weights[i .. i + layers * dim];
        i += rms_attention.len;
        const ffn_norm: Weights = weights[i .. i + layers * dim];
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

        const wq: Weights = weights[i .. i + layers * dim * (n_heads * head_size)];
        i += wq.len;
        const wk: Weights = weights[i .. i + layers * dim * (n_kv_heads * head_size)];
        i += wq.len;
        const wv: Weights = weights[i .. i + layers * dim * (n_kv_heads * head_size)];
        i += wv.len;
        const wo: Weights = weights[i .. i + layers * (n_heads * head_size) * dim];
        i += wo.len;

        // ff layers (7b)
        // w1:          [11008][4096]f32
        // w2:          [4096][11008]f32
        // w3:          [11008][4096]f32
        // output:      [vocab_size][4096]f32
        const w1: Weights = weights[i .. i + layers * dim * hidden_dim];
        i += w1.len;
        const w2: Weights = weights[i .. i + layers * hidden_dim * dim];
        i += w2.len;
        const w3: Weights = weights[i .. i + layers * dim * hidden_dim];
        i += w3.len;

        var out_classifier: ?Weights = null;
        if (config.shared_classifier) {
            out_classifier = token_embed;
        } else {
            out_classifier = weights[i .. i + dim * vocab];
            i += out_classifier.?.len;
        }

        const used_size = i * @sizeOf(f32) + header_len;

        std.debug.print("weights size: {d}, used: {d}\n", .{ fsize, used_size });
        std.debug.assert(fsize == used_size); // make sure we read the whole file.

        return TransformerV1{
            .fd = fd,
            .ptr = ptr,

            .rms_attention = rms_attention,
            .token_embed = token_embed,
            .feed_forward_norm = ffn_norm,
            .norm = norms,

            .w_q = wq,
            .w_k = wk,
            .w_v = wv,
            .w_o = wo,

            .w1 = w1,
            .w2 = w2,
            .w3 = w3,

            .classifier = out_classifier.?,
        };
    }

    fn deinit(self: *Self) void {
        std.debug.print("Transformer.deinit()\n", .{});
        std.posix.munmap(self.ptr.?);
        self.ptr = null;
        self.fd = -1;
    }
};
