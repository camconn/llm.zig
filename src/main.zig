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
        const prompt_example = "Hello\nworld! How are you today?";
        const prompt_dupe = try alloc.dupe(u8, prompt_example);
        defer alloc.free(prompt_dupe);

        const tokens = try tokenizer.encode(prompt_dupe, alloc);
        defer alloc.free(tokens);

        std.debug.print("Got {d} encoded tokens\n", .{tokens.len});
        for (0.., tokens) |i, tok| {
            std.debug.print("Token #{d} = {d}\n", .{ i, tok });
        }
    }

    var state = try State.init(alloc, config);
    defer state.deinit();

    // " Hello" = 15043
    _ = transformer.forward(&state, 15043, 0);

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
/// Do not leave the file open because we will mmap it.
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
    idx_to_token: []usize,

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
            // Some

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
            .idx_to_token = token_to_idx,
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
    fn encode(self: Self, text: []const u8, alloc: Allocator) ![]Token {
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
        const space = lookupIndex(chars, SPACE_PIECE).?;
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
                //const left_idx = findTokenById(ids, node.?.data).?;
                //const right_idx = findTokenById(ids, next.data).?;
                const left_idx = self.findTokenById(node.?.data).?;
                const right_idx = self.findTokenById(next.data).?;

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
        //if (std.sort.binarySearch([]u8, self.tokens.field(.chars), str, Searcher.compare)) |index| {
        if (std.sort.binarySearch([]u8, haystack, needle, Searcher.compare)) |index| {
            return index;
        } else {
            return null;
        }
    }

    /// Find a token by its index in `tokens`.
    fn findTokenById(self: Self, token: Token) ?usize {
        return self.idx_to_token[@intCast(token)];
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

/// Represents the current state of a transformer.
const State = struct {
    arena: ArenaAllocator,

    // x
    input: []f32,
    // xb
    residual: []f32,
    // xb2
    tmp: []f32,
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

    k_cache: []f32,
    v_cache: []f32,

    fn init(alloc: Allocator, config: Config) !State {
        var arena = ArenaAllocator.init(alloc);
        var a = arena.allocator();

        const kv_dim = config.dim * config.n_kv_heads / config.n_heads;

        const x = try a.alloc(f32, config.dim);
        const xb = try a.alloc(f32, config.dim);
        const xb2 = try a.alloc(f32, config.dim);
        const hb1 = try a.alloc(f32, config.hidden_dim);
        const hb2 = try a.alloc(f32, config.hidden_dim);

        const q = try a.alloc(f32, config.dim);
        const k = try a.alloc(f32, config.n_layers * config.dim * kv_dim);
        const v = try a.alloc(f32, config.n_layers * config.dim * kv_dim);

        const k_cache = try a.alloc(f32, config.n_layers * config.max_seq_length * kv_dim);
        const v_cache = try a.alloc(f32, config.n_layers * config.max_seq_length * kv_dim);

        const att = try a.alloc(f32, config.n_heads * config.max_seq_length);
        const logits = try a.alloc(f32, config.vocab_size);

        return .{
            .arena = arena,
            // layers and temporaries
            .input = x,
            .residual = xb,
            .tmp = xb2,
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

    fn deinit(self: @This()) void {
        self.arena.deinit();
    }
};

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

    // Config weights
    config: Config,

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

    const vector_len = std.simd.suggestVectorLength(f32) orelse 8;
    const Vec = @Vector(vector_len, f32);

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
            .config = config,
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

    /// Calculate a forward pass of the transformer with the next token `token` at
    /// position `n_token`.
    fn forward(self: Self, state: *State, token: Tokenizer.Token, n_token: usize) []f32 {
        const c = self.config;

        const token_offset: usize = @as(usize, @intCast(token)) * c.dim;
        const embeddings = self.token_embed[token_offset .. token_offset + c.dim];
        @memcpy(state.input[0..], embeddings);

        const kv_dim = c.dim * c.n_kv_heads / c.n_heads;

        for (0..c.n_layers) |i| {
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
            //    output = wo(output)
            //    return wo(output)

            // TransformerBlock.forward()
            // first handle the `attention_norm` call.
            const rms_offset = i * c.dim;
            rmsNorm(state.residual, state.input, self.rms_attention[rms_offset .. rms_offset + c.dim]);

            // TODO: Don't know if anything below this is correct.

            // Now handle attention matrix multiplies
            // xq = wq(x)
            const wq = self.w_q[i * c.dim * c.dim .. (i + 1) * c.dim * c.dim][0 .. c.dim * c.dim];
            matrixMul(state.q, wq, state.residual, c.dim, c.dim);

            // xk = wk(x)
            const kv_offset = i * c.dim * kv_dim;
            const kv_end = kv_offset + c.dim;
            const wk = self.w_k[i * c.dim * kv_dim .. (i + 1) * c.dim * kv_dim][0 .. c.dim * kv_dim];
            matrixMul(state.k[kv_offset..kv_end], wk, state.residual, c.dim, kv_dim);

            // xv = wv(x)
            const wv = self.w_v[i * c.dim * c.dim .. (i + 1) * c.dim * kv_dim][0 .. c.dim * kv_dim];
            matrixMul(state.v[kv_offset..kv_end], wv, state.residual, c.dim, kv_dim);

            std.debug.print("TODO: Handle layer {d} with token {d} and n_token {d}\n", .{ i, token, n_token });
        }

        // After the layers the hidden layer is normalized

        // Then logits are found

        return state.output;
    }

    /// Perform RMS Normalization on `x` with the weights `y` and store the result in `out`.
    /// Requires all inputs to have the same length.
    ///
    /// This method mirrors the implementation of `RMSNorm` in Meta's `model.py`.
    /// It implements the method described in [1], plus adds a small epsilon factor for numeric
    /// stability.
    /// [1]: https://arxiv.org/abs/1910.07467
    fn rmsNorm(out: []f32, x: []const f32, y: []const f32) void {
        if (!(x.len == y.len and y.len == out.len)) {
            std.debug.print("lengths: out={d}, x={d}, y={d}\n", .{ out.len, x.len, y.len });
            std.debug.assert(out.len == x.len);
            std.debug.assert(x.len == y.len);
            @panic("Mismatched lengths");
        }

        const chunks = x.len / vector_len;
        const leftover_offset = chunks * vector_len;

        var sum: f32 = 0;
        for (0..chunks) |i| {
            const idx = i * vector_len;
            const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
            const square = xs * xs;
            const chunk_sum = @reduce(.Add, square);
            sum += chunk_sum;
        }

        for (leftover_offset..x.len) |i| {
            const xs = x[i];
            const square = xs * xs;
            sum += square;
        }

        const epsilon = 1e-5;
        const x_f32: f32 = @floatFromInt(x.len);
        const mean = (sum + epsilon) / x_f32;
        const root = std.math.sqrt(mean);

        // Now perform division + multiply by weights.
        const divisor: Vec = @splat(root);
        for (0..chunks) |i| {
            const idx = i * vector_len;
            const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
            const ys: Vec = y[idx .. idx + vector_len][0..vector_len].*;

            const result = xs * ys / divisor;
            // The Zig compiler lets you got slice → vector, but doesn't like the following line:
            //out[idx .. idx + vector_len][0..vector_len] = result;
            out[idx .. idx + vector_len][0..vector_len].* = result;
            //@memcpy(out[idx .. idx + vector_len], result);
        }

        for (leftover_offset..x.len) |i| {
            const xs = x[i];
            const ys = y[i];

            const result = xs * ys / root;
            out[i] = result;
        }
    }

    /// Multiply a matrix `m` of (`rows`, `cols`) by a vector `x` of (`cols`) and store in `out`.
    fn matrixMul(out: []f32, m: []const f32, x: []const f32, rows: usize, cols: usize) void {
        std.debug.print("out: {d}, m: {d}, x: {d}, rows: {d}, cols: {d}\n", .{ out.len, m.len, x.len, rows, cols });
        std.debug.assert(out.len == rows);
        std.debug.assert(x.len == cols);
        std.debug.assert(m.len == rows * cols);

        const chunks = x.len / vector_len;
        const leftover_offset = chunks * vector_len;

        //var sum: usize = 0;
        for (0..rows) |row| {
            const m_off = row * cols;

            var sum: f32 = 0;
            for (0..chunks) |chunk| {
                const idx = chunk * vector_len;
                const m_idx = m_off + idx;
                const xs: Vec = x[idx .. idx + vector_len][0..vector_len].*;
                const ms: Vec = m[m_idx .. m_idx + vector_len][0..vector_len].*;

                const prod = xs * ms;
                const chunk_sum = @reduce(.Add, prod);
                sum += chunk_sum;
            }

            for (leftover_offset..cols) |i| {
                const xs = x[i];
                const ms = m[m_off + i];
                sum += xs * ms;
            }

            out[row] = sum;
        }
    }
};

const Params = struct {
    temperature: f32,
    top_p: f32,
    random: std.Random,

    fn init(temperature: f32, top_p: f32) Params {
        const now = std.time.milliTimestamp();
        const rng = std.Random.Xoroshiro128.init(@bitCast(now));
        return .{
            .temperature = temperature,
            .top_p = top_p,
            .random = rng,
        };
    }
};
