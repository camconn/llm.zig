// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! model: Model loading and runtime.
//! This module provides support for selection, loading, and running supported model architectures
//! in `llm.zig`. The main structure is the `Model`

const std = @import("std");

const llm = @import("root.zig");
const ggml = llm.ggml;
const tkn = llm.token;

pub const gpt2 = @import("model/gpt2.zig");
pub const llama = @import("model/llama.zig");

/// Possible error that may occur when loading a `Model`.
pub const ModelError = error{
    /// You attempted to load a file that has an invalid format. Usually this means the file had
    /// a bad header or the wrong version. This could also mean that the file is truncated or is
    /// corrupted. It could also mean there's trailing data at the end.
    BadFormat,
    /// You tried to load a file that has a correct format but is semantically incorrect. Usually
    /// this means you tried to load the wrong model.
    BadFile,
    /// Attempted to load a model with an invalid vocabulary.
    BadVocab,
    /// An unknown error occurred when loading the file.
    Other,
};

// zig fmt: off
/// Error union describing what can go wrong when loading a model.
pub const LoadError = ModelError || ggml.Error || tkn.TokenizerError || std.mem.Allocator.Error
    || std.fs.File.OpenError || std.posix.MMapError || std.posix.OpenError || std.posix.FStatError
    || std.posix.MadviseError || error{ EndOfStream, InvalidValue};
// zig fmt: on

// zig fmt: off
/// Error union describing what can go wrong when loading a model.
pub const RunError = ModelError || ggml.Error || tkn.TokenizerError || std.mem.Allocator.Error
    || error{ EndOfStream, InvalidValue, NoSpaceLeft };
// zig fmt: on

/// Supported `Model` architectures
pub const Architecture = enum {
    /// GPT-2
    gpt2,
    /// LLaMA 1 and LLaMA 2.
    llama,
};

/// An entry in the `supported_models` table.
/// First is a unique `Architecture` variant, which is different for every model architecture.
/// The next element is the `VTable` entry which tells us how to call the models entry points
/// (loading, tokenization, evaluation, de-initialization, etc.) dynamically.
pub const Entry = struct { Architecture, *const VTable };

/// Supported models for dynamic loading.
pub const supported_models = [_]Entry{
    .{ .gpt2, &gpt2.Gpt2Context.vtable },
    .{ .llama, &llama.LlamaContext.vtable },
};

/// Represents a loaded model in memory.
/// A `Model` is just a helper struct which performs calls on the defined `VTable` which each
/// supported model `Architecture` implements.
pub const Model = struct {
    const Self = @This();

    alloc: std.mem.Allocator,
    arch: Architecture,

    ptr: ?*anyopaque,
    vtable: *const VTable,

    /// Attempt to load a supported model from `file_path`.
    /// This function will attempt to load models one-by-one from the `supported_models` table
    /// until it finds a model that matches the provided metadata string in the `ggml.arch_key`
    /// value.
    /// If you wish to add support for a new model, this is the place.
    pub fn init(file_path: []const u8, allocator: std.mem.Allocator) LoadError!Model {
        var file = try ggml.GGUFFile.read_file(file_path, allocator);
        errdefer file.deinit();
        const model_info = try detectModelType(file);

        var arch: ?Architecture = null;
        var ptr: ?*anyopaque = null;
        var vtable: ?*const VTable = null;

        // Initialize the pointer based on the comptime name of the model.
        // Iterate the fields of the `Architecture` enum to find the one matching the
        // comptime-known name.
        inline for (comptime std.meta.fields(Architecture)) |field| {
            // `field` is a `std.builtin.Type.EnumField`

            // If we find a name matching the value of our enum, try to setup.
            if (std.mem.eql(u8, @tagName(model_info[0]), field.name)) {
                ptr = @ptrCast(@alignCast(try @call(.never_inline, model_info[1].init, .{ file, allocator })));
                vtable = model_info[1];
                arch = model_info[0];
                break;
            }
        }

        if (arch == null) {
            std.debug.print("Failed to select or load model. This is likely an `llm.zig` bug\n", .{});
            return ModelError.Other;
        }

        return .{
            .alloc = allocator,
            .ptr = ptr,
            .vtable = vtable.?, // impossible to fail, we set this before `arch` so it is non-null,
            .arch = arch.?, // already checked this is non-null
        };
    }

    fn detectModelType(file: ggml.GGUFFile) !Entry {
        if (file.getValueT(ggml.arch_key, .string)) |architecture| {
            const actual = architecture.string.str;
            for (supported_models) |entry| {
                const name = @tagName(entry[0]);
                if (std.ascii.eqlIgnoreCase(name, actual)) {
                    return entry;
                }
            }
            return ModelError.BadFile;
        } else {
            std.debug.print("model: File is missing \"{s}\" metadata string", .{ggml.arch_key});
            return ModelError.BadFormat;
        }
    }

    /// Perform forward-pass inference for a single `token` at position `n_token`.
    /// Returns a list of token probabilities
    pub fn forward(self: *Self, token: tkn.Token, n_token: usize) []f32 {
        return self.vtable.forward(self.ptr.?, token, n_token);
    }

    /// Convert a string into a slice of tokens owned by `allocator`.
    /// Use `add_start` to add a start-of-text token.
    /// Caller is responsible for freeing any allocated tokens.
    pub fn tokenize(self: *Self, str: []const u8, option: tkn.EncodingOption, allocator: std.mem.Allocator) RunError![]const tkn.Token {
        return self.vtable.tokenize(self.ptr.?, str, option, allocator);
    }

    /// Convert a slice of tokens into a string owned by `allocator`.
    /// Caller is responsible for freeing any returned string.
    pub fn detokenize(self: *Self, tokens: []const tkn.Token, allocator: std.mem.Allocator) RunError![]u8 {
        return self.vtable.detokenize(self.ptr.?, tokens, allocator);
    }

    /// Get the string slice corresponding to a single `token`, if it exists. Otherwise return null.
    /// Useful for incremental printing of processed output.
    pub fn toString(self: *Self, token: tkn.Token) ?[]const u8 {
        return self.vtable.to_string(self.ptr.?, token);
    }

    /// Return the size of this model's vocabulary.
    pub fn vocabSize(self: *Self) usize {
        return self.vtable.vocab_size(self.ptr.?);
    }

    /// Return the size of the model's context length.
    pub fn contextLength(self: *Self) usize {
        return self.vtable.context_len(self.ptr.?);
    }

    /// Free up any resources currently loaded or allocated for this Model.
    pub fn deinit(self: *Self) void {
        if (self.ptr) |ptr| {
            self.vtable.deinit(ptr);
            self.ptr = null;
        }
    }
};

/// Function call table for model operations.
/// Models must implement these methods to support dynamic loading with a `Model`.
pub const VTable = struct {
    /// Attempt to load a model from a `file` with the provided `allocator`.
    /// The returned `anyopaque` should be to a model context created with `Allocator.create`.
    /// The opaque pointer will eventally be destroyed with `Allocator.destroy`.
    ///
    /// If there is an error before this calls succeeds, `file` belongs to the caller.
    /// If the function succeeds then the callee takes ownership of `file`.
    init: *const fn (file: ggml.GGUFFile, allocator: std.mem.Allocator) LoadError!*anyopaque,

    /// Convert a string into a slice of tokens owned by `allocator`.
    /// Caller is responsible for freeing any allocated tokens.
    tokenize: *const fn (*anyopaque, str: []const u8, option: tkn.EncodingOption, allocator: std.mem.Allocator) RunError![]const tkn.Token,

    /// Convert a slice of tokens into a string owned by `allocator`.
    /// Caller is responsible for freeing any returned string.
    detokenize: *const fn (*anyopaque, tokens: []const tkn.Token, allocator: std.mem.Allocator) RunError![]u8,

    /// Get the string slice corresponding to a single `token`, if it exists. Otherwise return null.
    /// Useful for incremental printing of processed output.
    to_string: *const fn (*anyopaque, token: tkn.Token) ?[]const u8,

    /// Perform forward-pass inference for a single `token` at position `n_token`.
    /// Returns a list of token probabilities
    forward: *const fn (*anyopaque, token: tkn.Token, n_token: usize) []f32,

    /// Return the vocabulary size of the model's currently loaded tokenizer.
    vocab_size: *const fn (*anyopaque) usize,

    /// Return the size of the context window.
    context_len: *const fn (*anyopaque) usize,

    /// De-initialize the model context and any associated resources.
    /// Model implementations are responsible for calling `allocator.destroy(self)` during this function.
    deinit: *const fn (*anyopaque) void,
};

/// Metadata about a Model
pub const Info = struct {
    vocab_size: usize,
    context_len: usize,
    start_token: ?tkn.Token,
    end_token: ?tkn.Token,
};
