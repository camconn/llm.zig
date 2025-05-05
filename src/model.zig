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
const token = llm.token;

pub const gpt2 = @import("./model/gpt2.zig");
pub const llama = @import("./model/llama.zig");

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
pub const LoadError = ModelError || ggml.Error || token.TokenizerError || std.mem.Allocator.Error
    || std.fs.File.OpenError || std.posix.MMapError || std.posix.OpenError || std.posix.FStatError
    || std.posix.MadviseError || error{ EndOfStream, InvalidValue};
// zig fmt: on

/// Signature a model must implement for support dynamic loading.
pub const loader_fn = fn (file: ggml.GGUFFile, allocator: std.mem.Allocator) LoadError!*anyopaque;

/// An entry in the `supported_models` table.
pub const Entry = struct { Architecture, *const loader_fn };

/// Supported models for dynamic loading.
pub const supported_models = [_]Entry{
    .{ .gpt2, gpt2.Gpt2Context.initGeneric },
    .{ .llama, llama.LlamaContext.initGeneric },
};

/// Represents a loaded model in memory.
pub const Model = struct {
    const Self = @This();
    c: Context,

    /// Attempt to load a supported model from `file_path`.
    /// This function will attempt to load models one-by-one from the `supported_models` table
    /// until it finds a model that matches the provided metadata string in the `ggml.arch_key`
    /// value.
    /// If you wish to add support for a new model, this is the place.
    pub fn init(file_path: []const u8, alloc: std.mem.Allocator) LoadError!Model {
        var file = try ggml.GGUFFile.read_file(file_path, alloc);
        errdefer file.deinit();
        const model_info = try detectModelType(file);

        var context: Context = .none;

        // Initialize the Context union based on the comptime name of the model.
        // Iterate the fields of the `Context` struct to find the one matching the
        // comptime-known name.
        inline for (comptime std.meta.fields(Context)) |field| {
            // `field` is a `std.builtin.Type.UnionField`

            // Don't touch the `none` sentinel variant. This prevents a compile error.
            if (comptime std.mem.eql(u8, field.name, "none")) continue;

            // If we find a name matching the value of our Context field, try to setup.
            if (std.mem.eql(u8, @tagName(model_info[0]), field.name)) {
                context = @unionInit(
                    Context,
                    field.name,
                    @ptrCast(@alignCast(try @call(.never_inline, model_info[1], .{ file, alloc }))),
                );
                break;
            }
        }

        if (context == .none) {
            std.debug.print("Failed to select or load model. This is likely an `llm.zig` bug\n", .{});
            return ModelError.Other;
        }

        return .{
            .c = context,
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

    /// Helper method for identifying the currently loaded Model architecture.
    pub fn arch(self: Self) Architecture {
        return @as(Architecture, self.c);
    }

    /// Free up any resources currently loaded or allocated for this Model.
    pub fn deinit(self: *Self) void {
        switch (self.c) {
            .none => std.debug.print("deinit() none!\n", .{}),
            .llama => |c| c.deinit(),
            .gpt2 => |c| c.deinit(),
        }
    }
};

/// Supported `Model` architectures
pub const Architecture = enum {
    /// No architecture, the architecture is undefined.
    none,
    /// GPT-2
    gpt2,
    /// LLaMA 1 and LLaMA 2.
    llama,
};

/// Inner union type containing `Model` context.
const Context = union(Architecture) {
    /// Used for uninitialized unions
    none,
    gpt2: *gpt2.Gpt2Context,
    llama: *llama.LlamaContext,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    std.debug.print("Loading model\n", .{});

    const fpath = "llama2-7b-f32.gguf";
    var model = try Model.init(fpath, alloc);
    defer model.deinit();
    std.debug.print("Done loading model\n", .{});
}
