// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Cameron Conn

//! LLM.zig: A library for LLM functions.
//! Functionality for loading a model and running inference are located in the sub-modules listed
//! below.

pub const ggml = @import("ggml.zig");
pub const math = @import("math.zig");
pub const model = @import("model.zig");
pub const regex = @import("regex.zig");
pub const sample = @import("sample.zig");
pub const token = @import("token.zig");
pub const unicode = @import("unicode.zig");

test "Top level test reference" {
    const std = @import("std");
    std.testing.refAllDeclsRecursive(@This());
}
