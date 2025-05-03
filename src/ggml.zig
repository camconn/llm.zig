// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! ggml: This module contains methods and structs to parse the `.gguf` file format
//! from the [GGML](https://github.com/ggml-org/ggml/tree/master) library.

const std = @import("std");

const llm = @import("root.zig");
const math = llm.math;

const Alloc = std.mem.Allocator;
const Arena = std.heap.ArenaAllocator;

const page_size = std.heap.page_size_min;
const Reader = std.io.FixedBufferStream([]align(page_size) u8).Reader;

/// Custom error type to help the Zig compiler infer the error union from recursive
/// `try foo()` calls.
const Error = error{
    FileError,
    Alloc,
    Format,
    EOF,
    Unknown,
};

// ggml_type
pub const Type = enum(u32) {
    // zig fmt: off
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Used, but not supported according to GGUF docs
    //Q4_2    = 4,
    //Q4_3    = 5,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    // zig fmt: on
};

// gguf_metadata_value_type
pub const MetadataType = enum(u32) {
    // zig fmt: off
    uint8   = 0,
    int8    = 1,
    uint16  = 2,
    int16   = 3,
    uint32  = 4,
    int32   = 5,
    float32 = 6,
    /// 0 and 1 are valid. All other values mean the model or reader is buggy
    boolean = 7,
    /// UTF-8 non-null terminated
    string  = 8,
    /// Arrays can be nested. The "Length" of the array is the # of elems, not the size in bytes
    array   = 9,
    uint64  = 10,
    int64   = 11,
    float64 = 12,
    // zig fmt: on
};

// gguf_string_t
pub const String = struct {
    len: u64,
    str: []u8,

    fn read(reader: Reader, alloc: std.mem.Allocator) Error!String {
        const len = try (reader.readInt(u64, .little) catch Error.EOF);
        const buf = try (alloc.alloc(u8, len) catch Error.EOF);

        const n_read = reader.read(buf) catch 0;
        if (n_read == 0 or n_read != len) {
            return Error.FileError;
        }
        return .{ .len = len, .str = buf };
    }
};

// ggguf_metadata_value_t.array
pub const Array = struct {
    elem_type: MetadataType,
    // Number of elements, not the size in bytes
    len: u64,
    array: []Value,

    fn read(reader: Reader, alloc: std.mem.Allocator) Error!Array {
        const element_type = try (reader.readEnum(MetadataType, .little) catch Error.EOF);
        const len = try (reader.readInt(u64, .little) catch Error.EOF);

        var list = std.ArrayList(Value).init(alloc);
        errdefer list.deinit();

        for (0..len) |_| {
            const val = try Value.read(element_type, reader, alloc);
            try (list.append(val) catch Error.Alloc);
        }

        return .{
            .elem_type = element_type,
            .len = len,
            .array = try (list.toOwnedSlice() catch Error.Alloc),
        };
    }
};

// gguf_metadata_value_t
pub const Value = union(MetadataType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    boolean: bool,

    string: String,
    array: Array,

    uint64: u64,
    int64: i64,
    float64: f64,

    fn read(value_type: MetadataType, reader: Reader, alloc: std.mem.Allocator) Error!Value {
        // zig fmt: off
        return switch (value_type) {
            // Integer types
            .uint8   => Value{ .uint8   = try (reader.readInt(u8, .little) catch Error.EOF) },
            .int8    => Value{ .int8    = try (reader.readInt(i8, .little) catch Error.EOF) },
            .uint16  => Value{ .uint16  = try (reader.readInt(u16, .little) catch Error.EOF) },
            .int16   => Value{ .int16   = try (reader.readInt(i16, .little) catch Error.EOF) },
            .uint32  => Value{ .uint32  = try (reader.readInt(u32, .little) catch Error.EOF) },
            .int32   => Value{ .int32   = try (reader.readInt(i32, .little) catch Error.EOF) },
            .uint64  => Value{ .uint64  = try (reader.readInt(u64, .little) catch Error.EOF) },
            .int64   => Value{ .int64   = try (reader.readInt(i64, .little) catch Error.EOF) },
            .boolean => Value{ .boolean = try (reader.readInt(u8, .little) catch Error.EOF) == 1 },
            // Floating point
            .float32 => Value{ .float32 = @bitCast(try (reader.readInt(u32, .little) catch Error.EOF)) },
            .float64 => Value{ .float64 = @bitCast(try (reader.readInt(u64, .little) catch Error.EOF)) },
            // Complex types
            .string  => Value { .string = try String.read(reader, alloc) },
            .array   => Value { .array  = try Array.read(reader, alloc) },
        };
        // zig fmt: on
    }
};

// gguf_metadata_kv_t
/// A entry for metadata within a model.
///
/// The `value_type` field is omitted because that info can be obtained by coercing the `Value`
/// union to a `MetadataType` enum.
pub const MetadataKV = struct {
    /// The key for the metadata entry.
    key: String,
    /// The actual value for this entry.
    value: Value,

    fn read(reader: Reader, alloc: std.mem.Allocator) !MetadataKV {
        const key = try String.read(reader, alloc);
        const value_type = try reader.readEnum(MetadataType, .little);
        const value = try Value.read(value_type, reader, alloc);
        return .{
            .key = key,
            .value = value,
        };
    }
};

// gguf_header_t
pub const GGUFHeader = struct {
    const magic_str = "GGUF"; // 0x47, 0x47, 0x55, 0x46
    const magic_num = std.mem.readInt(u32, magic_str, .little);

    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    // metadata_kv is located in GGUFFile.

    pub fn read(reader: Reader) !GGUFHeader {
        const magic = try reader.readInt(u32, .little);
        if (magic != magic_num) {
            return Error.Format;
        }

        const version = try reader.readInt(u32, .little);
        if (version != 3) {
            return Error.Format;
        }

        const tensor_count = try reader.readInt(u64, .little);
        const metadata_kv_count = try reader.readInt(u64, .little);

        return .{
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
        };
    }
};

/// GGUF File Type. Located in the `general.file_type` metadata value.
pub const FileType = enum(u32) {
    // zig fmt: off
    ALL_F32              = 0,
    MOSTLY_F16           = 1,
    MOSTLY_Q4_0          = 2,
    MOSTLY_Q4_1          = 3,
    MOSTLY_Q4_1_SOME_F16 = 4,
    MOSTLY_Q4_2          = 5, // support removed
    MOSTLY_Q4_3          = 6, // support removed
    MOSTLY_Q8_0          = 7,
    MOSTLY_Q5_0          = 8,
    MOSTLY_Q5_1          = 9,
    MOSTLY_Q2_K          = 10,
    MOSTLY_Q3_K_S        = 11,
    MOSTLY_Q3_K_M        = 12,
    MOSTLY_Q3_K_L        = 13,
    MOSTLY_Q4_K_S        = 14,
    MOSTLY_Q4_K_M        = 15,
    MOSTLY_Q5_K_S        = 16,
    MOSTLY_Q5_K_M        = 17,
    MOSTLY_Q6_K          = 18,
    /// Sentinel value used to check range, not actually in `ggml.h`
    NUM_FILETYPE         = 19,
    // zig fmt: on
};
const max_file_type = @intFromEnum(FileType.NUM_FILETYPE);

// gguf_tensor_info_t
pub const TensorInfo = struct {
    const Self = @This();

    /// The name of this the tensor.
    name: String,
    // omit `dim` because that's implicitly stored in the `dimensions` slice as its length
    /// The dimensions of the tensor. The layout of the data within the dimensions varies
    /// based on the actual model.
    ///
    /// For Llama (2), the dimensions are in `(x, y)` order where `x` is the number of columns and
    /// `y` is the number of `rows`. Llama (2) weights are in row-major order, so the index of
    /// an element at position `(a, b)` within a flattened tensor is at `b*len(row) + a`.
    dimensions: []u64,
    /// The type of Tensor.
    ggml_type: Type,
    /// The relative offset of the tensor within the file, in bytes.
    /// This is a u64, but is represented as a usize to prevent a bunch of `@intCast` calls
    /// within model code.
    offset: usize,

    inline fn alignOffset(offset: usize, alignment: usize) usize {
        return offset + (alignment - (offset % alignment)) % alignment;
    }

    fn read(reader: Reader, alignment: usize, alloc: std.mem.Allocator) Error!TensorInfo {
        const name = try (String.read(reader, alloc) catch Error.EOF);
        const dim = try (reader.readInt(u32, .little) catch Error.EOF);
        std.debug.assert(dim > 0);
        const dimensions = try (alloc.alloc(u64, dim) catch Error.EOF);
        for (0..dim) |i| {
            const n = try (reader.readInt(u64, .little) catch Error.EOF);
            dimensions[i] = n;
        }
        const ggml_type = try (reader.readEnum(Type, .little) catch Error.EOF);
        const offset: usize = @intCast(try (reader.readInt(u64, .little) catch Error.EOF));

        const aligned = alignOffset(offset, alignment);
        if (offset != aligned) {
            std.debug.print("Unaligned: {s} offset {d} align {d} alignOffset {d}", .{
                name.str,
                offset,
                alignment,
                aligned,
            });
            return Error.FileError;
        }

        const rem = offset % alignment;
        if (rem != 0) {
            std.debug.print("Tensor {s} is not aligned: offset {d} alignment {d} has remainder {d}\n", .{
                name.str,
                offset,
                alignment,
                rem,
            });
            return Error.FileError;
        }

        return .{
            .name = name,
            .dimensions = dimensions,
            .ggml_type = ggml_type,
            .offset = offset,
        };
    }

    /// Get a raw slice of the Tensor's data in native model order.
    /// Get the elements of this tensor from the files `mmap(2)` pointer.
    pub fn getElems(self: Self, tensor_data: []const u8) math.Weights {
        const tensor = tensor_data[self.offset..];
        const target = @intFromPtr(&tensor[0]);

        var len: usize = 1;
        for (self.dimensions) |dim| {
            len *= @intCast(dim);
        }
        std.debug.assert(len >= 1);

        switch (self.ggml_type) {
            .F32 => {
                const ptr: [*]math.Block(.f32) = @ptrFromInt(target);
                return .{ .f32 = ptr[0..len] };
            },
            .F16 => {
                const ptr: [*]math.Block(.f16) = @ptrFromInt(target);
                return .{ .f16 = ptr[0..len] };
            },
            .Q8_0 => {
                const Block = comptime math.Block(.q8_0);
                const block_len = comptime math.blockUnitLen(Block);
                std.debug.assert(len % block_len == 0);
                const blocks = len / block_len;

                const ptr: [*]Block = @ptrFromInt(target);
                return .{ .q8_0 = ptr[0..blocks] };
            },
            else => {
                std.debug.print("ggml: Unsupported quantization, this is a library bug {}\n", .{self.ggml_type});
                @panic("Unsupported tensor quantization load");
            },
        }
    }
};

pub const alignment_key = "general.alignment";
pub const quant_version_key = "general.quantization_version";
pub const arch_key = "general.architecture";
pub const name_key = "general.name";
pub const file_type_key = "general.file_type";

// gguf_file_t
pub const GGUFFile = struct {
    // ========================================
    // gguf fields
    // ========================================
    header: GGUFHeader,
    metadata: []MetadataKV,
    // field is offset by GGUFHeader.tensor_count
    tensor_info: []TensorInfo,

    // padding
    // padding ends at sizeof(GGUFHeader) + sizeof(tensor_infos)
    /// How many bytes from the beginning of the file tensor data starts at.
    /// This is not an `mmap(2)` pointer, but the offset within the file, so you will manually
    /// need to add this to the `mmap(2)` ptr.
    tensor_data_offset: usize,
    //tensor_data: anyopaque,

    // ========================================
    // housekeeping fields
    // ========================================
    arena: Arena,

    fd: std.posix.fd_t,
    mmap_ptr: []align(page_size) u8,
    file_size: usize,

    const Self = @This();

    /// Open a GGUF file for reading.
    pub fn read_file(gguf_path: []const u8, alloc: std.mem.Allocator) !GGUFFile {
        std.debug.print("Opening GGUF file\n", .{});

        const fd = try std.posix.open(gguf_path, .{}, 0o440);
        errdefer std.posix.close(fd);

        // `mmap(2)` the weights file to try and save memory
        const stat = try std.posix.fstat(fd);
        const fsize: u64 = @intCast(stat.size);
        std.debug.print(
            "GGUF size: {d:.1} MiB\n",
            .{@as(f32, @floatFromInt(fsize)) / 1048576.0},
        );

        // TODO: Assert the file is sufficiently large enough for us

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

        // Get ready to parse the header and file metadata

        var stream = std.io.fixedBufferStream(ptr);
        const reader = stream.reader();

        var arena = Arena.init(alloc);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        // Actually parse the file.

        // Read header
        const header = try GGUFHeader.read(reader);
        // read metadata
        const metadata = try read_metadata(header.metadata_kv_count, reader, allocator);

        // Alignment must be a u32
        const alignment = if (getMetadataValue(alignment_key, metadata)) |val|
            switch (val) {
                .uint32 => val.uint32,
                else => {
                    std.debug.print("ggml: \"{s}\" must be a uint32\n", .{alignment_key});
                    return Error.Format;
                },
            }
        else
            32;
        // Read tensor info
        const tensor_info = try read_tensor_info(header.tensor_count, @intCast(alignment), reader, allocator);

        // Quantization version must be a u32
        const quantization = if (getMetadataValue(quant_version_key, metadata)) |val|
            switch (val) {
                .uint32 => val.uint32,
                else => {
                    std.debug.print("ggml: \"{s}\" must be a uint32\n", .{quant_version_key});
                    return Error.Format;
                },
            }
        else
            0;

        const arch = if (getMetadataValue(arch_key, metadata)) |arch|
            switch (arch) {
                .string => arch.string.str,
                else => {
                    std.debug.print("ggml: \"{s}\" must be a string\n", .{arch_key});
                    return Error.Format;
                },
            }
        else {
            std.debug.print("ggml: \"{s}\" is not present\n", .{arch_key});
            return Error.Format;
        };

        const name = if (getMetadataValue(name_key, metadata)) |name|
            switch (name) {
                .string => name.string.str,
                else => {
                    std.debug.print("ggml: \"{s}\" must be a string\n", .{name_key});
                    return Error.Format;
                },
            }
        else
            "unset";

        if (getMetadataValue(file_type_key, metadata)) |ft| {
            switch (ft) {
                .uint32 => {
                    const val = ft.uint32;
                    if (val >= max_file_type) {
                        std.debug.print("ggml: Invalid file type: read {d}, but max is {d}\n", .{ val, max_file_type });
                        return Error.Format;
                    }
                },
                else => {
                    std.debug.print("ggml: \"{s}\" must be a uint32\n", .{file_type_key});
                },
            }
        }

        std.debug.print("Model architecture: {s}\n", .{arch});
        std.debug.print("Model name: {s}\n", .{name});
        std.debug.print("Quantization version: {d}\n", .{quantization});

        const file_offset: usize = @intCast(try reader.context.getPos());
        const aligned = std.mem.alignForward(usize, file_offset, alignment);
        //std.debug.print("Aligning from {d} to {d}\n", .{ file_offset, aligned });

        const tensor_data_offset = aligned;
        //std.debug.print("Tensor_data starts at addr {d}, offset {d}\n", .{ tensor_data_offset, aligned });

        return .{
            // gguf fields
            .header = header,
            .metadata = metadata,
            .tensor_info = tensor_info,
            .tensor_data_offset = tensor_data_offset,

            // housekeeping fields
            .fd = fd,
            .mmap_ptr = ptr,
            .file_size = fsize,

            .arena = arena,
        };
    }

    fn read_metadata(
        metadata_count: u64,
        reader: Reader,
        alloc: std.mem.Allocator,
    ) ![]MetadataKV {
        var ret = std.ArrayList(MetadataKV).init(alloc);
        defer ret.deinit();
        try ret.ensureTotalCapacityPrecise(@intCast(metadata_count));

        for (0..metadata_count) |_| {
            const kv = try MetadataKV.read(reader, alloc);
            try ret.append(kv);
        }
        return ret.toOwnedSlice();
    }

    fn read_tensor_info(count: u64, alignment: usize, reader: Reader, alloc: std.mem.Allocator) ![]TensorInfo {
        var ret = std.ArrayList(TensorInfo).init(alloc);
        errdefer ret.deinit();
        try ret.ensureTotalCapacityPrecise(count);

        for (0..count) |_| {
            const tensor = try TensorInfo.read(reader, alignment, alloc);
            try ret.append(tensor);
        }
        return ret.toOwnedSlice();
    }

    fn getMetadataValue(key: []const u8, metadata: []MetadataKV) ?Value {
        for (metadata) |kv| {
            if (std.mem.eql(u8, key, kv.key.str)) {
                return kv.value;
            }
        }
        return null;
    }

    /// Get the Metadata value for a key if it is present or `null` otherwise.
    pub fn getValue(self: *const Self, key: []const u8) ?Value {
        return getMetadataValue(key, self.metadata);
    }

    /// Get the Metadata value for `key` if and only if it is present and of type `t`.
    /// Return `null` otherwise.
    pub fn getValueT(self: *const Self, key: []const u8, t: MetadataType) ?Value {
        if (self.getValue(key)) |inner| {
            const as_enum: MetadataType = inner;
            if (as_enum == t) {
                return inner;
            }
            std.debug.print("ggml: Mismatched type. Wanted {s} but got {s}\n", .{ @tagName(t), @tagName(as_enum) });
        }
        return null;
    }

    pub fn dumpMetadata(self: *const Self) void {
        for (self.metadata) |kv| {
            const value_type: MetadataType = kv.value;
            switch (value_type) {
                .string => std.debug.print("{s}={s}\n", .{ kv.key.str, kv.value.string.str }),
                .uint32 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.uint32 }),
                .uint64 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.uint64 }),
                .float32 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.float32 }),
                .float64 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.float64 }),
                .array => std.debug.print("{s}=array {d} elem, {}\n", .{ kv.key.str, kv.value.array.len, kv.value.array.elem_type }),
                else => std.debug.print("{s}={}\n", .{ kv.key.str, value_type }),
            }
        }
    }

    /// De-initialize this GGUF file and free up underlying resources.
    /// This invalidates any reference to memory of this GGUF file.
    pub fn deinit(self: *Self) void {
        _ = self.arena.reset(.free_all);
        self.arena.deinit();

        if (self.fd != -1) {
            std.posix.munmap(self.mmap_ptr);
            std.posix.close(self.fd);
        }
        self.fd = -1;
    }

    /// Retrieve the `TensorInfo` with `name` if it exists, or `null` if not.
    pub fn getTensorInfo(self: Self, name: []const u8) ?TensorInfo {
        for (self.tensor_info) |tensor| {
            if (std.mem.eql(u8, tensor.name.str, name)) {
                return tensor;
            }
        }
        return null;
    }

    /// Retrieve the `TensorWeights` for `name` if it exists or `null` otherwise.
    pub fn getTensorWeights(self: Self, name: []const u8) ?math.Weights {
        if (self.getTensorInfo(name)) |info| {
            const tensor_data = self.mmap_ptr[self.tensor_data_offset..];
            return info.getElems(tensor_data);
        }
        return null;
    }

    /// Retrieve the declared file type of this GGUF file.
    /// Returns `null` if there is no declared file type, or the value is invalid.
    pub fn fileType(self: Self) ?FileType {
        if (self.getValue(file_type_key)) |inner| {
            if (inner != .uint32) {
                const as_enum: MetadataType = inner;
                std.debug.print("ggml: {s} was present but had wrong type: {}\n", .{ file_type_key, as_enum });
                return null;
            }

            const val = inner.uint32;
            if (val < max_file_type) {
                return @enumFromInt(val);
            }
        }
        return null;
    }
};

pub fn main() !void {
    var GPA = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = GPA.allocator();

    const path = "llama2-7b-f32.gguf";
    std.debug.print("Reading file {s}\n", .{path});
    var file = try GGUFFile.read_file(path, alloc);
    defer file.deinit();
    std.debug.print("Read file {s}\n", .{path});

    std.debug.print("Printing metadata\n", .{});
    for (file.metadata) |kv| {
        const value_type: MetadataType = kv.value;
        switch (value_type) {
            .uint32 => std.debug.print("{s}: u32 {d}\n", .{ kv.key.str, kv.value.uint32 }),
            .uint64 => std.debug.print("{s}: u64 {d}\n", .{ kv.key.str, kv.value.uint64 }),
            .string => std.debug.print("{s}: {s}\n", .{ kv.key.str, kv.value.string.str }),
            .array => std.debug.print("{s}: array {} len {d}\n", .{ kv.key.str, kv.value.array.elem_type, kv.value.array.len }),
            else => std.debug.print("{s}: {}\n", .{ kv.key.str, value_type }),
        }
    }

    std.debug.print("\nPrinting tensor info\n", .{});
    for (file.tensor_info) |tensor| {
        const offset = tensor.offset + file.tensor_data_offset;
        std.debug.print("Got tensor {s} with dim {d} shape {any} ({}) at {d}\n", .{
            tensor.name.str,
            tensor.dimensions.len,
            tensor.dimensions,
            tensor.ggml_type,
            offset,
        });
    }

    std.debug.print("\n", .{});
    //file.dumpMetadata();

    std.debug.print("\nClosing file\n", .{});
}
