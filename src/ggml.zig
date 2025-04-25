// -*- coding: utf-8 -*-
// llm.zig - LLM implementation in Zig
//
// SPDX-License-Identifier: GPL-3.0-or-later
// © 2025 Cameron Conn

//! ggml: This module contains methods and structs to parse the `.gguf` file format
//! from the [GGML](https://github.com/ggml-org/ggml/tree/master) library.

const std = @import("std");

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
        std.debug.assert(n_read == len);
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
pub const MetadataKV = struct {
    key: String,
    value_type: MetadataType,
    value: Value,

    fn read(reader: Reader, alloc: std.mem.Allocator) !MetadataKV {
        const key = try String.read(reader, alloc);
        const value_type = try reader.readEnum(MetadataType, .little);
        const value = try Value.read(value_type, reader, alloc);
        return .{
            .key = key,
            .value_type = value_type,
            .value = value,
        };
    }
};

// gguf_header_t
pub const GGUFHeader = struct {
    const magic_str = "GGUF"; // 0x47, 0x47, 0x55, 0x46
    const magic_num = std.mem.readInt(u32, magic_str, .little);

    magic: u32,
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
            .magic = magic,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
        };
    }
};

// gguf_tensor_info_t
pub const TensorInfo = struct {
    const Self = @This();

    name: String,
    dim: u32,
    dimensions: []u64,
    ggml_type: Type,
    /// The relative offset of the tensor within the file, in bytes.
    /// This is a u64, but is represented as a usize to prevent a bunch of `@intCast` calls
    /// within model code.
    offset: usize,

    fn read(reader: Reader, alloc: std.mem.Allocator) Error!TensorInfo {
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

        return .{
            .name = name,
            .dim = dim,
            .dimensions = dimensions,
            .ggml_type = ggml_type,
            .offset = offset,
        };
    }

    /// Get the element type of this
    pub fn getElemType(self: Self) type {
        return switch (self.ggml_type) {
            .U8 => u8,
            .I8 => i8,
            .U16 => u16,
            .I16 => i16,
            .U32 => u32,
            .I32 => i32,
            .F32 => f32,
            .U64 => u32,
            .I64 => i32,
            .F64 => f64,

            else => {
                std.debug.print("Unimplemented type: {}\n", .{self.ggml_type});
                @panic("Unimplemented type");
            },
        };
    }

    /// Get a raw slice of the Tensor's data in native model order.
    /// Get the elements of this tensor from the files `mmap(2)` pointer.
    pub fn getElems(self: Self, T: type, data_start: usize) []const T {
        // TODO: Figure out a better way of enforcing type safety.
        //std.debug.assert(T == self.getElemType());

        const target = data_start + self.offset;

        var len: usize = 1;
        for (0..self.dim) |i| {
            len *= @intCast(self.dimensions[i]);
        }
        std.debug.assert(len >= 1);

        const ptr: [*]T = @ptrFromInt(target);
        return ptr[0..len];
    }
};

pub const alignment_key = "general.alignment";
pub const quant_version_key = "general.quantization_version";
pub const arch_key = "general.architecture";
pub const name_key = "general.name";

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
    tensor_data_offset: usize,
    //tensor_data: anyopaque,

    // ========================================
    // housekeeping fields
    // ========================================
    arena: Arena,

    fd: std.posix.fd_t,
    mmap_ptr: []align(page_size) u8,

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
        // Read tensor info
        const tensor_info = try read_tensor_info(header.tensor_count, reader, allocator);

        // Alignment must be a u32
        const alignment = if (getMetadataValue(alignment_key, metadata)) |val|
            switch (val.*) {
                .uint32 => val.*.uint32,
                else => {
                    std.debug.print("\"{s}\" must be a uint32\n", .{alignment_key});
                    return Error.Format;
                },
            }
        else
            32;
        // Quantization version must be a u32
        const quantization = if (getMetadataValue(quant_version_key, metadata)) |val|
            switch (val.*) {
                .uint32 => val.*.uint32,
                else => {
                    std.debug.print("\"{s}\" must be a uint32\n", .{quant_version_key});
                    return Error.Format;
                },
            }
        else
            0;

        const arch = if (getMetadataValue(arch_key, metadata)) |arch|
            switch (arch.*) {
                .string => arch.*.string.str,
                else => {
                    std.debug.print("\"{s}\" must be a string\n", .{arch_key});
                    return Error.Format;
                },
            }
        else {
            std.debug.print("\"{s}\" is not present\n", .{arch_key});
            return Error.Format;
        };

        const name = if (getMetadataValue(name_key, metadata)) |name|
            switch (name.*) {
                .string => name.*.string.str,
                else => {
                    std.debug.print("\"{s}\" must be a string\n", .{name_key});
                    return Error.Format;
                },
            }
        else
            "unset";

        std.debug.print("Model architecture: {s}\n", .{arch});
        std.debug.print("Model name: {s}\n", .{name});
        std.debug.print("desired alignment: {d}; quantization: {d}\n", .{ alignment, quantization });

        const offset: usize = @intCast(try reader.context.getPos());
        std.debug.print("Current file offset is {d}\n", .{offset});

        const tensor_data_offset = std.mem.alignForward(usize, offset + @intFromPtr(ptr.ptr), @intCast(alignment));
        std.debug.print("Tensor_data starts at {d}\n", .{tensor_data_offset});

        return .{
            // gguf fields
            .header = header,
            .metadata = metadata,
            .tensor_info = tensor_info,
            .tensor_data_offset = tensor_data_offset,

            // housekeeping fields
            .fd = fd,
            .mmap_ptr = ptr,

            .arena = arena,
        };
    }

    fn read_metadata(
        metadata_count: u64,
        reader: Reader,
        alloc: std.mem.Allocator,
    ) ![]MetadataKV {
        var ret = std.ArrayList(MetadataKV).init(alloc);
        errdefer ret.deinit();
        try ret.ensureTotalCapacityPrecise(@intCast(metadata_count));

        for (0..metadata_count) |_| {
            const kv = try MetadataKV.read(reader, alloc);
            try ret.append(kv);
        }
        return ret.toOwnedSlice();
    }

    fn read_tensor_info(count: u64, reader: Reader, alloc: std.mem.Allocator) ![]TensorInfo {
        var ret = std.ArrayList(TensorInfo).init(alloc);
        errdefer ret.deinit();
        try ret.ensureTotalCapacityPrecise(count);

        for (0..count) |_| {
            const tensor = try TensorInfo.read(reader, alloc);
            try ret.append(tensor);
        }
        return ret.toOwnedSlice();
    }

    fn getMetadataValue(key: []const u8, metadata: []MetadataKV) ?*const Value {
        for (metadata) |kv| {
            if (std.mem.eql(u8, key, kv.key.str)) {
                return &kv.value;
            }
        }
        return null;
    }

    /// Get the Metadata value for a key if it is present or `null` otherwise.
    pub fn getValue(self: *const Self, key: []const u8) ?*const Value {
        return getMetadataValue(key, self.metadata);
    }

    pub fn dumpMetadata(self: *const Self) void {
        for (self.metadata) |kv| {
            switch (kv.value_type) {
                .string => std.debug.print("{s}={s}\n", .{ kv.key.str, kv.value.string.str }),
                .uint32 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.uint32 }),
                .uint64 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.uint64 }),
                .float32 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.float32 }),
                .float64 => std.debug.print("{s}={d}\n", .{ kv.key.str, kv.value.float64 }),
                .array => std.debug.print("{s}=array {d} elem, {}\n", .{ kv.key.str, kv.value.array.len, kv.value.array.elem_type }),
                else => std.debug.print("{s}={}\n", .{ kv.key.str, kv.value_type }),
            }
        }
    }

    /// De-initialize this GGUF file and free up underlying resources.
    /// This invalidates any reference to memory of this GGUF file.
    pub fn deinit(self: *Self) void {
        _ = self.arena.reset(.free_all);
        self.arena.deinit();

        std.posix.munmap(self.mmap_ptr);
        std.posix.close(self.fd);
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
};

pub fn main() !void {
    var GPA = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = GPA.allocator();

    const path = "Llama-2-7B-F32.gguf";
    std.debug.print("Reading file {s}\n", .{path});
    var file = try GGUFFile.read_file(path, alloc);
    defer file.deinit();
    std.debug.print("Read file {s}\n", .{path});

    std.debug.print("Printing metadata\n", .{});
    for (file.metadata) |kv| {
        switch (kv.value_type) {
            .uint32 => std.debug.print("{s}: u32 {d}\n", .{ kv.key.str, kv.value.uint32 }),
            else => std.debug.print("{s}: {}\n", .{ kv.key.str, kv.value_type }),
        }
    }

    std.debug.print("\nPrinting tensor info\n", .{});
    for (file.tensor_info) |tensor| {
        std.debug.print("Got tensor {s} with dim {d} shape {any}\n", .{ tensor.name.str, tensor.dim, tensor.dimensions });
    }

    std.debug.print("\n", .{});
    //file.dumpMetadata();

    std.debug.print("\nClosing file\n", .{});
}
