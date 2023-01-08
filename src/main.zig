const std = @import("std");
const ascii = std.ascii;
const span = std.mem.span;
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const stdout = std.io.getStdOut();
const tokenizer = @import("./tokenizer.zig");
const codegen = @import("./codegen.zig");
const Node = codegen.Node;

const panic = std.debug.panic;

pub fn main() anyerror!void {
    var argv = std.os.argv;
    if (argv.len != 2) {
        panic("must have atleast 1 arg", .{});
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var token_stream = try tokenizer.text_to_stream(&std.os.argv[1], allocator);
    var top_node = try codegen.stream_to_ast(&token_stream, allocator);
    try codegen.generateProgram(top_node, allocator);
}
