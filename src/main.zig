const std = @import("std");
const ascii = std.ascii;
const span = std.mem.span;
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const stdout = std.io.getStdOut();
const tokenizer = @import("./tokenizer.zig");
const codegen = @import("./codegen.zig");
const Node = codegen.Node;
const POSIX_C_SOURCE = "200809L";
const panic = std.debug.panic;

fn debug_token_stream(s: *tokenizer.Stream) void {
    while (s.is_eof() == false) {
        var top = s.top();
        std.debug.print("{?}\n", .{top});
        s.advance();
    }
    s.*.idx = 0;
}
pub fn main() anyerror!void {
    var argv = std.os.argv;
    if (argv.len != 2) {
        panic("must have atleast 1 arg", .{});
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var text = try tokenizer.readFile(allocator, std.mem.span(std.os.argv[1]));
    var token_stream = try tokenizer.text_to_stream(@ptrCast(*[*:0]u8, &text), allocator);
    var top_node = try codegen.parse(&token_stream, allocator);
    try codegen.codegen(top_node);
}
