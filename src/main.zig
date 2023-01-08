const std = @import("std");
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const program =
    \\.global _start
    \\.align 2
    \\_start:
    \\mov X0, {s}
    \\ret
;
pub fn main() anyerror!void {
    var stdout = std.io.getStdOut();
    var argv = std.os.argv;
    if (argv.len != 2) {
        @panic("must have atleast 1 arg");
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var program_fmt_buf = try std.fmt.allocPrint(allocator, program, .{argv[1]});
    try stdout.writeAll(program_fmt_buf);
}

test "basic test" {
    try std.testing.expectEqual(10, 3 + 7);
}
