const std = @import("std");
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const program =
    \\.global _start
    \\.align 2
    \\_start:
    \\mov X0, {}
    \\mov X1, {}
    \\{s} X0, X0, X1
    \\ret
;
const add = "add";
const sub = "sub";
pub fn main() anyerror!void {
    var stdout = std.io.getStdOut();
    var argv = std.os.argv;
    if (argv.len != 4) {
        @panic("must have atleast 1 arg");
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var val1 = try strtol(argv[1]);
    var op = "add".*;
    if (argv[2][0] != '+' and argv[2][0] != '-') {
        @panic("invalid op");
    }
    if (argv[2][0] == '-') {
        op = "sub".*;
    }
    var val2 = try strtol(argv[3]);
    defer arena.deinit();
    var allocator = arena.allocator();
    var program_fmt_buf = try std.fmt.allocPrint(allocator, program, .{ val1, val2, op });
    try stdout.writeAll(program_fmt_buf);
}

fn strtol(y: [*:0]const u8) anyerror!i32 {
    if (y[0] == 0) {
        return error.Cantdothis;
    }
    var sign: i32 = 1;
    var acc: i32 = 0;
    var i: usize = 0;
    if (y[0] == '-' or y[1] == '+') {
        sign = if (y[0] == '-') -1 else 1;
        if (y[1] == 0) {
            return error.Cantdothis;
        }
        i = i + 1;
    }
    while (y[i] != 0) : (i += 1) {
        if (y[i] < '0' or y[i] > '9') {
            return error.Cantdothis;
        }
        acc = acc * 10 + (y[i] - '0');
    }
    return acc * sign;
}

test "basic test" {
    var value: [*:0]const u8 = "-233";
    try std.testing.expectEqual(strtol(value), -233);
}
