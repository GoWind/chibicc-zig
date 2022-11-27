const std = @import("std");
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const program_header =
    \\.global _start
    \\.align 2
    \\_start:
;
const panic = std.debug.panic;
const TokenKind = enum { punct, num, eof };
const Token = union(TokenKind) {
    punct: struct { ptr: [*:0]u8, len: usize },
    num: struct { val: i32 },
    eof: void,
};
const TokenList = std.ArrayList(Token);
const Stream = struct {
    const Self = @This();
    ts: TokenList,
    idx: usize,
    pub fn init_stream(ts: TokenList) Stream {
        return .{
            .ts = ts,
            .idx = @as(usize, 0),
        };
    }

    pub fn top(self: *Self) Token {
        return self.ts.items[self.idx];
    }

    pub fn consume(self: *Self) void {
        self.idx += 1;
    }

    pub fn is_eof(self: *Self) bool {
        if (self.idx >= self.ts.items.len) {
            return true;
        }
        switch (self.ts.items[self.idx]) {
            TokenKind.eof => {
                return true;
            },
            else => {
                return false;
            },
        }
    }
};

fn tokenize(stream_p: *[*:0]u8, list: *TokenList) !void {
    var stream = stream_p.*;
    while (stream[0] != 0) {
        if (isspace(stream[0]) == true) {
            stream += 1;
            continue;
        } else if (isdigit(stream[0]) == true) {
            var number = strtol(&stream);
            try list.append(Token{ .num = .{ .val = number.? } });
        } else if (stream[0] == '+' or stream[0] == '-') {
            try list.append(Token{ .punct = .{ .ptr = stream, .len = @as(usize, 1) } });
            stream += 1;
        } else {
            panic("Invalid token {c} from {s}\n", .{ stream[0], stream });
        }
    }
}
pub fn main() anyerror!void {
    var stdout = std.io.getStdOut();
    var argv = std.os.argv;
    if (argv.len != 2) {
        panic("must have atleast 1 arg", .{});
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var tlist = TokenList.init(allocator);
    defer tlist.deinit();
    try tokenize(&std.os.argv[1], &tlist);
    var token_stream = Stream.init_stream(tlist);
    var program = try std.fmt.allocPrint(allocator, "{s}\n", .{program_header});
    defer allocator.free(program);
    var val1 = try_get_number(&token_stream);
    program = try std.fmt.allocPrint(allocator, "{s}mov X0, {}\n", .{ program, val1 });
    while (token_stream.is_eof() == false) {
        switch (token_stream.top()) {
            TokenKind.punct => |v| {
                if (v.ptr[0] == '+') {
                    token_stream.consume();
                    var val2 = try_get_number(&token_stream);
                    program = try std.fmt.allocPrint(allocator, "{s}mov X1, {}\n", .{ program, val2 });
                    program = try std.fmt.allocPrint(allocator, "{s}add  X0, X0, X1\n", .{program});
                } else if (v.ptr[0] == '-') {
                    token_stream.consume();
                    var val2 = try_get_number(&token_stream);
                    program = try std.fmt.allocPrint(allocator, "{s}mov X1, {}\n", .{ program, val2 });
                    program = try std.fmt.allocPrint(allocator, "{s}sub  X0, X0, X1\n", .{program});
                }
            },
            else => {
                panic("this shouldn't have occured {any}\n", .{token_stream.top()});
            },
        }
    }
    program = try std.fmt.allocPrint(allocator, "{s}ret", .{program});
    try stdout.writeAll(program);
}

fn try_get_number(stream: *Stream) i32 {
    var first = stream.top();
    switch (first) {
        TokenKind.num => |v| {
            stream.consume();
            return v.val;
        },
        else => {
            panic("Expect number got {any}\n", .{first});
        },
    }
}
fn strtol(yp: *[*:0]const u8) ?i32 {
    var y = yp.*;
    if (y[0] == 0) {
        return null;
    }
    var sign: i32 = 1;
    var acc: i32 = 0;
    var i: usize = 0;
    if (y[0] == '-' or y[0] == '+') {
        sign = if (y[0] == '-') -1 else 1;
        if (y[1] == 0) { // 0 is null not '0` is not
            return null;
        }
        y = y + 1;
    }
    while (y[0] != 0) {
        if (y[i] < '0' or y[i] > '9') {
            break;
        }
        acc = acc * 10 + (y[i] - '0');
        y = y + 1;
    }
    yp.* = y;
    return acc * sign;
}

fn isspace(c: u8) bool {
    return (c == '\t' or c == '\n' or c == '\r' or c == ' ');
}
fn isdigit(n: u8) bool {
    if (n >= '0' and n <= '9') {
        return true;
    } else {
        return false;
    }
}
test "testing strtol" {
    var j = "+12346";
    var j_as_p = @as([*:0]const u8, j);
    try std.testing.expectEqual(@as(i32, 12346), strtol(&j_as_p).?);
}
