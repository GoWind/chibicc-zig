const std = @import("std");
const ascii = std.ascii;
const span = std.mem.span;
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const stdout = std.io.getStdOut();
const all_valid_chars = "()/*+-";
//This should be const, but since Token.eql() takes a *Token , and &LEFT_PAREN
//'s type is *const Token, the compiler is complaining all the time ! (Grrr)
const LEFT_PAREN = Token{ .punct = .{ .ptr = span(all_valid_chars[0..1]) } };
const RIGHT_PAREN = Token{ .punct = .{ .ptr = span(all_valid_chars[1..2]) } };
const DIV = Token{ .punct = .{ .ptr = span(all_valid_chars[2..3]) } };
const MUL = Token{ .punct = .{ .ptr = span(all_valid_chars[3..4]) } };
const PLUS = Token{ .punct = .{ .ptr = span(all_valid_chars[4..5]) } };
const MINUS = Token{ .punct = .{ .ptr = span(all_valid_chars[5..6]) } };
// In ARM assembler default alignment is a 4-byte boundart
// .align takes argument `exponent` and alignment = 2 power exponent
const program_header =
    \\.global _start
    \\.align 2
    \\_start:
;
const panic = std.debug.panic;
const TokenKind = enum { punct, num, eof };
const Token = union(TokenKind) {
    const Self = @This();
    punct: struct {
        ptr: []const u8,
    },
    num: struct { val: i32 },
    eof: void,

    fn equal(self: *Self, other: *const Token) bool {
        if (std.mem.eql(u8, @tagName(self.*), @tagName(other.*)) == false) {
            return false;
        }

        switch (self.*) {
            TokenKind.num => {
                return self.equal_nums(other);
            },
            TokenKind.punct => {
                return self.equal_puncts(other);
            },
            else => {
                panic("We shouldn't be here", .{});
            },
        }
    }

    fn equal_puncts(self: *Self, other: *const Token) bool {
        var struct_self = @field(self, "punct");
        var struct_other = @field(other, "punct");
        var ptr_self = @field(struct_self, "ptr");
        var ptr_other = @field(struct_other, "ptr");
        return std.mem.eql(u8, ptr_self, ptr_other);
    }

    fn equal_nums(self: *Self, other: *const Token) bool {
        var struct_self = @field(self, "num");
        var struct_other = @field(other, "num");
        var val_self = @field(struct_self, "val");
        var val_other = @field(struct_other, "val");
        return val_self == val_other;
    }

    fn format(self: Self, comptime _: []const u8, _: std.fmt.FormatOptions, out_stream: anytype) !void {
        switch (self) {
            TokenKind.num => |v| {
                try std.fmt.format(out_stream, "TokenKind.num {}\n", .{v.val});
            },
            TokenKind.punct => |v| {
                try std.fmt.format(out_stream, "TokenKind.punct {s}\n", .{v.ptr});
            },
            TokenKind.eof => {
                try std.fmt.format(out_stream, "TokenKind.eof \n", .{});
            },
        }
    }
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

    pub fn top(self: *Self) *Token {
        return &self.ts.items[self.idx];
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

    pub fn pos(self: *Self) usize {
        return self.idx;
    }

    pub fn skip(self: *Self, t: *const Token) void {
        if (self.is_eof()) {
            // Guess I should panic ?
            panic("Expected {}, found EOF", .{t});
        }
        var top_token = self.top();
        if (top_token.equal(t) == true) {
            self.consume();
        } else {
            panic(" stream.skip Expected token {?} got {?}\n", .{ t, top_token });
        }
    }
};

// Print an error with a pointer pointing to `stream_p.*[loc]`
fn errorToken(stream: [*:0]u8, t: Token) !void {
    var stdOut = std.io.getStdOut();
    try stdOut.writeAll(@tagName(t));
    try stdOut.writeAll(std.mem.span(stream));
}
fn tokenize(stream_p: *[*:0]u8, list: *TokenList) !void {
    var stream = stream_p.*;
    while (stream[0] != 0) {
        if (ascii.isSpace(stream[0]) == true) {
            stream += 1;
            continue;
        } else if (ascii.isDigit(stream[0]) == true) {
            //notice that we are modifying stream here
            var number = strtol(&stream);
            try list.append(Token{ .num = .{ .val = number.? } });
        } else if (ascii.isPunct(stream[0])) {
            try list.append(Token{ .punct = .{ .ptr = stream[0..1] } });
            stream += 1;
        } else {
            panic("Invalid token {c} from {s}\n", .{ stream[0], stream });
        }
    }
    try list.append(Token.eof);
}
pub fn main() anyerror!void {
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
    var top_node = try expr(&token_stream, allocator);
    try generateProgram(top_node, allocator);
}

fn generateProgram(n: *Node, alloc: Allocator) !void {
    var program = try std.fmt.allocPrint(alloc, "{s}\n", .{program_header});
    try stdout.writeAll(program);
    try gen_expr(n);
    try stdout.writer().print("ret\n", .{});
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

// Code emitter
const NodeKind = enum { Add, Sub, Mul, Div, Num };
const Node = struct {
    const Self = @This();
    kind: NodeKind,
    lhs: ?*Node = null,
    rhs: ?*Node = null,
    val: i32 = 0,

    // self is a pointer into global heap region or a fn-local heap
    fn from_binary(self: *Self, kind: NodeKind, lhs: ?*Node, rhs: ?*Node) void {
        self.* = .{ .kind = kind, .lhs = lhs, .rhs = rhs };
    }

    fn from_num(self: *Self, num: i32) void {
        self.* = .{ .kind = NodeKind.Num, .val = num };
    }
};

fn mul(s: *Stream, alloc: Allocator) !*Node {
    var lhs = try primary(s, alloc);
    var loop = true;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&MUL) == true) or (stream_top.equal(&DIV) == true)) {
            var op = if (stream_top.equal(&MUL)) NodeKind.Mul else NodeKind.Div;
            s.consume();
            var rhs = try primary(s, alloc);
            var expr_node = try alloc.create(Node);
            expr_node.from_binary(op, lhs, rhs);
            lhs = expr_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}
fn expr(s: *Stream, alloc: Allocator) !*Node {
    var lhs = try mul(s, alloc);
    var loop = true;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&PLUS) == true) or (stream_top.equal(&MINUS) == true)) {
            var op = if (stream_top.equal(&PLUS)) NodeKind.Add else NodeKind.Sub;
            s.consume();
            var rhs = try mul(s, alloc);
            var expr_node = try alloc.create(Node);
            expr_node.from_binary(op, lhs, rhs);
            lhs = expr_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}
fn primary(s: *Stream, alloc: Allocator) anyerror!*Node {
    var top_token = s.top();
    if (top_token.equal(&LEFT_PAREN)) {
        // var top_idx = s.pos();
        s.consume();
        var expression = try expr(s, alloc);
        s.skip(&RIGHT_PAREN);
        return expression;
    }
    if (std.mem.eql(u8, @tagName(top_token.*), "num")) {
        var struct_top = @field(top_token, "num");
        var val = @field(struct_top, "val");
        var num_node = try alloc.create(Node);
        num_node.from_num(val);
        s.consume();
        return num_node;
    } else {
        panic("unexpected token {?} to parse as primary", .{top_token});
    }
}

fn push() !void {
    try stdout.writer().print("str X0, [sp, -16]\n", .{});
    try stdout.writer().print("sub sp, sp, #16\n", .{});
}
fn pop(reg: []const u8) !void {
    try stdout.writer().print("ldr {s}, [sp]\n", .{reg});
    try stdout.writer().print("add sp, sp, #16\n", .{});
}
// Code generation
fn gen_expr(node: *Node) !void {
    if (node.kind == NodeKind.Num) {
        try stdout.writer().print("mov X0, {}\n", .{node.val});
    } else {
        // This is very inefficient for now, as aarch64 has 18 general
        // purpose registers and we are using only 2 of them
        // we need a way to figure out which regs are free and use them
        // instead of pushing and popping to the stack
        try gen_expr(node.rhs.?);
        try push();
        try gen_expr(node.lhs.?);
        try pop(span("x1"));
        switch (node.kind) {
            NodeKind.Add => {
                try stdout.writer().print("add x0, x0, x1\n", .{});
            },
            NodeKind.Sub => {
                try stdout.writer().print("sub x0, x0, x1\n", .{});
            },
            NodeKind.Mul => {
                try stdout.writer().print("mul x0, x0, x1\n", .{});
            },
            NodeKind.Div => {
                try stdout.writer().print("div x0, x0, x1\n", .{});
            },
            else => {
                panic("we shouldn't be here at all", .{});
            },
        }
    }
}
