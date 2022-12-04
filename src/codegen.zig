const std = @import("std");
const tokenizer = @import("./tokenizer.zig");
const Token = tokenizer.Token;
const Stream = tokenizer.Stream;
const Allocator = std.mem.Allocator;
const span = std.mem.span;
const panic = std.debug.panic;
const all_valid_chars = "()/*+-==!=<<=>>=";
const stdout = std.io.getStdOut();
pub const LEFT_PAREN = Token{ .punct = .{ .ptr = span(all_valid_chars[0..1]) } };
pub const RIGHT_PAREN = Token{ .punct = .{ .ptr = span(all_valid_chars[1..2]) } };
pub const DIV = Token{ .punct = .{ .ptr = span(all_valid_chars[2..3]) } };
pub const MUL = Token{ .punct = .{ .ptr = span(all_valid_chars[3..4]) } };
pub const PLUS = Token{ .punct = .{ .ptr = span(all_valid_chars[4..5]) } };
pub const MINUS = Token{ .punct = .{ .ptr = span(all_valid_chars[5..6]) } };
pub const EQEQ = Token{ .punct = .{ .ptr = span(all_valid_chars[6..8]) } };
pub const NOTEQ = Token{ .punct = .{ .ptr = span(all_valid_chars[8..10]) } };
pub const LT = Token{ .punct = .{ .ptr = span(all_valid_chars[10..11]) } };
pub const LTE = Token{ .punct = .{ .ptr = span(all_valid_chars[11..13]) } };
pub const GT = Token{ .punct = .{ .ptr = span(all_valid_chars[13..14]) } };
pub const GTE = Token{ .punct = .{ .ptr = span(all_valid_chars[14..16]) } };

// ** AST Generation ** //

// Code emitter
const NodeKind = enum { Add, Sub, Mul, Div, Unary, Num, Eq, Neq, Lt, Lte };
pub const Node = struct {
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
    fn from_unary(self: *Self, lhs: ?*Node) void {
        self.* = .{ .kind = NodeKind.Unary, .lhs = lhs };
    }
};

fn unary(stream: *Stream, alloc: Allocator) !*Node {
    var stream_top = stream.top();
    if (stream_top.equal(&PLUS)) {
        stream.consume();
        return unary(stream, alloc);
    } else if (stream_top.equal(&MINUS)) {
        stream.consume();
        var lhs = try unary(stream, alloc);
        var unary_node = try alloc.create(Node);
        unary_node.from_unary(lhs);
        return unary_node;
    }
    return primary(stream, alloc);
}
fn mul(s: *Stream, alloc: Allocator) !*Node {
    var lhs = try unary(s, alloc);
    var loop = true;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&MUL) == true) or (stream_top.equal(&DIV) == true)) {
            var op = if (stream_top.equal(&MUL)) NodeKind.Mul else NodeKind.Div;
            s.consume();
            var rhs = try unary(s, alloc);
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
    return equality(s, alloc);
}
fn add(s: *Stream, alloc: Allocator) !*Node {
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
fn equality(stream: *Stream, alloc: Allocator) !*Node {
    var lhs = try relational(stream, alloc);
    var loop = true;
    while (loop) {
        var stream_top = stream.top();
        if (stream_top.equal(&EQEQ) or stream_top.equal(&NOTEQ)) {
            var op = if (stream_top.equal(&EQEQ)) NodeKind.Eq else NodeKind.Neq;
            stream.consume();
            var rhs = try relational(stream, alloc);
            var rel_node = try alloc.create(Node);
            rel_node.from_binary(op, lhs, rhs);
            lhs = rel_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}
fn relational(stream: *Stream, alloc: Allocator) !*Node {
    var lhs = try add(stream, alloc);
    var loop = true;
    while (loop) {
        var stream_top = stream.top();
        if (stream_top.equal(&LT)) {
            var rel_node = try alloc.create(Node);
            stream.consume();
            var rhs = try add(stream, alloc);
            rel_node.from_binary(NodeKind.Lt, lhs, rhs);
            lhs = rel_node;
        } else if (stream_top.equal(&LTE)) {
            var rel_node = try alloc.create(Node);
            stream.consume();
            var rhs = try add(stream, alloc);
            rel_node.from_binary(NodeKind.Lte, lhs, rhs);
            lhs = rel_node;
            // Optimization, we need not have a NodeKind.Gte
            // we can just switch lhs and rhs with the same Lt, Lte ops
        } else if (stream_top.equal(&GT)) {
            var rel_node = try alloc.create(Node);
            stream.consume();
            var rhs = try add(stream, alloc);
            rel_node.from_binary(NodeKind.Lt, rhs, lhs);
            lhs = rel_node;
        } else if (stream_top.equal(&GTE)) {
            var rel_node = try alloc.create(Node);
            stream.consume();
            var rhs = try add(stream, alloc);
            rel_node.from_binary(NodeKind.Lte, rhs, lhs);
            lhs = rel_node;
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
//TODO: Turn this into a local variable within the `generateProgram` functions
var depth: u32 = 0;
// ** Code-Generation ** //
fn push() !void {
    depth += 1;
    try stdout.writer().print("str X0, [sp, -16]\n", .{});
    try stdout.writer().print("sub sp, sp, #16\n", .{});
}
fn pop(reg: []const u8) !void {
    depth -= 1;
    try stdout.writer().print("ldr {s}, [sp]\n", .{reg});
    try stdout.writer().print("add sp, sp, #16\n", .{});
}
// Code generation
pub fn gen_expr(node: *Node) !void {
    if (node.kind == NodeKind.Num) {
        try stdout.writer().print("mov X0, {}\n", .{node.val});
    } else if (node.kind == NodeKind.Unary) {
        try gen_expr(node.lhs.?);
        try stdout.writer().print("neg x0, x0\n", .{});
    } else {
        // idea, gen_expr, returns which register the end value of that expr is in
        // we can then use this as an input into the subsequent Add, Sub, Mul, Div
        // instructions, instead of pushing and popping from stack
        try gen_expr(node.rhs.?);
        try push();
        try gen_expr(node.lhs.?);
        try pop(span("x1"));
        // Idea: Add should be able to take a reg (x0..x18) as input and generate
        // instructions as per that
        // for each instruction, we keep track of which x register is free and then emit instructions
        // into that reg and then cross out that register as occupied
        switch (node.kind) {
            NodeKind.Add => {
                try stdout.writer().print("add x0, x0, x1\n", .{});
            },
            NodeKind.Sub => {
                try stdout.writer().print("sub x0, x0, x1\n", .{});
            },
            // This should be smul x0, w0, w1
            NodeKind.Mul => {
                try stdout.writer().print("mul x0, x0, x1\n", .{});
            },
            NodeKind.Div => {
                try stdout.writer().print("sdiv x0, x0, x1\n", .{});
            },
            NodeKind.Eq => {
                try stdout.writer().print("cmp x0, x1\n ", .{});
                try stdout.writer().print("cset x0, eq\n ", .{});
            },
            NodeKind.Neq => {
                try stdout.writer().print("cmp x0, x1\n ", .{});
                try stdout.writer().print("cset x0, ne\n ", .{});
            },
            NodeKind.Lt => {
                try stdout.writer().print("cmp x0, x1\n ", .{});
                try stdout.writer().print("cset x0, lt\n ", .{});
            },
            NodeKind.Lte => {
                try stdout.writer().print("cmp x0, x1\n ", .{});
                try stdout.writer().print("cset x0, le\n ", .{});
            },
            else => {
                panic("we shouldn't be here at all", .{});
            },
        }
    }
}

pub fn stream_to_ast(s: *Stream, alloc: Allocator) !*Node {
    return expr(s, alloc);
}

// In ARM assembler default alignment is a 4-byte boundart
// .align takes argument `exponent` and alignment = 2 power exponent
const program_header =
    \\.global _start
    \\.align 2
    \\_start:
;
pub fn generateProgram(n: *Node, alloc: Allocator) !void {
    depth = 0;
    var program = try std.fmt.allocPrint(alloc, "{s}\n", .{program_header});
    try stdout.writeAll(program);
    try gen_expr(n);
    std.debug.assert(depth == 0);
    try stdout.writer().print("ret\n", .{});
}
