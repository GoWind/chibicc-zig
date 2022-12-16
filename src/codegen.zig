const std = @import("std");
const tokenizer = @import("./tokenizer.zig");
const Token = tokenizer.Token;
const Stream = tokenizer.Stream;
const Allocator = std.mem.Allocator;
const span = std.mem.span;
const panic = std.debug.panic;
const all_valid_chars = "()/*+-==!=<<=>>=;={}";
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
pub const SEMICOLON = Token{ .punct = .{ .ptr = span(all_valid_chars[16..17]) } };
pub const ASSIGN = Token{ .punct = .{ .ptr = span(all_valid_chars[17..18]) } };
pub const LBRACE = Token{ .punct = .{ .ptr = span(all_valid_chars[18..19]) } };
pub const RBRACE = Token{ .punct = .{ .ptr = span(all_valid_chars[19..20]) } };

pub const RETURN = Token{ .keyword = .{ .ptr = span("return") } };
// ** AST Generation ** //

// Code emitter
const NodeKind = enum { Add, Sub, Mul, Div, Unary, Num, Eq, Neq, Lt, Lte, ExprStmt, Assign, Var, Ret, Block };
pub const Node = struct {
    const Self = @This();
    kind: NodeKind,
    next: ?*Node = null,
    lhs: ?*Node = null,
    rhs: ?*Node = null,
    body: ?*Node = null,
    val: i32 = 0, // Used when Kind = Num
    variable: ?*Obj = null, // Used when Kind = var

    // self is a pointer into global heap region or a fn-local heap
    fn from_binary(self: *Self, kind: NodeKind, lhs: ?*Node, rhs: ?*Node) void {
        self.* = .{ .kind = kind, .lhs = lhs, .rhs = rhs };
    }

    fn from_num(self: *Self, num: i32) void {
        self.* = .{ .kind = NodeKind.Num, .val = num };
    }
    fn from_unary(self: *Self, kind: NodeKind, lhs: ?*Node) void {
        self.* = .{ .kind = kind, .lhs = lhs };
    }
    fn from_expr_stmt(self: *Self, lhs: ?*Node) void {
        self.* = .{ .kind = NodeKind.ExprStmt, .lhs = lhs };
    }
    fn from_ident(self: *Self, variable: *Obj) void {
        self.* = .{ .kind = NodeKind.Var, .variable = variable };
    }
    fn from_block(self: *Self, body: ?*Node) void {
        self.* = .{ .kind = NodeKind.Block, .body = body };
    }
};

fn unary(p: *ParseContext) !*Node {
    var stream = p.stream;
    var stream_top = stream.top();
    if (stream_top.equal(&PLUS)) {
        stream.consume();
        return unary(p);
    } else if (stream_top.equal(&MINUS)) {
        stream.consume();
        var lhs = try unary(p);
        var unary_node = try p.alloc.create(Node);
        unary_node.from_unary(NodeKind.Unary, lhs);
        return unary_node;
    }
    return primary(p);
}
fn mul(p: *ParseContext) !*Node {
    var lhs = try unary(p);
    var loop = true;
    var s = p.stream;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&MUL) == true) or (stream_top.equal(&DIV) == true)) {
            var op = if (stream_top.equal(&MUL)) NodeKind.Mul else NodeKind.Div;
            s.consume();
            var rhs = try unary(p);
            var expr_node = try p.alloc.create(Node);
            expr_node.from_binary(op, lhs, rhs);
            lhs = expr_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}

fn assign(p: *ParseContext) !*Node {
    var lhs = try equality(p);
    var stream = p.stream;
    if (stream.top().equal(&ASSIGN)) {
        stream.consume();
        var rhs = try assign(p);
        var assign_node = try p.alloc.create(Node);
        assign_node.from_binary(NodeKind.Assign, lhs, rhs);
        return assign_node;
    }
    return lhs;
}
fn expr(p: *ParseContext) !*Node {
    return assign(p);
}
fn add(p: *ParseContext) !*Node {
    var lhs = try mul(p);
    var loop = true;
    var s = p.stream;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&PLUS) == true) or (stream_top.equal(&MINUS) == true)) {
            var op = if (stream_top.equal(&PLUS)) NodeKind.Add else NodeKind.Sub;
            s.consume();
            var rhs = try mul(p);
            var expr_node = try p.alloc.create(Node);
            expr_node.from_binary(op, lhs, rhs);
            lhs = expr_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}
fn equality(p: *ParseContext) !*Node {
    var lhs = try relational(p);
    var loop = true;
    var stream = p.stream;
    while (loop) {
        var stream_top = stream.top();
        if (stream_top.equal(&EQEQ) or stream_top.equal(&NOTEQ)) {
            var op = if (stream_top.equal(&EQEQ)) NodeKind.Eq else NodeKind.Neq;
            stream.consume();
            var rhs = try relational(p);
            var rel_node = try p.alloc.create(Node);
            rel_node.from_binary(op, lhs, rhs);
            lhs = rel_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}
fn relational(p: *ParseContext) !*Node {
    var lhs = try add(p);
    var loop = true;
    var stream = p.stream;
    while (loop) {
        var stream_top = stream.top();
        if (stream_top.equal(&LT)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lt, lhs, rhs);
            lhs = rel_node;
        } else if (stream_top.equal(&LTE)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lte, lhs, rhs);
            lhs = rel_node;
            // Optimization, we need not have a NodeKind.Gte
            // we can just switch lhs and rhs with the same Lt, Lte ops
        } else if (stream_top.equal(&GT)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lt, rhs, lhs);
            lhs = rel_node;
        } else if (stream_top.equal(&GTE)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lte, rhs, lhs);
            lhs = rel_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}
fn primary(p: *ParseContext) anyerror!*Node {
    var s = p.stream;
    var top_token = s.top();
    if (top_token.equal(&LEFT_PAREN)) {
        // var top_idx = s.pos();
        s.consume();
        var expression = try expr(p);
        s.skip(&RIGHT_PAREN);
        return expression;
    }
    if (std.mem.eql(u8, @tagName(top_token.*), "ident")) {
        var struct_top = @field(top_token, "ident");
        var name = @field(struct_top, "ptr");
        var variable = if (find_local_var(name, p.locals)) |local_var| local_var else try Obj.alloc_obj(p.alloc, name);
        try p.locals.append(variable);
        var variable_node = try p.alloc.create(Node);
        variable_node.from_ident(variable);
        s.consume();
        return variable_node;
    }

    if (std.mem.eql(u8, @tagName(top_token.*), "num")) {
        var struct_top = @field(top_token, "num");
        var val = @field(struct_top, "val");
        var num_node = try p.alloc.create(Node);
        num_node.from_num(val);
        s.consume();
        return num_node;
    } else {
        panic("unexpected token {?} to parse as primary", .{top_token});
    }
}

fn compound_statement(p: *ParseContext) anyerror!*Node {
    var first_stmt: ?*Node = null;
    var it = first_stmt;
    var stream = p.stream;
    while (stream.top().equal(&RBRACE) == false) {
        var statement = try stmt(p);
        if (first_stmt == null) {
            first_stmt = statement;
            it = statement;
        } else {
            it.?.next = statement;
            it = statement;
        }
    }
    var compound_stmt_node = try p.alloc.create(Node);
    compound_stmt_node.from_block(first_stmt);
    stream.consume();
    return compound_stmt_node;
}
// expr-stmt = expr? ;
fn expr_statement(p: *ParseContext) !*Node {
    var s = p.stream;
    if (s.top().equal(&SEMICOLON)) {
        s.consume();
        var empty_stmt = try p.alloc.create(Node);
        empty_stmt.from_block(null);
        return empty_stmt;
    }
    var expr_node = try p.alloc.create(Node);
    var lhs = try expr(p);
    expr_node.from_expr_stmt(lhs);
    s.skip(&SEMICOLON);
    return expr_node;
}
// stmt = "return" expr ";"
//      | "{" compound-stmt
//      | expr-stmt
fn stmt(p: *ParseContext) !*Node {
    var stream = p.stream;
    var stream_top = stream.top();
    if (stream_top.equal(&RETURN)) {
        stream.consume();
        var return_expr = try expr(p);
        stream.skip(&SEMICOLON);
        var return_node = try p.alloc.create(Node);
        return_node.from_unary(NodeKind.Ret, return_expr);
        return return_node;
    } else if (stream_top.equal(&LBRACE)) {
        stream.consume();
        return compound_statement(p);
    } else {
        return expr_statement(p);
    }
}
pub fn parse(s: *Stream, alloc: Allocator) !*Function {
    var parse_context = ParseContext{ .stream = s, .alloc = alloc, .locals = ObjList.init(alloc) };
    //FIXME: When do we deinit locals ?
    s.skip(&LBRACE);
    var program_body = try compound_statement(&parse_context);

    var f = try alloc.create(Function);
    f.fnbody = program_body;
    f.locals = parse_context.locals;
    f.stack_size = 0; //FIXME: What should this be ?
    return f;
}

// In ARM assembler default alignment is a 4-byte boundart
// .align takes argument `exponent` and alignment = 2 power exponent

const fn_prologue =
    \\.global _start
    \\.align 2
    \\_start:
    \\ str x29, [sp, -16]
    \\ sub sp, sp, #16
    \\ mov x29, sp
;
// each fn's body will sub sp based on the # of local vars after `fn_prologue`
// similarly each fn will add sp to its original location based on the # of local vars after `fn_epilogue`
const fn_epilogue =
    \\ return_label:
    \\ ldr x29, [x29]
    \\ add sp, sp, 16
    \\ ret
;
// At the end of this, X0 will have the addr
// of the variable being loaded
fn gen_addr(node: *Node) !void {
    if (node.kind == NodeKind.Var) {
        var offset = node.variable.?.offset;
        try stdout.writer().print(";; variable {s} at offset {}\n", .{ node.variable.?.name, offset });
        try stdout.writer().print("add x0, x29, #-{}\n", .{offset});
    } else {
        panic("Not an lvalue {?}", .{node});
    }
}
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
        try stdout.writer().print(";; loading immediate {} at\n", .{node.val});
        try stdout.writer().print("mov X0, {}\n", .{node.val});
    } else if (node.kind == NodeKind.Var) {
        try gen_addr(node); //x0 has address of variable
        try stdout.writer().print("ldr x0, [x0]\n", .{}); // now load val of variable into x0
    } else if (node.kind == NodeKind.Unary) {
        try gen_expr(node.lhs.?);
        try stdout.writer().print("neg x0, x0\n", .{});
    } else if (node.kind == NodeKind.Assign) {
        try gen_addr(node.lhs.?); //x0 has addr of variable
        try push(); //push x0 into stack
        try gen_expr(node.rhs.?); // x0 now has value
        try pop(span("x1")); // x1 has addr of variable
        try stdout.writer().print("str X0, [X1]\n", .{});
    } else {
        // Idea, gen_expr, returns which register the end value of that expr is in
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

fn gen_stmt(n: *Node) !void {
    switch (n.kind) {
        NodeKind.Block => {
            var maybe_it = n.body;
            while (maybe_it) |it| {
                try gen_stmt(it);
                maybe_it = it.next;
            }
        },
        NodeKind.ExprStmt => {
            try gen_expr(n.lhs.?);
            return;
        },
        NodeKind.Ret => {
            try gen_expr(n.lhs.?);
            var stdout_writer = stdout.writer();
            try stdout_writer.print("b return_label\n", .{});
            return;
        },
        else => {
            panic("Invalid node {?}\n", .{n});
        },
    }
}
pub fn codegen(f: *Function) !void {
    assign_lvar_offsets(f);
    var stdout_writer = stdout.writer();
    try stdout_writer.print("{s}\n", .{fn_prologue});
    try stdout_writer.print("sub sp, sp, #{}\n", .{f.stack_size});
    try gen_stmt(f.fnbody);
    try stdout_writer.print("add sp, sp, #{}\n", .{f.stack_size});
    try stdout_writer.print("{s}\n", .{fn_epilogue});
}

const Obj = struct {
    name: []u8,
    offset: usize = 0,
    const Self = @This();
    fn alloc_obj(alloc: Allocator, name: []const u8) !*Obj {
        var obj = try alloc.create(Self);
        obj.offset = 0;
        obj.name = try alloc.dupe(u8, name);
        return obj;
    }
};

const ObjList = std.ArrayList(*Obj);
const ParseContext = struct {
    stream: *Stream,
    alloc: Allocator,
    locals: ObjList,
};
const Function = struct { fnbody: *Node, locals: ObjList, stack_size: usize };

fn align_to(n: usize, al: u32) usize {
    return (n + al - 1) / al * al;
}

fn assign_lvar_offsets(prog: *Function) void {
    var offset: usize = 0;
    for (prog.locals.items) |*local| {
        offset += 8;
        local.*.offset = offset;
    }
    prog.*.stack_size = align_to(offset, 16);
}

fn find_local_var(ident: []const u8, locals: ObjList) ?*Obj {
    for (locals.items) |l| {
        if (std.mem.eql(u8, l.name, ident)) {
            return l;
        }
    }
    return null;
}
