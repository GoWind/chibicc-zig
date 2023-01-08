const std = @import("std");
const tokenizer = @import("./tokenizer.zig");
const Token = tokenizer.Token;
const Stream = tokenizer.Stream;
const Allocator = std.mem.Allocator;
const span = std.mem.span;
const panic = std.debug.panic;
const all_valid_chars = "()/*+-==!=<<=>>=;={}ifelse()forwhile";
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
pub const IF = Token{ .keyword = .{ .ptr = span(all_valid_chars[20..22]) } };
pub const ELSE = Token{ .keyword = .{ .ptr = span(all_valid_chars[22..26]) } };
pub const LPAREN = Token{ .punct = .{ .ptr = span(all_valid_chars[26..27]) } };
pub const RPAREN = Token{ .punct = .{ .ptr = span(all_valid_chars[27..28]) } };
pub const FOR = Token{ .keyword = .{ .ptr = span(all_valid_chars[28..31]) } };
pub const WHILE = Token{ .keyword = .{ .ptr = span(all_valid_chars[31..36]) } };

pub const RETURN = Token{ .keyword = .{ .ptr = span("return") } };
// ** AST Generation ** //

// Code emitter
// For and while use the same kind
// but in `while`, we disregard `inc`
const NodeKind = enum { Add, Sub, Mul, Div, Unary, Num, Eq, Neq, Lt, Lte, ExprStmt, Assign, Var, Ret, Block, If, For };

// A variable in our C code
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
// List of variables that we have encountered so far in our C code
const ObjList = std.ArrayList(*Obj);

pub const Node = struct {
    const Self = @This();
    kind: NodeKind,
    next: ?*Node = null, // Compound Statement
    lhs: ?*Node = null,
    rhs: ?*Node = null,
    body: ?*Node = null, // For Blocks
    // If else blocks, then is also used with `for`
    cond: ?*Node = null,
    then: ?*Node = null,
    els: ?*Node = null,

    // for loop and while loop
    init: ?*Node = null,
    inc: ?*Node = null,
    val: i32 = 0, // Used when Kind = Num
    variable: ?*Obj = null, // Used when Kind = var
    tok: ?*Token = null, // The token in the parse stream that was
    //used as a basis to create this node
    // self is a pointer into global heap region or a fn-local heap
    fn from_binary(self: *Self, kind: NodeKind, lhs: ?*Node, rhs: ?*Node, tok: ?*Token) void {
        self.* = .{ .kind = kind, .lhs = lhs, .rhs = rhs, .tok = tok };
    }

    fn from_num(self: *Self, num: i32, tok: ?*Token) void {
        self.* = .{ .kind = NodeKind.Num, .val = num, .tok = tok };
    }
    fn from_unary(self: *Self, kind: NodeKind, lhs: ?*Node, tok: ?*Token) void {
        self.* = .{ .kind = kind, .lhs = lhs, .tok = tok };
    }
    fn from_expr_stmt(self: *Self, lhs: ?*Node, tok: ?*Token) void {
        self.* = .{ .kind = NodeKind.ExprStmt, .lhs = lhs, .tok = tok };
    }
    fn from_ident(self: *Self, variable: *Obj, tok: ?*Token) void {
        self.* = .{ .kind = NodeKind.Var, .variable = variable, .tok = tok };
    }
    fn from_block(self: *Self, body: ?*Node, tok: ?*Token) void {
        self.* = .{ .kind = NodeKind.Block, .body = body, .tok = tok };
    }

    fn from_if_stmt(self: *Self, cond: ?*Node, then: ?*Node, els: ?*Node, tok: ?*Token) void {
        self.* = .{ .kind = NodeKind.If, .cond = cond, .then = then, .els = els, .tok = tok };
    }

    fn from_for(self: *Self, init: ?*Node, cond: ?*Node, inc: ?*Node, then: ?*Node, tok: ?*Token) void {
        self.* = .{ .kind = NodeKind.For, .init = init, .cond = cond, .inc = inc, .then = then, .tok = tok };
    }

    pub fn format(self: Self, comptime _: []const u8, _: std.fmt.FormatOptions, out_stream: anytype) !void {
        switch (self.kind) {
            NodeKind.Add, NodeKind.Sub, NodeKind.Mul, NodeKind.Div, NodeKind.Assign, NodeKind.Eq, NodeKind.Neq, NodeKind.Lt, NodeKind.Lte => {
                try std.fmt.format(out_stream, "Node {?} starting at {?} with lhs {?} and rhs {?}\n", .{ self.kind, self.tok, self.lhs, self.rhs });
            },
            NodeKind.Unary => {
                try std.fmt.format(out_stream, "Node {?} starting at {?} with lhs {?}\n", .{ self.kind, self.tok, self.lhs });
            },
            NodeKind.Num => {
                try std.fmt.format(out_stream, "Number node {?} with value {?}\n", .{ self.tok, self.val });
            },

            NodeKind.ExprStmt => {
                try std.fmt.format(out_stream, "Expression statement {?}\n", .{self.lhs});
            },
            NodeKind.Block => {
                try std.fmt.format(out_stream, "Block starting at {?}\n {?}", .{ self.tok, self.body });
            },
            NodeKind.Var => {
                try std.fmt.format(out_stream, "Variable Node {?} with value {?}\n", .{ self.tok, self.variable });
            },
            NodeKind.Ret => {
                try std.fmt.format(out_stream, "Return starting at {?}\n with expr {?}\n", .{ self.tok, self.variable });
            },
            NodeKind.If => {
                try std.fmt.format(out_stream, "If Node {?} with \ncond {?}\n then {?} else{?}\n", .{ self.tok, self.cond, self.then, self.els });
            },
            NodeKind.For => {
                try std.fmt.format(out_stream, "ForNode starting at {?} with init {?}\ncond {?}\n then {?} else{?}\n", .{ self.tok, self.init, self.cond, self.then, self.els });
            },
        }
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
        unary_node.from_unary(NodeKind.Unary, lhs, stream_top);
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
            expr_node.from_binary(op, lhs, rhs, stream_top);
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
    var top = stream.top();
    if (top.equal(&ASSIGN)) {
        stream.consume();
        var rhs = try assign(p);
        var assign_node = try p.alloc.create(Node);
        assign_node.from_binary(NodeKind.Assign, lhs, rhs, top);
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
            expr_node.from_binary(op, lhs, rhs, stream_top);
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
            var op_tok = stream_top;
            stream.consume();
            var rhs = try relational(p);
            var rel_node = try p.alloc.create(Node);
            rel_node.from_binary(op, lhs, rhs, op_tok);
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
            rel_node.from_binary(NodeKind.Lt, lhs, rhs, stream_top);
            lhs = rel_node;
        } else if (stream_top.equal(&LTE)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lte, lhs, rhs, stream_top);
            lhs = rel_node;
            // Optimization, we need not have a NodeKind.Gte
            // we can just switch lhs and rhs with the same Lt, Lte ops
        } else if (stream_top.equal(&GT)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lt, rhs, lhs, stream_top);
            lhs = rel_node;
        } else if (stream_top.equal(&GTE)) {
            var rel_node = try p.alloc.create(Node);
            stream.consume();
            var rhs = try add(p);
            rel_node.from_binary(NodeKind.Lte, rhs, lhs, stream_top);
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
        variable_node.from_ident(variable, top_token);
        s.consume();
        return variable_node;
    }

    if (std.mem.eql(u8, @tagName(top_token.*), "num")) {
        var struct_top = @field(top_token, "num");
        var val = @field(struct_top, "val");
        var num_node = try p.alloc.create(Node);
        num_node.from_num(val, top_token);
        s.consume();
        return num_node;
    } else {
        panic("unexpected token {?} to parse as primary\n", .{top_token.*});
    }
}
// stmt* }
fn compound_statement(p: *ParseContext) anyerror!*Node {
    var first_stmt: ?*Node = null;
    var it = first_stmt;
    var stream = p.stream;
    var s_top = stream.top();
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
    compound_stmt_node.from_block(first_stmt, s_top);
    stream.consume();
    return compound_stmt_node;
}
// expr-stmt = expr? ;
fn expr_statement(p: *ParseContext) !*Node {
    var s = p.stream;
    var top = s.top();
    if (top.equal(&SEMICOLON)) {
        s.consume();
        var empty_stmt = try p.alloc.create(Node);
        empty_stmt.from_block(null, top);
        return empty_stmt;
    }
    var expr_node = try p.alloc.create(Node);
    var lhs = try expr(p);
    expr_node.from_expr_stmt(lhs, top);
    s.skip(&SEMICOLON);
    return expr_node;
}
// stmt = "return" expr ";"
//      | "if" "(" expr ")" stmt ("else" stmt)?
//      | "for" "(" expr-stmt expr? ; expr? ")" stmt
//      | "while" "(" expr ")" stmt
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
        return_node.from_unary(NodeKind.Ret, return_expr, stream_top);
        return return_node;
    } else if (stream_top.equal(&IF)) {
        stream.consume();
        stream.skip(&LPAREN);
        var if_node = try p.alloc.create(Node);
        var if_condition = try expr(p);
        stream.skip(&RPAREN);
        var then_stmt = try stmt(p);
        if (stream.top().equal(&ELSE)) {
            stream.consume();
            var else_stmt = try stmt(p);
            if_node.from_if_stmt(if_condition, then_stmt, else_stmt, stream_top);
        } else {
            if_node.from_if_stmt(if_condition, then_stmt, null, stream_top);
        }
        return if_node;
    } else if (stream_top.equal(&FOR)) {
        stream.consume();
        stream.skip(&LPAREN);
        var for_init = try expr_statement(p);
        var for_cond: ?*Node = null;
        var for_inc: ?*Node = null;
        if (stream.top().equal(&SEMICOLON) == false) {
            for_cond = try expr(p);
        }
        stream.skip(&SEMICOLON);
        if (stream.top().equal(&RPAREN) == false) {
            for_inc = try expr(p);
        }
        stream.skip(&RPAREN);
        var for_then = try stmt(p);
        var for_node = try p.alloc.create(Node);
        for_node.from_for(for_init, for_cond, for_inc, for_then, stream_top);
        return for_node;
    } else if (stream_top.equal(&WHILE)) {
        stream.consume();
        stream.skip(&LPAREN);
        var while_cond = try expr(p);
        stream.skip(&RPAREN);
        var while_then = try stmt(p);
        var while_node = try p.alloc.create(Node);
        while_node.from_for(null, while_cond, null, while_then, stream_top);
        return while_node;
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
    var stdout_writer = stdout.writer();

    switch (n.kind) {
        NodeKind.If => {
            try gen_expr(n.cond.?);
            // x0 is 1. when `cond` holds
            // if x0 is != 1, then cond => false hence
            // we jump to else
            var branch_id = update_branch_count();
            try stdout_writer.print("cmp x0, #0\n", .{});
            try stdout_writer.print("b.eq else_label_{}\n", .{branch_id});
            try gen_stmt(n.then.?);
            try stdout_writer.print("b end_label_{}\n", .{branch_id});
            try stdout_writer.print("else_label_{}:\n", .{branch_id});
            if (n.els) |else_stmt| {
                try gen_stmt(else_stmt);
            }
            try stdout_writer.print("end_label_{}:\n", .{branch_id});
        },
        NodeKind.For => {
            var branch_id = update_branch_count();
            if (n.init) |for_init| {
                try gen_stmt(for_init);
            }
            try stdout_writer.print("for_label{}:\n", .{branch_id});
            if (n.cond) |for_cond| {
                try gen_expr(for_cond);
                try stdout_writer.print("cmp x0, #0\n", .{});
                try stdout_writer.print("b.eq for_end_label{}\n", .{branch_id});
            }
            try gen_stmt(n.then.?);
            if (n.inc) |inc| {
                try gen_expr(inc);
            }
            try stdout_writer.print("b for_label{}\n", .{branch_id});
            try stdout_writer.print("for_end_label{}:\n", .{branch_id});
        },
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

//FIXME: How do I make this a local within the main codegen function ?
var if_branch_val: u32 = 0;
fn update_branch_count() u32 {
    if_branch_val += 1;
    return if_branch_val;
}
