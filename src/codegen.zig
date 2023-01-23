const std = @import("std");
const tokenizer = @import("./tokenizer.zig");
const Token = tokenizer.Token;
const TokenKind = tokenizer.TokenKind;
const Stream = tokenizer.Stream;
const Allocator = std.mem.Allocator;
const span = std.mem.span;
const panic = std.debug.panic;
const all_valid_chars = "()/*+-==!=<<=>>=;={}ifelse()forwhile&*,[]";
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
pub const ADDR = Token{ .punct = .{ .ptr = span(all_valid_chars[36..37]) } };
pub const DEREF = Token{ .punct = .{ .ptr = span(all_valid_chars[37..38]) } };
pub const COMMA = Token{ .punct = .{ .ptr = span(all_valid_chars[38..39]) } };
pub const LSQ_BRACK = Token{ .punct = .{ .ptr = span(all_valid_chars[39..40]) } };
pub const RSQ_BRACK = Token{ .punct = .{ .ptr = span(all_valid_chars[40..41]) } };

pub const RETURN = Token{ .keyword = .{ .ptr = span("return") } };
pub const INT = Token{ .keyword = .{ .ptr = span("int") } };
pub const CHAR = Token{ .keyword = .{ .ptr = span("char") } };

pub const SIZEOF = Token{ .keyword = .{ .ptr = span("sizeof") } };

// ** AST Generation ** //

// Code emitter
// For and while use the same kind
// but in `while`, we disregard `inc`
const NodeKind = enum { Add, Sub, Mul, Div, Unary, Num, Eq, Neq, Lt, Lte, ExprStmt, Assign, Var, Ret, Block, If, For, Addr, Deref, Funcall };

const TypeKind = enum { Char, Int, Ptr, Func, Array };
// Type information about variables
const Type = struct {
    kind: TypeKind,
    // Pointer-to or array-of type. We intentionally use the same member
    // to represent pointer/array duality in C.
    //
    // In many contexts in which a pointer is expected, we examine this
    // member instead of "kind" member to determine whether a type is a
    // pointer or not. That means in many contexts "array of T" is
    // naturally handled as if it were "pointer to T", as required by
    // the C spec.
    base: ?*const Type = null,
    tok: *const Token,
    return_type: ?*const Type = null, // this is just for fns
    size: usize, // The size of a value of this type
    params: ?std.ArrayList(*Type) = null,
    next: ?*Type = null,
    array_len: usize = 0, // applies only when kind = Array
    const Self = @This();
    pub fn func_type(alloc: Allocator, return_type: *const Self) !*Type {
        var fn_type = try alloc.create(Self);
        fn_type.kind = TypeKind.Func;
        fn_type.return_type = return_type;
        return fn_type;
    }
    //TODO: This can be memoized
    fn pointer_to(alloc: Allocator, base: *const Type, tok: *const Token) !*Type {
        var derived = try alloc.create(Type);
        derived.kind = TypeKind.Ptr;
        derived.base = base;
        derived.tok = tok;
        derived.size = @as(usize, 8);
        return derived;
    }

    fn copy_type(alloc: Allocator, original: *const Type) !*Type {
        var copy = try alloc.create(Type);
        copy.* = original.*;
        return copy;
    }

    fn array_of(alloc: Allocator, base: *const Type, len: usize, tok: *const Token) !*Type {
        var array_type = try alloc.create(Type);
        array_type.* = .{ .base = base, .size = base.size * len, .kind = TypeKind.Array, .array_len = len, .tok = tok };
        return array_type;
    }

    fn isTypename(tok: *const Token) bool {
        return tok.equal(&CHAR) or tok.equal(&INT);
    }

    // is_integer in chibicc
    fn isIntegerLikeType(self: *const Self) bool {
        return self.kind == TypeKind.Int or self.kind == TypeKind.Char;
    }
};

// this should be a const, but that makes the pointer to this a *const Type and
// it is very annoying to change aftewards, so guess I will deal with this later
var IntBaseType = Type{ .kind = TypeKind.Int, .base = null, .tok = &INT, .params = null, .size = 8 };
var CharBaseType = Type{ .kind = TypeKind.Char, .base = null, .tok = &CHAR, .params = null, .size = 1 };

// A variable in our C code
const Variable = struct {
    name: []u8,
    offset: usize = 0,
    typ: *Type,
    //currently there are only 2 scopes : global or local
    //in future, there might be more, so this might be something else
    //apart from a boolean
    is_local: bool = false,
    const Self = @This();
    fn alloc_obj(alloc: Allocator, name: []const u8, ty: *Type, local: bool) !*Variable {
        var obj = try alloc.create(Self);
        obj.offset = 0;
        obj.name = try alloc.dupe(u8, name);
        obj.typ = ty;
        obj.is_local = local;
        return obj;
    }
    fn localVar(alloc: Allocator, name: []const u8, ty: *Type) !*Variable {
        var variable = try alloc.create(Self);
        variable.* = Self{ .offset = 0, .name = try alloc.dupe(u8, name), .typ = ty, .is_local = true };
        return variable;
    }
    fn globalVar(alloc: Allocator, name: []const u8, ty: *Type) !*Variable {
        var variable = try alloc.create(Self);
        variable.* = Self{ .offset = 0, .name = try alloc.dupe(u8, name), .typ = ty, .is_local = false };
        return variable;
    }
};
const VariableList = std.ArrayList(*Variable);

pub const Node = struct {
    const Self = @This();
    kind: NodeKind,
    n_type: ?*const Type = null, // Type information about this node, if it represents a
    // var or an immediate value etc.
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
    variable: ?*Variable = null, // Used when Kind = var
    tok: ?*const Token = null, // The token in the parse stream that was
    fn_name: []const u8 = span(""), // Used when Kind = Funcall
    fn_args: ?NodeList = null, //  Used when Kind = Funcall
    return_type: ?*Type = null,
    //used as a basis to create this node
    // self is a pointer into global heap region or a fn-local heap
    pub fn from_binary(alloc: Allocator, kind: NodeKind, lhs: ?*Node, rhs: ?*Node, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = kind, .lhs = lhs, .rhs = rhs, .tok = tok };
        return self;
    }

    pub fn from_num(alloc: Allocator, num: i32, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.Num, .val = num, .tok = tok };
        return self;
    }
    pub fn from_unary(alloc: Allocator, kind: NodeKind, lhs: ?*Node, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = kind, .lhs = lhs, .tok = tok };
        return self;
    }
    pub fn from_expr_stmt(alloc: Allocator, lhs: ?*Node, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.ExprStmt, .lhs = lhs, .tok = tok };
        return self;
    }
    pub fn from_ident(alloc: Allocator, variable: *Variable, tok: ?*const Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.Var, .variable = variable, .tok = tok };
        return self;
    }
    pub fn from_block(alloc: Allocator, body: ?*Node, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.Block, .body = body, .tok = tok };
        return self;
    }

    pub fn from_if_stmt(alloc: Allocator, cond: ?*Node, then: ?*Node, els: ?*Node, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.If, .cond = cond, .then = then, .els = els, .tok = tok };
        return self;
    }

    pub fn from_for(alloc: Allocator, init: ?*Node, cond: ?*Node, inc: ?*Node, then: ?*Node, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.For, .init = init, .cond = cond, .inc = inc, .then = then, .tok = tok };
        return self;
    }
    pub fn from_fncall(alloc: Allocator, fn_name: []const u8, fn_args: NodeList, tok: ?*Token) !*Self {
        var self = try alloc.create(Self);
        self.* = .{ .kind = NodeKind.Funcall, .fn_name = try alloc.dupe(u8, fn_name), .fn_args = fn_args, .tok = tok };
        return self;
    }

    pub fn format(self: Self, comptime _: []const u8, _: std.fmt.FormatOptions, out_stream: anytype) !void {
        switch (self.kind) {
            NodeKind.Add, NodeKind.Sub, NodeKind.Mul, NodeKind.Div, NodeKind.Assign, NodeKind.Eq, NodeKind.Neq, NodeKind.Lt, NodeKind.Lte => {
                try std.fmt.format(out_stream, "Node {?} starting at {?} with lhs {?} and rhs {?}\n", .{ self.kind, self.tok, self.lhs, self.rhs });
            },
            NodeKind.Unary, NodeKind.Addr, NodeKind.Deref => {
                try std.fmt.format(out_stream, "Node {?} starting at {?} with lhs {?}\n", .{ self.kind, self.tok, self.lhs });
            },
            NodeKind.Num => {
                try std.fmt.format(out_stream, "Number node {?} with value {?} and type {?}\n", .{ self.tok, self.val, self.n_type });
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
            NodeKind.Funcall => {
                try std.fmt.format(out_stream, "Funcall Node with name {s} \n", .{self.fn_name});
            },
        }
    }
};
const NodeList = std.ArrayList(*Node);

fn new_add(p: *ParseContext, lhs: ?*Node, rhs: ?*Node, tok: *Token) !*Node {
    if (lhs) |l|
        add_type(p.alloc, l);

    if (rhs) |r|
        add_type(p.alloc, r);

    if (lhs.?.n_type.?.isIntegerLikeType() and rhs.?.n_type.?.isIntegerLikeType()) {
        var add_node = try Node.from_binary(p.alloc, NodeKind.Add, lhs, rhs, tok);
        return add_node;
    }

    // when n_type.base != null, then then this is a Pointer type
    // in C, ptr + ptr is not possible
    // TypeKind.Ptr will be Int (for now just Int, other types will be added later)
    if (lhs.?.n_type.?.base != null and rhs.?.n_type.?.base != null) {
        panic("Invalid operands for arithmetic {?}\n", .{tok});
    }
    // change num + ptr to ptr + num
    var l: ?*Node = lhs;
    var r: ?*Node = rhs;
    if (lhs.?.n_type.?.isIntegerLikeType() and rhs.?.n_type.?.base != null) {
        l = rhs;
        r = lhs;
    }
    // stride = num * sizeof(typeOf(value pointer by ptr))
    // the following is rather unsafe, but works for now.
    // TODO: Figure out how to improve this later.

    var stride_val = @truncate(u32, l.?.n_type.?.base.?.size);
    var stride_node = try Node.from_num(p.alloc, @intCast(i32, stride_val), tok); // For now we recognize only 8 byte integers
    // Later we will utilize the size of the type from *Node `l`
    var new_rhs = try Node.from_binary(p.alloc, NodeKind.Mul, r, stride_node, tok);
    var new_binary = try Node.from_binary(p.alloc, NodeKind.Add, l, new_rhs, tok);
    return new_binary;
}

// Like `+`, `-` is overloaded for the pointer type.
fn new_sub(p: *ParseContext, lhs: ?*Node, rhs: ?*Node, tok: *Token) !*Node {
    if (lhs) |l| add_type(p.alloc, l);
    if (rhs) |r| add_type(p.alloc, r);

    if (lhs.?.n_type.?.isIntegerLikeType() and rhs.?.n_type.?.isIntegerLikeType()) {
        var sub_node = try Node.from_binary(p.alloc, NodeKind.Sub, lhs, rhs, tok);
        return sub_node;
    }

    var stride_val = @truncate(u32, lhs.?.n_type.?.base.?.size);
    var stride_node = try Node.from_num(p.alloc, @intCast(i32, stride_val), tok); // For now we recognize only 8 byte integers

    // ptr - num
    if (lhs.?.n_type.?.base != null and rhs.?.n_type.?.isIntegerLikeType()) {
        var new_rhs = try Node.from_binary(p.alloc, NodeKind.Mul, rhs, stride_node, tok);
        // we are creating a *new* rhs, so we need to add type info
        // to the new rhs node
        add_type(p.alloc, new_rhs);
        var new_binary = try Node.from_binary(p.alloc, NodeKind.Sub, lhs, new_rhs, tok);
        new_binary.n_type = lhs.?.n_type;
        return new_binary;
    }
    // ptr - ptr, which returns how many elements are between the two.
    // TODO: add check later that the base types of lhs and rhs are the same
    if (lhs.?.n_type.?.base != null and rhs.?.n_type.?.base != null) {
        // (ptr2-ptr)
        var sub_res = try Node.from_binary(p.alloc, NodeKind.Sub, lhs, rhs, tok);
        sub_res.n_type = &IntBaseType;
        // (ptr2-ptr1) / (sizeof(elementofptr1))
        var new_binary = Node.from_binary(p.alloc, NodeKind.Div, sub_res, stride_node, tok);
        return new_binary;
    }
    panic("Invalid token for Subtration operation {?}\n", .{tok});
}

fn fncall(p: *ParseContext) !*Node {
    var s = p.stream;
    var top = s.top();
    var args_list = NodeList.init(p.alloc);
    s.advance(); // consume fnname
    s.advance(); // consume "("
    while (s.top().equal(&RPAREN) == false) {
        if (args_list.items.len != 0) {
            s.skip(&COMMA);
        }
        var arg = try assign(p);
        try args_list.append(arg);
    }
    s.skip(&RPAREN);
    var fn_node = try Node.from_fncall(p.alloc, top.get_ident(), args_list, top);
    return fn_node;
}

// primary =      '(' expr ')'
//             |  ident func-args?
//             |  number
//             |  "sizeof" unary
fn primary(p: *ParseContext) anyerror!*Node {
    var s = p.stream;
    var top_token = s.top();
    if (top_token.equal(&LEFT_PAREN)) {
        // var top_idx = s.pos();
        s.advance();
        var expression = try expr(p);
        s.skip(&RIGHT_PAREN);
        return expression;
    }
    if (top_token.equal(&SIZEOF)) {
        s.advance();
        var unary_node = try unary(p);
        add_type(p.alloc, unary_node);
        var size_as_u32 = @truncate(u32, unary_node.n_type.?.size);
        return try Node.from_num(p.alloc, @intCast(i32, size_as_u32), top_token);
    } else {
        switch (top_token.*) {
            TokenKind.ident => |v| {
                if (s.next().?.equal(&LPAREN)) { // fn call
                    return fncall(p);
                }
                var variable: *Variable = undefined;
                if (findVar(p, v.ptr)) |local_var| {
                    variable = local_var;
                } else {
                    panic("Undefined variable {s}\n", .{v.ptr});
                }
                var variable_node = try Node.from_ident(p.alloc, variable, top_token);
                s.advance();
                return variable_node;
            },
            TokenKind.num => |v| {
                var num_node = try Node.from_num(p.alloc, v.val, top_token);
                s.advance();
                return num_node;
            },
            else => {
                panic("unexpected token {?} to parse as primary\n", .{top_token.*});
            },
        }
    }
}

// postfix = primary ("[" expr "]")*
fn postfix(p: *ParseContext) anyerror!*Node {
    var prim = try primary(p);
    var s = p.stream;
    while (s.top().equal(&LSQ_BRACK)) {
        var start = s.top();
        s.advance();
        var idx = try expr(p);
        s.skip(&RSQ_BRACK);
        // a[4] will be turned int soemthing like *(a+4)
        var idx_access = try new_add(p, prim, idx, start);
        prim = try Node.from_unary(p.alloc, NodeKind.Deref, idx_access, start);
    }
    return prim;
}
// unary = ( '+' | '-' | '*' | '&' | '*' ) unary
//         | postfix
fn unary(p: *ParseContext) !*Node {
    var stream = p.stream;
    var stream_top = stream.top();
    if (stream_top.equal(&PLUS)) {
        stream.advance();
        return unary(p);
    } else if (stream_top.equal(&MINUS)) {
        stream.advance();
        var lhs = try unary(p);
        var unary_node = try Node.from_unary(p.alloc, NodeKind.Unary, lhs, stream_top);
        return unary_node;
    } else if (stream_top.equal(&ADDR)) {
        stream.advance();
        var lhs = try unary(p);
        var unary_node = try Node.from_unary(p.alloc, NodeKind.Addr, lhs, stream_top);
        return unary_node;
    } else if (stream_top.equal(&DEREF)) {
        stream.advance();
        var lhs = try unary(p);
        var unary_node = try Node.from_unary(p.alloc, NodeKind.Deref, lhs, stream_top);
        return unary_node;
    }
    return postfix(p);
}

fn mul(p: *ParseContext) !*Node {
    var lhs = try unary(p);
    var loop = true;
    var s = p.stream;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&MUL) == true) or (stream_top.equal(&DIV) == true)) {
            var op = if (stream_top.equal(&MUL)) NodeKind.Mul else NodeKind.Div;
            s.advance();
            var rhs = try unary(p);
            var expr_node = try Node.from_binary(p.alloc, op, lhs, rhs, stream_top);
            lhs = expr_node;
        } else {
            loop = false;
        }
    }
    return lhs;
}

fn add(p: *ParseContext) !*Node {
    var lhs = try mul(p);
    var loop = true;
    var s = p.stream;
    while (loop == true) {
        var stream_top = s.top();
        if ((stream_top.equal(&PLUS) == true) or (stream_top.equal(&MINUS) == true)) {
            var op = if (stream_top.equal(&PLUS)) NodeKind.Add else NodeKind.Sub;
            s.advance();
            var rhs = try mul(p);
            var expr_node = if (op == NodeKind.Add) try new_add(p, lhs, rhs, stream_top) else try new_sub(p, lhs, rhs, stream_top);
            lhs = expr_node;
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
            stream.advance();
            var rhs = try add(p);
            var rel_node = try Node.from_binary(p.alloc, NodeKind.Lt, lhs, rhs, stream_top);
            lhs = rel_node;
        } else if (stream_top.equal(&LTE)) {
            stream.advance();
            var rhs = try add(p);
            var rel_node = try Node.from_binary(p.alloc, NodeKind.Lte, lhs, rhs, stream_top);
            lhs = rel_node;
            // Optimization, we need not have a NodeKind.Gte
            // we can just switch lhs and rhs with the same Lt, Lte ops
        } else if (stream_top.equal(&GT)) {
            stream.advance();
            var rhs = try add(p);
            var rel_node = try Node.from_binary(p.alloc, NodeKind.Lt, rhs, lhs, stream_top);
            lhs = rel_node;
        } else if (stream_top.equal(&GTE)) {
            stream.advance();
            var rhs = try add(p);
            var rel_node = try Node.from_binary(p.alloc, NodeKind.Lte, rhs, lhs, stream_top);
            lhs = rel_node;
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
            stream.advance();
            var rhs = try relational(p);
            var rel_node = try Node.from_binary(p.alloc, op, lhs, rhs, op_tok);
            lhs = rel_node;
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
        stream.advance();
        var rhs = try assign(p);
        var assign_node = try Node.from_binary(p.alloc, NodeKind.Assign, lhs, rhs, top);
        return assign_node;
    }
    return lhs;
}

fn expr(p: *ParseContext) !*Node {
    return assign(p);
}

// expr-stmt = expr? ;
fn expr_statement(p: *ParseContext) !*Node {
    var s = p.stream;
    var top = s.top();
    if (top.equal(&SEMICOLON)) {
        s.advance();
        var empty_stmt = Node.from_block(p.alloc, null, top);
        return empty_stmt;
    }
    var lhs = try expr(p);
    var expr_node = try Node.from_expr_stmt(p.alloc, lhs, top);
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
        stream.advance();
        var return_expr = try expr(p);
        stream.skip(&SEMICOLON);
        var return_node = try Node.from_unary(p.alloc, NodeKind.Ret, return_expr, stream_top);
        return return_node;
    } else if (stream_top.equal(&IF)) {
        stream.advance();
        stream.skip(&LPAREN);
        var if_node: ?*Node = null;
        var if_condition = try expr(p);
        stream.skip(&RPAREN);
        var then_stmt = try stmt(p);
        if (stream.top().equal(&ELSE)) {
            stream.advance();
            var else_stmt = try stmt(p);
            if_node = try Node.from_if_stmt(p.alloc, if_condition, then_stmt, else_stmt, stream_top);
        } else {
            if_node = try Node.from_if_stmt(p.alloc, if_condition, then_stmt, null, stream_top);
        }
        return if_node.?;
    } else if (stream_top.equal(&FOR)) {
        stream.advance();
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
        var for_node = try Node.from_for(p.alloc, for_init, for_cond, for_inc, for_then, stream_top);
        return for_node;
    } else if (stream_top.equal(&WHILE)) {
        stream.advance();
        stream.skip(&LPAREN);
        var while_cond = try expr(p);
        stream.skip(&RPAREN);
        var while_then = try stmt(p);
        var while_node = try Node.from_for(p.alloc, null, while_cond, null, while_then, stream_top);
        return while_node;
    } else if (stream_top.equal(&LBRACE)) {
        stream.advance();
        return compound_statement(p);
    } else {
        return expr_statement(p);
    }
}

// (declaration | stmt)* }
fn compound_statement(p: *ParseContext) anyerror!*Node {
    var first_stmt: ?*Node = null;
    var it = first_stmt;
    var stream = p.stream;
    var s_top = stream.top();
    while (stream.top().equal(&RBRACE) == false) {
        var statement = if (Type.isTypename(stream.top())) try declaration(p) else try stmt(p);
        add_type(p.alloc, statement);
        if (first_stmt == null) {
            first_stmt = statement;
            it = statement;
        } else {
            it.?.next = statement;
            it = statement;
        }
    }
    var compound_stmt_node = try Node.from_block(p.alloc, first_stmt, s_top);
    stream.advance();
    return compound_stmt_node;
}

//declarator type-suffix "{" compound_stmt
pub fn function(p: *ParseContext, return_typ: *Type) !void {
    var typ = try declarator(p, return_typ);
    // p.locals must always be empty before this
    var param_vars = VariableList.init(p.alloc);
    try create_param_lvars(p.alloc, typ.params, &param_vars);
    for (param_vars.items) |param| {
        try p.locals.insert(0, param);
    }

    p.stream.skip(&LBRACE);
    var fn_statements = try compound_statement(p);

    var f = Function{ .stack_size = 0, .name = Token.get_ident(typ.tok), .fnbody = fn_statements, .locals = p.locals, .params = param_vars };
    try p.globals.insert(0, Obj{ .kind = ObjKind.Function, .payload = ObjPayload{ .function = f } });
}

pub fn global_variable(p: *ParseContext, base_type: *Type) !void {
    var variables = VariableList.init(p.alloc);
    var s = p.stream;
    while (s.consume(&SEMICOLON) != true) {
        if (variables.items.len > 0) {
            s.skip(&COMMA);
        }
        var variable_type = try declarator(p, base_type);
        try variables.insert(0, try Variable.globalVar(p.alloc, variable_type.tok.get_ident(), variable_type));
    }
    for (variables.items) |v| {
        try p.globals.insert(0, Obj{ .kind = ObjKind.Variable, .payload = ObjPayload{ .variable = v } });
    }
    _ = variables.moveToUnmanaged();
}
pub fn parse(s: *Stream, alloc: Allocator) !std.ArrayList(Obj) {
    // Should `locals` also be ObjList or can it just be a VariableList (ie) can C have nested structs and functions ?
    var parse_context = ParseContext{ .stream = s, .alloc = alloc, .locals = VariableList.init(alloc), .globals = ObjList.init(alloc) };
    while (!s.is_eof()) {
        var decl = declspec(&parse_context);
        if (lookahead_is_function(&parse_context)) {
            try function(&parse_context, decl);
            //reset list of local variables for each function
            _ = parse_context.locals.moveToUnmanaged();
        } else {
            try global_variable(&parse_context, decl);
        }
    }
    return parse_context.globals;
}

const fn_prologue =
    \\ sub sp, sp, #16
    \\ stp x29, x30, [sp]
    \\ mov x29, sp
;
// each fn's body will sub sp based on the # of local vars after `fn_prologue`
// similarly each fn will add sp to its original location based on the # of local vars after `fn_epilogue`
const fn_epilogue =
    \\ sub sp, x29, -16
    \\ ldp x29, x30, [x29]
    \\ ret
;
// At the end of this, X0 will have the addr
// of the variable being loaded
fn gen_addr(node: *Node, ctx: *CodegenContext) !void {
    switch (node.kind) {
        NodeKind.Var => {
            var variable = node.variable.?;
            if (variable.is_local) {
                var offset = node.variable.?.offset;
                try stdout.writer().print(";; variable {s} at offset {}\n", .{ node.variable.?.name, offset });
                try stdout.writer().print("add x0, x29, #-{}\n", .{offset});
            } else {
                try stdout.writer().print(";; loading global variable {s}'s addr at x0\n", .{variable.name});
                try stdout.writer().print("adrp x0, {s}@PAGE\n", .{variable.name});
                try stdout.writer().print("add x0, x0, {s}@PAGEOFF\n", .{variable.name});
            }
        },
        // I don't know how we can pass a node of type `deref` to gen_addr yet.
        // Feels wrong, but maybe, it is for something like **x or ***x and so on ?
        NodeKind.Deref => {
            try gen_expr(node.lhs.?, ctx);
        },
        else => {
            panic("Not an lvalue {?}", .{node});
        },
    }
}
var depth: u32 = 0;
var args_regs = [_][]const u8{ "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7" };
var args_regs_32 = [_][]const u8{ "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7" };

// ** Code-Generation ** //
fn push(ctx: *CodegenContext) !void {
    ctx.stack_depth += 1;
    try stdout.writer().print("str X0, [sp, -16]\n", .{});
    try stdout.writer().print("sub sp, sp, #16\n", .{});
}
fn pop(reg: []const u8, ctx: *CodegenContext) !void {
    ctx.stack_depth -= 1;
    try stdout.writer().print("ldr {s}, [sp]\n", .{reg});
    try stdout.writer().print("add sp, sp, #16\n", .{});
}

// from an address in x0, load a value into x0
fn load(ty: *const Type) anyerror!void {
    if (ty.kind == TypeKind.Array) {
        // If it is an array, do not attempt to load a value to the
        // register because in general we can't load an entire array to a
        // register. As a result, the result of an evaluation of an array
        // becomes not the array itself but the address of the array.
        // This is where "array is automatically converted to a pointer to
        // the first element of the array in C" occurs.
        return;
    }
    if (ty.size == 1) {
        try stdout.writer().print("ldr w0, [x0]\n", .{});
    } else {
        try stdout.writer().print("ldr x0, [x0]\n", .{});
    }
}

// Store x0 to an address that the stack top is pointing to.
fn store(reg: []const u8, ty: *const Type, ctx: *CodegenContext) anyerror!void {
    try pop(reg, ctx);
    if (ty.size == 1) {
        try stdout.writer().print("strb w0, [{s}]\n", .{reg});
    } else {
        try stdout.writer().print("str x0, [{s}]\n", .{reg});
    }
}
// Code generation
// Depending on the kind of node, the state of registers might
// vary
pub fn gen_expr(node: *Node, ctx: *CodegenContext) anyerror!void {
    if (node.kind == NodeKind.Num) {
        try stdout.writer().print(";; loading immediate {} at\n", .{node.val});
        try stdout.writer().print("mov X0, {}\n", .{node.val});
    } else if (node.kind == NodeKind.Var) {
        try gen_addr(node, ctx); //x0 has address of variable
        try load(node.n_type.?);
    } else if (node.kind == NodeKind.Addr) {
        try gen_addr(node.lhs.?, ctx); // node should be something like a var whose address we can obtain
    } else if (node.kind == NodeKind.Deref) {
        try gen_expr(node.lhs.?, ctx); // x0 now should have an address (from a variable ideally)
        try load(node.n_type.?);
        //TODO: This should be neg, not Unary
    } else if (node.kind == NodeKind.Unary) {
        try gen_expr(node.lhs.?, ctx);
        try stdout.writer().print("neg x0, x0\n", .{});
    } else if (node.kind == NodeKind.Assign) {
        try gen_addr(node.lhs.?, ctx); //x0 has addr of variable
        try push(ctx); //push x0 into stack
        try gen_expr(node.rhs.?, ctx); // x0 now has value
        try store(span("x2"), node.n_type.?, ctx);
    } else if (node.kind == NodeKind.Funcall) {
        var args_len: usize = 0;
        for (node.fn_args.?.items) |arg| {
            try gen_expr(arg, ctx);
            try push(ctx);
            args_len += 1;
        }
        while (args_len > 0) : (args_len -= 1) {
            try pop(args_regs[args_len - 1], ctx);
            if (args_len == 0) {
                break;
            }
        }
        // I do not know how to solve this nicely yet, but
        // clang seems to emit symbols with a leading `_` for C fns
        // eg. if you have a fn int badaxe() in a C file that is compiled into
        // a relocatable object, the object file will have the symbol _badaxe
        // Until we figure out how we can fix it , lets just prefix all
        // function names with an `_`
        try stdout.writer().print("bl _{s}\n", .{node.fn_name});
    } else {
        // Idea, gen_expr, returns which register the end value of that expr is in
        // we can then use this as an input into the subsequent Add, Sub, Mul, Div
        // instructions, instead of pushing and popping from stack
        try gen_expr(node.rhs.?, ctx);
        try push(ctx);
        try gen_expr(node.lhs.?, ctx);
        try pop(span("x1"), ctx);
        // Idea: Add should be able to take a reg (x0..x18) as input and generate
        // instructions as per that
        // for each instruction, we keep track of which x register is free and then emit instructions
        // into that reg and then cross out that register as occupied
        switch (node.kind) {
            NodeKind.Add => {
                try stdout.writer().print("add x0, x1, x0\n", .{});
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

fn gen_stmt(n: *Node, ctx: *CodegenContext) !void {
    var stdout_writer = stdout.writer();

    switch (n.kind) {
        NodeKind.If => {
            try gen_expr(n.cond.?, ctx);
            // x0 is 1. when `cond` holds
            // if x0 is != 1, then cond => false hence
            // we jump to else
            var branch_id = update_branch_count(ctx);
            try stdout_writer.print("cmp x0, #0\n", .{});
            //TODO: Add fn name as prefix to branch tag
            try stdout_writer.print("b.eq else_label_{}\n", .{branch_id});

            try gen_stmt(n.then.?, ctx);

            try stdout_writer.print("b.eq else_label_{}\n", .{branch_id});

            try stdout_writer.print("b end_label_{}\n", .{branch_id});
            try stdout_writer.print("else_label_{}:\n", .{branch_id});
            if (n.els) |else_stmt| {
                try gen_stmt(else_stmt, ctx);
            }
            try stdout_writer.print("end_label_{}:\n", .{branch_id});
        },
        NodeKind.For => {
            var branch_id = update_branch_count(ctx);
            if (n.init) |for_init| {
                try gen_stmt(for_init, ctx);
            }
            try stdout_writer.print("for_label{}:\n", .{branch_id});
            if (n.cond) |for_cond| {
                try gen_expr(for_cond, ctx);
                try stdout_writer.print("cmp x0, #0\n", .{});
                try stdout_writer.print("b.eq for_end_label{}\n", .{branch_id});
            }
            try gen_stmt(n.then.?, ctx);
            if (n.inc) |inc| {
                try gen_expr(inc, ctx);
            }
            try stdout_writer.print("b for_label{}\n", .{branch_id});
            try stdout_writer.print("for_end_label{}:\n", .{branch_id});
        },
        NodeKind.Block => {
            var maybe_it = n.body;
            while (maybe_it) |it| {
                try gen_stmt(it, ctx);
                maybe_it = it.next;
            }
        },
        NodeKind.ExprStmt => {
            try gen_expr(n.lhs.?, ctx);
            return;
        },
        NodeKind.Ret => {
            try gen_expr(n.lhs.?, ctx);
            try stdout_writer.print("b return_label._{s}\n", .{ctx.current_func.name});
            return;
        },
        else => {
            panic("Invalid node {?}\n", .{n});
        },
    }
}
const CodegenContext = struct { current_func: Function, branch_count: u32 = 0, stack_depth: usize = 0 };
pub fn emit_text(objs: std.ArrayList(Obj)) !void {
    assign_lvar_offsets(objs);
    var stdout_writer = stdout.writer();

    try stdout_writer.print(" .globl _main\n", .{});
    try stdout_writer.print(".text\n.align 4\n", .{});

    for (objs.items) |*obj| {
        if (obj.kind != ObjKind.Function) {
            continue;
        }
        var f = obj.payload.function;
        var codegen_ctx = CodegenContext{ .current_func = f };
        try stdout_writer.print(".globl _{s}\n", .{f.name});
        try stdout_writer.print("_{s}:\n", .{f.name});

        try stdout_writer.print("{s}\n", .{fn_prologue});
        try stdout_writer.print(";; making space for local variables in stack\n", .{});
        try stdout_writer.print(";; stack space for {s} is {}\n", .{ f.name, f.stack_size });

        try stdout_writer.print("sub sp, sp, #{}\n", .{f.stack_size});
        for (f.params.items) |p, i| {
            if (p.typ.size == 1) {
                try stdout_writer.print("strb {s}, [x29, -{}]\n", .{ args_regs_32[i], p.offset });
            } else {
                try stdout_writer.print("str {s}, [x29, -{}]\n", .{ args_regs[i], p.offset });
            }
        }

        try gen_stmt(f.fnbody, &codegen_ctx);
        try stdout_writer.print("add sp, sp, #{}\n", .{f.stack_size});

        try stdout_writer.print("return_label._{s}:\n", .{f.name});
        try stdout_writer.print("{s}\n", .{fn_epilogue});
    }
}

const ParseContext = struct { stream: *Stream, alloc: Allocator, locals: VariableList, globals: std.ArrayList(Obj) };
const Function = struct { fnbody: *Node, locals: VariableList, stack_size: usize, name: []const u8, params: VariableList };

fn align_to(n: usize, al: u32) usize {
    return (n + al - 1) / al * al;
}

fn assign_lvar_offsets(objs: std.ArrayList(Obj)) void {
    for (objs.items) |*obj| {
        if (obj.kind != ObjKind.Function) {
            continue;
        }
        var f = obj.payload.function;
        var offset: usize = 0;
        for (f.locals.items) |*local| {
            offset += local.*.typ.size;
            local.*.offset = offset;
        }
        // will this change the value only inside the fn
        // or update the passed in objs
        f.stack_size = align_to(offset, 16);
        obj.* = Obj{ .kind = ObjKind.Function, .payload = ObjPayload{ .function = f } };
    }
}

fn find_local_var(ident: []const u8, locals: VariableList) ?*Variable {
    for (locals.items) |l| {
        if (std.mem.eql(u8, l.name, ident)) {
            return l;
        }
    }
    return null;
}

fn findVar(p: *ParseContext, ident: []const u8) ?*Variable {
    if (find_local_var(ident, p.locals)) |v| {
        return v;
    }
    for (p.globals.items) |g| {
        if (g.kind != ObjKind.Variable) {
            continue;
        }
        if (std.mem.eql(u8, ident, g.payload.variable.name)) {
            return g.payload.variable;
        }
    }

    return null;
}
fn update_branch_count(ctx: *CodegenContext) u32 {
    ctx.branch_count += 1;
    return ctx.branch_count;
}

fn add_type(ally: Allocator, n: *Node) void {
    if (n.n_type != null) { // already has type info , so just return
        return;
    }
    if (n.lhs) |lhs| {
        add_type(ally, lhs);
    }

    if (n.rhs) |rhs| {
        add_type(ally, rhs);
    }

    if (n.cond) |cond| {
        add_type(ally, cond);
    }

    if (n.then) |then| {
        add_type(ally, then);
    }

    if (n.els) |els| {
        add_type(ally, els);
    }

    if (n.init) |init| {
        add_type(ally, init);
    }

    if (n.inc) |inc| {
        add_type(ally, inc);
    }

    if (n.body) |body| {
        var b: ?*Node = body;
        while (b != null) : (b = b.?.next) {
            add_type(ally, b.?);
        }
    }

    if (n.fn_args) |fargs| {
        for (fargs.items) |fitem| {
            add_type(ally, fitem);
        }
    }

    switch (n.kind) {
        NodeKind.Add, NodeKind.Unary, NodeKind.Sub, NodeKind.Mul, NodeKind.Div => {
            n.n_type = n.lhs.?.n_type;
        },
        NodeKind.Eq, NodeKind.Neq, NodeKind.Lt, NodeKind.Lte, NodeKind.Num, NodeKind.Funcall => {
            n.n_type = &IntBaseType;
        },
        NodeKind.Assign => {
            if (n.lhs.?.n_type.?.kind == TypeKind.Array) {
                panic("Something of ArrayType cannot be an lvalue {?}\n", .{n});
            }
            n.n_type = n.lhs.?.n_type;
        },
        NodeKind.Var => {
            n.n_type = n.variable.?.typ;
        },
        NodeKind.Addr => {
            //TODO: The 3rd arg to pointer_to is a hack
            //as I couldn't figure out which is the right token to pass. fix it
            //when possible
            if (n.lhs.?.n_type.?.kind == TypeKind.Array) {
                n.n_type = Type.pointer_to(ally, n.lhs.?.n_type.?.base.?, n.lhs.?.n_type.?.base.?.tok) catch panic("failed to create Pointer type to {?}\n", .{n.lhs.?.n_type.?});
            }
            n.n_type = Type.pointer_to(ally, n.lhs.?.n_type.?, n.lhs.?.n_type.?.tok) catch panic("failed to create Pointer type to {?}\n", .{n.lhs.?.n_type.?});
        },
        NodeKind.Deref => {
            // we can deref either an array or a pointer
            // so don't check for `type == TypeKind.Ptr
            // but check if there is a `base` for our type
            if (n.lhs.?.n_type.?.base == null) {
                panic("invalid pointer dereference {?}\n", .{n.lhs});
            }
            n.n_type = n.lhs.?.n_type.?.base;
        },
        else => {
            // We just ignore all the other types of nodes
        },
    }
}

// declspec = "char" | "int"
// returns a `Type` from the stream
fn declspec(p: *ParseContext) *Type {
    var stream = p.stream;
    if (stream.top().equal(&CHAR)) {
        stream.advance();
        return &CharBaseType;
    }
    stream.skip(&INT);
    return &IntBaseType;
}

//declaration = declspec (declarator ("=" expr)? ("," declarator ("=" expr)?)*)? ";"
// e.g: int x;
// int x=5;
// int x=5,y;
// int x,y=4;
// int z,c;
// and also just `int` ?
fn declaration(p: *ParseContext) !*Node {
    var base_type = declspec(p); // This is the `int` part
    // Lines marked with ** are the only non-housekeeping part of the parser
    var head_node: ?*Node = null;
    var cur_node: ?*Node = head_node;
    var i: usize = 0;
    var s = p.stream;
    var dec_block_tok = s.top();
    while (s.top().equal(&SEMICOLON) == false) {
        if (i > 0) { // check for comma before every declaration after the first one
            s.skip(&COMMA);
        }
        i += 1;
        var top = s.top();
        var typ = try declarator(p, base_type);
        var identifier = try Variable.localVar(p.alloc, typ.tok.get_ident(), typ);
        try p.locals.insert(0, identifier);
        var lhs = try Node.from_ident(p.alloc, identifier, typ.tok);

        if (s.top().equal(&ASSIGN) == false) { // check if token is "="
            continue;
        }
        s.advance(); // ** consume the `=` token
        var rhs = try assign(p); // ** value of the ident
        var declaration_node = try Node.from_binary(p.alloc, NodeKind.Assign, lhs, rhs, top);
        // Why is this converted into a ExprStmt node ?
        var unary_node = try Node.from_unary(p.alloc, NodeKind.ExprStmt, declaration_node, top);
        if (cur_node == null) {
            head_node = unary_node;
            cur_node = unary_node;
        } else {
            cur_node.?.next = unary_node;
            cur_node = unary_node;
        }
    }
    s.advance();
    var declaration_block = try Node.from_block(p.alloc, head_node, dec_block_tok);
    return declaration_block;
}

////////////////
//  These are functions that parse for Type Nodes
//  not value or other types of expression nodes etc
////////////////

// declarator = "*"* ident type_suffix
fn declarator(p: *ParseContext, typ: *Type) !*Type {
    var stream = p.stream;
    var actual_type = typ;
    var top = stream.top();
    while (stream.consume(&DEREF)) {
        actual_type = try Type.pointer_to(p.alloc, actual_type, top);
    }
    top = stream.top();
    switch (top.*) {
        TokenKind.ident => {
            actual_type.tok = stream.top();
            stream.advance();
            var ty = try type_suffix(p, actual_type);
            ty.tok = top;
            return ty;
        },
        else => {
            panic("Expected Ident token for declarator, got {?}\n", .{stream.top()});
        },
    }
}
// func-params = (param (",", param)*)? ")"
fn func_params(p: *ParseContext, return_type: *Type) anyerror!*Type {
    var s = p.stream;
    var params = std.ArrayList(*Type).init(p.alloc);
    while (s.top().equal(&RPAREN) == false) {
        if (params.items.len > 0) {
            s.skip(&COMMA);
        }
        var param_type = declspec(p);
        var param = try declarator(p, param_type);
        try params.insert(0, try Type.copy_type(p.alloc, param));
    }
    s.advance(); // consume ")"

    var fn_type = try Type.func_type(p.alloc, return_type);
    fn_type.params = params;
    return fn_type;
}
// type-suffix =   "(" func-params
//               | "[" num "]"
//               | "[" num "]" type-suffix
//               | nothing
// func-params = param (, param)*
// param = declspec declarator
fn type_suffix(p: *ParseContext, return_type: *Type) anyerror!*Type {
    var s = p.stream;
    if (s.top().equal(&LPAREN)) {
        s.advance(); // consume "("
        // return try func_params(p, return_type);
        var fn_p = try func_params(p, return_type);
        return fn_p;
    } else if (s.top().equal(&LSQ_BRACK)) {
        var tok = s.top();
        s.advance();
        var siz = s.top().getNumber();
        if (siz < 0) {
            panic("cannot have array size {} that is < 0 at {?}\n", .{ siz, s.top() });
        }
        s.advance();
        var siz_i64 = @as(i64, siz);

        s.skip(&RSQ_BRACK);
        var next_dim_array = try type_suffix(p, return_type);
        return Type.array_of(p.alloc, next_dim_array, @bitCast(usize, siz_i64), tok);
    } else {
        return return_type;
    }
}

// create variables fn func parameters
fn create_param_lvars(alloc: Allocator, params_decl_list: ?std.ArrayList(*Type), vars_list: *VariableList) !void {
    if (params_decl_list) |decl_list| {
        for (decl_list.items) |d| {
            var arg = try Variable.localVar(alloc, d.tok.get_ident(), d);
            try vars_list.insert(0, arg);
        }
    }
}

const ObjKind = enum { Function, Variable };
const ObjPayload = union { variable: *Variable, function: Function };
const Obj = struct { kind: ObjKind, payload: ObjPayload };
const ObjList = std.ArrayList(Obj);

fn emit_data(objs: ObjList) !void {
    var stdout_writer = stdout.writer();
    try stdout_writer.print(".data\n", .{});

    for (objs.items) |*obj| {
        if (obj.kind != ObjKind.Variable) {
            continue;
        }
        var v = obj.payload.variable;
        try stdout_writer.print(".globl {s}\n", .{v.name});
        try stdout_writer.print("{s}: .skip {},0 \n", .{ v.name, v.typ.size });
    }
}

pub fn codegen(objs: std.ArrayList(Obj)) !void {
    assign_lvar_offsets(objs);
    try emit_data(objs);
    try emit_text(objs);
}

//lookahead functions parse the stream for a certain construct
//and returns true if the construct has been found ahead
//WITHOUT consuming the stream
//It does so by storing the current ctx (stream idx), looking ahead for a construct
//and restoring the current ctx
fn lookahead_is_function(p: *ParseContext) bool {
    var s = p.stream;
    if (s.top().equal(&SEMICOLON)) {
        return false;
    }
    var idx = s.idx;
    var dummy = &IntBaseType;
    // we try to parse as if the token contains a
    // declarator of something and then check if it is a fn
    dummy = declarator(p, dummy) catch return false;
    p.*.stream.idx = idx;

    return dummy.kind == TypeKind.Func;
}
