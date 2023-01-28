const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const panic = std.debug.panic;
const span = std.mem.span;
pub const TypeKind = enum { Char, Int, Ptr, Func, Array, String };

// Type information about variables
pub const Type = struct {
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
    pub fn pointer_to(alloc: Allocator, base: *const Type, tok: *const Token) !*Type {
        var derived = try alloc.create(Type);
        derived.kind = TypeKind.Ptr;
        derived.base = base;
        derived.tok = tok;
        derived.size = @as(usize, 8);
        return derived;
    }

    pub fn copy_type(alloc: Allocator, original: *const Type) !*Type {
        var copy = try alloc.create(Type);
        copy.* = original.*;
        return copy;
    }

    pub fn array_of(alloc: Allocator, base: *const Type, len: usize, tok: *const Token) !*Type {
        var array_type = try alloc.create(Type);
        array_type.* = .{ .base = base, .size = base.size * len, .kind = TypeKind.Array, .array_len = len, .tok = tok };
        return array_type;
    }

    pub fn isTypename(tok: *const Token) bool {
        return tok.equal(&CHAR) or tok.equal(&INT);
    }

    // is_integer in chibicc
    pub fn isIntegerLikeType(self: *const Self) bool {
        return self.kind == TypeKind.Int or self.kind == TypeKind.Char;
    }
};

const Keywords = [_][]const u8{ "return", "if", "else", "for", "while", "int", "sizeof", "char" };

pub const TokenKind = enum { Punct, Num, Eof, Ident, Keyword, StringLiteral };
pub const Token = union(TokenKind) {
    const Self = @This();
    Punct: struct {
        ptr: []const u8,
    },
    Num: struct { val: i32 },
    Eof: void,
    Ident: struct {
        ptr: []const u8,
    },
    Keyword: struct {
        ptr: []const u8,
    },
    StringLiteral: struct {
        ptr: []const u8,
    },

    pub fn isKeyword(word: []u8) bool {
        for (Keywords) |Keyword| {
            if (std.mem.eql(u8, word, Keyword)) {
                return true;
            }
        }
        return false;
    }
    pub fn equal(self: *const Self, other: *const Token) bool {
        if (std.mem.eql(u8, @tagName(self.*), @tagName(other.*)) == false) {
            return false;
        }

        switch (self.*) {
            TokenKind.Num => {
                return self.equal_nums(other);
            },
            TokenKind.Punct => {
                return self.equal_Puncts(other);
            },
            TokenKind.Keyword => |v| {
                var struct_other = @field(other, "Keyword");
                var Keyword = @field(struct_other, "ptr");
                return std.mem.eql(u8, v.ptr, Keyword);
            },
            else => {
                panic("We shouldn't be here", .{});
            },
        }
    }

    fn equal_Puncts(self: *const Self, other: *const Token) bool {
        var struct_self = @field(self, "Punct");
        var struct_other = @field(other, "Punct");
        var ptr_self = @field(struct_self, "ptr");
        var ptr_other = @field(struct_other, "ptr");
        return std.mem.eql(u8, ptr_self, ptr_other);
    }

    fn equal_Idents(self: *const Self, other: *const Token) bool {
        var struct_self = @field(self, "Ident");
        var struct_other = @field(other, "Ident");
        var ptr_self = @field(struct_self, "ptr");
        var ptr_other = @field(struct_other, "ptr");
        return std.mem.eql(u8, ptr_self, ptr_other);
    }

    fn equal_nums(self: *const Self, other: *const Token) bool {
        var struct_self = @field(self, "Num");
        var struct_other = @field(other, "Num");
        var val_self = @field(struct_self, "val");
        var val_other = @field(struct_other, "val");
        return val_self == val_other;
    }

    pub fn get_ident(tok: *const Self) []const u8 {
        switch (tok.*) {
            TokenKind.Ident => |v| {
                return v.ptr;
            },
            else => {
                panic("expected .Ident token, got {?}\n", .{tok});
            },
        }
    }

    pub fn getNumber(tok: *const Self) i32 {
        switch (tok.*) {
            TokenKind.Num => |v| {
                return v.val;
            },
            else => {
                panic("expected num token, got {?}\n", .{tok});
            },
        }
    }

    pub fn format(self: Self, comptime _: []const u8, _: std.fmt.FormatOptions, out_stream: anytype) !void {
        switch (self) {
            TokenKind.Num => |v| {
                try std.fmt.format(out_stream, "TokenKind.Num {}\n", .{v.val});
            },
            TokenKind.Punct => |v| {
                try std.fmt.format(out_stream, "TokenKind.Punct: '{s}'\n", .{v.ptr});
            },
            TokenKind.Eof => {
                try std.fmt.format(out_stream, "TokenKind.eof \n", .{});
            },
            TokenKind.Ident => |v| {
                try std.fmt.format(out_stream, "TokenKind.Ident : '{s}'\n", .{v.ptr});
            },
            TokenKind.Keyword => |v| {
                try std.fmt.format(out_stream, "TokenKind.Keyword {s}\n", .{v.ptr});
            },
            TokenKind.StringLiteral => |v| {
                try std.fmt.format(out_stream, "TokenKind.StringLiteral {s}\n", .{v.ptr});
            },
        }
    }
};
pub const TokenList = std.ArrayList(Token);
const all_valid_chars = "()/*+-==!=<<=>>=;={}ifelse()forwhile&*,[]";
pub const LEFT_PAREN = Token{ .Punct = .{ .ptr = span(all_valid_chars[0..1]) } };
pub const RIGHT_PAREN = Token{ .Punct = .{ .ptr = span(all_valid_chars[1..2]) } };
pub const DIV = Token{ .Punct = .{ .ptr = span(all_valid_chars[2..3]) } };
pub const MUL = Token{ .Punct = .{ .ptr = span(all_valid_chars[3..4]) } };
pub const PLUS = Token{ .Punct = .{ .ptr = span(all_valid_chars[4..5]) } };
pub const MINUS = Token{ .Punct = .{ .ptr = span(all_valid_chars[5..6]) } };
pub const EQEQ = Token{ .Punct = .{ .ptr = span(all_valid_chars[6..8]) } };
pub const NOTEQ = Token{ .Punct = .{ .ptr = span(all_valid_chars[8..10]) } };
pub const LT = Token{ .Punct = .{ .ptr = span(all_valid_chars[10..11]) } };
pub const LTE = Token{ .Punct = .{ .ptr = span(all_valid_chars[11..13]) } };
pub const GT = Token{ .Punct = .{ .ptr = span(all_valid_chars[13..14]) } };
pub const GTE = Token{ .Punct = .{ .ptr = span(all_valid_chars[14..16]) } };
pub const SEMICOLON = Token{ .Punct = .{ .ptr = span(all_valid_chars[16..17]) } };
pub const ASSIGN = Token{ .Punct = .{ .ptr = span(all_valid_chars[17..18]) } };
pub const LBRACE = Token{ .Punct = .{ .ptr = span(all_valid_chars[18..19]) } };
pub const RBRACE = Token{ .Punct = .{ .ptr = span(all_valid_chars[19..20]) } };
pub const IF = Token{ .Keyword = .{ .ptr = span(all_valid_chars[20..22]) } };
pub const ELSE = Token{ .Keyword = .{ .ptr = span(all_valid_chars[22..26]) } };
pub const LPAREN = Token{ .Punct = .{ .ptr = span(all_valid_chars[26..27]) } };
pub const RPAREN = Token{ .Punct = .{ .ptr = span(all_valid_chars[27..28]) } };
pub const FOR = Token{ .Keyword = .{ .ptr = span(all_valid_chars[28..31]) } };
pub const WHILE = Token{ .Keyword = .{ .ptr = span(all_valid_chars[31..36]) } };
pub const ADDR = Token{ .Punct = .{ .ptr = span(all_valid_chars[36..37]) } };
pub const DEREF = Token{ .Punct = .{ .ptr = span(all_valid_chars[37..38]) } };
pub const COMMA = Token{ .Punct = .{ .ptr = span(all_valid_chars[38..39]) } };
pub const LSQ_BRACK = Token{ .Punct = .{ .ptr = span(all_valid_chars[39..40]) } };
pub const RSQ_BRACK = Token{ .Punct = .{ .ptr = span(all_valid_chars[40..41]) } };

pub const RETURN = Token{ .Keyword = .{ .ptr = span("return") } };
pub const INT = Token{ .Keyword = .{ .ptr = span("int") } };
pub const CHAR = Token{ .Keyword = .{ .ptr = span("char") } };

pub const SIZEOF = Token{ .Keyword = .{ .ptr = span("sizeof") } };
