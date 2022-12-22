const std = @import("std");
const Allocator = std.mem.Allocator;
const span = std.mem.span;
const panic = std.debug.panic;
const ascii = std.ascii;
const keywords = [_][]const u8{ "return", "if", "else", "for", "while" };
pub const TokenKind = enum { punct, num, eof, ident, keyword };
pub const Token = union(TokenKind) {
    const Self = @This();
    punct: struct {
        ptr: []const u8,
    },
    num: struct { val: i32 },
    eof: void,
    ident: struct {
        ptr: []const u8,
    },
    keyword: struct {
        ptr: []const u8,
    },

    pub fn isKeyword(word: []u8) bool {
        for (keywords) |keyword| {
            if (std.mem.eql(u8, word, keyword)) {
                return true;
            }
        }
        return false;
    }
    pub fn equal(self: *Self, other: *const Token) bool {
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
            TokenKind.keyword => |v| {
                var struct_other = @field(other, "keyword");
                var keyword = @field(struct_other, "ptr");
                return std.mem.eql(u8, v.ptr, keyword);
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

    fn equal_idents(self: *Self, other: *const Token) bool {
        var struct_self = @field(self, "ident");
        var struct_other = @field(other, "ident");
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
                try std.fmt.format(out_stream, "TokenKind.num :D {}\n", .{v.val});
            },
            TokenKind.punct => |v| {
                try std.fmt.format(out_stream, "TokenKind.punct :D {s}\n", .{v.ptr});
            },
            TokenKind.eof => {
                try std.fmt.format(out_stream, "TokenKind.eof :D \n", .{});
            },
            TokenKind.ident => |v| {
                try std.fmt.format(out_stream, "TokenKind.punct :D {s}\n", .{v.ptr});
            },
        }
    }
};
pub const TokenList = std.ArrayList(Token);

pub const Stream = struct {
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
            panic("Expected {?}, found EOF", .{t});
        }
        var top_token = self.top();
        if (top_token.equal(t) == true) {
            self.consume();
        } else {
            panic(" stream.skip Expected token {?} got {?}\n", .{ t, top_token });
        }
    }
};
// Takes a program string and a TokenList as input and
// populates the TokenList with tokens from the program string
// Do not use the TokenList directly. Use the Stream instead.
pub fn tokenize(stream_p: *[*:0]u8, list: *TokenList) !void {
    var stream = stream_p.*;
    while (stream[0] != 0) {
        if (ascii.isSpace(stream[0]) == true) {
            stream += 1;
            continue;
            // Numbers
        } else if (ascii.isDigit(stream[0]) == true) {
            //notice that we are modifying stream here
            var number = strtol(&stream);
            try list.append(Token{ .num = .{ .val = number.? } });
            // Identifiers and Keywords
        } else if (isIdent1(stream[0])) {
            var nextIdx: usize = 1;
            while (isIdent2(stream[0 + nextIdx])) {
                nextIdx += 1;
            }
            // In chibicc, Ident Tokens were converted to keywords *AFTER*
            // the entire source was parsed. I don't know which method is better
            // of if there is a clear advantage to converting idents to keywords
            // later once all tokens are scanned
            var t = if (Token.isKeyword(stream[0..nextIdx]))
                Token{ .keyword = .{ .ptr = stream[0..nextIdx] } }
            else
                Token{ .ident = .{ .ptr = stream[0..nextIdx] } };
            try list.append(t);
            stream += nextIdx;
            // Operators
        } else if (readPunct(span(stream)) > 0) {
            var punct_len = readPunct(span(stream));
            try list.append(Token{ .punct = .{ .ptr = stream[0..punct_len] } });
            stream += punct_len;
        } else {
            panic("Invalid token {c} from {s}\n", .{ stream[0], stream });
        }
    }
    try list.append(Token.eof);
}

// A slight variation of the strtol func in C std lib.
// attempts to read a integer starting with + or -  from
// a pointer to a string, passed as an input
// If a number is present, modifies the pointer to the string
// to point to the first location after the number
// E;g. if yp = "+123abcd", after strtol(&yp), yp[0] is now 'a'
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

pub fn text_to_stream(text: *[*:0]u8, alloc: Allocator) !Stream {
    var tlist = TokenList.init(alloc);
    try tokenize(text, &tlist);
    var stream = Stream.init_stream(tlist);
    return stream;
}

fn readPunct(s: []const u8) usize {
    if (starts_with(s, span("==")) or starts_with(s, span("!=")) or
        starts_with(s, span("<=")) or starts_with(s, span(">=")))
    {
        return 2;
    }
    if (ascii.isPunct(s[0])) {
        return 1;
    } else {
        return 0;
    }
}

fn starts_with(q: []const u8, p: []const u8) bool {
    return std.mem.startsWith(u8, q, p);
}

fn isIdent1(c: u8) bool {
    return ('a' <= c and c <= 'z') or ('A' <= c and c <= 'Z') or c == '_';
}

fn isIdent2(c: u8) bool {
    return isIdent1(c) or '0' <= c and c <= '9';
}
