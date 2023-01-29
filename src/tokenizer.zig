const std = @import("std");
const data = @import("data.zig");
const Token = data.Token;
const TokenList = data.TokenList;
const TokenKind = data.TokenKind;
const Allocator = std.mem.Allocator;
const span = std.mem.span;
const panic = std.debug.panic;
const ascii = std.ascii;
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

    pub fn advance(self: *Self) void {
        self.idx += 1;
    }

    pub fn is_eof(self: *Self) bool {
        if (self.idx >= self.ts.items.len) {
            return true;
        }
        switch (self.ts.items[self.idx]) {
            TokenKind.Eof => {
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
            self.advance();
        } else {
            panic(" stream.skip Expected token {?} got {?} at idx {}\n", .{ t, top_token, self.idx });
        }
    }

    pub fn consume(self: *Self, t: *const Token) bool {
        if (self.is_eof()) {
            panic("found EOF, expected {?}\n", .{t});
        }
        var top_token = self.top();
        if (top_token.equal(t)) {
            self.advance();
            return true;
        }
        return false;
    }

    pub fn next(self: *Self) ?*Token {
        var next_id = self.idx + 1;
        if (next_id >= self.ts.items.len) {
            return null;
        }
        return &self.ts.items[next_id];
    }
};
// Takes a program string and a TokenList as input and
// populates the TokenList with tokens from the program string
// Do not use the TokenList directly. Use the Stream instead.
pub fn tokenize(stream_p: *[*:0]u8, list: *TokenList, alloc: Allocator) !void {
    var stream = stream_p.*;
    while (stream[0] != 0) {
        if (ascii.isSpace(stream[0]) == true) {
            stream += 1;
            continue;
            // Numbers
        } else if (ascii.isDigit(stream[0]) == true) {
            // Notice that we are mutating `stream` here
            var number = strtol(&stream);
            try list.append(Token{ .Num = .{ .val = number.? } });
            // Identifiers and Keywords
        } else if (stream[0] == '"') {
            var literal = try readStringLiteral(&stream);
            var buffer = try alloc.alloc(u8, literal.len);
            var skippedLiteral = skipEscapedChars(literal, buffer);
            try list.append(Token{ .StringLiteral = .{ .ptr = skippedLiteral } });
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
                Token{ .Keyword = .{ .ptr = stream[0..nextIdx] } }
            else
                Token{ .Ident = .{ .ptr = stream[0..nextIdx] } };
            try list.append(t);
            stream += nextIdx;
        } else if (readPunct(span(stream)) > 0) {
            var punct_len = readPunct(span(stream));
            try list.append(Token{ .Punct = .{ .ptr = stream[0..punct_len] } });
            stream += punct_len;
        } else {
            panic("Invalid token {c} from {s}\n", .{ stream[0], stream });
        }
    }
    try list.append(Token.Eof);
}

// A slight variation of the strtol func in C std lib.
// attempts to read a integer starting with + or -  from
// a pointer to a string, passed as an input
// If a number is present, modifies the pointer to the string
// to point to the first location after the number
// E;g. if yp = "+123abcd", after strtol(&yp), yp[0] is now 'a'
fn strtol(yp: *[*:0]const u8) ?i32 {
    var y = yp.*; // y is now a C string
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
// start is the index of the first "
fn readStringLiteral(sp: *[*:0]const u8) ![]const u8 {
    var start = sp.*;
    const s = start + 1;
    var idx: usize = 0;
    while (s[idx] != '"') : (idx += 1) {
        if (s[idx] == '\n' or s[idx] == 0) {
            std.debug.panic("Unclosed literal newline or null char at {s}\n", .{s[idx - 2 .. idx]});
        }
    }
    //idx now points to the trailing "
    sp.* = s + idx + 1;
    return s[0..idx];
}
pub fn text_to_stream(text: *[*:0]u8, alloc: Allocator) !Stream {
    var tlist = TokenList.init(alloc);
    try tokenize(text, &tlist, alloc);
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

fn readEscapedChar(c: u8) u8 {
    var escaped_char: u8 = switch (c) {
        'a' => 0x7,
        'b' => 0x8,
        't' => '\t',
        'n' => '\n',
        'v' => 0x0b,
        'f' => 0x0c,
        'r' => 0x0d,
        // GNU specific extension
        'e' => 0x1b,
        else => c,
    };
    return escaped_char;
}

fn skipEscapedChars(source: []const u8, dest: []u8) []const u8 {
    var input_idx: usize = 0;
    var output_idx: usize = 0;
    while (input_idx < source.len) {
        if (source[input_idx] == '\\') {
            dest[output_idx] = readEscapedChar(source[input_idx + 1]);
            input_idx += 2;
        } else {
            dest[output_idx] = source[input_idx];
            input_idx += 1;
        }
        output_idx += 1;
    }
    return dest[0..output_idx];
}
