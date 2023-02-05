const std = @import("std");
const ascii = std.ascii;
const span = std.mem.span;
const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;
const stdout = std.io.getStdOut();
const tokenizer = @import("./tokenizer.zig");
const codegen = @import("./codegen.zig");
const Node = codegen.Node;
const POSIX_C_SOURCE = "200809L";
const panic = std.debug.panic;

fn debug_token_stream(s: *tokenizer.Stream) void {
    while (s.is_eof() == false) {
        var top = s.top();
        std.debug.print("{?}\n", .{top});
        s.advance();
    }
    s.*.idx = 0;
}
pub fn main() anyerror!void {
    var argv = std.os.argv;
    if (argv.len < 2) {
        panic("must have atleast 1 arg", .{});
    }
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();
    var prog_args = Args{ .input_file = null, .output_file = null };
    try parseArgs(std.os.argv, &prog_args);
    if (prog_args.input_file == null) {
        panic("No input file provided\n", .{});
    }

    var text = try tokenizer.readFile(allocator, prog_args.input_file.?);
    var token_stream = try tokenizer.text_to_stream(@ptrCast(*[*:0]u8, &text), allocator);
    var top_node = try codegen.parse(&token_stream, allocator);
    try codegen.codegen(top_node, prog_args.output_file.?);
}

const Args = struct { input_file: ?[]const u8, output_file: ?[]const u8 };
fn usage(status: u8) !void {
    try stdout.writer().print("chibicc-zig [-o <path>] <file>\n", .{});
    std.os.exit(status);
}

fn parseArgs(prog_args: [][*:0]const u8, parsed_args: *Args) !void {
    var idx: usize = 1;
    while (idx < prog_args.len) : (idx += 1) {
        var arg = span(prog_args[idx]);
        if (std.mem.eql(u8, arg, "--help")) {
            try usage(0);
        } else if (std.mem.eql(u8, arg, "-o")) {
            if ((idx + 1) < prog_args.len) {
                parsed_args.*.output_file = span(prog_args[idx + 1]);
            }
        } else if (std.mem.startsWith(u8, arg, "-o")) {
            parsed_args.*.output_file = arg[2..];
        } else if (std.mem.startsWith(u8, arg, "-") and !std.mem.eql(u8, arg, "-")) {
            panic("unknown argument {s}\n", .{arg});
        } else {
            parsed_args.*.input_file = span(arg);
        }
    }
}
