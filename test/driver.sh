#!/bin/bash
echo "Running driver file"
tmp=`mktemp -d /tmp/chibicc-test-XXXXXX`
trap 'rm -rf $tmp' INT TERM HUP EXIT
echo > $tmp/empty.c

check() {
    if [ $? -eq 0 ]; then
        echo "testing $1 ... passed"
    else
        echo "testing $1 ... failed"
        exit 1
    fi
}

# -o
rm -f $tmp/out
./zig-out/bin/chibicc-zig -o $tmp/out $tmp/empty.c
[ -f $tmp/out ]
check -o

# --help
./zig-out/bin/chibicc-zig --help 2>&1 | grep -q chibicc-zig
check --help

echo OK
