#!/bin/bash
assert() {
  expected="$1"
  input="$2"
  zig build || exit
  ./zig-out/bin/chibicc-zig "$input" > tmp.s || exit
  as -arch arm64 -o tmp.asm tmp.s
  ld -o tmp tmp.asm -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64
  ./tmp
  actual="$?"

  if [ "$actual" = "$expected" ]; then
    echo "$input => $actual"
  else
    echo "$input => $expected expected, but got $actual"
    exit 1
  fi
}

assert 0 0
assert 42 42

echo OK
