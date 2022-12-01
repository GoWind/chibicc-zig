#!/bin/bash
zig build || exit
assert() {
  expected="$1"
  input="$2"

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
assert 21 '5+20-4'
assert 41 ' 12 + 34 - 5 '
assert 47 '5+6*7'
assert 15 '5*(9-6)'
assert 4 '(3+5)/2'
assert 10 '-10+20'
assert 10 '- -10'
assert 10 '- - +10'

echo OK 
