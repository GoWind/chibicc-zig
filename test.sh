#!/bin/bash
zig build || exit
./zig-out/bin/chibicc-zig "3" "+" "4" > tmp.s || exit
as -arch arm64 -o tmp.asm tmp.s
ld -o tmp tmp.asm -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64
./tmp
actual="$?"

if [ "$actual" = "7" ]; then
 echo "7 => $actual"
else
  echo "7 expected, but got $actual"
  exit 1
fi

echo OK
