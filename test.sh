#!/bin/bash
zig build || exit
./zig-out/bin/chibicc-zig "(33+ 44) - 7-10" > tmp.s || exit
as -arch arm64 -o tmp.asm tmp.s
ld -o tmp tmp.asm -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64
./tmp
actual="$?"

if [ "$actual" = "60" ]; then
 echo "60 => $actual"
else
  echo "7 expected, but got $actual"
  exit 1
fi

echo OK
