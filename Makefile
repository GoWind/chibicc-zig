CFLAGS=-std=c11 -g -fno-common

SRCS=$(wildcard src/*.zig)
TEST_SRCS=$(wildcard test/*.c)
TESTS=$(TEST_SRCS:.c=.exe)


./zig-out/bin/chibicc-zig: $(SRCS)
	echo $(ZIG_BIN_DIR)
	zig build	

test/%.exe: ./zig-out/bin/chibicc-zig test/%.c
	$(CC) -o- -E -P -C test/$*.c | ./zig-out/bin/chibicc-zig -o test/$*.s -
	# -x <language>           Treat subsequent input files as having type <language>
	$(CC) -o $@ test/$*.s -xc test/common

test: $(TESTS)
	for i in $^; do echo $$i; ./$$i || exit 1; echo; done
	test/driver.sh

clean:
	rm -rf chibicc-zig tmp* $(TESTS) test/*.s test/*.exe

.PHONY: test clean

