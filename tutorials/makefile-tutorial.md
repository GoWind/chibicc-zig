# An introduction to Makefiles
Let's use the Makefile from the project to illustrate how Make works
```
CFLAGS=-std=c11 -g -fno-common

SRCS=$(wildcard src/*.zig)
TEST_SRCS=$(wildcard test/*.c)
TESTS=$(TEST_SRCS:.c=.exe)

./zig-out/bin/chibicc-zig: $(SRCS)
	zig build	

test/%.exe: chibicc-zig test/%.c
	$(CC) -o- -E -P -C test/$*.c | ./zig-out/bin/chibicc-zig -o test/$*.s -
	# -x <language>           Treat subsequent input files as having type <language>
	$(CC) -o $@ test/$*.s -xc test/common

test: $(TESTS)
	for i in $^; do echo $$i; ./$$i || exit 1; echo; done
	test/driver.sh

clean:
	rm -rf chibicc-zig tmp* $(TESTS) test/*.s test/*.exe

.PHONY: test clean
```

Make was designed as way to build programs out of a large number of C, C++, Fortran
or pascal files etc. If you are using a modern(ish) language, like Java, Python etc
Make might not make sense and you should just stick to the lang. specific build tools.

### Rules, targets and recipes
##### The purpose of a Makefile is to fulfill goals. 
##### A goal is a file or a file-less task, such as `clean` in the Listing above.

A Makefile consists of `rules`. 
For example
```
./zig-out/bin/chibicc-zig: $(SRCS)
    zig build
```
is a rule. Each rule has 

1. A `target` : what is being built. In our case, the target is the file `./zig-out/bin/chibicc-zig`

2. A bunch of pre-requisites. When you run `make`, it compares the timestamp of the target with all of the pre-requisites and if any of the pre-requisite's timestamp is **newer**, it then runs the `recipe` to update the target
3. A recipe: A recipe is one or more commands that know how to update/create the target from the pre-requisites. In the example above our recipe consists of running `zig build`, which builds the executable from `Zig` source files

In summary:
```
target: pre-requisites
<TAB> recipe
```
A `TAB` is how you tell `make` that this line consists of a recipe command. The `TAB` can be swapped for another character(s) by setting the `RECIPEPREFIX` environment variable with the value you want, before running `make`

A target maybe intermediate or final. For example, `chibicc-zig` is final, as in no other targets depend on it. `test/%.exe:` meanwhile is a pre-requisite for `test`. 

A final target is a **goal**. In this Makefile, the goals are: `./zig-out/bin/chibicc-zig`, `clean` and `test`.

A Makefile has a **default** goal. The **default** goal is the first, non-pattern target in the file which does not start with `.` (or if it starts with `.`, then the the target must have one or more `/` in them)

In our case, `./zig-out/bin/chibicc-zig` is the **default** goal of the `Makefile`. When you just call `make`, it runs the **default** goal

A rule like `test/%.exe` is called a `pattern-rule` (more specifically a `static pattern rule`). The target describes a pattern of files to be created (test/`*.exe`) and each exe file depends on the corresponding `test/*.c` file and `chibicc-zig` executable. For example test/arith.exe depends on test/arith.c and the chibicc-zig executable. 


Since pattern rules may apply to a number of files, within the recipe, we would like to find which exact iteration (and file) the recipe is being run for. For example, if we have 2 files: `test/arith.c` and `test/function.c`, then our pattern `test/%.exe` will have 2 files and will run the recipe twice. 
[**Automatic Variables**](https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html#Automatic-Variables), capture the name of the file in each run of the rule, 

- %: the `stem` of the rule. In our case, it is `arith`
- $@: captures the target (e.g: `test/arith.exe`)
- $*: equivalent to `%` (that is arith)
- $<: captures the first pre-requisite (./zig-out-bin/chibicc-zig), but useless in our case as we don't need it in our recipe

### Phony targets
Now let us focus on `.PHONY: test clean`

`.PHONY` targets are targets without a file. There is no `test` or `clean` file and when you run `make clean`, Make will not search its built-in list of rules [**Implicit Rules**][https://www.gnu.org/software/make/manual/html_node/Implicit-Rules.html) to see if it has a recipe for making `clean`, thus making `make` run faster. 
Also, if a target is not declared as `.PHONY` and has no pre-requisites, it is considered always upto date and is never executed. By turning `test` and `clean` `.PHONY`, we are:  ensuring that 
1. They need to be explicitly run always (using `make clean` and `make test`)
2. The recipe of phony targets are **ALWAYS** run when invoked


### Functions
Make also has built-in [functions](https://www.gnu.org/software/make/manual/html_node/Functions.html), such as `wildcard`.


In the line `TEST_SRCS=$(wildcard test/*.c)`, we are creating a variable `TEST_SRCS`. The value of this variable is the output of the `wildcard` function. This function takes a pattern as input (`test/*.c`) and returns a list of path(s) of all filenames that match this pattern.
`
### Substitution Reference
The line `TESTS=$(TEST_SRCS:.c=.exe)` uses a feature called [Substitution Reference](https://www.gnu.org/software/make/manual/html_node/Substitution-Refs.html). It basically tells make to create a new variable `TESTS` from `TEST_SRCS` with the suffix of each value in `TEST_SRCS` replaced with ``.exe` 
#### Notes (skip this section for now, more for my own reference)
[Static pattern rules vs multiple targets rules](https://www.gnu.org/software/make/manual/html_node/Static-Pattern.html)
Multiple target rules: Multiple targets from **SAME** set of pre-requisite.

Static pattern rule: Different targets, each with a different set of pre-requisites

[Static pattern rules vs implicit rules](https://www.gnu.org/software/make/manual/html_node/Static-versus-Implicit.html)

The difference is in how make decides when the rule applies.
make will apply a implicit rule for a target matching a pattern, if and only if there are no other rules that apply to the target's pattern and if and only if it can find all of the target's pre-requisites

In contrast, a pattern rule will apply to all of the targets that match the rule

