# Synthetic PE32+ import fixtures

Run `tests/compat_fixtures/build.sh` on ARM64 macOS with native LLVM and LLD.
The script writes two x86-64 guest executables and their import listings under
`artifacts/compat-fixtures/`. Both variants come from the original C source
and are linked without the C runtime. The generated import library supplies
linker metadata for a synthetic DLL.

## Expected runtime behavior

The synthetic guest service resolver binds these imports from `XENORA-TEST.dll`.

- `xenora_transform` receives a writable guest range of three unsigned
  32-bit words. It doubles each word and returns the number written.
  The input `[2, 5, 11]` must become `[4, 10, 22]`.
- `xenora_report` reads three unsigned 32-bit words and returns their sum.
  The expected guest exit status is 36. Exit status 91 indicates that the
  transform did not return the expected word count.
- The second executable imports `xenora_missing` in place of
  `xenora_report`. The resolver must reject it before entering the guest.

The build script checks import metadata using LLVM. Run
`tests/compat_fixtures/verify.sh` to build both images, execute the positive
guest, compare the service trace and exit value, and check pre-entry rejection
of the unresolved symbol. Relocation and thread-local storage initialization
are outside this fixture's scope.

## Guest compute acceptance

Run `tests/compat_fixtures/verify-compute.sh` on native Apple Silicon with
LLVM and Metal. It builds the original `compute_guest.c` and exercises the
named-import runner with the Metal feature. The positive guest submits a
three-word resource and reports the Metal readback twice in separate runs.
The gate checks the guest's returned sum and the Metal validation banners.
The negative variants cover the command, handle, binding, count, unmapped
buffer and stale one-shot handle.

## Persistent resources and translated drawing

Run `tests/compat_fixtures/verify-resources.sh` on native Apple Silicon
with LLVM, Python and Metal. It builds the source-compiled
`resource_guest.c` PE32+ variants, runs each positive fixture twice and
checks the exact native readback, guest memory report and guest exit status.
Each run has a 15-second timeout. Metal API and shader validation must be
enabled.

- Compute submits a three-word view after a guard word. It updates the live
  buffer before completion, dispatches again, and queues work that completes
  from the old snapshot after release and slot reuse.
- Draw submits an original full-coverage triangle to the translated
  vertex and pixel stages in a 4×3 viewport. The guest directly loads and
  compares all 12 pixel words for each of the constant and interpolated
  draws. The verification script separately derives the interpolated
  values at the pixel centers. A control changes the guest's expected
  interior pixel and checks that the guest returns 93 even though the same
  native draw bytes are returned.
- BC1 selects the base or smaller mip from the second array layer of a
  pitched upload and reads the native texture through an authored Metal
  kernel. The smaller mip checks that the guest guard word after the
  output is preserved.
- Invalid and stale handles, bounds, offset overflow, a read-only output,
  an occupied or empty submission slot, malformed draw input and
  malformed or read-only texture operations fail before Metal execution.
  The abandoned submission returns 73 without executing the queued work.

The runner prints the guest's exit value and returns a nonzero host status
for load or interpreter faults. The gate compares both outcomes. These
fixtures exercise synthetic commands and local Apple Silicon readback.

## Module programs

`modules.c` is an original C program with no C library linked. It compares its
module handle, image range and entry point with the linker's `__ImageBase`, its
own start function and its mapped header, tries the `GetModuleHandleEx` flags and
their failures, counts references to the C runtime, and ends through an
`ExitProcess` it looks up by name, with 0x4D0D. A failed check ends it with its
number. `modules.cpp` is an original hosted C++ program that holds references with
a class and uses the Windows headers' `MODULEINFO` and the `psapi.dll` export.

`build-modules.sh [output directory]` builds both with mingw-w64 at `-O0`, `-O1`,
`-O2` and `-Os`. `MODULES_BREAK` is a control that fails its fifth check,
`MODULES_PIN` pins a module, and one build imports from API sets and KernelBase.
`cargo test -p xenora-loader --test modules` runs the script and the programs, and
fails when the compiler is missing. No executable is committed.

## Process startup programs

`startup_info.c` is an original C program with no C library linked. It reads its
startup information and command line, narrow and wide, checks that a second read
gives the same, splits the command line by Microsoft's documented rules after
trying its splitter on their six examples, saves what it found on the synthetic
`C:` drive and exits with 0x5157. A failed check exits with its number.
`startup_info.cpp` is an original hosted C++ program whose global constructor
reads the same before `main`. It uses the public Windows headers, so its layout
assertions are independent of the structure declared in the C program.

`build-startup-info.sh [output directory]` builds both with mingw-w64 at `-O0`,
`-O1`, `-O2` and `-Os`. `MINGW_GCC` names the compiler. Three more builds of the C
program are controls: `STARTUP_BREAK` expects a wrong size, one imports the four
functions from API sets, and one names an API set Microsoft's
pages do not list and must be refused before entry. `cargo test -p xenora-loader --test startup_info` runs
the script and the programs, and fails when the compiler is missing. No
executable is committed.

## A hosted C++ program

`guest_cpp.cpp` is an original C++ program that uses the C++ runtime as an
ordinary program does. It has a namespace-scope object with a constructor and
a destructor, a function-local static that is constructed on first use, a
`std::vector<int>` sorted with `std::sort` and a comparator, a `std::string`
compared with `memcmp`, a table of function pointers called through the
table, a base class and a derived class whose virtual method is called
through a base pointer, a `thread_local` variable, a second guest thread
created with `CreateThread` and waited for with `WaitForSingleObject`, a file
it writes with `fprintf` and reads back with `fgets`, and an `atexit`
registration. It prints ten lines and exits with the number of the first check
that fails.

`build-guest-cpp.sh [output directory]` builds it. `cargo test -p xenora-loader
--test guest_cpp_path` runs that script and then runs the result through
`run_synthetic_startup_scheduled`, the standard services and the DBT. No built
executable is committed: the test builds from source every time, in CI too.
The test fails, and does not skip, when the compiler is missing.

Compiler configuration:

| Step | Command |
| --- | --- |
| Compile and link | `x86_64-w64-mingw32-g++ -O<level> -fno-rtti -fno-exceptions -fno-threadsafe-statics -o <name>.exe guest_cpp.cpp` |

It is built at `-O0`, `-O1`, `-O2` and `-Os`, and once more as a control that
imports a service no Xenora service provides, so the loader must refuse the
image before the entry point runs.

`-fno-exceptions` and `-fno-rtti` are deliberate for this program: it links
the C++ runtime dynamically, and the guest runtime provides that runtime's
functions as services, which have no exception machinery or type-info
vtables. Exceptions are exercised by the next program, which links the C++
runtime statically. `std::thread` is not used either: it needs
`libwinpthread-1.dll` and its eighteen pthread entry points, where the Win32
thread API the loader already implements with its deterministic cooperative
scheduler is enough.

Tools. `MINGW_GXX` defaults to Homebrew's mingw-w64 when it is installed and
otherwise to the `x86_64-w64-mingw32-g++` on `PATH`. As with the other
mingw-built fixtures, the script requires the compiler driver, the compiler
proper, the assembler and the linker, and every library they load that is a
file on disk, to be ARM64, and checks each with `file` and `lipo -archs`.

## A C++ program that throws

`guest_cpp_exceptions.cpp` is an original C++ program that throws and
catches (#447). It throws an `int`; a `std::runtime_error` five calls deep,
each frame owning an object whose destructor counts; a `std::logic_error`
that a handler rethrows with `throw;`; a plain class caught with
`catch (...)`; a second exception from inside a handler; and an exception
through a frame with an 8 KiB stack array while two `double` values live
across the throwing call. It prints one line per check and returns the
number of checks that failed. Given the argument `uncaught`, it throws an
exception nobody catches; since #453 that exception reaches the C runtime's
startup filter, which lets `std::terminate` report it and `abort` end the
process with exit code 3.

`build-guest-cpp.sh` builds it too, at `-O0`, `-O1`, `-O2` and `-Os`:

| Step | Command |
| --- | --- |
| Compile and link | `x86_64-w64-mingw32-g++ -O<level> -Wall -Wextra -Werror -static-libstdc++ -static-libgcc -o guest-cpp-exceptions-<level>.exe guest_cpp_exceptions.cpp` |

Static linking puts GCC's SEH unwinder and C++ personality into the
executable as guest code, so the program needs the operating system's
exception services rather than C++ runtime services. `cargo test -p
xenora-loader --test guest_cpp_exceptions` builds and runs it.

## C structured exceptions

`guest_c_seh.c` is an original freestanding C program that uses MSVC-style
structured exception handling (#453). Each case is selected at compile time
with `-DCASE=<name>`, reports what ran by calling `xenora_note` (a service the
test registers under `xenora_seh.dll`) and exits with its own status. The
cases are a divide by zero caught by `__try`/`__except` (exit 42); a filter
that declines so the caller's filter catches it (43); a fault no filter
accepts; a write, a read and a call to unmapped addresses whose filter reports
the access-violation parameters (44 to 46); UD2 under a constant filter (47);
a `__finally` one frame below the handler and one in the handler's own frame
(48, 49); and a filter that continues execution after a raise (50). Each
case's notes and status are stated next to it in the source.

`build-guest-c-seh.sh` builds every case at `-O0`, `-O1` and `-O2`, with
import libraries made from definition files and no C runtime:

| Step | Command |
| --- | --- |
| Compile | `clang --target=x86_64-pc-windows-msvc -O<level> -fasync-exceptions -ffreestanding -fno-stack-protector -Wall -Wextra -Werror -Wno-unused-function -DCASE=<case> -c guest_c_seh.c` |
| Link | `lld-link /entry:entry /subsystem:console /nodefaultlib /machine:x64 <case>.obj kernel32.lib ucrtbase.lib xenora_seh.lib` |

`-fasync-exceptions` makes a CPU fault inside `__try` catchable. `cargo test
-p xenora-loader --test guest_c_seh` builds and runs every case.
The script finds its linker as `build-gate-a1.sh` does: `LLD_LINK`, Homebrew
`lld`, the `lld-link` beside a Homebrew LLVM, `lld-link` on `PATH`, then the
`rust-lld` in the active Rust toolchain, so a machine with only Rust and a
clang can build the cases.

## Gate A1: a C program with no C library

`gate_a1.c` is an original C program that links no C library. It imports
twelve functions from `ucrtbase.dll` and eight from `kernel32.dll` by name,
with prototypes written from the public Windows SDK and C standard
declarations. It allocates and frees memory, sorts with a callback, does
64-bit and double arithmetic, uses the C string functions, writes and reads
files on the synthetic `C:` drive, checks two error paths, writes a text
report and ends with `ExitProcess`.

`build-gate-a1.sh [output directory]` builds it. `cargo test -p xenora-loader
--test gate_a1` runs that script and then runs the result through
`run_synthetic_startup`, the standard services and the DBT. No built
executable is committed: the test builds from source every time, in CI too.

Compiler configuration:

| Step | Command |
| --- | --- |
| Compile | `clang --target=x86_64-pc-windows-msvc -O<level> -ffreestanding -fno-stack-protector -Wall -Wextra -Werror` |
| Import libraries | `lld-link /lib /machine:x64 /def:<kernel32 or ucrtbase>.def` from the name lists in the script |
| Link | `lld-link /entry:gate_entry /subsystem:console /nodefaultlib /timestamp:0` |

It is built at `-O0`, `-O1`, `-O2` and `-Os`, and three more times at `-O1` as
controls: `GATE_BREAK` (one wrong expectation), `GATE_RETURN` (the entry point
returns its status) and `GATE_MISSING` (an import no service provides).

Tools. `CLANG` defaults to Homebrew LLVM when it is installed and otherwise to
the `clang` on `PATH`. `LLD_LINK` defaults to Homebrew `lld` when it is
installed and otherwise to the `rust-lld` inside the active Rust toolchain,
which is an LLD. On macOS the script requires both, and every library they
load that is a file on disk, to be ARM64: it checks each tool with `file` and
`lipo -archs`, and each library `otool -L` lists with `lipo -archs`, following
the tool's own run paths. The script writes the versions it used to
`toolchain.txt`, which the test prints, and what the native checks looked at to
`toolchain-native.txt`. The test builds into a directory of its own for each
test process under the Cargo target directory, so two runs that share a target
directory do not overwrite each other's files; `cargo clean` removes them.

Provenance. The source, the name lists and the script were written for Xenora
and are GPL-3.0-or-later. Nothing from a Windows SDK, a C runtime or another
project is compiled or linked. The compiler and linker are LLVM projects
under the Apache License 2.0 with LLVM exceptions; they are tools, not inputs.

## Gate A2: a hosted C program

`gate_a2.c` is an original C program that uses the standard C library as an
ordinary program does. It includes only `<stdio.h>`, `<stdlib.h>` and
`<string.h>`. It prints its arguments, fills heap memory from a small
generator, sorts it with `qsort` and a callback, checksums it, writes a
report with `fprintf` and `fputs`, reads it back with `fgets`, `strtoul` and
`strtoull`, confirms that a missing file is reported, registers a function
with `atexit`, and returns from `main`. It checks every result itself and
exits with the number of the first check that fails. It uses no floating
point and reads no input.

`build-gate-a2.sh [output directory]` builds it. `cargo test -p xenora-loader
--test gate_a2` runs that script and then runs the result through
`run_synthetic_startup`, the standard services and the DBT. No built
executable is committed: the test builds from source every time, in CI too.
The test fails, and does not skip, when the compiler is missing.

Compiler configuration:

| Step | Command |
| --- | --- |
| Compile and link | `x86_64-w64-mingw32-gcc -O<level> -Wall -Wextra -Werror -o <name>.exe gate_a2.c` |

It is built at `-O0`, `-O1`, `-O2` and `-Os`, and three more times at `-O1` as
controls: `GATE_BREAK` (one wrong expectation), `GATE_EXIT` (`exit(42)` half
way) and `GATE_ABORT` (`abort()` half way).
One further build, `GATE_SPIN`, prints its arguments, counts to the first one
and ends; given 0 it never ends. It also writes one line to standard error. The gate does not run it;
`crates/cli/tests/run_program.rs` uses it to test the limits of `xenora run`.

Tools. `MINGW_GCC` defaults to Homebrew's mingw-w64 when it is installed and
otherwise to the `x86_64-w64-mingw32-gcc` on `PATH`. The toolchain must use
the Universal C Runtime, which Homebrew's does; a toolchain built for the
older `msvcrt.dll` produces a program that imports other functions. On macOS
the script requires the compiler driver, the compiler proper, the assembler
and the linker, and every library they load that is a file on disk, to be
ARM64: it checks each with `file` and `lipo -archs`, and each library
`otool -L` lists with `lipo -archs`. It writes the compiler version to
`toolchain.txt`, which the test prints, and what the native checks looked at
to `toolchain-native.txt`. The test builds into a directory of its own for
each test process under the Cargo target directory.

Provenance. The source and the script were written for Xenora and are
GPL-3.0-or-later. Unlike the Gate A1 program, each built executable contains
code that is not Xenora's: mingw-w64's startup objects and a little of GCC's
support library, as any program built with that toolchain does. Neither is
stored in the repository or in its history; the executables are test output.
mingw-w64's `COPYING` file says its runtime is under the Zope Public License
2.1 except for parts marked as public domain, BSD or LGPL; GCC's support
library is GPL-3.0-or-later with the GCC Runtime Library Exception. The C
library itself is not linked: the program imports it by name, and Xenora's
own services answer.

## The buffer forms of printf

`tests/compat_fixtures/crt_sprintf.c` (#466) is a hosted C program that
formats into a buffer with `snprintf`, `sprintf`, `_snprintf` and
`sprintf_s`, queries the needed size with a null buffer, and writes a
character with `putchar`. Built by `build-crt-sprintf.sh` with mingw-w64
(UCRT flavour) and `-fno-builtin`, the calls are imports of
`api-ms-win-crt-stdio-l1-1-0.dll`. Each check compares the result and the
bytes in and around the buffer with what Microsoft's `snprintf`,
`_snprintf` and `sprintf_s` pages describe: termination and truncation for
the standard, legacy and secure forms, the guards before and after the
buffer included. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_SPRINTF_BREAK`, which expects a wrong length on purpose as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.

## The encoding conversions

`tests/compat_fixtures/crt_codepages.c` (package 7 of #459) is a hosted C
program that converts between bytes and wide characters with
`MultiByteToWideChar` and `WideCharToMultiByte` in UTF-8 and in
Windows-1252, queries the pages with `GetACP`, `GetOEMCP`,
`IsValidCodePage` and `GetCPInfo`, and checks the documented error codes
of the strict and small-buffer failures. Built by `build-crt-codepages.sh`
with mingw-w64, the calls are imports of `kernel32.dll`. It is built at
`-O0` and `-O2`, and once at `-O1` with `CRT_CODEPAGES_BREAK`, which
expects a wrong count on purpose as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.

## The locale identity

`tests/compat_fixtures/crt_locale.c` (package 7 of #459) is a hosted C
program that asks the runtime for its one locale with the LCID, LANGID and
locale-name getters and for the en-US strings of `GetLocaleInfo` in its
three forms, checking the documented failures for other locales, unknown
types and small buffers. Built by `build-crt-locale.sh` with mingw-w64,
the calls are imports of `kernel32.dll`. It is built at `-O0` and `-O2`,
and once at `-O1` with `CRT_LOCALE_BREAK`, which expects a wrong count on
purpose as a control. `crates/cli/tests/run_program.rs` runs all three
through `xenora run`.
Provenance is as for Gate A2: the source and script are Xenora's, and the
executables are test output that contains mingw-w64 startup code.

## The standard input

`tests/compat_fixtures/crt_stdin.c` (#467) is a hosted C program that reads
lines from its standard input with a buffer of 8 and checks what ISO C says
at the edges. Built by `build-crt-stdin.sh` with mingw-w64 (UCRT flavour)
and `-fno-builtin`, the calls are imports of `api-ms-win-crt-stdio-l1-1-0.dll`.
The host pipes a fixed input: two short lines in CR LF, one line longer than
the buffer, and a last line with no newline; the checks count the reads, the
bytes after the text-mode replacement, the line feeds and any surviving
carriage return, and the end-of-file indicator. With the argument `empty`
and no input it expects every count to be zero. It is built at `-O0` and
`-O2`, and once at `-O1` with `CRT_STDIN_BREAK`, which expects a wrong read
count as a control. `crates/cli/tests/run_program.rs` runs all three through
`xenora run`, and feeds one 1 MiB + 1 byte to check the refusal. Provenance
is as for Gate A2.

## Formatted input

`tests/compat_fixtures/crt_scanf.c` (#605) is a hosted C program that reads
formatted input with `sscanf`, `sscanf_s`, `scanf`, `scanf_s`, `fscanf` and
`fscanf_s`. Built by `build-crt-scanf.sh` with mingw-w64 (UCRT flavour) and
`-fno-builtin`, the six functions are mingw-w64 wrappers that call the
imports `__stdio_common_vsscanf` and `__stdio_common_vfscanf` of
`api-ms-win-crt-stdio-l1-1-0.dll`. The checks cover integers in each base and
size, widths, strings, characters, scansets, decimal and hexadecimal floating
point, `%%`, `%n`, suppression, matching and input failures, and the buffer
sizes of the secure forms. The host pipes a fixed input in CR LF, which the
program reads with `scanf` and `getchar`; it then writes a file in text mode
and reads it back with `fscanf` in text and binary mode, checking each
position with `_ftelli64`. Every expected value follows from the input and
from ISO C 7.21.6.2 or Microsoft's pages; it exits with the number of the
first failed check, or 0. It is built at `-O0` and `-O2`, and once at `-O1`
with `CRT_SCANF_BREAK`, which expects the carriage return of a CR LF pair to
survive, so check 17 fails, as a control. `crates/cli/tests/run_program.rs`
runs all three through `xenora run`. Provenance is as for Gate A2. Nothing
here has been compared with Windows.

## C runtime file positions

`tests/compat_fixtures/crt_files.c` (#493) is a hosted C program that writes
files on Xenora's in-memory drive and moves through them with `_ftelli64`,
`_fseeki64`, `fgetpos` and `fsetpos`, in binary and text mode, then renames
and removes them. Every expected value follows from the bytes it writes; it
exits with the number of the first failed check, or 0. `build-crt-files.sh`
builds it with mingw-w64 (UCRT flavour) at `-O0` and `-O2`, and once at `-O1`
with `CRT_FILES_BREAK`, which expects a text position without the carriage
returns, so check 17 fails, as a control. `crates/cli/tests/run_program.rs`
runs all three through `xenora run`. Provenance is as for Gate A2.

## Temporary files

`tests/compat_fixtures/crt_tmp.c` (#493, last part) is a hosted C program
that asks `tmpnam` for names, checks they differ and hold no file, and
writes and reads a stream `tmpfile` opened. `build-crt-tmp.sh` builds it
with mingw-w64 (UCRT flavour) at `-O0` and `-O2`, and once at `-O1` with
`CRT_TMP_BREAK`, whose control check must fail.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.
Provenance is as for Gate A2.

## The descriptor layer

`tests/compat_fixtures/crt_fds.c` (#493, third part) is a hosted C program
that opens files with `_open`, moves through them with `_lseek`, reads and
writes with `_read` and `_write` in binary and text mode, takes the
descriptor of a stream with `_fileno` and writes through it so the stream's
position moves, and checks `_isatty`, `_access` and `_unlink`. Every
expected value follows from the bytes it writes and the Microsoft pages for
each function. `build-crt-fds.sh` builds it with mingw-w64 (UCRT flavour)
at `-O0` and `-O2`, and once at `-O1` with `CRT_FDS_BREAK`, whose control
check must fail. `crates/cli/tests/run_program.rs` runs all three through
`xenora run`. Provenance is as for Gate A2.

## The secure open and stream indicators

`tests/compat_fixtures/crt_secure.c` (#493, second part) is a hosted C
program that opens with `fopen_s`, reads a file to its end so the end-of-file
indicator is set, and clears it with `clearerr`. Every expected value follows
from the bytes it writes and the Microsoft pages for `fopen_s`, `feof` and
`clearerr`; it exits with the number of the first failed check, or 0.
`build-crt-secure.sh` builds it with mingw-w64 (UCRT flavour) at `-O0` and
`-O2`, and once at `-O1` with `CRT_SECURE_BREAK`, whose control check must
fail. `crates/cli/tests/run_program.rs` runs all three through `xenora run`.
Provenance is as for Gate A2.

## The C clock

`tests/compat_fixtures/crt_clock.c` (#459, package 5) is a hosted C program
that breaks down and composes known seconds since the epoch, checks the
CRT's string forms and `strftime`, and verifies `mktime`'s normalization.
`build-crt-clock.sh` builds it with mingw-w64 (UCRT flavour) at `-O0` and
`-O2`, and once at `-O1` with `CRT_CLOCK_BREAK`, whose control check must
fail. `crates/cli/tests/run_program.rs` runs all three through
`xenora run`. Provenance is as for Gate A2.

## Wide characters

`tests/compat_fixtures/crt_wide.c` (#459, package 6, first part) is a
hosted C program that converts between multibyte and wide strings in both
directions, checks the failure above character 255, the pointer advance
of the restartable forms, and the ASCII classification.
`build-crt-wide.sh` builds it with mingw-w64 (UCRT flavour) at `-O0` and
`-O2`, and once at `-O1` with `CRT_WIDE_BREAK`, whose control check must
fail. `crates/cli/tests/run_program.rs` runs all three through
`xenora run`. Provenance is as for Gate A2.

## C runtime wide stream io

`tests/compat_fixtures/crt_wstream.c` (#459) is a hosted C program that writes
and reads wide strings on file streams with `fputws` and `fgetws` in the C
locale, checks the C orientation both ways — byte io refused on a
wide-oriented stream, wide io refused on a byte-oriented one — and writes one
wide character to stdout with `putwchar`. Diagnostics go to stderr so stdout
stays unoriented until `putwchar` fixes it wide. Every expected value follows
from the bytes it writes; it exits with the number of the first failed check,
or 0. `build-crt-wstream.sh` builds it with mingw-w64 (UCRT flavour) at `-O0`
and `-O2`, and once at `-O1` with `CRT_WSTREAM_BREAK`, which expects `fputws`
to fail where it succeeds, so check 2 fails, as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.
Provenance is as for Gate A2.

## C runtime wide buffer formatting

`tests/compat_fixtures/crt_swprintf.c` (#459) is a hosted C program that
formats into wide buffers with `swprintf`: both string kinds and both
character kinds, padding, bases, a width from the arguments, a precision on
a wide string, and the CRT's default truncation behavior — count characters
without a terminator and -1, no size query. `build-crt-swprintf.sh` builds
it with mingw-w64 (UCRT flavour) at `-O0` and `-O2`, and once at `-O1` with
`CRT_SWPRINTF_BREAK`, which expects the integer conversion to fail, so check
1 fails, as a control. `crates/cli/tests/run_program.rs` runs all three
through `xenora run`. Provenance is as for Gate A2.

## C runtime mathematics

`tests/compat_fixtures/crt_math.c` (#468) is a hosted C program that calls the
49 `math.h` functions Xenora serves beyond `sqrt` and `fabs`. Built by
`build-crt-math.sh` with mingw-w64 (UCRT flavour) and `-fno-builtin`, each call
is an import of `api-ms-win-crt-math-l1-1-0.dll`; the arguments are volatile so
no call is folded at compile time. The program compares every result with a
value it carries and exits with the number of the first failed check, or 0.
Functions that IEC 60559 fixes exactly must match to the bit; the others may be
one unit in the last place from a reference computed with mpmath at 200 bits by
`crt_math_reference.py`. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_MATH_BREAK`, which expects a wrong value for check 67, as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.

## The Windows file services

`tests/compat_fixtures/crt_winfiles.c` (package 8 of #459) is a hosted C
program that creates, copies, moves, enumerates and describes files on the
synthetic drive through the narrow and wide services, checking the sharing
between widths, the failures and the unique temporary names. Built by
`build-crt-winfiles.sh` with mingw-w64, the calls are imports of
`kernel32.dll`. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_WINFILES_BREAK` as a control. `crates/cli/tests/run_program.rs` runs
all three through `xenora run`.

## The legacy memory services

`tests/compat_fixtures/crt_winmemory.c` (package 9 of #459) is a hosted C
program that allocates, writes, zero-checks and frees memory through the
legacy `GlobalAlloc`/`LocalAlloc` family, checking the null free, the
documented failure answers and the sharing with the process heap. Built by
`build-crt-winmemory.sh` with mingw-w64, the calls are imports of
`kernel32.dll`. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_WINMEMORY_BREAK` as a control. `crates/cli/tests/run_program.rs` runs
all three through `xenora run`.

## The error, environment and system services

`tests/compat_fixtures/crt_winsys.c` (package 9 of #459) is a hosted C
program that formats error messages in both widths, exchanges the error
mode, expands strings over the empty environment, walks the environment
block and reads the version, directories, computer name and processor
topology of the simulated machine. Built by `build-crt-winsys.sh` with
mingw-w64, the calls are imports of `kernel32.dll`. It is built at `-O0`
and `-O2`, and once at `-O1` with `CRT_WINSYS_BREAK` as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.
Provenance is as for Gate A2: the source and scripts are Xenora's, and the
executables are test output that contains mingw-w64 startup code.

## Byte-range locks and the file-mode services

`tests/compat_fixtures/crt_winlocks.c` is a hosted C program for the
kernel32 services that a database engine such as sqlite3 imports besides
plain file I/O. It takes and releases exclusive locks with `LockFile` and
shared and exclusive locks with `LockFileEx` through an `OVERLAPPED`,
checks the conflict and unlock errors, switches the file-API code page
mode, reads the free space of the current root, validates the process
heap, flushes a mapped view and compares `_getpid` with
`GetCurrentProcessId`. Built by `build-crt-winlocks.sh` with mingw-w64, it
is built at `-O0` and `-O2`, and once at `-O1` with `CRT_WINLOCKS_BREAK` as
a control. `crates/cli/tests/run_program.rs` runs all three through
`xenora run`. Provenance is as for Gate A2.

`tests/compat_fixtures/crt_crtutils.c` (the survey residue of #459) is a
hosted C program that exercises the small CRT utilities: narrow string
copying and concatenation, `strrchr`, `_strdup`, the in-place case
conversions and reversal, the C-locale collation forms, the Annex K copy
constraints, the character classes, the integer formatting family and
`getenv_s` over the empty environment. Built by `build-crt-crtutils.sh`
with mingw-w64, the calls are imports of `ucrtbase.dll`. It is built at
`-O0` and `-O2`, and once at `-O1` with `CRT_CRTUTILS_BREAK` as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.

## String comparison and case mapping

`tests/compat_fixtures/crt_compare.c` (package 7 of #459) is a hosted C
program that compares, classifies and case-maps strings with the `lstrcmp`
family, `CompareString`, `GetStringTypeW` and `LCMapStringW`, checking the
byte order, the ASCII fold, the terminator counting and each documented
failure. Built by `build-crt-compare.sh` with mingw-w64, the calls are
imports of `kernel32.dll`. It is built at `-O0` and `-O2`, and once at `-O1`
with `CRT_COMPARE_BREAK`, which expects a wrong count on purpose as a
control. `crates/cli/tests/run_program.rs` runs all three through
`xenora run`.

## The console

`tests/compat_fixtures/crt_console.c` (package 8 of #459) is a hosted C
program that asks for the standard handles, the console code pages and the
modes, writes narrow and wide text interleaved with `printf` to prove the
one capture, reads an input line, and checks each documented failure.
Built by `build-crt-console.sh` with mingw-w64, the calls are imports of
`kernel32.dll`. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_CONSOLE_BREAK` as a control. `crates/cli/tests/run_program.rs` runs
all three through `xenora run`.

## The Windows file services

`tests/compat_fixtures/crt_winfiles.c` (package 8 of #459) is a hosted C
program that creates, copies, moves, enumerates and describes files on the
synthetic drive through the narrow and wide services, checking the sharing
between widths, the failures and the unique temporary names. Built by
`build-crt-winfiles.sh` with mingw-w64, the calls are imports of
`kernel32.dll`. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_WINFILES_BREAK` as a control. `crates/cli/tests/run_program.rs` runs
all three through `xenora run`.

## The legacy memory services

`tests/compat_fixtures/crt_winmemory.c` (package 9 of #459) is a hosted C
program that allocates, writes, zero-checks and frees memory through the
legacy `GlobalAlloc`/`LocalAlloc` family, checking the null free, the
documented failure answers and the sharing with the process heap. Built by
`build-crt-winmemory.sh` with mingw-w64, the calls are imports of
`kernel32.dll`. It is built at `-O0` and `-O2`, and once at `-O1` with
`CRT_WINMEMORY_BREAK` as a control. `crates/cli/tests/run_program.rs` runs
all three through `xenora run`.

## The error, environment and system services

`tests/compat_fixtures/crt_winsys.c` (package 9 of #459) is a hosted C
program that formats error messages in both widths, exchanges the error
mode, expands strings over the empty environment, walks the environment
block and reads the version, directories, computer name and processor
topology of the simulated machine. Built by `build-crt-winsys.sh` with
mingw-w64, the calls are imports of `kernel32.dll`. It is built at `-O0`
and `-O2`, and once at `-O1` with `CRT_WINSYS_BREAK` as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.
Provenance is as for Gate A2: the source and scripts are Xenora's, and the
executables are test output that contains mingw-w64 startup code.

`tests/compat_fixtures/crt_crtutils.c` (the survey residue of #459) is a
hosted C program that exercises the small CRT utilities: narrow string
copying and concatenation, `strrchr`, `_strdup`, the in-place case
conversions and reversal, the C-locale collation forms, the Annex K copy
constraints, the character classes, the integer formatting family and
`getenv_s` over the empty environment. Built by `build-crt-crtutils.sh`
with mingw-w64, the calls are imports of `ucrtbase.dll`. It is built at
`-O0` and `-O2`, and once at `-O1` with `CRT_CRTUTILS_BREAK` as a control.
`crates/cli/tests/run_program.rs` runs all three through `xenora run`.
Provenance is as for Gate A2: the source and scripts are Xenora's, and the
executables are test output that contains mingw-w64 startup code.
