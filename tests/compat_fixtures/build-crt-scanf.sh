#!/bin/sh
# Build the crt_scanf compatibility fixture with mingw-w64 GCC: the normal
# program at two optimisation levels, and the control variant that must be
# caught. The output directory is the script's first argument.
set -eu
cd "$(dirname "$0")/../.."
out="$1"
mkdir -p "$out"
for level in 0 2; do
    x86_64-w64-mingw32-gcc -O$level -Wall -Wextra -Werror -fno-builtin \
        -o "$out/crt-scanf-O$level.exe" tests/compat_fixtures/crt_scanf.c
done
x86_64-w64-mingw32-gcc -O1 -DCRT_SCANF_BREAK -Wall -Wextra -Werror -fno-builtin \
    -o "$out/crt-scanf-break.exe" tests/compat_fixtures/crt_scanf.c
echo "built crt-scanf fixtures in $out"
