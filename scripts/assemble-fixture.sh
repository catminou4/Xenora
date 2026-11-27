#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu

cd "$(dirname "$0")/.."
llvm_bin=${LLVM_BIN:-$(brew --prefix llvm)/bin}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
case "${1:-}" in
  --check|"") ;;
  *) printf '%s\n' 'usage: scripts/assemble-fixture.sh [--check]' >&2; exit 2 ;;
esac

fixtures='add bitwise-branch float-exact or shifts ashr
    x9-constant-vs x9-constant-ps x9-varying-vs x9-varying-ps x9-transform-vs
    scalar-alu vector-compare scalar-logic scalar-compare scalar-unary scalar-saveexec
    scalar-arith
    scalar-immediate vector-integer vector-compare-ft vector-vcc vop3-integer
    vop3-compare vector-unary vector-compare-64 vector-compare-16 scalar-memory runtime-branch
    vector-round
    vector-f32
    image-sample-lz-dmask1 image-sample-lz-dmask8 image-sample-lz-dmask5
    image-sample-lz-dmaskf image-sample-l-lod0 image-sample-l-lod1'

for name in $fixtures; do
    "$llvm_bin/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -mattr=+wavefrontsize32 \
        -filetype=obj "fixtures/rdna2/$name.s" -o "$scratch/$name.o"
    "$llvm_bin/llvm-objcopy" -O binary --only-section=.text "$scratch/$name.o" "$scratch/$name.bin"
done

# A tool failure leaves every checked-in fixture at its original bytes.
for name in $fixtures; do
    if [ "${1:-}" = --check ]; then
        cmp "$scratch/$name.bin" "fixtures/rdna2/$name.bin"
    else
        cp "$scratch/$name.bin" "fixtures/rdna2/$name.bin"
    fi
done
