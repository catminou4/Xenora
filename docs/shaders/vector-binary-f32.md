# Wave32 VOP2 binary32 arithmetic

Xenora translates six gfx1030 VOP2 binary32 forms for the synthetic Wave32
compute ABI: `v_add_f32`, `v_sub_f32`, `v_subrev_f32`, `v_mul_f32`,
`v_min_f32` and `v_max_f32`, all `_e32`. The AMD RDNA 2 ISA Reference Guide
defines the operations in section 12.7, the encoding in section 13.3.1, and the
execution mask (EXEC) in section 3.3.

| Form | Result in each active lane | Guide |
| --- | --- | --- |
| `v_add_f32_e32` | `D.f32 = S0.f32 + S1.f32` | p131 |
| `v_sub_f32_e32` | `D.f32 = S0.f32 - S1.f32` | p131 |
| `v_subrev_f32_e32` | `D.f32 = S1.f32 - S0.f32` | p131 |
| `v_mul_f32_e32` | `D.f32 = S0.f32 * S1.f32` | p131 |
| `v_min_f32_e32` | The p132 minimum. | p132 |
| `v_max_f32_e32` | The p133 maximum. | p133 |

`v_subrev_f32_e32` reverses the operands: it is the second input minus the
first, so it is not the negation of `v_sub_f32_e32`.

## Operand limits

SRC0 accepts a VGPR, a normal scalar register `s0`–`s105`, `vcc_lo` (the low
half of the vector condition code), an inline integer, or a literal word.
VSRC1 must be a VGPR and VDST is a VGPR. Inactive lanes keep their old
values, and none of the six forms writes VCC, SCC or EXEC.

An integer inline constant is refused at decode: the guide does not say
whether a binary32 operation reads one as an integer bit pattern or as a
converted value, and the two readings differ. A float inline constant stays
outside the VOP2 subset for the same reason.

## The modelled domain, and what is refused

The six forms are modelled over the inputs whose result the CPU reference and
native Metal agree on. Everything outside that domain is refused rather than
approximated, so a program that needs one of these cases is told so instead of
receiving a plausible wrong answer.

- **Rounding.** MODE.FP_ROUND is pinned to 0, round to nearest with ties to
  even (guide p15). ADD, SUB, SUBREV and MUL are IEEE 754 binary32
  operations; an exact zero sum is `+0` unless both addends are `-0`, and an
  overflow gives a signed infinity.
- **Subnormal inputs are refused**, for all six forms. The pinned mode allows
  input denormals, but Metal flushes them in float arithmetic, so the two
  paths would disagree on the same program.
- **A NaN input or a NaN result is refused for ADD, SUB, SUBREV and MUL.** The
  guide gives no NaN result encoding, and Metal returns the same quiet NaN
  for every NaN-producing operation.
- **A signalling NaN input is refused for MIN and MAX.** The guide quiets a
  signalling NaN only under `IEEE_MODE`, and defines neither `IEEE_MODE` as a
  mode nor the `Quiet()` function it calls; `IEEE_MODE` is not pinned.
- **An ADD, SUB, SUBREV or MUL result whose exact value is nonzero and below
  2^-126 in magnitude is refused**, because output flushing is not pinned and
  Metal flushes.

The MIN and MAX rules are the guide's own pseudocode (pages 132 and 133), not
Metal's `fmin` and `fmax`, which differ on signed zeros. A quiet NaN operand
gives the other operand (`S1` when both are NaN); a `+0`/`-0` pair gives `-0`
for MIN and `+0` for MAX; otherwise the result is `S0 < S1 ? S0 : S1` for MIN
and `S0 >= S1 ? S0 : S1` for MAX. The guide's `IEEE_MODE` branch changes only
the signalling NaN cases and MAX's `>=` against `>`, and the latter selects
the same bits for equal non-zero operands.

## Where the domain is enforced

The compiler accepts every program whose encoding is in the subset. The
value domain above is enforced by the CPU interpreters, at run time, on the
value a lane actually holds: a value outside the domain is refused there
rather than answered with a number the two paths would not agree on. A
program whose operands are constants is not refused at compile time, because
the compiler does not evaluate the float domain; the two hand-written forms
this batch replaces used to fold an exact constant float operation and
refuse an inexact one at compile time, and neither happens now.

A translator therefore accepts a program that the interpreters will refuse if
its runtime values leave the domain, and Metal computes such a program with
Metal's own rules, which flush a subnormal to zero. The reference and the
Metal path therefore disagree for a program that uses an out-of-domain
value, and the disagreement is the documented limit of this batch rather
than a silent difference.

## How each path computes the result

The CPU reference computes binary32 arithmetic exactly, in integer
arithmetic on the bit patterns, so it does not inherit the host floating-point
unit's behaviour. It rounds the exact result once, to nearest even.

The Metal Shading Language path uses native binary32 arithmetic for ADD, SUB,
SUBREV and MUL, with contraction and reassociation turned off and the library
built in safe math mode. MIN and MAX are computed with integer tests on the
register bits, because no Metal float operation reproduces the guide's
signed-zero rule. The compare path used by the VOPC binary32 forms does the
same for the same reason.

## Evidence

The [guide-derived oracle](../../tests/validation/tests/vop2_float_arithmetic.rs)
holds hand-written expected values transcribed from the guide's definitions
and checks them against both CPU interpreters and, through the translator, on
native Metal. The [LLVM-assembled fixture](../../fixtures/rdna2/vector-f32.s)
exercises all six forms, including the reversed subtraction, a literal
multiplier, a minimum against a scalar register and a maximum against `-0`
under a partial EXEC mask.

Independent acceptance of the translator is pending. Xbox hardware agreement is
unknown.
