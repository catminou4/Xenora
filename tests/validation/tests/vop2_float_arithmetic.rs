// SPDX-License-Identifier: GPL-3.0-or-later

//! Independent per-lane expected values for the six VOP2 binary32 forms
//! (#784): `v_add_f32`, `v_sub_f32`, `v_subrev_f32`, `v_mul_f32`,
//! `v_min_f32` and `v_max_f32`, all `_e32`, Wave32.
//!
//! The expectations come from the public AMD "RDNA 2" Instruction Set
//! Architecture Reference Guide only: section 12.7 "VOP2 Instructions" (page
//! 131: ADD, SUB, SUBREV and MUL, "0.5ULP precision, denormals are
//! supported"; page 132: the MIN pseudocode; page 133: the MAX pseudocode),
//! section 3.5 (page 15: MODE.FP_DENORM, DX10_CLAMP and IEEE), section 13.3.1
//! for the VOP2 word and section 6.2 for SRC0. `KNOWN` is a table written by
//! hand from those definitions; `expect` transcribes them through exact
//! binary64 arithmetic. Nothing here comes from the ISA crate, the registry,
//! the decoder or the shader compiler.
//!
//! Readings of the guide (not hardware evidence):
//! - ADD, SUB, SUBREV and MUL are IEEE 754 binary32 operations rounded to
//!   nearest even (MODE.FP_ROUND 0, guide 3.5). An exact zero sum is +0
//!   unless both addends are -0; SUB is S0 + (-S1) and SUBREV S1 + (-S0).
//!   Overflow gives a signed infinity.
//! - MIN and MAX follow the page 132 and 133 pseudocode: a quiet NaN gives the
//!   other operand (S1 when both are NaN), a +0/-0 pair gives -0 for MIN and +0
//!   for MAX, otherwise S0 < S1 ? S0 : S1 (MIN) and S0 >= S1 ? S0 : S1 (MAX).
//!   IEEE_MODE changes only the signalling NaN cases and MAX's `>=` against
//!   `>`, which selects the same bits for equal non-zero operands.
//! - Refused (no expected value), each checked on the CPU interpreters only:
//!   - Signalling NaN inputs to MIN and MAX. They are quieted under IEEE_MODE
//!     and the guide does not define `Quiet()` (whether the payload is kept);
//!     IEEE_MODE is not pinned.
//!   - Every NaN input to ADD, SUB, SUBREV and MUL, and every NaN result
//!     (inf - inf, 0 * inf). The guide gives no NaN result encoding, and Metal
//!     returns 0x7fc00000 for each.
//!   - Subnormal inputs. The pinned mode allows input denormals, but Metal
//!     flushes them in float arithmetic (measured on Apple M4: 2^-149 + 2^-149
//!     gives +0). MIN and MAX refuse them too, since MODE.FP_DENORM output
//!     flushing (modes 1 against 3) is not pinned.
//!   - ADD, SUB, SUBREV and MUL results whose exact value is nonzero and below
//!     2^-126 in magnitude: output flushing is not pinned and Metal flushes.
//! - Refused at decode: integer inline constants (selectors 128-208) as SRC0,
//!   since the guide does not say whether a float operation reads them as
//!   integer bits or converted values. Float inline constants stay outside the
//!   VOP2 subset.
//! - None of the six writes VCC or SCC; inactive lanes keep their VGPRs
//!   (section 3.3).

#[path = "operand_domains/harness.rs"]
#[allow(dead_code)]
mod harness;
#[path = "../independent_ref.rs"]
mod independent_ref;

use std::collections::BTreeMap;

use harness::{Case, words};
use xenora_shader::{Architecture, CompileError, compile};

/// The six forms with their ENC_VOP2 opcodes (guide 12.7).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    Add = 3,
    Sub = 4,
    Subrev = 5,
    Mul = 8,
    Min = 15,
    Max = 16,
}

const FORMS: [(Form, &str); 6] = [
    (Form::Add, "v_add_f32"),
    (Form::Sub, "v_sub_f32"),
    (Form::Subrev, "v_subrev_f32"),
    (Form::Mul, "v_mul_f32"),
    (Form::Min, "v_min_f32"),
    (Form::Max, "v_max_f32"),
];

fn column(form: Form) -> usize {
    FORMS.iter().position(|&(f, _)| f == form).unwrap()
}

/// A refused input: no expected value.
const NO: Option<u32> = None;

const fn y(bits: u32) -> Option<u32> {
    Some(bits)
}

/// S0, S1 and the expected D for ADD, SUB, SUBREV, MUL, MIN and MAX, by hand.
#[rustfmt::skip]
const KNOWN: [(u32, u32, [Option<u32>; 6]); 36] = [
    (0x3f80_0000, 0x4000_0000, [y(0x4040_0000), y(0xbf80_0000), y(0x3f80_0000), y(0x4000_0000), y(0x3f80_0000), y(0x4000_0000)]), // 1, 2
    (0x0000_0000, 0x8000_0000, [y(0x0000_0000), y(0x0000_0000), y(0x8000_0000), y(0x8000_0000), y(0x8000_0000), y(0x0000_0000)]), // +0, -0
    (0x8000_0000, 0x0000_0000, [y(0x0000_0000), y(0x8000_0000), y(0x0000_0000), y(0x8000_0000), y(0x8000_0000), y(0x0000_0000)]), // -0, +0
    (0x8000_0000, 0x8000_0000, [y(0x8000_0000), y(0x0000_0000), y(0x0000_0000), y(0x0000_0000), y(0x8000_0000), y(0x8000_0000)]), // -0, -0
    (0x0000_0000, 0x0000_0000, [y(0x0000_0000), y(0x0000_0000), y(0x0000_0000), y(0x0000_0000), y(0x0000_0000), y(0x0000_0000)]), // +0, +0
    (0x7f80_0000, 0x3f80_0000, [y(0x7f80_0000), y(0x7f80_0000), y(0xff80_0000), y(0x7f80_0000), y(0x3f80_0000), y(0x7f80_0000)]), // +inf, 1
    (0x7f80_0000, 0xff80_0000, [NO, y(0x7f80_0000), y(0xff80_0000), y(0xff80_0000), y(0xff80_0000), y(0x7f80_0000)]),             // +inf, -inf
    (0x7f80_0000, 0x7f80_0000, [y(0x7f80_0000), NO, NO, y(0x7f80_0000), y(0x7f80_0000), y(0x7f80_0000)]),                         // +inf, +inf
    (0x7f80_0000, 0x0000_0000, [y(0x7f80_0000), y(0x7f80_0000), y(0xff80_0000), NO, y(0x0000_0000), y(0x7f80_0000)]),             // +inf, +0
    (0x8000_0000, 0xff80_0000, [y(0xff80_0000), y(0x7f80_0000), y(0xff80_0000), NO, y(0xff80_0000), y(0x8000_0000)]),             // -0, -inf
    (0x7fc0_0000, 0x3f80_0000, [NO, NO, NO, NO, y(0x3f80_0000), y(0x3f80_0000)]),                                                 // qNaN, 1
    (0x3f80_0000, 0x7fc0_0000, [NO, NO, NO, NO, y(0x3f80_0000), y(0x3f80_0000)]),                                                 // 1, qNaN
    (0x7fc0_0001, 0xffc0_0002, [NO, NO, NO, NO, y(0xffc0_0002), y(0xffc0_0002)]),                                                 // qNaN, -qNaN
    (0x7f80_0001, 0x3f80_0000, [NO, NO, NO, NO, NO, NO]),                                                                         // sNaN, 1
    (0x3f80_0000, 0xffbf_ffff, [NO, NO, NO, NO, NO, NO]),                                                                         // 1, -sNaN
    (0x7fc0_0000, 0x7f80_0000, [NO, NO, NO, NO, y(0x7f80_0000), y(0x7f80_0000)]),                                                 // qNaN, +inf
    (0x7fc0_0000, 0x8000_0000, [NO, NO, NO, NO, y(0x8000_0000), y(0x8000_0000)]),                                                 // qNaN, -0
    (0x3f80_0000, 0x3380_0000, [y(0x3f80_0000), y(0x3f7f_ffff), y(0xbf7f_ffff), y(0x3380_0000), y(0x3380_0000), y(0x3f80_0000)]), // 1, 2^-24: tie to even
    (0x3f80_0001, 0x3380_0000, [y(0x3f80_0002), y(0x3f80_0000), y(0xbf80_0000), y(0x3380_0001), y(0x3380_0000), y(0x3f80_0001)]), // 1 + ulp, 2^-24
    (0x3f80_0001, 0x3f80_0001, [y(0x4000_0001), y(0x0000_0000), y(0x0000_0000), y(0x3f80_0002), y(0x3f80_0001), y(0x3f80_0001)]), // (1 + ulp)^2 rounds
    (0x7f7f_ffff, 0x7f7f_ffff, [y(0x7f80_0000), y(0x0000_0000), y(0x0000_0000), y(0x7f80_0000), y(0x7f7f_ffff), y(0x7f7f_ffff)]), // max: overflow
    (0x7f7f_ffff, 0x7300_0000, [y(0x7f80_0000), y(0x7f7f_fffe), y(0xff7f_fffe), y(0x7f80_0000), y(0x7300_0000), y(0x7f7f_ffff)]), // max, half ulp
    (0x0080_0000, 0x3f00_0000, [y(0x3f00_0000), y(0xbf00_0000), y(0x3f00_0000), NO, y(0x0080_0000), y(0x3f00_0000)]),             // min normal, 0.5
    (0x0080_0001, 0x0080_0000, [y(0x0100_0000), NO, NO, NO, y(0x0080_0000), y(0x0080_0001)]),                                     // tiny difference
    (0x0080_0000, 0x8080_0000, [y(0x0000_0000), y(0x0100_0000), y(0x8100_0000), NO, y(0x8080_0000), y(0x0080_0000)]),             // x, -x
    (0x0000_0001, 0x3f80_0000, [NO, NO, NO, NO, NO, NO]),                                                                         // subnormal S0
    (0x3f80_0000, 0x807f_ffff, [NO, NO, NO, NO, NO, NO]),                                                                         // subnormal S1
    (0xc000_0000, 0xbf80_0000, [y(0xc040_0000), y(0xbf80_0000), y(0x3f80_0000), y(0x4000_0000), y(0xc000_0000), y(0xbf80_0000)]), // -2, -1
    (0xbf80_0000, 0x4000_0000, [y(0x3f80_0000), y(0xc040_0000), y(0x4040_0000), y(0xc000_0000), y(0xbf80_0000), y(0x4000_0000)]), // -1, 2
    (0x3fc0_0000, 0x3fc0_0000, [y(0x4040_0000), y(0x0000_0000), y(0x0000_0000), y(0x4010_0000), y(0x3fc0_0000), y(0x3fc0_0000)]), // 1.5, 1.5
    (0x3dcc_cccd, 0x3e4c_cccd, [y(0x3e99_999a), y(0xbdcc_cccd), y(0x3dcc_cccd), y(0x3ca3_d70b), y(0x3dcc_cccd), y(0x3e4c_cccd)]), // 0.1, 0.2
    (0x4040_0000, 0x3eaa_aaab, [y(0x4055_5555), y(0x402a_aaab), y(0xc02a_aaab), y(0x3f80_0000), y(0x3eaa_aaab), y(0x4040_0000)]), // 3, 1/3
    (0x3f80_0000, 0xbf80_0000, [y(0x0000_0000), y(0x4000_0000), y(0xc000_0000), y(0xbf80_0000), y(0xbf80_0000), y(0x3f80_0000)]), // 1, -1
    (0xbf80_0000, 0x3f80_0000, [y(0x0000_0000), y(0xc000_0000), y(0x4000_0000), y(0xbf80_0000), y(0xbf80_0000), y(0x3f80_0000)]), // -1, 1
    (0x1f80_0000, 0x1f80_0000, [y(0x2000_0000), y(0x0000_0000), y(0x0000_0000), NO, y(0x1f80_0000), y(0x1f80_0000)]),             // 2^-64 squared
    (0x5f80_0000, 0x5f80_0000, [y(0x6000_0000), y(0x0000_0000), y(0x0000_0000), y(0x7f80_0000), y(0x5f80_0000), y(0x5f80_0000)]), // 2^64 squared
];

fn is_nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > 0x7f80_0000
}

/// D for S0 = `a`, S1 = `b`: the guide's definitions through binary64. The
/// sum, difference or product of two binary32 values rounded once to binary64
/// and then to binary32 equals the directly rounded result (53 >= 2 * 24 + 2),
/// and binary64 holds every exact product and every exact sum below 2^-126.
/// `None` for a refused input or result.
fn expect(form: Form, a: u32, b: u32) -> Option<u32> {
    let subnormal = |x: u32| x & 0x7f80_0000 == 0 && x & 0x007f_ffff != 0;
    if subnormal(a) || subnormal(b) {
        return None;
    }
    if matches!(form, Form::Min | Form::Max) {
        let signalling = |x: u32| is_nan(x) && x & 0x0040_0000 == 0;
        if signalling(a) || signalling(b) {
            return None;
        }
        let (x, z) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
        // Pages 132-133, in the pseudocode's order.
        return Some(if is_nan(a) {
            b
        } else if is_nan(b) {
            a
        } else if a == 0x0000_0000 && b == 0x8000_0000 {
            if form == Form::Min { b } else { a }
        } else if a == 0x8000_0000 && b == 0x0000_0000 {
            if form == Form::Min { a } else { b }
        } else if form == Form::Min {
            if x < z { a } else { b }
        } else if x >= z {
            a
        } else {
            b
        });
    }
    if is_nan(a) || is_nan(b) {
        return None;
    }
    let (x, z) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
    let d = match form {
        Form::Add => x + z,
        Form::Sub => x - z,
        Form::Subrev => z - x,
        _ => x * z,
    };
    if d.is_nan() || (d != 0.0 && d.abs() < 2f64.powi(-126)) {
        return None;
    }
    Some((d as f32).to_bits())
}

const LITERAL: u32 = 255;
const VCC_LO: u32 = 106;

/// Bit 31 = 0, OP in bits 25-30, VDST 17-24, VSRC1 9-16, SRC0 0-8 (13.3.1).
fn vop2(op: u32, dst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | dst << 17 | vsrc1 << 9 | src0
}

const fn vgpr(n: u32) -> u32 {
    256 + n
}

/// `v_mov_b32 v<reg>, literal`.
fn vmov(reg: u32, value: u32) -> [u32; 2] {
    [0x7e00_02ff | reg << 17, value]
}

/// `s_mov_b32 s<reg>, literal`.
fn smov(reg: u32, value: u32) -> [u32; 2] {
    [0xbe80_03ff | reg << 16, value]
}

/// `v_xor_b32 v<dst>, v<a>, v<b>`.
fn vxor(dst: u32, a: u32, b: u32) -> u32 {
    29 << 25 | dst << 17 | b << 9 | vgpr(a)
}

const FULL_EXEC: u32 = 0xbefe_03c1;
const SCC_SET: u32 = 0xbf06_8080;
const SCC_CLEAR: u32 = 0xbf06_8180;
/// v6 = vcc_lo; v2 ^= v6; `s_cbranch_scc0` over v2 ^= v1.
const FLAG_TAIL: [u32; 4] = [0x7e0c_026a, 0x3a04_0506, 0xbf84_0001, 0x3a04_0501];
/// `buffer_load_dword v3, v0, s[0:3], 0 idxen` without the wait.
const PENDING_LOAD_V3: [u32; 2] = [0xe030_2000, 0x8000_0300];
/// XORed with v1 to give each lane its own old VGPR before a masked write.
const OLD: u32 = 0x0bad_f00d;
const EXEC_MASKS: [u32; 5] = [0, 0x5555_5555, 0xaaaa_aaaa, 0x8000_0001, u32::MAX];

/// Load v1 = lane input, `setup`, the ALU words, `after`, v2 = v<result>, store v2.
fn program(setup: &[u32], alu: &[u32], after: &[u32], result: u32) -> Vec<u8> {
    let mut code = vec![0xe030_2000, 0x8000_0100, 0xbf8c_3f70];
    code.extend(setup);
    code.extend(alu);
    code.extend(after);
    if result != 2 {
        code.push(0x7e04_0300 | result);
    }
    code.extend([0xe070_2000, 0x8001_0200, 0xbbfd_0000, 0xbf81_0000]);
    words(&code)
}

fn alu_offset(setup: &[u32]) -> usize {
    12 + 4 * setup.len()
}

/// Every operand of the known table, then quarter steps -8..8, values one ulp
/// around powers of two 2^-126..2^127 in steps of 2^11 and their negatives,
/// and the subnormals and NaNs the domain refuses.
fn values() -> Vec<u32> {
    let mut values: Vec<u32> = KNOWN.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    values.extend((-32..=32).map(|k| (k as f32 * 0.25).to_bits()));
    for e in (1..=254).step_by(11) {
        let p = e << 23;
        for bits in [p - 1, p, p + 1] {
            values.extend([bits, bits | 0x8000_0000]);
        }
    }
    values.sort_unstable();
    values.dedup();
    values
}

/// The S1 values each lane program is run with: the known table's S1 column.
fn uniform_s1() -> Vec<u32> {
    let mut s1: Vec<u32> = KNOWN.iter().map(|&(_, b, _)| b).collect();
    s1.sort_unstable();
    s1.dedup();
    s1
}

/// The S0 lanes `form` accepts against S1 = `b` (or, with `swap`, the S1
/// lanes it accepts against S0 = `b`).
fn lanes(form: Form, b: u32, swap: bool) -> Vec<u32> {
    let mut lanes = values();
    lanes.retain(|&x| {
        if swap {
            expect(form, b, x).is_some()
        } else {
            expect(form, x, b).is_some()
        }
    });
    lanes
}

fn case(
    name: &'static str,
    family: &'static str,
    label: String,
    code: Vec<u8>,
    input: Vec<u32>,
    lane: impl Fn(usize, u32) -> u32,
) -> Case {
    let expected = input.iter().enumerate().map(|(n, &x)| lane(n, x)).collect();
    Case {
        form: name,
        family,
        label: format!("{name} {label}"),
        code,
        input,
        expected,
    }
}

fn masked(exec: u32, n: usize, d: u32, old: u32) -> u32 {
    if exec >> (n % 32) & 1 == 1 { d } else { old }
}

/// Uniform SRC0 values for the SGPR, vcc_lo and literal cases.
const UNIFORM: [u32; 5] = [
    0xbf80_0000,
    0x3f80_0001,
    0x8000_0000,
    0x7fc0_0000,
    0x7f80_0000,
];

/// S1 (v3) for the register, EXEC and flag cases.
const FIXED_S1: u32 = 0x4020_0000;

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (form, name) in FORMS {
        let op = form as u32;
        let f = move |a, b| expect(form, a, b).expect("accepted input");
        // v2 = op(v1 lanes, v3 = S1) for every S1 of the table.
        for b in uniform_s1() {
            let all = lanes(form, b, false);
            if all.is_empty() {
                continue;
            }
            let code = program(&vmov(3, b), &[vop2(op, 2, vgpr(1), 3)], &[], 2);
            let label = format!("{} lanes, s1 {b:#x}", all.len());
            cases.push(case(name, "f32 lanes", label, code, all, move |_, x| {
                f(x, b)
            }));
        }
        let all = lanes(form, FIXED_S1, false);
        // VDST equal to SRC0, v255 as VDST, v255 as VSRC1, and v1 as both sources.
        let code = program(&vmov(3, FIXED_S1), &[vop2(op, 1, vgpr(1), 3)], &[], 1);
        cases.push(case(
            name,
            "f32 registers",
            "v1 = v1, v3".into(),
            code,
            all.clone(),
            |_, x| f(x, FIXED_S1),
        ));
        let code = program(&vmov(3, FIXED_S1), &[vop2(op, 255, vgpr(1), 3)], &[], 255);
        cases.push(case(
            name,
            "f32 registers",
            "v255 = v1, v3".into(),
            code,
            all.clone(),
            |_, x| f(x, FIXED_S1),
        ));
        let code = program(&vmov(255, FIXED_S1), &[vop2(op, 2, vgpr(1), 255)], &[], 2);
        cases.push(case(
            name,
            "f32 registers",
            "v2 = v1, v255".into(),
            code,
            all.clone(),
            |_, x| f(x, FIXED_S1),
        ));
        let mut same = values();
        same.retain(|&x| expect(form, x, x).is_some());
        let code = program(&[], &[vop2(op, 2, vgpr(1), 1)], &[], 2);
        cases.push(case(
            name,
            "f32 registers",
            "v2 = v1, v1".into(),
            code,
            same,
            |_, x| f(x, x),
        ));
        // Uniform SRC0 against v1 lanes: s8, s105, vcc_lo and a literal dword.
        for value in UNIFORM {
            let all = lanes(form, value, true);
            if all.is_empty() {
                // The form refuses this source for every lane, so it has no
                // expected value and is not an accepted case. The refusal is
                // checked in `refused_inputs`, where the source is in a
                // register the compiler cannot see through.
                continue;
            }
            for reg in [8, 105, VCC_LO] {
                let code = program(&smov(reg, value), &[vop2(op, 2, reg, 1)], &[], 2);
                let label = format!("s{reg} = {value:#x}");
                cases.push(case(
                    name,
                    "f32 uniform source",
                    label,
                    code,
                    all.clone(),
                    |_, x| f(value, x),
                ));
            }
            let code = program(&[], &[vop2(op, 2, LITERAL, 1), value], &[], 2);
            let label = format!("literal {value:#x}");
            cases.push(case(
                name,
                "f32 uniform source",
                label,
                code,
                all.clone(),
                |_, x| f(value, x),
            ));
        }
        // EXEC: inactive lanes keep v2 = v1 ^ OLD.
        let alu = [vop2(op, 2, vgpr(1), 3)];
        let mut setup = vmov(3, FIXED_S1).to_vec();
        setup.extend(vmov(5, OLD));
        setup.push(vxor(2, 5, 1));
        for exec in EXEC_MASKS {
            let mut before = setup.clone();
            before.extend(smov(126, exec));
            let code = program(&before, &alu, &[FULL_EXEC], 2);
            let label = format!("exec {exec:#x}");
            cases.push(case(name, "f32 exec", label, code, all.clone(), |n, x| {
                masked(exec, n, f(x, FIXED_S1), x ^ OLD)
            }));
        }
        // VCC_LO and SCC are not written: D ^ VCC_LO, XORed with S0 when SCC is set.
        for vcc in [0, u32::MAX] {
            for scc in [false, true] {
                let mut before = vmov(3, FIXED_S1).to_vec();
                before.extend(smov(VCC_LO, vcc));
                before.push(if scc { SCC_SET } else { SCC_CLEAR });
                let code = program(&before, &alu, &FLAG_TAIL, 2);
                let label = format!("vcc_lo {vcc:#x}, scc {scc}");
                cases.push(case(
                    name,
                    "f32 no vcc or scc write",
                    label,
                    code,
                    all.clone(),
                    |_, x| f(x, FIXED_S1) ^ vcc ^ if scc { x } else { 0 },
                ));
            }
        }
    }
    cases
}

/// The lane programs are one per S1 with an accepted lane; the rest per form:
/// four register cases, four uniform sources for five values, five EXEC masks
/// and four flag seeds.
fn counts(cases: &[Case]) -> BTreeMap<&'static str, usize> {
    let lane_programs: usize = FORMS
        .iter()
        .map(|&(form, _)| {
            uniform_s1()
                .into_iter()
                .filter(|&b| !lanes(form, b, false).is_empty())
                .count()
        })
        .sum();
    assert!(cases.iter().all(|case| !case.input.is_empty()));
    // A uniform source the form refuses for every lane has no accepted case.
    let uniform_sources: usize = FORMS
        .iter()
        .map(|&(form, _)| {
            UNIFORM
                .iter()
                .filter(|&&value| !lanes(form, value, true).is_empty())
                .count()
        })
        .sum();
    BTreeMap::from([
        ("f32 exec", 6 * 5),
        ("f32 lanes", lane_programs),
        ("f32 no vcc or scc write", 6 * 4),
        ("f32 registers", 6 * 4),
        ("f32 uniform source", 4 * uniform_sources),
    ])
}

/// Programs the decoders refuse at the ALU word: a missing literal, the
/// integer inline constants and every other selector outside the subset (with
/// the SDWA and DPP extension dword where one belongs), and a read and a write
/// of a VGPR with a load outstanding.
fn rejections() -> Vec<(String, Vec<u8>, usize)> {
    let setup = vmov(3, 0x3fc0_0000);
    let offset = alu_offset(&setup);
    let extension = |src0: u32| match src0 {
        249 => vec![0x0606_0601],
        250 => vec![0xff00_e401],
        233 | 234 => vec![0xfac6_8801],
        _ => vec![],
    };
    let mut all = Vec::new();
    for (form, name) in FORMS {
        let op = form as u32;
        let mut code = vec![0xe030_2000, 0x8000_0100, 0xbf8c_3f70];
        code.extend(setup);
        code.push(vop2(op, 2, LITERAL, 3));
        all.push((format!("{name} missing literal"), words(&code), offset));
        for src0 in 107..=254 {
            let mut alu = vec![vop2(op, 2, src0, 3)];
            alu.extend(extension(src0));
            let code = program(&setup, &alu, &[], 2);
            all.push((format!("{name} src0 {src0}"), code, offset));
        }
        let mut pending = setup.to_vec();
        pending.extend(PENDING_LOAD_V3);
        let at = alu_offset(&pending);
        for (dst, src0, src1, what) in [
            (2, 3, 1, "SRC0 read"),
            (2, 1, 3, "VSRC1 read"),
            (3, 1, 1, "write"),
        ] {
            let code = program(&pending, &[vop2(op, dst, vgpr(src0), src1)], &[], 2);
            all.push((
                format!("{name} {what} of v3 with its load outstanding"),
                code,
                at,
            ));
        }
    }
    all
}

const REJECTIONS: usize = 6 * (1 + 148 + 3);

/// Programs both interpreters refuse at run time at the ALU word: each refused
/// pair of the table as S0 in one lane against S1 = v3, and as a literal SRC0.
/// The compiler cannot see lane data.
fn refused_inputs() -> Vec<(String, Vec<u8>, Vec<u32>, usize)> {
    let mut all = Vec::new();
    for (form, name) in FORMS {
        let op = form as u32;
        for &(a, b, expected) in KNOWN.iter().filter(|(_, _, d)| d[column(form)].is_none()) {
            assert_eq!(expected[column(form)], expect(form, a, b));
            let setup = vmov(3, b);
            let at = alu_offset(&setup);
            let mut input = lanes(form, b, false);
            input.truncate(5);
            input.insert(input.len().min(3), a);
            let code = program(&setup, &[vop2(op, 2, vgpr(1), 3)], &[], 2);
            all.push((format!("{name} lane {a:#x}, s1 {b:#x}"), code, input, at));
            let code = program(&setup, &[vop2(op, 2, LITERAL, 3), a], &[], 2);
            all.push((
                format!("{name} literal {a:#x}, s1 {b:#x}"),
                code,
                vec![0, 1],
                at,
            ));
        }
        // A uniform SRC0 that the form refuses for every lane. The compiler
        // cannot see through the register, so the interpreters must refuse the
        // program at the ALU word instead of returning a value.
        for value in UNIFORM {
            if !lanes(form, value, true).is_empty() {
                continue;
            }
            let setup = smov(8, value);
            let at = alu_offset(&setup);
            let code = program(&setup, &[vop2(op, 2, 8, 1)], &[], 2);
            all.push((
                format!("{name} uniform src0 {value:#x}"),
                code,
                vec![0, 1],
                at,
            ));
        }
    }
    all
}

fn assert_rejected_at(label: &str, code: &[u8], offset: usize, input: &[u32]) {
    match xenora_ref::execute(code, input) {
        Ok(_) => panic!("reference accepted {label}"),
        Err(error) => assert_eq!(
            error.offset, offset,
            "reference rejected {label} elsewhere: {error:?}"
        ),
    }
    match independent_ref::execute(code, input) {
        Ok(_) => panic!("independent accepted {label}"),
        Err(error) => assert!(
            error.starts_with(&format!("{offset:#x}:")),
            "independent rejected {label} elsewhere: {error}"
        ),
    }
}

fn checked_cases() -> Vec<Case> {
    let cases = cases();
    let mut counted = BTreeMap::new();
    for case in &cases {
        *counted.entry(case.family).or_default() += 1;
    }
    assert_eq!(counted, counts(&cases));
    cases
}

fn failures(
    cases: &[Case],
    execute: impl Fn(&[u8], &[u32]) -> Result<Vec<u32>, String>,
) -> Vec<String> {
    cases
        .iter()
        .filter_map(|case| {
            let output = execute(&case.code, &case.input);
            (output.as_deref().ok() != Some(case.expected.as_slice()))
                .then(|| format!("  {}: {output:?}", case.label))
        })
        .collect()
}

fn lane_count(cases: &[Case]) -> usize {
    cases.iter().map(|case| case.input.len()).sum()
}

#[test]
fn transcription_reproduces_the_hand_table() {
    for (a, b, expected) in KNOWN {
        for (form, name) in FORMS {
            assert_eq!(
                expect(form, a, b),
                expected[column(form)],
                "{name}({a:#x}, {b:#x})"
            );
        }
    }
}

/// Negative control: the guide's MIN/MAX rule is not IEEE 754-2008 minNum
/// and maxNum as `f32::min`/`max` implement them on signed zeros, and NaN
/// propagation is not what the table expects of the arithmetic forms.
#[test]
fn negative_controls_disagree_with_the_table() {
    // f32::min(+0, -0) may return either zero; the guide requires -0 for both orders.
    let zero_pairs = [(0x0000_0000, 0x8000_0000), (0x8000_0000, 0x0000_0000)];
    // A signed-zero tie broken towards the first operand, or towards the
    // second, agrees with the table on one order and not the other, so the
    // table detects either rule. The control is written against those two
    // rules and not against the host's `f32::min`/`max`, whose signed-zero
    // tie-break is unspecified and on this target already agrees with the
    // guide, so a host comparison would prove nothing here.
    for form in [Form::Min, Form::Max] {
        for take_first in [true, false] {
            let tie = |a: u32, b: u32| if take_first { a } else { b };
            assert!(
                zero_pairs.iter().any(|&(a, b)| {
                    Some(tie(a, b)) != expect(form, a, b) || Some(tie(b, a)) != expect(form, b, a)
                }),
                "{form:?} accepts a signed-zero tie broken by the operand position"
            );
        }
    }
    // A plain `a < b ? a : b` would return NaN for (qNaN, 1); the guide gives 1.
    let naive = |a: u32, b: u32| {
        if f32::from_bits(a) < f32::from_bits(b) {
            a
        } else {
            b
        }
    };
    assert_ne!(
        Some(naive(0x3f80_0000, 0x7fc0_0000)),
        expect(Form::Min, 0x3f80_0000, 0x7fc0_0000)
    );
    // Truncating instead of rounding to nearest even changes (1 + ulp)^2 and 0.1 + 0.2.
    let truncate = |d: f64| f32::from_bits((d as f32).to_bits() - u32::from((d as f32) as f64 > d));
    assert_ne!(
        Some(
            truncate(
                f64::from(f32::from_bits(0x3dcc_cccd)) + f64::from(f32::from_bits(0x3e4c_cccd))
            )
            .to_bits()
        ),
        expect(Form::Add, 0x3dcc_cccd, 0x3e4c_cccd)
    );
    // Flushing the tiny difference to zero would be accepted silently; it is refused.
    assert_eq!(expect(Form::Sub, 0x0080_0001, 0x0080_0000), None);
}

#[test]
fn binary32_forms_match_the_guide_on_the_reference() {
    let cases = checked_cases();
    let failed = failures(&cases, |code, input| {
        xenora_ref::execute(code, input).map_err(|error| format!("{error:?}"))
    });
    assert!(
        failed.is_empty(),
        "reference disagrees with the guide:\n{}",
        failed.join("\n")
    );
    eprintln!(
        "vop2_float_arithmetic: {} reference cases ({} lanes)",
        cases.len(),
        lane_count(&cases)
    );
}

#[test]
fn binary32_forms_match_the_guide_on_the_independent_interpreter() {
    let cases = checked_cases();
    let failed = failures(&cases, independent_ref::execute);
    assert!(
        failed.is_empty(),
        "independent interpreter disagrees:\n{}",
        failed.join("\n")
    );
    eprintln!(
        "vop2_float_arithmetic: {} independent-interpreter cases",
        cases.len()
    );
}

#[test]
fn unsupported_binary32_encodings_and_inputs_are_refused() {
    let rejections = rejections();
    assert_eq!(rejections.len(), REJECTIONS);
    let input = lanes(Form::Add, FIXED_S1, false);
    for (label, code, offset) in &rejections {
        assert_rejected_at(label, code, *offset, &input);
    }
    let refused = refused_inputs();
    let expected: usize = FORMS
        .iter()
        .map(|&(form, _)| {
            2 * KNOWN
                .iter()
                .filter(|(_, _, d)| d[column(form)].is_none())
                .count()
                + UNIFORM
                    .iter()
                    .filter(|&&value| lanes(form, value, true).is_empty())
                    .count()
        })
        .sum();
    assert_eq!(refused.len(), expected);
    for (label, code, input, offset) in &refused {
        assert_rejected_at(label, code, *offset, input);
    }
    eprintln!(
        "vop2_float_arithmetic: {} rejected encodings, {} refused inputs",
        rejections.len(),
        refused.len()
    );
}

fn check_compiler_rejections() {
    for (label, code, offset) in rejections() {
        let at = match compile(Architecture::Gfx1030, &code) {
            Err(CompileError::Decode(error)) => error.offset,
            Err(CompileError::Verify { offset, .. }) => offset,
            other => panic!("compiler did not reject {label}: {other:?}"),
        };
        assert_eq!(at, offset, "compiler rejected {label} elsewhere");
    }
}

#[test]
fn compiler_matches_the_binary32_operand_domain() {
    for case in &checked_cases() {
        compile(Architecture::Gfx1030, &case.code)
            .unwrap_or_else(|error| panic!("compiler rejected {}: {error:?}", case.label));
    }
    check_compiler_rejections();
}

#[test]
#[ignore = "requires a native Metal device with API and shader validation"]
fn binary32_forms_match_the_guide_on_native_metal() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("MTL_SHADER_VALIDATION").as_deref(), Ok("1"));
    let cases = checked_cases();
    check_compiler_rejections();
    let (failures, programs) = harness::check_metal(&cases);
    failures.assert_none("native Metal disagrees with the guide");
    eprintln!(
        "vop2_float_arithmetic: {} Metal cases ({} lanes) in {programs} programs",
        cases.len(),
        lane_count(&cases)
    );
}
