// SPDX-License-Identifier: GPL-3.0-or-later

//! Independent integer semantics tests over LLVM-assembled instructions.
//!
//! `tests/x86/integer.txt` pairs encodings with Intel-syntax text (see
//! `tests/x86/assemble.py`). The reference below reads only that text, never the
//! bytes, the registry, the decoder or the production semantics. It works bit by
//! bit: ripple-carry addition, single-step shifts, shift-and-add multiplication
//! and long division, following the operation sections of the Intel SDM Vol. 2.
//! Flags the architecture leaves undefined are not compared with the reference,
//! but the interpreter and the DBT must still agree on the complete state.

use std::collections::{BTreeMap, BTreeSet};
use xenora_loader::cpu::{CpuState, ExitReason, Interpreter, RFlags, X87State};
use xenora_loader::dbt::{DbtEngine, is_native};
use xenora_loader::memory::{GuestMemory, MemoryError, MemoryPermissions};
use xenora_loader::x86_decode::decode;
use xenora_loader::x86_table::{FORMS, Op};

const FIXTURE: &str = include_str!("x86/integer.txt");
const ENTRY: u64 = 0x1000;
const DATA: u64 = 0x2000;
/// Address of `[rsi + 4*rdi + 16]` with RSI = DATA and RDI = 4, and the
/// absolute address of the `movabs` accumulator forms.
const OPERAND: u64 = DATA + 32;
const STACK: u64 = 0x3000;
const STACK_TOP: u64 = STACK + 0x80;

struct Case {
    code: Vec<u8>,
    text: &'static str,
}

fn cases() -> Vec<Case> {
    FIXTURE
        .lines()
        .map(|line| {
            let (hex, text) = line.split_once(" | ").expect("separator");
            let code = hex
                .split(' ')
                .map(|byte| u8::from_str_radix(byte, 16).expect("hex"))
                .collect();
            Case { code, text }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Independent reference
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Flags {
    cf: bool,
    zf: bool,
    sf: bool,
    of: bool,
    pf: bool,
    af: bool,
}

#[derive(Clone)]
struct Machine {
    gpr: [u64; 16],
    flags: Flags,
    /// Qword at the memory operand.
    operand: u64,
    /// Stack qwords at STACK_TOP - 8 and STACK_TOP.
    pushed: u64,
    top: u64,
    /// x87 control, status and tag words.
    x87: [u16; 3],
    /// The guest time-stamp counter: the value the next RDTSC reads.
    tsc: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Place {
    Reg { index: usize, bits: u32, high: bool },
    Mem { bits: u32 },
    Imm(i128),
}

fn parse_place(text: &str) -> Place {
    if let Some((width, _)) = text.split_once(" ptr ") {
        let bits = match width {
            "byte" => 8,
            "word" => 16,
            "dword" => 32,
            _ => 64,
        };
        return Place::Mem { bits };
    }
    const HIGH: [&str; 4] = ["ah", "ch", "dh", "bh"];
    if let Some(index) = HIGH.iter().position(|name| *name == text) {
        return Place::Reg {
            index,
            bits: 8,
            high: true,
        };
    }
    const LOW: [&str; 8] = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"];
    for (index, name) in LOW.iter().enumerate() {
        let byte = if index < 4 {
            format!("{}l", &name[..1])
        } else {
            format!("{name}l")
        };
        for (candidate, bits) in [
            (format!("r{name}"), 64),
            (format!("e{name}"), 32),
            (name.to_string(), 16),
            (byte, 8),
        ] {
            if candidate == text {
                return Place::Reg {
                    index,
                    bits,
                    high: false,
                };
            }
        }
    }
    for index in 8..16 {
        for (suffix, bits) in [("", 64), ("d", 32), ("w", 16), ("b", 8)] {
            if format!("r{index}{suffix}") == text {
                return Place::Reg {
                    index,
                    bits,
                    high: false,
                };
            }
        }
    }
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let value = match digits.strip_prefix("0x") {
        Some(hex) => i128::from_str_radix(hex, 16),
        None => digits.parse(),
    }
    .unwrap_or_else(|_| panic!("operand {text:?}"));
    Place::Imm(if negative { -value } else { value })
}

fn bits_of(place: Place) -> u32 {
    match place {
        Place::Reg { bits, .. } | Place::Mem { bits } => bits,
        Place::Imm(_) => 64,
    }
}

fn truncate(value: u128, bits: u32) -> u128 {
    value % (1u128 << bits)
}

fn bit(value: u128, index: u32) -> bool {
    (value / (1u128 << index)) % 2 == 1
}

/// Two's-complement value of the low `bits`.
fn signed(value: u128, bits: u32) -> i128 {
    let value = truncate(value, bits);
    if bit(value, bits - 1) {
        value as i128 - (1i128 << bits)
    } else {
        value as i128
    }
}

/// Representation of a signed value in `bits`.
fn encode(value: i128, bits: u32) -> u128 {
    value.rem_euclid(1i128 << bits) as u128
}

fn read(m: &Machine, place: Place, bits: u32) -> u128 {
    match place {
        Place::Reg { index, bits, high } => {
            truncate(u128::from(m.gpr[index]) >> if high { 8 } else { 0 }, bits)
        }
        Place::Mem { bits } => truncate(u128::from(m.operand), bits),
        Place::Imm(value) => encode(value, bits),
    }
}

fn replace(old: u64, shift: u32, bits: u32, value: u128) -> u64 {
    let mut out = u128::from(old);
    for index in 0..bits {
        let weight = 1u128 << (shift + index);
        if bit(out, shift + index) {
            out -= weight;
        }
        if bit(value, index) {
            out += weight;
        }
    }
    out as u64
}

fn write(m: &mut Machine, place: Place, value: u128) {
    match place {
        Place::Reg { index, bits, high } => {
            m.gpr[index] = match bits {
                32 => truncate(value, 32) as u64,
                _ => replace(m.gpr[index], if high { 8 } else { 0 }, bits, value),
            }
        }
        Place::Mem { bits } => m.operand = replace(m.operand, 0, bits, value),
        Place::Imm(_) => panic!("write to immediate"),
    }
}

/// Ripple-carry addition: result, carry out, carry into the top bit, carry out
/// of bit 3.
fn ripple(a: u128, b: u128, carry_in: bool, bits: u32) -> (u128, bool, bool, bool) {
    let (mut carry, mut into_top, mut nibble, mut out) = (carry_in, false, false, 0u128);
    for index in 0..bits {
        if index == bits - 1 {
            into_top = carry;
        }
        let total = u8::from(bit(a, index)) + u8::from(bit(b, index)) + u8::from(carry);
        if total % 2 == 1 {
            out += 1u128 << index;
        }
        carry = total >= 2;
        if index == 3 {
            nibble = carry;
        }
    }
    (out, carry, into_top, nibble)
}

fn set_result(flags: &mut Flags, value: u128, bits: u32) {
    flags.zf = (0..bits).all(|index| !bit(value, index));
    flags.sf = bit(value, bits - 1);
    flags.pf = (0..8).filter(|index| bit(value, *index)).count() % 2 == 0;
}

fn arithmetic(m: &mut Machine, a: u128, b: u128, subtract: bool, carry: bool, bits: u32) -> u128 {
    // a - b - borrow = a + !b + !borrow, with every carry inverted.
    let operand = if subtract { (1u128 << bits) - 1 - b } else { b };
    let (value, out, into_top, nibble) = ripple(a, operand, carry != subtract, bits);
    m.flags.cf = out != subtract;
    m.flags.af = nibble != subtract;
    m.flags.of = out != into_top;
    set_result(&mut m.flags, value, bits);
    value
}

fn holds(flags: &Flags, code: &str) -> bool {
    match code {
        "o" => flags.of,
        "no" => !flags.of,
        "b" => flags.cf,
        "ae" => !flags.cf,
        "e" => flags.zf,
        "ne" => !flags.zf,
        "be" => flags.cf || flags.zf,
        "a" => !flags.cf && !flags.zf,
        "s" => flags.sf,
        "ns" => !flags.sf,
        "p" => flags.pf,
        "np" => !flags.pf,
        "l" => flags.sf != flags.of,
        "ge" => flags.sf == flags.of,
        "le" => flags.zf || flags.sf != flags.of,
        "g" => !flags.zf && flags.sf == flags.of,
        _ => panic!("condition {code}"),
    }
}

/// Flags left undefined by an instruction, not compared with the reference.
#[derive(Clone, Copy, Default)]
struct Undefined {
    cf: bool,
    zf: bool,
    sf: bool,
    of: bool,
    pf: bool,
    af: bool,
    /// The destination itself is undefined (BSF/BSR of zero, or a 16-bit
    /// SHLD/SHRD count above 16).
    destination: bool,
}

/// Divide a 2n-bit value by schoolbook long division.
fn long_division(dividend: u128, divisor: u128, width: u32) -> (u128, u128) {
    let (mut quotient, mut remainder) = (0u128, 0u128);
    for index in (0..width).rev() {
        remainder = remainder * 2 + u128::from(bit(dividend, index));
        quotient *= 2;
        if remainder >= divisor {
            remainder -= divisor;
            quotient += 1;
        }
    }
    (quotient, remainder)
}

/// Product by shift-and-add of two values below 2^64.
fn shift_and_add(a: u128, b: u128) -> u128 {
    (0..64)
        .filter(|index| bit(b, *index))
        .fold(0, |sum, index| sum + a * (1u128 << index))
}

enum Outcome {
    Done(Undefined),
    DivideError,
}

fn reference(m: &mut Machine, text: &str) -> Outcome {
    // LOCK changes no result on a machine that runs one instruction at a time.
    let text = text.strip_prefix("lock ").unwrap_or(text);
    let (name, rest) = text.split_once(' ').unwrap_or((text, ""));
    let places: Vec<Place> = if rest.is_empty() || name == "lea" || name == "nop" {
        Vec::new()
    } else {
        rest.split(", ").map(parse_place).collect()
    };
    let bits = places.first().map_or(64, |place| bits_of(*place));
    let none = Undefined::default();
    let af = Undefined { af: true, ..none };
    const RAX: usize = 0;
    const RDX: usize = 2;
    let rax = |bits| Place::Reg {
        index: RAX,
        bits,
        high: false,
    };
    let rdx = |bits| Place::Reg {
        index: RDX,
        bits,
        high: false,
    };
    let ah = Place::Reg {
        index: RAX,
        bits: 8,
        high: true,
    };

    if let Some(code) = name.strip_prefix("cmov") {
        let value = read(
            m,
            if holds(&m.flags, code) {
                places[1]
            } else {
                places[0]
            },
            bits,
        );
        write(m, places[0], value);
        return Outcome::Done(none);
    }
    if let Some(code) = name.strip_prefix("set") {
        let value = u128::from(holds(&m.flags, code));
        write(m, places[0], value);
        return Outcome::Done(none);
    }
    match name {
        "add" | "adc" | "sub" | "sbb" | "cmp" => {
            let (a, b) = (read(m, places[0], bits), read(m, places[1], bits));
            let carry = m.flags.cf && matches!(name, "adc" | "sbb");
            let subtract = matches!(name, "sub" | "sbb" | "cmp");
            let value = arithmetic(m, a, b, subtract, carry, bits);
            if name != "cmp" {
                write(m, places[0], value);
            }
            Outcome::Done(none)
        }
        "and" | "or" | "xor" | "test" => {
            let (a, b) = (read(m, places[0], bits), read(m, places[1], bits));
            let mut value = 0;
            for index in 0..bits {
                let (x, y) = (bit(a, index), bit(b, index));
                let set = match name {
                    "or" => x || y,
                    "xor" => x != y,
                    _ => x && y,
                };
                value += u128::from(set) << index;
            }
            m.flags.cf = false;
            m.flags.of = false;
            set_result(&mut m.flags, value, bits);
            if name != "test" {
                write(m, places[0], value);
            }
            Outcome::Done(af)
        }
        "inc" | "dec" => {
            let carry = m.flags.cf;
            let a = read(m, places[0], bits);
            let value = arithmetic(m, a, 1, name == "dec", false, bits);
            m.flags.cf = carry;
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "neg" => {
            let a = read(m, places[0], bits);
            let value = arithmetic(m, 0, a, true, false, bits);
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "not" => {
            let a = read(m, places[0], bits);
            write(m, places[0], (1u128 << bits) - 1 - a);
            Outcome::Done(none)
        }
        "shl" | "shr" | "sar" | "rol" | "ror" | "rcl" | "rcr" => {
            let count = match rest.split(", ").nth(1) {
                Some("cl") => u32::from(m.gpr[1] as u8),
                _ => read(m, places[1], 8) as u32,
            } % if bits == 64 { 64 } else { 32 };
            let start = read(m, places[0], bits);
            let mut value = start;
            let top = 1u128 << (bits - 1);
            let carry_in = m.flags.cf;
            let mut carry = carry_in;
            // RCL and RCR rotate through CF, so `bits + 1` steps come round again.
            let steps = if matches!(name, "rcl" | "rcr") {
                count % (bits + 1)
            } else {
                count
            };
            for _ in 0..steps {
                // One step at a time, as the SDM loop does.
                let (high, low) = (bit(value, bits - 1), bit(value, 0));
                (value, carry) = match name {
                    "shl" => (truncate(value * 2, bits), high),
                    "rol" => (truncate(value * 2, bits) + u128::from(high), high),
                    "rcl" => (truncate(value * 2, bits) + u128::from(carry), high),
                    "rcr" => (value / 2 + if carry { top } else { 0 }, low),
                    "shr" => (value / 2, low),
                    "sar" => (value / 2 + if high { top } else { 0 }, low),
                    _ => (value / 2 + if low { top } else { 0 }, low),
                };
            }
            write(m, places[0], value);
            let mut undefined = none;
            if count != 0 {
                m.flags.cf = carry;
                let (msb, next) = (bit(value, bits - 1), bit(value, bits - 2));
                if count == 1 {
                    m.flags.of = match name {
                        "shl" | "rol" | "rcl" => msb != carry,
                        "rcr" => bit(start, bits - 1) != carry_in,
                        "shr" => bit(start, bits - 1),
                        "sar" => false,
                        _ => msb != next,
                    };
                } else {
                    undefined.of = true;
                }
                if matches!(name, "shl" | "shr" | "sar") {
                    set_result(&mut m.flags, value, bits);
                    undefined.af = true;
                    undefined.cf = name != "sar" && count >= bits;
                }
            }
            Outcome::Done(undefined)
        }
        "shld" | "shrd" => {
            // Intel SDM Vol. 2B, SHLD and SHRD: the count is masked to 5 bits
            // (6 for 64-bit operands); a count above the operand size leaves
            // the destination and the flags undefined.
            let count = read(m, places[2], 8) as u32 % if bits == 64 { 64 } else { 32 };
            let start = read(m, places[0], bits);
            let mut fill = read(m, places[1], bits);
            if count > bits {
                return Outcome::Done(Undefined {
                    cf: true,
                    zf: true,
                    sf: true,
                    of: true,
                    pf: true,
                    af: true,
                    destination: true,
                });
            }
            let mut value = start;
            let mut carry = m.flags.cf;
            for _ in 0..count {
                // One bit at a time: the bit leaving the destination goes to
                // CF and the next bit of the source enters from the other end.
                if name == "shld" {
                    carry = bit(value, bits - 1);
                    value = truncate(value * 2, bits) + u128::from(bit(fill, bits - 1));
                    fill = truncate(fill * 2, bits);
                } else {
                    carry = bit(value, 0);
                    value = value / 2 + if bit(fill, 0) { 1u128 << (bits - 1) } else { 0 };
                    fill /= 2;
                }
            }
            write(m, places[0], value);
            let mut undefined = none;
            if count != 0 {
                m.flags.cf = carry;
                set_result(&mut m.flags, value, bits);
                undefined.af = true;
                if count == 1 {
                    m.flags.of = bit(value, bits - 1) != bit(start, bits - 1);
                } else {
                    undefined.of = true;
                }
            }
            Outcome::Done(undefined)
        }
        "mul" | "imul" => {
            let undefined = Undefined {
                zf: true,
                sf: true,
                pf: true,
                af: true,
                ..none
            };
            let (a, b, destination) = match places.len() {
                1 => (read(m, rax(bits), bits), read(m, places[0], bits), None),
                2 => (
                    read(m, places[0], bits),
                    read(m, places[1], bits),
                    Some(places[0]),
                ),
                _ => (
                    read(m, places[1], bits),
                    read(m, places[2], bits),
                    Some(places[0]),
                ),
            };
            let (negative, left, right) = if name == "mul" {
                (false, a, b)
            } else {
                let (x, y) = (signed(a, bits), signed(b, bits));
                ((x < 0) != (y < 0), x.unsigned_abs(), y.unsigned_abs())
            };
            let magnitude = shift_and_add(left, right);
            // Two's complement in 2n bits; 2^128 wraps to zero for n = 64.
            let product = if negative {
                magnitude.wrapping_neg()
            } else {
                magnitude
            };
            let low = truncate(product, bits);
            let high = truncate(product >> bits, bits);
            let overflow = if name == "mul" {
                high != 0
            } else {
                let sign = if bit(low, bits - 1) {
                    (1u128 << bits) - 1
                } else {
                    0
                };
                high != sign
            };
            match destination {
                Some(place) => write(m, place, low),
                None if bits == 8 => write(m, rax(16), high * 256 + low),
                None => {
                    write(m, rax(bits), low);
                    write(m, rdx(bits), high);
                }
            }
            m.flags.cf = overflow;
            m.flags.of = overflow;
            Outcome::Done(undefined)
        }
        "div" | "idiv" => {
            let divisor = read(m, places[0], bits);
            let dividend = if bits == 8 {
                read(m, rax(16), 16)
            } else {
                (read(m, rdx(bits), bits) << bits) + read(m, rax(bits), bits)
            };
            if divisor == 0 {
                return Outcome::DivideError;
            }
            let (quotient, remainder) = if name == "div" {
                let (q, r) = long_division(dividend, divisor, 2 * bits);
                if q >= 1u128 << bits {
                    return Outcome::DivideError;
                }
                (q, r)
            } else {
                let negative_dividend = bit(dividend, 2 * bits - 1);
                let negative_divisor = bit(divisor, bits - 1);
                let magnitude = if !negative_dividend {
                    dividend
                } else if bits == 64 {
                    dividend.wrapping_neg()
                } else {
                    (1u128 << (2 * bits)) - dividend
                };
                let d = if negative_divisor {
                    (1u128 << bits) - divisor
                } else {
                    divisor
                };
                let (q, r) = long_division(magnitude, d, 2 * bits);
                let negative = negative_dividend != negative_divisor;
                let limit = 1u128 << (bits - 1);
                if q > limit || (q == limit && !negative) {
                    return Outcome::DivideError;
                }
                let q = encode(if negative { -(q as i128) } else { q as i128 }, bits);
                let r = encode(
                    if negative_dividend {
                        -(r as i128)
                    } else {
                        r as i128
                    },
                    bits,
                );
                (q, r)
            };
            if bits == 8 {
                write(m, rax(16), remainder * 256 + quotient);
            } else {
                write(m, rax(bits), quotient);
                write(m, rdx(bits), remainder);
            }
            Outcome::Done(Undefined {
                cf: true,
                zf: true,
                sf: true,
                of: true,
                pf: true,
                af: true,
                destination: false,
            })
        }
        "crc32" => {
            // The definition: reflect both, append 32 zeros, divide modulo 2 by
            // the polynomial, and reflect the remainder.
            let reflect = |value: u128, bits: u32| -> u128 {
                (0..bits)
                    .filter(|index| bit(value, *index))
                    .map(|index| 1u128 << (bits - 1 - index))
                    .sum()
            };
            let source_bits = bits_of(places[1]);
            let source = read(m, places[1], source_bits);
            let initial = read(m, places[0], bits) % (1 << 32);
            let mut rest = reflect(source, source_bits) << 32 ^ reflect(initial, 32) << source_bits;
            for position in (32..32 + 64).rev() {
                if bit(rest, position) {
                    rest ^= 0x1_1EDC_6F41u128 << (position - 32);
                }
            }
            write(m, places[0], reflect(rest, 32));
            Outcome::Done(none)
        }
        "bsf" | "bsr" | "tzcnt" | "lzcnt" | "popcnt" => {
            let value = read(m, places[1], bits);
            let first = (0..bits).find(|index| bit(value, *index));
            let last = (0..bits).rev().find(|index| bit(value, *index));
            let mut undefined = Undefined {
                sf: true,
                of: true,
                pf: true,
                af: true,
                ..none
            };
            match name {
                "bsf" | "bsr" => {
                    undefined.cf = true;
                    m.flags.zf = first.is_none();
                    match if name == "bsf" { first } else { last } {
                        Some(index) => write(m, places[0], u128::from(index)),
                        None => undefined.destination = true,
                    }
                }
                "popcnt" => {
                    let count = (0..bits).filter(|index| bit(value, *index)).count();
                    write(m, places[0], count as u128);
                    m.flags = Flags {
                        zf: count == 0,
                        ..Flags::default()
                    };
                    undefined = none;
                }
                _ => {
                    let count = if name == "tzcnt" {
                        first.unwrap_or(bits)
                    } else {
                        last.map_or(bits, |index| bits - 1 - index)
                    };
                    write(m, places[0], u128::from(count));
                    m.flags.cf = count == bits;
                    m.flags.zf = count == 0;
                }
            }
            Outcome::Done(undefined)
        }
        "andn" => {
            // Intel SDM Vol. 2A, ANDN: the first source inverted, bitwise
            // with the second. CF and OF clear; PF and AF are undefined.
            let a = read(m, places[1], bits);
            let b = read(m, places[2], bits);
            let mut value = 0;
            for index in 0..bits {
                if !bit(a, index) && bit(b, index) {
                    value += 1u128 << index;
                }
            }
            write(m, places[0], value);
            m.flags.cf = false;
            m.flags.of = false;
            m.flags.zf = value == 0;
            m.flags.sf = bit(value, bits - 1);
            Outcome::Done(Undefined {
                pf: true,
                af: true,
                ..none
            })
        }
        "blsi" | "blsmsk" | "blsr" => {
            // Intel SDM Vol. 2A: lowest-set-bit forms of the single source.
            let source = read(m, places[1], bits);
            let zero = source == 0;
            let minus_one = truncate(source.wrapping_sub(1), bits);
            let negative = truncate((!source).wrapping_add(1), bits);
            let mut value = 0;
            for index in 0..bits {
                let set = match name {
                    "blsi" => bit(negative, index) && bit(source, index),
                    "blsmsk" => bit(minus_one, index) != bit(source, index),
                    _ => bit(minus_one, index) && bit(source, index),
                };
                if set {
                    value += 1u128 << index;
                }
            }
            write(m, places[0], value);
            m.flags.cf = if name == "blsi" { !zero } else { zero };
            m.flags.of = false;
            m.flags.zf = name != "blsmsk" && value == 0;
            m.flags.sf = bit(value, bits - 1);
            Outcome::Done(Undefined {
                pf: true,
                af: true,
                ..none
            })
        }
        "bextr" | "bzhi" => {
            // Intel SDM Vol. 2A: BEXTR takes START from control[7:0] and LEN
            // from control[15:8] and extracts that window of the source, zero
            // filled past its width; BZHI clears the bits at and above the
            // index in control[7:0] and reports an out-of-range index in CF.
            let source = read(m, places[1], bits);
            let control = read(m, places[2], bits);
            let low = control % 256;
            let mut value = 0;
            if name == "bextr" {
                let len = (control / 256) % 256;
                for index in 0..bits {
                    let position = u128::from(index);
                    if position >= low && position - low < len && bit(source, index) {
                        value += 1u128 << (position - low);
                    }
                }
                m.flags.cf = false;
                m.flags.of = false;
                m.flags.zf = value == 0;
                write(m, places[0], value);
                Outcome::Done(Undefined {
                    sf: true,
                    pf: true,
                    af: true,
                    ..none
                })
            } else {
                for index in 0..bits {
                    if u128::from(index) < low && bit(source, index) {
                        value += 1u128 << index;
                    }
                }
                m.flags.cf = low >= u128::from(bits);
                m.flags.of = false;
                m.flags.zf = value == 0;
                m.flags.sf = bit(value, bits - 1);
                write(m, places[0], value);
                Outcome::Done(Undefined {
                    pf: true,
                    af: true,
                    ..none
                })
            }
        }
        "shlx" | "shrx" | "sarx" => {
            // Intel SDM Vol. 2B: the count is the second source masked to the
            // operand width; no flags are written. Bit by bit, one step at a
            // time, as the operation section describes.
            let count = (read(m, places[2], bits) % u128::from(bits)) as u32;
            let mut value = read(m, places[1], bits);
            for _ in 0..count {
                value = match name {
                    "shlx" => truncate(value * 2, bits),
                    "shrx" => value / 2,
                    _ => {
                        // Arithmetic: the sign bit re-enters at the top.
                        value / 2
                            + if bit(value, bits - 1) {
                                1u128 << (bits - 1)
                            } else {
                                0
                            }
                    }
                };
            }
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "rorx" => {
            // Intel SDM Vol. 2B: rotate right by the immediate, masked to the
            // operand width; no flags are written.
            let count = (read(m, places[2], bits) % u128::from(bits)) as u32;
            let start = read(m, places[1], bits);
            let mut value = 0;
            for index in 0..bits {
                let from = (index + count) % bits;
                if bit(start, from) {
                    value += 1u128 << index;
                }
            }
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "pdep" | "pext" => {
            // Intel SDM Vol. 2A/2B: the second operand is the bit source and
            // the third is the mask. PDEP consumes low source bits in mask
            // order; PEXT compresses masked source bits downwards. Both leave
            // every flag alone. Work a bit at a time, as the operation sections
            // do, independently of the executor's shift implementation.
            let source = read(m, places[1], bits);
            let control = read(m, places[2], bits);
            let mut value = 0;
            if name == "pdep" {
                let mut source_index = 0;
                for destination in 0..bits {
                    if bit(control, destination) {
                        if bit(source, source_index) {
                            value += 1u128 << destination;
                        }
                        source_index += 1;
                    }
                }
            } else {
                let mut destination = 0;
                for source_index in 0..bits {
                    if bit(control, source_index) {
                        if bit(source, source_index) {
                            value += 1u128 << destination;
                        }
                        destination += 1;
                    }
                }
            }
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "mulx" => {
            // Intel SDM Vol. 2B MULX: the r/m operand is the unsigned
            // multiplicand and RDX is the implicit unsigned multiplier. The
            // ModRM register receives the high product; VEX.vvvv, printed as
            // the second operand, receives the low product. Build the product
            // by shifted additions so this reference does not share a multiply
            // instruction with the executor. The low write happens first so
            // that an identical first and second operand keeps the high half.
            // No flag is written.
            let multiplicand = read(m, places[2], bits);
            let multiplier = read(m, rdx(bits), bits);
            let limit = 1u128 << bits;
            let mut product = 0u128;
            for index in 0..bits {
                if bit(multiplier, index) {
                    // The full product fits in u128 at both supported widths.
                    product += multiplicand << index;
                }
            }
            write(m, places[1], product % limit);
            write(m, places[0], product / limit);
            Outcome::Done(none)
        }
        "bt" | "bts" | "btr" | "btc" => {
            let value = read(m, places[0], bits);
            let index = (read(m, places[1], bits) % u128::from(bits)) as u32;
            let was = bit(value, index);
            let now = match name {
                "bts" => true,
                "btr" => false,
                "btc" => !was,
                _ => was,
            };
            let without = value - (u128::from(was) << index);
            if name != "bt" {
                write(m, places[0], without + (u128::from(now) << index));
            }
            m.flags.cf = was;
            Outcome::Done(Undefined {
                sf: true,
                of: true,
                pf: true,
                af: true,
                ..none
            })
        }
        // MOVNTI stores like MOV; the non-temporal hint changes no result.
        "mov" | "movabs" | "movnti" => {
            let value = read(m, places[1], bits);
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "movzx" => {
            let value = read(m, places[1], bits_of(places[1]));
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "movsx" | "movsxd" => {
            let width = bits_of(places[1]);
            let value = encode(signed(read(m, places[1], width), width), bits);
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "lea" => {
            let (target, address) = rest.split_once(", ").expect("lea operands");
            let value = match address {
                "[rsi + 4*rdi + 16]" => i128::from(m.gpr[6]) + 4 * i128::from(m.gpr[7]) + 16,
                "[r9 - 5]" => i128::from(m.gpr[9]) - 5,
                "[8*r10 + 64]" => 8 * i128::from(m.gpr[10]) + 64,
                other => panic!("address {other}"),
            };
            let place = parse_place(target);
            write(m, place, encode(value, bits_of(place)));
            Outcome::Done(none)
        }
        "xchg" => {
            let (a, b) = (read(m, places[0], bits), read(m, places[1], bits));
            write(m, places[0], b);
            write(m, places[1], a);
            Outcome::Done(none)
        }
        "xadd" => {
            // TEMP := SRC + DEST; SRC := DEST; DEST := TEMP.
            let (a, b) = (read(m, places[0], bits), read(m, places[1], bits));
            let sum = arithmetic(m, a, b, false, false, bits);
            write(m, places[1], a);
            write(m, places[0], sum);
            Outcome::Done(none)
        }
        "cmpxchg" => {
            // The accumulator is compared with DEST as CMP would. On a match
            // DEST := SRC; otherwise accumulator := DEST and DEST := DEST.
            let accumulator = read(m, rax(bits), bits);
            let (destination, source) = (read(m, places[0], bits), read(m, places[1], bits));
            arithmetic(m, accumulator, destination, true, false, bits);
            if accumulator == destination {
                write(m, places[0], source);
            } else {
                write(m, rax(bits), destination);
                write(m, places[0], destination);
            }
            Outcome::Done(none)
        }
        "bswap" => {
            let value = read(m, places[0], bits);
            let bytes = bits / 8;
            let swapped = (0..bytes).fold(0u128, |out, index| {
                out + (truncate(value >> (8 * index), 8) << (8 * (bytes - 1 - index)))
            });
            write(m, places[0], swapped);
            Outcome::Done(none)
        }
        "cbw" | "cwde" | "cdqe" => {
            let to = match name {
                "cbw" => 16,
                "cwde" => 32,
                _ => 64,
            };
            let value = encode(signed(read(m, rax(to), to / 2), to / 2), to);
            write(m, rax(to), value);
            Outcome::Done(none)
        }
        "cwd" | "cdq" | "cqo" => {
            let width = match name {
                "cwd" => 16,
                "cdq" => 32,
                _ => 64,
            };
            let negative = bit(read(m, rax(width), width), width - 1);
            write(
                m,
                rdx(width),
                if negative { (1u128 << width) - 1 } else { 0 },
            );
            Outcome::Done(none)
        }
        "clc" => {
            m.flags.cf = false;
            Outcome::Done(none)
        }
        "stc" => {
            m.flags.cf = true;
            Outcome::Done(none)
        }
        "cmc" => {
            m.flags.cf = !m.flags.cf;
            Outcome::Done(none)
        }
        // AH bits 7 to 0 are SF, ZF, 0, AF, 0, PF, 1 and CF; OF has no bit.
        "lahf" => {
            let f = m.flags;
            let bits = [f.cf, true, f.pf, false, f.af, false, f.zf, f.sf];
            let value = (0..8)
                .filter(|&index| bits[index])
                .map(|index| 1u128 << index)
                .sum();
            write(m, ah, value);
            Outcome::Done(none)
        }
        "sahf" => {
            let value = read(m, ah, 8);
            m.flags.cf = bit(value, 0);
            m.flags.pf = bit(value, 2);
            m.flags.af = bit(value, 4);
            m.flags.zf = bit(value, 6);
            m.flags.sf = bit(value, 7);
            Outcome::Done(none)
        }
        // The fences order memory accesses and compute nothing.
        "nop" | "pause" | "lfence" | "mfence" | "sfence" => Outcome::Done(none),
        "fninit" => {
            // Intel SDM Vol. 2A, "FINIT/FNINIT", Operation: FPUControlWord :=
            // 037FH, FPUStatusWord := 0, FPUTagWord := FFFFH. Nothing else here
            // is x87 state.
            m.x87 = [0x037F, 0, 0xFFFF];
            Outcome::Done(none)
        }
        "cpuid" => {
            // Xenora's guest CPUID contract, restated: the vendor "Xenora guest"
            // and the features whose every instruction exists, TSC (EDX
            // bit 4), CX8 (EDX bit 8), CMOV (EDX bit 15), CMPXCHG16B (ECX bit
            // 13) and POPCNT (ECX bit 23) in leaf 1, BMI1 (EBX bit 3) and BMI2
            // (EBX bit 8) in leaf 7 subleaf 0, LAHF/SAHF, LZCNT and long
            // mode in leaf 0x8000_0001.
            let word = |text: &[u8; 4]| u32::from_le_bytes(*text);
            let values = match m.gpr[RAX] as u32 {
                0 => [7, word(b"Xeno"), word(b"uest"), word(b"ra g")],
                1 => [0, 0, 1 << 13 | 1 << 23, 1 << 4 | 1 << 8 | 1 << 15],
                7 => {
                    if m.gpr[1] as u32 == 0 {
                        [0, 1 << 3 | 1 << 8, 0, 0]
                    } else {
                        [0; 4]
                    }
                }
                0x8000_0000 => [0x8000_0001, 0, 0, 0],
                0x8000_0001 => [0, 0, 1 | 1 << 5, 1 << 29],
                _ => [0; 4],
            };
            // EAX, EBX, ECX and EDX, each zero-extended.
            for (index, value) in [0, 3, 1, 2].into_iter().zip(values) {
                m.gpr[index] = u64::from(value);
            }
            Outcome::Done(none)
        }
        "rdtsc" => {
            // Intel SDM Vol. 2B, "RDTSC": EDX takes the high-order 32 bits of
            // the counter and EAX the low-order 32, and the high-order 32 bits
            // of RAX and RDX are cleared. Xenora's counter then counts this
            // read, modulo 2^64.
            m.gpr[0] = m.tsc % (1 << 32);
            m.gpr[2] = m.tsc / (1 << 32);
            m.tsc = m.tsc.wrapping_add(1);
            Outcome::Done(none)
        }
        "push" => {
            m.pushed = read(m, places[0], 64) as u64;
            m.gpr[4] -= 8;
            Outcome::Done(none)
        }
        "pop" => {
            m.gpr[4] += 8;
            let value = u128::from(m.top);
            write(m, places[0], value);
            Outcome::Done(none)
        }
        "leave" => {
            m.gpr[4] = STACK_TOP + 8;
            m.gpr[5] = m.top;
            Outcome::Done(none)
        }
        other => panic!("no reference for {other}"),
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// Deterministic generator (xorshift64*), seeded per test.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn value(&mut self) -> u64 {
        const EDGES: [u64; 14] = [
            0,
            1,
            2,
            0xF,
            0x7F,
            0x80,
            0xFF,
            0x7FFF,
            0x8000,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFF,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
        ];
        match self.next() % 5 {
            0 => EDGES[(self.next() % 14) as usize],
            1 => EDGES[(self.next() % 14) as usize].wrapping_neg(),
            2 => self.next() % 256,
            _ => self.next(),
        }
    }

    fn machine(&mut self) -> Machine {
        let mut gpr = [0; 16];
        for register in &mut gpr {
            *register = self.value();
        }
        // Small high halves make most divisions succeed.
        if self.next().is_multiple_of(2) {
            gpr[2] = if self.next().is_multiple_of(2) {
                0
            } else {
                u64::MAX
            };
        }
        gpr[4] = STACK_TOP;
        gpr[5] = STACK_TOP;
        gpr[6] = DATA;
        gpr[7] = 4;
        let bits = self.next();
        Machine {
            gpr,
            flags: Flags {
                cf: bits & 1 != 0,
                zf: bits & 2 != 0,
                sf: bits & 4 != 0,
                of: bits & 8 != 0,
                pf: bits & 16 != 0,
                af: bits & 32 != 0,
            },
            operand: self.value(),
            pushed: self.next(),
            top: self.value(),
            // Any bit pattern: nothing interprets these words.
            x87: [self.next() as u16, self.next() as u16, self.next() as u16],
            tsc: self.value(),
        }
    }
}

fn load(machine: &Machine, code: &[u8]) -> (CpuState, GuestMemory) {
    let mut memory = GuestMemory::new();
    memory
        .map_region_with_data(ENTRY, code.len(), MemoryPermissions::READ_EXECUTE, code)
        .unwrap();
    for base in [DATA, STACK] {
        memory
            .map_region(base, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
    }
    memory.write_u64(OPERAND, machine.operand).unwrap();
    memory.write_u64(STACK_TOP - 8, machine.pushed).unwrap();
    memory.write_u64(STACK_TOP, machine.top).unwrap();
    memory.advance_time_stamp_counter_to(machine.tsc);
    let mut cpu = CpuState::new(ENTRY, 0);
    cpu.gpr = machine.gpr;
    let f = machine.flags;
    cpu.flags = RFlags {
        cf: f.cf,
        zf: f.zf,
        sf: f.sf,
        of: f.of,
        pf: f.pf,
        af: f.af,
    };
    let [control, status, tag] = machine.x87;
    cpu.x87 = X87State {
        control,
        status,
        tag,
    };
    (cpu, memory)
}

fn bytes(memory: &GuestMemory) -> Vec<u8> {
    let mut out = vec![0; 0x200];
    memory.read_bytes(DATA, &mut out[..0x100]).unwrap();
    memory.read_bytes(STACK, &mut out[0x100..]).unwrap();
    out
}

#[test]
fn every_fixture_instruction_matches_the_reference_in_both_engines() {
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    let mut divide_errors = 0;
    let mut divisions = 0;
    for case in cases() {
        let text = case.text;
        let insn = decode(ENTRY, |address| {
            case.code.get((address - ENTRY) as usize).copied().ok_or(())
        })
        .unwrap_or_else(|error| panic!("{text}: {error:?}"));
        assert_eq!(usize::from(insn.len), case.code.len(), "{text}");
        let native = is_native(&insn);
        let mut engine = DbtEngine::new(64 * 1024).unwrap();
        for _ in 0..48 {
            let machine = random.machine();
            let mut expected = machine.clone();
            let outcome = reference(&mut expected, text);

            let (cpu, mut memory) = load(&machine, &case.code);
            let interpreted = Interpreter::new(cpu.clone(), &mut memory, 1).run();
            let (mut dbt_cpu, mut dbt_memory) = load(&machine, &case.code);
            let translated = engine.run(&mut dbt_cpu, &mut dbt_memory, 1);

            assert_eq!(translated.exit_reason, interpreted.exit_reason, "{text}");
            assert_eq!(
                translated.final_state, interpreted.final_state,
                "{text} from {cpu:x?}"
            );
            assert_eq!(bytes(&dbt_memory), bytes(&memory), "{text}");
            // Only RDTSC moves the counter, also when an instruction faults.
            for (engine, counter) in [
                ("interpreter", memory.time_stamp_counter()),
                ("DBT", dbt_memory.time_stamp_counter()),
            ] {
                assert_eq!(
                    counter, expected.tsc,
                    "{text}: time-stamp counter on the {engine} from {:#x}",
                    machine.tsc
                );
            }

            let undefined = match outcome {
                Outcome::DivideError => {
                    divide_errors += 1;
                    assert_eq!(interpreted.exit_reason, ExitReason::DivideError, "{text}");
                    assert_eq!(interpreted.final_state, cpu, "{text}");
                    continue;
                }
                Outcome::Done(undefined) => undefined,
            };
            divisions += usize::from(text.contains("div"));
            let state = &interpreted.final_state;
            assert_eq!(
                interpreted.exit_reason,
                ExitReason::StepLimitExceeded,
                "{text}"
            );
            assert_eq!(state.rip, ENTRY + case.code.len() as u64, "{text}");
            let x87 = state.x87;
            assert_eq!(
                [x87.control, x87.status, x87.tag],
                expected.x87,
                "{text}: x87 words from {:04x?}",
                machine.x87
            );
            if !undefined.destination {
                assert_eq!(state.gpr, expected.gpr, "{text} from {cpu:x?}");
                assert_eq!(
                    memory.read_u64(OPERAND).unwrap(),
                    expected.operand,
                    "{text} from {cpu:x?}"
                );
            }
            assert_eq!(
                memory.read_u64(STACK_TOP - 8).unwrap(),
                expected.pushed,
                "{text}"
            );
            for (name, actual, wanted, skip) in [
                ("CF", state.flags.cf, expected.flags.cf, undefined.cf),
                ("ZF", state.flags.zf, expected.flags.zf, undefined.zf),
                ("SF", state.flags.sf, expected.flags.sf, undefined.sf),
                ("OF", state.flags.of, expected.flags.of, undefined.of),
                ("PF", state.flags.pf, expected.flags.pf, undefined.pf),
                ("AF", state.flags.af, expected.flags.af, undefined.af),
            ] {
                assert!(
                    skip || actual == wanted,
                    "{text}: {name} = {actual} from {cpu:x?}"
                );
            }
        }
        // Native forms must never reach the helper, and helper forms always must.
        assert_eq!(engine.metrics.interpreter_fallbacks == 0, native, "{text}");
    }
    assert!(
        divide_errors > 50 && divisions > 50,
        "{divide_errors} {divisions}"
    );
}

#[test]
fn every_declared_form_is_executed_by_a_fixture() {
    let mut seen = BTreeSet::new();
    let vector = include_str!("x86/vector.txt");
    let float = include_str!("x86/float.txt");
    let avx = include_str!("x86/avx.txt");
    let forms = include_str!("x86/avx_forms.txt");
    let fma = include_str!("x86/fma.txt");
    let avx2 = include_str!("x86/avx2.txt");
    let f16c = include_str!("x86/f16c.txt");
    let lines = FIXTURE.lines().chain(vector.lines()).chain(float.lines());
    for line in lines
        .chain(avx.lines())
        .chain(forms.lines())
        .chain(fma.lines())
        .chain(avx2.lines())
        .chain(f16c.lines())
    {
        let (hex, _) = line.split_once(" | ").unwrap();
        let code: Vec<u8> = hex
            .split(' ')
            .map(|byte| u8::from_str_radix(byte, 16).unwrap())
            .collect();
        let insn = decode(ENTRY, |address| {
            code.get((address - ENTRY) as usize).copied().ok_or(())
        })
        .unwrap_or_else(|error| panic!("{line}: {error:?}"));
        seen.insert(usize::from(insn.form));
    }
    // Control flow and exits are covered by the tests below. String
    // instructions and the direction flag need their own memory layout and
    // are covered form by form in `tests/cpu_rep.rs` and, for LODS, SCAS and
    // CMPS, `tests/cpu_string_compare.rs`. CMPXCHG8B and CMPXCHG16B take an
    // 8- or 16-byte operand and two register pairs, and are covered in
    // `tests/x86_cmpxchg_pair.rs`.
    let separate = [
        Op::Ldmxcsr,
        Op::Jcc,
        Op::Loop,
        Op::Loope,
        Op::Loopne,
        Op::Jrcxz,
        Op::Jmp,
        Op::Call,
        Op::Ret,
        Op::Hlt,
        Op::Syscall,
        Op::Movs,
        Op::Stos,
        Op::Lods,
        Op::Scas,
        Op::Cmps,
        Op::Cld,
        Op::Std,
        Op::Cmpxchg8b,
    ];
    let mut missing: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (index, form) in FORMS.iter().enumerate() {
        if !separate.contains(&form.op) && !seen.contains(&index) {
            missing.entry(form.op.name()).or_default().push(format!(
                "{:?}/{:?}/{:?}",
                form.encoding, form.size, form.immediate
            ));
        }
    }
    assert!(missing.is_empty(), "forms without a fixture: {missing:#?}");
}

#[test]
fn random_sequences_leave_identical_state_in_both_engines() {
    // Exercises dead-flag elimination, flags across helper calls and block
    // boundaries, and faults in the middle of a block.
    let cases = cases();
    let mut random = Random(0xD1B5_4A32_D192_ED03);
    for round in 0..4000 {
        let count = 2 + random.next() % 30;
        let mut code = Vec::new();
        let mut text = Vec::new();
        for _ in 0..count {
            let case = &cases[(random.next() % cases.len() as u64) as usize];
            // Keep the address registers and the stack intact.
            if ["sil", "rsi", "leave"]
                .iter()
                .any(|name| case.text.starts_with(name))
                || case.text == "mov sil, dil"
            {
                continue;
            }
            code.extend_from_slice(&case.code);
            text.push(case.text);
        }
        code.push(0xF4);
        let machine = random.machine();
        let budget = 1 + (random.next() % 40) as usize;
        let (cpu, mut memory) = load(&machine, &code);
        let interpreted = Interpreter::new(cpu, &mut memory, budget).run();
        let (mut cpu, mut dbt_memory) = load(&machine, &code);
        let translated = DbtEngine::new(256 * 1024)
            .unwrap()
            .run(&mut cpu, &mut dbt_memory, budget);
        assert_eq!(
            translated.exit_reason, interpreted.exit_reason,
            "{round}: {text:#?}"
        );
        assert_eq!(
            translated.instructions_executed, interpreted.instructions_executed,
            "{round}: {text:#?}"
        );
        assert_eq!(
            translated.final_state, interpreted.final_state,
            "{round}: {text:#?}"
        );
        assert_eq!(bytes(&dbt_memory), bytes(&memory), "{round}: {text:#?}");
        assert_eq!(
            dbt_memory.time_stamp_counter(),
            memory.time_stamp_counter(),
            "{round}: {text:#?}"
        );
    }
}

#[test]
fn conditional_branches_follow_every_flag_combination() {
    const CODES: [&str; 16] = [
        "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g",
    ];
    for (index, code) in CODES.iter().enumerate() {
        // Short and near forms, backwards and forwards.
        let forms: [(Vec<u8>, u64); 3] = [
            (vec![0x70 + index as u8, 0x10], ENTRY + 2 + 0x10),
            (vec![0x70 + index as u8, 0xF0], ENTRY + 2 - 0x10),
            (
                vec![0x0F, 0x80 + index as u8, 0x00, 0x01, 0, 0],
                ENTRY + 6 + 0x100,
            ),
        ];
        for (bytes, target) in forms {
            for combination in 0..64u32 {
                let flags = Flags {
                    cf: combination & 1 != 0,
                    zf: combination & 2 != 0,
                    sf: combination & 4 != 0,
                    of: combination & 8 != 0,
                    pf: combination & 16 != 0,
                    af: combination & 32 != 0,
                };
                let mut machine = Random(1).machine();
                machine.flags = flags;
                let wanted = if holds(&flags, code) {
                    target
                } else {
                    ENTRY + bytes.len() as u64
                };
                let (cpu, mut memory) = load(&machine, &bytes);
                let interpreted = Interpreter::new(cpu.clone(), &mut memory, 1).run();
                let translated =
                    DbtEngine::new(16 * 1024)
                        .unwrap()
                        .run(&mut cpu.clone(), &mut memory, 1);
                for result in [interpreted, translated] {
                    let mut expected = cpu.clone();
                    expected.rip = wanted;
                    assert_eq!(result.final_state, expected, "j{code} {flags:?}");
                }
            }
        }
    }
}

#[test]
fn unsigned_branches_follow_a_compare_across_a_block_boundary() {
    // cmp rcx, rdx ends one translated run; the branch runs in another.
    for (a, b) in [
        (0u64, 0u64),
        (0, 1),
        (1, 0),
        (u64::MAX, 0),
        (0, u64::MAX),
        (1 << 63, 1),
        (1, 1 << 63),
    ] {
        for (opcode, taken) in [(0x72, a < b), (0x73, a >= b), (0x76, a <= b), (0x77, a > b)] {
            let code = [0x48, 0x39, 0xD1, opcode, 0x10]; // cmp rcx, rdx; jcc +0x10
            let mut machine = Random(7).machine();
            (machine.gpr[1], machine.gpr[2]) = (a, b);
            let (mut cpu, mut memory) = load(&machine, &code);
            let mut engine = DbtEngine::new(16 * 1024).unwrap();
            engine.run(&mut cpu, &mut memory, 1);
            engine.run(&mut cpu, &mut memory, 1);
            assert_eq!(engine.metrics.blocks_compiled, 2);
            let end = ENTRY + code.len() as u64;
            assert_eq!(
                cpu.rip,
                if taken { end + 0x10 } else { end },
                "{a:#x} {b:#x} {opcode:#x}"
            );
        }
    }
}

#[test]
fn calls_and_returns_agree_and_balance_the_stack() {
    // call +1; hlt; ret 16 -> returns to hlt having released 16 extra bytes.
    let code = [0xE8, 0x01, 0, 0, 0, 0xF4, 0xC2, 0x10, 0x00];
    let machine = Random(3).machine();
    let (cpu, mut memory) = load(&machine, &code);
    let interpreted = Interpreter::new(cpu, &mut memory, 8).run();
    let (mut cpu, mut dbt_memory) = load(&machine, &code);
    let translated = DbtEngine::new(16 * 1024)
        .unwrap()
        .run(&mut cpu, &mut dbt_memory, 8);
    for result in [interpreted, translated] {
        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(result.instructions_executed, 3);
        assert_eq!(result.final_state.gpr[4], STACK_TOP + 16);
        assert_eq!(result.final_state.rip, ENTRY + 6);
    }
    assert_eq!(memory.read_u64(STACK_TOP - 8).unwrap(), ENTRY + 5);
    assert_eq!(bytes(&memory), bytes(&dbt_memory));
}

#[test]
fn faulting_read_modify_write_changes_nothing() {
    // The operand is readable but not writable, so the read succeeds and the
    // store faults after the result is known.
    for case in cases() {
        let name = case.text.split(' ').next().unwrap();
        if !case.text.contains(&format!("{name} byte ptr"))
            && !case.text.contains(&format!("{name} word ptr"))
            && !case.text.contains(&format!("{name} dword ptr"))
            && !case.text.contains(&format!("{name} qword ptr"))
            || [
                "cmp", "test", "bt", "mul", "imul", "div", "idiv", "push", "nop", "mulx",
            ]
            .contains(&name)
        {
            continue;
        }
        let machine = Random(11).machine();
        for native in [false, true] {
            let (mut cpu, _) = load(&machine, &case.code);
            let mut memory = GuestMemory::new();
            memory
                .map_region_with_data(
                    ENTRY,
                    case.code.len(),
                    MemoryPermissions::READ_EXECUTE,
                    &case.code,
                )
                .unwrap();
            memory
                .map_region_with_data(DATA, 0x100, MemoryPermissions::READ_ONLY, &[0xA5; 0x100])
                .unwrap();
            memory
                .map_region(STACK, 0x100, MemoryPermissions::READ_WRITE)
                .unwrap();
            let before = cpu.clone();
            let result = if native {
                DbtEngine::new(16 * 1024)
                    .unwrap()
                    .run(&mut cpu, &mut memory, 1)
            } else {
                Interpreter::new(cpu, &mut memory, 1).run()
            };
            assert!(
                matches!(result.exit_reason, ExitReason::MemoryFault(_)),
                "{}: {:?}",
                case.text,
                result.exit_reason
            );
            assert_eq!(result.final_state, before, "{} native={native}", case.text);
            assert_eq!(result.instructions_executed, 0);
        }
    }
}

#[test]
fn pdep_pext_faulting_memory_mask_changes_nothing() {
    // The mask is the ModRM operand. Point RSI at an unmapped address so the
    // read faults before either engine writes the destination register.
    for case in cases() {
        let name = case.text.split(' ').next().unwrap();
        if !matches!(name, "pdep" | "pext") || !case.text.contains(" ptr ") {
            continue;
        }
        let machine = Random(13).machine();
        for native in [false, true] {
            let (mut cpu, mut memory) = load(&machine, &case.code);
            cpu.gpr[6] = 0x4000; // RSI
            cpu.gpr[7] = 0; // RDI
            let before = cpu.clone();
            let before_bytes = bytes(&memory);
            let result = if native {
                DbtEngine::new(16 * 1024)
                    .unwrap()
                    .run(&mut cpu, &mut memory, 1)
            } else {
                Interpreter::new(cpu, &mut memory, 1).run()
            };
            assert!(
                matches!(result.exit_reason, ExitReason::MemoryFault(_)),
                "{}: {:?}",
                case.text,
                result.exit_reason
            );
            assert_eq!(result.final_state, before, "{} native={native}", case.text);
            assert_eq!(result.instructions_executed, 0, "{}", case.text);
            assert_eq!(bytes(&memory), before_bytes, "{}", case.text);
        }
    }
}

#[test]
fn pdep_pext_edge_masks_match_known_answers_in_both_engines() {
    fn index(place: Place) -> usize {
        match place {
            Place::Reg { index, .. } => index,
            _ => panic!("this test uses register forms"),
        }
    }
    fn set(machine: &mut Machine, place: Place, value: u64) {
        match place {
            Place::Reg {
                index,
                bits: 32,
                high: false,
            } => machine.gpr[index] = u64::from(value as u32),
            Place::Reg {
                index,
                bits: 64,
                high: false,
            } => machine.gpr[index] = value,
            _ => panic!("this test uses 32- and 64-bit register forms"),
        }
    }

    // (fixture text, bit source, mask, destination): zero mask, all-ones
    // mask, sparse and adjacent masks, a high-bit-only result, aliased source
    // and mask, and VEX.vvvv naming the highest register. The expected values
    // are written here by hand from the operation sections, not produced by
    // the executor or by the text reference.
    let known: &[(&str, u64, u64, u64)] = &[
        ("pdep ebx, r9d, ecx", 0x0000_000B, 0x0000_0005, 0x0000_0005),
        ("pext ebx, r9d, ecx", 0x0000_000B, 0x0000_0005, 0x0000_0001),
        ("pdep ebx, r9d, ecx", 0x0000_0001, 0x8000_0000, 0x8000_0000),
        ("pext ebx, r9d, ecx", 0x8000_0000, 0x8000_0000, 0x0000_0001),
        ("pdep ebx, r9d, ecx", 0xFFFF_FFFF, 0xAAAA_AAAA, 0xAAAA_AAAA),
        ("pext ebx, r9d, ecx", 0xFFFF_FFFF, 0xAAAA_AAAA, 0x0000_FFFF),
        ("pdep ebx, r9d, ecx", 0xFFFF_FFFF, 0, 0),
        ("pext ebx, r9d, ecx", 0xFFFF_FFFF, 0, 0),
        ("pdep ebx, r9d, ecx", 0x8000_0001, 0xFFFF_FFFF, 0x8000_0001),
        ("pext ebx, r9d, ecx", 0x8000_0001, 0xFFFF_FFFF, 0x8000_0001),
        ("pdep edx, edx, edx", 0x0000_00F0, 0x0000_00F0, 0x0000_0000),
        ("pext edx, edx, edx", 0x0000_00F0, 0x0000_00F0, 0x0000_000F),
        ("pdep ebx, r9d, r15d", 0x0000_0003, 0x0000_0005, 0x0000_0005),
        ("pext ebx, r9d, r15d", 0x0000_000B, 0x0000_0005, 0x0000_0001),
        (
            "pdep rbx, r9, rcx",
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0,
        ),
        (
            "pext rbx, r9, rcx",
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
            1,
        ),
        (
            "pdep rbx, r9, rcx",
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
        ),
        (
            "pext rbx, r9, rcx",
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0000,
            0,
        ),
        (
            "pdep rbx, r9, rcx",
            u64::MAX,
            0xAAAA_AAAA_AAAA_AAAA,
            0xAAAA_AAAA_AAAA_AAAA,
        ),
        (
            "pext rbx, r9, rcx",
            u64::MAX,
            0xAAAA_AAAA_AAAA_AAAA,
            0x0000_0000_FFFF_FFFF,
        ),
        (
            "pext rbx, r9, r15",
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
            1,
        ),
        (
            "pdep rdx, rdx, rdx",
            0x0000_0000_0000_00F0,
            0x0000_0000_0000_00F0,
            0,
        ),
        (
            "pext rdx, rdx, rdx",
            0x0000_0000_0000_00F0,
            0x0000_0000_0000_00F0,
            0x0000_0000_0000_000F,
        ),
    ];
    let all = cases();
    for &(text, source, control, wanted) in known {
        let case = all.iter().find(|case| case.text == text).unwrap();
        let operands: Vec<_> = text.split_once(' ').unwrap().1.split(", ").collect();
        let places: Vec<_> = operands.iter().map(|text| parse_place(text)).collect();
        let mut machine = Random(1).machine();
        machine.gpr = [0; 16];
        set(&mut machine, places[1], source);
        set(&mut machine, places[2], control);

        let mut expected_machine = machine.clone();
        assert!(
            matches!(reference(&mut expected_machine, text), Outcome::Done(_)),
            "{text}"
        );
        assert_eq!(expected_machine.gpr[index(places[0])], wanted, "{text}");
        assert_eq!(expected_machine.flags, machine.flags, "{text}");

        let (mut expected_cpu, expected_memory) = load(&expected_machine, &case.code);
        expected_cpu.rip += case.code.len() as u64;
        let (cpu, mut memory) = load(&machine, &case.code);
        let interpreted = Interpreter::new(cpu, &mut memory, 1).run();
        let (mut dbt_cpu, mut dbt_memory) = load(&machine, &case.code);
        let translated = DbtEngine::new(16 * 1024)
            .unwrap()
            .run(&mut dbt_cpu, &mut dbt_memory, 1);
        for (engine, result, actual_memory) in [
            ("interpreter", interpreted, &memory),
            ("DBT", translated, &dbt_memory),
        ] {
            assert_eq!(
                result.exit_reason,
                ExitReason::StepLimitExceeded,
                "{text} on the {engine}"
            );
            assert_eq!(result.final_state, expected_cpu, "{text} on the {engine}");
            assert_eq!(bytes(actual_memory), bytes(&expected_memory), "{text}");
        }
    }
}

#[test]
fn undeclared_and_ambiguous_encodings_are_rejected() {
    let rejected: &[&[u8]] = &[
        &[0xF0, 0x39, 0x0E],       // lock cmp [rsi], ecx: CMP writes nothing
        &[0xF2, 0x90],             // repne nop
        &[0xF3, 0xC3],             // rep ret
        &[0xF2, 0xA4],             // repne movsb
        &[0xF2, 0xAC],             // repne lodsb
        &[0x67, 0xA4],             // movsb under the address-size override
        &[0x67, 0xA1, 0, 0, 0, 0], // mov eax, [moffs32]
        &[0x64, 0x2E, 0x8B, 0x06], // CS after FS
        &[0x66, 0x50],             // push ax
        &[0x66, 0xE8, 0, 0],       // call rel16
        &[0x66, 0x0F, 0xC9],       // bswap cx (undefined)
        &[0x66, 0x63, 0xC1],       // 16-bit movsxd
        &[0xF3, 0x41, 0x90],       // F3 on xchg r8, rax
        &[0x8D, 0xC0],             // lea with a register operand
        &[0x0F, 0xAE, 0x2E],       // 0F AE /5 with a memory operand
        &[0x0F, 0xAE, 0x36],       // 0F AE /6 with a memory operand
        &[0x0F, 0xC3, 0xC1],       // movnti with a register destination
        &[0x66, 0x0F, 0xC3, 0x06], // 66 with movnti
        &[0x0F, 0xA3, 0x06],       // bt [rsi], eax
        &[0x9B],                   // fwait
        &[0x9B, 0xDB, 0xE3],       // finit: fwait, then fninit
        &[0x66, 0xDB, 0xE3],       // 66 with fninit
        &[0xF3, 0xDB, 0xE3],       // F3 with fninit
        &[0xF2, 0xDB, 0xE3],       // F2 with fninit
        &[0xF0, 0xDB, 0xE3],       // lock fninit
        &[0xC6, 0xC8, 0x01],       // C6 /1
        &[0xFF, 0xF8],             // FF /7
        &[0xFE, 0xD0],             // FE /2
        &[0x8F, 0xC8],             // 8F /1
        &[0x0F, 0x0B],             // ud2
        &[0xCC],                   // int3
        &[0x06],                   // invalid in 64-bit mode
    ];
    for code in rejected {
        let mut padded = code.to_vec();
        padded.extend([0x90; 8]);
        let machine = Random(5).machine();
        for native in [false, true] {
            let (mut cpu, mut memory) = load(&machine, &padded);
            let before = cpu.clone();
            let result = if native {
                DbtEngine::new(16 * 1024)
                    .unwrap()
                    .run(&mut cpu, &mut memory, 1)
            } else {
                Interpreter::new(cpu, &mut memory, 1).run()
            };
            assert!(
                matches!(result.exit_reason, ExitReason::IllegalInstruction(_)),
                "{code:02X?}: {:?}",
                result.exit_reason
            );
            assert_eq!(result.final_state, before, "{code:02X?}");
        }
    }
}

/// FNINIT (DB E3) is the only x87 instruction. Every other ModRM byte under the
/// eight x87 escape opcodes, D8 to DF, stops both engines and changes nothing.
/// That includes FNINIT's neighbours of the same digit, such as FNCLEX (DB E2),
/// and the memory forms of that digit.
#[test]
fn fninit_is_the_only_x87_encoding_accepted() {
    let machine = Random(7).machine();
    assert_ne!(
        machine.x87,
        [0x037F, 0, 0xFFFF],
        "the reset must be visible"
    );
    let mut accepted = Vec::new();
    for opcode in 0xD8..=0xDF_u8 {
        for modrm in 0..=0xFF_u8 {
            // The padding is a SIB byte and displacement for the memory forms.
            let code = [opcode, modrm, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90];
            for native in [false, true] {
                let (mut cpu, mut memory) = load(&machine, &code);
                let before = cpu.clone();
                let data = bytes(&memory);
                let result = if native {
                    DbtEngine::new(16 * 1024)
                        .unwrap()
                        .run(&mut cpu, &mut memory, 1)
                } else {
                    Interpreter::new(cpu, &mut memory, 1).run()
                };
                let what = format!("{opcode:02X} {modrm:02X}, native {native}");
                assert_eq!(bytes(&memory), data, "{what}");
                if matches!(result.exit_reason, ExitReason::IllegalInstruction(_)) {
                    assert_eq!(result.final_state, before, "{what}");
                    continue;
                }
                accepted.push((opcode, modrm, native));
                assert_eq!(result.exit_reason, ExitReason::StepLimitExceeded, "{what}");
                // Only RIP and the three x87 words may differ.
                let mut wanted = before;
                wanted.rip = ENTRY + 2;
                wanted.x87 = X87State {
                    control: 0x037F,
                    status: 0,
                    tag: 0xFFFF,
                };
                assert_eq!(result.final_state, wanted, "{what}");
            }
        }
    }
    assert_eq!(accepted, [(0xDB, 0xE3, false), (0xDB, 0xE3, true)]);
}

/// A local rule, pinned so it cannot change unnoticed: a segment override or a
/// REX prefix before FNINIT is accepted and has no effect, as the decoder
/// treats them before any form without operands. It is not derived from a
/// manual statement. The prefixes that are refused are in
/// `undeclared_and_ambiguous_encodings_are_rejected`.
#[test]
fn mulx_destinations_hold_independent_halves_of_the_product() {
    const MEM32: &str = "mulx r10d, r12d, dword ptr [rsi + 4*rdi + 16]";
    const MEM64: &str = "mulx r10, r12, qword ptr [rsi + 4*rdi + 16]";
    let all = cases();
    // Text, source registers, exact destination answers. RDX is implicit and
    // is set explicitly in each input set. The asymmetric pairs have different
    // high and low halves, so exchanging the destinations fails; the last two
    // make the first two operands identical, where the SDM says the single
    // register keeps the high half.
    type RegInputs = [(usize, u64)];
    let checks: &[(&str, &RegInputs, &RegInputs)] = &[
        (
            "mulx ebx, r9d, ecx",
            &[(1, 3), (2, 0x8000_0001)],
            &[(3, 1), (9, 0x8000_0003)],
        ),
        (
            "mulx rbx, r9, rcx",
            &[(1, 0x8000_0000_0000_0001), (2, 3)],
            &[(3, 1), (9, 0x8000_0000_0000_0003)],
        ),
        (
            "mulx edx, r9d, edx",
            &[(2, 0xFFFF_FFFF)],
            &[(2, 0xFFFF_FFFE), (9, 1)],
        ),
        (MEM32, &[(2, 0x8000_0001)], &[(10, 1), (12, 0x8000_0003)]),
        (
            MEM64,
            &[(2, u64::MAX)],
            &[(10, 0xFFFF_FFFF_FFFF_FFFE), (12, 1)],
        ),
        (
            "mulx ebx, r9d, ecx",
            &[(1, 0x1_0002), (2, 0x3_0004)],
            &[(3, 3), (9, 0xA_0008)],
        ),
        (
            "mulx rbx, r9, rcx",
            &[(1, 0x1_0000_0003), (2, 0x1_0000_0005)],
            &[(3, 1), (9, 0x8_0000_000F)],
        ),
        ("mulx ebx, ebx, ecx", &[(1, 3), (2, 0x8000_0001)], &[(3, 1)]),
        (
            "mulx rbx, rbx, rcx",
            &[(1, 0x8000_0000_0000_0001), (2, 3)],
            &[(3, 1)],
        ),
    ];
    for &(text, inputs, outputs) in checks {
        let case = all
            .iter()
            .find(|case| case.text == text)
            .unwrap_or_else(|| panic!("missing fixture {text}"));
        let mut machine = Random(19).machine();
        machine.gpr[6] = DATA;
        machine.gpr[7] = 4;
        match text {
            MEM32 => machine.operand = 3,
            MEM64 => machine.operand = u64::MAX,
            _ => {}
        }
        for &(index, value) in inputs {
            machine.gpr[index] = value;
        }
        let (cpu, _) = load(&machine, &case.code);
        let mut expected = cpu;
        expected.rip += case.code.len() as u64;
        for &(index, value) in outputs {
            expected.gpr[index] = value;
        }
        let (state, exit, _) = run_both(&machine, &case.code, 1);
        assert_eq!(exit, ExitReason::StepLimitExceeded, "{text}");
        assert_eq!(state, expected, "{text}");
    }
}

#[test]
fn mulx_faulting_memory_source_changes_nothing() {
    for case in cases() {
        let name = case.text.split(' ').next().unwrap();
        if name != "mulx" || !case.text.contains(" ptr ") {
            continue;
        }
        let machine = Random(23).machine();
        for native in [false, true] {
            let (mut cpu, _) = load(&machine, &case.code);
            cpu.gpr[6] = 0x5000; // RSI: unmapped source address
            let mut memory = GuestMemory::new();
            memory
                .map_region_with_data(
                    ENTRY,
                    case.code.len(),
                    MemoryPermissions::READ_EXECUTE,
                    &case.code,
                )
                .unwrap();
            memory
                .map_region_with_data(DATA, 0x100, MemoryPermissions::READ_WRITE, &[0xA5; 0x100])
                .unwrap();
            memory
                .map_region(STACK, 0x100, MemoryPermissions::READ_WRITE)
                .unwrap();
            let before = cpu.clone();
            let before_bytes = bytes(&memory);
            let result = if native {
                DbtEngine::new(16 * 1024)
                    .unwrap()
                    .run(&mut cpu, &mut memory, 1)
            } else {
                Interpreter::new(cpu, &mut memory, 1).run()
            };
            assert!(
                matches!(result.exit_reason, ExitReason::MemoryFault(_)),
                "{}: {:?}",
                case.text,
                result.exit_reason
            );
            assert_eq!(result.final_state, before, "{} native={native}", case.text);
            assert_eq!(result.instructions_executed, 0, "{}", case.text);
            assert_eq!(bytes(&memory), before_bytes, "{}", case.text);
        }
    }
}

#[test]
fn prefixes_accepted_before_fninit_have_no_effect() {
    let machine = Random(11).machine();
    for prefix in [0x2E_u8, 0x64, 0x65, 0x40, 0x41, 0x48, 0x4F] {
        let code = [prefix, 0xDB, 0xE3, 0x90];
        let (cpu, _) = load(&machine, &code);
        let mut wanted = cpu;
        wanted.rip = ENTRY + 3;
        wanted.x87 = X87State {
            control: 0x037F,
            status: 0,
            tag: 0xFFFF,
        };
        let (state, exit, _) = run_both(&machine, &code, 1);
        assert_eq!(exit, ExitReason::StepLimitExceeded, "{prefix:02X}");
        assert_eq!(state, wanted, "{prefix:02X}");
    }
}

#[test]
fn raw_syscalls_are_refused_by_the_dbt_only() {
    let machine = Random(9).machine();
    let (cpu, mut memory) = load(&machine, &[0x0F, 0x05]);
    let interpreted = Interpreter::new(cpu.clone(), &mut memory, 1).run();
    assert_eq!(interpreted.exit_reason, ExitReason::Syscall(machine.gpr[0]));
    let translated = DbtEngine::new(16 * 1024)
        .unwrap()
        .run(&mut cpu.clone(), &mut memory, 1);
    assert_eq!(translated.exit_reason, ExitReason::IllegalInstruction(0x05));
}

/// Run one instruction in both engines, requiring identical outcomes.
fn run_both(machine: &Machine, code: &[u8], budget: usize) -> (CpuState, ExitReason, GuestMemory) {
    let (cpu, mut memory) = load(machine, code);
    let interpreted = Interpreter::new(cpu, &mut memory, budget).run();
    let (mut cpu, mut dbt_memory) = load(machine, code);
    let translated = DbtEngine::new(64 * 1024)
        .unwrap()
        .run(&mut cpu, &mut dbt_memory, budget);
    assert_eq!(
        translated.exit_reason, interpreted.exit_reason,
        "{code:02X?}"
    );
    assert_eq!(
        translated.final_state, interpreted.final_state,
        "{code:02X?}"
    );
    assert_eq!(bytes(&dbt_memory), bytes(&memory), "{code:02X?}");
    assert_eq!(
        dbt_memory.time_stamp_counter(),
        memory.time_stamp_counter(),
        "{code:02X?}"
    );
    (interpreted.final_state, interpreted.exit_reason, memory)
}

/// Code, RAX, RDX, divisor in RCX, and the quotient or `None` for #DE.
type Division = (&'static [u8], u64, u64, u64, Option<u64>);

#[test]
fn division_accepts_exactly_the_representable_quotients() {
    const DIV8: &[u8] = &[0xF6, 0xF1]; // div cl
    const IDIV8: &[u8] = &[0xF6, 0xF9]; // idiv cl
    const DIV32: &[u8] = &[0xF7, 0xF1]; // div ecx
    const IDIV32: &[u8] = &[0xF7, 0xF9]; // idiv ecx
    const DIV64: &[u8] = &[0x48, 0xF7, 0xF1]; // div rcx
    const IDIV64: &[u8] = &[0x48, 0xF7, 0xF9]; // idiv rcx
    const MIN64: u64 = 1 << 63;
    let cases: &[Division] = &[
        (DIV8, 0xFF, 0, 1, Some(0xFF)),
        (DIV8, 0x100, 0, 1, None),
        (IDIV8, 127, 0, 1, Some(127)),
        (IDIV8, 128, 0, 1, None),
        (IDIV8, 0xFF80, 0, 1, Some(0x80)), // -128 / 1
        (IDIV8, 0xFF7F, 0, 1, None),       // -129 / 1
        (IDIV8, 0xFF80, 0, 0xFF, None),    // -128 / -1
        (DIV32, 0xFFFF_FFFF, 0, 1, Some(0xFFFF_FFFF)),
        (DIV32, 0, 1, 1, None),
        (IDIV32, 0x7FFF_FFFF, 0, 1, Some(0x7FFF_FFFF)),
        (IDIV32, 0x8000_0000, 0, 1, None),
        (IDIV32, 0x8000_0000, 0xFFFF_FFFF, 1, Some(0x8000_0000)),
        (DIV64, u64::MAX, 0, 1, Some(u64::MAX)),
        (DIV64, 0, 1, 1, None),
        (IDIV64, MIN64 - 1, 0, 1, Some(MIN64 - 1)),
        (IDIV64, MIN64, 0, 1, None),
        (IDIV64, MIN64, u64::MAX, 1, Some(MIN64)),
        (IDIV64, MIN64, u64::MAX, u64::MAX, None),
        // -2^127 / -1 overflows even the intermediate 128-bit quotient.
        (IDIV64, 0, MIN64, u64::MAX, None),
        (IDIV64, 7, 0, 0, None),
    ];
    for &(code, rax, rdx, rcx, quotient) in cases {
        let mut machine = Random(13).machine();
        (machine.gpr[0], machine.gpr[1], machine.gpr[2]) = (rax, rcx, rdx);
        let (state, exit, _) = run_both(&machine, code, 1);
        match quotient {
            Some(value) => {
                assert_eq!(exit, ExitReason::StepLimitExceeded, "{code:02X?} {rax:#x}");
                let low = if code[0] == 0xF6 {
                    state.gpr[0] & 0xFF
                } else {
                    state.gpr[0]
                };
                assert_eq!(low, value, "{code:02X?} {rax:#x}");
            }
            None => {
                assert_eq!(exit, ExitReason::DivideError, "{code:02X?} {rax:#x}");
                assert_eq!(state.gpr, machine.gpr);
            }
        }
    }
}

/// The bit scans and the bit test keep the destination and the flags where
/// the architecture puts them.
///
/// The expectations come from the Intel SDM Volume 2B, "BSF", "BSR" and "BT":
/// BSF writes the index of the lowest set bit of the source and BSR the index
/// of the highest set bit, and both set the zero flag when the source is
/// zero, in which case the destination is undefined and keeps its value. BT
/// sets the carry flag from the tested bit, leaves the other flags alone and
/// masks the bit index to the operand width.
///
/// The encodings are `bsf rbx, r9` (49 0F BC D9), `bsr rbx, r9` (49 0F BD
/// D9), `bt rbx, r9` (4C 0F A3 CB), `bsf ebx, r9d` (41 0F BC D9),
/// `bsr ebx, r9d` (41 0F BD D9), `bt ebx, r9d` (44 0F A3 CB), `bsf bx, r9w`
/// (66 41 0F BC D9) and `bt bx, r9w` (66 44 0F A3 CB), as produced by LLVM
/// 23.1.6 `llvm-mc -show-encoding`.
#[test]
fn bit_scans_and_bit_tests_follow_the_architecture() {
    const BSF64: &[u8] = &[0x49, 0x0F, 0xBC, 0xD9]; // bsf rbx, r9
    const BSR64: &[u8] = &[0x49, 0x0F, 0xBD, 0xD9]; // bsr rbx, r9
    const BT64: &[u8] = &[0x4C, 0x0F, 0xA3, 0xCB]; // bt rbx, r9
    const BSF32: &[u8] = &[0x41, 0x0F, 0xBC, 0xD9]; // bsf ebx, r9d
    const BSR32: &[u8] = &[0x41, 0x0F, 0xBD, 0xD9]; // bsr ebx, r9d
    const BT32: &[u8] = &[0x44, 0x0F, 0xA3, 0xCB]; // bt ebx, r9d
    const BSF16: &[u8] = &[0x66, 0x41, 0x0F, 0xBC, 0xD9]; // bsf bx, r9w
    const BT16: &[u8] = &[0x66, 0x44, 0x0F, 0xA3, 0xCB]; // bt bx, r9w

    /// A machine with R9 and RBX set and every other register zero.
    fn machine(source: u64, destination: u64) -> Machine {
        let mut gpr = [0; 16];
        gpr[3] = destination;
        gpr[9] = source;
        Machine {
            gpr,
            flags: Flags::default(),
            operand: 0,
            pushed: 0,
            top: STACK_TOP,
            x87: [0x037F, 0, 0xFFFF],
            tsc: 0,
        }
    }

    // The lowest set bit of the source is the index BSF writes.
    let (state, _, _) = run_both(&machine(0x0000_0000_0000_0028, 0xFFFF), BSF64, 1);
    assert_eq!(state.gpr[3], 3);
    assert!(!state.flags.zf);

    // The highest set bit of the source is the index BSR writes.
    let (state, _, _) = run_both(&machine(0x8000_0000_0000_0028, 0xFFFF), BSR64, 1);
    assert_eq!(state.gpr[3], 63);
    assert!(!state.flags.zf);

    // A zero source sets the zero flag and leaves the destination alone.
    let (state, _, _) = run_both(&machine(0, 0xFFFF_FFFF_FFFF_FFFF), BSF64, 1);
    assert_eq!(state.gpr[3], 0xFFFF_FFFF_FFFF_FFFF);
    assert!(state.flags.zf);
    let (state, _, _) = run_both(&machine(0, 0xFFFF_FFFF_FFFF_FFFF), BSR64, 1);
    assert_eq!(state.gpr[3], 0xFFFF_FFFF_FFFF_FFFF);
    assert!(state.flags.zf);

    // A 32-bit destination holds a 32-bit index and clears the half above it.
    let (state, _, _) = run_both(&machine(0x0000_0028, 0xFFFF_FFFF_FFFF_FFFF), BSF32, 1);
    assert_eq!(state.gpr[3], 3);
    let (state, _, _) = run_both(&machine(0x8000_0000, 0xFFFF_FFFF_FFFF_FFFF), BSR32, 1);
    assert_eq!(state.gpr[3], 31);

    // A 16-bit destination keeps the bits above it.
    let (state, _, _) = run_both(&machine(0x0100, 0xFFFF_FFFF_FFFF_FFFF), BSF16, 1);
    assert_eq!(state.gpr[3], 0xFFFF_FFFF_FFFF_0008);

    // BT reports the tested bit in the carry flag and leaves the bit string
    // alone.
    let (state, _, _) = run_both(&machine(1, 0b1010), BT64, 1);
    assert!(state.flags.cf);
    assert_eq!(state.gpr[3], 0b1010);
    let (state, _, _) = run_both(&machine(0, 0b1010), BT64, 1);
    assert!(!state.flags.cf);
    assert_eq!(state.gpr[3], 0b1010);

    // The bit index is masked to the operand width, so 64 selects bit 0 of a
    // 64-bit bit string and 32 selects bit 0 of a 32-bit one.
    let (state, _, _) = run_both(&machine(64, 0b1), BT64, 1);
    assert!(state.flags.cf);
    let (state, _, _) = run_both(&machine(32, 0b1), BT32, 1);
    assert!(state.flags.cf);
    let (state, _, _) = run_both(&machine(16, 0b1), BT16, 1);
    assert!(state.flags.cf);

    // The bit scans and the register form of the bit test are lowered to
    // ARM64 instead of falling back to the shared semantics.
    for code in [BSF64, BSR64, BT64, BSF32, BSR32, BT32, BSF16, BT16] {
        let insn = decode(ENTRY, |address| {
            code.get((address - ENTRY) as usize).copied().ok_or(())
        })
        .expect("decode");
        assert!(is_native(&insn), "{code:02X?} is not native");
        let mut engine = DbtEngine::new(64 * 1024).unwrap();
        let (mut cpu, mut memory) = load(&machine(0x28, 0), code);
        engine.run(&mut cpu, &mut memory, 1);
        assert_eq!(engine.metrics.interpreter_fallbacks, 0, "{code:02X?}");
    }
}

#[test]
fn pop_rsp_and_push_rsp_use_the_architectural_stack_pointer() {
    let machine = Random(17).machine();
    // 5C = pop rsp: the popped value replaces the incremented pointer.
    let (state, _, _) = run_both(&machine, &[0x5C], 1);
    assert_eq!(state.gpr[4], machine.top);
    // 54 = push rsp stores the value before the decrement.
    let (state, _, memory) = run_both(&machine, &[0x54], 1);
    assert_eq!(state.gpr[4], STACK_TOP - 8);
    assert_eq!(memory.read_u64(STACK_TOP - 8).unwrap(), STACK_TOP);
    // 8F 04 24 = pop qword ptr [rsp] addresses with the incremented pointer.
    let (state, _, memory) = run_both(&machine, &[0x8F, 0x04, 0x24], 1);
    assert_eq!(state.gpr[4], STACK_TOP + 8);
    assert_eq!(memory.read_u64(STACK_TOP + 8).unwrap(), machine.top);
}

#[test]
fn flags_survive_when_the_next_flag_writer_is_never_reached() {
    // cmp rcx, rdx sets flags that the later cmp would overwrite, but the
    // instruction in between faults first, so they remain visible.
    let faulting: &[&[u8]] = &[
        &[0x48, 0x8B, 0x03], // mov rax, [rbx]
        &[0x48, 0x03, 0x03], // add rax, [rbx]
        &[0x53],             // push rbx (RSP unmapped)
        &[0x5B],             // pop rbx (RSP unmapped)
        &[0x48, 0xF7, 0xF6], // div rsi with RSI = 0
    ];
    for middle in faulting {
        let code = [&[0x48, 0x39, 0xD1][..], middle, &[0x48, 0x39, 0xC9, 0xF4]].concat();
        let mut machine = Random(19).machine();
        machine.gpr[3] = 0x9000;
        machine.gpr[4] = 0x9000;
        machine.gpr[6] = 0;
        (machine.gpr[1], machine.gpr[2]) = (1, 2);
        machine.flags = Flags::default();
        let (state, exit, _) = run_both(&machine, &code, 16);
        assert!(
            matches!(exit, ExitReason::MemoryFault(_) | ExitReason::DivideError),
            "{middle:02X?}: {exit:?}"
        );
        assert_eq!(state.rip, ENTRY + 3, "{middle:02X?}");
        // 1 - 2 borrows and is negative; it neither overflows nor is zero.
        assert!(state.flags.cf && state.flags.sf, "{middle:02X?}");
        assert!(!state.flags.zf && !state.flags.of, "{middle:02X?}");
    }
}

#[test]
fn crc32_reads_exactly_its_source_and_changes_only_its_destination() {
    let end = DATA + 0x100;
    let forms: [(&[u8], u64); 7] = [
        (&[0xF2, 0x0F, 0x38, 0xF0, 0x03], 1), // crc32 eax, byte ptr [rbx]
        (&[0x66, 0xF2, 0x0F, 0x38, 0xF0, 0x03], 1), // 66 has no effect on a byte source
        (&[0xF2, 0x48, 0x0F, 0x38, 0xF0, 0x03], 1), // crc32 rax, byte ptr [rbx]
        (&[0x66, 0xF2, 0x0F, 0x38, 0xF1, 0x03], 2), // crc32 eax, word ptr [rbx]
        (&[0xF2, 0x0F, 0x38, 0xF1, 0x03], 4), // crc32 eax, dword ptr [rbx]
        (&[0xF2, 0x48, 0x0F, 0x38, 0xF1, 0x03], 8), // crc32 rax, qword ptr [rbx]
        (&[0xF2, 0x66, 0x48, 0x0F, 0x38, 0xF1, 0x03], 8), // REX.W over 66
    ];
    for (code, width) in forms {
        let mut machine = Random(0xC3C).machine();
        machine.gpr[0] = 0xAAAA_AAAA_0000_0000;
        machine.flags = Flags {
            cf: true,
            of: true,
            af: true,
            ..Flags::default()
        };
        // The bytes 00 to 07 at the end of the region.
        let image = [0u8, 1, 2, 3, 4, 5, 6, 7];
        for native in [false, true] {
            let run = |rbx: u64, gs: u64, code: &[u8]| {
                let (mut cpu, mut memory) = load(&machine, code);
                memory.write_bytes(end - 8, &image).unwrap();
                cpu.gpr[3] = rbx;
                cpu.gs_base = gs;
                let before = cpu.clone();
                let result = if native {
                    DbtEngine::new(16 * 1024)
                        .unwrap()
                        .run(&mut cpu, &mut memory, 1)
                } else {
                    Interpreter::new(cpu, &mut memory, 1).run()
                };
                (before, result)
            };
            // An independent bitwise CRC of the bytes that should be read.
            let mut crc = 0u32;
            for byte in &image[8 - width as usize..] {
                for index in 0..8 {
                    let top = (crc ^ u32::from(byte >> index)) & 1;
                    crc = (crc / 2) ^ (top * 0x82F6_3B78);
                }
            }
            let (mut wanted, result) = run(end - width, 0, code);
            wanted.gpr[0] = u64::from(crc);
            wanted.rip += code.len() as u64;
            assert_eq!(result.exit_reason, ExitReason::StepLimitExceeded);
            assert_eq!(result.final_state, wanted, "{code:02X?}");

            let prefixed = [&[0x65][..], code].concat();
            let (_, result) = run(end - width - 0x777, 0x777, &prefixed);
            assert_eq!(result.final_state.gpr[0], u64::from(crc), "{code:02X?}");

            let (before, result) = run(end - width + 1, 0, code);
            let fault = MemoryError::UnmappedAddress(end);
            assert_eq!(result.exit_reason, ExitReason::MemoryFault(fault));
            assert_eq!(result.final_state, before, "{code:02X?}");

            const LIMIT: u64 = 0x0000_8000_0000_0000;
            for address in [LIMIT, LIMIT - width + 1, 1 << 63] {
                let (before, result) = run(address, 0, code);
                if address < LIMIT && width == 1 {
                    continue;
                }
                assert_eq!(result.exit_reason, ExitReason::GeneralProtection);
                assert_eq!(result.final_state, before, "{code:02X?}");
            }
        }
    }
}

#[test]
fn crc32_neighbours_are_rejected() {
    let rejected: &[&[u8]] = &[
        &[0x0F, 0x38, 0xF0, 0x03],             // movbe eax, [rbx]
        &[0x66, 0x0F, 0x38, 0xF1, 0x03],       // movbe [rbx], ax
        &[0xF3, 0x0F, 0x38, 0xF1, 0xC1],       // F3 selects no such form
        &[0xF2, 0xF3, 0x0F, 0x38, 0xF1, 0xC1], // both repeat prefixes
        &[0xF0, 0xF2, 0x0F, 0x38, 0xF1, 0x03], // LOCK
        &[0xF2, 0x0F, 0x38, 0xF2, 0xC1],       // unassigned
        &[0xF2, 0x0F, 0x3A, 0xF1, 0xC1, 0x00], // the other escape
    ];
    for code in rejected {
        let machine = Random(0xC4C).machine();
        let padded = [code, &[0x90; 8][..]].concat();
        let (state, exit, _) = run_both(&machine, &padded, 1);
        assert!(
            matches!(exit, ExitReason::IllegalInstruction(_)),
            "{code:02X?}: {exit:?}"
        );
        assert_eq!(state.rip, ENTRY, "{code:02X?}");
    }
}

#[test]
fn segment_bases_apply_to_every_addressing_form_in_both_engines() {
    const FS: u64 = 0x1_0000;
    const GS: u64 = 0x2_0000;
    // The linear address is the segment base plus the effective address, which
    // is RIP-relative to the next instruction for ModRM mode 0, rm 101b.
    let rip_forward = ENTRY + 8 + 0x1000;
    let rip_back = ENTRY + 8 - 0x400;
    let cases: &[(&[u8], u64)] = &[
        (
            &[0x64, 0x48, 0x8B, 0x05, 0x00, 0x10, 0, 0],
            FS + rip_forward,
        ), // mov rax, fs:[rip+0x1000]
        (
            &[0x65, 0x48, 0x8B, 0x05, 0x00, 0x10, 0, 0],
            GS + rip_forward,
        ),
        (
            &[0x64, 0x48, 0x8B, 0x05, 0x00, 0xFC, 0xFF, 0xFF],
            FS + rip_back,
        ),
        (
            &[0x65, 0x48, 0x8B, 0x05, 0x00, 0xFC, 0xFF, 0xFF],
            GS + rip_back,
        ),
        (
            &[0x64, 0x48, 0x89, 0x05, 0x00, 0x10, 0, 0],
            FS + rip_forward,
        ), // mov fs:[rip+0x1000], rax
        (
            &[0x65, 0x48, 0x89, 0x05, 0x00, 0xFC, 0xFF, 0xFF],
            GS + rip_back,
        ),
        (&[0x65, 0x48, 0x8B, 0x43, 0xF8], GS + 0x3000 - 8), // mov rax, gs:[rbx-8]
        (&[0x64, 0x48, 0x8B, 0x04, 0x8B], FS + 0x3000 + 16), // mov rax, fs:[rbx+rcx*4]
        (&[0x64, 0x65, 0x48, 0x8B, 0x03], GS + 0x3000),     // FS then GS
        (&[0x65, 0x64, 0x48, 0x8B, 0x03], GS + 0x3000),     // GS then FS
        (&[0x64, 0x65, 0x64, 0x48, 0x8B, 0x03], GS + 0x3000),
        (&[0x64, 0x48, 0xA1, 0x40, 0, 0, 0, 0, 0, 0, 0], FS + 0x40), // movabs rax, fs:[0x40]
        (
            &[
                0x65, 0x48, 0xA3, 0xF0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            ],
            GS - 0x10,
        ), // movabs gs:[-0x10], rax
    ];
    for &(code, address) in cases {
        let opcode = code[code.iter().position(|byte| *byte == 0x48).unwrap() + 1];
        let store = matches!(opcode, 0x89 | 0xA3);
        for native in [false, true] {
            for mapped in [true, false] {
                let mut memory = GuestMemory::new();
                memory
                    .map_region_with_data(ENTRY, code.len(), MemoryPermissions::READ_EXECUTE, code)
                    .unwrap();
                if mapped {
                    memory
                        .map_region(address, 8, MemoryPermissions::READ_WRITE)
                        .unwrap();
                    memory.write_u64(address, 0x1122_3344_5566_7788).unwrap();
                }
                let mut cpu = CpuState::new(ENTRY, 0);
                (cpu.fs_base, cpu.gs_base) = (FS, GS);
                cpu.gpr[0] = 0xAABB_CCDD_EEFF_0011;
                cpu.gpr[1] = 4;
                cpu.gpr[3] = 0x3000;
                let before = cpu.clone();
                let result = if native {
                    DbtEngine::new(16 * 1024)
                        .unwrap()
                        .run(&mut cpu, &mut memory, 1)
                } else {
                    Interpreter::new(cpu, &mut memory, 1).run()
                };
                if !mapped {
                    assert_eq!(
                        result.exit_reason,
                        ExitReason::MemoryFault(MemoryError::UnmappedAddress(address)),
                        "{code:02X?}"
                    );
                    assert_eq!(result.final_state, before, "{code:02X?}");
                } else if store {
                    assert_eq!(
                        memory.read_u64(address).unwrap(),
                        before.gpr[0],
                        "{code:02X?}"
                    );
                } else {
                    assert_eq!(
                        result.final_state.gpr[0], 0x1122_3344_5566_7788,
                        "{code:02X?}"
                    );
                }
            }
        }
    }
    // LEA computes the address without a segment base.
    let (state, _, _) = {
        let mut machine = Random(23).machine();
        machine.gpr[3] = 0x3000;
        run_both(&machine, &[0x65, 0x48, 0x8D, 0x43, 0x10], 1)
    };
    assert_eq!(state.gpr[0], 0x3010);
}
