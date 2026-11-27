// SPDX-License-Identifier: GPL-3.0-or-later

//! Verified compilation for Xenora's synthetic compute and graphics ABIs.

use std::collections::HashMap;
use std::fmt::{self, Write};

mod graphics;
pub use graphics::{PIXEL_ENTRY_POINT, VERTEX_ENTRY_POINT, compile_graphics};

pub use xenora_decoder::Architecture;
use xenora_decoder::{CompareRelation, Signedness, SoppControlOp};
use xenora_decoder::{
    DecodeError, DecodedInstruction, Instruction, ScalarSource, ScalarUnaryOp, VectorAluOp,
    VectorAluVop3Op, VectorBinary, VectorCompareOp, VectorLowering, VectorSource, VectorUnary,
    VectorVcc,
};

/// SGPR index of `vcc_lo`.
const VCC_LO: usize = 106;
/// SGPR index of `vcc_hi`.
const VCC_HI: usize = 107;
/// SGPR index of `exec_lo`.
const EXEC_LO: usize = 126;

/// Entry point of all kernels produced by this compiler.
pub const ENTRY_POINT: &str = "xenora_compute";
const MAX_CODE_BYTES: usize = 1024;

#[derive(Debug)]
pub struct CompiledShader {
    pub disassembly: String,
    pub ir: String,
    pub msl: String,
}

/// Optional translation passes applied after verification.
///
/// Default options preserve the historical IR shape: constants from scalar
/// evaluation and immediates remain, pure vector expressions between
/// constants are not rewritten, and duplicate pure expressions are not reused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileOptions {
    /// Replace pure vector unary/binary ops whose operands are already
    /// `const.u32` with a single constant. Floating-point ops fold only when
    /// the existing exact positive-normal rules accept the result. Loads,
    /// stores, lane masks, merges and branches are never folded.
    pub fold_constants: bool,
    /// Reuse the first result of an equivalent pure vector unary or binary
    /// expression. Equivalence is the same operation and the same operand
    /// values in the same order; constant operands compare by bit pattern.
    /// Loads, stores, constants, lane masks, merges, branches and other
    /// mask-dependent or side-effecting ops are never reused.
    pub eliminate_common_subexpressions: bool,
}

#[derive(Debug)]
pub enum CompileError {
    InvalidLength {
        bytes: usize,
    },
    Decode(DecodeError),
    GraphicsStage {
        stage: &'static str,
        error: Box<CompileError>,
    },
    GraphicsLink {
        reason: &'static str,
    },
    Verify {
        offset: usize,
        reason: &'static str,
    },
    /// An image sample must read descriptors that were loaded into s[8:19] by
    /// the synthetic binding ABI's `s_buffer_load_dwordx4` sequence.
    DescriptorProvenance {
        offset: usize,
        register: u8,
        expected: &'static str,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { bytes } => write!(
                formatter,
                "fixture length must be a nonzero multiple of four up to {MAX_CODE_BYTES} bytes, got {bytes}"
            ),
            Self::Decode(error) => error.fmt(formatter),
            Self::GraphicsStage { stage, error } => write!(formatter, "{stage} shader: {error}"),
            Self::GraphicsLink { reason } => write!(formatter, "graphics linking: {reason}"),
            Self::Verify { offset, reason } => {
                write!(
                    formatter,
                    "shader verification at byte 0x{offset:x}: {reason}"
                )
            }
            Self::DescriptorProvenance {
                offset,
                register,
                expected,
            } => write!(
                formatter,
                "shader verification at byte 0x{offset:x}: s{register} was not written by {expected}"
            ),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::GraphicsStage { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Value(usize);

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "t{}", self.0)
    }
}

#[derive(Clone, Copy)]
enum Op {
    Load,
    /// One word of a scalar buffer load: `input[word]`, or the register's old
    /// value where the dispatch has no such word. Wave-uniform.
    ScalarLoad {
        word: u32,
        old: Value,
    },
    Constant(u32),
    Binary(Binary, Value, Value),
    Unary(VectorUnary, Value),
    MergeLanes {
        mask: u32,
        active: Value,
        inactive: Value,
    },
    Store(Value),
    VccArithmetic {
        primitive: VectorVcc,
        left: Value,
        right: Value,
        mask: Value,
        flag: bool,
    },
    CarryMask {
        flag: Value,
        exec: u32,
    },
    ScalarBranch {
        target: usize,
        taken: bool,
    },
    /// Wave32 VOPC result: bit n is EXEC[n] & (lane n exists) & test(lane n).
    /// The value is wave-uniform; every thread computes it for its own wave.
    LaneMask {
        op: VectorCompareOp,
        /// Assembly name of the decoded encoding (`_e32` or `_e64`).
        name: &'static str,
        /// The explicit SGPR destination of the VOP3A encoding (IR text only).
        sdst: Option<usize>,
        src0: Value,
        src1: Value,
        high: Option<(Value, Value)>,
        exec: u32,
    },
    /// A wave-uniform SOPC or SOPK compare with a run-time operand: the SCC bit.
    ScalarTest {
        relation: CompareRelation,
        signed: bool,
        left: Value,
        right: Value,
    },
    /// A branch taken when `mask` is zero: `s_cbranch_vccz` on a lane-derived
    /// VCC_LO or `s_cbranch_scc0` on a run-time SCC (IR annotation only).
    RuntimeBranch {
        test: &'static str,
        mask: Value,
        target: usize,
    },
    /// The `s_branch` between the two parts of a run-time `if`/`else` (IR
    /// annotation only).
    RuntimeElse {
        target: usize,
    },
    /// Join after a runtime branch: the fall-through value when the mask is
    /// nonzero (branch not taken), else the value on the taken path.
    JoinRuntime {
        test: &'static str,
        mask: Value,
        fallthrough: Value,
        taken: Value,
    },
    /// The value is the earlier pure result `source`. The definition index stays
    /// so later operands do not need renumbering.
    Reuse(Value),
    /// MIMG 2D texture sample: the selected channel of
    /// `tex0.sample(samp0, float2(u, v), level(lod))` as float bits. The texture
    /// and sampler are the synthetic ABI bindings, not register operands.
    /// `exec` is the compile-time EXEC mask: only its lanes check the sample
    /// domain (finite coordinates and, for `explicit_lod`, a whole mip level).
    Sample {
        u: Value,
        v: Value,
        lod: Value,
        component: u8,
        explicit_lod: bool,
        exec: u32,
    },
}

impl Op {
    fn operands(self) -> Vec<Value> {
        match self {
            Self::Load | Self::Constant(_) | Self::ScalarBranch { .. } => Vec::new(),
            Self::ScalarLoad { old, .. } => vec![old],
            Self::Binary(_, left, right) => vec![left, right],
            Self::Unary(_, source) => vec![source],
            Self::MergeLanes {
                active, inactive, ..
            } => vec![active, inactive],
            Self::Store(value) => vec![value],
            Self::LaneMask {
                src0, src1, high, ..
            } => {
                let mut operands = vec![src0, src1];
                if let Some((left, right)) = high {
                    operands.extend([left, right]);
                }
                operands
            }
            Self::RuntimeBranch { mask, .. } => vec![mask],
            Self::RuntimeElse { .. } => Vec::new(),
            Self::ScalarTest { left, right, .. } => vec![left, right],
            Self::VccArithmetic {
                left, right, mask, ..
            } => vec![left, right, mask],
            Self::CarryMask { flag, .. } => vec![flag],
            Self::JoinRuntime {
                mask,
                fallthrough,
                taken,
                ..
            } => vec![mask, fallthrough, taken],
            Self::Reuse(source) => vec![source],
            Self::Sample { u, v, lod, .. } => vec![u, v, lod],
        }
    }

    /// The MSL expression of a per-thread value, with operand names, the
    /// dispatch index and the lane given by the caller.
    fn expression(self, name: &dyn Fn(Value) -> String, index: &str, lane: &str) -> String {
        match self {
            Self::Load => format!("input[{index}]"),
            Self::ScalarLoad { word, old } => {
                format!("{word}u < grid ? input[{word}u] : {}", name(old))
            }
            Self::Constant(value) => format!("0x{value:08x}u"),
            Self::Unary(op, source) => {
                let src = name(source);
                // AMD guide 12.8: all three scans return -1 if no bit matches.
                match op {
                    VectorUnary::Not => format!("~{src}"),
                    VectorUnary::ReverseBits => format!("reverse_bits({src})"),
                    VectorUnary::FindHighUnsigned => {
                        format!("{src} == 0u ? 0xffffffffu : clz({src})")
                    }
                    VectorUnary::FindLow => format!("{src} == 0u ? 0xffffffffu : ctz({src})"),
                    VectorUnary::FindHighSigned => format!(
                        "({src} == 0u || {src} == 0xffffffffu) ? 0xffffffffu : clz({src} ^ (0u - ({src} >> 31u)))"
                    ),
                    VectorUnary::TruncateF32
                    | VectorUnary::CeilingF32
                    | VectorUnary::RoundNearestEvenF32
                    | VectorUnary::FloorF32 => float_rounding_expression(op, &src),
                }
            }
            Self::Binary(operator, left, right) => {
                let (left, right) = (name(left), name(right));
                let Binary::Vector(operation) = operator;
                vector_expression(operation, &left, &right)
            }
            Self::MergeLanes {
                mask,
                active,
                inactive,
            } => format!(
                "((0x{mask:08x}u >> {lane}) & 1u) != 0u ? {} : {}",
                name(active),
                name(inactive)
            ),
            Self::JoinRuntime {
                mask,
                fallthrough,
                taken,
                ..
            } => format!(
                "{} != 0u ? {} : {}",
                name(mask),
                name(fallthrough),
                name(taken)
            ),
            Self::VccArithmetic {
                primitive,
                left,
                right,
                mask,
                flag,
            } => {
                let (left, right) = (name(left), name(right));
                let bit = format!("(({} >> {lane}) & 1u)", name(mask));
                // Guide 12.7: widen the carry/borrow check before adding the input bit.
                match (primitive, flag) {
                    (VectorVcc::Select, false) => format!("{bit} != 0u ? {right} : {left}"),
                    (VectorVcc::AddCarry, false) => format!("{left} + {right} + {bit}"),
                    (VectorVcc::AddCarry, true) => {
                        format!("uint((ulong({left}) + ulong({right}) + {bit}) >> 32u)")
                    }
                    (VectorVcc::SubtractBorrow, false) => format!("{left} - {right} - {bit}"),
                    (VectorVcc::SubtractBorrow, true) => {
                        format!("uint(ulong({right}) + {bit} > ulong({left}))")
                    }
                    _ => unreachable!("not a VCC arithmetic operation"),
                }
            }
            Self::Reuse(source) => name(source),
            Self::Sample {
                u,
                v,
                lod,
                component,
                ..
            } => {
                let channel = ["r", "g", "b", "a"][usize::from(component)];
                format!(
                    "as_type<uint>(tex0.sample(samp0, float2(as_type<float>({}), as_type<float>({})), level(as_type<float>({}))).{channel})",
                    name(u),
                    name(v),
                    name(lod),
                )
            }
            Self::ScalarTest {
                relation,
                signed,
                left,
                right,
            } => {
                let (left, right) = (name(left), name(right));
                let test = relation_expression(relation, signed, &left, &right);
                format!("({test}) ? 1u : 0u")
            }
            Self::Store(_)
            | Self::ScalarBranch { .. }
            | Self::CarryMask { .. }
            | Self::LaneMask { .. }
            | Self::RuntimeElse { .. }
            | Self::RuntimeBranch { .. } => unreachable!("not a per-thread expression"),
        }
    }
}

/// The MSL operator of a relation; `None` for F and T, which read no operand.
fn relation_symbol(relation: CompareRelation) -> Option<&'static str> {
    Some(match relation {
        CompareRelation::Less => "<",
        CompareRelation::Equal => "==",
        CompareRelation::LessOrEqual => "<=",
        CompareRelation::Greater => ">",
        CompareRelation::NotEqual => "!=",
        CompareRelation::GreaterOrEqual => ">=",
        CompareRelation::False | CompareRelation::True => return None,
    })
}

/// A 32-bit integer relation.
fn relation_expression(relation: CompareRelation, signed: bool, left: &str, right: &str) -> String {
    match relation_symbol(relation) {
        None => (relation == CompareRelation::True).to_string(),
        Some(symbol) if signed => {
            format!("as_type<int>({left}) {symbol} as_type<int>({right})")
        }
        Some(symbol) => format!("{left} {symbol} {right}"),
    }
}

fn compare_expression(op: VectorCompareOp, left: &str, right: &str) -> String {
    let Some(symbol) = relation_symbol(op.relation()) else {
        return (op.relation() == CompareRelation::True).to_string();
    };
    if op.semantics().width == 16 {
        // Guide §§6.2, 6.2.5 and 6.8: the e32 form reads the low half.
        let half = |value| {
            if op.signed() {
                format!("as_type<short>(ushort({value}))")
            } else {
                format!("ushort({value})")
            }
        };
        format!("{} {symbol} {}", half(left), half(right))
    } else if op.signed() && op.semantics().width == 64 {
        format!("as_type<long>({left}) {symbol} as_type<long>({right})")
    } else if op.semantics().width == 64 {
        format!("{left} {symbol} {right}")
    } else {
        relation_expression(op.relation(), op.signed(), left, right)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Binary {
    /// A registry VOP2 integer or binary32 operation (or a VOP3 integer
    /// encoding), lowered by its typed template.
    Vector(VectorBinary),
}

impl Binary {
    fn operator(self) -> (&'static str, &'static str) {
        match self {
            // The graphics path prints binary32 values with these symbols.
            Self::Vector(VectorBinary::AddF32) => ("add.f32", "+"),
            Self::Vector(VectorBinary::SubF32) => ("sub.f32", "-"),
            Self::Vector(VectorBinary::MulF32) => ("mul.f32", "*"),
            Self::Vector(operation) => (
                match operation {
                    VectorBinary::Add => "add.u32",
                    VectorBinary::Sub => "sub.u32",
                    VectorBinary::Xnor => "xnor.b32",
                    VectorBinary::MinSigned => "min.i32",
                    VectorBinary::MinUnsigned => "min.u32",
                    VectorBinary::MaxSigned => "max.i32",
                    VectorBinary::MaxUnsigned => "max.u32",
                    VectorBinary::Mul24Signed => "mul24.i32",
                    VectorBinary::Mul24Unsigned => "mul24.u32",
                    VectorBinary::MulHi24Signed => "mulhi24.i32",
                    VectorBinary::MulHi24Unsigned => "mulhi24.u32",
                    VectorBinary::And => "and.b32",
                    VectorBinary::Or => "or.b32",
                    VectorBinary::Xor => "xor.b32",
                    VectorBinary::ShiftLeft => "lshl.b32",
                    VectorBinary::ShiftRight => "lshr.b32",
                    VectorBinary::ArithmeticShiftRight => "ashr.i32",
                    VectorBinary::MinF32 => "min.f32",
                    VectorBinary::MaxF32 => "max.f32",
                    VectorBinary::AddF32 | VectorBinary::SubF32 | VectorBinary::MulF32 => {
                        unreachable!("binary32 arithmetic has a symbol")
                    }
                },
                "",
            ),
        }
    }
}

/// The MSL template of each per-lane registry operation. Integer arithmetic is
/// on `uint` (wrapping); signed forms reinterpret with `as_type`, and the
/// 24-bit forms sign-extend bit 23 with a shift pair or mask to 24 bits first.
fn vector_expression(operation: VectorBinary, left: &str, right: &str) -> String {
    let int = |value: &str| format!("as_type<int>({value})");
    let i24 = |value: &str| format!("(as_type<int>({value} << 8u) >> 8)");
    let u24 = |value: &str| format!("({value} & 0x00ffffffu)");
    match operation {
        VectorBinary::Add => format!("{left} + {right}"),
        VectorBinary::Sub => format!("{left} - {right}"),
        VectorBinary::Xnor => format!("~({left} ^ {right})"),
        VectorBinary::MinSigned => format!("as_type<uint>(min({}, {}))", int(left), int(right)),
        VectorBinary::MinUnsigned => format!("min({left}, {right})"),
        VectorBinary::MaxSigned => format!("as_type<uint>(max({}, {}))", int(left), int(right)),
        VectorBinary::MaxUnsigned => format!("max({left}, {right})"),
        // The low 32 bits of a product do not depend on signedness, so the
        // sign-extended factors multiply as wrapping uint.
        VectorBinary::Mul24Signed => format!(
            "as_type<uint>({}) * as_type<uint>({})",
            i24(left),
            i24(right)
        ),
        VectorBinary::Mul24Unsigned => format!("{} * {}", u24(left), u24(right)),
        VectorBinary::MulHi24Signed => {
            format!("as_type<uint>(mulhi({}, {}))", i24(left), i24(right))
        }
        VectorBinary::MulHi24Unsigned => format!("mulhi({}, {})", u24(left), u24(right)),
        VectorBinary::And => format!("{left} & {right}"),
        VectorBinary::Or => format!("{left} | {right}"),
        VectorBinary::Xor => format!("{left} ^ {right}"),
        // MSL leaves shifts by 32 or more undefined; the guide reads S0[4:0].
        VectorBinary::ShiftLeft => format!("{left} << ({right} & 31u)"),
        VectorBinary::ShiftRight => format!("{left} >> ({right} & 31u)"),
        VectorBinary::ArithmeticShiftRight => {
            format!("as_type<uint>(as_type<int>({left}) >> ({right} & 31u))")
        }
        // Guide 12.7 page 131: IEEE binary32 arithmetic, rounded to nearest
        // even. The kernel turns off contraction and reassociation, and the
        // library is built in safe math mode. Metal flushes subnormal inputs
        // and results, which the interpreters refuse (vector-binary-f32.md).
        VectorBinary::AddF32 => float_arithmetic(left, "+", right),
        VectorBinary::SubF32 => float_arithmetic(left, "-", right),
        VectorBinary::MulF32 => float_arithmetic(left, "*", right),
        VectorBinary::MinF32 => float_select_expression(true, left, right),
        VectorBinary::MaxF32 => float_select_expression(false, left, right),
    }
}

fn float_arithmetic(left: &str, symbol: &str, right: &str) -> String {
    format!("as_type<uint>(as_type<float>({left}) {symbol} as_type<float>({right}))")
}

/// Guide 12.7 V_MIN_F32 (page 132) and V_MAX_F32 (page 133) on the register
/// bits, with no Metal float operation (`fmin` and `fmax` differ on signed
/// zeros): a NaN operand gives the other operand (S1 when both are NaN; the
/// interpreters refuse signalling NaNs), a +0/-0 pair gives -0 for MIN and +0
/// for MAX, and otherwise the sign-magnitude words are compared as ordered
/// keys (S0 < S1 ? S0 : S1, or S0 >= S1 ? S0 : S1).
fn float_select_expression(minimum: bool, left: &str, right: &str) -> String {
    let nan = |value: &str| format!("({value} & 0x7fffffffu) > 0x7f800000u");
    let key = |value: &str| {
        format!("({value} ^ (as_type<uint>(as_type<int>({value}) >> 31) | 0x80000000u))")
    };
    let (zeros, order) = if minimum {
        (format!("{left} | {right}"), "<")
    } else {
        (format!("{left} & {right}"), ">=")
    };
    format!(
        "{} ? {right} : {} ? {left} : (({left} | {right}) & 0x7fffffffu) == 0u ? ({zeros}) : {} {order} {} ? {left} : {right}",
        nan(left),
        nan(right),
        key(left),
        key(right)
    )
}

// Each bit is either known zero, known one or unconstrained.
#[derive(Clone, Copy)]
struct KnownBits {
    zero: u32,
    one: u32,
    values: Option<[u32; 2]>,
}

impl KnownBits {
    const UNKNOWN: Self = Self {
        zero: 0,
        one: 0,
        values: None,
    };

    fn constant(value: u32) -> Self {
        Self {
            zero: !value,
            one: value,
            values: Some([value; 2]),
        }
    }

    fn and(self, other: Self) -> Self {
        Self {
            zero: self.zero | other.zero,
            one: self.one & other.one,
            values: None,
        }
    }

    fn or(self, other: Self) -> Self {
        Self {
            zero: self.zero & other.zero,
            one: self.one | other.one,
            values: None,
        }
    }

    fn xor(self, other: Self) -> Self {
        Self {
            zero: (self.zero & other.zero) | (self.one & other.one),
            one: (self.zero & other.one) | (self.one & other.zero),
            values: None,
        }
    }
}

/// Compiles explicitly tagged raw machine code to a synthetic Metal compute kernel.
///
/// Bind equal-length disjoint input and output word buffers and dispatch one
/// invocation per word in a one-dimensional grid. Equivalent to
/// [`compile_with`] using [`CompileOptions::default`].
pub fn compile(architecture: Architecture, bytes: &[u8]) -> Result<CompiledShader, CompileError> {
    compile_with(architecture, bytes, CompileOptions::default())
}

/// Compiles like [`compile`], with optional IR passes selected by `options`.
pub fn compile_with(
    architecture: Architecture,
    bytes: &[u8],
    options: CompileOptions,
) -> Result<CompiledShader, CompileError> {
    let decoded = decode_fixture(architecture, bytes)?;
    let mut operations = lift(&decoded, bytes.len())?;
    if options.fold_constants {
        fold_constant_ops(&mut operations);
    }
    // Fold first so a constant expression is `const.u32` before reuse. CSE does
    // not fold, and it does not turn those constants back into arithmetic.
    if options.eliminate_common_subexpressions {
        eliminate_common_subexpressions(&mut operations);
    }

    let disassembly = decoded.iter().map(|line| format!("{line}\n")).collect();
    let mut ir = String::new();
    let reads_grid = operations.iter().any(|(_, operation)| {
        matches!(
            operation,
            Op::LaneMask { .. } | Op::CarryMask { .. } | Op::ScalarLoad { .. }
        )
    });
    let grid = if reads_grid {
        ",\n    uint grid [[threads_per_grid]]"
    } else {
        ""
    };
    let samples = operations
        .iter()
        .any(|(_, operation)| matches!(operation, Op::Sample { .. }));
    // A texture and sampler pair must precede the grid attribute, so the
    // parameter list is built in one piece instead of rewriting the signature.
    let sampled = if samples {
        "    texture2d<float, access::sample> tex0 [[texture(0)]],\n    sampler samp0 [[sampler(0)]],\n    device atomic_uint* fault [[buffer(2)]],\n"
    } else {
        ""
    };
    let helpers = if samples { SAMPLE_DOMAIN_HELPERS } else { "" };
    let mut msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{helpers}kernel void {ENTRY_POINT}(\n    device const uint* input [[buffer(0)]],\n    device uint* output [[buffer(1)]],\n{sampled}    uint index [[thread_position_in_grid]]{grid}) {{\n"
    );
    if operations.iter().any(|(_, operation)| {
        matches!(
            operation,
            Op::Binary(
                Binary::Vector(VectorBinary::AddF32 | VectorBinary::SubF32 | VectorBinary::MulF32,),
                _,
                _
            ) | Op::Sample { .. }
        )
    }) {
        msl.push_str("    #pragma clang fp contract(off)\n    #pragma clang fp reassociate(off)\n");
    }
    // Wave-uniform values: identical in every lane of an RDNA wave.
    let mut uniform = Vec::with_capacity(operations.len());
    for (index, &(offset, operation)) in operations.iter().enumerate() {
        let dest = Value(index);
        uniform.push(match operation {
            Op::Constant(_) | Op::LaneMask { .. } | Op::CarryMask { .. } => true,
            Op::Load | Op::MergeLanes { .. } | Op::VccArithmetic { .. } => false,
            _ => operation.operands().iter().all(|value| uniform[value.0]),
        });
        let thread = |value: Value| value.to_string();
        let (step, statement) = match operation {
            Op::Load => (
                format!("{dest} = load.u32 input[index]"),
                format!(
                    "uint {dest} = {};",
                    operation.expression(&thread, "index", "")
                ),
            ),
            Op::ScalarLoad { word, old } => (
                format!("{dest} = load.uniform.u32 input[{word}], {old}"),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::Constant(value) => (
                format!("{dest} = const.u32 0x{value:08x}"),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::Unary(operator, source) => {
                let mnemonic = match operator {
                    VectorUnary::Not => "not.b32",
                    VectorUnary::ReverseBits => "brev.b32",
                    VectorUnary::FindHighUnsigned => "ffbh.u32",
                    VectorUnary::FindLow => "ffbl.b32",
                    VectorUnary::FindHighSigned => "ffbh.i32",
                    VectorUnary::TruncateF32 => "trunc.f32",
                    VectorUnary::CeilingF32 => "ceil.f32",
                    VectorUnary::RoundNearestEvenF32 => "rndne.f32",
                    VectorUnary::FloorF32 => "floor.f32",
                };
                (
                    format!("{dest} = {mnemonic} {source}"),
                    format!("uint {dest} = {};", operation.expression(&thread, "", "")),
                )
            }
            Op::Binary(operator, left, right) => (
                format!("{dest} = {} {left}, {right}", operator.operator().0),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::ScalarBranch { target, taken } => {
                let _ = writeln!(
                    ir,
                    "{offset:04x}: branch.uniform 0x{target:04x} taken={taken}"
                );
                continue;
            }
            Op::RuntimeBranch { test, mask, target } => {
                let _ = writeln!(ir, "{offset:04x}: branch.{test} {mask} -> 0x{target:04x}");
                continue;
            }
            Op::RuntimeElse { target } => {
                let _ = writeln!(ir, "{offset:04x}: else -> 0x{target:04x}");
                continue;
            }
            Op::ScalarTest {
                relation,
                signed,
                left,
                right,
            } => (
                format!(
                    "{dest} = test.uniform {relation:?}.{} {left}, {right}",
                    if signed { "i32" } else { "u32" }
                ),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::MergeLanes {
                mask,
                active,
                inactive,
            } => (
                format!("{dest} = merge.lanes mask=0x{mask:08x}, {active}, {inactive}"),
                format!(
                    "uint {dest} = {};",
                    operation.expression(&thread, "", "(index & 31u)")
                ),
            ),
            Op::JoinRuntime {
                test,
                mask,
                fallthrough,
                taken,
            } => (
                format!("{dest} = join.{test} {mask}, {fallthrough}, {taken}"),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::VccArithmetic {
                primitive,
                left,
                right,
                mask,
                flag,
            } => (
                format!(
                    "{dest} = vcc.{primitive:?}{suffix} {left}, {right}, {mask}",
                    suffix = if flag { ".flag" } else { "" }
                ),
                format!(
                    "uint {dest} = {};",
                    operation.expression(&thread, "", "(index & 31u)")
                ),
            ),
            Op::CarryMask { flag, exec } => {
                let _ = writeln!(
                    ir,
                    "{offset:04x}: {dest} = wave.carry {flag} exec=0x{exec:08x}"
                );
                emit_lane_mask(
                    &mut msl,
                    &operations,
                    &uniform,
                    index,
                    exec,
                    vec![flag],
                    |name| name(flag),
                );
                continue;
            }
            Op::LaneMask {
                op,
                name,
                sdst,
                src0,
                src1,
                high,
                exec,
            } => {
                let target = match sdst {
                    None => String::new(),
                    Some(VCC_LO) => " -> vcc_lo".to_string(),
                    Some(register) => format!(" -> s{register}"),
                };
                let (left, right) = if let Some((hi0, hi1)) = high {
                    (format!("{src0}:{hi0}"), format!("{src1}:{hi1}"))
                } else {
                    (src0.to_string(), src1.to_string())
                };
                let _ = writeln!(
                    ir,
                    "{offset:04x}: {dest} = wave.mask {name} {left}, {right}{target} exec=0x{exec:08x}"
                );
                let inputs = if matches!(
                    op.relation(),
                    xenora_decoder::CompareRelation::False | xenora_decoder::CompareRelation::True
                ) {
                    Vec::new()
                } else {
                    operation.operands()
                };
                emit_lane_mask(
                    &mut msl,
                    &operations,
                    &uniform,
                    index,
                    exec,
                    inputs,
                    |name| {
                        let (left, right) = if let Some((hi0, hi1)) = high {
                            let pair = |lo, hi| {
                                format!("((ulong({}) << 32u) | ulong({}))", name(hi), name(lo))
                            };
                            (pair(src0, hi0), pair(src1, hi1))
                        } else {
                            (name(src0), name(src1))
                        };
                        format!("({}) ? 1u : 0u", compare_expression(op, &left, &right))
                    },
                );
                continue;
            }
            Op::Store(value) => (
                format!("store.u32 output[index], {value}"),
                format!("output[index] = {value};"),
            ),
            Op::Reuse(source) => (
                format!("{dest} = reuse.u32 {source}"),
                format!("uint {dest} = {source};"),
            ),
            Op::Sample {
                u,
                v,
                lod,
                component,
                explicit_lod,
                exec,
            } => {
                let channel = ["r", "g", "b", "a"][usize::from(component)];
                let (form, level, lod_check) = if explicit_lod {
                    (
                        "sample_l.2d.f32",
                        format!(", lod {lod}"),
                        format!(
                            " if (!xenora_whole_level({lod}, tex0.get_num_mip_levels())) {{ bad |= {SAMPLE_FAULT_LOD}u; }}"
                        ),
                    )
                } else {
                    ("sample_lz.2d.f32", String::new(), String::new())
                };
                let active = if exec == u32::MAX {
                    String::new()
                } else {
                    format!(" && ((0x{exec:08x}u >> (index & 31u)) & 1u) != 0u")
                };
                (
                    format!("{form} tex0, samp0, ({u}, {v}){level}[{channel}] exec=0x{exec:08x}"),
                    format!(
                        "{{ uint bad = 0u; if (!xenora_finite({u}) || !xenora_finite({v})) {{ bad |= {SAMPLE_FAULT_COORDINATES}u; }}{lod_check} if (bad != 0u{active}) {{ atomic_fetch_or_explicit(fault, bad, memory_order_relaxed); }} }}\n    uint {dest} = {};",
                        operation.expression(&thread, "", "")
                    ),
                )
            }
        };
        // Writing to a String is infallible.
        let _ = writeln!(ir, "{offset:04x}: {step}");
        let _ = writeln!(msl, "    {statement}");
    }
    msl.push_str("}\n");
    Ok(CompiledShader {
        disassembly,
        ir,
        msl,
    })
}

/// Emit a Wave32 predicate mask without any cross-thread operation: the thread
/// recomputes the lane-dependent operands of each lane of its own RDNA wave
/// (dispatch indices `(index & ~31) + n`) from those lanes' input words.
/// Absent lanes of a final partial wave contribute 0.
fn emit_lane_mask(
    msl: &mut String,
    operations: &[(usize, Op)],
    uniform: &[bool],
    index: usize,
    exec: u32,
    mut pending: Vec<Value>,
    predicate: impl FnOnce(&dyn Fn(Value) -> String) -> String,
) {
    let mut cone = vec![false; index];
    while let Some(value) = pending.pop() {
        if uniform[value.0] || cone[value.0] {
            continue;
        }
        cone[value.0] = true;
        pending.extend(operations[value.0].1.operands());
    }
    let name = |value: Value| {
        if cone[value.0] {
            format!("l{}", value.0)
        } else {
            value.to_string()
        }
    };
    let dest = Value(index);
    let _ = writeln!(msl, "    uint {dest} = 0u;");
    let _ = writeln!(
        msl,
        "    for (uint lane = 0u; lane < 32u; ++lane) {{\n        uint lane_index = (index & ~31u) + lane;\n        if (lane_index >= grid) {{ break; }}"
    );
    for (position, &(_, operation)) in operations[..index].iter().enumerate() {
        if cone[position] {
            let _ = writeln!(
                msl,
                "        uint l{position} = {};",
                operation.expression(&name, "lane_index", "lane")
            );
        }
    }
    let _ = writeln!(
        msl,
        "        uint pass = {};\n        {dest} |= ((0x{exec:08x}u >> lane) & pass) << lane;\n    }}",
        predicate(&name)
    );
}

fn decode_fixture(
    architecture: Architecture,
    bytes: &[u8],
) -> Result<Vec<DecodedInstruction>, CompileError> {
    if bytes.is_empty() || bytes.len() > MAX_CODE_BYTES || !bytes.len().is_multiple_of(4) {
        return Err(CompileError::InvalidLength { bytes: bytes.len() });
    }
    xenora_decoder::decode(architecture, bytes).map_err(CompileError::Decode)
}

#[derive(Clone, Copy)]
enum ScalarValue {
    Uninitialized,
    Known(u32),
    RuntimeMask,
    /// A wave-uniform value known only at run time.
    Runtime(Value),
    /// The destination of a scalar load without a completed wait.
    Pending(Value),
}

/// SCC during translation.
#[derive(Clone, Copy)]
enum Scc {
    Uninitialized,
    Known(bool),
    /// A wave-uniform 0 or 1 known only at run time.
    Runtime(Value),
}

impl Scc {
    fn known(self, uninitialized: &'static str) -> Result<bool, &'static str> {
        match self {
            Self::Known(value) => Ok(value),
            Self::Runtime(_) => Err("instruction reads a run-time SCC"),
            Self::Uninitialized => Err(uninitialized),
        }
    }
}

/// The code a run-time branch may skip, and its `else` part if there is one.
struct Region {
    /// End of the part being translated, where the paths join after the last.
    end: usize,
    test: &'static str,
    mask: Value,
    /// The VGPRs at the branch.
    before: [Option<Value>; 256],
    /// The VGPRs after the fall-through part, while translating an `else`.
    fallthrough: Option<[Option<Value>; 256]>,
}

fn scalar_load_pending(scalars: &[ScalarValue]) -> bool {
    scalars
        .iter()
        .any(|value| matches!(value, ScalarValue::Pending(_)))
}

/// Bit set in the `[[buffer(2)]]` fault word when an active lane's explicit LOD
/// is not a whole mip level of the bound texture (the reference's `InvalidLod`).
pub const SAMPLE_FAULT_LOD: u32 = 1;
/// Bit set in the fault word when an active lane's coordinate is NaN or
/// infinite (the reference's `NonFiniteCoordinates`).
pub const SAMPLE_FAULT_COORDINATES: u32 = 2;

/// MSL helpers for the sample domain check. Both compare raw bits, so the
/// check does not depend on the compiler's floating-point mode. A whole level
/// is -0.0 or the exact binary32 value of a level below `levels`, the domain
/// `xenora_ref::texture_sampling::lod_to_mip` accepts.
const SAMPLE_DOMAIN_HELPERS: &str = "static inline bool xenora_finite(uint bits) {\n    return (bits & 0x7f800000u) != 0x7f800000u;\n}\nstatic inline bool xenora_whole_level(uint bits, uint levels) {\n    if (bits == 0x80000000u) { return true; }\n    for (uint level = 0u; level < levels; ++level) {\n        if (bits == as_type<uint>(float(level))) { return true; }\n    }\n    return false;\n}\n";

/// The synthetic binding ABI load that may provide descriptor register `slot`.
fn descriptor_load(slot: usize) -> Option<(usize, u32, &'static str)> {
    match slot {
        8..=11 => Some((8, 0x0, "s_buffer_load_dwordx4 s[8:11], s[0:3], 0x0")),
        12..=15 => Some((12, 0x10, "s_buffer_load_dwordx4 s[12:15], s[0:3], 0x10")),
        16..=19 => Some((16, 0x20, "s_buffer_load_dwordx4 s[16:19], s[0:3], 0x20")),
        _ => None,
    }
}

/// Forget the provenance of a descriptor register after any other write.
fn clear_descriptor_origin(origins: &mut [Option<u32>; 128], register: usize) {
    if (8..20).contains(&register) {
        origins[register] = None;
    }
}

/// Every image sample reads the three synthetic descriptor loads in order.
fn verify_descriptor_provenance(
    origins: &[Option<u32>; 128],
    offset: usize,
) -> Result<(), CompileError> {
    for (slot, origin) in origins.iter().enumerate().take(20).skip(8) {
        let (base, expected_offset, expected) =
            descriptor_load(slot).expect("descriptor registers are contiguous");
        debug_assert_eq!(base, 8 + ((slot - 8) / 4) * 4);
        if *origin != Some(expected_offset) {
            return Err(CompileError::DescriptorProvenance {
                offset,
                register: slot as u8,
                expected,
            });
        }
    }
    Ok(())
}

fn lift(decoded: &[DecodedInstruction], code_len: usize) -> Result<Vec<(usize, Op)>, CompileError> {
    let mut operations = Vec::new();
    let mut registers = [None::<Value>; 256];
    let mut facts = Vec::<KnownBits>::new();
    let mut pending_load = None;
    let mut stored = false;
    let mut store_pending = false;
    let mut scalars = [ScalarValue::Uninitialized; 128];
    // For s8..s19, the byte offset of the `s_buffer_load_dwordx4` that last
    // wrote the register, or `None` after any other write. The image sample
    // requires the synthetic binding ABI's three descriptor loads.
    let mut descriptor_origin = [None::<u32>; 128];
    let mut exec = u32::MAX; // The fixture starts with full Wave32 EXEC.
    scalars[126] = ScalarValue::Known(exec);
    let mut scc = Scc::Uninitialized;
    let mut skip_before = 0;
    // VCC_LO can forward a lane-derived mask to its supported runtime consumers.
    let mut lane_vcc: Option<Value> = None;
    let mut region: Option<Region> = None;

    // Every decoded instruction must satisfy the fixture layout and binding
    // contract, even when a known branch skips it.
    for (index, instruction) in decoded.iter().enumerate() {
        let offset = instruction.offset;
        if let Some(target) = instruction.branch_target()
            && (target <= offset
                || decoded
                    .binary_search_by_key(&target, |instruction| instruction.offset)
                    .is_err())
        {
            return Err(CompileError::Verify {
                offset,
                reason: "branch target must be a forward instruction boundary",
            });
        }
        let reason = match instruction.instruction {
            Instruction::BufferLoadDword {
                index, resource, ..
            } if (index, resource) != (0, 0) => "unsupported resource or buffer index",
            Instruction::BufferStoreDword {
                index, resource, ..
            } if (index, resource) != (0, 4) => "unsupported resource or buffer index",
            Instruction::ScalarBufferLoad { resource, .. } if resource != 0 => {
                "unsupported resource or buffer index"
            }
            Instruction::Export { .. } | Instruction::Interpolate { .. } => {
                "graphics instruction is unsupported in a compute fixture"
            }
            Instruction::ImageSampleLz {
                resource, sampler, ..
            } if (resource, sampler) != (8, 16) => {
                "sampling outside synthetic resource binding ABI s[8:15] and s[16:19]"
            }
            Instruction::ImageSampleL {
                resource, sampler, ..
            } if (resource, sampler) != (8, 16) => {
                "sampling outside synthetic resource binding ABI s[8:15] and s[16:19]"
            }
            Instruction::EndProgram if index + 1 != decoded.len() => {
                return Err(CompileError::Verify {
                    offset: decoded[index + 1].offset,
                    reason: "instruction follows program termination",
                });
            }
            _ => continue,
        };
        return Err(CompileError::Verify { offset, reason });
    }

    for instruction in decoded {
        let offset = instruction.offset;
        let error = |reason| CompileError::Verify { offset, reason };
        if let Some(Region {
            end,
            test,
            mask,
            before,
            fallthrough,
        }) = region.take_if(|region| offset >= region.end)
        {
            // Without an `else`, the taken path leaves the VGPRs as they were.
            let (fallthrough, taken) = match fallthrough {
                Some(fallthrough) => (fallthrough, registers),
                None => (registers, before),
            };
            for slot in 0..registers.len() {
                if let (Some(fallthrough), Some(taken)) = (fallthrough[slot], taken[slot])
                    && fallthrough.0 != taken.0
                {
                    let next = Value(operations.len());
                    operations.push((
                        end,
                        Op::JoinRuntime {
                            test,
                            mask,
                            fallthrough,
                            taken,
                        },
                    ));
                    let (left, right) = (facts[fallthrough.0], facts[taken.0]);
                    // Keep up to two candidate values when both edges have them.
                    let values = match (left.values, right.values) {
                        (Some([a, b]), Some([c, d])) => {
                            let mut union = vec![a, b, c, d];
                            union.sort_unstable();
                            union.dedup();
                            match union.as_slice() {
                                [only] => Some([*only; 2]),
                                [first, second] => Some([*first, *second]),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    facts.push(KnownBits {
                        zero: left.zero & right.zero,
                        one: left.one & right.one,
                        values,
                    });
                    registers[slot] = Some(next);
                }
            }
        }
        // docs/shaders/vector-compare.md section 6: checked whatever EXEC is.
        if let Some(region) = &mut region {
            // An `s_branch` that ends the fall-through part skips an `else`. The
            // `s_cbranch_vccz` contract has none.
            if let Instruction::Branch { target } = instruction.instruction
                && region.test == "scc0"
                && region.fallthrough.is_none()
                && offset + usize::from(instruction.byte_len) == region.end
            {
                region.fallthrough = Some(registers);
                registers = region.before;
                region.end = target;
                operations.push((offset, Op::RuntimeElse { target }));
                facts.push(KnownBits::UNKNOWN);
                continue;
            }
            let before = &region.before;
            let dst = match instruction.instruction {
                Instruction::MoveB32 { dst, .. }
                // Of the VOP3 variants, the contract admits only v_add_nc_u32_e64.
                | Instruction::VectorAluVop3 {
                    op: VectorAluVop3Op::AddNcU32,
                    dst,
                    ..
                }
                // Of the registry VOP2 forms, the contract admits only the six
                // logical and shift forms that were hand-written before (#215 M27).
                | Instruction::VectorAlu {
                    op:
                        VectorAluOp::LshlrevB32
                        | VectorAluOp::LshrrevB32
                        | VectorAluOp::AshrrevI32
                        | VectorAluOp::AndB32
                        | VectorAluOp::OrB32
                        | VectorAluOp::XorB32,
                    dst,
                    ..
                } => dst,
                _ => {
                    return Err(error(
                        "code skipped by a runtime branch may contain only v_mov_b32, v_add_nc_u32_e64 and the VOP2 logical and shift forms",
                    ));
                }
            };
            if before[usize::from(dst)].is_none() {
                return Err(error(
                    "a VGPR written under a runtime branch must be initialized before the branch",
                ));
            }
        }
        let read = |register: u8| {
            registers[usize::from(register)].ok_or_else(|| {
                error("source register is uninitialized, pending a load wait, or reserved for indexing")
            })
        };
        if offset < skip_before {
            continue;
        }
        let read_scalar = |source| match source {
            ScalarSource::Immediate(value) => Ok(value),
            ScalarSource::Register(register) => match scalars.get(usize::from(register)) {
                Some(ScalarValue::Known(value)) => Ok(*value),
                Some(ScalarValue::RuntimeMask) => Err(error(
                    "scalar source reads a runtime-only SGPR compare mask",
                )),
                Some(ScalarValue::Runtime(_)) => {
                    Err(error("scalar source reads a run-time loaded SGPR"))
                }
                Some(ScalarValue::Pending(_)) => Err(error(
                    "scalar source reads an SGPR pending a scalar load wait",
                )),
                Some(ScalarValue::Uninitialized) | None => {
                    Err(error("scalar source has no known value"))
                }
            },
        };
        let is_runtime = |source| {
            matches!(source, ScalarSource::Register(register) if matches!(
                scalars.get(usize::from(register)),
                Some(ScalarValue::Runtime(_))
            ))
        };
        // A wave-uniform operand of a vector instruction or a scalar compare.
        let read_uniform = |register: u8, operations: &mut Vec<_>, facts: &mut Vec<_>| {
            if let Some(ScalarValue::Runtime(value)) = scalars.get(usize::from(register)) {
                return Ok(*value);
            }
            let bits = read_scalar(ScalarSource::Register(register))?;
            Ok(constant(operations, facts, offset, bits))
        };
        let read_vector = |source, operations: &mut Vec<_>, facts: &mut Vec<_>| {
            Ok(match source {
                VectorSource::Register(register) => read(register)?,
                VectorSource::Scalar(register) if usize::from(register) == VCC_LO => {
                    match (lane_vcc, scalars[VCC_LO]) {
                        (Some(mask), _) => mask,
                        (None, ScalarValue::Known(bits)) => {
                            constant(operations, facts, offset, bits)
                        }
                        (None, _) => {
                            return Err(error("vector ALU reads uninitialized VCC_LO"));
                        }
                    }
                }
                VectorSource::Scalar(register) => read_uniform(register, operations, facts)?,
                VectorSource::Immediate(bits) => constant(operations, facts, offset, bits),
            })
        };
        // Scalar instructions and waits still execute when all lanes are inactive.
        if exec == 0 {
            match instruction.instruction {
                Instruction::VectorAlu { op, .. } if op.writes_vcc() => {
                    // Guide 3.9: inactive lanes contribute zero to carry-out.
                    lane_vcc = None;
                    scalars[VCC_LO] = ScalarValue::Known(0);
                    scalars[VCC_HI] = ScalarValue::Uninitialized;
                    continue;
                }
                Instruction::MoveB32 { .. }
                | Instruction::VectorAluVop3 { .. }
                | Instruction::VectorAlu { .. }
                | Instruction::VectorUnary { .. }
                | Instruction::VectorSwap { .. } => continue,
                _ => {}
            }
        }
        if exec != u32::MAX
            && matches!(
                instruction.instruction,
                Instruction::BufferLoadDword { .. } | Instruction::BufferStoreDword { .. }
            )
        {
            return Err(error("vector memory requires full Wave32 EXEC"));
        }
        if let Instruction::BufferLoadDword { resource, .. }
        | Instruction::BufferStoreDword { resource, .. } = instruction.instruction
        {
            let start = usize::from(resource);
            if scalars[start..start + 4]
                .iter()
                .any(|value| matches!(value, ScalarValue::RuntimeMask))
            {
                return Err(error(
                    "resource descriptor reads a runtime-only SGPR compare mask",
                ));
            }
        }
        if let Instruction::ScalarUnary { dst, .. }
        | Instruction::ScalarAlu { dst, .. }
        | Instruction::ScalarImmediate { dst, .. }
        | Instruction::VectorCompareVop3 { sdst: dst, .. } = instruction.instruction
            && matches!(scalars.get(usize::from(dst)), Some(ScalarValue::Pending(_)))
        {
            return Err(error(
                "instruction writes an SGPR pending a scalar load wait",
            ));
        }
        let next = Value(operations.len());
        let (dest, operation) = match instruction.instruction {
            Instruction::BufferLoadDword { data, .. } => {
                let slot = usize::from(data);
                if data == 0 || pending_load.is_some() {
                    return Err(error(
                        "load writes the index register or another load is outstanding",
                    ));
                }
                registers[slot] = None;
                pending_load = Some((slot, next));
                (None, Op::Load)
            }
            Instruction::BufferStoreDword { data, .. } => {
                if stored {
                    return Err(error("only one output store is supported"));
                }
                stored = true;
                store_pending = true;
                (None, Op::Store(read(data)?))
            }
            Instruction::WaitStore => {
                store_pending = false;
                continue;
            }
            // docs/shaders/scalar-memory.md.
            Instruction::ScalarBufferLoad {
                dst,
                offset: immediate,
                soffset,
                dwords,
                ..
            } => {
                if dst < 8 {
                    return Err(error("scalar load would modify a resource descriptor"));
                }
                let extra = match soffset {
                    Some(register) => read_scalar(ScalarSource::Register(register))?,
                    None => 0,
                };
                let first = (u64::from(immediate) + u64::from(extra)) / 4;
                let mut loaded = Vec::new();
                for index in 0..dwords {
                    let old = read_uniform(dst + index, &mut operations, &mut facts)?;
                    let word = first + u64::from(index);
                    loaded.push(match u32::try_from(word) {
                        Ok(word) if word < u32::MAX => {
                            let value = Value(operations.len());
                            operations.push((offset, Op::ScalarLoad { word, old }));
                            facts.push(KnownBits::UNKNOWN);
                            value
                        }
                        // Beyond any dispatch: the word is not loaded.
                        _ => old,
                    });
                }
                for (slot, value) in (usize::from(dst)..).zip(loaded) {
                    scalars[slot] = ScalarValue::Pending(value);
                }
                // A scalar load writes every destination register. Record the
                // synthetic descriptor loads so an image sample cannot consume
                // arbitrary SGPR content.
                for slot in usize::from(dst)..usize::from(dst) + usize::from(dwords) {
                    clear_descriptor_origin(&mut descriptor_origin, slot);
                }
                let expected = if soffset.is_none() && dwords == 4 {
                    match (dst, immediate) {
                        (8, 0x0) => Some(0x0),
                        (12, 0x10) => Some(0x10),
                        (16, 0x20) => Some(0x20),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(expected) = expected {
                    for slot in &mut descriptor_origin[usize::from(dst)..usize::from(dst) + 4] {
                        *slot = Some(expected);
                    }
                }
                continue;
            }
            Instruction::WaitScalarLoad => {
                for slot in &mut scalars {
                    if let ScalarValue::Pending(value) = *slot {
                        *slot = ScalarValue::Runtime(value);
                    }
                }
                continue;
            }
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Register(src),
            } => {
                let value = read(src)?;
                let slot = usize::from(dst);
                if dst == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                    return Err(error(
                        "move writes the index register or replaces a pending load",
                    ));
                }
                registers[slot] = Some(merge_lanes(
                    &mut operations,
                    &mut facts,
                    offset,
                    exec,
                    value,
                    registers[slot],
                )?);
                continue;
            }
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Immediate(value),
            } => (Some(dst), Op::Constant(value)),
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Scalar(register),
            } if usize::from(register) == VCC_LO
                && let Some(mask) = lane_vcc =>
            {
                let slot = usize::from(dst);
                if dst == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                    return Err(error(
                        "move writes the index register or replaces a pending load",
                    ));
                }
                registers[slot] = Some(merge_lanes(
                    &mut operations,
                    &mut facts,
                    offset,
                    exec,
                    mask,
                    registers[slot],
                )?);
                continue;
            }
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Scalar(register),
            } if let ScalarValue::Runtime(value) = scalars[usize::from(register)] => {
                let slot = usize::from(dst);
                if dst == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                    return Err(error(
                        "move writes the index register or replaces a pending load",
                    ));
                }
                registers[slot] = Some(merge_lanes(
                    &mut operations,
                    &mut facts,
                    offset,
                    exec,
                    value,
                    registers[slot],
                )?);
                continue;
            }
            // Other SGPR values are wave-uniform and known during translation.
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Scalar(register),
            } => (
                Some(dst),
                Op::Constant(read_scalar(ScalarSource::Register(register))?),
            ),
            Instruction::VectorAluVop3 {
                op,
                dst,
                src0,
                src1,
            } => {
                let op = op.base();
                let (left, right) = if op.swaps_sources() {
                    (read(src1)?, read(src0)?)
                } else {
                    (read(src0)?, read(src1)?)
                };
                let VectorLowering::Binary(binary) = op.lowering() else {
                    return Err(error("VOP3 with a lane condition is unsupported"));
                };
                (Some(dst), Op::Binary(Binary::Vector(binary), left, right))
            }
            Instruction::ScalarUnary {
                op: op @ ScalarUnaryOp::MovB32,
                dst,
                src,
            } => {
                let (value, _, _) = op.evaluate(read_scalar(src)?, exec);
                if dst < 8 {
                    return Err(error("scalar move would modify a resource descriptor"));
                }
                if dst == 126 {
                    exec = value;
                }
                if usize::from(dst) == VCC_LO {
                    lane_vcc = None;
                }
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                *slot = ScalarValue::Known(value);
                clear_descriptor_origin(&mut descriptor_origin, usize::from(dst));
                continue;
            }
            Instruction::ScalarUnary { op, dst, src } => {
                let src = read_scalar(src)?;
                if dst < 8 {
                    return Err(error("scalar unary would modify a resource descriptor"));
                }
                let (value, mask, flag) = op.evaluate(src, exec);
                if op.writes_exec() {
                    // The EXEC-writing forms combine the source with the mask and
                    // write the result back to EXEC_LO. The mask is a compile-time
                    // value here, so the destination and the mask agree.
                    exec = mask;
                    scalars[EXEC_LO] = ScalarValue::Known(mask);
                }
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                *slot = ScalarValue::Known(value);
                clear_descriptor_origin(&mut descriptor_origin, usize::from(dst));
                if let Some(flag) = flag {
                    scc = Scc::Known(flag);
                }
                continue;
            }
            Instruction::ScalarAlu {
                op,
                dst,
                src0,
                src1,
            } => {
                let (src0, src1) = (read_scalar(src0)?, read_scalar(src1)?);
                let scc_in = if op.reads_scc() {
                    scc.known("SOP2 SCC input reads uninitialized SCC")
                        .map_err(error)?
                } else {
                    false
                };
                if dst < 8 {
                    return Err(error("scalar ALU would modify a resource descriptor"));
                }
                let (value, flag) = op.evaluate(src0, src1, scc_in);
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                *slot = ScalarValue::Known(value);
                clear_descriptor_origin(&mut descriptor_origin, usize::from(dst));
                if let Some(flag) = flag {
                    scc = Scc::Known(flag);
                }
                continue;
            }
            Instruction::ScalarCompare { op, src0, src1 } => {
                let operand = |source, operations: &mut Vec<_>, facts: &mut Vec<_>| {
                    Ok(match source {
                        ScalarSource::Register(register) => {
                            read_uniform(register, operations, facts)?
                        }
                        ScalarSource::Immediate(bits) => constant(operations, facts, offset, bits),
                    })
                };
                if !(is_runtime(src0) || is_runtime(src1)) {
                    scc = Scc::Known(op.test(read_scalar(src0)?, read_scalar(src1)?));
                    continue;
                }
                let left = operand(src0, &mut operations, &mut facts)?;
                let right = operand(src1, &mut operations, &mut facts)?;
                let Some(relation) = op.semantics().relation else {
                    return Err(error("scalar compare without a relation"));
                };
                scc = Scc::Runtime(Value(operations.len()));
                operations.push((
                    offset,
                    Op::ScalarTest {
                        relation,
                        signed: op.semantics().signedness == Signedness::Signed,
                        left,
                        right,
                    },
                ));
                facts.push(KnownBits::UNKNOWN);
                continue;
            }
            // docs/shaders/scalar-immediate.md.
            Instruction::ScalarImmediate { op, dst, simm16 } => {
                if dst < 8 {
                    return Err(error("scalar immediate would modify a resource descriptor"));
                }
                let scc_in = if op.reads_scc() {
                    scc.known("SOPK SCC input reads uninitialized SCC")
                        .map_err(error)?
                } else {
                    false
                };
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                let old = if op.reads_dst() {
                    match *slot {
                        ScalarValue::Known(value) => value,
                        ScalarValue::RuntimeMask => {
                            return Err(error("SOPK reads a runtime-only SGPR compare mask"));
                        }
                        ScalarValue::Runtime(_) | ScalarValue::Pending(_) => {
                            return Err(error("SOPK reads a run-time loaded SGPR"));
                        }
                        ScalarValue::Uninitialized => {
                            return Err(error("SOPK reads an uninitialized SGPR"));
                        }
                    }
                } else {
                    0
                };
                let (value, flag) = op.evaluate(old, simm16, scc_in);
                if let Some(value) = value {
                    *slot = ScalarValue::Known(value);
                    clear_descriptor_origin(&mut descriptor_origin, usize::from(dst));
                }
                if let Some(flag) = flag {
                    scc = Scc::Known(flag);
                }
                continue;
            }
            Instruction::ScalarImmediateCompare { op, src, simm16 } => {
                if !is_runtime(ScalarSource::Register(src)) {
                    scc = Scc::Known(op.test(read_scalar(ScalarSource::Register(src))?, simm16));
                    continue;
                }
                let signed = op.semantics().signedness == Signedness::Signed;
                // Guide 12.2: the unsigned compares zero-extend SIMM16.
                let extended = if signed {
                    i32::from(simm16 as i16) as u32
                } else {
                    u32::from(simm16)
                };
                let left = read_uniform(src, &mut operations, &mut facts)?;
                let right = constant(&mut operations, &mut facts, offset, extended);
                scc = Scc::Runtime(Value(operations.len()));
                operations.push((
                    offset,
                    Op::ScalarTest {
                        relation: op.relation(),
                        signed,
                        left,
                        right,
                    },
                ));
                facts.push(KnownBits::UNKNOWN);
                continue;
            }
            // docs/shaders/vector-compare.md sections 3 and 5.
            // Both encodings share the Wave32 lane mask implementation.
            compare @ (Instruction::VectorCompare { .. }
            | Instruction::VectorCompareVop3 { .. }) => {
                let (op, name, explicit, sdst, src0, src1) = match compare {
                    Instruction::VectorCompare { op, src0, src1 } => {
                        (op, op.assembly_name(), false, VCC_LO, src0, src1)
                    }
                    Instruction::VectorCompareVop3 {
                        op,
                        sdst,
                        src0,
                        src1,
                    } => (
                        op.base(),
                        op.assembly_name(),
                        true,
                        usize::from(sdst),
                        VectorSource::Register(src0),
                        src1,
                    ),
                    _ => unreachable!("only vector compares reach this arm"),
                };
                if exec == 0 {
                    if sdst == VCC_LO {
                        scalars[VCC_HI] = ScalarValue::Uninitialized;
                        scalars[VCC_LO] = ScalarValue::Known(0);
                        lane_vcc = None;
                    } else {
                        scalars[sdst] = ScalarValue::RuntimeMask;
                        clear_descriptor_origin(&mut descriptor_origin, sdst);
                    }
                    continue;
                }
                let constant =
                    |operations: &mut Vec<(usize, Op)>, facts: &mut Vec<KnownBits>, bits: u32| {
                        let value = Value(operations.len());
                        operations.push((offset, Op::Constant(bits)));
                        facts.push(KnownBits::constant(bits));
                        value
                    };
                let high = if op.semantics().width == 64 {
                    // Guide §3.6.3 and contract §2a put the high word at base + 1.
                    let left = match src0 {
                        VectorSource::Register(base) => read(base + 1)?,
                        VectorSource::Scalar(base) => {
                            read_uniform(base + 1, &mut operations, &mut facts)?
                        }
                        VectorSource::Immediate(bits) => {
                            let high = ((bits as i32) >> 31) as u32;
                            constant(&mut operations, &mut facts, high)
                        }
                    };
                    Some((left, read(src1 + 1)?))
                } else {
                    None
                };
                let src0 = match src0 {
                    VectorSource::Register(register) => read(register)?,
                    VectorSource::Scalar(register) if usize::from(register) == VCC_LO => {
                        match (lane_vcc, scalars[VCC_LO]) {
                            (Some(mask), _) => mask,
                            (None, ScalarValue::Known(bits)) => {
                                constant(&mut operations, &mut facts, bits)
                            }
                            (None, _) => {
                                return Err(error("compare reads uninitialized VCC_LO"));
                            }
                        }
                    }
                    VectorSource::Scalar(register) => {
                        read_uniform(register, &mut operations, &mut facts)?
                    }
                    VectorSource::Immediate(bits) => constant(&mut operations, &mut facts, bits),
                };
                let src1 = read(src1)?;
                let mask = Value(operations.len());
                operations.push((
                    offset,
                    Op::LaneMask {
                        op,
                        name,
                        sdst: explicit.then_some(sdst),
                        src0,
                        src1,
                        high,
                        exec,
                    },
                ));
                facts.push(KnownBits::UNKNOWN);
                if sdst == VCC_LO {
                    scalars[VCC_LO] = ScalarValue::Uninitialized;
                    scalars[VCC_HI] = ScalarValue::Uninitialized;
                    lane_vcc = Some(mask);
                } else {
                    scalars[sdst] = ScalarValue::RuntimeMask;
                    clear_descriptor_origin(&mut descriptor_origin, sdst);
                }
                continue;
            }
            Instruction::Branch { .. }
            | Instruction::BranchSccZero { .. }
            | Instruction::BranchExecZero { .. }
            | Instruction::BranchVccZero { .. }
            | Instruction::WaitLoad
            | Instruction::EndProgram => {
                match instruction
                    .instruction
                    .sopp_control()
                    .expect("decoded SOPP control")
                {
                    SoppControlOp::WaitLoad => {
                        if let Some((register, value)) = pending_load.take() {
                            registers[register] = Some(value);
                        }
                        continue;
                    }
                    SoppControlOp::EndProgram => {
                        if !stored
                            || store_pending
                            || pending_load.is_some()
                            || scalar_load_pending(&scalars)
                        {
                            return Err(error(
                                "termination requires one completed store and no pending loads",
                            ));
                        }
                        return Ok(operations);
                    }
                    op => {
                        let target = instruction.branch_target().expect("decoded SOPP branch");
                        let runtime = match (op, lane_vcc, scc) {
                            (SoppControlOp::BranchVccZero, Some(mask), _) => Some(("vccz", mask)),
                            (SoppControlOp::BranchSccZero, _, Scc::Runtime(value)) => {
                                Some(("scc0", value))
                            }
                            _ => None,
                        };
                        if let Some((test, mask)) = runtime {
                            if pending_load.is_some() {
                                return Err(error(
                                    "a runtime branch requires no outstanding vector load",
                                ));
                            }
                            if scalar_load_pending(&scalars) {
                                return Err(error(
                                    "a runtime branch requires no outstanding scalar load",
                                ));
                            }
                            region = Some(Region {
                                end: target,
                                test,
                                mask,
                                before: registers,
                                fallthrough: None,
                            });
                            (None, Op::RuntimeBranch { test, mask, target })
                        } else {
                            let taken = match op {
                                SoppControlOp::Branch => true,
                                SoppControlOp::BranchSccZero => !scc
                                    .known("conditional branch reads uninitialized SCC")
                                    .map_err(error)?,
                                SoppControlOp::BranchVccZero => {
                                    let ScalarValue::Known(bits) = scalars[VCC_LO] else {
                                        return Err(error(
                                            "conditional branch reads uninitialized VCC_LO",
                                        ));
                                    };
                                    bits == 0
                                }
                                SoppControlOp::BranchExecZero => exec == 0,
                                SoppControlOp::WaitLoad | SoppControlOp::EndProgram => {
                                    unreachable!()
                                }
                            };
                            if taken {
                                skip_before = target;
                            }
                            (None, Op::ScalarBranch { target, taken })
                        }
                    }
                }
            }
            Instruction::VectorAlu {
                op,
                dst,
                src0,
                src1,
            } => {
                let src0 = read_vector(src0, &mut operations, &mut facts)?;
                let src1 = read(src1)?;
                let (left, right) = if op.swaps_sources() {
                    (src1, src0)
                } else {
                    (src0, src1)
                };
                match op.lowering() {
                    VectorLowering::Binary(binary) => {
                        (Some(dst), Op::Binary(Binary::Vector(binary), left, right))
                    }
                    VectorLowering::Vcc(primitive) => {
                        let mask = match (lane_vcc, scalars[VCC_LO]) {
                            (Some(mask), _) => mask,
                            (None, ScalarValue::Known(bits)) => {
                                constant(&mut operations, &mut facts, offset, bits)
                            }
                            (None, _) => {
                                return Err(error("vector ALU reads uninitialized VCC_LO"));
                            }
                        };
                        if op.writes_vcc() {
                            let flag = Value(operations.len());
                            operations.push((
                                offset,
                                Op::VccArithmetic {
                                    primitive,
                                    left,
                                    right,
                                    mask,
                                    flag: true,
                                },
                            ));
                            facts.push(KnownBits::UNKNOWN);
                            lane_vcc = Some(Value(operations.len()));
                            operations.push((offset, Op::CarryMask { flag, exec }));
                            facts.push(KnownBits::UNKNOWN);
                            scalars[VCC_LO] = ScalarValue::Uninitialized;
                            scalars[VCC_HI] = ScalarValue::Uninitialized;
                        }
                        (
                            Some(dst),
                            Op::VccArithmetic {
                                primitive,
                                left,
                                right,
                                mask,
                                flag: false,
                            },
                        )
                    }
                }
            }
            Instruction::VectorUnary { op, dst, src } => {
                let source = read_vector(src, &mut operations, &mut facts)?;
                (Some(dst), Op::Unary(op.lowering(), source))
            }
            Instruction::VectorSwap { dst, src, .. } => {
                // Guide 12.8: capture both old values before updating either VGPR.
                // read() also rejects v0 and registers with outstanding loads.
                let old_dst = read(dst)?;
                let old_src = read(src)?;
                if dst != src {
                    registers[usize::from(dst)] = Some(merge_lanes(
                        &mut operations,
                        &mut facts,
                        offset,
                        exec,
                        old_src,
                        Some(old_dst),
                    )?);
                    registers[usize::from(src)] = Some(merge_lanes(
                        &mut operations,
                        &mut facts,
                        offset,
                        exec,
                        old_dst,
                        Some(old_src),
                    )?);
                }
                continue;
            }
            Instruction::ImageSampleLz {
                dst, coord, dmask, ..
            }
            | Instruction::ImageSampleL {
                dst, coord, dmask, ..
            } => {
                // The synthetic binding (resource s[8:15] -> texture(0), sampler
                // s[16:19] -> sampler(0)) was already checked in the verify loop.
                // The descriptors must be the three s_buffer_load_dwordx4 words
                // of the synthetic input, with no later writes before the sample.
                if scalar_load_pending(&scalars) {
                    return Err(error("image sample before the scalar descriptor load wait"));
                }
                verify_descriptor_provenance(&descriptor_origin, offset)?;
                let explicit_lod =
                    matches!(instruction.instruction, Instruction::ImageSampleL { .. });
                let u = read(coord)?;
                let v = read(coord + 1)?;
                let lod = if explicit_lod {
                    read(coord + 2)?
                } else {
                    let zero = Value(operations.len());
                    facts.push(KnownBits::constant(0.0f32.to_bits()));
                    operations.push((offset, Op::Constant(0.0f32.to_bits())));
                    zero
                };
                // Guide Table 41, printed page 69: DMASK bit 0 = red, bit 1 =
                // green, bit 2 = blue, bit 3 = alpha; selected components are
                // written to consecutive destination VGPRs in that order.
                let mut next_dst = dst;
                for component in 0..4u8 {
                    if dmask & (1 << component) == 0 {
                        continue;
                    }
                    let sample = Value(operations.len());
                    facts.push(KnownBits::UNKNOWN);
                    operations.push((
                        offset,
                        Op::Sample {
                            u,
                            v,
                            lod,
                            component,
                            explicit_lod,
                            exec,
                        },
                    ));
                    let slot = usize::from(next_dst);
                    if next_dst == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                        return Err(error(
                            "instruction writes the index register or replaces a pending load",
                        ));
                    }
                    registers[slot] = Some(merge_lanes(
                        &mut operations,
                        &mut facts,
                        offset,
                        exec,
                        sample,
                        registers[slot],
                    )?);
                    next_dst += 1;
                }
                continue;
            }
            Instruction::Export { .. } | Instruction::Interpolate { .. } => {
                return Err(error(
                    "graphics instruction is unsupported in a compute fixture",
                ));
            }
        };
        let value_bits = match operation {
            Op::Constant(bits) => KnownBits::constant(bits),
            Op::Binary(Binary::Vector(VectorBinary::And), left, right) => {
                facts[left.0].and(facts[right.0])
            }
            Op::Binary(Binary::Vector(VectorBinary::Or), left, right) => {
                facts[left.0].or(facts[right.0])
            }
            Op::Binary(Binary::Vector(VectorBinary::Xor), left, right) => {
                facts[left.0].xor(facts[right.0])
            }
            _ => KnownBits::UNKNOWN,
        };
        // Arms may push operand constants first, so the result is the next slot
        // now rather than `next`.
        let produced = Value(operations.len());
        facts.push(value_bits);
        operations.push((offset, operation));
        if let Some(dest) = dest {
            let slot = usize::from(dest);
            if dest == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                return Err(error(
                    "instruction writes the index register or replaces a pending load",
                ));
            }
            registers[slot] = Some(merge_lanes(
                &mut operations,
                &mut facts,
                offset,
                exec,
                produced,
                registers[slot],
            )?);
        }
    }
    Err(CompileError::Verify {
        offset: code_len,
        reason: "missing program termination",
    })
}

/// Reuse the first equivalent pure vector unary or binary.
///
/// Equivalence is the operation plus its operand values in order. Constant
/// operands match by bit pattern, so two temps folded to the same `const.u32`
/// compare equal. That match only selects reuse: an op that is kept keeps
/// its own constant operands, so a shader with no repeat compiles unchanged.
/// Loads, stores, the constant definitions themselves, lane masks, merges,
/// branches and other mask-dependent or side-effecting ops are never
/// candidates. A reused definition stays at its index as `reuse.u32`.
fn eliminate_common_subexpressions(operations: &mut [(usize, Op)]) {
    let mut expr_alias = vec![None; operations.len()];
    let mut const_bits: HashMap<u32, usize> = HashMap::new();
    let mut seen: Vec<(PureExpr, usize)> = Vec::new();

    for index in 0..operations.len() {
        let current = operations[index].1;
        if let Op::Constant(bits) = current {
            const_bits.entry(bits).or_insert(index);
            continue;
        }
        let canonical = match current {
            Op::Unary(operator, source) => Op::Unary(
                operator,
                equivalent_value(&expr_alias, &const_bits, operations, source),
            ),
            Op::Binary(operator, left, right) => Op::Binary(
                operator,
                equivalent_value(&expr_alias, &const_bits, operations, left),
                equivalent_value(&expr_alias, &const_bits, operations, right),
            ),
            other => other,
        };
        let rewritten = map_uses(current, |value| reused_value(&expr_alias, value));
        if let Some(key) = pure_expr(canonical) {
            if let Some(&(_, first)) = seen.iter().find(|(seen_key, _)| *seen_key == key) {
                expr_alias[index] = Some(first);
                operations[index].1 = Op::Reuse(Value(first));
                continue;
            }
            seen.push((key, index));
        }
        operations[index].1 = rewritten;
    }
}

fn reused_value(expr_alias: &[Option<usize>], value: Value) -> Value {
    match expr_alias.get(value.0).and_then(|slot| *slot) {
        Some(index) => Value(index),
        None => value,
    }
}

/// SSA value to use when matching a pure expression. Follows a reuse edge, then
/// the earliest constant with the same bits.
fn equivalent_value(
    expr_alias: &[Option<usize>],
    const_bits: &HashMap<u32, usize>,
    operations: &[(usize, Op)],
    value: Value,
) -> Value {
    let root = reused_value(expr_alias, value);
    match operations.get(root.0).map(|(_, operation)| *operation) {
        Some(Op::Constant(bits)) => const_bits.get(&bits).copied().map(Value).unwrap_or(root),
        _ => root,
    }
}

fn map_uses(operation: Op, map: impl Fn(Value) -> Value) -> Op {
    match operation {
        Op::Load | Op::Constant(_) | Op::ScalarBranch { .. } => operation,
        Op::ScalarLoad { word, old } => Op::ScalarLoad {
            word,
            old: map(old),
        },
        Op::Binary(operator, left, right) => Op::Binary(operator, map(left), map(right)),
        Op::Unary(operator, source) => Op::Unary(operator, map(source)),
        Op::MergeLanes {
            mask,
            active,
            inactive,
        } => Op::MergeLanes {
            mask,
            active: map(active),
            inactive: map(inactive),
        },
        Op::Store(value) => Op::Store(map(value)),
        Op::VccArithmetic {
            primitive,
            left,
            right,
            mask,
            flag,
        } => Op::VccArithmetic {
            primitive,
            left: map(left),
            right: map(right),
            mask: map(mask),
            flag,
        },
        Op::CarryMask { flag, exec } => Op::CarryMask {
            flag: map(flag),
            exec,
        },
        Op::LaneMask {
            op,
            name,
            sdst,
            src0,
            src1,
            high,
            exec,
        } => Op::LaneMask {
            op,
            name,
            sdst,
            src0: map(src0),
            src1: map(src1),
            high: high.map(|(left, right)| (map(left), map(right))),
            exec,
        },
        Op::RuntimeBranch { test, mask, target } => Op::RuntimeBranch {
            test,
            mask: map(mask),
            target,
        },
        Op::RuntimeElse { .. } => operation,
        Op::ScalarTest {
            relation,
            signed,
            left,
            right,
        } => Op::ScalarTest {
            relation,
            signed,
            left: map(left),
            right: map(right),
        },
        Op::JoinRuntime {
            test,
            mask,
            fallthrough,
            taken,
        } => Op::JoinRuntime {
            test,
            mask: map(mask),
            fallthrough: map(fallthrough),
            taken: map(taken),
        },
        Op::Reuse(source) => Op::Reuse(map(source)),
        Op::Sample {
            u,
            v,
            lod,
            component,
            explicit_lod,
            exec,
        } => Op::Sample {
            u: map(u),
            v: map(v),
            lod: map(lod),
            component,
            explicit_lod,
            exec,
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PureExpr {
    Unary(VectorUnary, usize),
    Binary(Binary, usize, usize),
}

fn pure_expr(operation: Op) -> Option<PureExpr> {
    match operation {
        Op::Unary(operator, source) => Some(PureExpr::Unary(operator, source.0)),
        Op::Binary(operator, left, right) => Some(PureExpr::Binary(operator, left.0, right.0)),
        _ => None,
    }
}

/// Replace pure constant expressions with `Op::Constant` in place.
///
/// Operands stay addressable by their original `Value` indices so later uses
/// need no rewrite. Side-effecting and mask-dependent ops are left unchanged.
fn fold_constant_ops(operations: &mut [(usize, Op)]) {
    for index in 0..operations.len() {
        let folded = match operations[index].1 {
            Op::Unary(operator, source) => {
                const_value(operations, source).and_then(|bits| eval_unary(operator, bits))
            }
            Op::Binary(operator, left, right) => {
                match (
                    const_value(operations, left),
                    const_value(operations, right),
                ) {
                    (Some(left), Some(right)) => eval_binary(operator, left, right),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some(bits) = folded {
            operations[index].1 = Op::Constant(bits);
        }
    }
}

fn const_value(operations: &[(usize, Op)], value: Value) -> Option<u32> {
    match operations.get(value.0).map(|(_, operation)| *operation)? {
        Op::Constant(bits) => Some(bits),
        _ => None,
    }
}

/// MSL for a binary32 rounding form, on the register bits only: no Metal
/// float operation runs, so fast math, subnormal flushing and NaN
/// canonicalization cannot change the result. A magnitude of at least 2^23
/// (including infinities and NaNs) is already integral and passes through;
/// below 1.0 the result is a signed zero or one; otherwise the `m` fraction
/// bits are cleared and, when rounding away from zero, one unit is added to the
/// magnitude (a carry into the exponent is the next binade).
fn float_rounding_expression(op: VectorUnary, src: &str) -> String {
    let magnitude = format!("({src} & 0x7fffffffu)");
    let sign = format!("({src} & 0x80000000u)");
    let fraction = format!("(0x007fffffu >> (((({src} >> 23u) & 0xffu) - 127u) & 31u))");
    let below_one = match op {
        VectorUnary::TruncateF32 => sign.clone(),
        VectorUnary::FloorF32 => format!("({src} > 0x80000000u ? 0xbf800000u : {sign})"),
        VectorUnary::CeilingF32 => {
            format!("(({src} != 0u && {src} < 0x80000000u) ? 0x3f800000u : {sign})")
        }
        VectorUnary::RoundNearestEvenF32 => {
            format!("({magnitude} > 0x3f000000u ? ({sign} | 0x3f800000u) : {sign})")
        }
        _ => unreachable!("not a binary32 rounding form"),
    };
    let whole = format!("({src} & ~{fraction})");
    if op == VectorUnary::TruncateF32 {
        return format!(
            "{magnitude} >= 0x4b000000u ? {src} : {magnitude} < 0x3f800000u ? {below_one} : {whole}"
        );
    }
    let frac = format!("({src} & {fraction})");
    let unit = format!("({fraction} + 1u)");
    let away = match op {
        VectorUnary::FloorF32 => format!("({frac} != 0u && {sign} != 0u)"),
        VectorUnary::CeilingF32 => format!("({frac} != 0u && {sign} == 0u)"),
        VectorUnary::RoundNearestEvenF32 => format!(
            "({frac} > ({unit} >> 1u) || ({frac} == ({unit} >> 1u) && ({src} & {unit}) != 0u))"
        ),
        _ => unreachable!("not a binary32 rounding form"),
    };
    format!(
        "{magnitude} >= 0x4b000000u ? {src} : {magnitude} < 0x3f800000u ? {below_one} : ({whole} + ({away} ? {unit} : 0u))"
    )
}

fn eval_unary(operator: VectorUnary, source: u32) -> Option<u32> {
    Some(match operator {
        // Not folded: constant sources still run the MSL above, so the
        // translator has one implementation of these forms.
        VectorUnary::TruncateF32
        | VectorUnary::CeilingF32
        | VectorUnary::RoundNearestEvenF32
        | VectorUnary::FloorF32 => return None,
        VectorUnary::Not => !source,
        VectorUnary::ReverseBits => source.reverse_bits(),
        VectorUnary::FindHighUnsigned => {
            if source == 0 {
                0xffff_ffff
            } else {
                source.leading_zeros()
            }
        }
        VectorUnary::FindLow => {
            if source == 0 {
                0xffff_ffff
            } else {
                source.trailing_zeros()
            }
        }
        VectorUnary::FindHighSigned => {
            if source == 0 || source == 0xffff_ffff {
                0xffff_ffff
            } else {
                (source ^ (0u32.wrapping_sub(source >> 31))).leading_zeros()
            }
        }
    })
}

fn eval_binary(operator: Binary, left: u32, right: u32) -> Option<u32> {
    match operator {
        // Binary32 results are not folded on the CPU: the MSL is the only
        // translator path for them.
        Binary::Vector(operation) if operation.is_float() => None,
        Binary::Vector(operation) => Some(eval_vector_binary(operation, left, right)),
    }
}

fn eval_vector_binary(operation: VectorBinary, left: u32, right: u32) -> u32 {
    let i24 = |value: u32| ((value as i32) << 8) >> 8;
    let u24 = |value: u32| value & 0x00ff_ffff;
    match operation {
        VectorBinary::Add => left.wrapping_add(right),
        VectorBinary::Sub => left.wrapping_sub(right),
        VectorBinary::Xnor => !(left ^ right),
        VectorBinary::MinSigned => (left as i32).min(right as i32) as u32,
        VectorBinary::MinUnsigned => left.min(right),
        VectorBinary::MaxSigned => (left as i32).max(right as i32) as u32,
        VectorBinary::MaxUnsigned => left.max(right),
        VectorBinary::Mul24Signed => (i24(left) as u32).wrapping_mul(i24(right) as u32),
        VectorBinary::Mul24Unsigned => u24(left).wrapping_mul(u24(right)),
        VectorBinary::MulHi24Signed => {
            ((i64::from(i24(left)) * i64::from(i24(right))) >> 32) as u32
        }
        VectorBinary::MulHi24Unsigned => {
            ((u64::from(u24(left)) * u64::from(u24(right))) >> 32) as u32
        }
        VectorBinary::And => left & right,
        VectorBinary::Or => left | right,
        VectorBinary::Xor => left ^ right,
        VectorBinary::ShiftLeft => left << (right & 31),
        VectorBinary::ShiftRight => left >> (right & 31),
        VectorBinary::ArithmeticShiftRight => ((left as i32) >> (right & 31)) as u32,
        VectorBinary::AddF32
        | VectorBinary::SubF32
        | VectorBinary::MulF32
        | VectorBinary::MinF32
        | VectorBinary::MaxF32 => unreachable!("binary32 operations are not folded"),
    }
}

/// A wave-uniform constant operand at `offset`.
fn constant(
    operations: &mut Vec<(usize, Op)>,
    facts: &mut Vec<KnownBits>,
    offset: usize,
    bits: u32,
) -> Value {
    let value = Value(operations.len());
    operations.push((offset, Op::Constant(bits)));
    facts.push(KnownBits::constant(bits));
    value
}

// Keeping a value for inactive lanes requires a previously initialized destination.
fn merge_lanes(
    operations: &mut Vec<(usize, Op)>,
    facts: &mut Vec<KnownBits>,
    offset: usize,
    mask: u32,
    active: Value,
    inactive: Option<Value>,
) -> Result<Value, CompileError> {
    if mask == u32::MAX {
        return Ok(active);
    }
    let inactive = inactive.ok_or(CompileError::Verify {
        offset,
        reason: "partial EXEC requires an initialized vector destination",
    })?;
    let next = Value(operations.len());
    operations.push((
        offset,
        Op::MergeLanes {
            mask,
            active,
            inactive,
        },
    ));
    facts.push(KnownBits {
        zero: facts[active.0].zero & facts[inactive.0].zero,
        one: facts[active.0].one & facts[inactive.0].one,
        values: None,
    });
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::{
        KnownBits, Op, Value, VectorBinary, VectorUnary, eval_unary, eval_vector_binary,
        vector_expression,
    };

    /// Every foldable vector op must keep a matching MSL template and eval arm
    /// so the fold table cannot drift from emission. Match arms in
    /// `vector_expression` / `eval_vector_binary` / unary `expression` are
    /// exhaustive; this test calls both sides for each variant.
    #[test]
    fn every_foldable_vector_op_has_matching_msl_template() {
        for operation in [
            VectorBinary::Add,
            VectorBinary::Sub,
            VectorBinary::Xnor,
            VectorBinary::MinSigned,
            VectorBinary::MinUnsigned,
            VectorBinary::MaxSigned,
            VectorBinary::MaxUnsigned,
            VectorBinary::Mul24Signed,
            VectorBinary::Mul24Unsigned,
            VectorBinary::MulHi24Signed,
            VectorBinary::MulHi24Unsigned,
            VectorBinary::And,
            VectorBinary::Or,
            VectorBinary::Xor,
            VectorBinary::ShiftLeft,
            VectorBinary::ShiftRight,
            VectorBinary::ArithmeticShiftRight,
        ] {
            let expression = vector_expression(operation, "left", "right");
            assert!(
                !expression.is_empty(),
                "missing MSL template for {operation:?}"
            );
            let _ = eval_vector_binary(operation, 0x11, 0x22);
        }
        for operation in [
            VectorUnary::Not,
            VectorUnary::ReverseBits,
            VectorUnary::FindHighUnsigned,
            VectorUnary::FindLow,
            VectorUnary::FindHighSigned,
        ] {
            let expression = Op::Unary(operation, Value(0)).expression(&|v| v.to_string(), "", "");
            assert!(
                !expression.is_empty(),
                "missing MSL template for {operation:?}"
            );
            assert!(eval_unary(operation, 0x8000_0001).is_some());
        }
        for operation in [
            VectorUnary::TruncateF32,
            VectorUnary::CeilingF32,
            VectorUnary::RoundNearestEvenF32,
            VectorUnary::FloorF32,
        ] {
            let expression = Op::Unary(operation, Value(0)).expression(&|v| v.to_string(), "", "");
            assert!(!expression.contains("as_type"), "float op in {operation:?}");
            assert!(eval_unary(operation, 0x3fc0_0000).is_none());
        }
    }

    #[test]
    fn propagates_known_bits_through_integer_operators() {
        let left = KnownBits::constant(0b1100);
        let right = KnownBits::constant(0b1010);
        let and = left.and(right);
        assert_eq!((and.zero, and.one), (!0b1000, 0b1000));
        let xor = left.xor(right);
        assert_eq!((xor.zero, xor.one), (!0b0110, 0b0110));
    }
}
