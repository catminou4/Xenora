//! Typed vocabulary of the declarative semantic registry (`crates/isa/semantics/rdna2.toml`).
//!
//! The registry is data. `crates/isa/generate_semantics.py` validates it against
//! this closed vocabulary and the pinned AMD XML snapshot, then emits
//! `semantics_gen.rs`: the typed operation enums, their opcode dispatch and one
//! static [`FormSemantics`] row per form. Nothing here interprets metadata at run
//! time; the decoder and translator call the generated `const fn`s and `match`es.
//!
//! The reference interpreter (`crates/ref`) and the independent interpreter
//! (`tests/validation/independent_ref.rs`) must not use this module.

use crate::CompareRelation;
use crate::meta::EncodingFamily;

pub use crate::semantics_gen::{FORM_IDS, FORMS, GENERATED_SCHEMA};

/// Version of the registry schema this crate was generated against.
pub const SCHEMA: u32 = 1;

/// Reusable semantic primitive selected by a form. Closed: the generator rejects
/// any other name, and each primitive has one implementation in `primitive.rs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    /// Integer addition.
    Add,
    /// Integer subtraction.
    Sub,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise exclusive OR.
    Xor,
    /// Bitwise AND with the complement of S1.
    AndNot,
    /// Bitwise OR with the complement of S1.
    OrNot,
    /// Bitwise AND with the complement of S0.
    AndNot0,
    /// Bitwise OR with the complement of S0.
    OrNot0,
    /// Complement of bitwise AND.
    Nand,
    /// Complement of bitwise OR.
    Nor,
    /// Complement of bitwise exclusive OR.
    Xnor,
    /// Shift S0 left by the masked count in S1, filling with zeros.
    ShiftLeft,
    /// Shift S0 right by the masked count in S1, filling with zeros.
    LogicalShiftRight,
    /// Shift S0 right by the masked count in S1, filling with its sign bit.
    ArithmeticShiftRight,
    /// Compare S0 with S1 under the form's relation, producing SCC or a lane mask.
    Compare,
    /// Bitwise complement of S0.
    Not,
    /// S0 with its bit order reversed.
    ReverseBits,
    /// Number of zero bits in S0.
    CountZeros,
    /// Number of one bits in S0.
    CountOnes,
    /// Index of the lowest zero bit of S0, or -1 when there is none.
    FindFirstZero,
    /// Index of the lowest one bit of S0, or -1 when there is none.
    FindFirstOne,
    /// Unsigned: count of zeros above the highest one bit; signed: count of bits
    /// above the highest bit that differs from the sign bit. -1 when none.
    FindLastBit,
    /// Sign-extend the low `source_bits` of S0 to 32 bits.
    SignExtend,
    /// Two's-complement absolute value; the most negative value maps to itself.
    Absolute,
    /// S0 when it is strictly less than S1, otherwise S1.
    Minimum,
    /// S0 when it is strictly greater than S1, otherwise S1.
    Maximum,
    /// S0 when SCC is set, otherwise S1.
    Select,
    /// `((1 << S0[4:0]) - 1) << S1[4:0]`.
    BitfieldMask,
    /// Low 32 bits of the product.
    Multiply,
    /// High 32 bits of the 64-bit product.
    MultiplyHigh,
    /// The field of S0 at offset S1[4:0] with width S1[22:16]; sign-extended when signed.
    BitfieldExtract,
    /// Absolute value of the wrapped 32-bit difference S0 - S1.
    AbsoluteDifference,
    /// `(S0 << shift_amount) + S1`.
    ShiftLeftAdd,
    /// Low 32 bits of the product of S0[23:0] and S1[23:0] (sign-extended when signed).
    Multiply24,
    /// Bits 63:32 of the product of S0[23:0] and S1[23:0] (sign-extended when signed).
    MultiplyHigh24,
    /// SOPK: SDST receives the sign-extended SIMM16.
    Move,
    /// SOPK: SDST receives the sign-extended SIMM16 when SCC is set.
    ConditionalMove,
    /// Terminate the program.
    EndProgram,
    /// Unconditional PC-relative branch.
    Branch,
    /// Branch when SCC is clear.
    BranchSccZero,
    /// Branch when VCC is zero.
    BranchVccZero,
    /// Branch when EXEC is zero.
    BranchExecZero,
    /// Complete an outstanding vector load.
    WaitLoad,
    /// Swap the two old per-lane VGPR values (guide 12.8).
    Swap,
    /// Binary32 integral part, rounding toward zero (guide 12.8).
    Truncate,
    /// Binary32 rounding up to an integral value (guide 12.8).
    Ceiling,
    /// Binary32 rounding to the nearest integral value, ties to even (guide 12.8).
    RoundNearestEven,
    /// Binary32 rounding down to an integral value (guide 12.8).
    Floor,
    /// MIMG `image_sample_lz`: sample a 2D texture at level of detail zero.
    ImageSampleLz,
    /// MIMG `image_sample_l`: sample a 2D texture at an explicit level of
    /// detail read from the address tuple.
    ImageSampleL,
}

/// Per-lane operation with one input and one result: integer and bit forms,
/// and binary32 rounding to an integral value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorUnary {
    /// Bitwise complement.
    Not,
    /// Reverse bit order.
    ReverseBits,
    /// Count leading zeros, or return all ones for zero.
    FindHighUnsigned,
    /// Count trailing zeros, or return all ones for zero.
    FindLow,
    /// Count leading sign bits, or return all ones if every bit matches.
    FindHighSigned,
    /// Binary32 rounding toward zero.
    TruncateF32,
    /// Binary32 rounding toward +infinity.
    CeilingF32,
    /// Binary32 rounding to nearest, ties to even.
    RoundNearestEvenF32,
    /// Binary32 rounding toward -infinity.
    FloorF32,
}

/// The per-lane integer operation a vector form lowers to. One reusable MSL
/// template exists per variant; a form selects it through its declared
/// operation and signedness, never through its opcode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorBinary {
    /// Wrapping `a + b`.
    Add,
    /// Wrapping `a - b`.
    Sub,
    /// `!(a ^ b)`.
    Xnor,
    /// Two's-complement minimum.
    MinSigned,
    /// Unsigned minimum.
    MinUnsigned,
    /// Two's-complement maximum.
    MaxSigned,
    /// Unsigned maximum.
    MaxUnsigned,
    /// Low 32 bits of `sext(a[23:0]) * sext(b[23:0])`.
    Mul24Signed,
    /// Low 32 bits of `a[23:0] * b[23:0]`.
    Mul24Unsigned,
    /// Bits 63:32 of `sext(a[23:0]) * sext(b[23:0])`.
    MulHi24Signed,
    /// Bits 63:32 of `a[23:0] * b[23:0]`.
    MulHi24Unsigned,
    /// `a & b`.
    And,
    /// `a | b`.
    Or,
    /// `a ^ b`.
    Xor,
    /// `a << (b & 31)`.
    ShiftLeft,
    /// Logical `a >> (b & 31)`.
    ShiftRight,
    /// Arithmetic (sign-filling) `a >> (b & 31)`.
    ArithmeticShiftRight,
    /// Binary32 `a + b`, round to nearest even (guide 12.7).
    AddF32,
    /// Binary32 `a - b`, round to nearest even (guide 12.7).
    SubF32,
    /// Binary32 `a * b`, round to nearest even (guide 12.7).
    MulF32,
    /// Binary32 minimum by the guide 12.7 V_MIN_F32 rule: a quiet NaN gives
    /// the other operand and a signed-zero pair gives -0.
    MinF32,
    /// Binary32 maximum by the guide 12.7 V_MAX_F32 rule: a quiet NaN gives
    /// the other operand and a signed-zero pair gives +0.
    MaxF32,
}

impl VectorBinary {
    /// The binary32 operations, which read float operands.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::AddF32 | Self::SubF32 | Self::MulF32 | Self::MinF32 | Self::MaxF32
        )
    }
}

/// How the operands are interpreted. `Bitwise` exactly when the operation is logical.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Signedness {
    /// Unsigned integers.
    Unsigned,
    /// Two's-complement integers.
    Signed,
    /// Untyped bits.
    Bitwise,
    /// IEEE 754 binary32 values.
    Float,
}

/// Per-lane VOP2 operation selected by the semantic registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorLowering {
    /// Two explicit operands, with no implicit lane input.
    Binary(VectorBinary),
    /// Two explicit operands and the lane's VCC bit.
    Vcc(VectorVcc),
}

/// VOP2 primitives with an implicit per-lane VCC input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorVcc {
    /// Choose S1 when the VCC bit is set, otherwise S0.
    Select,
    /// Unsigned addition with carry-in and carry-out.
    AddCarry,
    /// Unsigned subtraction with borrow-in and borrow-out.
    SubtractBorrow,
}

/// Implicit condition or carry input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CarryIn {
    /// No implicit condition or carry input.
    None,
    /// SCC as carry-in (add), borrow-in (sub) or the select condition (select).
    Scc,
    /// Per-lane Wave32 VCC_LO bit.
    Vcc,
}

/// What a form writes to SCC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccEffect {
    /// SCC is not written.
    Unchanged,
    /// Unsigned carry-out of the top bit.
    CarryOut,
    /// Unsigned borrow (subtrahend plus borrow-in exceeds the minuend).
    Borrow,
    /// Two's-complement overflow of the exact result.
    SignedOverflow,
    /// The result is not zero.
    Nonzero,
    /// The compare's relation holds for S0 and S1.
    Relation,
    /// The result is S0 by the strict comparison (min, max): equal operands give 0.
    SelectedSrc0,
}

/// Where a form writes its lane mask, if any.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskEffect {
    /// Not written.
    Unchanged,
    /// Wave32 VOPC result (VCC only): `VCC_LO` bit n is EXEC[n] & (lane n
    /// exists) & test(lane n), fully written; `VCC_HI` has no known value after
    /// it (docs/shaders/vector-compare.md section 3).
    LaneMask,
    /// Wave32 VOP3A compare mask written to SDST, which may be `vcc_lo`.
    SdstLaneMask,
    /// The form combines its source with EXEC_LO, writes the result to EXEC_LO
    /// and leaves the previous EXEC_LO in the destination. Guide 12.3, printed
    /// pages 120-122.
    ExecSave,
    /// The form combines its source with EXEC_LO and writes the result to both
    /// EXEC_LO and the destination. Guide 12.3, printed page 122.
    ExecWrite,
}

/// Wave-size restriction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wave {
    /// Behaviour does not depend on the wave size.
    Any,
    /// Only Wave32 behaviour is declared; Wave64 stays unsupported.
    Wave32,
}

/// Shared operand rule. The decoder implements each rule once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperandRule {
    /// SOP2: SDST is s0-s105; SSRC0/SSRC1 are s0-s105, inline integers -16..64
    /// or one literal word. Everything else is rejected as unsupported.
    Sop2Sgpr,
    /// SOPC: no destination; SSRC0/SSRC1 as for SOP2 sources (s0-s105, inline
    /// integers -16..64 or one literal word). Everything else is rejected.
    SopcScalar,
    /// SOP1 unary: SDST is s0-s105; SSRC0 is s0-s105, an inline integer -16..64 or
    /// a literal word. Everything else is rejected as unsupported.
    Sop1Sgpr,
    /// SOP1 move: the same sources as Sop1Sgpr, with VCC and EXEC halves
    /// also allowed as explicit destinations.
    Sop1Move,
    /// SOP1 EXEC-writing: the same sources and destination as Sop1Sgpr, with the
    /// form also writing EXEC_LO and SCC.
    Sop1SaveExec,
    /// SOPK integer: SDST is s0-s105 and, except for compares, written (never s0-s7,
    /// which hold resource descriptors); SIMM16 is the 16-bit immediate. The word
    /// never takes a literal. Everything else is rejected as unsupported.
    SopkSgpr,
    /// SOPK integer compare: SDST (s0-s105) is the left operand, SIMM16 the right
    /// (sign-extended for `_i32`, zero-extended for `_u32`); only SCC is written.
    SopkCompare,
    /// SOPP termination with zero in the reserved immediate field.
    SoppEnd,
    /// SOPP branch with a signed dword displacement from the next instruction.
    SoppBranch,
    /// SOPP wait with exactly vmcnt(0) and the other counters disabled.
    SoppWait,
    /// VOPC vector compare: the destination is `vcc_lo` (Wave32); SRC0 is a VGPR,
    /// s0-s105, `vcc_lo`, an inline integer -16..64 or a literal word; VSRC1 is a
    /// VGPR. Everything else is rejected as unsupported.
    VopcVcc,
    /// Wave32 VOPC with a VGPR pair, even normal SGPR pair or inline integer
    /// as SRC0 and a VGPR pair as VSRC1. The destination is `vcc_lo`.
    VopcVccPair,
    /// VOPC with low-half operands and no literal selector.
    VopcVccHalf,
    /// VOP3 encoding of a VOP2 vector form: VDST is v0-v255, SRC0 and SRC1 are
    /// VGPRs, no input or output modifiers, SRC2 unused. Everything else is
    /// rejected as unsupported.
    Vop3Vgpr2,
    /// VOP3A compare: VDST encodes s0-s105 or `vcc_lo`; SRC0 and SRC1 are
    /// VGPRs, SRC2 is unused and all modifiers are clear.
    Vop3CmpVgpr2,
    /// VOP2 vector: VDST is v0-v255; SRC0 is a VGPR, s0-s105, `vcc_lo`, an inline
    /// integer -16..64 or a literal word; VSRC1 is a VGPR. Everything else is
    /// rejected as unsupported.
    Vop2Vgpr,
    /// VOP2 general SRC0 and VGPR VSRC1 with an implicit VCC_LO input.
    Vop2Vcc,
    /// VOP2 vector, VGPR sources only: VDST is v0-v255; SRC0 and VSRC1 are VGPRs.
    /// Everything else (SGPR, inline, literal SRC0) is rejected as unsupported.
    Vop2Vgpr2,
    /// VOP1 unary: VGPR VDST and the same SRC0 domain as VOP2.
    Vop1Vgpr,
    /// VOP1 binary32 rounding: as `Vop1Vgpr` without integer inline constants.
    Vop1F32,
    /// VOP2 binary32 arithmetic: as `Vop2Vgpr` without integer inline constants.
    Vop2F32,
    /// VOP1 swap: VDST and SRC0 each read and write a VGPR.
    Vop1Swap,
    /// MIMG 2D sample (IMAGE_SAMPLE_LZ and IMAGE_SAMPLE_L): VDATA is the first
    /// destination VGPR, VADDR the first address VGPR, SRSRC the base of an
    /// eight-aligned SGPR resource descriptor and SSAMP the base of a
    /// four-aligned SGPR sampler descriptor. Non-NSA, DIM 2D, R128 clear,
    /// DMASK 1-15; everything else is rejected as unsupported.
    Mimg2D,
}

/// Evidence a form needs before it counts as verified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Evidence {
    /// `crates/ref` lockstep reference.
    Reference,
    /// `tests/validation/independent_ref.rs`.
    Independent,
    /// Native Metal known-answer runs.
    Metal,
    /// LLVM assembler/disassembler encoding agreement.
    LlvmEncoding,
}

/// Public document a [`SpecRef`] points at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Document {
    /// AMD RDNA 2 ISA reference guide (PDF).
    Guide,
    /// Pinned AMD machine-readable XML.
    Xml,
}

/// One public specification reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecRef {
    /// Source document.
    pub document: Document,
    /// Section number, for example `"12.1"`.
    pub section: &'static str,
    /// Page in the document, or 0 when the page has not been recorded.
    pub page: u16,
}

/// One validated registry row, emitted as a static table by the generator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormSemantics {
    /// XML instruction name, the stable form identity.
    pub name: &'static str,
    /// Encoding family (matches [`crate::meta::FORMS`]).
    pub family: EncodingFamily,
    /// Opcode within the family.
    pub opcode: u16,
    /// Primitive.
    pub operation: Operation,
    /// Operand interpretation.
    pub signedness: Signedness,
    /// SCC input.
    pub carry_in: CarryIn,
    /// Operand and result width in bits.
    pub width: u8,
    /// Low bits of S1 used as the shift count by shift operations; 0 otherwise.
    pub shift_count_bits: u8,
    /// Fixed left shift of S0 for shift-and-add operations (1 to 4); 0 otherwise.
    pub shift_amount: u8,
    /// Whether the operation reads its sources in reverse order (`D = op(S1, S0)`).
    pub swap_sources: bool,
    /// Low bits of S0 that the operation reads: `width`, or 8 or 16 for sign extension.
    pub source_bits: u8,
    /// The relation tested by compares (S0 on the left); `None` otherwise.
    pub relation: Option<CompareRelation>,
    /// SCC write.
    pub scc: SccEffect,
    /// Implicit VCC write or explicit scalar destination for a compare mask.
    pub vcc: MaskEffect,
    /// EXEC write.
    pub exec: MaskEffect,
    /// Wave-size restriction.
    pub wave: Wave,
    /// Operand rule.
    pub operands: OperandRule,
    /// Public references.
    pub spec: &'static [SpecRef],
    /// Required evidence.
    pub verify: &'static [Evidence],
}
