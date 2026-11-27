// SPDX-License-Identifier: GPL-3.0-or-later

//! Independent CPU execution of the synthetic gfx1030 fixture subset, reference sampling, and host-side references for depth-tested rendering, indexed draws and blending.

pub mod blend;
pub mod depth;
pub mod indexed;
pub mod texture_sampling;

use std::fmt;

const MAX_CODE_BYTES: usize = 1024;
const MAX_INPUT_WORDS: usize = 4096;

#[derive(Debug, PartialEq, Eq)]
pub struct ReferenceError {
    pub offset: usize,
    pub detail: &'static str,
}

impl fmt::Display for ReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reference byte {:#x}: {}", self.offset, self.detail)
    }
}

impl std::error::Error for ReferenceError {}

/// Register index or immediate bits. The opcode determines the register bank.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    Register(u8),
    Immediate(u32),
}

/// Operations decoded independently by the reference interpreter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    Load {
        data: u8,
        index: u8,
        resource: u8,
    },
    Store {
        data: u8,
        index: u8,
        resource: u8,
    },
    WaitRead,
    WaitWrite,
    /// SMEM `S_BUFFER_LOAD_DWORD`, `_X2` and `_X4` into consecutive SGPRs.
    ScalarLoad {
        data: u8,
        resource: u8,
        offset: u32,
        register: Option<u8>,
        dwords: u8,
    },
    WaitScalar,
    Move(u8, Source),
    Add(u8, u8, u8),
    ScalarMove(u8, Source),
    /// SOP1 unary operation writing an SGPR (s0-s105), and SCC for some.
    ScalarUnary(ScalarUnary, u8, Source),
    /// VOP1 move of a normal SGPR or `vcc_lo` (106) into a VGPR.
    MoveScalar(u8, u8),
    /// VOPC 32-bit integer compare writing the Wave32 mask to `vcc_lo`.
    Compare(Comparison, CompareSource, u8),
    /// VOP3A integer compare writing a Wave32 mask to SDST.
    CompareVop3(u8, Comparison, CompareSource, u8),
    /// VOPC 16- or 64-bit integer compare writing the Wave32 mask to
    /// `vcc_lo`. For 64-bit forms a `Vector` or `Scalar` source and VSRC1 name
    /// the low register of a pair.
    WideCompare(Comparison, CompareWidth, CompareSource, u8),
    /// SOP2 operation writing an SGPR and SCC.
    Scalar(ScalarAlu, u8, Source, Source),
    /// SOPC `s_cmp_eq_{i32,u32}` (opcodes 0 and 6, identical per guide 12.4).
    CompareEqual(Source, Source),
    /// The other SOPC 32-bit integer compares (opcodes 1-5 and 7-11), writing
    /// only SCC: LG, GT, GE, LT, LE, signed for 1-5, unsigned for 7-11.
    ScalarCompare(Comparison, Source, Source),
    /// SOPK operation on SGPR `SDST` (s0-s105) and the raw SIMM16.
    ScalarImmediate(ScalarImmediate, u8, u16),
    Branch {
        target: usize,
        condition: BranchCondition,
    },
    Vector(VectorAlu, u8, u8, u8),
    /// VOP2 32-bit integer form: VDST, SRC0 (VGPR, SGPR, `vcc_lo`, inline
    /// integer or literal) and VSRC1, with implicit VCC for select/carry forms.
    VectorInteger(VectorInteger, u8, CompareSource, u8),
    /// VOP2 binary32 form: VDST, SRC0 (VGPR, SGPR, `vcc_lo` or literal; no
    /// integer inline constant) and VSRC1. Writes neither VCC nor SCC.
    VectorFloat(VectorFloat, u8, CompareSource, u8),
    /// VOP1 32-bit integer or bit form: VDST and SRC0 (VGPR, SGPR, `vcc_lo`,
    /// inline integer or literal). Writes neither VCC nor SCC.
    VectorUnary(VectorUnary, u8, CompareSource),
    /// VOP1 `v_swap_b32` (opcode 101): exchanges VDST and the SRC0 VGPR.
    Swap(u8, u8),
    /// MIMG 2D image sample (IMAGE_SAMPLE_LZ or IMAGE_SAMPLE_L) writing one
    /// VGPR per DMASK-selected component; `explicit_lod` reads the LOD from
    /// `coord + 2` instead of level zero.
    ImageSample {
        dst: u8,
        coord: u8,
        resource: u8,
        sampler: u8,
        dmask: u8,
        explicit_lod: bool,
    },
    End,
}

/// Relation and operand interpretation of a VOPC integer compare.
///
/// AMD RDNA 2 ISA guide section 12.9, Tables 59 and 61: opcode 128 + offset
/// for signed and 192 + offset for unsigned 32-bit operands, with offsets
/// F 0, LT 1, EQ 2, LE 3, GT 4, NE 5, GE 6, T 7.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Comparison {
    /// Opcode offset 0-7 within the 32-bit integer group.
    pub offset: u8,
    /// Whether the operands are two's-complement integers.
    pub signed: bool,
}

impl Comparison {
    /// The per-lane result for 32-bit operands.
    fn holds(self, s0: u32, s1: u32) -> bool {
        self.holds_at(CompareWidth::Word, u64::from(s0), u64::from(s1))
    }

    /// The per-lane result, from an exact comparison of the operands widened
    /// from their low `width` bits; higher bits are ignored.
    fn holds_at(self, width: CompareWidth, s0: u64, s1: u64) -> bool {
        let widen = |bits: u64| -> i128 {
            match (width, self.signed) {
                (CompareWidth::Half, true) => i128::from(bits as u16 as i16),
                (CompareWidth::Half, false) => i128::from(bits as u16),
                (CompareWidth::Word, true) => i128::from(bits as u32 as i32),
                (CompareWidth::Word, false) => i128::from(bits as u32),
                (CompareWidth::Double, true) => i128::from(bits as i64),
                (CompareWidth::Double, false) => i128::from(bits),
            }
        };
        let (s0, s1) = (widen(s0), widen(s1));
        match self.offset {
            // Table 61: V_CMP_F_{I,U}{32,64} `D[threadId] = 0`.
            0 => false,
            1 => s0 < s1,
            2 => s0 == s1,
            3 => s0 <= s1,
            4 => s0 > s1,
            5 => s0 != s1,
            6 => s0 >= s1,
            // Table 61: V_CMP_T_{I,U}{32,64} `D[threadId] = 1`.
            _ => true,
        }
    }
}

/// Operand width of a VOPC integer compare (guide 6.2: the data size is the
/// one named by the instruction).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareWidth {
    /// `_I16`/`_U16`: bits 15:0 of each 32-bit source (6.2.5 and 6.8: VOPC
    /// selects another half only through SDWA, which is not accepted).
    Half,
    /// `_I32`/`_U32`.
    Word,
    /// `_I64`/`_U64`: register pairs with the low dword in the lower register
    /// (3.6.3 for SGPRs; for VGPRs the guide states it for 64-bit addresses,
    /// 9.1: "ADDR has the LSBs, and ADDR+1 has the MSBs").
    Double,
}

impl CompareWidth {
    /// Guide 12.9 Tables 59-61: the base opcode of each accepted group plus
    /// the relation offset; 16-bit groups have no F (offset 0) or T (7).
    fn from_opcode(opcode: u16) -> Option<(Comparison, Self)> {
        let (base, signed, width) = match opcode {
            128..=135 => (128, true, Self::Word),
            192..=199 => (192, false, Self::Word),
            137..=142 => (136, true, Self::Half),
            169..=174 => (168, false, Self::Half),
            160..=167 => (160, true, Self::Double),
            224..=231 => (224, false, Self::Double),
            _ => return None,
        };
        let offset = (opcode - base) as u8;
        Some((Comparison { offset, signed }, width))
    }
}

/// First source of a VOPC compare or a VOP2 integer form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareSource {
    /// VGPR index.
    Vector(u8),
    /// SGPR index s0-s105, or 106 for `vcc_lo`.
    Scalar(u8),
    /// Inline integer or literal bits.
    Immediate(u32),
}

/// Scalar condition tested by a forward branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchCondition {
    /// Take the branch unconditionally.
    Always,
    /// Take the branch when SCC is clear.
    SccZero,
    /// Take the branch when the low half of VCC is clear.
    VccZero,
    /// Take the branch when the low half of VCC is nonzero.
    VccNonzero,
    /// Take the branch when the low half of EXEC is clear.
    ExecZero,
}

/// SOP2 operations decoded by the reference interpreter.
///
/// AMD RDNA 2 XML descriptions: additions store the carry-out in SCC,
/// subtractions the borrow; ADDC/SUBB add or subtract SCC; bitwise
/// operations and shifts set SCC iff the result is nonzero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarAlu {
    Add,
    Subtract,
    AddWithCarry,
    SubtractWithBorrow,
    /// SCC = signed overflow (RDNA 2 ISA guide 12.1).
    AddSigned,
    SubtractSigned,
    And,
    Or,
    Xor,
    /// RDNA 2 ISA guide 12.1: D = S0 & ~S1, SCC = (D != 0).
    AndNot,
    /// D = S0 | ~S1, SCC = (D != 0).
    OrNot,
    /// D = ~(S0 & S1), SCC = (D != 0).
    Nand,
    /// D = ~(S0 | S1), SCC = (D != 0).
    Nor,
    /// D = ~(S0 ^ S1), SCC = (D != 0).
    Xnor,
    /// D = S0 << S1[4:0], SCC = (D != 0).
    ShiftLeft,
    /// D = S0 >> S1[4:0] (logical), SCC = (D != 0).
    ShiftRight,
    /// D = signext(S0) >> S1[4:0], SCC = (D != 0).
    ArithmeticShiftRight,
    /// Guide 12.1: D = (S0 < S1) ? S0 : S1, SCC = (S0 < S1); signed or not.
    Minimum {
        signed: bool,
    },
    /// D = (S0 > S1) ? S0 : S1, SCC = (S0 > S1).
    Maximum {
        signed: bool,
    },
    /// D = SCC ? S0 : S1; reads SCC and leaves it unchanged.
    Select,
    /// D = ((1 << S0[4:0]) - 1) << S1[4:0]; SCC unchanged.
    BitfieldMask,
    /// D = low 32 bits of S0 * S1; SCC unchanged.
    Multiply,
    /// D = (S0 >> S1[4:0]) & ((1 << S1[22:16]) - 1), sign-extended from the
    /// field's top bit when signed (S0 then shifts arithmetically);
    /// SCC = (D != 0).
    BitfieldExtract {
        signed: bool,
    },
    /// D = |S0 - S1| with the 32-bit difference (0x80000000 stays),
    /// SCC = (D != 0).
    AbsoluteDifference,
    /// D = (S0 << N) + S1, SCC = ((S0 << N) + S1 >= 2^32) with N = 1..4.
    ShiftLeftAdd(u8),
    /// D = high 32 bits of S0 * S1; SCC unchanged.
    MultiplyHigh {
        signed: bool,
    },
}

/// SOP1 unary operations decoded by the reference interpreter (RDNA 2 ISA
/// guide 12.3; SCC per Tables 12 and 14).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarUnary {
    /// D = ~S0, SCC = (D != 0).
    Not,
    /// D[31:0] = S0[0:31].
    ReverseBits,
    /// D = number of zero bits, SCC = (D != 0).
    CountZeros,
    /// D = number of one bits, SCC = (D != 0).
    CountOnes,
    /// D = index of the lowest zero bit, or -1.
    FindFirstZero,
    /// D = index of the lowest one bit, or -1.
    FindFirstOne,
    /// D = number of zeros above the highest one bit, or -1 for 0.
    FindLastBit,
    /// D = number of leading bits equal to the sign bit, or -1 for 0 and -1.
    FindLastSignBit,
    /// D = signext(S0[7:0]).
    SignExtendByte,
    /// D = signext(S0[15:0]).
    SignExtendShort,
    /// D = |S0| (0x80000000 stays), SCC = (D != 0).
    Absolute,
    /// 60 `s_and_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = S0 & EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 120).
    AndSaveExec,
    /// 61 `s_or_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = S0 | EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    OrSaveExec,
    /// 62 `s_xor_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = S0 ^ EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    XorSaveExec,
    /// 63 `s_andn2_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = S0 & ~EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    Andn2SaveExec,
    /// 64 `s_orn2_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = S0 | ~EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    Orn2SaveExec,
    /// 65 `s_nand_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = ~(S0 & EXEC_LO), SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    NandSaveExec,
    /// 66 `s_nor_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = ~(S0 | EXEC_LO), SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    NorSaveExec,
    /// 67 `s_xnor_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = ~(S0 ^ EXEC_LO), SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 121).
    XnorSaveExec,
    /// 68 `s_andn1_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = ~S0 & EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 122).
    Andn1SaveExec,
    /// 69 `s_orn1_saveexec_b32`: D = the previous EXEC_LO, then
    /// EXEC_LO = ~S0 | EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 122).
    Orn1SaveExec,
    /// 70 `s_andn1_wrexec_b32`: EXEC_LO = ~S0 & EXEC_LO, then
    /// D = the new EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 122).
    Andn1Wrexec,
    /// 71 `s_andn2_wrexec_b32`: EXEC_LO = S0 & ~EXEC_LO, then
    /// D = the new EXEC_LO, SCC = (EXEC_LO != 0)
    /// (RDNA 2 ISA guide 12.3, printed page 122).
    Andn2Wrexec,
}

/// SOPK operations decoded by the reference interpreter (RDNA 2 ISA guide
/// 12.2; SCC per Tables 11, 12, 13 and 14).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarImmediate {
    /// 0 `s_movk_i32`: D = signext(SIMM16).
    Move,
    /// 2 `s_cmovk_i32`: if SCC, D = signext(SIMM16); SCC unchanged.
    ConditionalMove,
    /// 3-14 `s_cmpk_*`: SCC = (D <relation> K), K = signext(SIMM16) for the
    /// signed forms 3-8 and SIMM16 zero-extended for the unsigned 9-14.
    Compare(Comparison),
    /// 15 `s_addk_i32`: D = D + signext(SIMM16), SCC = signed overflow.
    Add,
    /// 16 `s_mulk_i32`: D = low 32 bits of D * signext(SIMM16); SCC unchanged.
    Multiply,
}

/// VOP2 operations decoded by the reference interpreter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorAlu {
    ShiftLeft,
    ShiftRight,
    ArithmeticShiftRight,
    And,
    Or,
    Xor,
}

/// VOP2 32-bit integer forms with a general SRC0 (AMD RDNA 2 ISA guide
/// section 12.7). None has a carry-in or carry-out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorInteger {
    /// 1: select S1 when the lane's VCC bit is set.
    Select,
    /// 40: unsigned add with carry input and output.
    AddCarry,
    /// 41 / 42: unsigned subtract with borrow input and output.
    SubtractBorrow { reversed: bool },
    /// 9 / 11: D = low 32 bits of S0[23:0] * S1[23:0], sign-extended from
    /// bit 23 when `signed`.
    Multiply24 { signed: bool },
    /// 10 / 12: D = bits 63:32 of the same 48-bit product.
    MultiplyHigh24 { signed: bool },
    /// 17 / 19: D = S0 < S1 ? S0 : S1.
    Minimum { signed: bool },
    /// 18 / 20: D = S0 >= S1 ? S0 : S1.
    Maximum { signed: bool },
    /// 30: D = ~(S0 ^ S1).
    Xnor,
    /// 37: D = S0 + S1.
    Add,
    /// 38: D = S0 - S1.
    Subtract,
    /// 39: D = S1 - S0.
    SubtractReverse,
}

impl VectorInteger {
    /// Opcode bits 30-25 of the VOP2 word.
    fn from_opcode(opcode: u8) -> Option<Self> {
        Some(match opcode {
            1 => Self::Select,
            40 => Self::AddCarry,
            41 | 42 => Self::SubtractBorrow {
                reversed: opcode == 42,
            },
            9 => Self::Multiply24 { signed: true },
            10 => Self::MultiplyHigh24 { signed: true },
            11 => Self::Multiply24 { signed: false },
            12 => Self::MultiplyHigh24 { signed: false },
            17 => Self::Minimum { signed: true },
            18 => Self::Maximum { signed: true },
            19 => Self::Minimum { signed: false },
            20 => Self::Maximum { signed: false },
            30 => Self::Xnor,
            37 => Self::Add,
            38 => Self::Subtract,
            39 => Self::SubtractReverse,
            _ => return None,
        })
    }

    fn reads_vcc(self) -> bool {
        matches!(
            self,
            Self::Select | Self::AddCarry | Self::SubtractBorrow { .. }
        )
    }

    fn writes_vcc(self) -> bool {
        matches!(self, Self::AddCarry | Self::SubtractBorrow { .. })
    }

    // Guide 12.7: carry/borrow is determined from the exact mathematical result.
    fn apply_vcc(self, a: u32, b: u32, bit: u32) -> (u32, bool) {
        match self {
            Self::Select => (if bit == 0 { a } else { b }, false),
            Self::AddCarry => {
                let exact = u64::from(a) + u64::from(b) + u64::from(bit);
                (exact as u32, exact > u64::from(u32::MAX))
            }
            Self::SubtractBorrow { reversed } => {
                let (minuend, subtrahend) = if reversed { (b, a) } else { (a, b) };
                let exact = i64::from(minuend) - i64::from(subtrahend) - i64::from(bit);
                (exact as u32, exact < 0)
            }
            _ => unreachable!("VOP2 operation does not read VCC"),
        }
    }

    /// D for S0 = `a`, S1 = `b`, from exact i64 arithmetic reduced modulo 2^32.
    fn apply(self, a: u32, b: u32) -> u32 {
        let field = |bits: u32, signed: bool| {
            if signed {
                i64::from(((bits << 8) as i32) >> 8)
            } else {
                i64::from(bits & 0x00ff_ffff)
            }
        };
        let widen = |bits: u32, signed: bool| {
            if signed {
                i64::from(bits as i32)
            } else {
                i64::from(bits)
            }
        };
        match self {
            Self::Multiply24 { signed } => (field(a, signed) * field(b, signed)) as u32,
            Self::MultiplyHigh24 { signed } => ((field(a, signed) * field(b, signed)) >> 32) as u32,
            Self::Minimum { signed } => {
                if widen(a, signed) < widen(b, signed) {
                    a
                } else {
                    b
                }
            }
            Self::Maximum { signed } => {
                if widen(a, signed) >= widen(b, signed) {
                    a
                } else {
                    b
                }
            }
            Self::Xnor => !(a ^ b),
            Self::Add => a.wrapping_add(b),
            Self::Subtract => a.wrapping_sub(b),
            Self::SubtractReverse => b.wrapping_sub(a),
            Self::Select | Self::AddCarry | Self::SubtractBorrow { .. } => {
                unreachable!("VOP2 operation requires VCC")
            }
        }
    }
}

/// VOP2 binary32 arithmetic (AMD RDNA 2 ISA guide section 12.7: ADD, SUB,
/// SUBREV and MUL on page 131, MIN on page 132, MAX on page 133).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorFloat {
    /// 3: D = S0 + S1.
    Add,
    /// 4: D = S0 - S1.
    Subtract,
    /// 5: D = S1 - S0.
    SubtractReverse,
    /// 8: D = S0 * S1.
    Multiply,
    /// 15: the page 132 minimum.
    Minimum,
    /// 16: the page 133 maximum.
    Maximum,
}

impl VectorFloat {
    /// Opcode bits 30-25 of the VOP2 word.
    fn from_opcode(opcode: u8) -> Option<Self> {
        Some(match opcode {
            3 => Self::Add,
            4 => Self::Subtract,
            5 => Self::SubtractReverse,
            8 => Self::Multiply,
            15 => Self::Minimum,
            16 => Self::Maximum,
            _ => return None,
        })
    }

    /// D for S0 = `a`, S1 = `b`, or `None` for an input or result outside the
    /// modeled domain (docs/shaders/vector-binary-f32.md): a subnormal input,
    /// a NaN input to ADD/SUB/SUBREV/MUL or a signalling NaN to MIN/MAX, and
    /// an arithmetic result that is NaN or whose exact value is nonzero and
    /// below 2^-126 in magnitude.
    fn apply(self, a: u32, b: u32) -> Option<u32> {
        if f32_subnormal(a) || f32_subnormal(b) {
            return None;
        }
        const SIGN: u32 = 0x8000_0000;
        match self {
            Self::Minimum | Self::Maximum => f32_select(self == Self::Minimum, a, b),
            _ if f32_nan(a) || f32_nan(b) => None,
            Self::Add => f32_add(a, b),
            Self::Subtract => f32_add(a, b ^ SIGN),
            Self::SubtractReverse => f32_add(b, a ^ SIGN),
            Self::Multiply => f32_multiply(a, b),
        }
    }
}

fn f32_nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > 0x7f80_0000
}

fn f32_subnormal(bits: u32) -> bool {
    bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0
}

/// Sign, integer significand and power of two of a finite binary32 value
/// that is zero or normal: |x| = significand * 2^power.
fn f32_split(bits: u32) -> (bool, u128, i32) {
    let biased = ((bits >> 23) & 0xff) as i32;
    let significand = if biased == 0 {
        0
    } else {
        u128::from(bits & 0x007f_ffff | 0x0080_0000)
    };
    (bits >> 31 == 1, significand, biased - 150)
}

/// Guide pages 132 and 133, IEEE_MODE read as irrelevant once signalling NaNs
/// are refused: a quiet NaN gives the other operand (S1 when both are NaN), a
/// +0/-0 pair gives -0 for MIN and +0 for MAX, otherwise S0 < S1 ? S0 : S1
/// (MIN) or S0 >= S1 ? S0 : S1 (MAX; S0 > S1 outside IEEE mode selects the
/// same bits). The order is the integer order of the sign-magnitude words.
fn f32_select(minimum: bool, a: u32, b: u32) -> Option<u32> {
    let signalling = |x: u32| f32_nan(x) && x & 0x0040_0000 == 0;
    if signalling(a) || signalling(b) {
        return None;
    }
    if f32_nan(a) {
        return Some(b);
    }
    if f32_nan(b) {
        return Some(a);
    }
    if (a | b) & 0x7fff_ffff == 0 {
        return Some(if minimum { a | b } else { a & b });
    }
    // Negative values order by decreasing magnitude, positive by increasing.
    let key = |x: u32| -> i64 {
        let magnitude = i64::from(x & 0x7fff_ffff);
        if x >> 31 == 1 { -magnitude } else { magnitude }
    };
    let take_a = if minimum {
        key(a) < key(b)
    } else {
        key(a) >= key(b)
    };
    Some(if take_a { a } else { b })
}

/// Correctly rounded (nearest even) binary32 sum of two non-NaN words.
fn f32_add(a: u32, b: u32) -> Option<u32> {
    const INFINITY: u32 = 0x7f80_0000;
    let infinite = |x: u32| x & 0x7fff_ffff == INFINITY;
    match (infinite(a), infinite(b)) {
        // +inf + -inf is NaN.
        (true, true) => return (a == b).then_some(a),
        (true, false) => return Some(a),
        (false, true) => return Some(b),
        (false, false) => {}
    }
    let (sign_a, mut significand_a, power_a) = f32_split(a);
    let (sign_b, mut significand_b, power_b) = f32_split(b);
    if significand_a == 0 && significand_b == 0 {
        // Only -0 + -0 is -0 under round to nearest.
        return Some(a & b);
    }
    if significand_a == 0 {
        return Some(b);
    }
    if significand_b == 0 {
        return Some(a);
    }
    // Align on the lower power. A gap above 100 leaves the smaller operand
    // far below a quarter unit of the larger normal one, which is then exact.
    let power = power_a.min(power_b);
    let gap = power_a.abs_diff(power_b);
    if gap > 100 {
        return Some(if power_a > power_b { a } else { b });
    }
    if power_a > power_b {
        significand_a <<= gap;
    } else {
        significand_b <<= gap;
    }
    let signed = |negative: bool, value: u128| {
        let value = value as i128;
        if negative { -value } else { value }
    };
    let sum = signed(sign_a, significand_a) + signed(sign_b, significand_b);
    if sum == 0 {
        // An exact cancellation is +0 under round to nearest.
        return Some(0);
    }
    f32_round(sum < 0, sum.unsigned_abs(), power)
}

/// Correctly rounded (nearest even) binary32 product of two non-NaN words.
fn f32_multiply(a: u32, b: u32) -> Option<u32> {
    let sign = (a ^ b) & 0x8000_0000;
    let infinite = |x: u32| x & 0x7fff_ffff == 0x7f80_0000;
    let zero = |x: u32| x & 0x7fff_ffff == 0;
    if infinite(a) || infinite(b) {
        // 0 * inf is NaN.
        return (!zero(a) && !zero(b)).then_some(sign | 0x7f80_0000);
    }
    let (_, significand_a, power_a) = f32_split(a);
    let (_, significand_b, power_b) = f32_split(b);
    if significand_a == 0 || significand_b == 0 {
        return Some(sign);
    }
    f32_round(sign != 0, significand_a * significand_b, power_a + power_b)
}

/// Encode `magnitude * 2^power` (nonzero) rounded to nearest even, or `None`
/// when the exact value is below 2^-126 (MODE.FP_DENORM output flushing is
/// not pinned). Overflow gives infinity.
fn f32_round(negative: bool, magnitude: u128, power: i32) -> Option<u32> {
    let width = 128 - magnitude.leading_zeros() as i32;
    // magnitude * 2^power lies in [2^(width - 1 + power), 2^(width + power)).
    if width - 1 + power < -126 {
        return None;
    }
    let shift = width - 24;
    let (mut significand, mut shift) = if shift > 0 {
        let kept = magnitude >> shift;
        let remainder = magnitude - (kept << shift);
        let half = 1_u128 << (shift - 1);
        let up = remainder > half || (remainder == half && kept % 2 == 1);
        (kept + u128::from(up), shift)
    } else {
        (magnitude << -shift, shift)
    };
    if significand == 1 << 24 {
        significand >>= 1;
        shift += 1;
    }
    let biased = shift + power + 23 + 127;
    let sign = u32::from(negative) << 31;
    if biased >= 255 {
        return Some(sign | 0x7f80_0000);
    }
    Some(sign | (biased as u32) << 23 | (significand as u32 & 0x007f_ffff))
}

/// VOP1 32-bit integer and bit forms with a general SRC0, and the binary32
/// rounding forms (AMD RDNA 2 ISA guide section 12.8). None writes VCC or SCC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorUnary {
    /// 55: D = ~S0.
    Not,
    /// 56: D[31:0] = S0[0:31].
    ReverseBits,
    /// 57: zeros before the first one from the MSB, -1 if none.
    FindFirstOneFromMsb,
    /// 58: position of the first one from the LSB, -1 if none.
    FindFirstOneFromLsb,
    /// 59: bits after the sign bit that equal it, from the MSB; -1 if all
    /// 32 bits are equal.
    FindSignChange,
    /// 33: binary32 integral part, toward zero.
    Truncate,
    /// 34: binary32 rounded toward +infinity.
    Ceiling,
    /// 35: binary32 rounded to nearest, ties to even.
    RoundNearestEven,
    /// 36: binary32 rounded toward -infinity.
    Floor,
}

impl VectorUnary {
    /// Opcode bits 16-9 of the VOP1 word.
    fn from_opcode(opcode: u8) -> Option<Self> {
        Some(match opcode {
            55 => Self::Not,
            56 => Self::ReverseBits,
            57 => Self::FindFirstOneFromMsb,
            58 => Self::FindFirstOneFromLsb,
            59 => Self::FindSignChange,
            33 => Self::Truncate,
            34 => Self::Ceiling,
            35 => Self::RoundNearestEven,
            36 => Self::Floor,
            _ => return None,
        })
    }

    /// The binary32 rounding forms, which refuse integer inline constants.
    fn is_float(self) -> bool {
        matches!(
            self,
            Self::Truncate | Self::Ceiling | Self::RoundNearestEven | Self::Floor
        )
    }

    /// D for S0, or `None` for an input outside the modeled domain.
    fn apply(self, s0: u32) -> Option<u32> {
        if self.is_float() {
            return round_f32(self, s0);
        }
        Some(self.apply_integer(s0))
    }

    fn apply_integer(self, s0: u32) -> u32 {
        // -1 (all ones) when the scan finds nothing.
        let found = |index: u32| if index == 32 { u32::MAX } else { index };
        match self {
            Self::Not => !s0,
            Self::ReverseBits => s0.reverse_bits(),
            Self::FindFirstOneFromMsb => found(s0.leading_zeros()),
            Self::FindFirstOneFromLsb => found(s0.trailing_zeros()),
            Self::FindSignChange => {
                // Normalize to a non-negative word: the run after the sign bit
                // is then a run of zeros.
                let positive = if (s0 as i32) < 0 { !s0 } else { s0 };
                found(positive.leading_zeros())
            }
            _ => unreachable!("binary32 rounding is handled by round_f32"),
        }
    }
}

/// Guide 12.8 binary32 rounding of S0 to an integral value.
///
/// The value is split into sign, integer significand and power of two, the
/// integer part and the discarded remainder are compared with half a unit, and
/// the rounded magnitude is encoded again. Subnormal inputs are read as their
/// exact values (MODE.FP_DENORM single-precision input denormals allowed, the
/// pinned ABI mode). RNDNE follows the guide's "round-to-nearest-even
/// semantics" (IEEE 754 roundTiesToEven), so a zero result keeps the sign of
/// S0. Refused (`None`): every NaN, whose result the guide leaves to MODE.IEEE
/// and MODE.DX10_CLAMP.
fn round_f32(kind: VectorUnary, s0: u32) -> Option<u32> {
    let negative = s0 >> 31 == 1;
    let biased = (s0 >> 23) & 0xff;
    let fraction = s0 & 0x007f_ffff;
    if biased == 0xff {
        // Infinity is integral; NaN is refused.
        return (fraction == 0).then_some(s0);
    }
    // |S0| = significand * 2^(power), exactly.
    let (significand, power) = if biased == 0 {
        (u64::from(fraction), -149_i32)
    } else {
        (u64::from(fraction | 0x0080_0000), biased as i32 - 150)
    };
    if power >= 0 {
        return Some(s0);
    }
    let shift = power.unsigned_abs().min(63);
    let whole = significand >> shift;
    let remainder = significand - (whole << shift);
    let half = 1_u64 << (shift - 1);
    let up = match kind {
        VectorUnary::Truncate => false,
        VectorUnary::Ceiling => !negative && remainder != 0,
        VectorUnary::Floor => negative && remainder != 0,
        _ => remainder > half || (remainder == half && whole % 2 == 1),
    };
    let magnitude = whole + u64::from(up);
    let sign = s0 & 0x8000_0000;
    if magnitude == 0 {
        return Some(sign);
    }
    // At most 2^24, so the conversion is exact.
    Some(sign | (magnitude as f32).to_bits())
}

/// Execute original little-endian gfx1030 bytes with the synthetic binding ABI.
///
/// Each dispatched lane reads the corresponding input word and may write only
/// its own output word. Integer operations are exact modulo 2^32.
pub fn execute(code: &[u8], input: &[u32]) -> Result<Vec<u32>, ReferenceError> {
    if input.is_empty() || input.len() > MAX_INPUT_WORDS {
        return Err(error(0, "input word count outside reference limits"));
    }
    if code.len() > MAX_CODE_BYTES {
        return Err(error(0, "instruction stream exceeds reference limit"));
    }
    let program = decode(code)?;
    for instruction in &program {
        match instruction.op {
            Op::End if instruction.offset + instruction.byte_len != code.len() => {
                return Err(error(
                    instruction.offset + instruction.byte_len,
                    "trailing code after s_endpgm",
                ));
            }
            Op::Load {
                index, resource, ..
            } if (index, resource) != (0, 0) => {
                return Err(error(
                    instruction.offset,
                    "load outside synthetic binding ABI",
                ));
            }
            Op::Store {
                index, resource, ..
            } if (index, resource) != (0, 4) => {
                return Err(error(
                    instruction.offset,
                    "store outside synthetic binding ABI",
                ));
            }
            Op::ScalarLoad { resource, .. } if resource != 0 => {
                return Err(error(
                    instruction.offset,
                    "scalar load outside synthetic binding ABI",
                ));
            }
            _ => {}
        }
    }
    if !program
        .last()
        .is_some_and(|instruction| instruction.op == Op::End)
    {
        return Err(error(code.len(), "missing s_endpgm"));
    }
    // Waves are consecutive groups of 32 dispatch indices; a final partial
    // wave has absent lanes, which contribute 0 to compare masks.
    let mut output = Vec::with_capacity(input.len());
    for wave in input.chunks(32) {
        output.extend(execute_wave(&program, wave, input)?);
    }
    Ok(output)
}

fn error(offset: usize, detail: &'static str) -> ReferenceError {
    ReferenceError { offset, detail }
}

/// Read the consecutive SGPRs of a validated image descriptor.
fn read_descriptor_words(
    scalars: &[Option<u32>; 128],
    base: u8,
    count: usize,
    offset: usize,
) -> Result<Vec<u32>, ReferenceError> {
    (0..count)
        .map(|index| {
            scalars[usize::from(base) + index].ok_or_else(|| {
                error(
                    offset,
                    "image sample reads an uninitialized descriptor SGPR",
                )
            })
        })
        .collect()
}

/// The reference detail string for a texture sampling failure.
fn descriptor_detail(detail: texture_sampling::SamplingError) -> &'static str {
    match detail {
        texture_sampling::SamplingError::InvalidDimensions => "invalid texture dimensions",
        texture_sampling::SamplingError::InvalidMipCount => "invalid mip count",
        texture_sampling::SamplingError::MipOutOfBounds => "selected mip index is out of bounds",
        texture_sampling::SamplingError::InvalidRowPitch => "invalid row pitch for mip level",
        texture_sampling::SamplingError::InvalidOffset => "invalid buffer offset",
        texture_sampling::SamplingError::BufferTooSmall => "buffer is too small for texture layout",
        texture_sampling::SamplingError::NonFiniteCoordinates => "non-finite texture coordinates",
        texture_sampling::SamplingError::InvalidResourceType => {
            "resource descriptor is not a 2D image"
        }
        texture_sampling::SamplingError::InvalidResourceFormat => {
            "resource descriptor is not RGBA8 unorm"
        }
        texture_sampling::SamplingError::InvalidResourceLevel => {
            "unsupported resource level, channel select or base level"
        }
        texture_sampling::SamplingError::InvalidResourceExtent => {
            "resource width or height sets a reserved high bit"
        }
        texture_sampling::SamplingError::UnsupportedSwizzleMode => {
            "resource swizzle mode is not linear"
        }
        texture_sampling::SamplingError::InvalidResourceMinLod => "resource minimum LOD is nonzero",
        texture_sampling::SamplingError::InvalidResourceLevelRange => {
            "resource last level is below base level"
        }
        texture_sampling::SamplingError::ResourceExtensionNonzero => {
            "nonzero 256-bit resource extension words"
        }
        texture_sampling::SamplingError::ResourceUnsupportedBits { .. } => {
            "resource descriptor sets bits the validator does not support"
        }
        texture_sampling::SamplingError::SamplerAddressing => {
            "sampler addressing is not clamp-to-last-texel with normalized coordinates"
        }
        texture_sampling::SamplingError::SamplerFilter => "sampler filtering is not point",
        texture_sampling::SamplingError::SamplerMipFilter => "sampler mip filtering is not point",
        texture_sampling::SamplingError::SamplerLodRange => {
            "sampler LOD range does not cover the resource mip levels"
        }
        texture_sampling::SamplingError::SamplerUnsupportedBits { .. } => {
            "sampler sets bits the validator does not support"
        }
        texture_sampling::SamplingError::InvalidLod => {
            "explicit LOD is not a representable integer mip level"
        }
    }
}

/// An operation and its span in the original byte stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedInstruction {
    pub offset: usize,
    pub byte_len: usize,
    pub op: Op,
}

/// Decode the supported gfx1030 subset without applying the execution ABI.
///
/// AMD RDNA 2 ISA guide, Instruction Encoding. Identify the forms by bytes
/// and independently extract their variable fields.
///
/// # Errors
///
/// Returns the failing byte offset for an unsupported, truncated
/// or invalid instruction stream.
pub fn decode(code: &[u8]) -> Result<Vec<DecodedInstruction>, ReferenceError> {
    let mut program = Vec::new();
    let mut offset = 0;
    while let Some(bytes) = code.get(offset..).filter(|bytes| !bytes.is_empty()) {
        let (op, length) = match bytes {
            [
                0,
                0x20,
                opcode @ (0x30 | 0x70),
                0xe0,
                index,
                data,
                group @ 0..=25,
                0x80,
                ..,
            ] => {
                let (data, index, resource) = (*data, *index, *group * 4);
                let op = if *opcode == 0x30 {
                    Op::Load {
                        data,
                        index,
                        resource,
                    }
                } else {
                    Op::Store {
                        data,
                        index,
                        resource,
                    }
                };
                (op, 8)
            }
            [0x70, 0x3f, 0x8c, 0xbf, ..] => (Op::WaitRead, 4),
            [0x7f, 0xc0, 0x8c, 0xbf, ..] => (Op::WaitScalar, 4),
            [_, _, 0x20 | 0x24 | 0x28, 0xf4] => {
                return Err(error(offset, "truncated scalar load"));
            }
            // SMEM (guide 13.2.1): byte 3 is the encoding and opcode bits 6-7,
            // byte 2 the other opcode bits with GLC clear, SBASE and SDATA
            // across bytes 0 and 1. An even SBASE is an aligned quad.
            [
                low,
                middle @ 0..=0x1f,
                opcode @ (0x20 | 0x24 | 0x28),
                0xf4,
                first,
                second,
                third @ 0..=0x0f,
                selector,
                ..,
            ] if low & 1 == 0 && selector & 1 == 0 => {
                let dwords = 1 << ((opcode >> 2) & 3);
                let data = (middle << 2) | (low >> 6);
                let resource = (low & 0x3f) * 2;
                if data % dwords != 0 || data + dwords > 106 || resource > 100 {
                    return Err(error(offset, "unsupported scalar load registers"));
                }
                let register = match selector >> 1 {
                    125 => None,
                    register @ 0..=105 => Some(register),
                    _ => return Err(error(offset, "unsupported scalar load offset")),
                };
                let immediate = u32::from_le_bytes([*first, *second, *third, 0]);
                (
                    Op::ScalarLoad {
                        data,
                        resource,
                        offset: immediate,
                        register,
                        dwords,
                    },
                    8,
                )
            }
            [0x00, 0x00, 0xfd, 0xbb, ..] => (Op::WaitWrite, 4),
            [0x00, 0x00, 0x81, 0xbf, ..] => (Op::End, 4),
            [
                source,
                3,
                destination @ (0x80..=0xeb | 0xfe..=0xff),
                0xbe,
                tail @ ..,
            ] => {
                let (source, length) = decode_scalar(*source, tail, offset)?;
                (Op::ScalarMove(destination & 0x7f, source), length)
            }
            // SOP1: ENCODING 0b10_1111101 in bits 23-31, SDST in 16-22 (s0-s105
            // only), OP in 8-15.
            [
                source,
                op @ (7 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 | 26 | 52 | 60..=71),
                destination @ 0x80..=0xe9,
                0xbe,
                tail @ ..,
            ] => {
                let kind = match op {
                    7 => ScalarUnary::Not,
                    11 => ScalarUnary::ReverseBits,
                    13 => ScalarUnary::CountZeros,
                    15 => ScalarUnary::CountOnes,
                    17 => ScalarUnary::FindFirstZero,
                    19 => ScalarUnary::FindFirstOne,
                    21 => ScalarUnary::FindLastBit,
                    23 => ScalarUnary::FindLastSignBit,
                    25 => ScalarUnary::SignExtendByte,
                    26 => ScalarUnary::SignExtendShort,
                    // The twelve EXEC-writing forms of guide 12.3, printed
                    // pages 120-122.
                    60 => ScalarUnary::AndSaveExec,
                    61 => ScalarUnary::OrSaveExec,
                    62 => ScalarUnary::XorSaveExec,
                    63 => ScalarUnary::Andn2SaveExec,
                    64 => ScalarUnary::Orn2SaveExec,
                    65 => ScalarUnary::NandSaveExec,
                    66 => ScalarUnary::NorSaveExec,
                    67 => ScalarUnary::XnorSaveExec,
                    68 => ScalarUnary::Andn1SaveExec,
                    69 => ScalarUnary::Orn1SaveExec,
                    70 => ScalarUnary::Andn1Wrexec,
                    71 => ScalarUnary::Andn2Wrexec,
                    _ => ScalarUnary::Absolute,
                };
                let (source, length) = decode_scalar(*source, tail, offset)?;
                (Op::ScalarUnary(kind, destination & 0x7f, source), length)
            }
            // SOPC: ENCODING 0b10_1111110 in bits 23-31, OP in 16-22 (guide
            // 13.1.4); 0-5 signed and 6-11 unsigned EQ, LG, GT, GE, LT, LE.
            [left, right, op @ 0..=11, 0xbf, tail @ ..] if (*left, *right) != (255, 255) => {
                let (left, left_length) = decode_scalar(*left, tail, offset)?;
                let (right, right_length) = decode_scalar(*right, tail, offset)?;
                // VOPC relation offsets: LT 1, EQ 2, LE 3, GT 4, NE 5, GE 6.
                let relation = [2, 5, 4, 6, 1, 3][usize::from(op % 6)];
                let comparison = Comparison {
                    offset: relation,
                    signed: *op < 6,
                };
                let op = if relation == 2 {
                    Op::CompareEqual(left, right)
                } else {
                    Op::ScalarCompare(comparison, left, right)
                };
                (op, left_length.max(right_length))
            }
            // SOPK (guide 13.1.2): ENCODING 0b1011 in bits 28-31, OP 23-27,
            // SDST 16-22 (s0-s105 only), SIMM16 0-15. The format never takes
            // a literal dword (guide 12.2), so the word is always 4 bytes.
            [lo, hi, low, high @ 0xb0..=0xb8, ..]
                if low & 0x7f <= 105 && matches!((high & 0xf) << 1 | low >> 7, 0 | 2..=16) =>
            {
                let kind = match (high & 0xf) << 1 | low >> 7 {
                    0 => ScalarImmediate::Move,
                    2 => ScalarImmediate::ConditionalMove,
                    // EQ, LG, GT, GE, LT, LE as VOPC relation offsets (see SOPC).
                    op @ 3..=14 => ScalarImmediate::Compare(Comparison {
                        offset: [2, 5, 4, 6, 1, 3][usize::from((op - 3) % 6)],
                        signed: op <= 8,
                    }),
                    15 => ScalarImmediate::Add,
                    _ => ScalarImmediate::Multiply,
                };
                let simm16 = u16::from_le_bytes([*lo, *hi]);
                (Op::ScalarImmediate(kind, low & 0x7f, simm16), 4)
            }
            // SOP2: OP in bits 23-29, SDST in bits 16-22, SSRC1, SSRC0.
            [left, right, low, high @ 0x80..=0xbf, tail @ ..]
                if (*left, *right) != (255, 255)
                    && low & 0x7f <= 105
                    && matches!(
                        (high & 0x3f) << 1 | low >> 7,
                        0..=10 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 | 34
                            | 36 | 38..=40 | 44 | 46..=49 | 53 | 54
                    ) =>
            {
                let kind = match (high & 0x3f) << 1 | low >> 7 {
                    0 => ScalarAlu::Add,
                    1 => ScalarAlu::Subtract,
                    2 => ScalarAlu::AddSigned,
                    3 => ScalarAlu::SubtractSigned,
                    4 => ScalarAlu::AddWithCarry,
                    5 => ScalarAlu::SubtractWithBorrow,
                    6 => ScalarAlu::Minimum { signed: true },
                    7 => ScalarAlu::Minimum { signed: false },
                    8 => ScalarAlu::Maximum { signed: true },
                    9 => ScalarAlu::Maximum { signed: false },
                    10 => ScalarAlu::Select,
                    14 => ScalarAlu::And,
                    16 => ScalarAlu::Or,
                    18 => ScalarAlu::Xor,
                    20 => ScalarAlu::AndNot,
                    22 => ScalarAlu::OrNot,
                    24 => ScalarAlu::Nand,
                    26 => ScalarAlu::Nor,
                    28 => ScalarAlu::Xnor,
                    30 => ScalarAlu::ShiftLeft,
                    32 => ScalarAlu::ShiftRight,
                    34 => ScalarAlu::ArithmeticShiftRight,
                    36 => ScalarAlu::BitfieldMask,
                    38 => ScalarAlu::Multiply,
                    39 => ScalarAlu::BitfieldExtract { signed: false },
                    40 => ScalarAlu::BitfieldExtract { signed: true },
                    44 => ScalarAlu::AbsoluteDifference,
                    op @ 46..=49 => ScalarAlu::ShiftLeftAdd(op - 45),
                    53 => ScalarAlu::MultiplyHigh { signed: false },
                    _ => ScalarAlu::MultiplyHigh { signed: true },
                };
                let (left, left_length) = decode_scalar(*left, tail, offset)?;
                let (right, right_length) = decode_scalar(*right, tail, offset)?;
                (
                    Op::Scalar(kind, low & 0x7f, left, right),
                    left_length.max(right_length),
                )
            }
            [
                lo,
                hi,
                opcode @ (0x82 | 0x84 | 0x86 | 0x87 | 0x88),
                0xbf,
                ..,
            ] => {
                let words = i16::from_le_bytes([*lo, *hi]);
                let words = usize::try_from(words).map_err(|_| error(offset, "backward branch"))?;
                (
                    Op::Branch {
                        target: offset + 4 + words * 4,
                        condition: match opcode {
                            0x84 => BranchCondition::SccZero,
                            0x86 => BranchCondition::VccZero,
                            0x87 => BranchCondition::VccNonzero,
                            0x88 => BranchCondition::ExecZero,
                            _ => BranchCondition::Always,
                        },
                    },
                    4,
                )
            }
            [first, second, third, opcode, ..]
                if matches!(opcode & 0xfe, 0x2c | 0x30 | 0x34 | 0x36 | 0x38 | 0x3a)
                    && second & 1 != 0 =>
            {
                let kind = match opcode & 0xfe {
                    0x34 => VectorAlu::ShiftLeft,
                    0x2c => VectorAlu::ShiftRight,
                    0x30 => VectorAlu::ArithmeticShiftRight,
                    0x36 => VectorAlu::And,
                    0x38 => VectorAlu::Or,
                    _ => VectorAlu::Xor,
                };
                let dst = (third >> 1) | ((opcode & 1) << 7);
                let src1 = (second >> 1) | ((third & 1) << 7);
                (Op::Vector(kind, dst, *first, src1), 4)
            }
            // Guide §§12.7.1, 12.9.1, 12.12 and 13.3.4: VOP3A extends
            // VOP2 by 0x100 and preserves the VOPC opcode. This subset
            // fixes unused SRC2 to inline 0 and excludes FP ABS/NEG/OMOD,
            // halfword OP_SEL, and CLAMP (integer saturation or FP compare
            // exceptions, §6.5).
            [
                dst,
                0,
                op,
                tag @ (0xd4 | 0xd5),
                lo,
                middle,
                hi @ 2..=3,
                0x02,
                ..,
            ] if middle & 1 != 0 => {
                let src0 = CompareSource::Vector(*lo);
                let src1 = (middle >> 1) | ((hi & 1) << 7);
                match tag {
                    0xd4 if matches!(op, 128..=135 | 192..=199) && *dst <= 106 => (
                        Op::CompareVop3(
                            *dst,
                            Comparison {
                                offset: op & 7,
                                signed: *op < 192,
                            },
                            src0,
                            src1,
                        ),
                        8,
                    ),
                    0xd5 if *op == 37 => (Op::Add(*dst, *lo, src1), 8),
                    0xd5 => (
                        Op::VectorInteger(
                            VectorInteger::from_opcode(*op)
                                .ok_or_else(|| error(offset, "unsupported VOP3 opcode"))?,
                            *dst,
                            src0,
                            src1,
                        ),
                        8,
                    ),
                    _ => return Err(error(offset, "unsupported VOP3 opcode or destination")),
                }
            }
            // VOPC (bits 31-25 = 0b0111110): SRC0 bits 0-8, VSRC1 bits 9-16, OP bits 17-24.
            [lo, packed, upper, tag @ 0x7c..=0x7d, tail @ ..]
                if CompareWidth::from_opcode(
                    (u16::from(*tag & 1) << 7) | u16::from(*upper >> 1),
                )
                .is_some() =>
            {
                let opcode = (u16::from(*tag & 1) << 7) | u16::from(*upper >> 1);
                let (comparison, width) = CompareWidth::from_opcode(opcode).unwrap();
                let vsrc1 = (packed >> 1) | ((upper & 1) << 7);
                let selector = u16::from(*lo) | (u16::from(packed & 1) << 8);
                // docs/shaders/vector-compare.md subset: no literal for 16-bit;
                // 64-bit takes v[n:n+1] (n <= 254), s[2k:2k+1] (k <= 52) or
                // an inline integer, with VSRC1 v[m:m+1] (m <= 254).
                let accepted = match width {
                    CompareWidth::Half => selector != 255,
                    CompareWidth::Word => true,
                    CompareWidth::Double => {
                        vsrc1 != 255
                            && matches!(selector, 0..=104 | 128..=208 | 256..=510)
                            && (selector >= 128 || selector % 2 == 0)
                    }
                };
                if !accepted {
                    return Err(error(offset, "unsupported VOPC source"));
                }
                let (src0, length) =
                    decode_vector_source(*lo, *packed, tail, offset, "unsupported VOPC source")?;
                let op = if width == CompareWidth::Word {
                    Op::Compare(comparison, src0, vsrc1)
                } else {
                    Op::WideCompare(comparison, width, src0, vsrc1)
                };
                (op, length)
            }
            // VOP2 binary32 forms (bit 31 = 0), same fields as the integer forms.
            [lo, packed, upper, tag @ 0x00..=0x7b, tail @ ..]
                if VectorFloat::from_opcode(tag >> 1).is_some() =>
            {
                let kind = VectorFloat::from_opcode(tag >> 1).unwrap();
                let dst = (upper >> 1) | ((tag & 1) << 7);
                let vsrc1 = (packed >> 1) | ((upper & 1) << 7);
                let selector = u16::from(*lo) | (u16::from(packed & 1) << 8);
                if (128..=208).contains(&selector) {
                    return Err(error(offset, "integer inline constant as binary32 source"));
                }
                let (src0, length) =
                    decode_vector_source(*lo, *packed, tail, offset, "unsupported VOP2 source")?;
                (Op::VectorFloat(kind, dst, src0, vsrc1), length)
            }
            // VOP2 integer forms (bit 31 = 0): OP bits 25-30, VDST 17-24,
            // VSRC1 9-16, SRC0 0-8.
            [lo, packed, upper, tag @ 0x00..=0x7b, tail @ ..]
                if VectorInteger::from_opcode(tag >> 1).is_some() =>
            {
                let kind = VectorInteger::from_opcode(tag >> 1).unwrap();
                let dst = (upper >> 1) | ((tag & 1) << 7);
                let vsrc1 = (packed >> 1) | ((upper & 1) << 7);
                let (src0, length) =
                    decode_vector_source(*lo, *packed, tail, offset, "unsupported VOP2 source")?;
                (Op::VectorInteger(kind, dst, src0, vsrc1), length)
            }
            // VOP1 integer forms (bits 31-25 = 0b0111111): VDST bits 17-24,
            // OP 9-16, SRC0 0-8.
            [lo, packed, upper, tag @ 0x7e..=0x7f, tail @ ..]
                if VectorUnary::from_opcode((packed >> 1) | ((upper & 1) << 7)).is_some() =>
            {
                let kind = VectorUnary::from_opcode((packed >> 1) | ((upper & 1) << 7)).unwrap();
                let dst = (upper >> 1) | ((tag & 1) << 7);
                let selector = u16::from(*lo) | (u16::from(packed & 1) << 8);
                if kind.is_float() && (128..=208).contains(&selector) {
                    return Err(error(offset, "integer inline constant as binary32 source"));
                }
                let (src0, length) =
                    decode_vector_source(*lo, *packed, tail, offset, "unsupported VOP1 source")?;
                (Op::VectorUnary(kind, dst, src0), length)
            }
            // v_swap_b32: SRC0 must be a VGPR (256-511), since it is written.
            [lo, packed, upper, tag @ 0x7e..=0x7f, ..]
                if (packed >> 1) | ((upper & 1) << 7) == 101 =>
            {
                if packed & 1 == 0 {
                    return Err(error(offset, "v_swap_b32 SRC0 is not a VGPR"));
                }
                (Op::Swap((upper >> 1) | ((tag & 1) << 7), *lo), 4)
            }
            [lo, packed, upper, tag @ 0x7e..=0x7f, tail @ ..]
                if packed & 0xfe == 2 && upper & 1 == 0 =>
            {
                let dst = (upper >> 1) | ((tag & 1) << 7);
                let selector = u16::from(*lo) | (u16::from(packed & 1) << 8);
                if selector <= 106 {
                    // VOP1 SRC0 0-105 selects a normal SGPR, 106 vcc_lo.
                    program.push(DecodedInstruction {
                        offset,
                        byte_len: 4,
                        op: Op::MoveScalar(dst, *lo),
                    });
                    offset += 4;
                    continue;
                }
                let (src, length) = match selector {
                    128..=192 => (Source::Immediate(u32::from(selector - 128)), 4),
                    193..=208 => (
                        Source::Immediate(0_u32.wrapping_sub(u32::from(selector - 192))),
                        4,
                    ),
                    242 => (Source::Immediate(1.0f32.to_bits()), 4),
                    255 => {
                        let literal = tail
                            .first_chunk::<4>()
                            .ok_or_else(|| error(offset, "truncated literal"))?;
                        (Source::Immediate(u32::from_le_bytes(*literal)), 8)
                    }
                    256..=511 => (Source::Register(*lo), 4),
                    _ => return Err(error(offset, "unsupported VOP1 source")),
                };
                (Op::Move(dst, src), length)
            }
            // MIMG (guide 8.2 and 13.9 Table 100, printed page 277): byte 3
            // carries the ENCODING 0b111100 and the opcode's high bit; byte 2
            // its low six bits and byte 0 its low bit. Non-NSA is two dwords.
            [low, flags, op_lo, 0xf0, vaddr, vdata, srsrc, ssamp, ..] => {
                let opcode = u16::from(*low & 1) << 7 | u16::from(*op_lo >> 2);
                let explicit_lod = match opcode {
                    39 => false,
                    36 => true,
                    _ => return Err(error(offset, "unsupported MIMG opcode")),
                };
                // Bounded scope (guide Table 41, printed page 69): non-NSA, DIM
                // 2D, DLC/UNORM/GLC/DA/R128/TFE/LWE clear, a nonzero DMASK, and
                // clear reserved/A16/D16 bits.
                if *low & 0xfe != 0x08
                    || *flags & 0xf0 != 0
                    || *op_lo & 0x3 != 0
                    || *ssamp & 0xfc != 0
                {
                    return Err(error(offset, "unsupported MIMG encoding or data mask"));
                }
                let dmask = *flags & 0xf;
                let resource = (*srsrc & 0x1f) * 4;
                let sampler = ((*srsrc >> 5) | ((*ssamp & 0x3) << 3)) * 4;
                if resource > 98 || sampler > 102 {
                    return Err(error(offset, "unsupported MIMG descriptor registers"));
                }
                let components = u16::from(dmask.count_ones() as u8);
                let address = if explicit_lod { 3 } else { 2 };
                if u16::from(*vdata) + components - 1 > 255 || u16::from(*vaddr) + address - 1 > 255
                {
                    return Err(error(offset, "unsupported MIMG register span"));
                }
                (
                    Op::ImageSample {
                        dst: *vdata,
                        coord: *vaddr,
                        resource,
                        sampler,
                        dmask,
                        explicit_lod,
                    },
                    8,
                )
            }
            _ => {
                return Err(error(
                    offset,
                    "unsupported or truncated gfx1030 instruction",
                ));
            }
        };
        program.push(DecodedInstruction {
            offset,
            byte_len: length,
            op,
        });
        offset += length;
    }
    for instruction in &program {
        if let Op::Branch { target, .. } = instruction.op
            && program
                .binary_search_by_key(&target, |item| item.offset)
                .is_err()
        {
            return Err(error(
                instruction.offset,
                "branch target is not an instruction boundary",
            ));
        }
    }
    Ok(program)
}

fn decode_scalar(
    selector: u8,
    tail: &[u8],
    offset: usize,
) -> Result<(Source, usize), ReferenceError> {
    match selector {
        0..=105 => Ok((Source::Register(selector), 4)),
        128..=192 => Ok((Source::Immediate(u32::from(selector - 128)), 4)),
        193..=208 => Ok((
            Source::Immediate(0u32.wrapping_sub(u32::from(selector - 192))),
            4,
        )),
        255 => {
            let literal = tail
                .first_chunk::<4>()
                .ok_or_else(|| error(offset, "truncated scalar literal"))?;
            Ok((Source::Immediate(u32::from_le_bytes(*literal)), 8))
        }
        _ => Err(error(offset, "unsupported scalar source")),
    }
}

/// A vector SRC0 selector (bits 0-8: `lo` and bit 0 of `packed`): s0-s105 or
/// `vcc_lo` (106), an inline integer, a literal dword from `tail`, or a VGPR.
fn decode_vector_source(
    lo: u8,
    packed: u8,
    tail: &[u8],
    offset: usize,
    unsupported: &'static str,
) -> Result<(CompareSource, usize), ReferenceError> {
    let selector = u16::from(lo) | (u16::from(packed & 1) << 8);
    Ok(match selector {
        0..=106 => (CompareSource::Scalar(lo), 4),
        128..=192 => (CompareSource::Immediate(u32::from(selector - 128)), 4),
        193..=208 => (
            CompareSource::Immediate(0_u32.wrapping_sub(u32::from(selector - 192))),
            4,
        ),
        255 => {
            let literal = tail
                .first_chunk::<4>()
                .ok_or_else(|| error(offset, "truncated literal"))?;
            (CompareSource::Immediate(u32::from_le_bytes(*literal)), 8)
        }
        256..=511 => (CompareSource::Vector(lo), 4),
        _ => return Err(error(offset, unsupported)),
    })
}

fn vector_alu(op: VectorAlu, a: u32, b: u32) -> u32 {
    match op {
        VectorAlu::ShiftLeft => b << (a & 31),
        VectorAlu::ShiftRight => b >> (a & 31),
        VectorAlu::ArithmeticShiftRight => ((b as i32) >> (a & 31)) as u32,
        VectorAlu::And => a & b,
        VectorAlu::Or => a | b,
        VectorAlu::Xor => a ^ b,
    }
}

const VCC_LO: usize = 106;
const VCC_HI: usize = 107;
const EXEC_LO: usize = 126;

/// Runs one Wave32 wave in lockstep. Scalar state is shared by the lanes;
/// VGPRs, transfers and outputs are per lane.
/// `input` is the wave's words and `buffer` those of the whole dispatch, which a
/// scalar load addresses.
fn execute_wave(
    program: &[DecodedInstruction],
    input: &[u32],
    buffer: &[u32],
) -> Result<Vec<u32>, ReferenceError> {
    let lanes = input.len();
    let mut registers = vec![[None::<u32>; 256]; lanes];
    let mut scalars = [None; 128];
    let mut exec = u32::MAX;
    scalars[EXEC_LO] = Some(exec);
    let mut scc = None;
    // SGPRs holding a loaded value, or one computed from such, and whether SCC
    // came from one. Only the rules for skipped code depend on this.
    let mut loaded = [false; 128];
    let mut runtime_scc = false;
    // Whether VCC_LO was last written by a compare with nonzero EXEC.
    let mut lane_derived_vcc = false;
    let mut target = 0;
    let mut pending_read = None;
    // Destinations of scalar loads without a completed wait, and their values.
    let mut pending_scalars: Vec<(usize, u32)> = Vec::new();
    let mut pending_write = vec![None; lanes];
    let mut output = vec![None; lanes];
    for (position, &DecodedInstruction { offset, op, .. }) in program.iter().enumerate() {
        if offset < target {
            continue;
        }
        let read_scalar = |source| match source {
            Source::Register(register) => {
                let index = usize::from(register);
                if pending_scalars.iter().any(|(slot, _)| *slot == index) {
                    return Err(error(offset, "SGPR read before scalar load wait"));
                }
                scalars[index].ok_or_else(|| error(offset, "read from uninitialized SGPR"))
            }
            Source::Immediate(value) => Ok(value),
        };
        let runtime = scalar_sources(op).any(|slot| loaded[slot]);
        if let Some(destination) = scalar_destination(op) {
            if pending_scalars.iter().any(|(slot, _)| *slot == destination) {
                return Err(error(offset, "SGPR write before scalar load wait"));
            }
            loaded[destination] = runtime;
        }
        let read = |lane: usize, register: u8| {
            if register == 0 {
                return Err(error(offset, "v0 is reserved for the resource index"));
            }
            if pending_read == Some(register) {
                return Err(error(offset, "VGPR read before load wait"));
            }
            registers[lane][usize::from(register)]
                .ok_or_else(|| error(offset, "read from uninitialized VGPR"))
        };
        // Lanes that execute a vector ALU instruction under the current EXEC.
        let executes = |lane: usize| exec & (1 << lane) != 0;
        if exec != u32::MAX {
            match op {
                Op::Move(dst, _)
                | Op::MoveScalar(dst, _)
                | Op::Add(dst, ..)
                | Op::Vector(_, dst, ..)
                | Op::VectorInteger(_, dst, ..)
                | Op::VectorFloat(_, dst, ..)
                | Op::VectorUnary(_, dst, _)
                | Op::Swap(dst, _) => {
                    if exec != 0
                        && registers
                            .iter()
                            .any(|lane| lane[usize::from(dst)].is_none())
                    {
                        return Err(error(offset, "partial EXEC requires an initialized VGPR"));
                    }
                    // As in translation, an SGPR source must be known even for an inactive lane.
                    if let Op::MoveScalar(_, src) = op
                        && exec != 0
                    {
                        read_scalar(Source::Register(src))?;
                    }
                }
                Op::Load { .. } | Op::Store { .. } => {
                    return Err(error(offset, "vector memory requires full Wave32 EXEC"));
                }
                _ => {}
            }
        }
        // Per-lane results of a vector ALU instruction, applied after all lanes read.
        let mut results: Vec<Option<(u8, u32)>> = vec![None; lanes];
        match op {
            Op::Load { data: dst, .. } => {
                if dst == 0 || pending_read.is_some() {
                    return Err(error(
                        offset,
                        "invalid load destination or overlapping load",
                    ));
                }
                pending_read = Some(dst);
            }
            Op::Store { data: src, .. } => {
                for lane in 0..lanes {
                    let value = read(lane, src)?;
                    if pending_write[lane].is_some() || output[lane].is_some() {
                        return Err(error(offset, "multiple writes to the output"));
                    }
                    pending_write[lane] = Some(value);
                }
            }
            Op::WaitRead => {
                if let Some(dst) = pending_read.take() {
                    for (lane, &word) in input.iter().enumerate() {
                        results[lane] = Some((dst, word));
                    }
                }
            }
            Op::ScalarLoad {
                data,
                offset: immediate,
                register,
                dwords,
                ..
            } => {
                if data < 8 {
                    return Err(error(offset, "write to a reserved resource SGPR"));
                }
                let extra = match register {
                    Some(register) => read_scalar(Source::Register(register))?,
                    None => 0,
                };
                // Guide 7.2.1: the two low bits of the address are ignored.
                let byte = u64::from(immediate) + u64::from(extra);
                let mut values = Vec::new();
                for index in 0..u64::from(dwords) {
                    let slot = usize::from(data) + index as usize;
                    let old = read_scalar(Source::Register(slot as u8))?;
                    // Guide 7.5: an out-of-range word is not loaded.
                    let word = usize::try_from(byte / 4 + index).ok();
                    let value = word.and_then(|word| buffer.get(word)).copied();
                    values.push((slot, value.unwrap_or(old)));
                }
                pending_scalars.extend(values);
            }
            Op::WaitScalar => {
                for (slot, value) in pending_scalars.drain(..) {
                    scalars[slot] = Some(value);
                    loaded[slot] = true;
                }
            }
            Op::WaitWrite => {
                for lane in 0..lanes {
                    if let Some(value) = pending_write[lane].take() {
                        output[lane] = Some(value);
                    }
                }
            }
            Op::Move(dst, source) => {
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let value = match source {
                        Source::Register(src) => read(lane, src)?,
                        Source::Immediate(value) => value,
                    };
                    results[lane] = Some((dst, value));
                }
            }
            Op::Add(dst, lhs, rhs) => {
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let value = read(lane, lhs)?.wrapping_add(read(lane, rhs)?);
                    results[lane] = Some((dst, value));
                }
            }
            Op::MoveScalar(dst, src) => {
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    results[lane] = Some((dst, read_scalar(Source::Register(src))?));
                }
            }
            Op::Vector(operator, dst, lhs, rhs) => {
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let value = vector_alu(operator, read(lane, lhs)?, read(lane, rhs)?);
                    results[lane] = Some((dst, value));
                }
            }
            Op::VectorInteger(kind, dst, src0, src1) => {
                if exec != 0 {
                    // As for a compare, sources are checked once for the wave,
                    // whichever present lanes are active.
                    let check = |register: u8| {
                        (0..lanes).try_for_each(|lane| read(lane, register).map(|_| ()))
                    };
                    match src0 {
                        CompareSource::Vector(register) => check(register)?,
                        CompareSource::Scalar(register) => {
                            read_scalar(Source::Register(register))?;
                        }
                        CompareSource::Immediate(_) => {}
                    }
                    check(src1)?;
                }
                let incoming = if exec != 0 && kind.reads_vcc() {
                    read_scalar(Source::Register(VCC_LO as u8))?
                } else {
                    0
                };
                let mut outgoing = 0;
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let s0 = match src0 {
                        CompareSource::Vector(register) => read(lane, register)?,
                        CompareSource::Scalar(register) => read_scalar(Source::Register(register))?,
                        CompareSource::Immediate(bits) => bits,
                    };
                    let s1 = read(lane, src1)?;
                    let value = if kind.reads_vcc() {
                        let (value, carry) = kind.apply_vcc(s0, s1, (incoming >> lane) & 1);
                        outgoing |= u32::from(carry) << lane;
                        value
                    } else {
                        kind.apply(s0, s1)
                    };
                    results[lane] = Some((dst, value));
                }
                // Guide 3.9: only executing lanes can contribute a carry.
                if kind.writes_vcc() {
                    scalars[VCC_LO] = Some(outgoing);
                    scalars[VCC_HI] = None;
                    lane_derived_vcc = exec != 0;
                }
            }
            Op::VectorFloat(kind, dst, src0, src1) => {
                if exec != 0 {
                    // Sources are checked once for the wave, as for VOP2 integers.
                    match src0 {
                        CompareSource::Vector(register) => {
                            (0..lanes).try_for_each(|lane| read(lane, register).map(|_| ()))?
                        }
                        CompareSource::Scalar(register) => {
                            read_scalar(Source::Register(register))?;
                        }
                        CompareSource::Immediate(_) => {}
                    }
                    (0..lanes).try_for_each(|lane| read(lane, src1).map(|_| ()))?;
                }
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let s0 = match src0 {
                        CompareSource::Vector(register) => read(lane, register)?,
                        CompareSource::Scalar(register) => read_scalar(Source::Register(register))?,
                        CompareSource::Immediate(bits) => bits,
                    };
                    let value = kind.apply(s0, read(lane, src1)?).ok_or_else(|| {
                        error(offset, "binary32 operand or result outside domain")
                    })?;
                    results[lane] = Some((dst, value));
                }
            }
            Op::VectorUnary(kind, dst, src0) => {
                if exec != 0 {
                    // Sources are checked once for the wave, as for VOP2.
                    match src0 {
                        CompareSource::Vector(register) => {
                            (0..lanes).try_for_each(|lane| read(lane, register).map(|_| ()))?
                        }
                        CompareSource::Scalar(register) => {
                            read_scalar(Source::Register(register))?;
                        }
                        CompareSource::Immediate(_) => {}
                    }
                }
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let s0 = match src0 {
                        CompareSource::Vector(register) => read(lane, register)?,
                        CompareSource::Scalar(register) => read_scalar(Source::Register(register))?,
                        CompareSource::Immediate(bits) => bits,
                    };
                    let value = kind
                        .apply(s0)
                        .ok_or_else(|| error(offset, "binary32 input outside rounding domain"))?;
                    results[lane] = Some((dst, value));
                }
            }
            // Guide 12.8: tmp = D; D = S0; S0 = tmp, in each active lane.
            Op::Swap(dst, src0) => {
                if exec != 0 {
                    for register in [dst, src0] {
                        (0..lanes).try_for_each(|lane| read(lane, register).map(|_| ()))?;
                    }
                }
                let mut swapped = Vec::new();
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    swapped.push((lane, read(lane, dst)?, read(lane, src0)?));
                }
                if !swapped.is_empty()
                    && pending_read.is_some_and(|register| register == dst || register == src0)
                {
                    return Err(error(offset, "write to immutable or pending VGPR"));
                }
                for (lane, old_dst, old_src0) in swapped {
                    registers[lane][usize::from(dst)] = Some(old_src0);
                    registers[lane][usize::from(src0)] = Some(old_dst);
                }
            }
            Op::ScalarMove(dst, src) => {
                if dst < 8 {
                    return Err(error(offset, "write to a reserved resource SGPR"));
                }
                let value = read_scalar(src)?;
                if usize::from(dst) == EXEC_LO {
                    exec = value;
                }
                if usize::from(dst) == VCC_LO {
                    lane_derived_vcc = false;
                }
                scalars[usize::from(dst)] = Some(value);
            }
            Op::ScalarUnary(kind, dst, src) => {
                if dst < 8 {
                    return Err(error(offset, "write to a reserved resource SGPR"));
                }
                let value = read_scalar(src)?;
                // -1 (all ones) when no bit is found.
                let found = |index: u32| if index == 32 { u32::MAX } else { index };
                // The destination value and, for the EXEC-writing forms of
                // guide 12.3 (printed pages 120-122), the new EXEC_LO. The ten
                // SAVEEXEC forms leave the previous EXEC_LO in the
                // destination, the two WREXEC forms the new one.
                let (result, new_exec) = match kind {
                    ScalarUnary::Not => (!value, None),
                    ScalarUnary::ReverseBits => (value.reverse_bits(), None),
                    ScalarUnary::CountZeros => (value.count_zeros(), None),
                    ScalarUnary::CountOnes => (value.count_ones(), None),
                    ScalarUnary::FindFirstZero => (found(value.trailing_ones()), None),
                    ScalarUnary::FindFirstOne => (found(value.trailing_zeros()), None),
                    ScalarUnary::FindLastBit => (found(value.leading_zeros()), None),
                    ScalarUnary::FindLastSignBit => {
                        let same = if value >> 31 == 1 {
                            value.leading_ones()
                        } else {
                            value.leading_zeros()
                        };
                        (found(same), None)
                    }
                    ScalarUnary::SignExtendByte => (value as u8 as i8 as i32 as u32, None),
                    ScalarUnary::SignExtendShort => (value as u16 as i16 as i32 as u32, None),
                    ScalarUnary::Absolute => ((value as i32).wrapping_abs() as u32, None),
                    ScalarUnary::AndSaveExec => (exec, Some(value & exec)),
                    ScalarUnary::OrSaveExec => (exec, Some(value | exec)),
                    ScalarUnary::XorSaveExec => (exec, Some(value ^ exec)),
                    ScalarUnary::Andn2SaveExec => (exec, Some(value & !exec)),
                    ScalarUnary::Orn2SaveExec => (exec, Some(value | !exec)),
                    ScalarUnary::NandSaveExec => (exec, Some(!(value & exec))),
                    ScalarUnary::NorSaveExec => (exec, Some(!(value | exec))),
                    ScalarUnary::XnorSaveExec => (exec, Some(!(value ^ exec))),
                    ScalarUnary::Andn1SaveExec => (exec, Some(!value & exec)),
                    ScalarUnary::Orn1SaveExec => (exec, Some(!value | exec)),
                    ScalarUnary::Andn1Wrexec => {
                        let exec = !value & exec;
                        (exec, Some(exec))
                    }
                    ScalarUnary::Andn2Wrexec => {
                        let exec = value & !exec;
                        (exec, Some(exec))
                    }
                };
                scalars[usize::from(dst)] = Some(result);
                if let Some(new_exec) = new_exec {
                    // The mask does not gate this instruction, but it does
                    // gate the vector work that follows it.
                    exec = new_exec;
                    scalars[EXEC_LO] = Some(new_exec);
                    runtime_scc = runtime;
                    scc = Some(new_exec != 0);
                } else if matches!(
                    kind,
                    ScalarUnary::Not
                        | ScalarUnary::CountZeros
                        | ScalarUnary::CountOnes
                        | ScalarUnary::Absolute
                ) {
                    runtime_scc = runtime;
                    scc = Some(result != 0);
                }
            }
            Op::CompareEqual(lhs, rhs) => {
                runtime_scc = runtime;
                scc = Some(read_scalar(lhs)? == read_scalar(rhs)?);
            }
            // Guide 12.4: SCC = (S0 <relation> S1); no SGPR is written.
            Op::ScalarCompare(comparison, lhs, rhs) => {
                runtime_scc = runtime;
                scc = Some(comparison.holds(read_scalar(lhs)?, read_scalar(rhs)?));
            }
            // Guide 12.2: SDST is also the second source; SIMM16 is
            // sign-extended except by the unsigned compares.
            Op::ScalarImmediate(kind, dst, simm16) => {
                let extended = i32::from(simm16 as i16);
                if let ScalarImmediate::Compare(comparison) = kind {
                    let constant = if comparison.signed {
                        extended as u32
                    } else {
                        u32::from(simm16)
                    };
                    let value = read_scalar(Source::Register(dst))?;
                    runtime_scc = runtime;
                    scc = Some(comparison.holds(value, constant));
                } else {
                    if dst < 8 {
                        return Err(error(offset, "write to a reserved resource SGPR"));
                    }
                    let value = match kind {
                        ScalarImmediate::ConditionalMove
                            if !scc
                                .ok_or_else(|| error(offset, "move on uninitialized SCC"))? =>
                        {
                            None
                        }
                        ScalarImmediate::Add => {
                            let exact = i64::from(read_scalar(Source::Register(dst))? as i32)
                                + i64::from(extended);
                            // SCC = the exact signed sum does not fit in i32.
                            runtime_scc = runtime;
                            scc = Some(i32::try_from(exact).is_err());
                            Some(exact as u32)
                        }
                        ScalarImmediate::Multiply => {
                            let exact = i64::from(read_scalar(Source::Register(dst))? as i32)
                                * i64::from(extended);
                            Some(exact.rem_euclid(1 << 32) as u32)
                        }
                        _ => Some(extended as u32),
                    };
                    if let Some(value) = value {
                        scalars[usize::from(dst)] = Some(value);
                    }
                }
            }
            Op::Scalar(kind, dst, lhs, rhs) => {
                if dst < 8 {
                    return Err(error(offset, "write to a reserved resource SGPR"));
                }
                let (lhs, rhs) = (i64::from(read_scalar(lhs)?), i64::from(read_scalar(rhs)?));
                let carry = match kind {
                    ScalarAlu::AddWithCarry | ScalarAlu::SubtractWithBorrow => i64::from(
                        scc.ok_or_else(|| error(offset, "carry-in from uninitialized SCC"))?,
                    ),
                    ScalarAlu::Select => {
                        i64::from(scc.ok_or_else(|| error(offset, "select on uninitialized SCC"))?)
                    }
                    _ => 0,
                };
                let (signed_lhs, signed_rhs) =
                    (i64::from(lhs as u32 as i32), i64::from(rhs as u32 as i32));
                // Exact integer arithmetic, then reduction modulo 2^32.
                let (exact, flag) = match kind {
                    ScalarAlu::Add | ScalarAlu::AddWithCarry => {
                        let sum = lhs + rhs + carry;
                        (sum, sum >= 1 << 32)
                    }
                    ScalarAlu::Subtract | ScalarAlu::SubtractWithBorrow => {
                        let difference = lhs - rhs - carry;
                        (difference, difference < 0)
                    }
                    // The exact signed result must fit in i32, else SCC = overflow.
                    ScalarAlu::AddSigned | ScalarAlu::SubtractSigned => {
                        let signed = |value: i64| i64::from(value as u32 as i32);
                        let exact = if kind == ScalarAlu::AddSigned {
                            signed(lhs) + signed(rhs)
                        } else {
                            signed(lhs) - signed(rhs)
                        };
                        (exact, i32::try_from(exact).is_err())
                    }
                    ScalarAlu::And | ScalarAlu::Or | ScalarAlu::Xor => {
                        let bits = match kind {
                            ScalarAlu::And => lhs & rhs,
                            ScalarAlu::Or => lhs | rhs,
                            _ => lhs ^ rhs,
                        };
                        (bits, bits != 0)
                    }
                    // Guide 12.1: strict comparisons, so equal operands give
                    // S1 (the same value) and SCC = 0.
                    ScalarAlu::Minimum { signed } | ScalarAlu::Maximum { signed } => {
                        let (a, b) = if signed {
                            (signed_lhs, signed_rhs)
                        } else {
                            (lhs, rhs)
                        };
                        let first = if matches!(kind, ScalarAlu::Minimum { .. }) {
                            a < b
                        } else {
                            a > b
                        };
                        (if first { lhs } else { rhs }, first)
                    }
                    // The SCC-less forms; `flag` is unused for them below.
                    ScalarAlu::Select => (if carry == 1 { lhs } else { rhs }, false),
                    ScalarAlu::BitfieldMask => (((1 << (lhs & 0x1f)) - 1) << (rhs & 0x1f), false),
                    ScalarAlu::Multiply => (signed_lhs * signed_rhs, false),
                    ScalarAlu::MultiplyHigh { signed } => {
                        // The unsigned product fits only in u64.
                        let high = if signed {
                            (signed_lhs * signed_rhs) >> 32
                        } else {
                            ((lhs as u64 * rhs as u64) >> 32) as i64
                        };
                        (high, false)
                    }
                    // Offset S1[4:0], width S1[22:16] (0-127). A width of 32 or
                    // more keeps every bit of the shifted word.
                    ScalarAlu::BitfieldExtract { signed } => {
                        let (shift, width) = (rhs & 0x1f, (rhs >> 16) & 0x7f);
                        let data = if signed { signed_lhs } else { lhs };
                        let shifted = data >> shift;
                        let value = if width == 0 {
                            0
                        } else if width >= 32 {
                            shifted
                        } else {
                            let field = shifted & ((1 << width) - 1);
                            if signed && field >> (width - 1) == 1 {
                                field - (1 << width)
                            } else {
                                field
                            }
                        };
                        (value, value.rem_euclid(1 << 32) != 0)
                    }
                    // The 32-bit difference first (guide examples), then |D|.
                    ScalarAlu::AbsoluteDifference => {
                        let difference = (signed_lhs - signed_rhs) as u32 as i32;
                        let value = difference.unsigned_abs();
                        (i64::from(value), value != 0)
                    }
                    // The carry out of the exact (S0 << N) + S1.
                    ScalarAlu::ShiftLeftAdd(n) => {
                        let sum = (lhs << n) + rhs;
                        (sum, sum >= 1 << 32)
                    }
                    // Guide 12.1: shifts use only S1[4:0]; all set SCC = (D != 0).
                    _ => {
                        let (a, b) = (lhs as u32, rhs as u32);
                        let bits = match kind {
                            ScalarAlu::AndNot => a & !b,
                            ScalarAlu::OrNot => a | !b,
                            ScalarAlu::Nand => !(a & b),
                            ScalarAlu::Nor => !(a | b),
                            ScalarAlu::Xnor => !(a ^ b),
                            ScalarAlu::ShiftLeft => a << (b & 0x1f),
                            ScalarAlu::ShiftRight => a >> (b & 0x1f),
                            _ => ((a as i32) >> (b & 0x1f)) as u32,
                        };
                        (i64::from(bits), bits != 0)
                    }
                };
                let value = exact.rem_euclid(1 << 32) as u32;
                scalars[usize::from(dst)] = Some(value);
                // Guide 12.1 and Tables 11, 12, 14: these forms leave SCC unchanged.
                if !matches!(
                    kind,
                    ScalarAlu::Select
                        | ScalarAlu::BitfieldMask
                        | ScalarAlu::Multiply
                        | ScalarAlu::MultiplyHigh { .. }
                ) {
                    runtime_scc = runtime;
                    scc = Some(flag);
                }
            }
            // RDNA 2 ISA section 3.9: VCC[n] = EXEC[n] & test(n), fully written.
            Op::Compare(..) | Op::WideCompare(..) | Op::CompareVop3(..) => {
                let (dst, comparison, width, src0, src1) = match op {
                    Op::Compare(comparison, src0, src1) => {
                        (VCC_LO, comparison, CompareWidth::Word, src0, src1)
                    }
                    Op::CompareVop3(dst, comparison, src0, src1) => {
                        (usize::from(dst), comparison, CompareWidth::Word, src0, src1)
                    }
                    Op::WideCompare(comparison, width, src0, src1) => {
                        (VCC_LO, comparison, width, src0, src1)
                    }
                    _ => unreachable!(),
                };
                let mut mask = 0u32;
                if exec != 0 {
                    // The registers each source occupies: a pair for 64-bit forms.
                    let span = |register: u8| {
                        if width == CompareWidth::Double {
                            vec![register, register + 1]
                        } else {
                            vec![register]
                        }
                    };
                    // Sources are checked once for the wave, even if no present lane is active.
                    let check = |register: u8| {
                        span(register).into_iter().try_for_each(|register| {
                            (0..lanes).try_for_each(|lane| read(lane, register).map(|_| ()))
                        })
                    };
                    let join = |values: Vec<u32>| {
                        values
                            .iter()
                            .rev()
                            .fold(0u64, |acc, &value| (acc << 32) | u64::from(value))
                    };
                    let read_pair = |lane: usize, register: u8| {
                        span(register)
                            .into_iter()
                            .map(|register| read(lane, register))
                            .collect::<Result<Vec<_>, _>>()
                            .map(join)
                    };
                    let read_scalar_pair = |register: u8| {
                        span(register)
                            .into_iter()
                            .map(|register| read_scalar(Source::Register(register)))
                            .collect::<Result<Vec<_>, _>>()
                            .map(join)
                    };
                    match src0 {
                        CompareSource::Vector(register) => check(register)?,
                        CompareSource::Scalar(register) => {
                            read_scalar_pair(register)?;
                        }
                        CompareSource::Immediate(_) => {}
                    }
                    check(src1)?;
                    for lane in (0..lanes).filter(|&lane| executes(lane)) {
                        let s0 = match src0 {
                            CompareSource::Vector(register) => read_pair(lane, register)?,
                            CompareSource::Scalar(register) => read_scalar_pair(register)?,
                            // Table 20: an inline integer is the value 0..64 or
                            // -1..-16 at the operand's width (sign extended).
                            CompareSource::Immediate(bits) => bits as i32 as i64 as u64,
                        };
                        if comparison.holds_at(width, s0, read_pair(lane, src1)?) {
                            mask |= 1 << lane;
                        }
                    }
                }
                scalars[dst] = Some(mask);
                if dst == VCC_LO {
                    scalars[VCC_HI] = None;
                    lane_derived_vcc = exec != 0;
                }
            }
            Op::Branch {
                target: destination,
                condition,
            } => {
                let tests_vcc = matches!(
                    condition,
                    BranchCondition::VccZero | BranchCondition::VccNonzero
                );
                if tests_vcc && scalars[VCC_LO].is_none() {
                    return Err(error(offset, "branch with uninitialized VCC_LO"));
                }
                let tests_runtime_scc = condition == BranchCondition::SccZero && runtime_scc;
                if (tests_vcc && lane_derived_vcc) || tests_runtime_scc {
                    check_runtime_region(
                        &program[position + 1..],
                        destination,
                        &registers,
                        &scalars,
                        exec,
                        pending_read.is_some() || !pending_scalars.is_empty(),
                        tests_runtime_scc,
                        offset,
                    )?;
                }
                let take_branch = match condition {
                    BranchCondition::Always => true,
                    BranchCondition::SccZero => {
                        !scc.ok_or_else(|| error(offset, "branch with uninitialized SCC"))?
                    }
                    BranchCondition::VccZero => scalars[VCC_LO] == Some(0),
                    BranchCondition::VccNonzero => scalars[VCC_LO] != Some(0),
                    BranchCondition::ExecZero => exec == 0,
                };
                if take_branch {
                    target = destination;
                }
            }
            Op::End => {
                if !pending_scalars.is_empty()
                    || pending_read.is_some()
                    || pending_write.iter().any(Option::is_some)
                {
                    return Err(error(offset, "outstanding transfer at termination"));
                }
                return output
                    .iter()
                    .map(|value| {
                        value.ok_or_else(|| error(offset, "lane did not write its output"))
                    })
                    .collect();
            }
            Op::ImageSample {
                dst,
                coord,
                resource,
                sampler,
                dmask,
                explicit_lod,
            } => {
                // The T# and S# descriptors are wave-uniform: read and validate
                // them once, then sample for each active lane.
                let resource_words = read_descriptor_words(&scalars, resource, 8, offset)?;
                let sampler_words = read_descriptor_words(&scalars, sampler, 4, offset)?;
                let (desc, base) = texture_sampling::parse_resource_descriptor(
                    resource_words[..8].try_into().unwrap(),
                )
                .map_err(|detail| error(offset, descriptor_detail(detail)))?;
                texture_sampling::validate_sampler_descriptor(
                    sampler_words.try_into().unwrap(),
                    desc.mips - 1,
                )
                .map_err(|detail| error(offset, descriptor_detail(detail)))?;
                let bytes = buffer
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<u8>>();
                let required = desc
                    .required_size()
                    .map_err(|detail| error(offset, descriptor_detail(detail)))?;
                let data = bytes
                    .get(base..base.saturating_add(required))
                    .ok_or_else(|| error(offset, "texture data outside the input buffer"))?;

                // Active lanes sample and write one VGPR per DMASK component.
                // All sources are read before any destination is written, so a
                // destination may overlap a coordinate register.
                let mut writes: Vec<(usize, u8, u32)> = Vec::new();
                for lane in (0..lanes).filter(|&lane| executes(lane)) {
                    let u = read(lane, coord)?;
                    let v = read(lane, coord + 1)?;
                    let mip = if explicit_lod {
                        texture_sampling::lod_to_mip(read(lane, coord + 2)?, &desc)
                            .map_err(|detail| error(offset, descriptor_detail(detail)))?
                    } else {
                        0
                    };
                    let texel = texture_sampling::sample_rgba8_2d(
                        data,
                        &desc,
                        mip,
                        f32::from_bits(u),
                        f32::from_bits(v),
                    )
                    .map_err(|detail| error(offset, descriptor_detail(detail)))?;
                    let mut slot = dst;
                    for component in 0..4u8 {
                        if dmask & (1 << component) == 0 {
                            continue;
                        }
                        writes.push((
                            lane,
                            slot,
                            (f32::from(texel[usize::from(component)]) / 255.0).to_bits(),
                        ));
                        slot += 1;
                    }
                }
                for (lane, dst, value) in writes {
                    if dst == 0 || pending_read == Some(dst) {
                        return Err(error(offset, "write to immutable or pending VGPR"));
                    }
                    registers[lane][usize::from(dst)] = Some(value);
                }
            }
        }
        for (lane, result) in results.into_iter().enumerate() {
            if let Some((dst, value)) = result {
                if dst == 0 || pending_read == Some(dst) {
                    return Err(error(offset, "write to immutable or pending VGPR"));
                }
                registers[lane][usize::from(dst)] = Some(value);
            }
        }
    }
    Err(error(0, "missing decoded termination"))
}

/// docs/shaders/vector-compare.md section 6: code skipped by `s_cbranch_vccz`
/// on a lane-derived VCC_LO may hold only integer vector ALU instructions
/// whose destinations and sources are ready at the branch. Checked on the
/// code itself, whether or not the branch is taken.
/// The normal SGPR an operation writes, apart from a scalar load.
fn scalar_destination(op: Op) -> Option<usize> {
    match op {
        Op::ScalarMove(dst, _)
        | Op::ScalarUnary(_, dst, _)
        | Op::Scalar(_, dst, ..)
        | Op::CompareVop3(dst, ..) => Some(usize::from(dst)),
        Op::ScalarImmediate(kind, dst, _) if !matches!(kind, ScalarImmediate::Compare(_)) => {
            Some(usize::from(dst))
        }
        _ => None,
    }
}

/// The normal SGPRs a scalar operation reads, as far as their origin matters.
fn scalar_sources(op: Op) -> impl Iterator<Item = usize> {
    let sources = match op {
        Op::ScalarMove(_, source) | Op::ScalarUnary(_, _, source) => [Some(source), None],
        Op::Scalar(_, _, left, right)
        | Op::CompareEqual(left, right)
        | Op::ScalarCompare(_, left, right) => [Some(left), Some(right)],
        Op::ScalarImmediate(kind, register, _) if kind != ScalarImmediate::Move => {
            [Some(Source::Register(register)), None]
        }
        _ => [None, None],
    };
    sources
        .into_iter()
        .flatten()
        .filter_map(|source| match source {
            Source::Register(register) => Some(usize::from(register)),
            Source::Immediate(_) => None,
        })
}

#[allow(clippy::too_many_arguments)]
fn check_runtime_region(
    following: &[DecodedInstruction],
    end: usize,
    registers: &[[Option<u32>; 256]],
    scalars: &[Option<u32>; 128],
    exec: u32,
    load_pending: bool,
    allow_else: bool,
    branch: usize,
) -> Result<(), ReferenceError> {
    if load_pending {
        return Err(error(branch, "runtime branch with an outstanding load"));
    }
    // An `s_branch` that ends the skipped part makes the code up to its target
    // an `else` part, under the same rules.
    let mut end = end;
    let mut has_else = false;
    let ready = |register: u8| {
        register != 0
            && registers
                .iter()
                .all(|lane| lane[usize::from(register)].is_some())
    };
    for instruction in following {
        let offset = instruction.offset;
        if offset >= end {
            break;
        }
        if let Op::Branch {
            target,
            condition: BranchCondition::Always,
        } = instruction.op
            && allow_else
            && !has_else
            && offset + instruction.byte_len == end
        {
            (end, has_else) = (target, true);
            continue;
        }
        let (dst, vector_sources, scalar_source) = match instruction.op {
            Op::Move(dst, Source::Register(src)) => (dst, vec![src], None),
            Op::Move(dst, Source::Immediate(_)) => (dst, vec![], None),
            Op::MoveScalar(dst, src) => (dst, vec![], Some(src)),
            Op::Add(dst, lhs, rhs) => (dst, vec![lhs, rhs], None),
            Op::Vector(_, dst, lhs, rhs) => (dst, vec![lhs, rhs], None),
            _ => {
                return Err(error(
                    offset,
                    "runtime branch skips a non-integer-vector instruction",
                ));
            }
        };
        if !ready(dst) {
            return Err(error(
                offset,
                "runtime branch skips a write to an uninitialized VGPR",
            ));
        }
        if exec == 0 {
            continue;
        }
        if vector_sources.into_iter().any(|register| !ready(register)) {
            return Err(error(offset, "runtime branch skips an unready VGPR read"));
        }
        if let Some(register) = scalar_source
            && scalars[usize::from(register)].is_none()
        {
            return Err(error(offset, "runtime branch skips an unknown SGPR read"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/add.bin");

    #[test]
    fn computes_the_independent_known_answers() {
        let input = [0, 1, u32::MAX, 0u32.wrapping_sub(0x1234_5678)];
        let expected = [0x1234_5678, 0x1234_5679, 0x1234_5677, 0];
        assert_eq!(execute(FIXTURE, &input), Ok(expected.to_vec()));
    }

    #[test]
    fn executes_both_scalar_branch_outcomes() {
        let fixture = include_bytes!("../../../fixtures/rdna2/bitwise-branch.bin");
        let inputs = [0, u32::MAX, 0x1234_5678];
        assert_eq!(
            execute(fixture, &inputs),
            Ok(vec![0x1234_5678, 0x12cb_5687, 0x1200_5600])
        );
        let mut equal = *fixture;
        equal[17] = 0x87; // Compare s8 with 7, so the SCC-zero edge is not taken.
        assert_eq!(execute(&equal, &inputs), Ok(vec![0; 3]));
    }

    #[test]
    fn rejects_invalid_branch_even_when_not_taken() {
        let fixture = include_bytes!("../../../fixtures/rdna2/bitwise-branch.bin");
        let mut invalid = *fixture;
        invalid[17] = 0x87;
        invalid[20] = 3; // Target byte 36, the VOP1 literal.
        assert_eq!(
            execute(&invalid, &[0]).unwrap_err().detail,
            "branch target is not an instruction boundary"
        );
        invalid[20..22].copy_from_slice(&(-1i16).to_le_bytes());
        assert_eq!(
            execute(&invalid, &[0]).unwrap_err().detail,
            "backward branch"
        );
    }

    #[test]
    fn executes_exact_float_fixture_and_rejects_subnormal_operand() {
        let fixture = include_bytes!("../../../fixtures/rdna2/float-exact.bin");
        assert_eq!(
            execute(fixture, &[0, u32::MAX]),
            Ok(vec![0x410c_0000, 0x4120_0000])
        );
        let mut broken = *fixture;
        broken[36..40].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            execute(&broken, &[0]).unwrap_err().detail,
            "binary32 operand or result outside domain"
        );
    }

    #[test]
    fn rejects_truncated_programs_and_trailing_bytes() {
        for length in 0..FIXTURE.len() {
            assert!(
                execute(&FIXTURE[..length], &[0]).is_err(),
                "length {length}"
            );
        }
        let mut trailing = FIXTURE.to_vec();
        trailing.push(0);
        assert_eq!(execute(&trailing, &[0]).unwrap_err().offset, FIXTURE.len());
    }

    #[test]
    fn rejects_access_before_wait() {
        let mut no_read_wait = FIXTURE.to_vec();
        no_read_wait.drain(8..12);
        assert_eq!(
            execute(&no_read_wait, &[1]).unwrap_err().detail,
            "VGPR read before load wait"
        );
        let mut no_write_wait = FIXTURE.to_vec();
        no_write_wait.drain(36..40);
        assert_eq!(
            execute(&no_write_wait, &[1]).unwrap_err().detail,
            "outstanding transfer at termination"
        );
    }

    #[test]
    fn decodes_high_register_indices() {
        // V_MOV_B32 v255, v254 and V_ADD_NC_U32 v255, v128, v255.
        let code = [
            0xfe, 0x03, 0xfe, 0x7f, 0xff, 0, 0x25, 0xd5, 0x80, 0xff, 0x03, 0x02, 0, 0, 0x81, 0xbf,
        ];
        assert_eq!(
            decode(&code),
            Ok(vec![
                DecodedInstruction {
                    offset: 0,
                    byte_len: 4,
                    op: Op::Move(255, Source::Register(254))
                },
                DecodedInstruction {
                    offset: 4,
                    byte_len: 8,
                    op: Op::Add(255, 128, 255)
                },
                DecodedInstruction {
                    offset: 12,
                    byte_len: 4,
                    op: Op::End
                },
            ])
        );
    }

    const IMAGE_SAMPLE_LZ_FIXTURE: &[u8] =
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask1.bin");
    const IMAGE_SAMPLE_L_LOD1_FIXTURE: &[u8] =
        include_bytes!("../../../fixtures/rdna2/image-sample-l-lod1.bin");

    /// The 20 per-lane (u, v) cases. Every value has a zero low 16-bit half so
    /// the fixture can pack both into one word (see `packed`).
    const IMAGE_SAMPLE_CASES: [(f32, f32); 20] = [
        (0.125, 0.125), // mip 0 texel (0, 0) centre
        (0.375, 0.125), // texel (1, 0)
        (0.625, 0.125), // texel (2, 0)
        (0.875, 0.125), // texel (3, 0)
        (0.125, 0.375), // texel (0, 1)
        (0.125, 0.625), // texel (0, 2)
        (0.125, 0.875), // texel (0, 3)
        (0.875, 0.875), // texel (3, 3)
        (0.25, 0.25),   // exactly on the edge: 0.25 * 4 = 1.0
        (0.5, 0.5),     // exactly on the edge: 0.5 * 4 = 2.0
        (-0.5, 0.125),  // u < 0 clamps to x = 0
        (1.5, 0.125),   // u > 1 clamps to x = 3
        (0.125, 1.5),   // v > 1 clamps to y = 3
        (-0.5, -0.5),   // both clamp to the first texel
        (1.5, 1.5),     // both clamp to the last texel
        (0.0, 0.0),     // first texel
        (1.0, 1.0),     // last texel
        (0.75, 0.25),   // texel (3, 1)
        (0.25, 0.75),   // texel (1, 3)
        (0.5, 0.25),    // texel (2, 1)
    ];

    fn packed(u: f32, v: f32) -> u32 {
        (u.to_bits() >> 16) | (v.to_bits() & 0xffff_0000)
    }

    /// Texel byte `(level << 6) | (y << 4) | (x << 2) | channel` makes every
    /// channel of every texel distinct.
    fn texel_byte(level: u32, x: usize, y: usize, channel: usize) -> u8 {
        ((level << 6) | ((y as u32) << 4) | ((x as u32) << 2) | channel as u32) as u8
    }

    /// The 4x4 mip 0 followed by the 2x2 mip 1, packed as little-endian words.
    fn image_sample_texels() -> Vec<u32> {
        let mut bytes = Vec::new();
        for level in 0..2u32 {
            let extent = 4usize >> level;
            for y in 0..extent {
                for x in 0..extent {
                    for channel in 0..4 {
                        bytes.push(texel_byte(level, x, y, channel));
                    }
                }
            }
        }
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    /// The synthetic input: a 4x4 RGBA8 unorm T# with two levels (base field 1
    /// = byte offset 0x100), an S# with point mip filtering and a 1.0 maximum
    /// LOD, the 20 packed coordinate cases at words 12..32, and the texels at
    /// word 64.
    fn image_sample_input() -> Vec<u32> {
        let mut input = vec![
            0x0000_0001, // T# base-address field 1 -> byte offset 0x100
            0xC380_0000, // min lod 0, format 56, width-1 3 high bits
            0x8000_C000, // width-1 3 low bits, height-1 3, resource level 1
            0x9001_0fac, // dst_sel RGBA, base 0, last 1, type 9 (2D)
            0,
            0,
            0,
            0,           // 256-bit extension words (zero for a 2D texture)
            0x0000_0092, // S# clamp_x/y/z = ClampLastTexel (2)
            0x0010_0000, // S# max lod 0x100 = 1.0 (4.8 fixed point)
            0x0400_0000, // S# mip filter point (1) at 91:90
            0,
        ];
        input.extend(IMAGE_SAMPLE_CASES.map(|(u, v)| packed(u, v)));
        input.resize(64, 0);
        input.extend(image_sample_texels());
        input
    }

    fn texel_index(u: f32, v: f32, mip: u32) -> (usize, usize) {
        let extent = 4usize >> mip;
        let x = ((u.clamp(0.0, 1.0) * extent as f32).floor() as usize).min(extent - 1);
        let y = ((v.clamp(0.0, 1.0) * extent as f32).floor() as usize).min(extent - 1);
        (x, y)
    }

    /// One output word per dispatch index. Every lane loads its packed coordinate
    /// word `input[lane]`, splits it into u (low half shifted into the float's
    /// high half) and v (high half), and samples `component` of the mip.
    fn image_sample_expected(input: &[u32], mip: u32, component: usize) -> Vec<u32> {
        input
            .iter()
            .map(|&word| {
                let u = f32::from_bits((word & 0xffff) << 16);
                let v = f32::from_bits(word & 0xffff_0000);
                let (x, y) = texel_index(u, v, mip);
                (f32::from(texel_byte(mip, x, y, component)) / 255.0).to_bits()
            })
            .collect()
    }

    #[test]
    fn decodes_and_executes_image_sample_lz() {
        let program = decode(IMAGE_SAMPLE_LZ_FIXTURE).expect("decode image-sample fixture");
        assert!(program.iter().any(|instruction| matches!(
            instruction.op,
            Op::ImageSample {
                explicit_lod: false,
                ..
            }
        )));
        let input = image_sample_input();
        let output = execute(IMAGE_SAMPLE_LZ_FIXTURE, &input).expect("execute");
        assert_eq!(output, image_sample_expected(&input, 0, 0));
        // The selected red channel differs across the grid; the test would not
        // pass with a single texel or a dropped coordinate.
        assert!(output.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn decodes_and_executes_image_sample_l_at_lod_one() {
        let program = decode(IMAGE_SAMPLE_L_LOD1_FIXTURE).expect("decode image-sample_l fixture");
        assert!(program.iter().any(|instruction| matches!(
            instruction.op,
            Op::ImageSample {
                explicit_lod: true,
                ..
            }
        )));
        let input = image_sample_input();
        let output = execute(IMAGE_SAMPLE_L_LOD1_FIXTURE, &input).expect("execute");
        assert_eq!(output, image_sample_expected(&input, 1, 0));
        assert_ne!(output, image_sample_expected(&input, 0, 0));
    }

    /// The image-sample program with a partial EXEC mask: the masked lanes
    /// sample and the rest keep the value their destination already held.
    /// The words match the assembled fixture, with `s_mov_b32 exec_lo, mask`
    /// between the destination initialisation and the sample.
    fn image_sample_program(mask: u32) -> Vec<u8> {
        let mut words: Vec<u32> = (0..12).map(|index| 0xbe88_0380 + (index << 16)).collect();
        words.extend([
            0xf428_0200,
            0xfa00_0000, // s_buffer_load_dwordx4 s[8:11], s[0:3], 0x0
            0xf428_0300,
            0xfa00_0010, // s_buffer_load_dwordx4 s[12:15], s[0:3], 0x10
            0xf428_0400,
            0xfa00_0020, // s_buffer_load_dwordx4 s[16:19], s[0:3], 0x20
            0xbf8c_c07f, // s_waitcnt lgkmcnt(0)
            0xe030_2000,
            0x8000_0100, // buffer_load_dword v1, v0, s[0:3], 0 idxen
            0xbf8c_3f70, // s_waitcnt vmcnt(0)
            0x7e16_0290, // v_mov_b32_e32 v11, 16
            0x3404_030b, // v_lshlrev_b32_e32 v2, v11, v1
            0x7e14_02ff,
            0xffff_0000,        // v_mov_b32_e32 v10, 0xffff0000
            0x3606_030a,        // v_and_b32_e32 v3, v10, v1
            0x7e0a_0290,        // v_mov_b32_e32 v5, 16: the prior destination value
            0xbefe_0380 | mask, // s_mov_b32 exec_lo, mask
            0xf09c_0108,
            0x0082_0502, // image_sample_lz v5, v[2:3], s[8:15], s[16:19] dmask:0x1
            0xbefe_03c1, // s_mov_b32 exec_lo, -1: the store needs a full wave
            0xe070_2000,
            0x8001_0500, // buffer_store_dword v5, v0, s[4:7], 0 idxen
            0xbbfd_0000, // s_waitcnt_vscnt null, 0
            0xbf81_0000, // s_endpgm
        ]);
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// An inactive lane keeps the value its destination already held: the
    /// sample writes only the lanes the EXEC mask selects.
    #[test]
    fn image_sample_keeps_the_prior_destination_in_inactive_lanes() {
        let input = image_sample_input();
        let sampled = image_sample_expected(&input, 0, 0);
        // Lanes 0 and 2 of every wave sample; the rest keep the sixteen the
        // destination was initialised with.
        let expected: Vec<u32> = sampled
            .iter()
            .enumerate()
            .map(|(index, value)| match index % 32 {
                0 | 2 => *value,
                _ => 16,
            })
            .collect();
        let output = execute(&image_sample_program(0b101), &input).expect("execute");
        assert_eq!(output, expected);
        // An active lane holds the sample and an inactive one its prior value.
        assert_eq!(output[0], sampled[0]);
        assert_eq!(output[1], 16);
        assert_eq!(output[3], 16);
    }

    #[test]
    fn image_sample_lz_rejects_invalid_descriptors() {
        let rejects = |input: Vec<u32>, detail: &str| {
            assert_eq!(
                execute(IMAGE_SAMPLE_LZ_FIXTURE, &input).unwrap_err().detail,
                detail
            );
        };
        // A nonzero 256-bit extension word is refused.
        let mut input = image_sample_input();
        input[4] = 1;
        rejects(input, "nonzero 256-bit resource extension words");
        // A sampler that is not point-and-clamp is refused.
        let mut input = image_sample_input();
        input[8] = 0; // clamp_x = Wrap (0), not ClampLastTexel (2)
        rejects(
            input,
            "sampler addressing is not clamp-to-last-texel with normalized coordinates",
        );
        // A resource of the wrong type is refused.
        let mut input = image_sample_input();
        input[3] = 0x8000_0fac; // type 8 (1D), not 9 (2D)
        rejects(input, "resource descriptor is not a 2D image");
        // Reserved width and height bits are refused.
        let mut input = image_sample_input();
        input[2] |= 1 << 12;
        rejects(input, "resource width or height sets a reserved high bit");
        // A swizzled resource is refused.
        let mut input = image_sample_input();
        input[3] |= 1 << 20;
        rejects(input, "resource swizzle mode is not linear");
        // A nonzero resource minimum LOD is refused.
        let mut input = image_sample_input();
        input[1] |= 1 << 8;
        rejects(input, "resource minimum LOD is nonzero");
        // A last level below the base level is refused.
        let mut input = image_sample_input();
        input[3] = (input[3] & !(0xf << 16)) | (1 << 12);
        rejects(input, "resource last level is below base level");
        // A mip filter of none is refused.
        let mut input = image_sample_input();
        input[10] = 0;
        rejects(input, "sampler mip filtering is not point");
        // A nonzero z filter, a mismatched clamp z, a LOD bias, a nonzero
        // minimum LOD and a maximum LOD below the last level are all refused.
        let mut input = image_sample_input();
        input[10] |= 1 << 24;
        rejects(input, "sampler filtering is not point");
        let mut input = image_sample_input();
        input[8] = 0x12;
        rejects(
            input,
            "sampler addressing is not clamp-to-last-texel with normalized coordinates",
        );
        let mut input = image_sample_input();
        input[10] |= 1;
        rejects(input, "sampler sets bits the validator does not support");
        let mut input = image_sample_input();
        input[9] = 1;
        rejects(
            input,
            "sampler LOD range does not cover the resource mip levels",
        );
        let mut input = image_sample_input();
        input[9] = 0;
        rejects(
            input,
            "sampler LOD range does not cover the resource mip levels",
        );
    }
}
