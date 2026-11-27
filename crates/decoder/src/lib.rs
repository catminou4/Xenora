//! Decoder for Xenora's bounded little-endian gfx1030 instruction stream.

use std::fmt;

pub use xenora_isa::{
    Architecture, CompareRelation, ExportTarget, ImageSampleOp, Instruction, InterpolationPhase,
    ScalarAluOp, ScalarCompareOp, ScalarImmediateCompareOp, ScalarImmediateOp, ScalarSource,
    ScalarUnaryOp, SoppControlOp, VectorAluOp, VectorAluVop3Op, VectorCompareOp,
    VectorCompareVop3Op, VectorSource, VectorSwapOp, VectorUnaryOp,
};
use xenora_isa::{
    BUFFER_LOAD_DWORD, BUFFER_STORE_DWORD, EXP, S_WAITCNT_VSCNT, V_INTERP_P1_F32, V_INTERP_P2_F32,
    V_MOV_B32, VCC_LO,
};

use xenora_isa::meta::EncodingFamily;
use xenora_isa::semantics::{FORM_IDS, OperandRule};
pub use xenora_isa::semantics::{Signedness, VectorBinary, VectorLowering, VectorUnary, VectorVcc};

/// Metadata forms that [`decode`] turns into an [`Instruction`] for at least one
/// operand combination: the hand-written forms below followed by every form of
/// the semantic registry ([`xenora_isa::semantics::FORM_IDS`]). Operand
/// restrictions are listed in docs/status.toml; every other form in
/// [`xenora_isa::meta::FORMS`] is rejected as unsupported.
pub const SEMANTIC_FORMS: [(&str, EncodingFamily); HANDWRITTEN_FORMS.len() + FORM_IDS.len()] =
    concat_forms();

const fn concat_forms<const N: usize>() -> [(&'static str, EncodingFamily); N] {
    let mut forms = [("", EncodingFamily::Sop2); N];
    let mut index = 0;
    while index < N {
        forms[index] = if index < HANDWRITTEN_FORMS.len() {
            HANDWRITTEN_FORMS[index]
        } else {
            FORM_IDS[index - HANDWRITTEN_FORMS.len()]
        };
        index += 1;
    }
    forms
}

/// Decoded forms not yet described by the semantic registry.
const HANDWRITTEN_FORMS: [(&str, EncodingFamily); 10] = [
    ("BUFFER_LOAD_DWORD", EncodingFamily::Mubuf),
    ("BUFFER_STORE_DWORD", EncodingFamily::Mubuf),
    ("EXP", EncodingFamily::Exp),
    ("S_BUFFER_LOAD_DWORD", EncodingFamily::Smem),
    ("S_BUFFER_LOAD_DWORDX2", EncodingFamily::Smem),
    ("S_BUFFER_LOAD_DWORDX4", EncodingFamily::Smem),
    ("S_WAITCNT_VSCNT", EncodingFamily::Sopk),
    ("V_INTERP_P1_F32", EncodingFamily::Vintrp),
    ("V_INTERP_P2_F32", EncodingFamily::Vintrp),
    ("V_MOV_B32", EncodingFamily::Vop1),
];

/// A successfully decoded instruction and its location in the input stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedInstruction {
    /// Byte offset from the start of the stream.
    pub offset: usize,
    /// Total size in bytes, including any literal word.
    pub byte_len: u8,
    /// The decoded instruction and operand fields.
    pub instruction: Instruction,
}

impl DecodedInstruction {
    /// Destination byte offset of a branch, from the start of the stream.
    #[must_use]
    pub fn branch_target(&self) -> Option<usize> {
        match self.instruction {
            Instruction::Branch { target }
            | Instruction::BranchSccZero { target }
            | Instruction::BranchExecZero { target }
            | Instruction::BranchVccZero { target } => Some(target),
            _ => None,
        }
    }
}

impl fmt::Display for DecodedInstruction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:04x}: {}", self.offset, self.instruction)
    }
}

/// Decoder failures classified for callers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeErrorKind {
    /// An instruction, second word or literal extends past the available input.
    Truncated {
        /// Number of bytes required at the instruction start.
        needed: u8,
        /// Number of bytes available at the instruction start.
        available: usize,
    },
    /// No supported opcode pattern matches the first word.
    UnsupportedOpcode {
        /// First instruction word in host integer order.
        word: u32,
    },
    /// A supported opcode uses an operand or modifier outside the subset.
    UnsupportedForm {
        /// Opcode mnemonic.
        mnemonic: &'static str,
        /// Details of the unsupported or reserved fields.
        detail: &'static str,
    },
    /// A forward branch leaves the stream or lands inside an instruction.
    InvalidBranchTarget {
        /// Destination byte offset from the start of the stream.
        target: usize,
    },
}

/// A decoder error and the offset of the offending instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError {
    /// Byte offset from the start of the stream.
    pub offset: usize,
    /// Decoder-specific failure.
    pub kind: DecodeErrorKind,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "gfx1030 decode at byte 0x{:x}: ", self.offset)?;
        match self.kind {
            DecodeErrorKind::Truncated { needed, available } => {
                write!(
                    formatter,
                    "instruction requires {needed} bytes, found {available}"
                )
            }
            DecodeErrorKind::UnsupportedOpcode { word } => {
                write!(formatter, "unsupported opcode in 0x{word:08x}")
            }
            DecodeErrorKind::UnsupportedForm { mnemonic, detail } => {
                write!(formatter, "unsupported {mnemonic} form: {detail}")
            }
            DecodeErrorKind::InvalidBranchTarget { target } => {
                write!(
                    formatter,
                    "branch target 0x{target:x} is outside instruction boundaries"
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decodes a stream of instructions. Program termination, memory hazards and
/// register initialization are checked by later stages.
///
/// # Errors
///
/// Returns the failing byte offset for a truncated instruction, unsupported
/// opcode, unsupported form or invalid branch target. An empty stream
/// decodes to an empty list.
pub fn decode(
    architecture: Architecture,
    bytes: &[u8],
) -> Result<Vec<DecodedInstruction>, DecodeError> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let decoded = decode_at(architecture, bytes, offset)?;
        offset += usize::from(decoded.byte_len);
        result.push(decoded);
    }
    for decoded in &result {
        if let Some(target) = decoded.branch_target()
            && result
                .binary_search_by_key(&target, |item| item.offset)
                .is_err()
        {
            return Err(DecodeError {
                offset: decoded.offset,
                kind: DecodeErrorKind::InvalidBranchTarget { target },
            });
        }
    }
    Ok(result)
}

/// Decodes the one instruction that starts at `offset` in `bytes`. Branch
/// targets are stream offsets, but whether they land on an instruction
/// boundary is left to the caller, which knows the rest of the stream.
///
/// # Errors
///
/// Returns `offset` with a truncated instruction, unsupported opcode or
/// unsupported form. An `offset` at or past the end is truncated.
pub fn decode_at(
    architecture: Architecture,
    bytes: &[u8],
    offset: usize,
) -> Result<DecodedInstruction, DecodeError> {
    let decode_instruction = match architecture {
        Architecture::Gfx1030 => decode_gfx1030,
    };
    let rest = bytes.get(offset..).unwrap_or_default();
    let (instruction, byte_len) =
        decode_instruction(rest, offset).map_err(|kind| DecodeError { offset, kind })?;
    Ok(DecodedInstruction {
        offset,
        byte_len,
        instruction,
    })
}

fn word(bytes: &[u8], end: u8) -> Result<u32, DecodeErrorKind> {
    let end_index = usize::from(end);
    let Some(word) = bytes
        .get(end_index - 4..end_index)
        .and_then(|word| <[u8; 4]>::try_from(word).ok())
    else {
        return Err(DecodeErrorKind::Truncated {
            needed: end,
            available: bytes.len(),
        });
    };
    Ok(u32::from_le_bytes(word))
}

fn unsupported(mnemonic: &'static str, detail: &'static str) -> DecodeErrorKind {
    DecodeErrorKind::UnsupportedForm { mnemonic, detail }
}

fn decode_gfx1030(bytes: &[u8], offset: usize) -> Result<(Instruction, u8), DecodeErrorKind> {
    let first = word(bytes, 4)?;

    if BUFFER_LOAD_DWORD.matches(first) {
        let len = BUFFER_LOAD_DWORD.byte_len();
        let second = word(bytes, len)?;
        let (data, index, resource) = buffer_operands(first, second, "buffer_load_dword")?;
        return Ok((
            Instruction::BufferLoadDword {
                data,
                index,
                resource,
            },
            len,
        ));
    }
    if BUFFER_STORE_DWORD.matches(first) {
        let len = BUFFER_STORE_DWORD.byte_len();
        let second = word(bytes, len)?;
        let (data, index, resource) = buffer_operands(first, second, "buffer_store_dword")?;
        return Ok((
            Instruction::BufferStoreDword {
                data,
                index,
                resource,
            },
            len,
        ));
    }
    if EXP.matches(first) {
        let len = EXP.byte_len();
        let sources = word(bytes, len)?.to_le_bytes();
        // RDNA 2 ISA Table 56. Match all operand and reserved bits, including
        // the four enable bits, compression, DONE and valid-mask flags.
        let target = match first & 0x03ff_ffff {
            0x08cf => ExportTarget::Position0,
            0x020f => ExportTarget::Parameter0,
            0x180f => ExportTarget::RenderTarget0,
            _ => {
                return Err(unsupported(
                    "exp",
                    "expected full uncompressed POS0 done, PARAM0, or MRT0 done vm",
                ));
            }
        };
        return Ok((Instruction::Export { target, sources }, len));
    }
    if let Some(op) = ImageSampleOp::from_word(first) {
        // Non-NSA MIMG is two dwords; the form never takes a literal.
        let len = 8;
        let second = word(bytes, len)?;
        let (dst, coord, resource, sampler, dmask) = image_sample_operands(first, second, op)?;
        return Ok((
            match op {
                ImageSampleOp::ImageSampleLz => Instruction::ImageSampleLz {
                    dst,
                    coord,
                    resource,
                    sampler,
                    dmask,
                },
                ImageSampleOp::ImageSampleL => Instruction::ImageSampleL {
                    dst,
                    coord,
                    resource,
                    sampler,
                    dmask,
                },
            },
            len,
        ));
    }
    for (opcode, phase) in [
        (V_INTERP_P1_F32, InterpolationPhase::First),
        (V_INTERP_P2_F32, InterpolationPhase::Second),
    ] {
        if opcode.matches(first) {
            return Ok((
                Instruction::Interpolate {
                    phase,
                    dst: (first >> 18) as u8,
                    coordinate: first as u8,
                    attribute: ((first >> 10) & 0x3f) as u8,
                    component: ((first >> 8) & 3) as u8,
                },
                opcode.byte_len(),
            ));
        }
    }
    if V_MOV_B32.matches(first) {
        return move_b32(first, bytes);
    }
    // AMD RDNA 2 ISA guide 13.3.2: VDST occupies bits 17–24 and SRC0 bits 0–8.
    if let Some(op) = VectorUnaryOp::from_word(first) {
        let dst = (first >> 17) as u8;
        // The guide does not define how a binary32 rounding form reads an
        // integer inline constant (selectors 128-208), so they are refused.
        if op.semantics().operands == OperandRule::Vop1F32 && matches!(first & 0x1ff, 128..=208) {
            return Err(unsupported(
                op.assembly_name(),
                "expected a VGPR, normal SGPR, vcc_lo or literal binary32 source",
            ));
        }
        let (src, len) = vector_src0(first, bytes, op.assembly_name())?;
        return Ok((Instruction::VectorUnary { op, dst, src }, len));
    }
    if let Some(op) = VectorSwapOp::from_word(first) {
        if first & 0x100 == 0 {
            return Err(unsupported(
                op.assembly_name(),
                "expected a VGPR SRC0 for writeback",
            ));
        }
        let dst = (first >> 17) as u8;
        let src = first as u8;
        return Ok((Instruction::VectorSwap { op, dst, src }, 4));
    }
    if let Some(op) = VectorCompareOp::from_word(first) {
        return vector_compare(op, first, bytes);
    }
    if let Some(op) = VectorCompareVop3Op::from_word(first) {
        let second = word(bytes, 8)?;
        let sdst = first as u8;
        if sdst > VCC_LO {
            return Err(unsupported(
                op.assembly_name(),
                "expected a destination of s0-s105 or vcc_lo",
            ));
        }
        let (src0, src1) = vop3_vgpr2_sources(op.assembly_name(), first, second)?;
        return Ok((
            Instruction::VectorCompareVop3 {
                op,
                sdst,
                src0,
                src1,
            },
            8,
        ));
    }
    if let Some(op) = VectorAluVop3Op::from_word(first) {
        let second = word(bytes, 8)?;
        return Ok((vop3_vgpr2(op, first, second)?, 8));
    }
    if let Some(op) = ScalarUnaryOp::from_word(first) {
        // SOP1: SSRC0 in bits 0-7, SDST in bits 16-22.
        let mnemonic = op.mnemonic();
        let dst = ((first >> 16) & 0x7f) as u8;
        match op.semantics().operands {
            OperandRule::Sop1Move if !matches!(dst, 0..=107 | 126..=127) => {
                return Err(unsupported(
                    mnemonic,
                    "expected a normal SGPR or EXEC/VCC destination",
                ));
            }
            OperandRule::Sop1Sgpr if dst > 105 => {
                return Err(unsupported(mnemonic, "expected a normal SGPR destination"));
            }
            OperandRule::Sop1SaveExec if dst > 105 => {
                // Guide 5.3, printed page 33: an out-of-range destination leaves
                // no SGPR written, but SCC and EXEC are still written. The form
                // is refused here rather than executed with a discarded result.
                return Err(unsupported(mnemonic, "expected a normal SGPR destination"));
            }
            _ => {}
        }
        let (src, len) = scalar_source(first as u8, bytes, mnemonic)?;
        return Ok((Instruction::ScalarUnary { op, dst, src }, len));
    }
    if let Some(op) = ScalarCompareOp::from_word(first) {
        let mnemonic = op.mnemonic();
        let src0_bits = first as u8;
        let src1_bits = (first >> 8) as u8;
        if src0_bits == 255 && src1_bits == 255 {
            return Err(unsupported(
                mnemonic,
                "only one source may select a literal",
            ));
        }
        let (src0, len0) = scalar_source(src0_bits, bytes, mnemonic)?;
        let (src1, len1) = scalar_source(src1_bits, bytes, mnemonic)?;
        return Ok((
            Instruction::ScalarCompare { op, src0, src1 },
            len0.max(len1),
        ));
    }
    if let Some(op) = ScalarImmediateOp::from_word(first) {
        // SOPK: SIMM16 in bits 0-15, SDST in bits 16-22; the word has no literal.
        let dst = ((first >> 16) & 0x7f) as u8;
        if dst > 105 {
            return Err(unsupported(
                op.mnemonic(),
                "expected a normal SGPR destination",
            ));
        }
        let simm16 = first as u16;
        return Ok((Instruction::ScalarImmediate { op, dst, simm16 }, 4));
    }
    if let Some(op) = ScalarImmediateCompareOp::from_word(first) {
        let src = ((first >> 16) & 0x7f) as u8;
        if src > 105 {
            return Err(unsupported(op.mnemonic(), "expected a normal SGPR operand"));
        }
        let simm16 = first as u16;
        return Ok((Instruction::ScalarImmediateCompare { op, src, simm16 }, 4));
    }
    if first == S_WAITCNT_LGKMCNT_0 {
        return Ok((Instruction::WaitScalarLoad, 4));
    }
    if let Some(dwords) = scalar_buffer_load_dwords(first) {
        let second = word(bytes, 8)?;
        return Ok((scalar_buffer_load(first, second, dwords)?, 8));
    }
    if let Some(op) = SoppControlOp::from_word(first) {
        let immediate = first as u16;
        if op
            .required_immediate()
            .is_some_and(|exact| immediate != exact)
        {
            let detail = if op == SoppControlOp::EndProgram {
                "expected zero immediate"
            } else {
                "expected vmcnt(0) alone"
            };
            return Err(unsupported(op.mnemonic(), detail));
        }
        let instruction = op.decode(|| branch_target(immediate, offset, op.mnemonic()))?;
        return Ok((instruction, 4));
    }
    if let Some(op) = ScalarAluOp::from_word(first) {
        return scalar_alu(op, first, bytes);
    }
    if let Some(op) = VectorAluOp::from_word(first) {
        // VOP2: SRC0 in bits 0-8, VSRC1 in bits 9-16, VDST in bits 17-24.
        let dst = ((first >> 17) & 0xff) as u8;
        let src1 = ((first >> 9) & 0xff) as u8;
        let (src0, len) = match op.operand_rule() {
            // Operand rule `vop2_vgpr2`: SRC0 must select a VGPR (256-511).
            OperandRule::Vop2Vgpr2 => {
                let (_, src0, _) = vop2_registers(first, op.assembly_name())?;
                (VectorSource::Register(src0), 4)
            }
            OperandRule::Vop2Vgpr | OperandRule::Vop2Vcc => {
                vector_src0(first, bytes, op.assembly_name())?
            }
            // Operand rule `vop2_f32`: the guide does not define how a binary32
            // form reads an integer inline constant (selectors 128-208).
            OperandRule::Vop2F32 if matches!(first & 0x1ff, 128..=208) => {
                return Err(unsupported(
                    op.assembly_name(),
                    "expected a VGPR, normal SGPR, vcc_lo or literal binary32 source",
                ));
            }
            OperandRule::Vop2F32 => vector_src0(first, bytes, op.assembly_name())?,
            _ => unreachable!("non-VOP2 operand rule for VOP2 instruction"),
        };
        return Ok((
            Instruction::VectorAlu {
                op,
                dst,
                src0,
                src1,
            },
            len,
        ));
    }
    if S_WAITCNT_VSCNT.matches(first) {
        return scalar_form(
            first,
            S_WAITCNT_VSCNT.byte_len(),
            0xbbfd_0000,
            Instruction::WaitStore,
            "expected null, 0",
        );
    }

    Err(DecodeErrorKind::UnsupportedOpcode { word: first })
}

/// SOP2: SSRC0 in bits 0-7, SSRC1 in bits 8-15, SDST in bits 16-22.
/// Other SOP2 opcodes stay unsupported.
fn scalar_alu(
    op: ScalarAluOp,
    first: u32,
    bytes: &[u8],
) -> Result<(Instruction, u8), DecodeErrorKind> {
    let mnemonic = op.mnemonic();
    let dst = ((first >> 16) & 0x7f) as u8;
    if dst > 105 {
        return Err(unsupported(mnemonic, "expected a normal SGPR destination"));
    }
    let src0_bits = first as u8;
    let src1_bits = (first >> 8) as u8;
    if src0_bits == 255 && src1_bits == 255 {
        return Err(unsupported(
            mnemonic,
            "only one source may select a literal",
        ));
    }
    let (src0, len0) = scalar_source(src0_bits, bytes, mnemonic)?;
    let (src1, len1) = scalar_source(src1_bits, bytes, mnemonic)?;
    Ok((
        Instruction::ScalarAlu {
            op,
            dst,
            src0,
            src1,
        },
        len0.max(len1),
    ))
}

fn scalar_source(
    bits: u8,
    bytes: &[u8],
    mnemonic: &'static str,
) -> Result<(ScalarSource, u8), DecodeErrorKind> {
    let value = u32::from(bits);
    match bits {
        0..=105 => Ok((ScalarSource::Register(bits), 4)),
        128..=192 => Ok((ScalarSource::Immediate(value - 128), 4)),
        193..=208 => Ok((ScalarSource::Immediate(192u32.wrapping_sub(value)), 4)),
        255 => Ok((ScalarSource::Immediate(word(bytes, 8)?), 8)),
        _ => Err(unsupported(
            mnemonic,
            "expected a normal SGPR, inline integer or literal",
        )),
    }
}

fn branch_target(
    immediate: u16,
    offset: usize,
    mnemonic: &'static str,
) -> Result<usize, DecodeErrorKind> {
    let displacement = immediate as i16;
    let words = usize::try_from(displacement)
        .map_err(|_| unsupported(mnemonic, "expected a forward branch"))?;
    offset
        .checked_add(4 + words * 4)
        .ok_or_else(|| unsupported(mnemonic, "target byte offset overflowed"))
}

fn vop2_registers(first: u32, mnemonic: &'static str) -> Result<(u8, u8, u8), DecodeErrorKind> {
    if first & 0x100 == 0 {
        return Err(unsupported(
            mnemonic,
            "expected two unmodified VGPR sources",
        ));
    }
    Ok(((first >> 17) as u8, first as u8, (first >> 9) as u8))
}

fn image_sample_operands(
    first: u32,
    second: u32,
    op: ImageSampleOp,
) -> Result<(u8, u8, u8, u8, u8), DecodeErrorKind> {
    let mnemonic = op.mnemonic();
    // The generated `from_word` dispatch matches the encoding and opcode; the
    // bounded 2D scope is enforced here. Fixed bits: NSA 0 (1-2), DIM 1 = 2D
    // (3-5), reserved bit 6 and DLC (7) clear, UNORM/GLC/DA/R128/TFE/LWE clear
    // (12-17) and SLC clear (25). The data mask (8-11) and opcode stay variable.
    const MIMG_FIXED_MASK: u32 = 0x0203_f0fe;
    const MIMG_FIXED_2D: u32 = 0x0000_0008;
    if first & MIMG_FIXED_MASK != MIMG_FIXED_2D {
        return Err(unsupported(
            mnemonic,
            "expected dim:SQ_RSRC_IMG_2D with zero modifier flags",
        ));
    }
    let dmask = ((first >> 8) & 0xf) as u8;
    if dmask == 0 {
        return Err(unsupported(mnemonic, "expected a nonzero data mask"));
    }
    // In word 1: bits 0..7 are VADDR, 8..15 VDATA, 16..20 SRSRC (>> 2) and
    // 21..25 SSAMP (>> 2). The reserved bits 26..29 and the A16/D16 flags must
    // be clear.
    if second & 0xfc00_0000 != 0 {
        return Err(unsupported(
            mnemonic,
            "expected 32-bit coordinates and data without A16/D16",
        ));
    }

    let coord = (second & 0xff) as u8;
    let dst = ((second >> 8) & 0xff) as u8;
    let resource = (((second >> 16) & 0x1f) * 4) as u8;
    let sampler = (((second >> 21) & 0x1f) * 4) as u8;

    // One destination VGPR per DMASK-selected component.
    let components = u16::from(dmask.count_ones() as u8);
    if u16::from(dst) + components - 1 > 255 {
        return Err(unsupported(mnemonic, "destination span exceeds v255"));
    }
    // The address tuple is (u, v) for LZ and (u, v, lod) for L.
    let address_components: u16 = match op {
        ImageSampleOp::ImageSampleLz => 2,
        ImageSampleOp::ImageSampleL => 3,
    };
    if u16::from(coord) + address_components - 1 > 255 {
        return Err(unsupported(mnemonic, "coordinate tuple exceeds v255"));
    }

    // The RDNA 2 resource descriptor (T#) spans eight SGPRs and the sampler
    // (S#) four, both four-aligned (guide 8.2, printed page 69); the range is
    // s[0:105] for normal allocatable SGPRs.
    if resource > 98 {
        return Err(unsupported(
            mnemonic,
            "resource descriptor must fit in normal SGPRs s[0:105]",
        ));
    }
    if sampler > 102 {
        return Err(unsupported(
            mnemonic,
            "sampler descriptor must fit in normal SGPRs s[0:105]",
        ));
    }

    Ok((dst, coord, resource, sampler, dmask))
}

/// SOPP `s_waitcnt lgkmcnt(0)` alone: the other counters at their maximum.
const S_WAITCNT_LGKMCNT_0: u32 = 0xbf8c_c07f;

/// Words loaded by an SMEM `s_buffer_load_dword`, `_x2` or `_x4` (guide Tables
/// 73 and 74: encoding in bits 26-31, opcode in bits 18-25).
fn scalar_buffer_load_dwords(first: u32) -> Option<u8> {
    match first >> 18 {
        0x3d08 => Some(1),
        0x3d09 => Some(2),
        0x3d0a => Some(4),
        _ => None,
    }
}

/// Guide 7.1 and 7.5: SBASE names an SGPR pair and must be even, SDATA is
/// aligned to the number of words, and the offset must be positive.
fn scalar_buffer_load(first: u32, second: u32, dwords: u8) -> Result<Instruction, DecodeErrorKind> {
    const NULL: u8 = 125;
    let mnemonic = match dwords {
        1 => "s_buffer_load_dword",
        2 => "s_buffer_load_dwordx2",
        _ => "s_buffer_load_dwordx4",
    };
    // Bits 13-17 hold DLC, GLC and reserved bits; 21-24 of the second word are
    // reserved.
    if first & 0x0003_e000 != 0 || second & 0x01e0_0000 != 0 {
        return Err(unsupported(
            mnemonic,
            "expected cache policy and reserved bits to be zero",
        ));
    }
    let base = (first & 0x3f) as u8;
    let resource = base * 2;
    if !base.is_multiple_of(2) || resource > 102 {
        return Err(unsupported(
            mnemonic,
            "resource descriptor must be four aligned normal SGPRs",
        ));
    }
    let dst = ((first >> 6) & 0x7f) as u8;
    if !dst.is_multiple_of(dwords) || dst + dwords > 106 {
        return Err(unsupported(
            mnemonic,
            "expected an aligned destination in normal SGPRs",
        ));
    }
    let offset = second & 0x001f_ffff;
    if offset >> 20 != 0 {
        return Err(unsupported(
            mnemonic,
            "a buffer load needs a positive offset",
        ));
    }
    let soffset = match (second >> 25) as u8 {
        NULL => None,
        register @ 0..=105 => Some(register),
        _ => {
            return Err(unsupported(
                mnemonic,
                "expected a normal SGPR offset or null",
            ));
        }
    };
    Ok(Instruction::ScalarBufferLoad {
        dst,
        resource,
        offset,
        soffset,
        dwords,
    })
}

fn buffer_operands(
    first: u32,
    second: u32,
    mnemonic: &'static str,
) -> Result<(u8, u8, u8), DecodeErrorKind> {
    // OFFSET, OFFEN, IDXEN, GLC, DLC, LDS and the reserved bit before OP.
    const ADDRESS_MODE_MASK: u32 = 0x0003_ffff;
    const IDXEN: u32 = 1 << 13;

    // SOFFSET is inline zero. TFE, SLC and the reserved high operand bit are zero.
    if first & ADDRESS_MODE_MASK != IDXEN || second & 0xffe0_0000 != 0x8000_0000 {
        return Err(unsupported(
            mnemonic,
            "expected idxen alone with zero offsets",
        ));
    }

    let index = second as u8;
    let data = (second >> 8) as u8;
    let resource = ((second >> 16) & 0x1f) as u8 * 4;
    // AMD RDNA 2 ISA §3.6.2. A resource descriptor requires four normal SGPRs.
    if resource > 102 {
        return Err(unsupported(
            mnemonic,
            "resource descriptor must fit in normal SGPRs",
        ));
    }
    Ok((data, index, resource))
}

/// VOPC: SRC0 in bits 0-8, VSRC1 in bits 9-16 (RDNA 2 ISA section 12.9). The
/// destination is implicitly `vcc_lo`. SRC0 excludes inline float constants,
/// `vcc_hi`, EXEC, M0, TTMPs and the condition selectors (SDWA 249, DPP 250/233).
fn vector_compare(
    op: VectorCompareOp,
    first: u32,
    bytes: &[u8],
) -> Result<(Instruction, u8), DecodeErrorKind> {
    let src1 = ((first >> 9) & 0xff) as u8;
    let mnemonic = op.assembly_name();
    if op.semantics().width == 64 {
        // Guide §3.6.3 requires even SGPR bases. Reject invalid selectors
        // before reading a literal word or considering the execution mask.
        let source = first & 0x1ff;
        if src1 == 255
            || !matches!(source, 0..=104 | 128..=208 | 256..=510)
            || (source <= 104 && !source.is_multiple_of(2))
        {
            return Err(unsupported(mnemonic, "invalid 64-bit compare source"));
        }
    }
    if op.semantics().width == 16 && first & 0x1ff == 255 {
        return Err(unsupported(mnemonic, "literal in 16-bit compare"));
    }
    let (src0, len) = vector_src0(first, bytes, mnemonic)?;
    Ok((Instruction::VectorCompare { op, src0, src1 }, len))
}

/// The 9-bit SRC0 of a VOP1, VOP2 or VOPC word: a VGPR, `s0-s105`, `vcc_lo`, an
/// inline integer -16..64 or a literal word; everything else is unsupported.
fn vector_src0(
    first: u32,
    bytes: &[u8],
    mnemonic: &'static str,
) -> Result<(VectorSource, u8), DecodeErrorKind> {
    let source = first & 0x1ff;
    let base_len = 4;
    Ok(match source {
        0..=106 => (VectorSource::Scalar(source as u8), base_len),
        128..=192 => (VectorSource::Immediate(source - 128), base_len),
        193..=208 => (
            VectorSource::Immediate(192u32.wrapping_sub(source)),
            base_len,
        ),
        255 => {
            let len = base_len + 4;
            (VectorSource::Immediate(word(bytes, len)?), len)
        }
        256..=511 => (VectorSource::Register(source as u8), base_len),
        _ => {
            return Err(unsupported(
                mnemonic,
                "expected a VGPR, normal SGPR, vcc_lo, inline integer or literal",
            ));
        }
    })
}

fn move_b32(first: u32, bytes: &[u8]) -> Result<(Instruction, u8), DecodeErrorKind> {
    let dst = (first >> 17) as u8;
    let source = first & 0x1ff;
    let base_len = V_MOV_B32.byte_len();
    let (src, len) = match source {
        0..=105 => (VectorSource::Scalar(source as u8), base_len),
        // vcc_lo, which may hold a lane-derived compare mask (docs/shaders/vector-compare.md).
        106 => (VectorSource::Scalar(VCC_LO), base_len),
        128..=192 => (VectorSource::Immediate(source - 128), base_len),
        193..=208 => (
            VectorSource::Immediate(192u32.wrapping_sub(source)),
            base_len,
        ),
        // AMD RDNA 2 ISA Table 77: VOP1 source selector 242 supplies +1.0f.
        242 => (VectorSource::Immediate(0x3f80_0000), base_len),
        255 => {
            let len = base_len + 4;
            (VectorSource::Immediate(word(bytes, len)?), len)
        }
        256..=511 => (VectorSource::Register(source as u8), base_len),
        _ => {
            return Err(unsupported(
                "v_mov_b32_e32",
                "expected a VGPR, normal SGPR, vcc_lo, inline integer, inline +1.0f or literal",
            ));
        }
    };
    Ok((Instruction::MoveB32 { dst, src }, len))
}

fn vop3_vgpr2(
    op: VectorAluVop3Op,
    first: u32,
    second: u32,
) -> Result<Instruction, DecodeErrorKind> {
    let (src0, src1) = vop3_vgpr2_sources(op.assembly_name(), first, second)?;
    Ok(Instruction::VectorAluVop3 {
        op,
        dst: first as u8,
        src0,
        src1,
    })
}

/// Common VOP3A source and modifier checks (RDNA 2 ISA section 13.3.4).
/// SRC0 and SRC1 are VGPRs, the unused SRC2 is inline zero, and ABS, OP_SEL,
/// CLAMP, OMOD and NEG are clear.
fn vop3_vgpr2_sources(
    mnemonic: &'static str,
    first: u32,
    second: u32,
) -> Result<(u8, u8), DecodeErrorKind> {
    let src0 = second & 0x1ff;
    let src1 = (second >> 9) & 0x1ff;
    if first & 0xff00 != 0 || second >> 18 != 128 || src0 < 256 || src1 < 256 {
        return Err(unsupported(
            mnemonic,
            "expected two VGPR sources and no modifiers",
        ));
    }
    Ok((src0 as u8, src1 as u8))
}

fn scalar_form(
    word: u32,
    byte_len: u8,
    expected: u32,
    instruction: Instruction,
    detail: &'static str,
) -> Result<(Instruction, u8), DecodeErrorKind> {
    if word == expected {
        Ok((instruction, byte_len))
    } else {
        Err(unsupported(instruction.mnemonic(), detail))
    }
}
