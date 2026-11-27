//! Typed operands and instruction forms for the bounded gfx1030 compute and graphics subsets.

use std::fmt;

pub mod meta;
pub mod primitive;
mod rdna2;
pub mod semantics;
#[rustfmt::skip]
mod rdna2_meta;
#[rustfmt::skip]
mod semantics_gen;
pub use rdna2::{
    BUFFER_LOAD_DWORD, BUFFER_STORE_DWORD, EXP, IMAGE_SAMPLE_LZ, S_BRANCH, S_CBRANCH_EXECZ,
    S_CBRANCH_SCC0, S_CBRANCH_VCCZ, S_ENDPGM, S_MOV_B32, S_WAITCNT, S_WAITCNT_VSCNT, V_ADD_F32,
    V_ADD_NC_U32, V_AND_B32, V_ASHRREV_I32, V_CMP_EQ_I32, V_CMP_EQ_U32, V_CMP_GE_I32, V_CMP_GE_U32,
    V_CMP_GT_I32, V_CMP_GT_U32, V_CMP_LE_I32, V_CMP_LE_U32, V_CMP_LT_I32, V_CMP_LT_U32,
    V_CMP_NE_I32, V_CMP_NE_U32, V_INTERP_P1_F32, V_INTERP_P2_F32, V_LSHLREV_B32, V_LSHRREV_B32,
    V_MOV_B32, V_MUL_F32, V_OR_B32, V_XOR_B32,
};
pub use semantics_gen::{
    ImageSampleOp, ScalarAluOp, ScalarCompareOp, ScalarImmediateCompareOp, ScalarImmediateOp,
    ScalarUnaryOp, SoppControlOp, VectorAluOp, VectorAluVop3Op, VectorCompareOp,
    VectorCompareVop3Op, VectorSwapOp, VectorUnaryOp,
};

impl VectorAluOp {
    /// Whether this encoding takes a per-lane VCC_LO bit.
    pub fn reads_vcc(self) -> bool {
        self.semantics().carry_in == semantics::CarryIn::Vcc
    }

    /// Whether this encoding replaces the Wave32 VCC_LO mask.
    pub fn writes_vcc(self) -> bool {
        self.semantics().vcc == semantics::MaskEffect::LaneMask
    }
}

/// SGPR index of `vcc_lo`, the Wave32 vector condition code.
pub const VCC_LO: u8 = 106;

/// Selects the instruction encoding independently of the container and execution ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    /// LLVM's gfx1030 target for AMD RDNA 2.
    Gfx1030,
}

/// A generated first-word opcode pattern and its base length, before literal extensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Opcode {
    mask: u32,
    bits: u32,
    bytes: u8,
}

impl Opcode {
    /// Checks the format and opcode bits. The decoder must still validate operands.
    pub fn matches(self, word: u32) -> bool {
        word & self.mask == self.bits
    }

    /// Returns the base instruction size in bytes.
    pub fn byte_len(self) -> u8 {
        self.bytes
    }
}

/// A 32-bit source for an unmodified vector move.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorSource {
    /// VGPR index.
    Register(u8),
    /// Normal SGPR index (s0-s105) or `vcc_lo` (106), read as a wave-uniform value.
    Scalar(u8),
    /// Inline integer or literal word, represented by its bits.
    Immediate(u32),
}

impl fmt::Display for VectorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(register) => write!(formatter, "v{register}"),
            Self::Scalar(VCC_LO) => formatter.write_str("vcc_lo"),
            Self::Scalar(register) => write!(formatter, "s{register}"),
            Self::Immediate(value) => write!(formatter, "0x{value:x}"),
        }
    }
}

/// A scalar 32-bit source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarSource {
    /// A normal SGPR index.
    Register(u8),
    /// Inline integer or literal word, represented by its bits.
    Immediate(u32),
}

impl fmt::Display for ScalarSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Register(register) => write!(formatter, "s{register}"),
            Self::Immediate(value) => match value as i32 {
                integer @ -16..=64 => write!(formatter, "{integer}"),
                _ => write!(formatter, "0x{value:x}"),
            },
        }
    }
}

/// Relation tested by a SOPC or VOPC integer compare; S0 (SSRC0 or SRC0) is on
/// the left.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareRelation {
    /// S0 < S1 (opcode offset 1).
    Less,
    /// S0 == S1 (offset 2).
    Equal,
    /// S0 <= S1 (offset 3).
    LessOrEqual,
    /// S0 > S1 (offset 4).
    Greater,
    /// S0 != S1 (offset 5).
    NotEqual,
    /// S0 >= S1 (offset 6).
    GreaterOrEqual,
    /// Never holds (VOPC F, offset 0).
    False,
    /// Always holds (VOPC T, offset 7).
    True,
}

/// Output targeted by a supported uncompressed four-component export.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExportTarget {
    /// Vertex clip position.
    Position0,
    /// Vertex parameter slot zero.
    Parameter0,
    /// Pixel color attachment zero.
    RenderTarget0,
}

/// Phase of a pixel parameter interpolation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpolationPhase {
    /// Initialize the destination with the base and I-coordinate terms.
    First,
    /// Add the J-coordinate term to the destination.
    Second,
}

/// Instruction forms accepted by the gfx1030 decoder.
///
/// Vector operations use VGPR indices and scalar operations use SGPR indices.
/// The buffer `resource` is the first SGPR of its four-word descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum Instruction {
    /// Indexed 32-bit buffer read into a VGPR.
    BufferLoadDword { data: u8, index: u8, resource: u8 },
    /// Indexed 32-bit buffer write from a VGPR.
    BufferStoreDword { data: u8, index: u8, resource: u8 },
    /// `s_waitcnt vmcnt(0)`.
    WaitLoad,
    /// SMEM `s_buffer_load_dword`, `_x2` and `_x4`: `dwords` consecutive words
    /// into SGPRs from `dst`, at a byte offset from a buffer descriptor.
    ScalarBufferLoad {
        dst: u8,
        resource: u8,
        /// Unsigned 20-bit immediate byte offset.
        offset: u32,
        /// SGPR holding a further unsigned byte offset; `None` encodes `null`.
        soffset: Option<u8>,
        dwords: u8,
    },
    /// `s_waitcnt lgkmcnt(0)`.
    WaitScalarLoad,
    /// `s_waitcnt_vscnt null, 0`.
    WaitStore,
    /// VOP1 unmodified 32-bit vector move.
    MoveB32 { dst: u8, src: VectorSource },
    /// VOP1 unary integer operation with a single VGPR result.
    VectorUnary {
        op: VectorUnaryOp,
        dst: u8,
        src: VectorSource,
    },
    /// VOP1 register swap. Both operands are inputs and outputs.
    VectorSwap { op: VectorSwapOp, dst: u8, src: u8 },
    /// VOP3 encoding of a registry VOP2 integer operation (operand rule
    /// `vop3_vgpr2`): two VGPR sources and no input or output modifiers.
    VectorAluVop3 {
        op: VectorAluVop3Op,
        dst: u8,
        src0: u8,
        src1: u8,
    },
    /// SOP1 32-bit unary integer operation. The move also allows an explicit
    /// VCC or EXEC half destination. Some unary operations write SCC.
    ScalarUnary {
        op: ScalarUnaryOp,
        dst: u8,
        src: ScalarSource,
    },
    /// SOP2 32-bit integer operation writing a normal SGPR (s0-s105) and, for
    /// most operations, SCC; `s_cselect_b32` also reads SCC.
    ScalarAlu {
        op: ScalarAluOp,
        dst: u8,
        src0: ScalarSource,
        src1: ScalarSource,
    },
    /// SOPC 32-bit integer compare writing only SCC.
    ScalarCompare {
        op: ScalarCompareOp,
        src0: ScalarSource,
        src1: ScalarSource,
    },
    /// SOPK 32-bit integer operation on SDST (s0-s105, never s0-s7) and the
    /// 16-bit immediate `simm16`; `s_addk_i32` also writes SCC and
    /// `s_cmovk_i32` reads it.
    ScalarImmediate {
        op: ScalarImmediateOp,
        dst: u8,
        simm16: u16,
    },
    /// SOPK 32-bit integer compare of `src` (s0-s105) with the 16-bit
    /// immediate, writing only SCC.
    ScalarImmediateCompare {
        op: ScalarImmediateCompareOp,
        src: u8,
        simm16: u16,
    },
    /// VOPC integer compare writing a Wave32 mask to `vcc_lo`.
    /// For 64-bit forms register operands name pair bases and the immediate
    /// contains the low word of a sign-extended inline integer.
    VectorCompare {
        op: VectorCompareOp,
        src0: VectorSource,
        src1: u8,
    },
    /// VOP3A 32-bit integer compare writing a Wave32 mask to s0-s105 or `vcc_lo`.
    /// Both sources are VGPR indices.
    VectorCompareVop3 {
        op: VectorCompareVop3Op,
        sdst: u8,
        src0: u8,
        src1: u8,
    },
    /// VOP2 32-bit per-lane integer operation from the semantic registry.
    /// `src0` is a VGPR, `s0-s105`, `vcc_lo`, an inline integer or a literal;
    /// `src1` is a VGPR. Inactive lanes keep `dst`.
    VectorAlu {
        op: VectorAluOp,
        dst: u8,
        src0: VectorSource,
        src1: u8,
    },
    /// SOPP jump to a stream byte offset when SCC is clear.
    BranchSccZero { target: usize },
    /// SOPP jump when the Wave32 execution mask is empty.
    BranchExecZero { target: usize },
    /// SOPP jump when the Wave32 vector condition mask is empty.
    BranchVccZero { target: usize },
    /// SOPP unconditional jump to a stream byte offset.
    Branch { target: usize },
    /// Full, uncompressed four-component graphics export.
    Export {
        target: ExportTarget,
        sources: [u8; 4],
    },
    /// Parameter interpolation with a source barycentric coordinate and an attribute component.
    Interpolate {
        phase: InterpolationPhase,
        dst: u8,
        coordinate: u8,
        attribute: u8,
        component: u8,
    },
    /// MIMG image_sample_lz 2D RGBA8 texture sample with level of detail zero.
    ImageSampleLz {
        /// First destination VGPR of the DMASK-selected components.
        dst: u8,
        /// Two contiguous coordinate VGPRs [coord..coord+1] containing (u, v).
        coord: u8,
        /// First SGPR of the eight-word resource descriptor s[resource..resource+7].
        resource: u8,
        /// First SGPR of the four-word sampler descriptor s[sampler..sampler+3].
        sampler: u8,
        /// Data mask (guide Table 41, printed page 69): bit 0 = red, bit 1 =
        /// green, bit 2 = blue, bit 3 = alpha. The selected components are
        /// written to consecutive VGPRs starting at `dst`.
        dmask: u8,
    },
    /// MIMG image_sample_l 2D RGBA8 texture sample with an explicit level of
    /// detail read from the address tuple.
    ImageSampleL {
        /// First destination VGPR of the DMASK-selected components.
        dst: u8,
        /// Three contiguous coordinate VGPRs [coord..coord+2] containing
        /// (u, v, lod).
        coord: u8,
        /// First SGPR of the eight-word resource descriptor s[resource..resource+7].
        resource: u8,
        /// First SGPR of the four-word sampler descriptor s[sampler..sampler+3].
        sampler: u8,
        /// Data mask as for [`Self::ImageSampleLz`].
        dmask: u8,
    },
    /// Terminates the wave and waits for outstanding memory operations.
    EndProgram,
}

/// Formats an SGPR destination with LLVM's names for the EXEC and VCC halves.
struct ScalarDestination(u8);

impl fmt::Display for ScalarDestination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            106 => formatter.write_str("vcc_lo"),
            107 => formatter.write_str("vcc_hi"),
            126 => formatter.write_str("exec_lo"),
            127 => formatter.write_str("exec_hi"),
            register => write!(formatter, "s{register}"),
        }
    }
}

impl Instruction {
    /// Returns the mnemonic for the exact encoding family represented.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::BufferLoadDword { .. } => "buffer_load_dword",
            Self::BufferStoreDword { .. } => "buffer_store_dword",
            Self::WaitLoad | Self::WaitScalarLoad => "s_waitcnt",
            Self::ScalarBufferLoad { dwords: 1, .. } => "s_buffer_load_dword",
            Self::ScalarBufferLoad { dwords: 2, .. } => "s_buffer_load_dwordx2",
            Self::ScalarBufferLoad { .. } => "s_buffer_load_dwordx4",
            Self::WaitStore => "s_waitcnt_vscnt",
            Self::MoveB32 { .. } => "v_mov_b32_e32",
            Self::VectorUnary { op, .. } => op.assembly_name(),
            Self::VectorSwap { op, .. } => op.assembly_name(),
            Self::VectorAluVop3 { op, .. } => op.assembly_name(),
            Self::ScalarUnary { op, .. } => op.mnemonic(),
            Self::ScalarAlu { op, .. } => op.mnemonic(),
            Self::ScalarCompare { op, .. } => op.mnemonic(),
            Self::ScalarImmediate { op, .. } => op.mnemonic(),
            Self::ScalarImmediateCompare { op, .. } => op.mnemonic(),
            Self::VectorCompare { op, .. } => op.assembly_name(),
            Self::VectorCompareVop3 { op, .. } => op.assembly_name(),
            Self::VectorAlu { op, .. } => op.assembly_name(),
            Self::BranchSccZero { .. } => "s_cbranch_scc0",
            Self::BranchExecZero { .. } => "s_cbranch_execz",
            Self::BranchVccZero { .. } => "s_cbranch_vccz",
            Self::Branch { .. } => "s_branch",
            Self::Export { .. } => "exp",
            Self::Interpolate {
                phase: InterpolationPhase::First,
                ..
            } => "v_interp_p1_f32_e32",
            Self::Interpolate {
                phase: InterpolationPhase::Second,
                ..
            } => "v_interp_p2_f32_e32",
            Self::ImageSampleLz { .. } => "image_sample_lz",
            Self::ImageSampleL { .. } => "image_sample_l",
            Self::EndProgram => "s_endpgm",
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.mnemonic())?;
        match *self {
            Self::BufferLoadDword {
                data,
                index,
                resource,
            }
            | Self::BufferStoreDword {
                data,
                index,
                resource,
            } => {
                let last = u16::from(resource) + 3;
                write!(
                    formatter,
                    " v{data}, v{index}, s[{resource}:{last}], 0 idxen"
                )
            }
            Self::WaitLoad => formatter.write_str(" vmcnt(0)"),
            Self::WaitScalarLoad => formatter.write_str(" lgkmcnt(0)"),
            Self::ScalarBufferLoad {
                dst,
                resource,
                offset,
                soffset,
                dwords,
            } => {
                let last = u16::from(resource) + 3;
                if dwords == 1 {
                    write!(formatter, " s{dst}")?;
                } else {
                    write!(formatter, " s[{dst}:{}]", u16::from(dst + dwords) - 1)?;
                }
                write!(formatter, ", s[{resource}:{last}], ")?;
                match soffset {
                    None if offset == 0 => formatter.write_str("null"),
                    None => write!(formatter, "{offset:#x}"),
                    Some(register) if offset == 0 => write!(formatter, "s{register}"),
                    Some(register) => write!(formatter, "s{register} offset:{offset:#x}"),
                }
            }
            Self::WaitStore => formatter.write_str(" null, 0"),
            Self::MoveB32 { dst, src } | Self::VectorUnary { dst, src, .. } => {
                write!(formatter, " v{dst}, {src}")
            }
            Self::VectorSwap { dst, src, .. } => write!(formatter, " v{dst}, v{src}"),
            Self::VectorAluVop3 {
                dst, src0, src1, ..
            } => {
                write!(formatter, " v{dst}, v{src0}, v{src1}")
            }
            Self::ScalarUnary { dst, src, .. } => {
                write!(formatter, " {}, {src}", ScalarDestination(dst))
            }
            Self::ScalarAlu {
                dst, src0, src1, ..
            } => write!(formatter, " {}, {src0}, {src1}", ScalarDestination(dst)),
            Self::ScalarCompare { src0, src1, .. } => write!(formatter, " {src0}, {src1}"),
            Self::ScalarImmediate { dst, simm16, .. } => {
                write!(formatter, " s{dst}, {simm16:#x}")
            }
            Self::ScalarImmediateCompare { src, simm16, .. } => {
                write!(formatter, " s{src}, {simm16:#x}")
            }
            Self::VectorCompare { op, src0, src1 } if op.semantics().width == 64 => {
                formatter.write_str(" vcc_lo, ")?;
                match src0 {
                    VectorSource::Register(base) => {
                        write!(formatter, "v[{base}:{}]", u16::from(base) + 1)?;
                    }
                    VectorSource::Scalar(base) => {
                        write!(formatter, "s[{base}:{}]", u16::from(base) + 1)?;
                    }
                    VectorSource::Immediate(bits) => write!(formatter, "{}", bits as i32)?,
                }
                write!(formatter, ", v[{src1}:{}]", u16::from(src1) + 1)
            }
            Self::VectorCompare { src0, src1, .. } => {
                write!(formatter, " vcc_lo, {src0}, v{src1}")
            }
            Self::VectorCompareVop3 {
                sdst, src0, src1, ..
            } => write!(formatter, " {}, v{src0}, v{src1}", ScalarDestination(sdst)),
            Self::VectorAlu {
                op,
                dst,
                src0,
                src1,
            } => {
                write!(formatter, " v{dst}, ")?;
                if op.writes_vcc() {
                    write!(formatter, "vcc_lo, ")?;
                }
                write!(formatter, "{src0}, v{src1}")?;
                if op.reads_vcc() {
                    write!(formatter, ", vcc_lo")?;
                }
                Ok(())
            }
            Self::Branch { target }
            | Self::BranchSccZero { target }
            | Self::BranchExecZero { target }
            | Self::BranchVccZero { target } => {
                write!(formatter, " -> 0x{target:04x}")
            }
            Self::Export {
                target,
                sources: [a, b, c, d],
            } => {
                let (name, flags) = match target {
                    ExportTarget::Position0 => ("pos0", " done"),
                    ExportTarget::Parameter0 => ("param0", ""),
                    ExportTarget::RenderTarget0 => ("mrt0", " done vm"),
                };
                write!(formatter, " {name}, v{a}, v{b}, v{c}, v{d}{flags}")
            }
            Self::Interpolate {
                dst,
                coordinate,
                attribute,
                component,
                ..
            } => {
                let channel = ["x", "y", "z", "w"]
                    .get(usize::from(component))
                    .ok_or(fmt::Error)?;
                write!(
                    formatter,
                    " v{dst}, v{coordinate}, attr{attribute}.{channel}"
                )
            }
            Self::ImageSampleLz {
                dst,
                coord,
                resource,
                sampler,
                dmask,
            }
            | Self::ImageSampleL {
                dst,
                coord,
                resource,
                sampler,
                dmask,
            } => {
                let components = dmask.count_ones() as usize;
                let destination = if components == 1 {
                    format!("v{dst}")
                } else {
                    format!("v[{dst}:{}]", dst + components as u8 - 1)
                };
                let last_coord = coord
                    + if matches!(self, Self::ImageSampleL { .. }) {
                        2
                    } else {
                        1
                    };
                let last_res = resource + 7;
                let last_samp = sampler + 3;
                write!(
                    formatter,
                    " {destination}, v[{coord}:{last_coord}], s[{resource}:{last_res}], s[{sampler}:{last_samp}] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_2D"
                )
            }
            Self::EndProgram => Ok(()),
        }
    }
}
