// SPDX-License-Identifier: GPL-3.0-or-later

//! A second, structurally independent interpreter for the demo instruction
//! subset, used only by the semantic-review tests.
//!
//! This module deliberately imports neither `xenora-decoder`, `xenora-isa` nor
//! `xenora-ref`. It re-decodes the gfx1030 encodings from the AMD RDNA 2 ISA
//! field layout and executes them with an explicit program-counter machine, so
//! a defect in the production decoder or reference cannot cancel out here.
//!
//! Scope matches the demo fixtures: scalar move, eleven SOP1 unary forms
//! (NOT, BREV, BCNT0/1, FF0/1, FLBIT, FLBIT_I32, SEXT_I8/I16, ABS), the
//! twelve SOP1 SAVEEXEC and WREXEC forms (AND, OR, XOR, ANDN2, ORN2, NAND,
//! NOR, XNOR, ANDN1, ORN1, ANDN1_WREXEC and ANDN2_WREXEC), the twelve SOPC 32-bit integer
//! compares, branch, SOP2 unsigned
//! and signed add and subtract, add-with-carry, subtract-with-borrow, AND, OR
//! and XOR, the stage C forms ANDN2, ORN2, NAND, NOR, XNOR, LSHL, LSHR and
//! ASHR (32-bit), the 32-bit integer forms MIN/MAX (I32, U32), CSELECT, BFM,
//! MUL_I32, BFE (U32, I32), ABSDIFF, LSHL1-4_ADD and MUL_HI (U32, I32), VOP1 move
//! (including from an SGPR), the VOP1 integer forms NOT, BFREV, FFBH_U32,
//! FFBL_B32 and FFBH_I32 (any SRC0 a VOPC compare takes; bit scans as the
//! guide's loops over bits from integer division), the binary32 rounding
//! forms TRUNC, CEIL, RNDNE and FLOOR (host binary64 rounding of the exactly
//! widened input; NaN inputs and integer inline constants are refused) and SWAP_B32 (VGPR SRC0),
//! VOP2 bitwise, shift and arithmetic-shift, the VOP2 binary32 forms
//! ADD/SUB/SUBREV/MUL/MIN/MAX_F32 (host binary32 arithmetic; subnormal and
//! NaN inputs, signalling NaNs to MIN/MAX, NaN results and results below
//! 2^-126 are refused; integer inline constants are refused), the VOP2 integer forms
//! MUL_I32_I24, MUL_HI_I32_I24, MUL_U32_U24, MUL_HI_U32_U24, MIN/MAX (I32,
//! U32), XNOR and ADD/SUB/SUBREV_NC_U32 (any SRC0 a VOPC compare takes; a
//! shift-and-add multiplier and floor division for the high half, the ripple
//! adder for add and subtract), VOP3A integer arithmetic and compares, MUBUF
//! load/store, SMEM `s_buffer_load_dword`/`_x2`/`_x4` with
//! `s_waitcnt lgkmcnt(0)`, MIMG 2D `image_sample_lz`/`image_sample_l`, and the
//! two vector waits plus `s_endpgm`. Anything else is a controlled error. The
//! arithmetic shift is evaluated as a floor division by a power of
//! two rather than with a host signed shift, so the oracle does not share the
//! production operator. SOP2 arithmetic runs through a bit-serial ripple
//! adder (subtraction as addition of the complement), not host integer ops.
//!
//! Wave32 VOPC integer compares
//! (`v_cmp_{f,lt,eq,le,gt,ne,ge,t}_{i32,u32,i64,u64}_e32` and
//! `v_cmp_{lt,eq,le,gt,ne,ge}_{i16,u16}_e32`, contract
//! `docs/shaders/vector-compare.md`) need all lanes of a wave, so execution
//! runs lockstep waves of 32 consecutive dispatch indices with wave-uniform
//! scalar state and per-lane vector state. A lane-local failure stops only that
//! lane and the lowest-index error is reported, so programs without compares
//! fail exactly as independent per-lane runs did. Signed compares flip the sign
//! bit of both operands and then compare unsigned. VCC_LO is uninitialized,
//! known or lane-derived; a lane-derived VCC_LO feeds `s_cbranch_vccz` only
//! after a static check of the skipped region.

const MAX_CODE_BYTES: usize = 1024;
const MAX_INPUT_WORDS: usize = 4096;

/// The SGPR number of EXEC_LO (guide 3.3, printed page 13). EXEC_HI is 127
/// and is never written by the forms in this subset.
const EXEC_LO: usize = 126;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Src {
    Reg(u8),
    Imm(u32),
}

/// A VOPC SRC0 operand (contract 2 allowlist).
#[derive(Clone, Copy, Debug)]
enum CmpSrc {
    Sgpr(u8),
    Vcc,
    Imm(u32),
    Vgpr(u8),
}

#[derive(Clone, Copy, Debug)]
enum Insn {
    SMov(u8, Src),
    /// SOP1 unary opcode 7, 11, 13, 15, 17, 19, 21, 23, 25, 26 or 52.
    SUnary(u32, u8, Src),
    /// SOP1 opcode 60-71: the ten SAVEEXEC forms (60-69) and the two WREXEC
    /// forms (70-71). Each writes the destination and EXEC_LO, in the order
    /// the guide's pseudocode gives.
    SSaveExec(u32, u8, Src),
    /// SOP2 opcode 0-10, 14, 16, 18, an even opcode 20-36, 38-40, 44, 46-49,
    /// 53 or 54.
    SAlu(u32, u8, Src, Src),
    /// SOPC opcode 0-11: `s_cmp_{eq,lg,gt,ge,lt,le}_i32` then `_u32`.
    SCmp(u32, Src, Src),
    /// SOPK opcode 0 or 2-16 on SGPR `SDST` (s0-s105) with the raw SIMM16.
    SImm(u32, u8, u16),
    BranchScc0(usize),
    BranchMask {
        target: usize,
        register: usize,
        nonzero: bool,
    },
    Branch(usize),
    VMov(u8, Src),
    VMovFromSgpr(u8, u8),
    /// `v_mov_b32_e32 vD, vcc_lo`.
    VMovFromVcc(u8),
    /// Integer vector compare into `dst`: relation 0 f, 1 lt, 2 eq, 3 le, 4 gt, 5 ne, 6 ge, 7 t.
    /// `bits` is 16, 32 or 64; 64-bit register operands name the low register of a pair.
    VCmp {
        dst: u8,
        relation: u8,
        signed: bool,
        bits: u32,
        src0: CmpSrc,
        src1: u8,
    },
    VAddNc(u8, u8, u8),
    VAnd(u8, u8, u8),
    VOr(u8, u8, u8),
    VShiftLeft(u8, u8, u8),
    VShiftRight(u8, u8, u8),
    VShiftRightArithmetic(u8, u8, u8),
    VXor(u8, u8, u8),
    /// VOP2 integer opcode 9-12, 17-20, 30 or 37-39, or binary32 opcode 3-5,
    /// 8, 15 or 16: VDST, SRC0, VSRC1.
    VInt(u32, u8, CmpSrc, u8),
    VIntVcc(u32, u8, CmpSrc, u8),
    /// VOP1 integer opcode 55-59 or binary32 rounding opcode 33-36: VDST, SRC0.
    VUnary(u32, u8, CmpSrc),
    /// `v_swap_b32 vD, vS` (VOP1 opcode 101).
    VSwap(u8, u8),
    Load(u8),
    Store(u8),
    WaitRead,
    WaitWrite,
    /// SMEM `s_buffer_load_dword`, `_x2` or `_x4` from the synthetic input
    /// buffer (`s[0:3]`, base zero) into `dwords` consecutive SGPRs.
    SLoad {
        dst: u8,
        dwords: u8,
        offset: u32,
    },
    /// `s_waitcnt lgkmcnt(0)`. The synthetic scalar load completes immediately.
    WaitScalar,
    /// MIMG 2D image sample into `dst..` with one VGPR per DMASK component;
    /// the LOD is level zero or read from `coord + 2` (`explicit_lod`).
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

/// Execute the synthetic fixture ABI and return one output word per input word.
pub fn execute(code: &[u8], input: &[u32]) -> Result<Vec<u32>, String> {
    if input.is_empty() || input.len() > MAX_INPUT_WORDS {
        return Err("input word count outside limits".to_string());
    }
    if code.len() > MAX_CODE_BYTES {
        return Err("code exceeds limit".to_string());
    }
    let program = decode(code)?;
    let mut output = Vec::with_capacity(input.len());
    for wave in input.chunks(32) {
        output.extend(run_wave(&program, wave, input)?);
    }
    Ok(output)
}

/// Per-lane vector and memory state of one wave lane.
struct Lane {
    input: u32,
    vgpr: [Option<u32>; 256],
    pending_read: Option<u8>,
    pending_write: Option<u32>,
    written: Option<u32>,
    /// A lane-local failure; the lane stops executing.
    error: Option<String>,
}

/// The lowest-index failed lane's error, in dispatch order.
fn first_error(lanes: &[Lane]) -> Option<String> {
    lanes.iter().find_map(|lane| lane.error.clone())
}

/// Fail every still-running lane with a wave-uniform error and report the
/// lowest-index failure, exactly as independent per-lane runs would.
fn wave_error(lanes: &mut [Lane], error: String) -> String {
    for lane in lanes.iter_mut() {
        lane.error.get_or_insert_with(|| error.clone());
    }
    first_error(lanes).unwrap_or(error)
}

fn run_wave(program: &[(usize, Insn)], wave: &[u32], buffer: &[u32]) -> Result<Vec<u32>, String> {
    let mut lanes: Vec<Lane> = wave
        .iter()
        .map(|&input| Lane {
            input,
            vgpr: [None; 256],
            pending_read: None,
            pending_write: None,
            written: None,
            error: None,
        })
        .collect();
    // Scalar state is wave-uniform and shared by all lanes.
    let mut sgpr = [None::<u32>; 128];
    sgpr[EXEC_LO] = Some(u32::MAX);
    let mut scc: Option<bool> = None;
    // sgpr[106] holds VCC_LO's value; this flag marks it lane-derived.
    let mut vcc_lane_derived = false;
    let mut pc = 0usize;

    while pc < program.len() {
        let (offset, insn) = program[pc];
        if lanes.iter().all(|lane| lane.error.is_some()) {
            break;
        }
        let read_s = |sgpr: &[Option<u32>; 128], src: Src| -> Result<u32, String> {
            match src {
                Src::Imm(value) => Ok(value),
                Src::Reg(reg) => {
                    sgpr[reg as usize].ok_or_else(|| format!("{offset:#x}: uninitialized s{reg}"))
                }
            }
        };
        let execution_mask = match read_s(&sgpr, Src::Reg(EXEC_LO as u8)) {
            Ok(mask) => mask,
            Err(error) => return Err(wave_error(&mut lanes, error)),
        };

        let scalar: Result<Option<usize>, String> = match insn {
            Insn::End => {
                for lane in lanes.iter_mut().filter(|lane| lane.error.is_none()) {
                    if lane.pending_read.is_some() || lane.pending_write.is_some() {
                        lane.error = Some(format!("{offset:#x}: outstanding transfer at end"));
                    } else if lane.written.is_none() {
                        lane.error = Some(format!("{offset:#x}: missing output store"));
                    }
                }
                if let Some(error) = first_error(&lanes) {
                    return Err(error);
                }
                return Ok(lanes.iter().map(|lane| lane.written.unwrap()).collect());
            }
            Insn::VCmp {
                dst,
                relation,
                signed,
                bits,
                src0,
                src1,
            } => {
                // A compare couples lanes; a failed lane has no defined
                // contribution, so the wave stops with its error.
                if let Some(error) = first_error(&lanes) {
                    return Err(error);
                }
                compare(
                    &lanes,
                    &sgpr,
                    execution_mask,
                    (relation, signed, bits, src0, src1),
                    offset,
                )
                .map(|mask| {
                    sgpr[usize::from(dst)] = Some(mask);
                    if dst == 106 {
                        // VCC_HI has no known value after a VCC compare (contract 3).
                        sgpr[107] = None;
                        vcc_lane_derived = execution_mask != 0;
                    }
                    None
                })
            }
            Insn::VIntVcc(opcode, dst, src0, src1) => {
                if let Some(error) = first_error(&lanes) {
                    return Err(error);
                }
                vector_vcc(
                    &mut lanes,
                    &sgpr,
                    execution_mask,
                    (opcode, dst, src0, src1),
                    offset,
                )
                .map(|mask| {
                    if opcode != 1 {
                        sgpr[106] = Some(mask);
                        sgpr[107] = None;
                        vcc_lane_derived = execution_mask != 0;
                    }
                    None
                })
            }
            Insn::SUnary(opcode, dst, src) => {
                if dst < 8 {
                    Err(format!("{offset:#x}: write to a resource SGPR"))
                } else {
                    read_s(&sgpr, src).map(|value| {
                        let (result, flag) = scalar_unary(opcode, value);
                        sgpr[dst as usize] = Some(result);
                        if let Some(flag) = flag {
                            scc = Some(flag);
                        }
                        None
                    })
                }
            }
            Insn::SSaveExec(opcode, dst, src) => {
                if dst < 8 {
                    Err(format!("{offset:#x}: write to a resource SGPR"))
                } else {
                    let exec_lo = sgpr[EXEC_LO]
                        .ok_or_else(|| format!("{offset:#x}: uninitialized exec_lo"))?;
                    read_s(&sgpr, src).map(|value| {
                        // Guide 12.3, printed pages 120-122: the ten SAVEEXEC
                        // forms write the previous EXEC_LO to the destination
                        // and then update EXEC_LO; the two WREXEC forms update
                        // EXEC_LO first and write the new value. The mask does
                        // not affect scalar execution, so the form runs even
                        // when EXEC_LO is zero.
                        let updated = saveexec_mask(opcode, value, exec_lo);
                        let destination = if opcode <= 69 { exec_lo } else { updated };
                        sgpr[dst as usize] = Some(destination);
                        sgpr[EXEC_LO] = Some(updated);
                        scc = Some(updated != 0);
                        None
                    })
                }
            }
            Insn::SLoad {
                dst,
                dwords,
                offset,
            } => {
                if dst < 8 {
                    return Err(wave_error(
                        &mut lanes,
                        format!("{offset:#x}: scalar load into a resource descriptor"),
                    ));
                }
                for index in 0..usize::from(dwords) {
                    let word = offset as usize / 4 + index;
                    let value = buffer.get(word).copied().ok_or_else(|| {
                        format!("{offset:#x}: scalar load outside the input buffer")
                    })?;
                    sgpr[usize::from(dst) + index] = Some(value);
                }
                Ok(None)
            }
            Insn::WaitScalar => Ok(None),
            Insn::SMov(dst, src) => {
                if dst < 8 {
                    Err(format!("{offset:#x}: write to a resource SGPR"))
                } else {
                    read_s(&sgpr, src).map(|value| {
                        sgpr[dst as usize] = Some(value);
                        if dst == 106 {
                            vcc_lane_derived = false;
                        }
                        None
                    })
                }
            }
            Insn::SCmp(opcode, a, b) => read_s(&sgpr, a).and_then(|a| {
                read_s(&sgpr, b).map(|b| {
                    scc = Some(scalar_compare(opcode, a, b));
                    None
                })
            }),
            Insn::SImm(opcode, dst, simm16) => {
                scalar_immediate(&mut sgpr, &mut scc, (opcode, dst, simm16), offset).map(|()| None)
            }
            Insn::SAlu(opcode, dst, a, b) => {
                scalar_alu(&mut sgpr, &mut scc, (opcode, dst, a, b), offset).map(|()| None)
            }
            Insn::BranchScc0(target) => scc
                .ok_or_else(|| format!("{offset:#x}: uninitialized SCC"))
                .and_then(|condition| {
                    if condition {
                        Ok(None)
                    } else {
                        index_of(program, target, offset).map(Some)
                    }
                }),
            Insn::BranchMask {
                target,
                register,
                nonzero,
            } => {
                let takes = |mask| (mask != 0) == nonzero;
                if register == 106 && vcc_lane_derived {
                    let live = lanes.iter().find(|lane| lane.error.is_none()).unwrap();
                    check_runtime_region(program, pc, target, live, &sgpr, execution_mask, offset)
                        .and_then(|()| {
                            if takes(sgpr[106].unwrap()) {
                                index_of(program, target, offset).map(Some)
                            } else {
                                Ok(None)
                            }
                        })
                } else {
                    sgpr[register]
                        .ok_or_else(|| format!("{offset:#x}: uninitialized mask"))
                        .and_then(|mask| {
                            if takes(mask) {
                                index_of(program, target, offset).map(Some)
                            } else {
                                Ok(None)
                            }
                        })
                }
            }
            Insn::Branch(target) => index_of(program, target, offset).map(Some),
            Insn::ImageSample {
                dst,
                coord,
                resource,
                sampler,
                dmask,
                explicit_lod,
            } => {
                if let Some(error) = first_error(&lanes) {
                    return Err(error);
                }
                image_sample(
                    &mut lanes,
                    &sgpr,
                    execution_mask,
                    buffer,
                    (dst, coord, resource, sampler, dmask, explicit_lod),
                    offset,
                )
                .map(|()| None)
            }
            _ => {
                for (lane_index, lane) in lanes.iter_mut().enumerate() {
                    if lane.error.is_none()
                        && let Err(error) =
                            lane_step(lane, &sgpr, execution_mask, insn, offset, lane_index)
                    {
                        lane.error = Some(error);
                    }
                }
                Ok(None)
            }
        };
        match scalar {
            Ok(Some(next)) => pc = next,
            Ok(None) => pc += 1,
            Err(error) => return Err(wave_error(&mut lanes, error)),
        }
    }
    Err(first_error(&lanes).unwrap_or_else(|| "stream ended without s_endpgm".to_string()))
}

/// The consecutive SGPRs of a validated image descriptor.
fn resource_words(
    sgpr: &[Option<u32>; 128],
    base: u8,
    count: usize,
    offset: usize,
) -> Result<Vec<u32>, String> {
    (0..count)
        .map(|index| {
            sgpr[usize::from(base) + index]
                .ok_or_else(|| format!("{offset:#x}: uninitialized descriptor SGPR"))
        })
        .collect()
}

/// Independent nearest-neighbor, clamp-to-edge RGBA8 sample: the coordinate is
/// clamped to [0,1], scaled by the mip dimensions and truncated toward zero to
/// the texel index (the host reference floors a binary64 product instead, so the
/// two oracles do not share the coordinate arithmetic).
fn sample_texel(
    wave: &[u32],
    base: usize,
    width: usize,
    height: usize,
    mip: u32,
    u_bits: u32,
    v_bits: u32,
) -> Result<[u8; 4], String> {
    let u = f32::from_bits(u_bits).clamp(0.0, 1.0);
    let v = f32::from_bits(v_bits).clamp(0.0, 1.0);
    let w = width >> mip;
    let h = height >> mip;
    let x = ((u * w as f32) as usize).min(w - 1);
    let y = ((v * h as f32) as usize).min(h - 1);
    let mut offset = base / 4;
    for level in 0..mip {
        offset += (width >> level) * (height >> level);
    }
    let word = wave
        .get(offset + y * w + x)
        .ok_or_else(|| "texture data outside the input buffer".to_string())?;
    Ok(word.to_le_bytes())
}

/// Execute one MIMG image sample for the whole wave, writing one VGPR per
/// DMASK-selected component of each active lane. Descriptor validation follows
/// guide Tables 45 and 46 directly from the SGPR words.
#[allow(clippy::too_many_arguments)]
fn image_sample(
    lanes: &mut [Lane],
    sgpr: &[Option<u32>; 128],
    exec: u32,
    buffer: &[u32],
    (dst, coord, resource, sampler, dmask, explicit_lod): (u8, u8, u8, u8, u8, bool),
    offset: usize,
) -> Result<(), String> {
    let r = resource_words(sgpr, resource, 8, offset)?;
    if r[4..].iter().any(|&word| word != 0) {
        return Err(format!(
            "{offset:#x}: nonzero 256-bit resource extension words"
        ));
    }
    let ty = (r[3] >> 28) & 0xf;
    let format = (r[1] >> 20) & 0x1ff;
    let width1 = (r[1] >> 30) | ((r[2] & 0x3fff) << 2);
    let height1 = (r[2] >> 14) & 0xffff;
    if width1 & 0xc000 != 0 || height1 & 0xc000 != 0 {
        return Err(format!(
            "{offset:#x}: resource width or height sets a reserved high bit"
        ));
    }
    let width = width1 as usize + 1;
    let height = height1 as usize + 1;
    if ty != 9 {
        return Err(format!(
            "{offset:#x}: resource descriptor is not a 2D image"
        ));
    }
    if format != 56 {
        return Err(format!(
            "{offset:#x}: resource descriptor is not RGBA8 unorm"
        ));
    }
    if (r[3] >> 20) & 0x1f != 0 {
        return Err(format!("{offset:#x}: resource swizzle mode is not linear"));
    }
    if (r[1] >> 8) & 0xfff != 0 {
        return Err(format!("{offset:#x}: resource minimum LOD is nonzero"));
    }
    let base_level = (r[3] >> 12) & 0xf;
    let last_level = (r[3] >> 16) & 0xf;
    if last_level < base_level {
        return Err(format!(
            "{offset:#x}: resource last level is below base level"
        ));
    }
    // Resource level 1, an identity RGBA channel select and base level 0.
    if (r[2] >> 31) & 1 != 1 || (r[3] & 0xfff) != 0xfac || base_level != 0 {
        return Err(format!(
            "{offset:#x}: unsupported resource level or channel select"
        ));
    }
    let s = resource_words(sgpr, sampler, 4, offset)?;
    let clamp = (s[0] & 0x3f) == 0x12 && (s[0] >> 6) & 0x7 == 2;
    let point = ((s[2] >> 20) & 0xf) == 0 && (s[2] >> 24) & 0x3 == 0;
    let mip_filter = (s[2] >> 26) & 0x3 == 1;
    let bias = (s[2] & 0xfff) == 0 && (s[2] >> 12) & 0x3 == 0;
    let min_lod = s[1] & 0xfff;
    let max_lod = (s[1] >> 12) & 0xfff;
    if !clamp || !point || !mip_filter || !bias || min_lod != 0 || max_lod < last_level << 8 {
        return Err(format!(
            "{offset:#x}: sampler is not point filtering with clamp addressing and an explicit mip level"
        ));
    }
    let base = ((r[0] as u64) | (u64::from(r[1] & 0xff) << 32)) << 8;
    let base = base as usize;
    let mips = last_level - base_level + 1;
    for (lane_index, lane) in lanes.iter_mut().enumerate() {
        if exec & (1 << lane_index) == 0 {
            continue;
        }
        let u = read_vgpr(lane, coord, offset)?;
        let v = read_vgpr(lane, coord + 1, offset)?;
        let mip = if explicit_lod {
            let lod = f32::from_bits(read_vgpr(lane, coord + 2, offset)?);
            if !lod.is_finite() || lod < 0.0 || lod.fract() != 0.0 || lod as u32 >= mips {
                return Err(format!(
                    "{offset:#x}: explicit LOD is not a representable integer mip level"
                ));
            }
            lod as u32
        } else {
            0
        };
        let texel = sample_texel(buffer, base, width, height, mip, u, v)?;
        let mut slot = dst;
        for component in 0..4u8 {
            if dmask & (1 << component) == 0 {
                continue;
            }
            check_destination(slot, lane.pending_read, offset)?;
            lane.vgpr[usize::from(slot)] =
                Some((f32::from(texel[usize::from(component)]) / 255.0).to_bits());
            slot += 1;
        }
    }
    Ok(())
}

/// Guide 12.7 vector select and carry/borrow, with the Wave32 mask from 3.9.
fn vector_vcc(
    lanes: &mut [Lane],
    sgpr: &[Option<u32>; 128],
    exec: u32,
    (opcode, dst, src0, src1): (u32, u8, CmpSrc, u8),
    offset: usize,
) -> Result<u32, String> {
    if exec == 0 {
        return Ok(0);
    }
    let incoming = sgpr[106].ok_or_else(|| format!("{offset:#x}: uninitialized vcc_lo"))?;
    let mut mask = 0;
    for (index, lane) in lanes.iter_mut().enumerate() {
        if exec != u32::MAX && lane.vgpr[usize::from(dst)].is_none() {
            return Err(format!("{offset:#x}: partial EXEC on uninitialized VGPR"));
        }
        let a = match src0 {
            CmpSrc::Sgpr(reg) => sgpr[usize::from(reg)]
                .ok_or_else(|| format!("{offset:#x}: uninitialized s{reg}"))?,
            CmpSrc::Vcc => incoming,
            CmpSrc::Imm(value) => value,
            CmpSrc::Vgpr(reg) => read_vgpr(lane, reg, offset)?,
        };
        let b = read_vgpr(lane, src1, offset)?;
        let lane_bit = 2_u32.pow(index as u32);
        if exec & lane_bit == 0 {
            continue;
        }
        check_destination(dst, lane.pending_read, offset)?;
        let bit = incoming / lane_bit % 2;
        let (value, flag) = match opcode {
            1 => (if bit == 1 { b } else { a }, false),
            40 => {
                let (sum, overflow) = a.overflowing_add(b);
                let (value, carry) = sum.overflowing_add(bit);
                (value, overflow || carry)
            }
            41 | 42 => {
                let (lhs, rhs) = if opcode == 42 { (b, a) } else { (a, b) };
                let (difference, underflow) = lhs.overflowing_sub(rhs);
                let (value, borrow) = difference.overflowing_sub(bit);
                (value, underflow || borrow)
            }
            _ => unreachable!("not a VCC-reading VOP2 opcode"),
        };
        lane.vgpr[usize::from(dst)] = Some(value);
        if flag {
            mask |= lane_bit;
        }
    }
    Ok(mask)
}

/// One SOPK instruction, per the operation text of guide 12.2. SDST names
/// the destination and, for every form but the two moves, the first source.
fn scalar_immediate(
    sgpr: &mut [Option<u32>; 128],
    scc: &mut Option<bool>,
    (opcode, dst, simm16): (u32, u8, u16),
    offset: usize,
) -> Result<(), String> {
    let d = |sgpr: &[Option<u32>; 128]| {
        sgpr[dst as usize].ok_or_else(|| format!("{offset:#x}: uninitialized s{dst}"))
    };
    // signext(SIMM16[15:0]): copy bit 15 into bits 16-31.
    let sext = u32::from(simm16) | if simm16 >> 15 == 1 { 0xffff_0000 } else { 0 };
    // The unsigned compares use SIMM16[15:0] as is, zero-extended.
    let zext = u32::from(simm16);
    if (3..=14).contains(&opcode) {
        let s0 = d(sgpr)?;
        let (a, k) = (s0 as i32, sext as i32);
        *scc = Some(match opcode {
            3 => a == k,
            4 => a != k,
            5 => a > k,
            6 => a >= k,
            7 => a < k,
            8 => a <= k,
            9 => s0 == zext,
            10 => s0 != zext,
            11 => s0 > zext,
            12 => s0 >= zext,
            13 => s0 < zext,
            _ => s0 <= zext,
        });
        return Ok(());
    }
    if dst < 8 {
        return Err(format!("{offset:#x}: write to a resource SGPR"));
    }
    let value = match opcode {
        0 => sext,
        2 => {
            if !scc.ok_or_else(|| format!("{offset:#x}: uninitialized SCC"))? {
                return Ok(());
            }
            sext
        }
        15 => {
            let tmp = d(sgpr)?;
            let sum = tmp.wrapping_add(sext);
            // tmp[31] == SIMM16[15] && tmp[31] != D[31].
            *scc = Some(tmp >> 31 == u32::from(simm16 >> 15) && tmp >> 31 != sum >> 31);
            sum
        }
        // 16: the low 32 bits of the product, the same for both signednesses.
        _ => d(sgpr)?.wrapping_mul(sext),
    };
    sgpr[dst as usize] = Some(value);
    Ok(())
}

fn scalar_alu(
    sgpr: &mut [Option<u32>; 128],
    scc: &mut Option<bool>,
    (opcode, dst, a, b): (u32, u8, Src, Src),
    offset: usize,
) -> Result<(), String> {
    if dst < 8 {
        return Err(format!("{offset:#x}: write to a resource SGPR"));
    }
    let read = |src: Src| match src {
        Src::Imm(value) => Ok(value),
        Src::Reg(reg) => {
            sgpr[reg as usize].ok_or_else(|| format!("{offset:#x}: uninitialized s{reg}"))
        }
    };
    let (a, b) = (read(a)?, read(b)?);
    let carry_in = || scc.ok_or_else(|| format!("{offset:#x}: uninitialized SCC"));
    let bit = |word: u32, i: u32| (word >> i) & 1 == 1;
    let (value, flag) = match opcode {
        0 => ripple(a, b, false),
        // RDNA 2 ISA guide 12.1: MIN/MAX pick S0 iff S0 < S1 (> for MAX), the
        // order a borrow out of S0 - S1 (or S1 - S0) shows; signed forms flip
        // both sign bits first. SCC = whether S0 was picked (0 on a tie).
        6..=9 => {
            let flip = if opcode % 2 == 0 { 0x8000_0000 } else { 0 };
            let (x, y) = (a ^ flip, b ^ flip);
            let (p, q) = if opcode < 8 { (x, y) } else { (y, x) };
            let first = !ripple(p, !q, true).1;
            (if first { a } else { b }, first)
        }
        // CSELECT: D = SCC ? S0 : S1.
        10 => {
            let pick_first = carry_in()?;
            (if pick_first { a } else { b }, false)
        }
        // BFM: bit i of D is set iff S1[4:0] <= i < S1[4:0] + S0[4:0].
        36 => {
            let (size, start) = (a % 32, b % 32);
            let value = (0..32)
                .filter(|&i| i >= start && i < start + size)
                .fold(0, |d, i| d | 1 << i);
            (value, false)
        }
        // MUL_I32 and MUL_HI: the 64-bit unsigned product by shift and add;
        // the signed high word subtracts S1 when S0 < 0 and S0 when S1 < 0
        // (two's complement correction).
        38 | 53 | 54 => {
            let product = (0..32)
                .filter(|&i| bit(b, i))
                .fold(0_u64, |sum, i| sum + (u64::from(a) << i));
            let (low, mut high) = (product as u32, (product >> 32) as u32);
            if opcode == 54 {
                if bit(a, 31) {
                    high = ripple(high, !b, true).0;
                }
                if bit(b, 31) {
                    high = ripple(high, !a, true).0;
                }
            }
            (if opcode == 38 { low } else { high }, false)
        }
        // BFE: D[i] = S0[offset + i] for i < width (offset S1[4:0], width
        // S1[22:16]); positions past bit 31 read 0 (U32) or S0[31] (I32), and
        // I32 copies D[width - 1] into the bits at and above width. SCC = D != 0.
        39 | 40 => {
            let (start, width) = (b % 32, (b >> 16) % 128);
            let source = |i: u32| match start + i {
                at @ 0..=31 => bit(a, at),
                _ => opcode == 40 && bit(a, 31),
            };
            let value = (0..32).fold(0, |d, i| {
                let set = if i < width {
                    source(i)
                } else {
                    opcode == 40 && width > 0 && source(width - 1)
                };
                d | u32::from(set) << i
            });
            (value, value != 0)
        }
        // ABSDIFF: the 32-bit S0 - S1, negated when bit 31 is set
        // (0x80000000 stays). SCC = D != 0.
        44 => {
            let difference = ripple(a, !b, true).0;
            let value = if bit(difference, 31) {
                ripple(!difference, 0, true).0
            } else {
                difference
            };
            (value, value != 0)
        }
        // LSHLn_ADD, n = 1-4: D = (S0 << n) + S1; SCC = 1 iff the exact sum
        // reaches 2^32, i.e. bits shifted out of S0 or a carry out of the add.
        46..=49 => {
            let n = opcode - 45;
            let (value, carry) = ripple(a << n, b, false);
            (value, carry || a >> (32 - n) != 0)
        }
        // RDNA 2 ISA guide 12.1: signed overflow from the sign bits,
        // add: S0[31] == S1[31] != D[31]; subtract: S0[31] != S1[31] != D[31].
        2 | 3 => {
            let value = if opcode == 2 {
                ripple(a, b, false).0
            } else {
                ripple(a, !b, true).0
            };
            let sign = |word: u32| word & 0x8000_0000 != 0;
            let same = sign(a) == sign(b);
            let overflow = (if opcode == 2 { same } else { !same }) && sign(a) != sign(value);
            (value, overflow)
        }
        4 => ripple(a, b, carry_in()?),
        // a - b - borrow = a + !b + (1 - borrow); borrow-out = !carry-out.
        1 | 5 => {
            let borrow = opcode == 5 && carry_in()?;
            let (value, carry) = ripple(a, !b, !borrow);
            (value, !carry)
        }
        // RDNA 2 ISA guide 12.1: the shifts use only S1[4:0], taken here as
        // the remainder modulo 32, and scale by a power of two in exact
        // arithmetic; ASHR is floor division of the signed S0. SCC = D != 0.
        30 | 32 | 34 => {
            let scale = 1_i64 << (b % 32);
            let value = match opcode {
                30 => (i64::from(a) * scale).rem_euclid(1 << 32),
                32 => i64::from(a) / scale,
                _ => i64::from(a as i32).div_euclid(scale).rem_euclid(1 << 32),
            } as u32;
            (value, value != 0)
        }
        _ => {
            let mut value = 0;
            for bit in 0..32 {
                let (x, y) = ((a >> bit) & 1 == 1, (b >> bit) & 1 == 1);
                let set = match opcode {
                    14 => x && y,
                    16 => x || y,
                    18 => x != y,
                    20 => x && !y,
                    22 => x || !y,
                    24 => !(x && y),
                    26 => !(x || y),
                    _ => x == y,
                };
                value |= u32::from(set) << bit;
            }
            (value, value != 0)
        }
    };
    sgpr[dst as usize] = Some(value);
    // CSELECT, BFM, MUL_I32 and MUL_HI have no SCC line in guide 12.1 ("n" in
    // Tables 11, 12 and 14): SCC stays as it was, possibly unset.
    if !matches!(opcode, 10 | 36 | 38 | 53 | 54) {
        *scc = Some(flag);
    }
    Ok(())
}

/// RDNA 2 ISA guide 12.4: SCC = (S0 <relation> S1). Signed operands (opcodes
/// 0-5) are ordered by flipping both sign bits and comparing unsigned.
fn scalar_compare(opcode: u32, a: u32, b: u32) -> bool {
    let (a, b) = if opcode < 6 {
        (a ^ 0x8000_0000, b ^ 0x8000_0000)
    } else {
        (a, b)
    };
    match opcode % 6 {
        0 => a == b,
        1 => a != b,
        2 => a > b,
        3 => a >= b,
        4 => a < b,
        _ => a <= b,
    }
}

/// RDNA 2 ISA guide 12.3, evaluated with arithmetic on the value (no host bit
/// intrinsics): D and, for NOT, BCNT0, BCNT1 and ABS, SCC = (D != 0).
fn scalar_unary(opcode: u32, value: u32) -> (u32, Option<bool>) {
    // The bits from the LSB, as repeated halving produces them.
    let mut bits = [false; 32];
    let mut rest = u64::from(value);
    for slot in &mut bits {
        *slot = rest % 2 == 1;
        rest /= 2;
    }
    let word = |bits: &dyn Fn(usize) -> bool| {
        (0..32).rev().fold(0u64, |w, i| w * 2 + u64::from(bits(i))) as u32
    };
    let position = |found: Option<usize>| found.map_or(u32::MAX, |i| i as u32);
    let ones = bits.iter().filter(|&&b| b).count() as u32;
    let result = match opcode {
        7 => 0xffff_ffff - value,
        11 => word(&|i| bits[31 - i]),
        13 => 32 - ones,
        15 => ones,
        17 => position(bits.iter().position(|&b| !b)),
        19 => position(bits.iter().position(|&b| b)),
        21 => position(bits.iter().rev().position(|&b| b)),
        23 => position(
            bits.iter()
                .rev()
                .skip(1)
                .position(|&b| b != bits[31])
                .map(|i| i + 1),
        ),
        // signext of the low 8 or 16 bits: subtract 2^n when the value mod 2^n
        // is at least 2^(n-1).
        25 | 26 => {
            let span = if opcode == 25 { 256 } else { 65_536 };
            let low = i64::from(value) % span;
            let signed = if low >= span / 2 { low - span } else { low };
            signed.rem_euclid(1 << 32) as u32
        }
        // |S0| modulo 2^32, so INT_MIN stays 0x80000000.
        _ => i64::from(value as i32).abs().rem_euclid(1 << 32) as u32,
    };
    let flag = matches!(opcode, 7 | 13 | 15 | 52).then_some(result != 0);
    (result, flag)
}

/// RDNA 2 ISA guide 12.3, printed pages 120-122: the new EXEC_LO of the
/// twelve SAVEEXEC and WREXEC forms, one bit at a time, so that the oracle
/// does not use the host's bitwise operators on the mask. The guide's
/// pseudocode names the bits of S0 and of the old EXEC_LO; the ten SAVEEXEC
/// forms (60-69) and the two WREXEC forms (70-71) differ only in the order
/// of the two writes, which the caller applies.
fn saveexec_mask(opcode: u32, s0: u32, exec: u32) -> u32 {
    let bits = |value: u32| {
        let mut bits = [false; 32];
        let mut rest = u64::from(value);
        for slot in &mut bits {
            *slot = rest % 2 == 1;
            rest /= 2;
        }
        bits
    };
    let (a, b) = (bits(s0), bits(exec));
    let word =
        |bits: [bool; 32]| (0..32).rev().fold(0u64, |w, i| w * 2 + u64::from(bits[i])) as u32;
    let mut out = [false; 32];
    for i in 0..32 {
        // `a` is a bit of S0 and `b` a bit of the old EXEC_LO.
        out[i] = match opcode {
            60 => a[i] && b[i],
            61 => a[i] || b[i],
            62 => a[i] != b[i],
            63 => a[i] && !b[i],
            64 => a[i] || !b[i],
            65 => !(a[i] && b[i]),
            66 => !(a[i] || b[i]),
            67 => !(a[i] != b[i]),
            68 => !a[i] && b[i],
            69 => !a[i] || b[i],
            70 => !a[i] && b[i],
            71 => a[i] && !b[i],
            _ => unreachable!("not a SAVEEXEC or WREXEC opcode"),
        };
    }
    word(out)
}

fn read_vgpr(lane: &Lane, reg: u8, offset: usize) -> Result<u32, String> {
    if reg == 0 {
        return Err(format!("{offset:#x}: v0 is the resource index"));
    }
    if lane.pending_read == Some(reg) {
        return Err(format!("{offset:#x}: read before load wait"));
    }
    lane.vgpr[reg as usize].ok_or_else(|| format!("{offset:#x}: uninitialized v{reg}"))
}

/// Evaluate one integer vector compare for the whole wave and return its mask.
fn compare(
    lanes: &[Lane],
    sgpr: &[Option<u32>; 128],
    execution_mask: u32,
    (relation, signed, bits, src0, src1): (u8, bool, u32, CmpSrc, u8),
    offset: usize,
) -> Result<u32, String> {
    if execution_mask == 0 {
        // Encoding already validated; nothing is read (contract 5).
        return Ok(0);
    }
    // Registers per operand: a 64-bit operand is {reg (bits 31:0), reg+1 (63:32)}.
    let count: u8 = if bits == 64 { 2 } else { 1 };
    let read_s = |reg: u8| -> Result<u64, String> {
        let mut value = 0u64;
        for half in (0..count).rev() {
            let r = reg + half;
            let word =
                sgpr[r as usize].ok_or_else(|| format!("{offset:#x}: uninitialized s{r}"))?;
            value = value * (1 << 32) + u64::from(word);
        }
        Ok(value)
    };
    let read_v = |lane: &Lane, reg: u8| -> Result<u64, String> {
        let mut value = 0u64;
        for half in (0..count).rev() {
            value = value * (1 << 32) + u64::from(read_vgpr(lane, reg + half, offset)?);
        }
        Ok(value)
    };
    let scalar_src0 = match src0 {
        CmpSrc::Sgpr(reg) => Some(read_s(reg)?),
        CmpSrc::Vcc => {
            Some(u64::from(sgpr[106].ok_or_else(|| {
                format!("{offset:#x}: uninitialized vcc_lo")
            })?))
        }
        // Inline integers -16..64 keep their value at every width; a
        // 32-bit literal (32-bit forms only) is used as its 32 bits.
        CmpSrc::Imm(value) if bits == 64 && value >= 0x8000_0000 => {
            Some(u64::from(value) + 0xffff_ffff_0000_0000)
        }
        CmpSrc::Imm(value) => Some(u64::from(value)),
        CmpSrc::Vgpr(_) => None,
    };
    // VGPR readiness is wave-uniform; validate it on every existing lane so
    // acceptance does not depend on which lanes are active.
    for lane in lanes {
        if let CmpSrc::Vgpr(reg) = src0 {
            read_v(lane, reg)?;
        }
        read_v(lane, src1)?;
    }
    // Only the low `bits` bits take part (16-bit forms ignore bits 31:16).
    let keep = u64::MAX >> (64 - bits);
    let sign = 1u64 << (bits - 1);
    let mut mask = 0u32;
    for (n, lane) in lanes.iter().enumerate() {
        if (execution_mask >> n) & 1 == 0 {
            continue;
        }
        let a = match (scalar_src0, src0) {
            (Some(value), _) => value,
            (None, CmpSrc::Vgpr(reg)) => read_v(lane, reg)?,
            (None, _) => unreachable!(),
        };
        let b = read_v(lane, src1)?;
        let (a, b) = (a & keep, b & keep);
        // Signed order equals unsigned order after flipping the sign bit.
        let (a, b) = if signed { (a ^ sign, b ^ sign) } else { (a, b) };
        // Guide 12.9 Table 59 offsets; F (0) and T (7) ignore the operand
        // values but still read them (contract 5: sources are validated).
        let pass = match relation {
            0 => false,
            1 => a < b,
            2 => a == b,
            3 => a <= b,
            4 => a > b,
            5 => a != b,
            6 => a >= b,
            _ => true,
        };
        mask |= u32::from(pass) << n;
    }
    Ok(mask)
}

/// Static acceptance of the region skipped by a runtime `s_cbranch_vccz`
/// (contract 6), independent of lane data, EXEC and the branch outcome.
fn check_runtime_region(
    program: &[(usize, Insn)],
    pc: usize,
    target: usize,
    lane: &Lane,
    sgpr: &[Option<u32>; 128],
    execution_mask: u32,
    offset: usize,
) -> Result<(), String> {
    // Contract 6 forbids an outstanding load; a pending store is waited for later.
    if lane.pending_read.is_some() {
        return Err(format!("{offset:#x}: load outstanding at a runtime branch"));
    }
    let end = index_of(program, target, offset)?;
    for &(at, insn) in &program[pc + 1..end] {
        // (destination, VGPR sources, SGPR source)
        let (dst, vgprs, scalar): (u8, Vec<u8>, Option<u8>) = match insn {
            Insn::VMov(dst, Src::Reg(reg)) => (dst, vec![reg], None),
            Insn::VMov(dst, Src::Imm(_)) => (dst, vec![], None),
            Insn::VMovFromSgpr(dst, src) => (dst, vec![], Some(src)),
            Insn::VMovFromVcc(dst) => (dst, vec![], Some(106)),
            Insn::VAddNc(dst, a, b)
            | Insn::VAnd(dst, a, b)
            | Insn::VOr(dst, a, b)
            | Insn::VXor(dst, a, b)
            | Insn::VShiftLeft(dst, a, b)
            | Insn::VShiftRight(dst, a, b)
            | Insn::VShiftRightArithmetic(dst, a, b) => (dst, vec![a, b], None),
            _ => {
                return Err(format!(
                    "{at:#x}: instruction not allowed in a runtime branch region"
                ));
            }
        };
        if dst == 0 || lane.vgpr[dst as usize].is_none() {
            return Err(format!(
                "{at:#x}: runtime branch region writes an uninitialized VGPR"
            ));
        }
        // Sources are read only when some lane executes (contract 6).
        if execution_mask == 0 {
            continue;
        }
        for reg in vgprs {
            read_vgpr(lane, reg, at)?;
        }
        if let Some(reg) = scalar {
            sgpr[reg as usize].ok_or_else(|| format!("{at:#x}: uninitialized s{reg}"))?;
        }
    }
    Ok(())
}

/// One per-lane vector, memory or wait instruction; behaviour and errors match
/// the former independent per-lane interpreter.
fn lane_step(
    lane: &mut Lane,
    sgpr: &[Option<u32>; 128],
    execution_mask: u32,
    insn: Insn,
    offset: usize,
    lane_index: usize,
) -> Result<(), String> {
    let read_s = |reg: u8| -> Result<u32, String> {
        sgpr[reg as usize].ok_or_else(|| format!("{offset:#x}: uninitialized s{reg}"))
    };
    let read_vcc = || sgpr[106].ok_or_else(|| format!("{offset:#x}: uninitialized vcc_lo"));
    if execution_mask.count_ones() < 32 {
        match insn {
            Insn::Load(_) | Insn::Store(_) => {
                return Err(format!(
                    "{offset:#x}: incomplete EXEC on a buffer instruction"
                ));
            }
            Insn::VMov(dst, _)
            | Insn::VMovFromSgpr(dst, _)
            | Insn::VMovFromVcc(dst)
            | Insn::VAddNc(dst, ..)
            | Insn::VAnd(dst, ..)
            | Insn::VOr(dst, ..)
            | Insn::VXor(dst, ..)
            | Insn::VShiftLeft(dst, ..)
            | Insn::VShiftRight(dst, ..)
            | Insn::VShiftRightArithmetic(dst, ..)
            | Insn::VInt(_, dst, ..)
            | Insn::VUnary(_, dst, _)
            | Insn::VSwap(dst, _) => {
                if execution_mask != 0 && lane.vgpr[usize::from(dst)].is_none() {
                    return Err(format!("{offset:#x}: partial EXEC on uninitialized VGPR"));
                }
                if execution_mask != 0 {
                    match insn {
                        Insn::VMovFromSgpr(_, src) => {
                            read_s(src)?;
                        }
                        Insn::VMovFromVcc(_) => {
                            read_vcc()?;
                        }
                        // Sources must be readable even in an inactive lane.
                        Insn::VInt(_, _, src0, src1) => {
                            match src0 {
                                CmpSrc::Sgpr(reg) => {
                                    read_s(reg)?;
                                }
                                CmpSrc::Vcc => {
                                    read_vcc()?;
                                }
                                CmpSrc::Vgpr(reg) => {
                                    read_vgpr(lane, reg, offset)?;
                                }
                                CmpSrc::Imm(_) => {}
                            }
                            read_vgpr(lane, src1, offset)?;
                        }
                        Insn::VUnary(_, _, src0) => match src0 {
                            CmpSrc::Sgpr(reg) => {
                                read_s(reg)?;
                            }
                            CmpSrc::Vcc => {
                                read_vcc()?;
                            }
                            CmpSrc::Vgpr(reg) => {
                                read_vgpr(lane, reg, offset)?;
                            }
                            CmpSrc::Imm(_) => {}
                        },
                        Insn::VSwap(dst, src0) => {
                            read_vgpr(lane, dst, offset)?;
                            read_vgpr(lane, src0, offset)?;
                        }
                        _ => {}
                    }
                }
                if (execution_mask.rotate_right(lane_index as u32) & 1) == 0 {
                    return Ok(());
                }
            }
            _ => {}
        }
    }

    let read_v = |reg: u8| read_vgpr(lane, reg, offset);
    let pending_read = lane.pending_read;
    let value = match insn {
        Insn::Load(data) => {
            if data == 0 || pending_read.is_some() {
                return Err(format!("{offset:#x}: invalid or overlapping load"));
            }
            lane.pending_read = Some(data);
            return Ok(());
        }
        Insn::Store(data) => {
            let value = read_v(data)?;
            if lane.pending_write.is_some() || lane.written.is_some() {
                return Err(format!("{offset:#x}: multiple output writes"));
            }
            lane.pending_write = Some(value);
            return Ok(());
        }
        Insn::WaitRead => {
            if let Some(dst) = lane.pending_read.take() {
                lane.vgpr[dst as usize] = Some(lane.input);
            }
            return Ok(());
        }
        Insn::WaitWrite => {
            if let Some(value) = lane.pending_write.take() {
                lane.written = Some(value);
            }
            return Ok(());
        }
        Insn::VMov(dst, src) => {
            let value = match src {
                Src::Imm(value) => value,
                Src::Reg(reg) => read_v(reg)?,
            };
            check_destination(dst, pending_read, offset)?;
            (dst, value)
        }
        Insn::VMovFromSgpr(dst, src) => {
            let value = read_s(src)?;
            check_destination(dst, pending_read, offset)?;
            (dst, value)
        }
        Insn::VMovFromVcc(dst) => {
            let value = read_vcc()?;
            check_destination(dst, pending_read, offset)?;
            (dst, value)
        }
        Insn::VAddNc(dst, a, b) => {
            check_destination(dst, pending_read, offset)?;
            let sum = u64::from(read_v(a)?) + u64::from(read_v(b)?);
            (dst, sum as u32)
        }
        Insn::VAnd(dst, a, b) => {
            check_destination(dst, pending_read, offset)?;
            (dst, read_v(a)? & read_v(b)?)
        }
        Insn::VOr(dst, a, b) => {
            check_destination(dst, pending_read, offset)?;
            (dst, read_v(a)? | read_v(b)?)
        }
        Insn::VShiftLeft(dst, count, value) => {
            check_destination(dst, pending_read, offset)?;
            let factor = 2_u64.pow(read_v(count)? % 32);
            let product = u64::from(read_v(value)?) * factor;
            (dst, (product % (1_u64 << 32)) as u32)
        }
        Insn::VShiftRight(dst, count, value) => {
            check_destination(dst, pending_read, offset)?;
            let divisor = 2_u32.pow(read_v(count)? % 32);
            (dst, read_v(value)? / divisor)
        }
        Insn::VShiftRightArithmetic(dst, count, value) => {
            check_destination(dst, pending_read, offset)?;
            // Floor division of the sign-extended value, reduced modulo
            // 2^32. This is arithmetic shift without using `>>`.
            let shifted =
                i64::from(read_v(value)? as i32).div_euclid(1_i64 << (read_v(count)? % 32));
            (dst, shifted.rem_euclid(1_i64 << 32) as u32)
        }
        Insn::VXor(dst, a, b) => {
            check_destination(dst, pending_read, offset)?;
            (dst, read_v(a)? ^ read_v(b)?)
        }
        Insn::VInt(opcode, dst, src0, src1) => {
            check_destination(dst, pending_read, offset)?;
            let a = match src0 {
                CmpSrc::Sgpr(reg) => read_s(reg)?,
                CmpSrc::Vcc => read_vcc()?,
                CmpSrc::Imm(value) => value,
                CmpSrc::Vgpr(reg) => read_v(reg)?,
            };
            let b = read_v(src1)?;
            let d = if matches!(opcode, 3..=5 | 8 | 15 | 16) {
                vector_float(opcode, a, b, offset)?
            } else {
                vector_integer(opcode, a, b)
            };
            (dst, d)
        }
        Insn::VUnary(opcode, dst, src0) => {
            check_destination(dst, pending_read, offset)?;
            let a = match src0 {
                CmpSrc::Sgpr(reg) => read_s(reg)?,
                CmpSrc::Vcc => read_vcc()?,
                CmpSrc::Imm(value) => value,
                CmpSrc::Vgpr(reg) => read_v(reg)?,
            };
            let d = if (33..=36).contains(&opcode) {
                vector_round(opcode, a, offset)?
            } else {
                vector_unary(opcode, a)
            };
            (dst, d)
        }
        // Guide 12.8: tmp = D; D = S0; S0 = tmp. Both are VGPR writes.
        Insn::VSwap(dst, src0) => {
            let (old_dst, old_src0) = (read_v(dst)?, read_v(src0)?);
            check_destination(dst, pending_read, offset)?;
            check_destination(src0, pending_read, offset)?;
            lane.vgpr[dst as usize] = Some(old_src0);
            lane.vgpr[src0 as usize] = Some(old_dst);
            return Ok(());
        }
        _ => unreachable!("scalar instruction in lane_step"),
    };
    lane.vgpr[value.0 as usize] = Some(value.1);
    Ok(())
}

/// D of VOP1 binary32 rounding opcode 33-36 (guide 12.8) for S0 = `bits`.
///
/// Widening to binary64 is exact, subnormals included (input denormals
/// allowed), and the binary64 integral result converts back exactly.
fn vector_round(opcode: u32, bits: u32, offset: usize) -> Result<u32, String> {
    let x = f64::from(f32::from_bits(bits));
    if x.is_nan() {
        return Err(format!(
            "{offset:#x}: NaN rounding input is outside the domain"
        ));
    }
    let rounded = match opcode {
        33 => x.trunc(),
        34 => x.ceil(),
        35 => x.round_ties_even(),
        _ => x.floor(),
    };
    Ok((rounded as f32).to_bits())
}

/// D of VOP1 integer opcode `opcode` (guide 12.8 pseudocode) for S0 = `value`.
fn vector_unary(opcode: u32, value: u32) -> u32 {
    // Bit i of S0, by integer division rather than a host shift.
    let bit = |i: u32| (value / 2_u32.pow(i)) % 2 == 1;
    // D = -1 unless the loop finds a position.
    let found = |position: Option<u32>| position.unwrap_or(u32::MAX);
    match opcode {
        55 => u32::MAX - value,
        // D[31:0] = S0[0:31]: S0 bit 0 becomes D bit 31.
        56 => (0..32).fold(0, |word, i| word * 2 + u32::from(bit(i))),
        57 => found((0..32).find(|&i| bit(31 - i))),
        58 => found((0..32).find(|&i| bit(i))),
        _ => found((1..32).find(|&i| bit(31 - i) != bit(31))),
    }
}

/// D of VOP2 integer opcode `opcode` (guide 12.7) for S0 = `a`, S1 = `b`.
fn vector_integer(opcode: u32, a: u32, b: u32) -> u32 {
    // Signed order equals unsigned order after flipping the sign bit.
    let below = |signed: bool| {
        let flip = if signed { 0x8000_0000 } else { 0 };
        (a ^ flip) < (b ^ flip)
    };
    match opcode {
        9 => product_word(multiply_field(a, b, 24, true), false),
        10 => product_word(multiply_field(a, b, 24, true), true),
        11 => product_word(multiply_field(a, b, 24, false), false),
        12 => product_word(multiply_field(a, b, 24, false), true),
        17 | 19 => {
            if below(opcode == 17) {
                a
            } else {
                b
            }
        }
        18 | 20 => {
            if below(opcode == 18) {
                b
            } else {
                a
            }
        }
        30 => (a & b) | (!a & !b),
        37 => ripple(a, b, false).0,
        38 => ripple(a, !b, true).0,
        _ => ripple(b, !a, true).0,
    }
}

/// The exact product of the low `width` bits of `a` and `b`, read as two's
/// complement when `signed`, by shift-and-add on the magnitudes.
fn multiply_field(a: u32, b: u32, width: u32, signed: bool) -> i128 {
    let read = |value: u32| {
        let low = i128::from(value) % (1_i128 << width);
        if signed && low >= 1_i128 << (width - 1) {
            low - (1_i128 << width)
        } else {
            low
        }
    };
    let (x, y) = (read(a), read(b));
    let (mx, my) = (x.unsigned_abs(), y.unsigned_abs());
    let mut magnitude = 0_u128;
    for bit in 0..width {
        if (my >> bit) & 1 == 1 {
            magnitude += mx << bit;
        }
    }
    let magnitude = magnitude as i128;
    if (x < 0) != (y < 0) {
        -magnitude
    } else {
        magnitude
    }
}

/// Bits 31:0 (or 63:32 when `high`) of `product` in 64-bit two's complement,
/// by Euclidean division rather than a shift.
fn product_word(product: i128, high: bool) -> u32 {
    let word = 1_i128 << 32;
    let part = if high {
        product.div_euclid(word)
    } else {
        product
    };
    part.rem_euclid(word) as u32
}

/// Bit-serial addition with carry-in; returns the sum and the carry-out.
fn ripple(a: u32, b: u32, carry_in: bool) -> (u32, bool) {
    let mut carry = carry_in;
    let mut sum = 0;
    for bit in 0..32 {
        let (x, y) = ((a >> bit) & 1 == 1, (b >> bit) & 1 == 1);
        sum |= u32::from(x ^ y ^ carry) << bit;
        carry = (x && y) || (carry && (x ^ y));
    }
    (sum, carry)
}

fn check_destination(dst: u8, load: Option<u8>, offset: usize) -> Result<(), String> {
    if dst == 0 || load == Some(dst) {
        return Err(format!("{offset:#x}: write to the index or pending VGPR"));
    }
    Ok(())
}

fn index_of(program: &[(usize, Insn)], target: usize, offset: usize) -> Result<usize, String> {
    program
        .iter()
        .position(|(at, _)| *at == target)
        .ok_or_else(|| format!("{offset:#x}: branch target {target:#x} is not a boundary"))
}

/// D of VOP2 binary32 opcode 3, 4, 5, 8, 15 or 16 (guide 12.7: ADD, SUB,
/// SUBREV and MUL on page 131, MIN on page 132, MAX on page 133) for S0 =
/// `a_bits`, S1 = `b_bits`, from host binary32 arithmetic and comparisons.
///
/// Refused: subnormal inputs (Metal flushes them; MODE.FP_DENORM is pinned
/// only for inputs), NaN inputs to the arithmetic forms, signalling NaN inputs
/// to MIN and MAX (IEEE_MODE and Quiet() are not pinned), and arithmetic
/// results that are NaN or whose exact value is nonzero and below 2^-126.
/// Binary64 holds the exact product, and the exact sum whenever it is that
/// small, so the size check is made on the binary64 result.
fn vector_float(opcode: u32, a_bits: u32, b_bits: u32, offset: usize) -> Result<u32, String> {
    let refuse = |what: &str| {
        Err(format!(
            "{offset:#x}: binary32 {what} is outside the domain"
        ))
    };
    let (a, b) = (f32::from_bits(a_bits), f32::from_bits(b_bits));
    if a.is_subnormal() || b.is_subnormal() {
        return refuse("subnormal input");
    }
    if opcode == 15 || opcode == 16 {
        let signalling = |x: f32, bits: u32| x.is_nan() && bits & 0x0040_0000 == 0;
        if signalling(a, a_bits) || signalling(b, b_bits) {
            return refuse("signalling NaN input");
        }
        // The pseudocode's order of cases.
        return Ok(if a.is_nan() {
            b_bits
        } else if b.is_nan() {
            a_bits
        } else if a == 0.0 && b == 0.0 && a.is_sign_negative() != b.is_sign_negative() {
            if opcode == 15 {
                (-0.0f32).to_bits()
            } else {
                0.0f32.to_bits()
            }
        } else if (opcode == 15 && a < b) || (opcode == 16 && a >= b) {
            a_bits
        } else {
            b_bits
        });
    }
    if a.is_nan() || b.is_nan() {
        return refuse("NaN input");
    }
    let (wide, d) = match opcode {
        3 => (f64::from(a) + f64::from(b), a + b),
        4 => (f64::from(a) - f64::from(b), a - b),
        5 => (f64::from(b) - f64::from(a), b - a),
        _ => (f64::from(a) * f64::from(b), a * b),
    };
    if d.is_nan() {
        return refuse("NaN result");
    }
    if wide != 0.0 && wide.abs() < f64::from(f32::MIN_POSITIVE) {
        return refuse("result below 2^-126");
    }
    Ok(d.to_bits())
}

fn decode(code: &[u8]) -> Result<Vec<(usize, Insn)>, String> {
    let mut program = Vec::new();
    let mut offset = 0;
    while offset < code.len() {
        let (insn, len) = decode_one(code, offset)?;
        if matches!(insn, Insn::End) && offset + len != code.len() {
            return Err(format!("{offset:#x}: code after s_endpgm"));
        }
        program.push((offset, insn));
        offset += len;
    }
    for &(at, insn) in &program {
        if let Insn::Branch(target) | Insn::BranchScc0(target) | Insn::BranchMask { target, .. } =
            insn
            && !program.iter().any(|(boundary, _)| *boundary == target)
        {
            return Err(format!(
                "{at:#x}: branch target {target:#x} is not a boundary"
            ));
        }
    }
    Ok(program)
}

fn decode_one(code: &[u8], offset: usize) -> Result<(Insn, usize), String> {
    let first = word(code, offset)?;
    let fail = |detail: &str| format!("{offset:#x}: {detail}");

    // The synthetic ABI permits only idxen with zero offsets. Only the
    // data VGPR varies across each complete eight-byte encoding.
    if matches!(first, 0xe030_2000 | 0xe070_2000) {
        let load = first == 0xe030_2000;
        let expected = if load { 0x8000_0000 } else { 0x8001_0000 };
        let second = word(code, offset + 4)?;
        if second & 0xffff_00ff != expected {
            return Err(fail("buffer operand outside synthetic ABI"));
        }
        let data = (second >> 8) as u8;
        return Ok((
            if load {
                Insn::Load(data)
            } else {
                Insn::Store(data)
            },
            8,
        ));
    }
    if first & 0xfe01_fe00 == 0x7e00_0200 {
        let dst = (first >> 17) as u8;
        let selector = first & 0x1ff;
        if selector <= 105 {
            return Ok((Insn::VMovFromSgpr(dst, selector as u8), 4));
        }
        if selector == 106 {
            return Ok((Insn::VMovFromVcc(dst), 4));
        }
        let (src, len) = if selector == 255 {
            (Src::Imm(word(code, offset + 4)?), 8)
        } else {
            (
                vector_source(selector).ok_or_else(|| fail("bad VOP1 source"))?,
                4,
            )
        };
        return Ok((Insn::VMov(dst, src), len));
    }
    // VOP1 integer forms: bits 31:25 = 0b0111111, VDST 24:17, OP 16:9, SRC0 8:0.
    if first >> 25 == 0b011_1111 {
        let dst = (first >> 17) as u8;
        match (first >> 9) & 0xff {
            opcode @ (33..=36 | 55..=59) => {
                if opcode <= 36 && (128..=208).contains(&(first & 0x1ff)) {
                    return Err(fail("integer inline constant as a binary32 source"));
                }
                let (src0, len) = vector_src0(code, offset, first, "VOP1")?;
                return Ok((Insn::VUnary(opcode, dst, src0), len));
            }
            // SRC0 is written back, so only a VGPR is meaningful.
            101 => {
                return match vector_src0(code, offset, first, "VOP1")? {
                    (CmpSrc::Vgpr(src0), _) => Ok((Insn::VSwap(dst, src0), 4)),
                    _ => Err(fail("v_swap_b32 needs a VGPR SRC0")),
                };
            }
            _ => {}
        }
    }
    // VOPC: bits 31:25 = 0b0111110, OP 24:17, VSRC1 16:9, SRC0 8:0.
    if first >> 25 == 0b011_1110 {
        let opcode = (first >> 17) & 0xff;
        // Guide 12.9 Table 61 opcode groups (16-bit groups lack F and T).
        let (signed, bits, relation) = match opcode {
            128..=135 => (true, 32, opcode - 128),
            192..=199 => (false, 32, opcode - 192),
            137..=142 => (true, 16, opcode - 136),
            169..=174 => (false, 16, opcode - 168),
            160..=167 => (true, 64, opcode - 160),
            224..=231 => (false, 64, opcode - 224),
            _ => return Err(fail("unsupported VOPC opcode")),
        };
        let (src0, len) = vector_src0(code, offset, first, "VOPC")?;
        let src1 = (first >> 9) as u8;
        // Accepted subset: no literal for 16-bit forms; 64-bit forms take an
        // even SGPR pair below VCC, a VGPR pair below v255 or an inline integer.
        let allowed = match (bits, src0) {
            (16 | 64, CmpSrc::Imm(_)) if len == 8 => false,
            (64, CmpSrc::Vcc) => false,
            (64, CmpSrc::Sgpr(reg)) => reg % 2 == 0 && src1 != 255,
            (64, CmpSrc::Vgpr(reg)) => reg != 255 && src1 != 255,
            (64, _) => src1 != 255,
            _ => true,
        };
        if !allowed {
            return Err(fail("unsupported VOPC operand"));
        }
        let insn = Insn::VCmp {
            dst: 106,
            relation: relation as u8,
            signed,
            bits,
            src0,
            src1,
        };
        return Ok((insn, len));
    }
    // VOP2 integer forms: bit 31 = 0, OP 30:25, VDST 24:17, VSRC1 16:9, SRC0 8:0.
    let vop2_op = (first >> 25) & 0x3f;
    if first >> 31 == 0 && matches!(vop2_op, 1 | 40..=42) {
        let (src0, len) = vector_src0(code, offset, first, "VOP2")?;
        return Ok((
            Insn::VIntVcc(vop2_op, (first >> 17) as u8, src0, (first >> 9) as u8),
            len,
        ));
    }
    // Binary32 forms: integer inline constants (128-208) are refused.
    if first >> 31 == 0
        && matches!(vop2_op, 3..=5 | 8 | 15 | 16)
        && (128..=208).contains(&(first & 0x1ff))
    {
        return Err(fail("integer inline constant as a binary32 source"));
    }
    if first >> 31 == 0 && matches!(vop2_op, 3..=5 | 8..=12 | 15..=20 | 30 | 37..=39) {
        let (src0, len) = vector_src0(code, offset, first, "VOP2")?;
        let insn = Insn::VInt(vop2_op, (first >> 17) as u8, src0, (first >> 9) as u8);
        return Ok((insn, len));
    }
    if first >> 26 == 0b110101 {
        let dst = first as u8;
        let opcode = (first >> 16) & 0x3ff;
        let second = word(code, offset + 4).map_err(|_| fail("truncated VOP3A"))?;
        let src0 = second & 0x1ff;
        let src1 = (second >> 9) & 0x1ff;
        if first & 0xff00 != 0 || second >> 18 != 128 || src0 < 256 || src1 < 256 {
            return Err(fail("unsupported VOP3A modifiers or sources"));
        }
        // Guide §§3.9, 12.7.1 and 12.9.1: re-encoded VOP2 writes a VGPR,
        // while VOPC uses the destination field for the scalar mask.
        let (src0, src1) = (src0 as u8, src1 as u8);
        let insn = match opcode {
            128..=135 | 192..=199 if dst <= 106 => Insn::VCmp {
                dst,
                relation: (opcode % 8) as u8,
                signed: opcode < 192,
                bits: 32,
                src0: CmpSrc::Vgpr(src0),
                src1,
            },
            293 => Insn::VAddNc(dst, src0, src1),
            265..=268 | 273..=276 | 286 | 294..=295 => {
                Insn::VInt(opcode - 256, dst, CmpSrc::Vgpr(src0), src1)
            }
            _ => return Err(fail("unsupported VOP3A opcode or destination")),
        };
        return Ok((insn, 8));
    }
    // SOP1: ENCODING 0b10_1111101 in bits 23-31, SDST 16-22, OP 8-15.
    let sop1_op = (first >> 8) & 0xff;
    if first >> 23 == 0b1_0111_1101
        && matches!(sop1_op, 7 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 | 26 | 52)
    {
        let dst = ((first >> 16) & 0x7f) as u8;
        if dst > 105 {
            return Err(fail("unsupported SOP1 destination"));
        }
        let (src, len) = scalar_source(code, offset, first as u8)?;
        return Ok((Insn::SUnary(sop1_op, dst, src), len));
    }
    // The twelve SAVEEXEC and WREXEC forms share the SOP1 encoding. Guide
    // 12.3, printed pages 120-122: the destination is a normal SGPR, so
    // EXEC_LO and EXEC_HI are refused as destinations.
    if first >> 23 == 0b1_0111_1101 && (60..=71).contains(&sop1_op) {
        let dst = ((first >> 16) & 0x7f) as u8;
        if dst > 105 {
            return Err(fail("unsupported SOP1 destination"));
        }
        let (src, len) = scalar_source(code, offset, first as u8)?;
        return Ok((Insn::SSaveExec(sop1_op, dst, src), len));
    }
    if first & 0xff80_ff00 == 0xbe80_0300 {
        let dst = ((first >> 16) & 0x7f) as u8;
        if dst > 107 && dst < 126 {
            return Err(fail("unsupported SOP1 destination"));
        }
        let (src, len) = scalar_source(code, offset, first as u8)?;
        return Ok((Insn::SMov(dst, src), len));
    }
    // SOPK: ENCODING 0b1011 in bits 28-31, OP 23-27, SDST 16-22, SIMM16 0-15
    // (guide 13.1.2). Guide 12.2: no literal dword may follow.
    if first >> 28 == 0b1011 && matches!((first >> 23) & 0x1f, 0 | 2..=16) {
        let dst = ((first >> 16) & 0x7f) as u8;
        if dst > 105 {
            return Err(fail("unsupported SOPK register"));
        }
        return Ok((Insn::SImm((first >> 23) & 0x1f, dst, first as u16), 4));
    }
    // SOP2: ENCODING 0b10 in bits 30-31, OP in bits 23-29, SDST in 16-22.
    if first >> 30 == 0b10
        && matches!(
            (first >> 23) & 0x7f,
            0..=10
                | 14
                | 16
                | 18
                | 20
                | 22
                | 24
                | 26
                | 28
                | 30
                | 32
                | 34
                | 36
                | 38..=40
                | 44
                | 46..=49
                | 53
                | 54
        )
    {
        let dst = ((first >> 16) & 0x7f) as u8;
        if dst > 105 {
            return Err(fail("unsupported SOP2 destination"));
        }
        let (a, b) = (first as u8, (first >> 8) as u8);
        if a == 255 && b == 255 {
            return Err(fail("two scalar literals"));
        }
        let (sa, la) = scalar_source(code, offset, a)?;
        let (sb, lb) = scalar_source(code, offset, b)?;
        return Ok((Insn::SAlu((first >> 23) & 0x7f, dst, sa, sb), la.max(lb)));
    }
    // SOPC: ENCODING 0b10_1111110 in bits 23-31, OP in 16-22.
    if first >> 23 == 0b1_0111_1110 && (first >> 16) & 0x7f <= 11 {
        let a = first as u8;
        let b = (first >> 8) as u8;
        if a == 255 && b == 255 {
            return Err(fail("two scalar literals"));
        }
        let (sa, la) = scalar_source(code, offset, a)?;
        let (sb, lb) = scalar_source(code, offset, b)?;
        return Ok((Insn::SCmp((first >> 16) & 0x7f, sa, sb), la.max(lb)));
    }
    if first & 0xffff_0000 == 0xbf82_0000 {
        return Ok((Insn::Branch(branch_target(first, offset)?), 4));
    }
    if first & 0xffff_0000 == 0xbf84_0000 {
        return Ok((Insn::BranchScc0(branch_target(first, offset)?), 4));
    }
    let tested_register = match first >> 16 {
        0xbf86 | 0xbf87 => Some(106),
        0xbf88 => Some(126),
        _ => None,
    };
    if let Some(register) = tested_register {
        return Ok((
            Insn::BranchMask {
                target: branch_target(first, offset)?,
                register,
                nonzero: first >> 16 == 0xbf87,
            },
            4,
        ));
    }
    for (mask, value, kind) in [
        (0xfe00_0000, 0x3600_0000u32, 0u8),
        (0xfe00_0000, 0x3a00_0000, 1),
        (0xfe00_0000, 0x3800_0000, 4),
        (0xfe00_0000, 26 << 25, 5),
        (0xfe00_0000, 22 << 25, 6),
        (0xfe00_0000, 24 << 25, 7),
    ] {
        if first & mask == value {
            if first & 0x100 == 0 {
                return Err(fail("VOP2 source modifier"));
            }
            let dst = (first >> 17) as u8;
            let a = first as u8;
            let b = (first >> 9) as u8;
            let insn = match kind {
                0 => Insn::VAnd(dst, a, b),
                1 => Insn::VXor(dst, a, b),
                4 => Insn::VOr(dst, a, b),
                5 => Insn::VShiftLeft(dst, a, b),
                6 => Insn::VShiftRight(dst, a, b),
                _ => Insn::VShiftRightArithmetic(dst, a, b),
            };
            return Ok((insn, 4));
        }
    }
    if first & 0xffff_0000 == 0xbf8c_0000 {
        return match first {
            0xbf8c_3f70 => Ok((Insn::WaitRead, 4)),
            0xbf8c_c07f => Ok((Insn::WaitScalar, 4)),
            _ => Err(fail("waitcnt not vmcnt(0) or lgkmcnt(0)")),
        };
    }
    if first & 0xff80_0000 == 0xbb80_0000 {
        if first != 0xbbfd_0000 {
            return Err(fail("waitcnt_vscnt not null,0"));
        }
        return Ok((Insn::WaitWrite, 4));
    }
    if first & 0xffff_0000 == 0xbf81_0000 {
        if first != 0xbf81_0000 {
            return Err(fail("endpgm immediate"));
        }
        return Ok((Insn::End, 4));
    }
    // SMEM s_buffer_load_dword/_x2/_x4 (guide 13.2.1, Tables 73 and 74). The
    // synthetic ABI uses the input buffer as s[0:3] with no additional offset.
    if first & 0xff00_0000 == 0xf400_0000 {
        let opcode = (first >> 16) & 0xff;
        let dwords = match opcode {
            0x20 => 1u8,
            0x24 => 2,
            0x28 => 4,
            _ => return Err(fail("unsupported scalar load width")),
        };
        let low = first & 0xff;
        let middle = (first >> 8) & 0x1f;
        let data = ((middle << 2) | (low >> 6)) as u8;
        let resource = ((low & 0x3f) * 2) as u8;
        let second = word(code, offset + 4)?;
        let byte_offset = second & 0x00ff_ffff;
        let selector = (second >> 24) as u8;
        if resource != 0 || selector >> 1 != 125 {
            return Err(fail("scalar load outside the synthetic ABI"));
        }
        if u16::from(data) + u16::from(dwords) > 106 {
            return Err(fail("scalar load register span"));
        }
        return Ok((
            Insn::SLoad {
                dst: data,
                dwords,
                offset: byte_offset,
            },
            8,
        ));
    }
    // MIMG (guide 8.2 and 13.9 Table 100, printed page 277): byte 3 (bits
    // 24-31) carries the ENCODING 0b111100 with the opcode's high bit and SLC
    // clear. Only the two explicit-LOD sample forms are accepted.
    if first >> 24 == 0xf0 {
        let opcode = ((first & 1) << 7) | ((first >> 18) & 0x7f);
        let explicit_lod = match opcode {
            36 => true,
            39 => false,
            _ => return Err(fail("unsupported MIMG opcode")),
        };
        // Non-NSA, DIM 2D, DLC/UNORM/GLC/DA/R128/TFE/LWE and the reserved bit
        // clear, DMASK nonzero.
        if first & 0x0203_f0fe != 0x8 || (first >> 8) & 0xf == 0 {
            return Err(fail("unsupported MIMG encoding or data mask"));
        }
        let second = word(code, offset + 4).map_err(|_| fail("truncated MIMG"))?;
        if second & 0xfc00_0000 != 0 {
            return Err(fail("unsupported MIMG A16/D16"));
        }
        let dmask = ((first >> 8) & 0xf) as u8;
        let coord = (second & 0xff) as u8;
        let dst = ((second >> 8) & 0xff) as u8;
        let resource = (((second >> 16) & 0x1f) * 4) as u8;
        let sampler = (((second >> 21) & 0x1f) * 4) as u8;
        if resource > 98 || sampler > 102 {
            return Err(fail("unsupported MIMG descriptor registers"));
        }
        let components = u16::from(dmask.count_ones() as u8);
        let address = if explicit_lod { 3 } else { 2 };
        if u16::from(dst) + components - 1 > 255 || u16::from(coord) + address - 1 > 255 {
            return Err(fail("unsupported MIMG register span"));
        }
        return Ok((
            Insn::ImageSample {
                dst,
                coord,
                resource,
                sampler,
                dmask,
                explicit_lod,
            },
            8,
        ));
    }
    Err(fail("unsupported opcode"))
}

/// SRC0 (bits 8:0 of `first`) of a VOPC or VOP2 integer word: an SGPR,
/// `vcc_lo`, an inline integer, a literal dword or a VGPR.
fn vector_src0(
    code: &[u8],
    offset: usize,
    first: u32,
    what: &str,
) -> Result<(CmpSrc, usize), String> {
    let selector = first & 0x1ff;
    Ok(match selector {
        0..=105 => (CmpSrc::Sgpr(selector as u8), 4),
        106 => (CmpSrc::Vcc, 4),
        128..=192 => (CmpSrc::Imm(selector - 128), 4),
        193..=208 => (CmpSrc::Imm(192u32.wrapping_sub(selector)), 4),
        255 => (
            CmpSrc::Imm(
                word(code, offset + 4)
                    .map_err(|_| format!("{offset:#x}: truncated {what} literal"))?,
            ),
            8,
        ),
        256..=511 => (CmpSrc::Vgpr(selector as u8), 4),
        _ => return Err(format!("{offset:#x}: bad {what} source")),
    })
}

fn vector_source(selector: u32) -> Option<Src> {
    match selector {
        128..=192 => Some(Src::Imm(selector - 128)),
        193..=208 => Some(Src::Imm(192u32.wrapping_sub(selector))),
        242 => Some(Src::Imm(0x3f80_0000)),
        256..=511 => Some(Src::Reg(selector as u8)),
        _ => None,
    }
}

fn scalar_source(code: &[u8], offset: usize, selector: u8) -> Result<(Src, usize), String> {
    match selector {
        0..=105 => Ok((Src::Reg(selector), 4)),
        128..=192 => Ok((Src::Imm(u32::from(selector - 128)), 4)),
        193..=208 => Ok((Src::Imm(192u32.wrapping_sub(u32::from(selector))), 4)),
        255 => Ok((Src::Imm(word(code, offset + 4)?), 8)),
        _ => Err(format!("{offset:#x}: bad scalar source")),
    }
}

fn branch_target(first: u32, offset: usize) -> Result<usize, String> {
    let displacement = first as u16 as i16;
    let words =
        usize::try_from(displacement).map_err(|_| format!("{offset:#x}: backward branch"))?;
    Ok(offset + 4 + words * 4)
}

fn word(code: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = code
        .get(offset..offset + 4)
        .ok_or_else(|| format!("{offset:#x}: truncated instruction"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
