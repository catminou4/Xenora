use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use xenora_decoder::{
    Architecture, CompareRelation, DecodeError, DecodeErrorKind, DecodedInstruction, Instruction,
    ScalarAluOp, ScalarImmediateOp, ScalarSource, ScalarUnaryOp, VectorAluOp, VectorAluVop3Op,
    VectorCompareOp, VectorCompareVop3Op, VectorSource, VectorUnaryOp, decode, decode_at,
};
use xenora_isa::semantics::Signedness;
use xenora_ref::{
    BranchCondition, CompareSource, CompareWidth, Comparison, Op, ScalarAlu, ScalarImmediate,
    ScalarUnary, Source, VectorAlu, VectorFloat, VectorInteger, VectorUnary,
};

const FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/add.bin");
const BITWISE_FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/bitwise-branch.bin");
const SHIFT_FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/shifts.bin");
const OR_FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/or.bin");
const FLOAT_FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/float-exact.bin");

#[test]
fn decodes_each_fixture_instruction_with_offset_and_length() {
    let instructions = decode(Architecture::Gfx1030, FIXTURE).unwrap();
    let wanted = [
        (
            0,
            8,
            Instruction::BufferLoadDword {
                data: 1,
                index: 0,
                resource: 0,
            },
        ),
        (8, 4, Instruction::WaitLoad),
        (
            12,
            8,
            Instruction::MoveB32 {
                dst: 2,
                src: VectorSource::Immediate(0x1234_5678),
            },
        ),
        (
            20,
            8,
            Instruction::VectorAluVop3 {
                op: VectorAluVop3Op::AddNcU32,
                dst: 1,
                src0: 1,
                src1: 2,
            },
        ),
        (
            28,
            8,
            Instruction::BufferStoreDword {
                data: 1,
                index: 0,
                resource: 4,
            },
        ),
        (36, 4, Instruction::WaitStore),
        (40, 4, Instruction::EndProgram),
    ]
    .map(|(offset, byte_len, instruction)| DecodedInstruction {
        offset,
        byte_len,
        instruction,
    });
    assert_eq!(instructions, wanted);
}

#[test]
fn decodes_wave32_vop3_compares_from_llvm_words() {
    use VectorCompareVop3Op::*;

    // llvm-mc -triple=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize32 -show-encoding
    let cases: [(u32, VectorCompareVop3Op, u8); 19] = [
        // v_cmp_lt_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x81,0xd4,0x01,0x05,0x02,0x02]
        (0xd481_0004, CmpLtI32, 4),
        // v_cmp_eq_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x82,0xd4,0x01,0x05,0x02,0x02]
        (0xd482_0004, CmpEqI32, 4),
        // v_cmp_le_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x83,0xd4,0x01,0x05,0x02,0x02]
        (0xd483_0004, CmpLeI32, 4),
        // v_cmp_gt_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x84,0xd4,0x01,0x05,0x02,0x02]
        (0xd484_0004, CmpGtI32, 4),
        // v_cmp_ne_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x85,0xd4,0x01,0x05,0x02,0x02]
        (0xd485_0004, CmpNeI32, 4),
        // v_cmp_ge_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x86,0xd4,0x01,0x05,0x02,0x02]
        (0xd486_0004, CmpGeI32, 4),
        // v_cmp_lt_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc1,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c1_0004, CmpLtU32, 4),
        // v_cmp_eq_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc2,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c2_0004, CmpEqU32, 4),
        // v_cmp_le_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc3,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c3_0004, CmpLeU32, 4),
        // v_cmp_gt_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc4,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c4_0004, CmpGtU32, 4),
        // v_cmp_ne_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc5,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c5_0004, CmpNeU32, 4),
        // v_cmp_ge_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc6,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c6_0004, CmpGeU32, 4),
        // v_cmp_f_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x80,0xd4,0x01,0x05,0x02,0x02]
        (0xd480_0004, CmpFI32, 4),
        // v_cmp_t_i32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0x87,0xd4,0x01,0x05,0x02,0x02]
        (0xd487_0004, CmpTI32, 4),
        // v_cmp_f_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc0,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c0_0004, CmpFU32, 4),
        // v_cmp_t_u32_e64 s4, v1, v2 ; encoding: [0x04,0x00,0xc7,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c7_0004, CmpTU32, 4),
        // v_cmp_eq_u32_e64 s0, v1, v2 ; encoding: [0x00,0x00,0xc2,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c2_0000, CmpEqU32, 0),
        // v_cmp_eq_u32_e64 s105, v1, v2 ; encoding: [0x69,0x00,0xc2,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c2_0069, CmpEqU32, 105),
        // v_cmp_eq_u32_e64 vcc_lo, v1, v2 ; encoding: [0x6a,0x00,0xc2,0xd4,0x01,0x05,0x02,0x02]
        (0xd4c2_006a, CmpEqU32, 106),
    ];
    for (first, op, sdst) in cases {
        let code: Vec<_> = [first, 0x0202_0501]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let decoded = decode(Architecture::Gfx1030, &code).unwrap();
        let dst = if sdst == 106 {
            "vcc_lo".to_owned()
        } else {
            format!("s{sdst}")
        };
        assert_eq!(
            decoded,
            [DecodedInstruction {
                offset: 0,
                byte_len: 8,
                instruction: Instruction::VectorCompareVop3 {
                    op,
                    sdst,
                    src0: 1,
                    src1: 2
                },
            }],
        );
        assert_eq!(
            decoded[0].instruction.to_string(),
            format!("{} {dst}, v1, v2", op.assembly_name())
        );
    }
}

#[test]
fn rejects_noncanonical_wave32_vop3_compare_operands() {
    // v_cmp_eq_u32_e64 s4, v1, v2
    let first = 0xd4c2_0004;
    let second = 0x0202_0501;
    let rejects = |first, second| {
        let code: Vec<_> = [first, second]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let error = decode(Architecture::Gfx1030, &code).unwrap_err();
        assert_eq!(error.offset, 0);
        assert!(
            matches!(
                error.kind,
                DecodeErrorKind::UnsupportedForm {
                    mnemonic: "v_cmp_eq_u32_e64",
                    ..
                }
            ),
            "{first:#010x} {second:#010x}: {error}"
        );
    };

    // Includes vcc_hi (107), TTMPs (108-123), null (124), m0 (125) and EXEC (126-127).
    for sdst in 107..=255 {
        rejects((first & !0xff) | sdst, second);
    }
    for bit in 8..=15 {
        // ABS, OP_SEL and CLAMP.
        rejects(first | (1 << bit), second);
    }
    for bit in 27..=31 {
        // OMOD and NEG.
        rejects(first, second | (1 << bit));
    }
    rejects(first, second ^ (1 << 18)); // SRC2 = 129 instead of 128.
    for shift in [0, 9] {
        for selector in [4, 106, 128, 240, 255] {
            // SGPR, VCC_LO, inline integer/float, literal.
            rejects(first, (second & !(0x1ff << shift)) | (selector << shift));
        }
    }
}

#[test]
fn every_non_boundary_prefix_reports_the_incomplete_instruction() {
    let lengths = [8, 4, 8, 8, 8, 4, 4];
    let mut offset = 0;
    for len in lengths {
        for available in 1..len {
            let prefix = &FIXTURE[..offset + usize::from(available)];
            let needed = if available < 4 { 4 } else { len };
            assert_eq!(
                decode(Architecture::Gfx1030, prefix).unwrap_err(),
                DecodeError {
                    offset,
                    kind: DecodeErrorKind::Truncated {
                        needed,
                        available: usize::from(available),
                    },
                }
            );
            assert_eq!(xenora_ref::decode(prefix).unwrap_err().offset, offset);
        }
        offset += usize::from(len);
    }
}

#[test]
fn rejects_other_opcodes_and_noncanonical_operands() {
    let cases: &[(&[u32], &str)] = &[
        (&[0xe030_0000, 0x8000_0100], "buffer_load_dword"),
        (&[0xe070_2000, 0x8101_0100], "buffer_store_dword"),
        (&[0xe070_2000, 0x801b_0100], "buffer_store_dword"),
        (&[0xbf8c_3f71], "s_waitcnt"),
        (&[0xbbfd_0001], "s_waitcnt_vscnt"),
        (&[0x7e04_026b], "v_mov_b32_e32"), // VCC_HI is outside the source subset.
        (&[0x7d02_046b], "v_cmp_lt_i32_e32"), // VCC_HI source.
        (&[0x7d02_047e], "v_cmp_lt_i32_e32"), // EXEC_LO source.
        (&[0x7d02_04f0], "v_cmp_lt_i32_e32"), // Inline 0.5 source.
        (&[0x7d84_04f9, 0x0606_0601], "v_cmp_eq_u32_e32"), // SDWA.
        (&[0x7d84_04fa, 0], "v_cmp_eq_u32_e32"), // DPP16.
        (&[0x7e04_02fa, 0], "v_mov_b32_e32"),
        (&[0xd525_8001, 0x0202_0501], "v_add_nc_u32_e64"),
        (&[0xd525_0001, 0x0202_0480], "v_add_nc_u32_e64"),
        (&[0xbf81_0001], "s_endpgm"),
        (&[0xbefc_0387], "s_mov_b32"),
        (&[0xbf06_87fe], "s_cmp_eq_u32"),
        (&[0xbf06_ffff, 0], "s_cmp_eq_u32"),
        (&[0xbf82_ffff], "s_branch"),
        (&[0xbf84_fffe], "s_cbranch_scc0"),
        (&[0x3602_0680], "v_and_b32_e32"),
        (&[0x3803_fe80], "v_or_b32_e32"),
        (&[0x3803_fe08], "v_or_b32_e32"),
        (&[0x3803_feff, 0xdead_beef], "v_or_b32_e32"),
        (&[0x3803_fefa, 0xff00_e401], "v_or_b32_e32"),
        (&[0x3803_fef9, 0x0606_0601], "v_or_b32_e32"),
        (&[0x3402_0480], "v_lshlrev_b32_e32"),
        (&[0x2dff_fe08], "v_lshrrev_b32_e32"),
        (&[0x3402_04ff, 0xdead_beef], "v_lshlrev_b32_e32"),
        (&[0x2dff_fefa, 0xff00_e401], "v_lshrrev_b32_e32"),
        (&[0x3402_04f9, 0x0606_0601], "v_lshlrev_b32_e32"),
        (&[0x1002_06f2], "v_mul_f32_e32"),
    ];
    for (words, mnemonic) in cases {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let error = decode(Architecture::Gfx1030, &bytes).unwrap_err();
        assert_eq!(error.offset, 0);
        assert_eq!(xenora_ref::decode(&bytes).unwrap_err().offset, 0);
        assert!(matches!(
            error.kind,
            DecodeErrorKind::UnsupportedForm { mnemonic: actual, .. } if actual == *mnemonic
        ));
    }

    assert_eq!(xenora_ref::decode(&[0xff; 4]).unwrap_err().offset, 0);
    assert_eq!(
        decode(Architecture::Gfx1030, &[0xff; 4]).unwrap_err().kind,
        DecodeErrorKind::UnsupportedOpcode { word: 0xffff_ffff }
    );
}

#[test]
fn decodes_image_sample_lz_and_rejects_malformed_variants() {
    // image_sample_lz v[4:7], v[2:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    // bytes: [0x08, 0x0f, 0x9c, 0xf0, 0x02, 0x04, 0x40, 0x00]
    let words = [0xf09c_0f08_u32, 0x0040_0402_u32];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let decoded = decode(Architecture::Gfx1030, &bytes).expect("decode image_sample_lz");
    assert_eq!(decoded.len(), 1);
    assert_eq!(
        decoded[0],
        DecodedInstruction {
            offset: 0,
            byte_len: 8,
            instruction: Instruction::ImageSampleLz {
                dst: 4,
                coord: 2,
                resource: 0,
                sampler: 8,
                dmask: 0xf,
            },
        }
    );

    // A nonzero DMASK below 0xf selects a single component (guide Table 41).
    let single = [0xf09c_0108_u32, 0x0040_0402_u32];
    let single_bytes: Vec<u8> = single.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(
        decode(Architecture::Gfx1030, &single_bytes).expect("decode image_sample_lz dmask:0x1")[0]
            .instruction,
        Instruction::ImageSampleLz {
            dst: 4,
            coord: 2,
            resource: 0,
            sampler: 8,
            dmask: 0x1,
        }
    );

    // image_sample_l carries an explicit LOD in a third address VGPR.
    let sample_l = [0xf090_0f08_u32, 0x0040_0402_u32];
    let sample_l_bytes: Vec<u8> = sample_l.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(
        decode(Architecture::Gfx1030, &sample_l_bytes).expect("decode image_sample_l")[0]
            .instruction,
        Instruction::ImageSampleL {
            dst: 4,
            coord: 2,
            resource: 0,
            sampler: 8,
            dmask: 0xf,
        }
    );

    // Negative tests:
    // 1. Truncated (less than 8 bytes)
    assert!(matches!(
        decode(Architecture::Gfx1030, &bytes[..4]).unwrap_err().kind,
        DecodeErrorKind::Truncated {
            needed: 8,
            available: 4
        }
    ));

    // 2. Unsupported dim (dim != 1, e.g. dim = 0, bit 3 clear)
    let bad_dim = [0xf09c_0f00_u32, 0x0040_0402_u32];
    let bad_dim_bytes: Vec<u8> = bad_dim.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(matches!(
        decode(Architecture::Gfx1030, &bad_dim_bytes)
            .unwrap_err()
            .kind,
        DecodeErrorKind::UnsupportedForm {
            mnemonic: "image_sample_lz",
            ..
        }
    ));

    // 3. Unsupported dmask (dmask = 0 selects no component)
    let bad_dmask = [0xf09c_0008_u32, 0x0040_0402_u32];
    let bad_dmask_bytes: Vec<u8> = bad_dmask.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(matches!(
        decode(Architecture::Gfx1030, &bad_dmask_bytes)
            .unwrap_err()
            .kind,
        DecodeErrorKind::UnsupportedForm {
            mnemonic: "image_sample_lz",
            ..
        }
    ));

    // 4. Modifiers set (e.g. A16 bit 30 set in second word)
    let bad_a16 = [0xf09c_0f08_u32, 0x4040_0402_u32];
    let bad_a16_bytes: Vec<u8> = bad_a16.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(matches!(
        decode(Architecture::Gfx1030, &bad_a16_bytes)
            .unwrap_err()
            .kind,
        DecodeErrorKind::UnsupportedForm {
            mnemonic: "image_sample_lz",
            ..
        }
    ));

    // 5. Out of range SGPR for resource (resource > 98)
    // 25 << 16 * 4 = 100 > 98
    let bad_res = [0xf09c_0f08_u32, 0x0059_0402_u32];
    let bad_res_bytes: Vec<u8> = bad_res.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(matches!(
        decode(Architecture::Gfx1030, &bad_res_bytes)
            .unwrap_err()
            .kind,
        DecodeErrorKind::UnsupportedForm {
            mnemonic: "image_sample_lz",
            ..
        }
    ));
}

#[test]
fn rejects_targets_at_eof_or_inside_a_literal() {
    for (words, target) in [
        ([0xbf82_0002u32, 0x7e04_02ff, 0x1234_5678], 12),
        ([0xbf84_0001u32, 0x7e04_02ff, 0x1234_5678], 8),
    ] {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(
            decode(Architecture::Gfx1030, &bytes).unwrap_err(),
            DecodeError {
                offset: 0,
                kind: DecodeErrorKind::InvalidBranchTarget { target },
            }
        );
    }
}

#[test]
fn accepts_zero_displacement_to_the_following_instruction() {
    let bytes = [0, 0, 0x82, 0xbf, 0, 0, 0x81, 0xbf];
    let instructions = decode(Architecture::Gfx1030, &bytes).unwrap();
    assert_eq!(
        instructions[0].instruction,
        Instruction::Branch { target: 4 }
    );
}

#[test]
fn scalar_sources_keep_literal_width_and_register_class() {
    // s_mov_b32 s10, 0x11223344 and s_cmp_eq_u32 s10, 0x11223344.
    let words = [0xbe8a_03ffu32, 0x1122_3344, 0xbf06_ff0a, 0x1122_3344];
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let decoded = decode(Architecture::Gfx1030, &bytes).unwrap();
    assert_eq!(decoded[0].byte_len, 8);
    assert_eq!(decoded[1].offset, 8);
    assert_eq!(decoded[1].byte_len, 8);
    assert_eq!(
        decoded[1].instruction.to_string(),
        "s_cmp_eq_u32 s10, 0x11223344"
    );
    assert_eq!(
        decode(Architecture::Gfx1030, &bytes[..12]).unwrap_err(),
        DecodeError {
            offset: 8,
            kind: DecodeErrorKind::Truncated {
                needed: 8,
                available: 4
            },
        }
    );
}

fn reference_source(vector: VectorSource) -> Source {
    match vector {
        VectorSource::Register(reg) => Source::Register(reg),
        VectorSource::Immediate(bits) => Source::Immediate(bits),
        VectorSource::Scalar(_) => unreachable!("mapped to Op::MoveScalar"),
    }
}

fn reference_scalar(scalar: ScalarSource) -> Source {
    match scalar {
        ScalarSource::Register(reg) => Source::Register(reg),
        ScalarSource::Immediate(bits) => Source::Immediate(bits),
    }
}

fn reference_comparison(op: VectorCompareOp) -> Comparison {
    Comparison {
        offset: match op.relation() {
            CompareRelation::Less => 1,
            CompareRelation::Equal => 2,
            CompareRelation::LessOrEqual => 3,
            CompareRelation::Greater => 4,
            CompareRelation::NotEqual => 5,
            CompareRelation::GreaterOrEqual => 6,
            CompareRelation::False => 0,
            CompareRelation::True => 7,
        },
        signed: op.signed(),
    }
}

fn reference_vector_integer(op: VectorAluOp) -> VectorInteger {
    match op {
        VectorAluOp::CndmaskB32 => VectorInteger::Select,
        VectorAluOp::AddCoCiU32 => VectorInteger::AddCarry,
        VectorAluOp::SubCoCiU32 => VectorInteger::SubtractBorrow { reversed: false },
        VectorAluOp::SubrevCoCiU32 => VectorInteger::SubtractBorrow { reversed: true },
        VectorAluOp::MulI32I24 => VectorInteger::Multiply24 { signed: true },
        VectorAluOp::MulHiI32I24 => VectorInteger::MultiplyHigh24 { signed: true },
        VectorAluOp::MulU32U24 => VectorInteger::Multiply24 { signed: false },
        VectorAluOp::MulHiU32U24 => VectorInteger::MultiplyHigh24 { signed: false },
        VectorAluOp::MinI32 => VectorInteger::Minimum { signed: true },
        VectorAluOp::MaxI32 => VectorInteger::Maximum { signed: true },
        VectorAluOp::MinU32 => VectorInteger::Minimum { signed: false },
        VectorAluOp::MaxU32 => VectorInteger::Maximum { signed: false },
        VectorAluOp::XnorB32 => VectorInteger::Xnor,
        VectorAluOp::AddNcU32 => VectorInteger::Add,
        VectorAluOp::SubNcU32 => VectorInteger::Subtract,
        VectorAluOp::SubrevNcU32 => VectorInteger::SubtractReverse,
        _ => panic!("{op:?} decodes only with a VGPR SRC0 (operand rule vop2_vgpr2)"),
    }
}

fn reference_op(instruction: Instruction) -> Op {
    match instruction {
        Instruction::BufferLoadDword {
            data,
            index,
            resource,
        } => Op::Load {
            data,
            index,
            resource,
        },
        Instruction::BufferStoreDword {
            data,
            index,
            resource,
        } => Op::Store {
            data,
            index,
            resource,
        },
        Instruction::WaitLoad => Op::WaitRead,
        Instruction::WaitScalarLoad => Op::WaitScalar,
        Instruction::ScalarBufferLoad {
            dst,
            resource,
            offset,
            soffset,
            dwords,
        } => Op::ScalarLoad {
            data: dst,
            resource,
            offset,
            register: soffset,
            dwords,
        },
        Instruction::WaitStore => Op::WaitWrite,
        Instruction::MoveB32 {
            dst,
            src: VectorSource::Scalar(src),
        } => Op::MoveScalar(dst, src),
        Instruction::MoveB32 { dst, src } => Op::Move(dst, reference_source(src)),
        Instruction::VectorUnary { op, dst, src } => Op::VectorUnary(
            match op {
                VectorUnaryOp::NotB32 => VectorUnary::Not,
                VectorUnaryOp::BfrevB32 => VectorUnary::ReverseBits,
                VectorUnaryOp::FfbhU32 => VectorUnary::FindFirstOneFromMsb,
                VectorUnaryOp::FfblB32 => VectorUnary::FindFirstOneFromLsb,
                VectorUnaryOp::FfbhI32 => VectorUnary::FindSignChange,
                VectorUnaryOp::TruncF32 => VectorUnary::Truncate,
                VectorUnaryOp::CeilF32 => VectorUnary::Ceiling,
                VectorUnaryOp::RndneF32 => VectorUnary::RoundNearestEven,
                VectorUnaryOp::FloorF32 => VectorUnary::Floor,
            },
            dst,
            match src {
                VectorSource::Register(reg) => CompareSource::Vector(reg),
                VectorSource::Scalar(reg) => CompareSource::Scalar(reg),
                VectorSource::Immediate(bits) => CompareSource::Immediate(bits),
            },
        ),
        Instruction::VectorSwap { dst, src, .. } => Op::Swap(dst, src),
        Instruction::ScalarAlu {
            op,
            dst,
            src0,
            src1,
        } => Op::Scalar(
            match op {
                ScalarAluOp::AddU32 => ScalarAlu::Add,
                ScalarAluOp::SubU32 => ScalarAlu::Subtract,
                ScalarAluOp::AddI32 => ScalarAlu::AddSigned,
                ScalarAluOp::SubI32 => ScalarAlu::SubtractSigned,
                ScalarAluOp::AddcU32 => ScalarAlu::AddWithCarry,
                ScalarAluOp::SubbU32 => ScalarAlu::SubtractWithBorrow,
                ScalarAluOp::AndB32 => ScalarAlu::And,
                ScalarAluOp::OrB32 => ScalarAlu::Or,
                ScalarAluOp::XorB32 => ScalarAlu::Xor,
                ScalarAluOp::Andn2B32 => ScalarAlu::AndNot,
                ScalarAluOp::Orn2B32 => ScalarAlu::OrNot,
                ScalarAluOp::NandB32 => ScalarAlu::Nand,
                ScalarAluOp::NorB32 => ScalarAlu::Nor,
                ScalarAluOp::XnorB32 => ScalarAlu::Xnor,
                ScalarAluOp::LshlB32 => ScalarAlu::ShiftLeft,
                ScalarAluOp::LshrB32 => ScalarAlu::ShiftRight,
                ScalarAluOp::AshrI32 => ScalarAlu::ArithmeticShiftRight,
                ScalarAluOp::MinI32 => ScalarAlu::Minimum { signed: true },
                ScalarAluOp::MinU32 => ScalarAlu::Minimum { signed: false },
                ScalarAluOp::MaxI32 => ScalarAlu::Maximum { signed: true },
                ScalarAluOp::MaxU32 => ScalarAlu::Maximum { signed: false },
                ScalarAluOp::CselectB32 => ScalarAlu::Select,
                ScalarAluOp::BfmB32 => ScalarAlu::BitfieldMask,
                ScalarAluOp::MulI32 => ScalarAlu::Multiply,
                ScalarAluOp::BfeU32 => ScalarAlu::BitfieldExtract { signed: false },
                ScalarAluOp::BfeI32 => ScalarAlu::BitfieldExtract { signed: true },
                ScalarAluOp::AbsdiffI32 => ScalarAlu::AbsoluteDifference,
                ScalarAluOp::Lshl1AddU32 => ScalarAlu::ShiftLeftAdd(1),
                ScalarAluOp::Lshl2AddU32 => ScalarAlu::ShiftLeftAdd(2),
                ScalarAluOp::Lshl3AddU32 => ScalarAlu::ShiftLeftAdd(3),
                ScalarAluOp::Lshl4AddU32 => ScalarAlu::ShiftLeftAdd(4),
                ScalarAluOp::MulHiU32 => ScalarAlu::MultiplyHigh { signed: false },
                ScalarAluOp::MulHiI32 => ScalarAlu::MultiplyHigh { signed: true },
            },
            dst,
            reference_scalar(src0),
            reference_scalar(src1),
        ),
        Instruction::VectorAluVop3 {
            op: VectorAluVop3Op::AddNcU32,
            dst,
            src0,
            src1,
        } => Op::Add(dst, src0, src1),
        Instruction::VectorAluVop3 {
            op,
            dst,
            src0,
            src1,
        } => Op::VectorInteger(
            reference_vector_integer(op.base()),
            dst,
            CompareSource::Vector(src0),
            src1,
        ),
        Instruction::ScalarUnary { op, dst, src } => {
            let src = reference_scalar(src);
            let unary = match op {
                ScalarUnaryOp::MovB32 => return Op::ScalarMove(dst, src),
                ScalarUnaryOp::NotB32 => ScalarUnary::Not,
                ScalarUnaryOp::BrevB32 => ScalarUnary::ReverseBits,
                ScalarUnaryOp::Bcnt0I32B32 => ScalarUnary::CountZeros,
                ScalarUnaryOp::Bcnt1I32B32 => ScalarUnary::CountOnes,
                ScalarUnaryOp::Ff0I32B32 => ScalarUnary::FindFirstZero,
                ScalarUnaryOp::Ff1I32B32 => ScalarUnary::FindFirstOne,
                ScalarUnaryOp::FlbitI32B32 => ScalarUnary::FindLastBit,
                ScalarUnaryOp::FlbitI32 => ScalarUnary::FindLastSignBit,
                ScalarUnaryOp::SextI32I8 => ScalarUnary::SignExtendByte,
                ScalarUnaryOp::SextI32I16 => ScalarUnary::SignExtendShort,
                ScalarUnaryOp::AbsI32 => ScalarUnary::Absolute,
                ScalarUnaryOp::AndSaveExecB32 => ScalarUnary::AndSaveExec,
                ScalarUnaryOp::OrSaveExecB32 => ScalarUnary::OrSaveExec,
                ScalarUnaryOp::XorSaveExecB32 => ScalarUnary::XorSaveExec,
                ScalarUnaryOp::Andn2SaveExecB32 => ScalarUnary::Andn2SaveExec,
                ScalarUnaryOp::Orn2SaveExecB32 => ScalarUnary::Orn2SaveExec,
                ScalarUnaryOp::NandSaveExecB32 => ScalarUnary::NandSaveExec,
                ScalarUnaryOp::NorSaveExecB32 => ScalarUnary::NorSaveExec,
                ScalarUnaryOp::XnorSaveExecB32 => ScalarUnary::XnorSaveExec,
                ScalarUnaryOp::Andn1SaveExecB32 => ScalarUnary::Andn1SaveExec,
                ScalarUnaryOp::Orn1SaveExecB32 => ScalarUnary::Orn1SaveExec,
                ScalarUnaryOp::Andn1WrexecB32 => ScalarUnary::Andn1Wrexec,
                ScalarUnaryOp::Andn2WrexecB32 => ScalarUnary::Andn2Wrexec,
            };
            Op::ScalarUnary(unary, dst, src)
        }
        Instruction::ScalarCompare { op, src0, src1 } => {
            let (src0, src1) = (reference_scalar(src0), reference_scalar(src1));
            // The reference keeps CompareEqual for EQ and uses its VOPC relation
            // offsets (LT 1, EQ 2, LE 3, GT 4, NE 5, GE 6) for the rest.
            match op.relation() {
                CompareRelation::Equal => Op::CompareEqual(src0, src1),
                relation => Op::ScalarCompare(
                    Comparison {
                        offset: relation as u8 + 1,
                        signed: op.semantics().signedness == Signedness::Signed,
                    },
                    src0,
                    src1,
                ),
            }
        }
        Instruction::ScalarImmediate { op, dst, simm16 } => Op::ScalarImmediate(
            match op {
                ScalarImmediateOp::MovkI32 => ScalarImmediate::Move,
                ScalarImmediateOp::CmovkI32 => ScalarImmediate::ConditionalMove,
                ScalarImmediateOp::AddkI32 => ScalarImmediate::Add,
                ScalarImmediateOp::MulkI32 => ScalarImmediate::Multiply,
            },
            dst,
            simm16,
        ),
        Instruction::ScalarImmediateCompare { op, src, simm16 } => Op::ScalarImmediate(
            ScalarImmediate::Compare(Comparison {
                // The reference's VOPC relation offsets, as for SOPC above.
                offset: op.relation() as u8 + 1,
                signed: op.semantics().signedness == Signedness::Signed,
            }),
            src,
            simm16,
        ),
        Instruction::BranchSccZero { target } => Op::Branch {
            target,
            condition: BranchCondition::SccZero,
        },
        Instruction::BranchExecZero { target } => Op::Branch {
            target,
            condition: BranchCondition::ExecZero,
        },
        Instruction::BranchVccZero { target } => Op::Branch {
            target,
            condition: BranchCondition::VccZero,
        },
        Instruction::Branch { target } => Op::Branch {
            target,
            condition: BranchCondition::Always,
        },
        // The six logical and shift forms migrated from hand-written code
        // (#215 M27) keep the reference's VGPR-only VectorAlu ops.
        Instruction::VectorAlu {
            op:
                op @ (VectorAluOp::AndB32
                | VectorAluOp::OrB32
                | VectorAluOp::XorB32
                | VectorAluOp::LshlrevB32
                | VectorAluOp::LshrrevB32
                | VectorAluOp::AshrrevI32),
            dst,
            src0: VectorSource::Register(src0),
            src1,
        } => Op::Vector(
            match op {
                VectorAluOp::AndB32 => VectorAlu::And,
                VectorAluOp::OrB32 => VectorAlu::Or,
                VectorAluOp::XorB32 => VectorAlu::Xor,
                VectorAluOp::LshlrevB32 => VectorAlu::ShiftLeft,
                VectorAluOp::LshrrevB32 => VectorAlu::ShiftRight,
                _ => VectorAlu::ArithmeticShiftRight,
            },
            dst,
            src0,
            src1,
        ),
        Instruction::ImageSampleLz {
            dst,
            coord,
            resource,
            sampler,
            dmask,
        } => Op::ImageSample {
            dst,
            coord,
            resource,
            sampler,
            dmask,
            explicit_lod: false,
        },
        Instruction::ImageSampleL {
            dst,
            coord,
            resource,
            sampler,
            dmask,
        } => Op::ImageSample {
            dst,
            coord,
            resource,
            sampler,
            dmask,
            explicit_lod: true,
        },
        Instruction::Export { .. } | Instruction::Interpolate { .. } => {
            panic!("compute reference matrix received a graphics instruction")
        }

        Instruction::VectorCompare { op, src0, src1 } => {
            let comparison = reference_comparison(op);
            let src0 = match src0 {
                VectorSource::Register(reg) => CompareSource::Vector(reg),
                VectorSource::Scalar(reg) => CompareSource::Scalar(reg),
                VectorSource::Immediate(bits) => CompareSource::Immediate(bits),
            };
            match op.semantics().width {
                16 => Op::WideCompare(comparison, CompareWidth::Half, src0, src1),
                32 => Op::Compare(comparison, src0, src1),
                64 => Op::WideCompare(comparison, CompareWidth::Double, src0, src1),
                width => panic!("unexpected VOPC operand width {width}"),
            }
        }
        Instruction::VectorCompareVop3 {
            op,
            sdst,
            src0,
            src1,
        } => Op::CompareVop3(
            sdst,
            reference_comparison(op.base()),
            CompareSource::Vector(src0),
            src1,
        ),
        Instruction::VectorAlu {
            op:
                op @ (VectorAluOp::AddF32
                | VectorAluOp::SubF32
                | VectorAluOp::SubrevF32
                | VectorAluOp::MulF32
                | VectorAluOp::MinF32
                | VectorAluOp::MaxF32),
            dst,
            src0,
            src1,
        } => Op::VectorFloat(
            match op {
                VectorAluOp::AddF32 => VectorFloat::Add,
                VectorAluOp::SubF32 => VectorFloat::Subtract,
                VectorAluOp::SubrevF32 => VectorFloat::SubtractReverse,
                VectorAluOp::MulF32 => VectorFloat::Multiply,
                VectorAluOp::MinF32 => VectorFloat::Minimum,
                _ => VectorFloat::Maximum,
            },
            dst,
            match src0 {
                VectorSource::Register(reg) => CompareSource::Vector(reg),
                VectorSource::Scalar(reg) => CompareSource::Scalar(reg),
                VectorSource::Immediate(bits) => CompareSource::Immediate(bits),
            },
            src1,
        ),
        Instruction::VectorAlu {
            op,
            dst,
            src0,
            src1,
        } => Op::VectorInteger(
            reference_vector_integer(op),
            dst,
            match src0 {
                VectorSource::Register(reg) => CompareSource::Vector(reg),
                VectorSource::Scalar(reg) => CompareSource::Scalar(reg),
                VectorSource::Immediate(bits) => CompareSource::Immediate(bits),
            },
            src1,
        ),
        Instruction::EndProgram => Op::End,
    }
}

/// RDNA 2 ISA guide 7.1, 7.5 and Table 73. Words from llvm-mc, then changed by
/// hand where LLVM refuses the operand.
#[test]
fn scalar_buffer_load_operands_are_checked_when_decoding() {
    let decode_words = |first: u32, second: u32| {
        let bytes = [first.to_le_bytes(), second.to_le_bytes()].concat();
        decode(Architecture::Gfx1030, &bytes).map(|decoded| decoded[0].to_string())
    };
    const NULL: u32 = 125 << 25;
    for (first, second, text) in [
        (
            0xf420_0200,
            NULL | 0x14,
            "s_buffer_load_dword s8, s[0:3], 0x14",
        ),
        (0xf420_0000, NULL, "s_buffer_load_dword s0, s[0:3], null"),
        (
            0xf420_1a72,
            NULL,
            "s_buffer_load_dword s105, s[100:103], null",
        ),
        (
            0xf424_1a00,
            105 << 25,
            "s_buffer_load_dwordx2 s[104:105], s[0:3], s105",
        ),
        (
            0xf428_1900,
            12 << 25 | 0xf_ffff,
            "s_buffer_load_dwordx4 s[100:103], s[0:3], s12 offset:0xfffff",
        ),
    ] {
        let decoded = decode_words(first, second).unwrap();
        assert!(decoded.ends_with(text), "{decoded}");
    }
    for (first, second) in [
        (0xf420_0201, NULL),           // an odd SBASE
        (0xf420_0234, NULL),           // s[104:107]
        (0xf420_1a80, NULL),           // vcc_lo
        (0xf420_1f00, NULL),           // m0
        (0xf424_0240, NULL),           // s[9:10]
        (0xf428_0280, NULL),           // s[10:13]
        (0xf428_1a00, NULL),           // s[104:107]
        (0xf420_0200, 106 << 25),      // an offset in vcc_lo
        (0xf420_0200, 124 << 25),      // an offset in m0
        (0xf420_0200, 127 << 25),      // an offset in exec_hi
        (0xf420_0200, NULL | 1 << 20), // a negative offset
        (0xf421_0200, NULL),           // GLC
        (0xf420_4200, NULL),           // DLC
    ] {
        let error = decode_words(first, second).unwrap_err();
        assert!(
            matches!(error.kind, DecodeErrorKind::UnsupportedForm { .. }),
            "{first:08x} {second:08x}: {error:?}"
        );
    }
}

#[test]
fn displays_single_component_image_destination_like_llvm() {
    // llvm-mc prints a one-component VDATA as `v5`, not `v[5:5]`; a multi-
    // component mask keeps the range form.
    for (fixture, expected) in [
        (
            &include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask1.bin")[..],
            "image_sample_lz v5, v[2:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        ),
        (
            &include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmaskf.bin")[..],
            "image_sample_lz v[5:8], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D",
        ),
        (
            &include_bytes!("../../../fixtures/rdna2/image-sample-l-lod0.bin")[..],
            "image_sample_l v[5:8], v[2:4], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D",
        ),
    ] {
        let decoded = decode(Architecture::Gfx1030, fixture).unwrap();
        let image = decoded
            .iter()
            .find(|instruction| {
                instruction
                    .instruction
                    .mnemonic()
                    .starts_with("image_sample")
            })
            .expect("image sample instruction");
        assert_eq!(image.instruction.to_string(), expected);
    }
}

#[test]
#[ignore = "requires LLVM with the gfx1030 AMDGPU target"]
fn compares_fixture_instructions_against_llvm() {
    let high_registers: &[u8] = &[
        0xfe, 0x03, 0xfe, 0x7f, 0xff, 0, 0x25, 0xd5, 0x80, 0xff, 0x03, 0x02, 0, 0, 0x81, 0xbf,
    ];
    let matrix = assemble_matrix();
    let mut forms = std::collections::BTreeSet::new();
    let mut count = 0;
    for fixture in [
        FIXTURE,
        BITWISE_FIXTURE,
        FLOAT_FIXTURE,
        OR_FIXTURE,
        SHIFT_FIXTURE,
        include_bytes!("../../../fixtures/rdna2/ashr.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-alu.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-logic.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-compare.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-unary.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-saveexec.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-arith.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-immediate.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-integer.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-vcc.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-compare.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-compare-ft.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-unary.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-round.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-f32.bin"),
        include_bytes!("../../../fixtures/rdna2/vector-compare-16.bin"),
        include_bytes!("../../../fixtures/rdna2/vop3-integer.bin"),
        include_bytes!("../../../fixtures/rdna2/vop3-compare.bin"),
        include_bytes!("../../../fixtures/rdna2/scalar-memory.bin"),
        include_bytes!("../../../fixtures/rdna2/runtime-branch.bin"),
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask1.bin"),
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask8.bin"),
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask5.bin"),
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmaskf.bin"),
        include_bytes!("../../../fixtures/rdna2/image-sample-l-lod0.bin"),
        include_bytes!("../../../fixtures/rdna2/image-sample-l-lod1.bin"),
        high_registers,
        &matrix,
    ] {
        count += compare_against_llvm(fixture, &mut forms);
    }
    assert_eq!(forms.len(), 194, "forms compared: {forms:?}");
    eprintln!(
        "{count} three-way instruction comparisons across {} forms",
        forms.len()
    );
}

fn assemble_matrix() -> Vec<u8> {
    use std::fmt::Write as _;
    // Cover register banks and extrema, immediate classes in both scalar
    // compare positions, zero displacement and branches over literal words.
    let mut assembly = String::from(
        "
        buffer_load_dword v255, v128, s[100:103], 0 idxen
        buffer_store_dword v128, v255, s[96:99], 0 idxen
        s_buffer_load_dword s0, s[100:103], 0x0
        s_buffer_load_dword s105, s[0:3], 0xfffff
        s_buffer_load_dword s1, s[4:7], s105
        s_buffer_load_dword s1, s[0:3], s0 offset:0x13
        s_buffer_load_dwordx2 s[104:105], s[96:99], 0x1
        s_buffer_load_dwordx2 s[2:3], s[0:3], s104 offset:0xfffff
        s_buffer_load_dwordx4 s[100:103], s[0:3], 0x40
        s_buffer_load_dwordx4 s[8:11], s[100:103], s7
        s_waitcnt lgkmcnt(0)
        v_and_b32_e32 v255, v128, v254
        v_xor_b32_e32 v128, v255, v127
        v_or_b32_e32 v255, v255, v255
        v_or_b32_e32 v0, v0, v0
        v_lshlrev_b32_e32 v255, v128, v0
        v_lshrrev_b32_e32 v0, v255, v128
        v_ashrrev_i32_e32 v255, v0, v1
        v_add_f32_e32 v255, v254, v128
        v_mul_f32_e32 v128, v127, v255
        s_mov_b32 s105, s104
        s_mov_b32 s0, 0xdeadbeef
        s_cmp_eq_u32 s105, s0
        s_cmp_eq_u32 s105, 0xdeadbeef
        s_cmp_eq_u32 0xdeadbeef, s105
        s_cmp_eq_u32 -16, 64
        s_cmp_eq_u32 -16, s105
        s_cmp_eq_u32 s105, -16
        s_cmp_eq_u32 -16, 0xdeadbeef
        s_cmp_eq_u32 0xdeadbeef, -16
        s_cbranch_scc0 0
        s_mov_b32 exec_lo, 0
        s_mov_b32 exec_hi, -1
        s_mov_b32 vcc_lo, 0
        s_mov_b32 vcc_hi, -1
        s_cbranch_execz 0
        s_cbranch_vccz 0
        s_branch 2
        v_mov_b32_e32 v128, 0x12345678
        v_mov_b32_e32 v255, lit(0x1)
        s_mov_b32 s105, lit(0x1)
        s_cmp_eq_u32 s105, lit(0x1)
        s_add_u32 s105, s104, 0xdeadbeef
        s_add_u32 s8, -16, 64
        s_sub_u32 s105, s1, s2
        s_sub_u32 s9, 0xdeadbeef, -16
        s_addc_u32 s104, 0xdeadbeef, s105
        s_subb_u32 s0, s0, s0
        s_add_i32 s101, 0x7fffffff, s100
        s_sub_i32 s100, -16, 0x80000000
        s_and_b32 s1, -1, s2
        s_or_b32 s103, s3, 0x80000000
        s_xor_b32 s102, s104, -16
        s_xor_b32 s7, s7, lit(0x1)
        s_movk_i32 s105, 0xffff
        s_movk_i32 s8, 0
        s_cmovk_i32 s105, 0x8000
        s_addk_i32 s9, 0x7fff
        s_addk_i32 s105, 0xffff
        s_mulk_i32 s8, 0x8001
        v_mov_b32_e32 v255, s105
        v_mov_b32_e32 v0, s0
        v_mov_b32_e32 v128, vcc_lo
    ",
    );
    // Every supported VOPC form with each source class: VGPR, SGPR, vcc_lo,
    // inline integers at both ends, inline zero and a literal.
    for relation in ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"] {
        for kind in ["i32", "u32"] {
            for src0 in [
                "v255",
                "v0",
                "s0",
                "s105",
                "vcc_lo",
                "-16",
                "64",
                "0",
                "0x80000000",
            ] {
                writeln!(assembly, "v_cmp_{relation}_{kind}_e32 vcc_lo, {src0}, v128").unwrap();
            }
            writeln!(assembly, "v_cmp_{relation}_{kind}_e32 vcc_lo, v1, v255").unwrap();
        }
    }
    // Every binary32 rounding form with each accepted SRC0 class: VGPRs at both
    // ends, SGPRs at both ends, vcc_lo and a literal (no integer inline).
    for name in ["trunc", "ceil", "rndne", "floor"] {
        for src0 in ["v255", "v0", "s0", "s105", "vcc_lo", "0x80000001"] {
            writeln!(assembly, "v_{name}_f32_e32 v128, {src0}").unwrap();
        }
    }
    // Every VOP2 binary32 form with each accepted SRC0 class (no integer
    // inline) and VSRC1 at both ends.
    for name in ["add", "sub", "subrev", "mul", "min", "max"] {
        for src0 in ["v255", "v0", "s0", "s105", "vcc_lo", "0x80000001"] {
            for vsrc1 in ["v0", "v255"] {
                writeln!(assembly, "v_{name}_f32_e32 v128, {src0}, {vsrc1}").unwrap();
            }
        }
    }
    // Every SOPK compare on the lowest and highest readable SGPR.
    for relation in ["eq", "lg", "gt", "ge", "lt", "le"] {
        for kind in ["i32", "u32"] {
            for register in ["s0", "s7", "s8", "s105"] {
                writeln!(assembly, "s_cmpk_{relation}_{kind} {register}, 0x8000").unwrap();
            }
            writeln!(assembly, "s_cmpk_{relation}_{kind} s9, 0").unwrap();
        }
    }
    for value in -16..=64 {
        writeln!(assembly, "v_mov_b32_e32 v255, {value}").unwrap();
        writeln!(assembly, "s_mov_b32 s105, {value}").unwrap();
        writeln!(assembly, "s_cmp_eq_u32 {value}, {value}").unwrap();
        writeln!(assembly, "s_add_u32 s10, {value}, s11").unwrap();
    }
    assembly.push_str("s_endpgm\n");
    let output = llvm_mc(&[], assembly.as_bytes());
    assert_llvm_success(&output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let encodings: Vec<_> = stdout
        .lines()
        .filter_map(|line| line.split_once("; encoding: ["))
        .map(|(_, encoding)| encoding_bytes(encoding))
        .collect();
    assert_eq!(
        encodings.len(),
        assembly
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count(),
        "LLVM must encode every matrix row"
    );
    encodings.into_iter().flatten().collect()
}

fn encoding_bytes(encoding: &str) -> Vec<u8> {
    encoding
        .trim_end_matches(']')
        .split(',')
        .map(|byte| u8::from_str_radix(byte.trim().trim_start_matches("0x"), 16).unwrap())
        .collect()
}

fn llvm_mc(mode: &[&str], input: &[u8]) -> std::process::Output {
    let executable = std::env::var_os("LLVM_BIN").map_or_else(
        || PathBuf::from("llvm-mc"),
        |directory| PathBuf::from(directory).join("llvm-mc"),
    );
    let mut child = Command::new(executable)
        .args(mode)
        .args([
            "-show-encoding",
            "-triple=amdgcn-amd-amdhsa",
            "-mcpu=gfx1030",
            "-mattr=+wavefrontsize32",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch llvm-mc (provide its directory in LLVM_BIN or put it on PATH)");
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

fn assert_llvm_success(output: &std::process::Output) {
    assert!(
        output.status.success() && output.stderr.is_empty(),
        "llvm-mc failed or emitted diagnostics: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn compare_against_llvm(
    fixture: &[u8],
    forms: &mut std::collections::BTreeSet<&'static str>,
) -> usize {
    let output = llvm_disassemble(fixture);
    assert_llvm_success(&output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let llvm: Vec<_> = stdout
        .lines()
        .filter_map(|line| line.split_once("; encoding: ["))
        .collect();
    let xenora = decode(Architecture::Gfx1030, fixture).unwrap();
    assert_eq!(llvm.len(), xenora.len());

    let reference = xenora_ref::decode(fixture).unwrap();
    let count = llvm.len();
    assert_eq!(count, reference.len());
    let mut llvm_offset = 0;
    for (((assembly, encoding), decoded), reference) in llvm.into_iter().zip(xenora).zip(reference)
    {
        let llvm_bytes = encoding_bytes(encoding);
        let llvm_end = llvm_offset + llvm_bytes.len();
        assert_eq!(decoded.offset, llvm_offset);
        assert_eq!(reference.offset, llvm_offset);
        assert_eq!(usize::from(decoded.byte_len), llvm_bytes.len());
        assert_eq!(reference.byte_len, llvm_bytes.len());
        assert_eq!(llvm_bytes, fixture[llvm_offset..llvm_end]);
        let assembly = normalize_llvm_assembly(assembly.trim(), llvm_end);
        assert_eq!(
            decoded.instruction.to_string(),
            assembly,
            "LLVM at {llvm_offset:#x}"
        );
        assert_eq!(
            reference.op,
            reference_op(decoded.instruction),
            "reference at {llvm_offset:#x}"
        );
        forms.insert(decoded.instruction.mnemonic());
        llvm_offset = llvm_end;
    }
    assert_eq!(llvm_offset, fixture.len());
    count
}

fn llvm_disassemble(fixture: &[u8]) -> std::process::Output {
    let input = fixture
        .iter()
        .map(|byte| format!("0x{byte:02x} "))
        .collect::<String>();
    llvm_mc(&["-disassemble"], input.as_bytes())
}

#[test]
#[ignore = "requires LLVM with the gfx1030 AMDGPU target"]
fn distinguishes_llvm_invalid_encodings_from_xenora_subset_rejections() {
    let malformed: &[&[u32]] = &[
        &[0xffff_ffff],
        &[0xe030_2000],
        &[0xd525_0001],
        &[0xd51a_0001],
        &[0xd516_0001],
    ];
    let outside_subset: &[&[u32]] = &[
        &[0x7e04_026b],                                        // VOP1 VCC_HI source
        &[0x7da2_0501],                                        // v_cmpx_eq_u32_e32
        &[0xd482_006a, 0x0002_0501],                           // v_cmp_eq_i32_e64 with SRC2 = 0
        &[0x8788_0908],                                        // s_and_b64
        &[0x8008_ffff, 0x1234_5678],                           // SOP2 with two literal sources
        &[0x806c_0908],                                        // SOP2 TTMP destination
        &[0x807e_0908],                                        // SOP2 EXEC_LO destination
        &[0x806a_0908],                                        // SOP2 VCC_LO destination
        &[0x8008_096a],                                        // SOP2 VCC_LO source
        &[0x3803_fe80],                                        // OR with an inline SRC0
        &[0x3803_fe08],                                        // OR with an SGPR SRC0
        &[0x3803_feff, 0xdead_beef],                           // OR with a literal SRC0
        &[0x3803_fefa, 0xff00_e401],                           // OR DPP
        &[0x3803_fef9, 0x0606_0601],                           // OR SDWA
        &[0xd51c_0001, 0x0203_ff01],                           // OR e64
        &[0x3402_0480],                                        // Shift with inline SRC0
        &[0x2dff_fe08],                                        // Shift with SGPR SRC0
        &[0x3402_04ff, 0xdead_beef],                           // Shift with literal SRC0
        &[0x2dff_fefa, 0xff00_e401],                           // Shift DPP
        &[0x3402_04f9, 0x0606_0601],                           // Shift SDWA
        &[0xd51a_0001, 0x0202_0501],                           // Left shift e64
        &[0xd516_0001, 0x0202_0501],                           // Right shift e64
        &[0x3602_0680],                                        // VOP2 inline source
        &[0xbf8c_3f71],                                        // Different wait counter
        &[0xbf82_ffff],                                        // Backward branch
        &[0xbf82_0001, 0xbf81_0000],                           // Branch to EOF
        &[0xbf84_0001, 0x7e04_02ff, 0x1234_5678, 0xbf81_0000], // Branch into literal
    ];
    for (words, llvm_accepts) in malformed
        .iter()
        .map(|words| (words, false))
        .chain(outside_subset.iter().map(|words| (words, true)))
    {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(decode(Architecture::Gfx1030, &bytes).unwrap_err().offset, 0);
        assert_eq!(xenora_ref::decode(&bytes).unwrap_err().offset, 0);
        let output = llvm_disassemble(&bytes);
        if llvm_accepts {
            assert_llvm_success(&output);
            assert!(!output.stdout.is_empty());
        } else {
            assert!(
                String::from_utf8_lossy(&output.stderr).contains("invalid instruction encoding")
            );
            assert!(output.stdout.is_empty());
        }
    }
    eprintln!(
        "{} malformed and {} out-of-subset LLVM probes",
        malformed.len(),
        outside_subset.len()
    );
}

fn literal_bits(text: &str) -> Option<u32> {
    let digits = text
        .strip_prefix("lit(")?
        .strip_suffix(')')?
        .strip_prefix("0x")?;
    u32::from_str_radix(digits, 16).ok()
}

fn normalize_llvm_assembly(assembly: &str, next_offset: usize) -> String {
    let (mnemonic, operands) = assembly.split_once(' ').unwrap_or((assembly, ""));
    match mnemonic {
        "s_waitcnt_vscnt" if operands == "null, 0x0" => format!("{mnemonic} null, 0"),
        "s_branch" | "s_cbranch_scc0" | "s_cbranch_execz" | "s_cbranch_vccz" => {
            let displacement: usize = operands.parse().unwrap();
            let target = next_offset + displacement * 4;
            format!("{mnemonic} -> 0x{target:04x}")
        }
        // VOPC and registry VOP2 forms print SRC0 as bits.
        mnemonic
            if mnemonic.starts_with("v_cmp_")
                || VectorAluOp::ALL
                    .iter()
                    .any(|op| op.assembly_name() == mnemonic) =>
        {
            let mut parts: Vec<String> = operands.split(", ").map(str::to_owned).collect();
            let src = if parts.len() == 5 { 2 } else { 1 };
            parts[src] = match literal_bits(&parts[src]) {
                Some(bits) => format!("0x{bits:x}"),
                None => parts[src].parse::<i32>().map_or_else(
                    |_| parts[src].clone(),
                    |value| format!("0x{:x}", value as u32),
                ),
            };
            format!("{mnemonic} {}", parts.join(", "))
        }
        "v_mov_b32_e32" => {
            let (destination, source) = operands.split_once(", ").unwrap();
            let normalized = match source {
                "1.0" => "0x3f800000".to_owned(),
                literal if literal_bits(literal).is_some() => {
                    format!("0x{:x}", literal_bits(literal).unwrap())
                }
                _ => source.parse::<i32>().map_or_else(
                    |_| source.to_owned(),
                    |value| format!("0x{:x}", value as u32),
                ),
            };
            format!("{mnemonic} {destination}, {normalized}")
        }
        "s_mov_b32" | "s_cmp_eq_u32" => {
            let (left, right) = operands.split_once(", ").unwrap();
            let scalar = |operand: &str| match literal_bits(operand) {
                Some(bits) => match bits as i32 {
                    inline @ -16..=64 => inline.to_string(),
                    _ => format!("0x{bits:x}"),
                },
                None => operand.to_owned(),
            };
            format!("{mnemonic} {}, {}", scalar(left), scalar(right))
        }
        "s_add_u32" | "s_sub_u32" | "s_add_i32" | "s_sub_i32" | "s_addc_u32" | "s_subb_u32"
        | "s_and_b32" | "s_or_b32" | "s_xor_b32" => {
            let scalar = |operand: &str| match literal_bits(operand) {
                Some(bits) => match bits as i32 {
                    inline @ -16..=64 => inline.to_string(),
                    _ => format!("0x{bits:x}"),
                },
                None => operand.to_owned(),
            };
            let operands: Vec<_> = operands.split(", ").map(scalar).collect();
            format!("{mnemonic} {}", operands.join(", "))
        }
        _ => assembly.to_owned(),
    }
}

#[test]
fn displays_decoded_instructions_and_decode_errors() {
    let decoded = DecodedInstruction {
        offset: 0x1a,
        byte_len: 4,
        instruction: Instruction::EndProgram,
    };
    assert_eq!(decoded.to_string(), "001a: s_endpgm");
    for (error, expected) in [
        (
            DecodeError {
                offset: 0x10,
                kind: DecodeErrorKind::Truncated {
                    needed: 8,
                    available: 4,
                },
            },
            "gfx1030 decode at byte 0x10: instruction requires 8 bytes, found 4",
        ),
        (
            DecodeError {
                offset: 0,
                kind: DecodeErrorKind::UnsupportedOpcode { word: 0xdead_beef },
            },
            "gfx1030 decode at byte 0x0: unsupported opcode in 0xdeadbeef",
        ),
        (
            DecodeError {
                offset: 0,
                kind: DecodeErrorKind::UnsupportedForm {
                    mnemonic: "s_mov_b32",
                    detail: "expected a normal SGPR or EXEC/VCC destination",
                },
            },
            "gfx1030 decode at byte 0x0: unsupported s_mov_b32 form: expected a normal SGPR or EXEC/VCC destination",
        ),
        (
            DecodeError {
                offset: 0,
                kind: DecodeErrorKind::InvalidBranchTarget { target: 0x20 },
            },
            "gfx1030 decode at byte 0x0: branch target 0x20 is outside instruction boundaries",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn decodes_inline_negative_and_register_sources() {
    // s_mov_b32 s10, 5; s_mov_b32 s11, -3; v_mov_b32_e32 v2, 5;
    // v_mov_b32_e32 v2, -3; v_mov_b32_e32 v2, v1.
    let words = [
        0xbe8a_0385u32,
        0xbe8b_03c3,
        0x7e04_0285,
        0x7e04_02c3,
        0x7e04_0301,
    ];
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let decoded = decode(Architecture::Gfx1030, &bytes).unwrap();
    assert_eq!(
        decoded[0].instruction,
        Instruction::ScalarUnary {
            op: ScalarUnaryOp::MovB32,
            dst: 10,
            src: ScalarSource::Immediate(5)
        }
    );
    assert_eq!(
        decoded[1].instruction,
        Instruction::ScalarUnary {
            op: ScalarUnaryOp::MovB32,
            dst: 11,
            src: ScalarSource::Immediate(0xffff_fffd)
        }
    );
    assert_eq!(
        decoded[2].instruction,
        Instruction::MoveB32 {
            dst: 2,
            src: VectorSource::Immediate(5)
        }
    );
    assert_eq!(
        decoded[3].instruction,
        Instruction::MoveB32 {
            dst: 2,
            src: VectorSource::Immediate(0xffff_fffd)
        }
    );
    assert_eq!(
        decoded[4].instruction,
        Instruction::MoveB32 {
            dst: 2,
            src: VectorSource::Register(1)
        }
    );
}

#[test]
fn accepts_scalar_destination_105_and_vgpr_zero_add_sources() {
    // s_mov_b32 s105, 0 is the highest normal SGPR destination.
    let valid = 0xbee9_0380u32.to_le_bytes();
    assert_eq!(
        decode(Architecture::Gfx1030, &valid).unwrap()[0].instruction,
        Instruction::ScalarUnary {
            op: ScalarUnaryOp::MovB32,
            dst: 105,
            src: ScalarSource::Immediate(0)
        }
    );
    // s_mov_b32 ttmp0, 0 selects a privileged destination.
    let invalid = 0xbeec_0380u32.to_le_bytes();
    assert_eq!(
        decode(Architecture::Gfx1030, &invalid).unwrap_err().kind,
        DecodeErrorKind::UnsupportedForm {
            mnemonic: "s_mov_b32",
            detail: "expected a normal SGPR or EXEC/VCC destination",
        }
    );

    // v_add_nc_u32_e64 v1, v0, v0 uses selector 256 (V0) for both sources.
    let words = [0xd525_0001u32, 0x0202_0100];
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    assert_eq!(
        decode(Architecture::Gfx1030, &bytes).unwrap()[0].instruction,
        Instruction::VectorAluVop3 {
            op: VectorAluVop3Op::AddNcU32,
            dst: 1,
            src0: 0,
            src1: 0
        }
    );
}

#[test]
fn decode_at_resolves_branch_targets_from_the_stream_offset() {
    let decoded = decode_at(Architecture::Gfx1030, BITWISE_FIXTURE, 0x1c).unwrap();
    assert_eq!(decoded.offset, 0x1c);
    assert_eq!(decoded.byte_len, 4);
    assert_eq!(decoded.branch_target(), Some(0x3c));
    // Alone, the same branch still reports its stream target; only the whole
    // stream can say whether that target is an instruction boundary.
    let isolated = decode_at(Architecture::Gfx1030, &BITWISE_FIXTURE[..0x20], 0x1c).unwrap();
    assert_eq!(isolated.branch_target(), Some(0x3c));
    assert_eq!(
        decode_at(
            Architecture::Gfx1030,
            BITWISE_FIXTURE,
            BITWISE_FIXTURE.len()
        ),
        Err(DecodeError {
            offset: BITWISE_FIXTURE.len(),
            kind: DecodeErrorKind::Truncated {
                needed: 4,
                available: 0
            },
        })
    );
}

// llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding
const VOP3_INTEGER_ENCODINGS: [(VectorAluVop3Op, u32); 11] = [
    // v_mul_i32_i24_e64 v255, v0, v127        ; encoding: [0xff,0x00,0x09,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MulI32I24, 0xd50900ff),
    // v_mul_hi_i32_i24_e64 v255, v0, v127     ; encoding: [0xff,0x00,0x0a,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MulHiI32I24, 0xd50a00ff),
    // v_mul_u32_u24_e64 v255, v0, v127        ; encoding: [0xff,0x00,0x0b,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MulU32U24, 0xd50b00ff),
    // v_mul_hi_u32_u24_e64 v255, v0, v127     ; encoding: [0xff,0x00,0x0c,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MulHiU32U24, 0xd50c00ff),
    // v_min_i32_e64 v255, v0, v127            ; encoding: [0xff,0x00,0x11,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MinI32, 0xd51100ff),
    // v_max_i32_e64 v255, v0, v127            ; encoding: [0xff,0x00,0x12,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MaxI32, 0xd51200ff),
    // v_min_u32_e64 v255, v0, v127            ; encoding: [0xff,0x00,0x13,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MinU32, 0xd51300ff),
    // v_max_u32_e64 v255, v0, v127            ; encoding: [0xff,0x00,0x14,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::MaxU32, 0xd51400ff),
    // v_xnor_b32_e64 v255, v0, v127           ; encoding: [0xff,0x00,0x1e,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::XnorB32, 0xd51e00ff),
    // v_sub_nc_u32_e64 v255, v0, v127         ; encoding: [0xff,0x00,0x26,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::SubNcU32, 0xd52600ff),
    // v_subrev_nc_u32_e64 v255, v0, v127      ; encoding: [0xff,0x00,0x27,0xd5,0x00,0xff,0x02,0x02]
    (VectorAluVop3Op::SubrevNcU32, 0xd52700ff),
];
const VOP3_INTEGER_SOURCE_WORD: u32 = 0x0202_ff00;

#[test]
fn decodes_registry_vop3_integer_encodings() {
    for (op, first) in VOP3_INTEGER_ENCODINGS {
        let bytes = [first.to_le_bytes(), VOP3_INTEGER_SOURCE_WORD.to_le_bytes()].concat();
        let decoded = decode(Architecture::Gfx1030, &bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].byte_len, 8);
        assert_eq!(
            decoded[0].instruction,
            Instruction::VectorAluVop3 {
                op,
                dst: 255,
                src0: 0,
                src1: 127,
            }
        );
    }
}

#[test]
fn decodes_vop3_integer_fixture() {
    use VectorAluVop3Op::*;

    let code = include_bytes!("../../../fixtures/rdna2/vop3-integer.bin");
    let decoded = decode(Architecture::Gfx1030, code).unwrap();
    let expected = [
        (MulI32I24, 2, 1, 1),
        (MulHiI32I24, 3, 2, 1),
        (MulU32U24, 4, 2, 1),
        (MulHiU32U24, 255, 4, 1),
        (MinI32, 6, 1, 2),
        (MaxI32, 7, 1, 4),
        (MinU32, 8, 3, 1),
        (MaxU32, 9, 255, 2),
        (XnorB32, 10, 1, 255),
        (SubNcU32, 11, 6, 10),
        (SubrevNcU32, 1, 1, 9),
    ];
    let actual: Vec<_> = decoded
        .iter()
        .filter_map(|decoded| match decoded.instruction {
            Instruction::VectorAluVop3 {
                op,
                dst,
                src0,
                src1,
            } => Some((decoded.offset, decoded.byte_len, op, dst, src0, src1)),
            _ => None,
        })
        .collect();
    let expected: Vec<_> = expected
        .into_iter()
        .enumerate()
        .map(|(index, (op, dst, src0, src1))| (12 + 8 * index, 8, op, dst, src0, src1))
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn rejects_registry_vop3_integer_modifiers_and_other_sources() {
    for (op, first) in VOP3_INTEGER_ENCODINGS {
        let second = VOP3_INTEGER_SOURCE_WORD;
        let reject = |first: u32, second: u32| {
            let bytes = [first.to_le_bytes(), second.to_le_bytes()].concat();
            let error = decode(Architecture::Gfx1030, &bytes)
                .expect_err(&format!("{op:?} accepted {first:#010x} {second:#010x}"));
            assert_eq!(error.offset, 0);
            assert!(matches!(
                error.kind,
                DecodeErrorKind::UnsupportedForm { mnemonic, .. }
                    if mnemonic == op.assembly_name()
            ));
        };

        // First word: ABS[10:8], OP_SEL[14:11], CLAMP[15].
        for bit in 8..=15 {
            reject(first | (1 << bit), second);
        }
        // Second word: OMOD[28:27], NEG[31:29].
        for bit in 27..=31 {
            reject(first, second | (1 << bit));
        }

        reject(first, second & !0x1ff); // SRC0 = s0
        reject(first, (second & !(0x1ff << 9)) | (128 << 9)); // SRC1 = inline zero
        reject(first, second | (1 << 18)); // SRC2 = inline one
    }
}

#[test]
fn decodes_sopk_immediates_and_rejects_non_sgpr_operands() {
    use xenora_decoder::ScalarImmediateCompareOp;
    let decode_word = |word: u32| decode(Architecture::Gfx1030, &word.to_le_bytes());
    // Words assembled by LLVM for gfx1030 (llvm-mc --show-encoding).
    // s_movk_i32 s8, 0x8000
    let decoded = decode_word(0xb008_8000).unwrap();
    assert_eq!(decoded[0].byte_len, 4);
    assert_eq!(
        decoded[0].instruction,
        Instruction::ScalarImmediate {
            op: ScalarImmediateOp::MovkI32,
            dst: 8,
            simm16: 0x8000
        }
    );
    // s_cmovk_i32 s105, 0xffff keeps the full 16 bits: the sign extension is
    // the translator's job, not the decoder's.
    assert_eq!(
        decode_word(0xb169_ffff).unwrap()[0].instruction,
        Instruction::ScalarImmediate {
            op: ScalarImmediateOp::CmovkI32,
            dst: 105,
            simm16: 0xffff
        }
    );
    // s_cmpk_gt_u32 s0, 0x1234 and s_cmpk_le_i32 s7, 0xffff read SDST s0-s7.
    assert_eq!(
        decode_word(0xb580_1234).unwrap()[0].instruction,
        Instruction::ScalarImmediateCompare {
            op: ScalarImmediateCompareOp::CmpkGtU32,
            src: 0,
            simm16: 0x1234
        }
    );
    assert_eq!(
        decode_word(0xb407_ffff).unwrap()[0].instruction,
        Instruction::ScalarImmediateCompare {
            op: ScalarImmediateCompareOp::CmpkLeI32,
            src: 7,
            simm16: 0xffff
        }
    );
    // s_addk_i32 s9, 0x7fff and s_mulk_i32 s105, 0
    assert_eq!(
        decode_word(0xb789_7fff).unwrap()[0].instruction,
        Instruction::ScalarImmediate {
            op: ScalarImmediateOp::AddkI32,
            dst: 9,
            simm16: 0x7fff
        }
    );
    assert_eq!(
        decode_word(0xb869_0000).unwrap()[0].instruction,
        Instruction::ScalarImmediate {
            op: ScalarImmediateOp::MulkI32,
            dst: 105,
            simm16: 0
        }
    );
    // vcc_lo, exec_lo, null and ttmp0 as SDST, and s_cmpk_eq_i32 m0: refused.
    for (word, mnemonic, detail) in [
        (
            0xb06a_0001,
            "s_movk_i32",
            "expected a normal SGPR destination",
        ),
        (
            0xb07e_0001,
            "s_movk_i32",
            "expected a normal SGPR destination",
        ),
        (
            0xb07d_0001,
            "s_movk_i32",
            "expected a normal SGPR destination",
        ),
        (
            0xb06c_0001,
            "s_movk_i32",
            "expected a normal SGPR destination",
        ),
        (
            0xb1fc_0001,
            "s_cmpk_eq_i32",
            "expected a normal SGPR operand",
        ),
    ] {
        assert_eq!(
            decode_word(word).unwrap_err().kind,
            DecodeErrorKind::UnsupportedForm { mnemonic, detail },
            "{word:#010x}"
        );
    }
    // Other SOPK opcodes (opcode 1 s_version, 17 s_getreg_b32) stay unsupported.
    for word in [0xb080_0000u32, 0xb888_0000] {
        assert!(decode_word(word).is_err(), "{word:#010x}");
    }
    // The fixed store fence is still the one decoded SOPK word outside the registry.
    assert_eq!(
        decode_word(0xbbfd_0000).unwrap()[0].instruction,
        Instruction::WaitStore
    );
}
