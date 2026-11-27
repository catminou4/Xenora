// SPDX-License-Identifier: GPL-3.0-or-later

use xenora_shader::{Architecture, CompileError, CompileOptions, compile, compile_with};

const FIXTURE: &[u8] = include_bytes!("../../../fixtures/rdna2/add.bin");

#[test]
fn compiles_original_fixture() {
    let result = compile(Architecture::Gfx1030, FIXTURE).unwrap();
    assert_eq!(result.ir, include_str!("../../../fixtures/golden/add.ir"));
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/add.metal")
    );
}

#[test]
fn vop3_compare_to_vcc_lo_drives_a_runtime_vccz_branch() {
    let code: Vec<_> = [
        0x7e02_0281u32, // v_mov_b32_e32 v1, 1
        0x7e04_0282,    // v_mov_b32_e32 v2, 2
        0xd4c2_006a,    // v_cmp_eq_u32_e64 vcc_lo, v1, v2
        0x0202_0501,
        0xbf86_0001, // s_cbranch_vccz 1
        0x7e02_0302, // v_mov_b32_e32 v1, v2
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .chain(FIXTURE[28..].iter().copied())
    .collect();
    let shader = compile(Architecture::Gfx1030, &code).unwrap();
    assert!(
        shader
            .disassembly
            .contains("v_cmp_eq_u32_e64 vcc_lo, v1, v2")
    );
    assert!(shader.ir.contains("0008: t2 = wave.mask"));
    assert!(shader.ir.contains("0010: branch.vccz t2 -> 0x0018"));
    assert!(shader.ir.contains("join.vccz t2"));
    assert!(shader.msl.contains("lane_index >= grid"));
}

#[test]
fn vop3_compare_to_s4_rejects_later_scalar_reads() {
    for (setup, offset) in [
        (&[0x7e02_0281u32, 0x7e04_0282][..], 16), // Initialize both VGPRs.
        (&[0xbefe_0380][..], 12),                 // Empty EXEC. The VGPRs need not be initialized.
    ] {
        let code: Vec<_> = setup
            .iter()
            .copied()
            .chain([
                0xd4c2_0004, // v_cmp_eq_u32_e64 s4, v1, v2
                0x0202_0501,
                0xbf06_8004, // s_cmp_eq_u32 s4, 0
                0xbefe_03c1, // s_mov_b32 exec_lo, -1
                0x7e02_0281, // v_mov_b32_e32 v1, 1
            ])
            .flat_map(u32::to_le_bytes)
            .chain(FIXTURE[28..].iter().copied())
            .collect();
        let error = compile(Architecture::Gfx1030, &code).unwrap_err();
        assert!(
            matches!(
                error,
                CompileError::Verify {
                    offset: actual,
                    reason: "scalar source reads a runtime-only SGPR compare mask",
                } if actual == offset
            ),
            "{error}"
        );
    }
}

#[test]
fn vop3_compare_mask_cannot_be_read_as_a_buffer_descriptor() {
    let code: Vec<_> = [
        0x7e02_0281u32, // v_mov_b32_e32 v1, 1
        0x7e04_0282,    // v_mov_b32_e32 v2, 2
        0xd4c2_0004,    // v_cmp_eq_u32_e64 s4, v1, v2
        0x0202_0501,
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .chain(FIXTURE[28..].iter().copied())
    .collect();
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify {
            offset: 16,
            reason: "resource descriptor reads a runtime-only SGPR compare mask",
        })
    ));
}

#[test]
fn vop3_compare_to_normal_sgpr_preserves_vcc_and_can_be_overwritten() {
    let code: Vec<_> = [
        0x7e02_0281u32, // v_mov_b32_e32 v1, 1
        0x7e04_0282,    // v_mov_b32_e32 v2, 2
        0xbeea_0381,    // s_mov_b32 vcc_lo, 1
        0xd4c2_0008,    // v_cmp_eq_u32_e64 s8, v1, v2
        0x0202_0501,
        0xbf86_0001, // s_cbranch_vccz 1
        0xbe88_0381, // s_mov_b32 s8, 1
        0xbf06_8008, // s_cmp_eq_u32 s8, 0
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .chain(FIXTURE[28..].iter().copied())
    .collect();
    let shader = compile(Architecture::Gfx1030, &code).unwrap();
    assert!(
        shader
            .ir
            .contains("0014: branch.uniform 0x001c taken=false")
    );
}

#[test]
fn load_wait_is_required_before_using_loaded_data() {
    let mut code = FIXTURE.to_vec();
    code.drain(8..12);
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 16, .. })
    ));
}

#[test]
fn store_wait_is_required_before_termination() {
    let mut code = FIXTURE.to_vec();
    code.drain(36..40);
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 36, .. })
    ));
}

#[test]
fn rejects_unaligned_length_before_decoding() {
    assert!(matches!(
        compile(Architecture::Gfx1030, &FIXTURE[..FIXTURE.len() - 1]),
        Err(CompileError::InvalidLength { .. })
    ));
}

const BRANCH: &[u8] = include_bytes!("../../../fixtures/rdna2/bitwise-branch.bin");
const FLOAT: &[u8] = include_bytes!("../../../fixtures/rdna2/float-exact.bin");

#[test]
fn uniform_branch_selects_each_known_path() {
    let taken = compile(Architecture::Gfx1030, BRANCH).unwrap();
    assert_eq!(
        taken.ir,
        include_str!("../../../fixtures/golden/bitwise-branch.ir")
    );
    assert_eq!(
        taken.msl,
        include_str!("../../../fixtures/golden/bitwise-branch.metal")
    );

    let mut equal = BRANCH.to_vec();
    equal[17] = 0x87; // Compare s8 with 7 instead of 6.
    let fallthrough = compile(Architecture::Gfx1030, &equal).unwrap();
    assert!(fallthrough.ir.contains("branch.uniform 0x0020 taken=false"));
    assert!(!fallthrough.ir.contains("and.b32"));
    assert!(fallthrough.ir.contains("const.u32 0x00000000"));
}

#[test]
fn rejects_branch_into_a_literal_word() {
    let mut invalid = BRANCH.to_vec();
    invalid[20] = 3; // Branch to byte 36, inside the literal move at byte 32.
    assert!(matches!(
        compile(Architecture::Gfx1030, &invalid),
        Err(CompileError::Decode(error)) if error.offset == 20
    ));
}

#[test]
fn compiles_only_proven_exact_float_operations() {
    let result = compile(Architecture::Gfx1030, FLOAT).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/float-exact.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/float-exact.metal")
    );

    // The guide's rule for the arithmetic forms (12.7, page 131) is a rounded
    // binary32 operation, so the compiler does not refuse a value whose
    // domain it cannot see: the operands are in VGPRs and their values are
    // only known at run time. A value outside the domain is refused there by
    // the interpreters, which the semantic-review crate checks.
    let mut invalid = FLOAT.to_vec();
    invalid[16..20].copy_from_slice(&0x007f_ffffu32.to_le_bytes());
    assert!(compile(Architecture::Gfx1030, &invalid).is_ok());
}

#[test]
fn enforces_the_fixture_size_limit_before_decoding() {
    assert!(matches!(
        compile(Architecture::Gfx1030, &[]),
        Err(CompileError::InvalidLength { bytes: 0 })
    ));
    let mut code = FIXTURE.to_vec();
    // Prepend redundant s_waitcnt vmcnt(0) instructions to reach 1024 bytes.
    code.splice(
        ..0,
        [0x70, 0x3f, 0x8c, 0xbf].repeat((1024 - FIXTURE.len()) / 4),
    );
    assert!(compile(Architecture::Gfx1030, &code).is_ok());
    code.extend_from_slice(&[0; 4]);
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::InvalidLength { bytes: 1028 })
    ));
}

#[test]
fn preserves_the_decoder_diagnostic_and_error_source() {
    let error = compile(Architecture::Gfx1030, &[0xff; 4]).unwrap_err();
    let expected = "gfx1030 decode at byte 0x0: unsupported opcode in 0xffffffff";
    assert_eq!(error.to_string(), expected);
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        expected
    );
}

#[test]
fn rejects_register_moves_into_index_and_outstanding_load_destinations() {
    for (offset, instruction) in [
        (12, [0x01, 0x03, 0x00, 0x7e]), // v_mov_b32_e32 v0, v1
        (24, [0x02, 0x03, 0x02, 0x7e]), // v_mov_b32_e32 v1, v2
    ] {
        let mut code = FIXTURE.to_vec();
        if offset == 24 {
            // Move the load wait after the register move. v2 is initialized.
            code.drain(8..12);
            code.splice(16..16, instruction);
        } else {
            code.splice(offset..offset, instruction);
        }
        let expected_offset = if offset == 24 { 16 } else { offset };
        assert!(matches!(
            compile(Architecture::Gfx1030, &code),
            Err(CompileError::Verify {
                offset,
                reason: "move writes the index register or replaces a pending load",
            }) if offset == expected_offset
        ));
    }
}

#[test]
fn refuses_to_terminate_with_an_outstanding_load() {
    let mut code = FIXTURE.to_vec();
    // Add a second input load after the completed store. An output exists,
    // but the new load still needs its own wait.
    code.splice(40..40, FIXTURE[..8].iter().copied());
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify {
            offset: 48,
            reason: "termination requires one completed store and no pending loads",
        })
    ));
}

#[test]
fn protects_reserved_scalar_resource_registers() {
    let mut code = FIXTURE.to_vec();
    code.splice(..0, 0xbe87_0380u32.to_le_bytes()); // s_mov_b32 s7, 0
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify {
            offset: 0,
            reason: "scalar move would modify a resource descriptor",
        })
    ));
    code[2] = 0x88; // s8 is outside both resource descriptors.
    assert!(compile(Architecture::Gfx1030, &code).is_ok());
}

#[test]
fn proves_a_float_product_using_multiplication() {
    let mut code: Vec<_> = [
        0x7e02_02ffu32, // v_mov_b32_e32 v1, 0x4e800000 (2^30)
        0x4e80_0000,
        0x7e04_02ff, // v_mov_b32_e32 v2, 0x40000000 (2.0)
        0x4000_0000,
        0x1006_0501, // v_mul_f32_e32 v3, v1, v2
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect();
    code.extend_from_slice(&FIXTURE[28..]);
    code[25] = 3; // Store v3.
    // The product 2^31 is exact, while 2^30 + 2 is not representable.
    let shader = compile(Architecture::Gfx1030, &code).unwrap();
    assert!(shader.ir.contains("mul.f32 t0, t1"));
    assert!(
        shader
            .msl
            .contains("as_type<float>(t0) * as_type<float>(t1)")
    );
}

#[test]
fn rejects_load_into_index_or_before_waiting_on_previous_load() {
    let mut code = FIXTURE.to_vec();
    code[5] = 0; // Load into reserved v0.
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 0, .. })
    ));
    let mut code = FIXTURE.to_vec();
    code.splice(8..8, FIXTURE[..8].iter().copied());
    assert!(matches!(
        compile(Architecture::Gfx1030, &code),
        Err(CompileError::Verify { offset: 8, .. })
    ));
}

#[test]
fn checks_fixture_layout_and_bindings_even_in_skipped_code() {
    for (prefix, expected_offset, expected_reason) in [
        (
            &[0xbf82_0001u32, 0xbf81_0000][..], // Skip an early s_endpgm.
            8,
            "instruction follows program termination",
        ),
        (
            &[0xbf82_0002, 0xe030_2000, 0x8001_0100], // Skip load from s[4:7].
            4,
            "unsupported resource or buffer index",
        ),
        (
            &[0xbf82_0002, 0xe070_2000, 0x8000_0100], // Skip store to s[0:3].
            4,
            "unsupported resource or buffer index",
        ),
    ] {
        let code: Vec<_> = prefix
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .chain(FIXTURE.iter().copied())
            .collect();
        assert!(matches!(
            compile(Architecture::Gfx1030, &code),
            Err(CompileError::Verify { offset, reason })
                if offset == expected_offset && reason == expected_reason
        ));
    }

    // A structurally valid instruction on the skipped path need not read an
    // initialized register. Here v_mov_b32_e32 v1, v255 never runs.
    let code: Vec<_> = [0xbf82_0001u32, 0x7e02_03ff]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .chain(FIXTURE.iter().copied())
        .collect();
    let compiled = compile(Architecture::Gfx1030, &code).unwrap();
    assert!(compiled.ir.contains("branch.uniform 0x0008 taken=true"));
    assert!(compiled.msl.contains("uint t1 = input[index];"));
    assert!(compiled.msl.contains("output[index] = t3;"));
}

#[test]
fn scalar_alu_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-alu.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-alu.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-alu.metal")
    );
}

#[test]
fn scalar_logic_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-logic.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-logic.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-logic.metal")
    );
}

#[test]
fn scalar_compare_results_resolve_every_branch() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-compare.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-compare.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-compare.metal")
    );
}

#[test]
fn scalar_immediate_results_resolve_every_branch() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-immediate.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-immediate.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-immediate.metal")
    );
}

#[test]
fn scalar_unary_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-unary.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-unary.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-unary.metal")
    );
}

#[test]
fn scalar_arith_results_reach_the_kernel_as_constants() {
    let code = include_bytes!("../../../fixtures/rdna2/scalar-arith.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/scalar-arith.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/scalar-arith.metal")
    );
}

#[test]
fn vector_integer_forms_lower_to_per_lane_templates() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-integer.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vector-integer.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vector-integer.metal")
    );
}

#[test]
fn vop3_integer_forms_lower_to_per_lane_templates() {
    let code = include_bytes!("../../../fixtures/rdna2/vop3-integer.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vop3-integer.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vop3-integer.metal")
    );
}

#[test]
fn vector_f_and_t_compares_lower_to_operand_free_wave_masks() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-compare-ft.bin");
    let result = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/vector-compare-ft.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/vector-compare-ft.metal")
    );
}

#[test]
fn vector_compares_lower_to_per_thread_wave_masks_and_selects() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-compare.bin");
    let result = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/vector-compare.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/vector-compare.metal")
    );
}

#[test]
fn vop3_compare_forms_lower_to_wave_masks() {
    let code = include_bytes!("../../../fixtures/rdna2/vop3-compare.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vop3-compare.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vop3-compare.metal")
    );
}

#[test]
fn vector_vcc_fixture_matches_ir_and_msl_goldens() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-vcc.bin");
    let result = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        result.ir,
        include_str!("../../../fixtures/golden/vector-vcc.ir")
    );
    assert_eq!(
        result.msl,
        include_str!("../../../fixtures/golden/vector-vcc.metal")
    );
}

#[test]
fn vop1_integer_fixture_lowers_unary_and_masked_swap() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-unary.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vector-unary.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vector-unary.metal")
    );
}

#[test]
fn vop1_float_rounding_fixture_lowers_to_integer_bit_operations() {
    let code = include_bytes!("../../../fixtures/rdna2/vector-round.bin");
    let compiled = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        compiled.ir,
        include_str!("../../../fixtures/golden/vector-round.ir")
    );
    assert_eq!(
        compiled.msl,
        include_str!("../../../fixtures/golden/vector-round.metal")
    );
    // No Metal float operation or reinterpretation: fast math cannot apply.
    assert!(!compiled.msl.contains("as_type") && !compiled.msl.contains("float"));
}

// --- Opt-in constant folding: hand-computed known-answer fixtures -------------
// Instruction words are built by hand to match the existing RDNA2 fixture shape
// (load, wait, mov literals, op, store). Expected constants are derived from the
// RDNA2 guide / MSL templates in lib.rs, then checked against fold-on IR. Fold-off
// (and default compile) must leave the op mnemonic in the IR.

/// The three synthetic binding ABI descriptor loads of the image-sample
/// fixtures: s[8:11] from offset 0x0, s[12:15] from 0x10, s[16:19] from 0x20.
const IMAGE_SAMPLE_LOADS: [u32; 6] = [
    0xf428_0200,
    0xfa00_0000,
    0xf428_0300,
    0xfa00_0010,
    0xf428_0400,
    0xfa00_0020,
];

fn image_sample_program(loads: &[u32], extra: &[u32]) -> Vec<u8> {
    let mut program: Vec<u32> = (0..12).map(|index| 0xbe88_0380 + (index << 16)).collect();
    program.extend_from_slice(loads);
    program.push(0xbf8c_c07f); // s_waitcnt lgkmcnt(0)
    program.extend_from_slice(extra);
    program.extend([
        0x7e04_0280, // v_mov_b32_e32 v2, 0
        0x7e06_0280, // v_mov_b32_e32 v3, 0
        0xf09c_0108, // image_sample_lz v5, v[2:3], s[8:15], s[16:19]
        0x0082_0502,
        0xe070_2000, // buffer_store_dword v5, v0, s[4:7], 0 idxen
        0x8001_0500,
        0xbbfd_0000, // s_waitcnt_vscnt null, 0
        0xbf81_0000, // s_endpgm
    ]);
    words(&program)
}

#[test]
fn compiles_image_sample_with_the_three_descriptor_loads() {
    compile(
        Architecture::Gfx1030,
        &image_sample_program(&IMAGE_SAMPLE_LOADS, &[]),
    )
    .expect("the synthetic descriptor loads satisfy the binding contract");
}

#[test]
fn refuses_image_sample_without_descriptor_loads() {
    let error = compile(Architecture::Gfx1030, &image_sample_program(&[], &[])).unwrap_err();
    assert!(
        matches!(
            error,
            CompileError::DescriptorProvenance {
                register: 8,
                expected: "s_buffer_load_dwordx4 s[8:11], s[0:3], 0x0",
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn refuses_image_sample_with_a_descriptor_load_at_the_wrong_offset() {
    let mut loads = IMAGE_SAMPLE_LOADS;
    loads[3] = 0xfa00_0000; // s[12:15] from offset 0x0 instead of 0x10
    let error = compile(Architecture::Gfx1030, &image_sample_program(&loads, &[])).unwrap_err();
    assert!(
        matches!(
            error,
            CompileError::DescriptorProvenance { register: 12, .. }
        ),
        "{error:?}"
    );
}

#[test]
fn refuses_image_sample_with_a_narrow_descriptor_load() {
    let mut loads = IMAGE_SAMPLE_LOADS;
    loads[4] = 0xf424_0400; // dwordx2 into s[16:17] instead of dwordx4
    let error = compile(Architecture::Gfx1030, &image_sample_program(&loads, &[])).unwrap_err();
    assert!(
        matches!(
            error,
            CompileError::DescriptorProvenance { register: 16, .. }
        ),
        "{error:?}"
    );
}

#[test]
fn refuses_image_sample_after_a_later_descriptor_write() {
    // s_mov_b32 s8, 0 after the completed descriptor loads.
    let error = compile(
        Architecture::Gfx1030,
        &image_sample_program(&IMAGE_SAMPLE_LOADS, &[0xbe88_0380]),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            CompileError::DescriptorProvenance { register: 8, .. }
        ),
        "{error:?}"
    );
}

/// The six assembled MIMG image-sample fixtures. Each writes a different DMASK
/// component or an explicit LOD, so the compiled IR and MSL name the form, the
/// sampled channels and the level.
const IMAGE_SAMPLE_FIXTURES: [(&[u8], &str, &str, u32); 6] = [
    (
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask1.bin"),
        "sample_lz.2d.f32",
        "r",
        0x0000_0000,
    ),
    (
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask8.bin"),
        "sample_lz.2d.f32",
        "a",
        0x0000_0000,
    ),
    (
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmask5.bin"),
        "sample_lz.2d.f32",
        "rb",
        0x0000_0000,
    ),
    (
        include_bytes!("../../../fixtures/rdna2/image-sample-lz-dmaskf.bin"),
        "sample_lz.2d.f32",
        "rgba",
        0x0000_0000,
    ),
    (
        include_bytes!("../../../fixtures/rdna2/image-sample-l-lod0.bin"),
        "sample_l.2d.f32",
        "rgba",
        0x0000_0000,
    ),
    (
        include_bytes!("../../../fixtures/rdna2/image-sample-l-lod1.bin"),
        "sample_l.2d.f32",
        "rgba",
        0x3f80_0000,
    ),
];

/// The channel each DMASK-selected component selects in the generated MSL, in
/// the order the components are written.
fn sampled_channels(msl: &str) -> String {
    msl.lines()
        .filter(|line| line.contains("tex0.sample("))
        .map(|line| {
            let (_, tail) = line.rsplit_once('.').expect("a sampled channel");
            tail.chars().next().expect("a sampled channel")
        })
        .collect()
}

/// The MSL operand of the sample's `level(...)` argument.
fn sample_level_operand(msl: &str) -> String {
    let line = msl
        .lines()
        .find(|line| line.contains("tex0.sample("))
        .expect("a sampled texture");
    let start = line
        .find("level(as_type<float>(")
        .expect("an explicit level")
        + "level(as_type<float>(".len();
    let tail = &line[start..];
    tail[..tail.find("))").expect("the end of the level")].to_string()
}

/// The DMASK bit order is R, G, B, A (guide Table 41, printed page 69), so a
/// masked sample reads exactly those channels of the texel, in that order, and
/// an explicit LOD reaches the level the fixture asks for.
#[test]
fn image_sample_lowers_to_the_documented_channels_and_level() {
    for (fixture, form, channels, level) in IMAGE_SAMPLE_FIXTURES {
        let compiled = compile(Architecture::Gfx1030, fixture).expect("compile the fixture");
        assert!(
            compiled.ir.contains(form),
            "the IR must name {form}: {}",
            compiled.ir
        );
        assert_eq!(
            sampled_channels(&compiled.msl),
            channels,
            "the MSL must sample the DMASK channels in R, G, B, A order: {}",
            compiled.msl
        );
        let operand = sample_level_operand(&compiled.msl);
        assert!(
            compiled
                .msl
                .contains(&format!("uint {operand} = 0x{level:08x}u;")),
            "the level must be the fixture's LOD: {}",
            compiled.msl
        );
    }
}

/// The image-sample program with an initialised destination and a partial EXEC
/// mask: the masked lanes sample and the rest keep the prior destination value.
fn image_sample_program_with_exec(mask: u32) -> Vec<u8> {
    let mut program: Vec<u32> = (0..12).map(|index| 0xbe88_0380 + (index << 16)).collect();
    program.extend_from_slice(&IMAGE_SAMPLE_LOADS);
    program.extend([
        0xbf8c_c07f,        // s_waitcnt lgkmcnt(0)
        0x7e04_0280,        // v_mov_b32_e32 v2, 0
        0x7e06_0280,        // v_mov_b32_e32 v3, 0
        0x7e0a_0280,        // v_mov_b32_e32 v5, 0
        0xbefe_0380 | mask, // s_mov_b32 exec_lo, mask
        0xf09c_0108,        // image_sample_lz v5, v[2:3], s[8:15], s[16:19] dmask:0x1
        0x0082_0502,
        0xbefe_03c1, // s_mov_b32 exec_lo, -1: the store needs a full wave
        0xe070_2000, // buffer_store_dword v5, v0, s[4:7], 0 idxen
        0x8001_0500,
        0xbbfd_0000, // s_waitcnt_vscnt null, 0
        0xbf81_0000, // s_endpgm
    ]);
    words(&program)
}

/// A partially active image sample writes only the lanes the EXEC mask selects
/// and leaves the other lanes with the value their destination already held.
#[test]
fn image_sample_under_a_partial_exec_keeps_the_prior_destination() {
    let compiled = compile(Architecture::Gfx1030, &image_sample_program_with_exec(5))
        .expect("compile a partially active image sample");
    assert!(
        compiled.ir.contains("merge.lanes mask=0x00000005"),
        "the IR must merge the sample with the prior destination: {}",
        compiled.ir
    );
    assert!(
        compiled
            .msl
            .contains("((0x00000005u >> (index & 31u)) & 1u) != 0u ?"),
        "the sampled value must reach only the lanes the mask selects: {}",
        compiled.msl
    );
}

/// Every translated sample checks its domain before the result is used: finite
/// coordinates always, a whole mip level for an explicit LOD, and only in the
/// lanes EXEC selects. A failing lane sets the `[[buffer(2)]]` fault word.
#[test]
fn image_samples_guard_their_runtime_domain() {
    for (fixture, form, _, _) in IMAGE_SAMPLE_FIXTURES {
        let compiled = compile(Architecture::Gfx1030, fixture).expect("compile image sample");
        let msl = &compiled.msl;
        assert!(
            msl.contains("device atomic_uint* fault [[buffer(2)]]"),
            "{msl}"
        );
        assert!(
            msl.contains("static inline bool xenora_finite(uint bits)")
                && msl.contains("static inline bool xenora_whole_level(uint bits, uint levels)"),
            "{msl}"
        );
        let samples = msl.matches(".sample(samp0").count();
        let guards = msl
            .matches("atomic_fetch_or_explicit(fault, bad, memory_order_relaxed)")
            .count();
        assert_eq!(samples, guards, "every sample is guarded: {msl}");
        assert_eq!(
            msl.matches("bad |= 2u;").count(),
            samples,
            "coordinate check: {msl}"
        );
        let lod_checks = msl
            .matches("tex0.get_num_mip_levels())) { bad |= 1u; }")
            .count();
        if form == "sample_l.2d.f32" {
            assert_eq!(lod_checks, samples, "explicit LOD check: {msl}");
            assert!(
                compiled.ir.contains(", lod t"),
                "IR names the LOD: {}",
                compiled.ir
            );
        } else {
            assert_eq!(lod_checks, 0, "LZ has no LOD to check: {msl}");
        }
        // Full EXEC: the guard has no lane condition.
        assert!(msl.contains("if (bad != 0u) {"), "{msl}");
    }
    assert_eq!(xenora_shader::SAMPLE_FAULT_LOD, 1);
    assert_eq!(xenora_shader::SAMPLE_FAULT_COORDINATES, 2);
}

/// Under a partial EXEC only the selected lanes may report a domain fault, so
/// an inactive lane's coordinate cannot refuse the dispatch.
#[test]
fn image_sample_domain_guard_follows_a_partial_exec() {
    let compiled = compile(Architecture::Gfx1030, &image_sample_program_with_exec(5))
        .expect("compile a partially active image sample");
    assert!(
        compiled
            .msl
            .contains("if (bad != 0u && ((0x00000005u >> (index & 31u)) & 1u) != 0u) {"),
        "{}",
        compiled.msl
    );
    assert!(compiled.ir.contains("exec=0x00000005"), "{}", compiled.ir);
}

/// A program without an image sample keeps the two-buffer kernel signature.
#[test]
fn programs_without_samples_bind_no_fault_word() {
    for fixture in [FIXTURE, BRANCH, FLOAT] {
        let compiled = compile(Architecture::Gfx1030, fixture).expect("compile fixture");
        assert!(!compiled.msl.contains("fault"), "{}", compiled.msl);
        assert!(!compiled.msl.contains("xenora_finite"), "{}", compiled.msl);
    }
}

fn words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn mov_lit(dst: u32) -> u32 {
    0x7e00_0000 | (dst << 17) | (1 << 9) | 255
}

fn vop2(op: u32, dst: u32, src0_vgpr: u32, src1_vgpr: u32) -> u32 {
    (op << 25) | (dst << 17) | (src1_vgpr << 9) | (0x100 + src0_vgpr)
}

fn vop1(op: u32, dst: u32, src_vgpr: u32) -> u32 {
    0x7e00_0000 | (dst << 17) | (op << 9) | (0x100 + src_vgpr)
}

/// load v1; mov v2,A; mov v3,B; VOP2 dst=v2 (or v4); store the result VGPR.
fn binary_const_fixture(op: u32, left: u32, right: u32, dst_is_v4: bool) -> Vec<u8> {
    let (dst, store_hi) = if dst_is_v4 {
        (4u32, 0x8001_0400u32)
    } else {
        (2, 0x8001_0200)
    };
    // For shift-rev forms the ISA encodes count in SRC0 and value in SRC1; the
    // caller passes (count, value) already swapped into (src0, src1) via left/right
    // naming below when needed. Non-rev forms use SRC0=v2, SRC1=v3.
    let op_word = if dst_is_v4 {
        // Shift-rev: v_l*rev dst, v3(count), v2(value) → SRC0=v3, SRC1=v2.
        vop2(op, dst, 3, 2)
    } else {
        vop2(op, dst, 2, 3)
    };
    words(&[
        0xe030_2000,
        0x8000_0100, // buffer_load_dword v1
        0xbf8c_3f70, // s_waitcnt vmcnt(0)
        mov_lit(2),
        left, // v_mov_b32 v2, left
        mov_lit(3),
        right, // v_mov_b32 v3, right
        op_word,
        0xe070_2000,
        store_hi,    // buffer_store_dword result
        0xbbfd_0000, // s_waitcnt_vscnt
        0xbf81_0000, // s_endpgm
    ])
}

fn unary_const_fixture(op: u32, source: u32) -> Vec<u8> {
    words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        source,         // v_mov_b32 v2, source
        vop1(op, 3, 2), // v_* v3, v2
        0xe070_2000,
        0x8001_0300, // buffer_store_dword v3
        0xbbfd_0000,
        0xbf81_0000,
    ])
}

/// load; mov v2,A; mov v3,B; and v2,v2,v3; xor v1,v1,v2; store.
/// Hand-computed: 0x12345678 & 0xff00ff00 == 0x12005600.
fn constant_and_fixture() -> Vec<u8> {
    const A: u32 = 0x1234_5678;
    const B: u32 = 0xff00_ff00;
    words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        A,
        mov_lit(3),
        B,
        0x3604_0702, // v_and_b32 v2, v2, v3 (OP=27)
        0x3a02_0501, // v_xor_b32 v1, v1, v2
        0xe070_2000,
        0x8001_0100,
        0xbbfd_0000,
        0xbf81_0000,
    ])
}

fn compile_fold_off(code: &[u8]) -> xenora_shader::CompiledShader {
    compile_with(
        Architecture::Gfx1030,
        code,
        CompileOptions {
            fold_constants: false,
            ..CompileOptions::default()
        },
    )
    .unwrap()
}

fn compile_fold_on(code: &[u8]) -> xenora_shader::CompiledShader {
    compile_with(
        Architecture::Gfx1030,
        code,
        CompileOptions {
            fold_constants: true,
            ..CompileOptions::default()
        },
    )
    .unwrap()
}

fn assert_folds_to(code: &[u8], mnemonic: &str, expected: u32) {
    let plain = compile_fold_off(code);
    let folded = compile_fold_on(code);
    let default = compile(Architecture::Gfx1030, code).unwrap();
    assert_eq!(
        default.ir, plain.ir,
        "default compile must match fold_constants: false"
    );
    assert!(
        plain.ir.contains(mnemonic),
        "fold-off IR must keep {mnemonic}: {}",
        plain.ir
    );
    assert!(
        !folded.ir.contains(mnemonic),
        "fold-on IR must drop {mnemonic}: {}",
        folded.ir
    );
    let const_ir = format!("const.u32 0x{expected:08x}");
    assert!(
        folded.ir.contains(&const_ir),
        "fold-on IR must contain {const_ir}: {}",
        folded.ir
    );
    let const_msl = format!("0x{expected:08x}u");
    assert!(
        folded.msl.contains(&const_msl),
        "fold-on MSL must contain {const_msl}: {}",
        folded.msl
    );
}

#[test]
fn default_compile_matches_fold_constants_off() {
    let code = constant_and_fixture();
    let baseline = compile(Architecture::Gfx1030, &code).unwrap();
    let explicit = compile_fold_off(&code);
    assert_eq!(baseline.ir, explicit.ir);
    assert_eq!(baseline.msl, explicit.msl);
    assert!(baseline.ir.contains("and.b32"));
}

#[test]
fn folds_constant_and_b32_to_known_answer() {
    // 0x12345678 & 0xff00ff00 = 0x12005600 (bitwise and of each bit).
    assert_folds_to(&constant_and_fixture(), "and.b32", 0x1200_5600);
}

#[test]
fn folding_is_noop_on_add_fixture() {
    let folded = compile_fold_on(FIXTURE);
    let plain = compile(Architecture::Gfx1030, FIXTURE).unwrap();
    // add.bin has no pure constant-constant vector op, so folding is a no-op.
    assert_eq!(folded.ir, plain.ir);
    assert_eq!(folded.msl, plain.msl);
    assert_eq!(plain.ir, include_str!("../../../fixtures/golden/add.ir"));
}

#[test]
fn folds_lshl_by_32_as_shift_by_0() {
    // Guide / MSL: count masked with S0[4:0] → 32 & 31 == 0, so 0xf0 << 0 == 0xf0.
    // v_lshlrev_b32 OP=26; fixture stores into v4 (rev encoding).
    let code = binary_const_fixture(26, 0x0000_00f0, 32, true);
    assert_folds_to(&code, "lshl.b32", 0x0000_00f0);
}

#[test]
fn folds_lshl_by_33_as_shift_by_1() {
    // 33 & 31 == 1, so 0xf0 << 1 == 0x1e0.
    let code = binary_const_fixture(26, 0x0000_00f0, 33, true);
    assert_folds_to(&code, "lshl.b32", 0x0000_01e0);
}

#[test]
fn folds_lshr_by_32_as_shift_by_0() {
    // 32 & 31 == 0, so 0xf0 >> 0 == 0xf0. v_lshrrev_b32 OP=22.
    let code = binary_const_fixture(22, 0x0000_00f0, 32, true);
    assert_folds_to(&code, "lshr.b32", 0x0000_00f0);
}

#[test]
fn folds_lshr_by_33_as_shift_by_1() {
    // 33 & 31 == 1, so 0xf0 >> 1 == 0x78.
    let code = binary_const_fixture(22, 0x0000_00f0, 33, true);
    assert_folds_to(&code, "lshr.b32", 0x0000_0078);
}

#[test]
fn folds_ashr_of_negative_value() {
    // (-16 as i32) >> 4 == -1 → 0xffffffff. v_ashrrev_i32 OP=24.
    let code = binary_const_fixture(24, 0xffff_fff0, 4, true);
    assert_folds_to(&code, "ashr.i32", 0xffff_ffff);
}

#[test]
fn folds_min_i32_and_min_u32_differ_on_sign_bit() {
    // min_i32(0x80000000, 1) = min(-2^31, 1) = 0x80000000.
    // min_u32(0x80000000, 1) = min(2^31, 1) = 1.
    // They must differ. OP MinI32=17, MinU32=19.
    assert_folds_to(
        &binary_const_fixture(17, 0x8000_0000, 1, false),
        "min.i32",
        0x8000_0000,
    );
    assert_folds_to(
        &binary_const_fixture(19, 0x8000_0000, 1, false),
        "min.u32",
        1,
    );
}

#[test]
fn folds_mul_hi_i32_i24_with_negative_input() {
    // i24(0x00ffffff) = ((0x00ffffff << 8) >> 8) = -1; i24(2) = 2;
    // (-1 * 2) as i64 >> 32 = -1 → 0xffffffff. v_mul_hi_i32_i24 OP=10.
    let code = binary_const_fixture(10, 0x00ff_ffff, 2, false);
    assert_folds_to(&code, "mulhi24.i32", 0xffff_ffff);
}

#[test]
fn folds_ffbh_i32_on_zero_all_ones_and_one() {
    // RDNA2 / MSL: return 0xffffffff when every bit matches the sign bit (0 or -1);
    // else clz(x ^ (0 - (x >> 31))). For 1: clz(1) = 31. v_ffbh_i32 OP=59.
    assert_folds_to(&unary_const_fixture(59, 0), "ffbh.i32", 0xffff_ffff);
    assert_folds_to(
        &unary_const_fixture(59, 0xffff_ffff),
        "ffbh.i32",
        0xffff_ffff,
    );
    assert_folds_to(&unary_const_fixture(59, 1), "ffbh.i32", 31);
}

#[test]
fn does_not_fold_a_constant_float_add() {
    // 1.0 + 2.0 = 3.0 exactly, but a float add is not folded: the guide's
    // rule (12.7, page 131) is a rounded binary32 operation, so the value is
    // computed where the two paths agree on it.
    let code = binary_const_fixture(3, 0x3f80_0000, 0x4000_0000, false);
    let plain = compile_fold_off(&code);
    let folded = compile_fold_on(&code);
    assert!(plain.ir.contains("add.f32"), "{}", plain.ir);
    assert_eq!(
        folded.ir, plain.ir,
        "the fold pass must not fold a float add"
    );
    assert!(!plain.ir.contains("0x40400000"));
}

#[test]
fn float_add_from_load_stays_unfolded_with_fold_on() {
    // float-exact.bin feeds add.f32 from a load-derived value (not a pure
    // constant), so the op must remain even with fold_constants. Constant
    // denormal / NaN / negative / inexact float pairs are rejected at verify
    // (see rejects_inexact_float_add_of_constants_at_verify) and never reach
    // the folder as stay-unfolded ops.
    let plain = compile_fold_off(FLOAT);
    let folded = compile_fold_on(FLOAT);
    assert!(
        plain.ir.contains("add.f32"),
        "fold-off must keep add.f32: {}",
        plain.ir
    );
    assert!(
        folded.ir.contains("add.f32"),
        "fold-on must not rewrite load-derived add.f32: {}",
        folded.ir
    );
    assert!(
        folded.ir.contains("mul.f32"),
        "fold-on must not rewrite load-derived mul.f32: {}",
        folded.ir
    );
}

#[test]
fn accepts_an_inexact_float_add_of_constants() {
    // Both operands are positive normals, but 1.0 + 2^-24 is not
    // representable in f32 (ulp of 1.0 is 2^-23). The guide's rule (12.7,
    // page 131) rounds it to nearest even, so the program is compiled and
    // the value is computed at run time.
    let code = binary_const_fixture(3, 0x3f80_0000, 0x3380_0000, false);
    for fold in [false, true] {
        let result = compile_with(
            Architecture::Gfx1030,
            &code,
            CompileOptions {
                fold_constants: fold,
                ..CompileOptions::default()
            },
        );
        assert!(
            result.is_ok(),
            "fold={fold} must accept the constant add.f32: {result:?}"
        );
        assert!(
            result.unwrap().ir.contains("add.f32"),
            "fold={fold} must keep the add"
        );
    }
}

// --- Opt-in common-subexpression elimination (#142) ---------------------------
// Expected IR is derived from the lifter's value order and the reuse rule
// (same unary/binary op, same operand values in order), not from the pass
// output. Constant bits cited below are computed by hand.

fn compile_options(code: &[u8], fold: bool, cse: bool) -> xenora_shader::CompiledShader {
    compile_with(
        Architecture::Gfx1030,
        code,
        CompileOptions {
            fold_constants: fold,
            eliminate_common_subexpressions: cse,
        },
    )
    .unwrap()
}

fn vopc(op: u32, src0_vgpr: u32, src1_vgpr: u32) -> u32 {
    // VOPC: encoding bits 0x7c00_0000, OP in bits 17-24, VSRC1 in bits 9-16,
    // SRC0 in bits 0-8 (VGPR selector 256 + register).
    0x7c00_0000 | (op << 17) | (src1_vgpr << 9) | (0x100 + src0_vgpr)
}

/// load; mov v2, 0xff00ff00; and v3 and v4 with v1, v2; xor each and with v2.
/// The two ands are the same pure op. The two xors become the same pure op
/// only after the second and is reused, so the second xor must reuse the first.
fn duplicate_and_xor_fixture() -> Vec<u8> {
    words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        0xff00_ff00,
        vop2(27, 3, 1, 2), // v_and_b32 v3, v1, v2
        vop2(27, 4, 1, 2), // v_and_b32 v4, v1, v2
        vop2(29, 5, 3, 2), // v_xor_b32 v5, v3, v2
        vop2(29, 6, 4, 2), // v_xor_b32 v6, v4, v2
        0xe070_2000,
        0x8001_0600, // buffer_store_dword v6
        0xbbfd_0000,
        0xbf81_0000,
    ])
}

const DUPLICATE_AND_OFF: &str = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0xff00ff00
0014: t2 = and.b32 t0, t1
0018: t3 = and.b32 t0, t1
001c: t4 = xor.b32 t2, t1
0020: t5 = xor.b32 t3, t1
0024: store.u32 output[index], t5
";

const DUPLICATE_AND_ON: &str = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0xff00ff00
0014: t2 = and.b32 t0, t1
0018: t3 = reuse.u32 t2
001c: t4 = xor.b32 t2, t1
0020: t5 = reuse.u32 t4
0024: store.u32 output[index], t4
";

const DUPLICATE_AND_MSL_OFF: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void xenora_compute(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    uint t0 = input[index];
    uint t1 = 0xff00ff00u;
    uint t2 = t0 & t1;
    uint t3 = t0 & t1;
    uint t4 = t2 ^ t1;
    uint t5 = t3 ^ t1;
    output[index] = t5;
}
";

const DUPLICATE_AND_MSL_ON: &str = "\
#include <metal_stdlib>
using namespace metal;
kernel void xenora_compute(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    uint t0 = input[index];
    uint t1 = 0xff00ff00u;
    uint t2 = t0 & t1;
    uint t3 = t2;
    uint t4 = t2 ^ t1;
    uint t5 = t4;
    output[index] = t4;
}
";

#[test]
fn default_compile_matches_cse_off() {
    assert_eq!(
        CompileOptions::default(),
        CompileOptions {
            fold_constants: false,
            eliminate_common_subexpressions: false,
        }
    );
    let code = duplicate_and_xor_fixture();
    let baseline = compile(Architecture::Gfx1030, &code).unwrap();
    let explicit = compile_options(&code, false, false);
    assert_eq!(baseline.ir, explicit.ir);
    assert_eq!(baseline.msl, explicit.msl);
    assert_eq!(baseline.ir, DUPLICATE_AND_OFF);
    assert_eq!(baseline.msl, DUPLICATE_AND_MSL_OFF);
    // add.bin has no duplicate pure expression. CSE on must not touch it.
    let add = compile(Architecture::Gfx1030, FIXTURE).unwrap();
    let add_cse = compile_options(FIXTURE, false, true);
    assert_eq!(add.ir, include_str!("../../../fixtures/golden/add.ir"));
    assert_eq!(add_cse.ir, add.ir);
    assert_eq!(add_cse.msl, add.msl);
}

#[test]
fn reuses_duplicate_and_and_the_xor_of_that_and() {
    // Off: two and.b32 and two xor.b32. On: one of each, plus reuse.u32.
    // The xor is not evaluated: xor of a value with a different operand stays
    // xor.b32. (xor with itself is not what this fixture computes.)
    let code = duplicate_and_xor_fixture();
    let off = compile_options(&code, false, false);
    let on = compile_options(&code, false, true);
    assert_eq!(off.ir, DUPLICATE_AND_OFF);
    assert_eq!(off.msl, DUPLICATE_AND_MSL_OFF);
    assert_eq!(on.ir, DUPLICATE_AND_ON);
    assert_eq!(on.msl, DUPLICATE_AND_MSL_ON);
    assert_eq!(off.ir.matches("and.b32").count(), 2);
    assert_eq!(on.ir.matches("and.b32").count(), 1);
    assert_eq!(off.ir.matches("xor.b32").count(), 2);
    assert_eq!(on.ir.matches("xor.b32").count(), 1);
    assert_eq!(on.ir.matches("store.u32").count(), 1);
}

#[test]
fn reuses_duplicate_not() {
    // v_not_b32 twice on the loaded word. ~x computed once.
    let code = words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        vop1(55, 2, 1), // v_not_b32 v2, v1 (VOP1 opcode 55)
        vop1(55, 3, 1), // v_not_b32 v3, v1
        0xe070_2000,
        0x8001_0300,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let off = "\
0000: t0 = load.u32 input[index]
000c: t1 = not.b32 t0
0010: t2 = not.b32 t0
0014: store.u32 output[index], t2
";
    let on = "\
0000: t0 = load.u32 input[index]
000c: t1 = not.b32 t0
0010: t2 = reuse.u32 t1
0014: store.u32 output[index], t1
";
    assert_eq!(compile_options(&code, false, false).ir, off);
    assert_eq!(compile_options(&code, false, true).ir, on);
    let msl = compile_options(&code, false, true).msl;
    assert!(msl.contains("uint t1 = ~t0;"));
    assert!(msl.contains("uint t2 = t1;"));
    assert_eq!(msl.matches("~t0").count(), 1);
}

#[test]
fn does_not_reuse_swapped_subtract() {
    // sub is not commutative: if the load is 0, v1 - 1 = 0xffffffff and
    // 1 - v1 = 1. The third sub repeats the first and is reused.
    let code = words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        1,
        vop2(38, 3, 1, 2), // v_sub_nc_u32 v3, v1, v2
        vop2(38, 4, 2, 1), // v_sub_nc_u32 v4, v2, v1
        vop2(38, 5, 1, 2), // v_sub_nc_u32 v5, v1, v2
        0xe070_2000,
        0x8001_0500,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let off = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x00000001
0014: t2 = sub.u32 t0, t1
0018: t3 = sub.u32 t1, t0
001c: t4 = sub.u32 t0, t1
0020: store.u32 output[index], t4
";
    let on = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x00000001
0014: t2 = sub.u32 t0, t1
0018: t3 = sub.u32 t1, t0
001c: t4 = reuse.u32 t2
0020: store.u32 output[index], t2
";
    assert_eq!(compile_options(&code, false, false).ir, off);
    assert_eq!(compile_options(&code, false, true).ir, on);
    assert_eq!(
        compile_options(&code, false, true)
            .ir
            .matches("sub.u32")
            .count(),
        2
    );
}

#[test]
fn does_not_reuse_a_different_opcode() {
    // and vs or of the same bits differ. 0xff00ff00 & 0x0f0f0f0f = 0x0f000f00;
    // 0xff00ff00 | 0x0f0f0f0f = 0xff0fff0f.
    let code = words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        0xff00_ff00,
        vop2(27, 3, 1, 2), // v_and_b32
        vop2(28, 4, 1, 2), // v_or_b32
        0xe070_2000,
        0x8001_0300,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let ir = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0xff00ff00
0014: t2 = and.b32 t0, t1
0018: t3 = or.b32 t0, t1
001c: store.u32 output[index], t2
";
    let off = compile_options(&code, false, false);
    let on = compile_options(&code, false, true);
    assert_eq!(off.ir, ir);
    assert_eq!(on.ir, ir);
    assert!(!on.ir.contains("reuse.u32"));
}

#[test]
fn does_not_merge_loads_or_duplicate_constant_definitions() {
    // Two loads of input[index] stay two loads. Two movs of the same immediate
    // stay two const.u32 definitions. The ands use different constant temps
    // with equal bits, so the pure and is reused and the consts remain.
    let code = words(&[
        0xe030_2000,
        0x8000_0100, // buffer_load_dword v1
        0xbf8c_3f70,
        0xe030_2000,
        0x8000_0200, // buffer_load_dword v2
        0xbf8c_3f70,
        mov_lit(3),
        0x1234_5678,
        mov_lit(4),
        0x1234_5678,
        vop2(27, 5, 1, 3), // v_and_b32 v5, v1, v3
        vop2(27, 6, 1, 4), // v_and_b32 v6, v1, v4
        0xe070_2000,
        0x8001_0600,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let off = "\
0000: t0 = load.u32 input[index]
000c: t1 = load.u32 input[index]
0018: t2 = const.u32 0x12345678
0020: t3 = const.u32 0x12345678
0028: t4 = and.b32 t0, t2
002c: t5 = and.b32 t0, t3
0030: store.u32 output[index], t5
";
    // Constant temps compare by bits, so the second and reuses the first.
    // The load temps differ, and neither load is removed.
    let on = "\
0000: t0 = load.u32 input[index]
000c: t1 = load.u32 input[index]
0018: t2 = const.u32 0x12345678
0020: t3 = const.u32 0x12345678
0028: t4 = and.b32 t0, t2
002c: t5 = reuse.u32 t4
0030: store.u32 output[index], t4
";
    assert_eq!(compile_options(&code, false, false).ir, off);
    assert_eq!(compile_options(&code, false, true).ir, on);
    assert_eq!(
        compile_options(&code, false, true)
            .ir
            .matches("load.u32")
            .count(),
        2
    );
    assert_eq!(
        compile_options(&code, false, true)
            .ir
            .matches("const.u32 0x12345678")
            .count(),
        2
    );
}

#[test]
fn does_not_merge_lane_masks() {
    // Two identical v_cmp_eq_u32_e32 (VOPC opcode 194) write the wave mask.
    // The results are lane-mask values and are not reused.
    let code = words(&[
        mov_lit(1),
        1,
        mov_lit(2),
        2,
        vopc(194, 1, 2),
        vopc(194, 1, 2),
        mov_lit(3),
        0,
        0xe070_2000,
        0x8001_0300,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let ir = "\
0000: t0 = const.u32 0x00000001
0008: t1 = const.u32 0x00000002
0010: t2 = wave.mask v_cmp_eq_u32_e32 t0, t1 exec=0xffffffff
0014: t3 = wave.mask v_cmp_eq_u32_e32 t0, t1 exec=0xffffffff
0018: t4 = const.u32 0x00000000
0020: store.u32 output[index], t4
";
    let off = compile_options(&code, false, false);
    let on = compile_options(&code, false, true);
    assert_eq!(off.ir, ir);
    assert_eq!(on.ir, ir);
    assert_eq!(on.ir.matches("wave.mask").count(), 2);
    assert!(!on.ir.contains("reuse.u32"));
}

#[test]
fn does_not_merge_lane_merges_under_partial_exec() {
    // EXEC = 1 merges each and with that destination's previous value.
    // The and itself is pure and is reused. The two merges see different
    // inactive values (the two zero temps) and both remain.
    let code = words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        0xff00_ff00,
        mov_lit(3),
        0, // v3 initialized before the partial write
        mov_lit(4),
        0,                 // v4 initialized before the partial write
        0xbefe_0381,       // s_mov_b32 exec_lo, 1
        vop2(27, 3, 1, 2), // v_and_b32 v3, v1, v2
        vop2(27, 4, 1, 2), // v_and_b32 v4, v1, v2
        0xbefe_03c1,       // s_mov_b32 exec_lo, -1
        0xe070_2000,
        0x8001_0300, // buffer_store_dword v3
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let off = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0xff00ff00
0014: t2 = const.u32 0x00000000
001c: t3 = const.u32 0x00000000
0028: t4 = and.b32 t0, t1
0028: t5 = merge.lanes mask=0x00000001, t4, t2
002c: t6 = and.b32 t0, t1
002c: t7 = merge.lanes mask=0x00000001, t6, t3
0034: store.u32 output[index], t5
";
    let on = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0xff00ff00
0014: t2 = const.u32 0x00000000
001c: t3 = const.u32 0x00000000
0028: t4 = and.b32 t0, t1
0028: t5 = merge.lanes mask=0x00000001, t4, t2
002c: t6 = reuse.u32 t4
002c: t7 = merge.lanes mask=0x00000001, t4, t3
0034: store.u32 output[index], t5
";
    assert_eq!(compile_options(&code, false, false).ir, off);
    assert_eq!(compile_options(&code, false, true).ir, on);
    assert_eq!(
        compile_options(&code, false, true)
            .ir
            .matches("merge.lanes")
            .count(),
        2
    );
}

#[test]
fn does_not_merge_vcc_select() {
    // v_cndmask_b32 reads VCC. Two identical selects stay two vcc.Select ops.
    let code = words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        0x1111_1111,
        0xbeea_0381,      // s_mov_b32 vcc_lo, 1
        vop2(1, 3, 1, 2), // v_cndmask_b32 v3, v1, v2
        vop2(1, 4, 1, 2), // v_cndmask_b32 v4, v1, v2
        0xe070_2000,
        0x8001_0300,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let ir = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x11111111
0018: t2 = const.u32 0x00000001
0018: t3 = vcc.Select t0, t1, t2
001c: t4 = const.u32 0x00000001
001c: t5 = vcc.Select t0, t1, t4
0020: store.u32 output[index], t3
";
    let off = compile_options(&code, false, false);
    let on = compile_options(&code, false, true);
    assert_eq!(off.ir, ir);
    assert_eq!(on.ir, ir);
    assert_eq!(on.ir.matches("vcc.Select").count(), 2);
    assert!(!on.ir.contains("reuse.u32"));
}

#[test]
fn cse_leaves_branch_and_float_fixtures_unchanged() {
    // No duplicate pure expression, so the pass is a no-op, including on the
    // uniform branches and the load-derived float ops.
    for (code, ir) in [
        (
            BRANCH,
            include_str!("../../../fixtures/golden/bitwise-branch.ir"),
        ),
        (
            FLOAT,
            include_str!("../../../fixtures/golden/float-exact.ir"),
        ),
    ] {
        let plain = compile(Architecture::Gfx1030, code).unwrap();
        let on = compile_options(code, false, true);
        assert_eq!(plain.ir, ir);
        assert_eq!(on.ir, plain.ir);
        assert_eq!(on.msl, plain.msl);
    }
}

#[test]
fn cse_leaves_every_golden_fixture_unchanged() {
    // None of these fixtures repeats a pure op. Several use equal-bit
    // constants in different ops whose operands must not be renamed.
    macro_rules! golden {
        ($name:literal) => {{
            let code = include_bytes!(concat!("../../../fixtures/rdna2/", $name, ".bin"));
            let shader = compile_options(code, false, true);
            assert_eq!(
                shader.ir,
                include_str!(concat!("../../../fixtures/golden/", $name, ".ir")),
                "ir changed for {}",
                $name
            );
            assert_eq!(
                shader.msl,
                include_str!(concat!("../../../fixtures/golden/", $name, ".metal")),
                "msl changed for {}",
                $name
            );
        }};
    }
    golden!("add");
    golden!("bitwise-branch");
    golden!("float-exact");
    golden!("scalar-alu");
    golden!("scalar-arith");
    golden!("scalar-compare");
    golden!("scalar-immediate");
    golden!("scalar-logic");
    golden!("scalar-unary");
    golden!("vector-compare");
    golden!("vector-compare-ft");
    golden!("vector-integer");
    golden!("vector-unary");
    golden!("vector-round");
}

/// 1 + 2 = 3, and the same add twice, then and the load with each sum.
fn fold_and_cse_fixture() -> Vec<u8> {
    words(&[
        0xe030_2000,
        0x8000_0100,
        0xbf8c_3f70,
        mov_lit(2),
        1,
        mov_lit(3),
        2,
        vop2(37, 4, 2, 3), // v_add_nc_u32 v4, v2, v3
        vop2(37, 5, 2, 3), // v_add_nc_u32 v5, v2, v3
        vop2(27, 6, 1, 4), // v_and_b32 v6, v1, v4
        vop2(27, 7, 1, 5), // v_and_b32 v7, v1, v5
        0xe070_2000,
        0x8001_0700,
        0xbbfd_0000,
        0xbf81_0000,
    ])
}

#[test]
fn cse_interacts_with_constant_folding() {
    // Hand-computed: 1 + 2 = 3 (0x00000003). Folding rewrites the adds and
    // does not reuse the ands. CSE reuses both the add and the and, and does
    // not invent the constant 3. Both passes: the adds are constants first,
    // the two const 3 temps stay, and the ands reuse because those temps have
    // the same bits.
    let code = fold_and_cse_fixture();
    let off = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x00000001
0014: t2 = const.u32 0x00000002
001c: t3 = add.u32 t1, t2
0020: t4 = add.u32 t1, t2
0024: t5 = and.b32 t0, t3
0028: t6 = and.b32 t0, t4
002c: store.u32 output[index], t6
";
    let cse_only = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x00000001
0014: t2 = const.u32 0x00000002
001c: t3 = add.u32 t1, t2
0020: t4 = reuse.u32 t3
0024: t5 = and.b32 t0, t3
0028: t6 = reuse.u32 t5
002c: store.u32 output[index], t5
";
    let fold_only = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x00000001
0014: t2 = const.u32 0x00000002
001c: t3 = const.u32 0x00000003
0020: t4 = const.u32 0x00000003
0024: t5 = and.b32 t0, t3
0028: t6 = and.b32 t0, t4
002c: store.u32 output[index], t6
";
    let both = "\
0000: t0 = load.u32 input[index]
000c: t1 = const.u32 0x00000001
0014: t2 = const.u32 0x00000002
001c: t3 = const.u32 0x00000003
0020: t4 = const.u32 0x00000003
0024: t5 = and.b32 t0, t3
0028: t6 = reuse.u32 t5
002c: store.u32 output[index], t5
";
    assert_eq!(compile_options(&code, false, false).ir, off);
    assert_eq!(compile_options(&code, false, true).ir, cse_only);
    assert_eq!(compile_options(&code, true, false).ir, fold_only);
    assert_eq!(compile_options(&code, true, true).ir, both);
    assert!(!cse_only.contains("0x00000003"));
    assert_eq!(fold_only.matches("const.u32 0x00000003").count(), 2);
    assert_eq!(both.matches("and.b32").count(), 1);
    assert!(!both.contains("add.u32"));
}

#[test]
fn cse_reuses_a_float_add_without_folding_it() {
    // 1.0 + 2.0 = 3.0 exactly, but a float add is not folded, so CSE reuses
    // the operation itself: the second add becomes a reuse of the first.
    // The fold pass does not fold a float add, so it changes nothing.
    let code = words(&[
        mov_lit(2),
        0x3f80_0000,
        mov_lit(3),
        0x4000_0000,
        vop2(3, 4, 2, 3), // v_add_f32 v4, v2, v3
        vop2(3, 5, 2, 3), // v_add_f32 v5, v2, v3
        0xe070_2000,
        0x8001_0500,
        0xbbfd_0000,
        0xbf81_0000,
    ]);
    let plain_ir = "\
0000: t0 = const.u32 0x3f800000
0008: t1 = const.u32 0x40000000
0010: t2 = add.f32 t0, t1
0014: t3 = add.f32 t0, t1
0018: store.u32 output[index], t3
";
    let cse_ir = "\
0000: t0 = const.u32 0x3f800000
0008: t1 = const.u32 0x40000000
0010: t2 = add.f32 t0, t1
0014: t3 = reuse.u32 t2
0018: store.u32 output[index], t2
";
    let cse_msl = "\
#include <metal_stdlib>
using namespace metal;
kernel void xenora_compute(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    uint t0 = 0x3f800000u;
    uint t1 = 0x40000000u;
    uint t2 = as_type<uint>(as_type<float>(t0) + as_type<float>(t1));
    uint t3 = t2;
    output[index] = t2;
}
";
    assert_eq!(compile_options(&code, false, false).ir, plain_ir);
    assert_eq!(compile_options(&code, false, true).ir, cse_ir);
    assert_eq!(compile_options(&code, false, true).msl, cse_msl);
    assert_eq!(compile_options(&code, true, false).ir, plain_ir);
    assert_eq!(compile_options(&code, true, true).ir, cse_ir);
    assert!(!cse_ir.contains("0x40400000"));
}

#[test]
fn cse_does_not_change_an_inexact_float_add() {
    // Same pair as accepts_an_inexact_float_add_of_constants: 1.0 + 2^-24 is
    // not an f32 value, but the guide's rule rounds it, so the program is
    // compiled whichever optional pass runs.
    let code = binary_const_fixture(3, 0x3f80_0000, 0x3380_0000, false);
    for fold in [false, true] {
        for cse in [false, true] {
            let result = compile_with(
                Architecture::Gfx1030,
                &code,
                CompileOptions {
                    fold_constants: fold,
                    eliminate_common_subexpressions: cse,
                },
            );
            assert!(
                result.is_ok(),
                "fold={fold} cse={cse} must accept the add: {result:?}"
            );
        }
    }
}
