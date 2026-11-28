// SPDX-License-Identifier: GPL-3.0-or-later

//! The guest CPUID contract (issue #130).
//!
//! CPUID reports a fixed synthetic processor. A feature bit is set only when
//! every instruction of that feature is implemented, so each bit that is set
//! is checked against instructions that run, and each bit software commonly
//! tests that is clear has a witness instruction of that feature which both
//! engines refuse. The witness must be removed before its bit can be set. A
//! feature whose instructions all run can also stay clear pending independent
//! acceptance; F16C has its own test for that case.
//!
//! Bit positions and the feature each instruction belongs to are from the
//! Intel SDM Vol. 2, CPUID and the instruction entries. A feature whose own
//! instructions all run stays clear while one it builds on is clear, and has
//! its instruction list checked instead. Encodings were
//! assembled with `clang -target x86_64-apple-macos` and read with `objdump`.

use xenora_loader::cpu::{
    CPUID_CMOV, CPUID_CMPXCHG16B, CPUID_CX8, CPUID_LAHF_SAHF, CPUID_LONG_MODE, CPUID_LZCNT,
    CPUID_POPCNT, CPUID_TSC, CpuState, ExitReason, Interpreter, RFlags, cpuid,
};
use xenora_loader::dbt::DbtEngine;
use xenora_loader::memory::{GuestMemory, MemoryPermissions};
use xenora_loader::x86_decode::decode;

const ENTRY: u64 = 0x1000;
const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
const RBX: usize = 3;
const RSI: usize = 6;

/// Index of a CPUID result, which come in the order EAX, EBX, ECX, EDX.
const EBX: usize = 1;
const ECX: usize = 2;
const EDX: usize = 3;

fn image(code: &[u8]) -> GuestMemory {
    let mut memory = GuestMemory::new();
    memory
        .map_region_with_data(ENTRY, code.len(), MemoryPermissions::READ_EXECUTE, code)
        .unwrap();
    memory
}

/// Final state and exit reason from the interpreter, then from the DBT.
fn both(cpu: &CpuState, code: &[u8], budget: usize) -> [(&'static str, CpuState, ExitReason); 2] {
    let mut memory = image(code);
    let interpreted = Interpreter::new(cpu.clone(), &mut memory, budget).run();
    let mut memory = image(code);
    let mut state = cpu.clone();
    let translated = DbtEngine::new(64 * 1024)
        .unwrap()
        .run(&mut state, &mut memory, budget);
    [
        (
            "interpreter",
            interpreted.final_state,
            interpreted.exit_reason,
        ),
        ("DBT", translated.final_state, translated.exit_reason),
    ]
}

const SET: RFlags = RFlags {
    cf: true,
    zf: true,
    sf: true,
    of: true,
    pf: true,
    af: true,
};

#[test]
fn every_leaf_returns_its_documented_values() {
    // (leaf, subleaf, [EAX, EBX, ECX, EDX])
    let leaves: &[(u32, u32, [u32; 4])] = &[
        // Highest basic leaf 7, and "Xeno", "uest", "ra g" as little-endian text.
        (0, 0, [7, 0x6F6E_6558, 0x7473_6575, 0x6720_6172]),
        // CMPXCHG16B is ECX bit 13, POPCNT ECX bit 23, TSC EDX bit 4,
        // CMPXCHG8B (CX8) EDX bit 8 and CMOV EDX bit 15. No signature.
        (1, 0, [0, 0, 0x0080_2000, 0x0000_8110]),
        (1, 0xFFFF_FFFF, [0, 0, 0x0080_2000, 0x0000_8110]),
        // Highest extended leaf.
        (0x8000_0000, 0, [0x8000_0001, 0, 0, 0]),
        // LAHF/SAHF is ECX bit 0, LZCNT ECX bit 5 and long mode EDX bit 29.
        (0x8000_0001, 0, [0, 0, 0x0000_0021, 0x2000_0000]),
        (0x8000_0001, 7, [0, 0, 0x0000_0021, 0x2000_0000]),
        // Everything else, above or between the highest leaves, is zeros.
        (2, 0, [0; 4]),
        (4, 0, [0; 4]),
        (6, 0, [0; 4]),
        // Leaf 7 subleaf 0: EAX zero further subleaves, BMI1 EBX bit 3 and
        // BMI2 EBX bit 8. Every other subleaf is undefined space.
        (7, 0, [0, 0x108, 0, 0]),
        (7, 1, [0; 4]),
        (7, 0xFFFF_FFFF, [0; 4]),
        (0xB, 0, [0; 4]),
        (0xD, 1, [0; 4]),
        (0x16, 0, [0; 4]),
        (0x4000_0000, 0, [0; 4]),
        (0x7FFF_FFFF, 0, [0; 4]),
        (0x8000_0002, 0, [0; 4]),
        (0x8000_0008, 0, [0; 4]),
        (0xFFFF_FFFF, 0, [0; 4]),
    ];
    for &(leaf, subleaf, wanted) in leaves {
        // Only EAX and ECX select the leaf: the upper halves are noise.
        let mut cpu = CpuState::new(ENTRY, 0);
        for (index, register) in cpu.gpr.iter_mut().enumerate() {
            *register = 0xA5A5_0000_1111_0000 + index as u64;
        }
        cpu.gpr[RAX] = 0xDEAD_BEEF_0000_0000 | u64::from(leaf);
        cpu.gpr[RCX] = 0xFEED_FACE_0000_0000 | u64::from(subleaf);
        cpu.flags = SET;
        // Each result is a 32-bit register write, so it clears the upper half.
        // CPUID changes no flag and no other register.
        let mut expected = cpu.clone();
        expected.rip = ENTRY + 2;
        for (register, value) in [RAX, RBX, RCX, RDX].into_iter().zip(wanted) {
            expected.gpr[register] = u64::from(value);
        }
        for (engine, state, exit) in both(&cpu, &[0x0F, 0xA2], 1) {
            let what = format!("leaf {leaf:#x} subleaf {subleaf:#x} on the {engine}");
            assert_eq!(exit, ExitReason::StepLimitExceeded, "{what}");
            assert_eq!(state, expected, "{what}");
        }
        // The function a runtime service would read agrees with the instruction.
        assert_eq!(cpuid(leaf, subleaf), wanted, "leaf {leaf:#x}");
    }

    let [_, ebx, ecx, edx] = cpuid(0, 0);
    let vendor: Vec<u8> = [ebx, edx, ecx]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    assert_eq!(vendor, b"Xenora guest");
    assert_eq!(
        (
            CPUID_POPCNT,
            CPUID_CMOV,
            CPUID_TSC,
            CPUID_CX8,
            CPUID_CMPXCHG16B,
            CPUID_LAHF_SAHF,
            CPUID_LZCNT,
            CPUID_LONG_MODE
        ),
        (
            1 << 23,
            1 << 15,
            1 << 4,
            1 << 8,
            1 << 13,
            1,
            1 << 5,
            1 << 29
        )
    );
}

#[test]
fn reported_features_have_all_their_instructions_in_the_fixtures() {
    // `tests/x86_integer.rs` runs every line of this file through both engines
    // against an independent reference. Here each reported feature must have
    // its whole instruction set among those lines.
    let fixture = include_str!("x86/integer.txt");
    let decodable = |text: &str| {
        let line = fixture
            .lines()
            .find(|line| line.split_once(" | ").is_some_and(|(_, asm)| asm == text))
            .unwrap_or_else(|| panic!("no fixture for {text}"));
        let code: Vec<u8> = line
            .split_once(" | ")
            .unwrap()
            .0
            .split(' ')
            .map(|byte| u8::from_str_radix(byte, 16).unwrap())
            .collect();
        decode(ENTRY, |at| {
            code.get((at - ENTRY) as usize).copied().ok_or(())
        })
        .unwrap_or_else(|error| panic!("{text}: {error:?}"));
    };
    let memory = |bits: u32| {
        let width = match bits {
            16 => "word",
            32 => "dword",
            _ => "qword",
        };
        format!("{width} ptr [rsi + 4*rdi + 16]")
    };
    // 16-, 32- and 64-bit destinations (RBX and R10) from a register and from memory.
    let sizes = [
        (16, "bx", "r9w", "r10w"),
        (32, "ebx", "r9d", "r10d"),
        (64, "rbx", "r9", "r10"),
    ];

    assert_ne!(cpuid(1, 0)[EDX] & CPUID_CMOV, 0);
    for condition in "o no b ae e ne be a s ns p np l ge le g".split(' ') {
        for (bits, low, source, high) in sizes {
            decodable(&format!("cmov{condition} {low}, {source}"));
            decodable(&format!("cmov{condition} {high}, {}", memory(bits)));
        }
    }
    for (feature, bit, name) in [
        (cpuid(1, 0)[ECX], CPUID_POPCNT, "popcnt"),
        (cpuid(0x8000_0001, 0)[ECX], CPUID_LZCNT, "lzcnt"),
    ] {
        assert_ne!(feature & bit, 0, "{name}");
        for (bits, low, source, high) in sizes {
            decodable(&format!("{name} {low}, {source}"));
            decodable(&format!("{name} {high}, {}", memory(bits)));
        }
    }
    // CMPXCHG8B and CMPXCHG16B are one memory-only form, 8 or 16 bytes by
    // REX.W. The reference of `tests/x86_integer.rs` has no operand that wide,
    // so `tests/x86_cmpxchg_pair.rs` and the SDM known answers run them instead.
    assert_ne!(cpuid(1, 0)[EDX] & CPUID_CX8, 0);
    assert_ne!(cpuid(1, 0)[ECX] & CPUID_CMPXCHG16B, 0);
    for code in [&[0x0F, 0xC7, 0x0E][..], &[0x48, 0x0F, 0xC7, 0x0E]] {
        decode(ENTRY, |at| {
            code.get((at - ENTRY) as usize).copied().ok_or(())
        })
        .unwrap_or_else(|error| panic!("{code:02X?}: {error:?}"));
    }
    // The SDM's TSC bit says that RDTSC is supported, including CR4.TSD for
    // controlling privilege. RDTSC is one form without operands. CR4.TSD
    // belongs to an operating system and is not modelled; the instruction runs
    // as it does while that flag is clear. RDTSCP has a bit of its own
    // (0x8000_0001 EDX bit 27), which stays clear.
    assert_ne!(cpuid(1, 0)[EDX] & CPUID_TSC, 0);
    decodable("rdtsc");
    // BMI1 and BMI2: every form of the fourteen instructions has fixture
    // rows in integer.txt, all of which `tests/x86_integer.rs` runs through
    // both engines against the independent reference. Here each instruction
    // must have its rows and every one of them must decode.
    use xenora_loader::cpu::{CPUID7_BMI1, CPUID7_BMI2};
    assert_ne!(cpuid(7, 0)[EBX] & CPUID7_BMI1, 0);
    assert_ne!(cpuid(7, 0)[EBX] & CPUID7_BMI2, 0);
    for name in
        "andn bextr blsi blsmsk blsr tzcnt bzhi mulx pdep pext rorx sarx shlx shrx".split(' ')
    {
        let rows: Vec<(&str, &str)> = fixture
            .lines()
            .filter_map(|line| line.split_once(" | "))
            .filter(|(_, asm)| asm.split(' ').next() == Some(name))
            .collect();
        assert!(!rows.is_empty(), "no fixture rows for {name}");
        for (hex, asm) in rows {
            let code: Vec<u8> = hex
                .split(' ')
                .map(|byte| u8::from_str_radix(byte, 16).unwrap())
                .collect();
            decode(ENTRY, |at| {
                code.get((at - ENTRY) as usize).copied().ok_or(())
            })
            .unwrap_or_else(|error| panic!("{asm}: {error:?}"));
        }
    }
    // LAHF and SAHF have one form each and no operands.
    assert_ne!(cpuid(0x8000_0001, 0)[ECX] & CPUID_LAHF_SAHF, 0);
    decodable("lahf");
    decodable("sahf");
    // Long mode: 64-bit operands and the registers only it has.
    assert_ne!(cpuid(0x8000_0001, 0)[EDX] & CPUID_LONG_MODE, 0);
    decodable("add r9, r12");
    decodable("movabs r15, 0x8877665544332211");
}

#[test]
fn unreported_features_each_have_an_instruction_that_is_refused() {
    // (feature, leaf, result register, bit, an instruction of that feature)
    let witnesses: &[(&str, u32, usize, u32, &[u8])] = &[
        ("FPU", 1, EDX, 0, &[0xD9, 0xE8]),               // fld1
        ("CLFSH", 1, EDX, 19, &[0x0F, 0xAE, 0x3E]),      // clflush [rsi]
        ("MMX", 1, EDX, 23, &[0x0F, 0x77]),              // emms
        ("FXSR", 1, EDX, 24, &[0x0F, 0xAE, 0x06]),       // fxsave [rsi]
        ("SSE", 1, EDX, 25, &[0x0F, 0x52, 0xC1]),        // rsqrtps xmm0, xmm1
        ("SSE2", 1, EDX, 26, &[0x66, 0x0F, 0xF7, 0xC1]), // maskmovdqu xmm0, xmm1
        ("SSE3", 1, ECX, 0, &[0xF2, 0x0F, 0x7C, 0xC1]),  // haddps xmm0, xmm1
        // Every legacy imm8 is accepted: SDM Table 4-13 ignores all bits but
        // 0 and 4. The VEX form remains refused and is the witness.
        (
            "PCLMULQDQ",
            1,
            ECX,
            1,
            &[0xC4, 0xE3, 0x71, 0x44, 0xC2, 0x00],
        ), // vpclmulqdq xmm0, xmm1, xmm2, 0
        ("SSSE3", 1, ECX, 9, &[0x0F, 0x38, 0x00, 0xC1]), // pshufb mm0, mm1
        ("MOVBE", 1, ECX, 22, &[0x0F, 0x38, 0xF0, 0x06]), // movbe eax, [rsi]
        ("AESNI", 1, ECX, 25, &[0x66, 0x0F, 0x38, 0xDC, 0xC1]), // aesenc xmm0, xmm1
        ("XSAVE", 1, ECX, 26, &[0x0F, 0x01, 0xD0]),      // xgetbv
        ("AVX", 1, ECX, 28, &[0xC5, 0xFC, 0x53, 0xC1]),  // vrcpps ymm0, ymm1
        ("RDRAND", 1, ECX, 30, &[0x0F, 0xC7, 0xF0]),     // rdrand eax
        // AVX2 cross-lane permutes, broadcasts, blends, per-lane shifts and
        // the 256-bit immediate shuffles are implemented; moving the 256-bit
        // byte mask to a general register is still refused.
        ("AVX2", 7, EBX, 5, &[0xC5, 0xFD, 0xD7, 0xC1]), // vpmovmskb eax, ymm1
        ("RDSEED", 7, EBX, 18, &[0x0F, 0xC7, 0xF8]),    // rdseed eax
        ("ADX", 7, EBX, 19, &[0x66, 0x0F, 0x38, 0xF6, 0xC1]), // adcx eax, ecx
        ("SHA", 7, EBX, 29, &[0x0F, 0x38, 0xC9, 0xC1]), // sha1msg1 xmm0, xmm1
        ("PREFETCHW", 0x8000_0001, ECX, 8, &[0x0F, 0x0D, 0x0E]), // prefetchw [rsi]
        ("SYSCALL/SYSRET", 0x8000_0001, EDX, 11, &[0x48, 0x0F, 0x07]), // sysretq
        ("RDTSCP", 0x8000_0001, EDX, 27, &[0x0F, 0x01, 0xF9]), // rdtscp
    ];
    for &(feature, leaf, register, bit, code) in witnesses {
        assert_eq!(
            cpuid(leaf, 0)[register] >> bit & 1,
            0,
            "{feature} is reported"
        );
        let mut padded = code.to_vec();
        padded.extend([0x90; 8]);
        let mut cpu = CpuState::new(ENTRY, 0);
        cpu.gpr[RSI] = ENTRY;
        for (engine, state, exit) in both(&cpu, &padded, 1) {
            assert!(
                matches!(exit, ExitReason::IllegalInstruction(_)),
                "{feature}: {code:02X?} ran on the {engine} ({exit:?}); report the feature only \
                 once all of it is implemented"
            );
            assert_eq!(state, cpu, "{feature} on the {engine}");
        }
    }
}

/// F16C is fully implemented but its CPUID bit stays clear until independent
/// acceptance. Unlike the witnesses above, the instruction runs in both engines.
#[test]
fn f16c_is_implemented_but_unreported() {
    assert_eq!(cpuid(1, 0)[ECX] >> 29 & 1, 0);
    let code = [0xC4, 0xE2, 0x79, 0x13, 0xC1]; // vcvtph2ps xmm0, xmm1
    let mut cpu = CpuState::new(ENTRY, 0);
    cpu.xmm[1] = 0x3C00; // binary16 1.0
    for (engine, state, exit) in both(&cpu, &code, 1) {
        assert!(
            !matches!(exit, ExitReason::IllegalInstruction(_)),
            "{engine}: {exit:?}"
        );
        assert_eq!(state.xmm[0], 0x3F80_0000, "{engine}");
    }
}

/// BMI1 and BMI2 are complete — every form of ANDN, BEXTR, BLSI, BLSMSK,
/// BLSR, TZCNT, BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX and SHRX executes in
/// both engines — so leaf 7 reports both bits, and a guest that tests them
/// before using the forms reaches working instructions.
#[test]
fn bmi1_and_bmi2_are_complete_and_reported() {
    use xenora_loader::cpu::{CPUID7_BMI1, CPUID7_BMI2};
    let ebx = cpuid(7, 0)[EBX];
    assert_ne!(ebx & CPUID7_BMI1, 0, "BMI1 reported");
    assert_ne!(ebx & CPUID7_BMI2, 0, "BMI2 reported");
    assert_eq!(cpuid(7, 0)[0], 0, "subleaf 0 is the only feature leaf");
    // One execution sample per set: the fixture suites cover every form in
    // both engines; here the bit a guest tests answers the truth.
    for (code, what) in [
        (&[0xC4, 0xE2, 0x33, 0xF5, 0xD9][..], "pdep ebx, r9d, ecx"),
        (&[0xC4, 0xE2, 0x73, 0xF6, 0xC2][..], "mulx eax, ecx, edx"),
    ] {
        let cpu = CpuState::new(ENTRY, 0);
        for (engine, _state, exit) in both(&cpu, code, 1) {
            assert!(
                !matches!(exit, ExitReason::IllegalInstruction(_)),
                "{what} refused on the {engine}: {exit:?}"
            );
        }
    }
}

/// SDM Vol. 1, 12.12: an application checks for SSE, SSE2, SSE3 and SSSE3 before
/// SSE4.1, and for SSE, SSE2, SSSE3 and SSE4.1 before most of SSE4.2. Those are not
/// complete, so the two bits stay clear although every instruction of both runs.
/// Each has a fixture that the tests of `x86_integer.rs`,
/// `x86_vector.rs` or `x86_float.rs` run in both engines.
#[test]
fn sse4_is_complete_but_unreported_while_its_prerequisites_are() {
    let fixtures = [
        include_str!("x86/integer.txt"),
        include_str!("x86/vector.txt"),
        include_str!("x86/float.txt"),
    ]
    .concat();
    let has = |name: &str, memory: bool| {
        fixtures.lines().any(|line| {
            let text = line.split_once(" | ").unwrap().1;
            text.split(' ').next() == Some(name) && text.contains(" ptr ") == memory
        })
    };
    // Every instruction the SDM assigns to the features; POPCNT has its own bit,
    // which is reported. MOVNTDQA has no register
    // form; PEXTRW belongs to SSE4.1 only as a store.
    let names = "blendpd blendps blendvpd blendvps dppd dpps extractps insertps mpsadbw packusdw \
                 pblendvb pblendw pcmpeqq pextrb pextrd pextrq phminposuw pinsrb pinsrd pinsrq \
                 pmaxsb pmaxsd pmaxud pmaxuw pminsb pminsd pminud pminuw pmovsxbw pmovsxbd \
                 pmovsxbq pmovsxwd pmovsxwq pmovsxdq pmovzxbw pmovzxbd pmovzxbq pmovzxwd \
                 pmovzxwq pmovzxdq pmuldq pmulld ptest roundpd roundps roundsd roundss \
                 crc32 pcmpestri pcmpestrm pcmpgtq pcmpistri pcmpistrm";
    for name in names.split_whitespace() {
        assert!(has(name, false) && has(name, true), "{name}");
    }
    assert!(has("movntdqa", true) && has("pextrw", true));

    let [_, _, ecx, edx] = cpuid(1, 0);
    let prerequisites = [edx >> 25, edx >> 26, ecx, ecx >> 9].map(|bit| bit & 1);
    assert_eq!(
        (prerequisites, ecx >> 19 & 3),
        ([0; 4], 0),
        "SSE, SSE2, SSE3 and SSSE3 come before SSE4"
    );
}

/// The whole FMA3 family runs, in both engines and through `tests/x86_fma.rs`,
/// but the bit stays clear until independent acceptance.
#[test]
fn fma_is_complete_but_unreported() {
    let fixture = include_str!("x86/fma.txt");
    let has = |name: &str, memory: bool| {
        fixture.lines().any(|line| {
            let text = line.split_once(" | ").unwrap().1;
            text.split(' ').next() == Some(name) && text.contains(" ptr ") == memory
        })
    };
    for family in ["fmadd", "fmsub", "fnmadd", "fnmsub", "fmaddsub", "fmsubadd"] {
        let alternating = family.starts_with("fmaddsub") || family.starts_with("fmsubadd");
        for ordering in ["132", "213", "231"] {
            for kind in ["ps", "pd"] {
                let name = format!("v{family}{ordering}{kind}");
                assert!(has(&name, false) && has(&name, true), "{name}");
            }
            if alternating {
                continue;
            }
            for kind in ["ss", "sd"] {
                let name = format!("v{family}{ordering}{kind}");
                assert!(has(&name, false) && has(&name, true), "{name}");
            }
        }
    }
    let [_, _, ecx, _] = cpuid(1, 0);
    assert_eq!(ecx >> 12 & 1, 0, "FMA is reported before acceptance");
}

#[test]
fn a_guest_can_use_an_instruction_after_testing_its_feature_bit() {
    //       mov eax, 1
    //       cpuid
    //       bt ecx, 23        ; POPCNT
    //       jnc done
    //       popcnt rax, rsi
    // done: hlt
    let code = [
        0xB8, 0x01, 0x00, 0x00, 0x00, 0x0F, 0xA2, 0x0F, 0xBA, 0xE1, 0x17, 0x73, 0x05, 0xF3, 0x48,
        0x0F, 0xB8, 0xC6, 0xF4,
    ];
    let mut cpu = CpuState::new(ENTRY, 0);
    cpu.gpr[RSI] = 0xF0F0_0000_0000_0001;
    for (engine, state, exit) in both(&cpu, &code, 16) {
        assert_eq!(exit, ExitReason::Halt, "{engine}");
        // Nine bits are set in RSI. Leaf 1 leaves EAX zero, so a skipped
        // POPCNT would show as 0.
        assert_eq!(state.gpr[RAX], 9, "{engine}");
        assert_eq!(state.rip, ENTRY + code.len() as u64, "{engine}");
    }
}
