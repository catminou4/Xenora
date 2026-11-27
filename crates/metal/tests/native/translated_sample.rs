// SPDX-License-Identifier: GPL-3.0-or-later

use xenora_metal::sampler::MetalSampler;
use xenora_metal::texture_pipeline::{MetalRgba8Texture, Rgba8MipLevel, Rgba8TextureDesc};
use xenora_shader::{Architecture, ENTRY_POINT, compile};
use xenora_texture::SamplerDesc;

use super::context;

const LZ_DMASK1: &[u8] = include_bytes!("../../../../fixtures/rdna2/image-sample-lz-dmask1.bin");
const LZ_DMASK8: &[u8] = include_bytes!("../../../../fixtures/rdna2/image-sample-lz-dmask8.bin");
const LZ_DMASK5: &[u8] = include_bytes!("../../../../fixtures/rdna2/image-sample-lz-dmask5.bin");
const LZ_DMASKF: &[u8] = include_bytes!("../../../../fixtures/rdna2/image-sample-lz-dmaskf.bin");
const L_LOD0: &[u8] = include_bytes!("../../../../fixtures/rdna2/image-sample-l-lod0.bin");
const L_LOD1: &[u8] = include_bytes!("../../../../fixtures/rdna2/image-sample-l-lod1.bin");

/// The 20 per-lane (u, v) cases, in the same order as the CPU-oracle test. Every
/// value has a zero low 16-bit half so the fixture can pack both into one word.
const CASES: [(f32, f32); 20] = [
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

fn texels() -> (Vec<u8>, Vec<u8>) {
    let mut mip0 = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            for channel in 0..4 {
                mip0.push(texel_byte(0, x, y, channel));
            }
        }
    }
    let mut mip1 = Vec::new();
    for y in 0..2 {
        for x in 0..2 {
            for channel in 0..4 {
                mip1.push(texel_byte(1, x, y, channel));
            }
        }
    }
    (mip0, mip1)
}

/// The synthetic input: a 4x4 RGBA8 unorm T# with two levels (base field 1 =
/// byte offset 0x100), an S# with point mip filtering and a 1.0 maximum LOD,
/// the 20 packed coordinate cases at words 12..32, and the texels at word 64.
/// The native kernel reads every word as a packed coordinate; the texture
/// itself is bound separately.
fn input() -> Vec<u32> {
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
    input.extend(CASES.map(|(u, v)| packed(u, v)));
    input.resize(64, 0);
    let (mip0, mip1) = texels();
    for word in mip0
        .as_chunks::<4>()
        .0
        .iter()
        .chain(mip1.as_chunks::<4>().0.iter())
    {
        input.push(u32::from_le_bytes(*word));
    }
    input
}

fn bound_texture() -> MetalRgba8Texture {
    let (mip0, mip1) = texels();
    let desc = Rgba8TextureDesc::new(4, 4, 2).expect("valid 4x4 two-level texture");
    let level0 = Rgba8MipLevel::new_packed(0, 4, 4, &mip0);
    let level1 = Rgba8MipLevel::new_packed(1, 2, 2, &mip1);
    MetalRgba8Texture::upload(context().device(), desc, &[level0, level1])
        .expect("upload RGBA8 4x4 + 2x2 texture to Metal")
}

fn bound_sampler() -> MetalSampler {
    MetalSampler::new(context().device(), SamplerDesc::nearest_clamp())
        .expect("create native Metal sampler")
}

fn texel_index(u: f32, v: f32, mip: u32) -> (usize, usize) {
    let extent = 4usize >> mip;
    let x = ((u.clamp(0.0, 1.0) * extent as f32).floor() as usize).min(extent - 1);
    let y = ((v.clamp(0.0, 1.0) * extent as f32).floor() as usize).min(extent - 1);
    (x, y)
}

/// One output word per dispatch index. Every lane loads its packed coordinate
/// word `input[lane]`, splits it into u and v, and samples `component` of the
/// mip. The descriptor and texel words follow the same formula.
fn expected(input: &[u32], mip: u32, component: usize) -> Vec<u32> {
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

/// The destination VGPR of every DMASK-selected component, in R, G, B, A order
/// of the set bits. VDATA is v5.
fn destinations(dmask: u8) -> Vec<(u8, usize)> {
    let mut destinations = Vec::new();
    let mut slot = 5u8;
    for component in 0..4 {
        if dmask & (1 << component) != 0 {
            destinations.push((slot, component));
            slot += 1;
        }
    }
    destinations
}

/// Rewrite the single output store to read a different destination VGPR.
fn store_destination(code: &[u8], vgpr: u8) -> Vec<u8> {
    let mut code = code.to_vec();
    let store = code.len() - 16;
    assert_eq!(code[store + 7], 0x80, "buffer_store encoding");
    code[store + 5] = vgpr;
    code
}

fn check_fixture(fixture: &[u8], dmask: u8, mip: u32, explicit: bool) {
    let input = input();
    let texture = bound_texture();
    let sampler = bound_sampler();
    for (destination, component) in destinations(dmask) {
        let code = store_destination(fixture, destination);
        let compiled = compile(Architecture::Gfx1030, &code)
            .expect("compile translated image sample compute shader");
        let form = if explicit {
            "sample_l.2d.f32"
        } else {
            "sample_lz.2d.f32"
        };
        assert!(
            compiled.ir.contains(form),
            "IR must name {form}: {}",
            compiled.ir
        );
        assert!(
            compiled
                .msl
                .contains("texture2d<float, access::sample> tex0 [[texture(0)]]")
        );
        assert!(compiled.msl.contains("sampler samp0 [[sampler(0)]]"));

        let wanted = expected(&input, mip, component);
        let oracle = xenora_ref::execute(&code, &input).expect("CPU reference");
        assert_eq!(
            oracle, wanted,
            "independent CPU oracle for destination v{destination}"
        );
        let actual = context()
            .run_words_sampled(&compiled.msl, ENTRY_POINT, &input, &texture, &sampler)
            .expect("compiled and completed native Metal dispatch");
        assert_eq!(
            actual, wanted,
            "GPU destination v{destination} must match the CPU oracle"
        );
    }
}

#[test]
fn translated_image_sample_lz_dmask_matrix() {
    for (fixture, dmask) in [
        (LZ_DMASK1, 0x1),
        (LZ_DMASK8, 0x8),
        (LZ_DMASK5, 0x5),
        (LZ_DMASKF, 0xf),
    ] {
        check_fixture(fixture, dmask, 0, false);
    }
    println!(
        "device={} verified LZ DMASK 1/8/5/F",
        context().device_name()
    );
}

#[test]
fn translated_image_sample_l_lod_matrix() {
    check_fixture(L_LOD0, 0xf, 0, true);
    check_fixture(L_LOD1, 0xf, 1, true);
    println!(
        "device={} verified IMAGE_SAMPLE_L at LOD 0 and 1",
        context().device_name()
    );
}

#[test]
fn translated_image_sample_rejects_disagreeing_bindings() {
    let input = input();
    let compiled = compile(Architecture::Gfx1030, LZ_DMASK1)
        .expect("compile translated image sample compute shader");
    let sampler = bound_sampler();

    let small = Rgba8TextureDesc::new(2, 2, 1).expect("valid 2x2 texture");
    let pixel = [0u8; 16];
    let small_texture = MetalRgba8Texture::upload(
        context().device(),
        small,
        &[Rgba8MipLevel::new_packed(0, 2, 2, &pixel)],
    )
    .expect("upload 2x2 texture");
    let error = context()
        .run_words_sampled(&compiled.msl, ENTRY_POINT, &input, &small_texture, &sampler)
        .expect_err("a texture whose dimensions disagree with the T# must be refused");
    assert_eq!(error.operation, "image resource descriptor");

    let limited = MetalSampler::new(
        context().device(),
        SamplerDesc::nearest_clamp().with_lod_range(0.0, 0.0),
    )
    .expect("create sampler with a 0.0 maximum LOD");
    let error = context()
        .run_words_sampled(
            &compiled.msl,
            ENTRY_POINT,
            &input,
            &bound_texture(),
            &limited,
        )
        .expect_err("a sampler whose LOD range cannot reach the last level must be refused");
    assert_eq!(error.operation, "image sampler descriptor");
}

#[test]
fn translated_image_sample_rejects_invalid_descriptors() {
    let compiled = compile(Architecture::Gfx1030, LZ_DMASK1)
        .expect("compile translated image sample compute shader");
    let texture = bound_texture();
    let sampler = bound_sampler();

    let mut words = input();
    words[3] = 0x8000_0fac; // type 8 (1D), not 9 (2D)
    let error = context()
        .run_words_sampled(&compiled.msl, ENTRY_POINT, &words, &texture, &sampler)
        .expect_err("a non-2D resource descriptor must be refused");
    assert_eq!(error.operation, "image resource descriptor");

    let mut words = input();
    words[10] = 0; // mip filter none (0) instead of point (1)
    let error = context()
        .run_words_sampled(&compiled.msl, ENTRY_POINT, &words, &texture, &sampler)
        .expect_err("a sampler without point mip filtering must be refused");
    assert_eq!(error.operation, "image sampler descriptor");
}

// Runtime sample-domain enforcement. The CPU reference refuses a dispatch in
// which any active lane has a non-whole explicit LOD or a NaN or infinite
// coordinate; the translated kernel must refuse the same dispatches instead of
// letting Metal round or clamp the value.

/// The EXEC mask of the runtime-LOD program: lanes 20..31 of every wave. Lanes
/// 0..11 hold the descriptors and lanes 0..19 of wave 2 the texels, so every
/// active lane reads a coordinate word the test controls or a zero pad word.
const RUNTIME_EXEC: u32 = 0xfff0_0000;

fn code_words(code: &[u8]) -> Vec<u32> {
    code.as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

fn code_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The LOD-1 fixture with the LOD taken from each lane's own v coordinate
/// (`v_mov_b32 v4, v3` replaces `v_mov_b32 v4, 1.0`, inline constant 0xf2)
/// and the sample run under
/// [`RUNTIME_EXEC`]. The LOD is guest data read at run time, not a constant.
fn runtime_lod_program() -> Vec<u8> {
    let words = code_words(L_LOD1);
    let mov = words
        .iter()
        .position(|&word| word == 0x7e08_02f2)
        .expect("v_mov_b32 v4, 1.0 in the LOD-1 fixture");
    let sample = words
        .iter()
        .position(|word| word >> 26 == 0x3c)
        .expect("MIMG image_sample_l in the LOD-1 fixture");
    let mut program = words[..mov].to_vec();
    program.push(0x7e08_0303); // v_mov_b32_e32 v4, v3
    program.extend_from_slice(&words[mov + 1..sample]);
    program.extend([0xbefe_03ff, RUNTIME_EXEC]); // s_mov_b32 exec_lo, mask
    program.extend_from_slice(&words[sample..sample + 2]);
    program.push(0xbefe_03c1); // s_mov_b32 exec_lo, -1: the store needs a full wave
    program.extend_from_slice(&words[sample + 2..]);
    code_bytes(&program)
}

/// `input()` with every coordinate lane the runtime-LOD program activates
/// holding a whole-level v (0.0 or 1.0), so the dispatch is in the domain.
fn runtime_lod_input() -> Vec<u32> {
    let mut input = input();
    let us = [0.125f32, 0.375, 0.625, 0.875, -0.5, 1.5];
    for lane in 20..32 {
        let v = if lane % 2 == 0 { 0.0f32 } else { 1.0 };
        input[lane] = packed(us[lane % us.len()], v);
    }
    input
}

fn set_v(input: &mut [u32], lane: usize, v: f32) {
    input[lane] = (input[lane] & 0xffff) | (v.to_bits() & 0xffff_0000);
}

/// Red of every lane: active lanes sample the mip their v names, inactive
/// lanes keep the zero their destination was initialised to.
fn runtime_lod_expected(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .enumerate()
        .map(|(lane, &word)| {
            if (RUNTIME_EXEC >> (lane & 31)) & 1 == 0 {
                return 0;
            }
            let u = f32::from_bits((word & 0xffff) << 16);
            let v = f32::from_bits(word & 0xffff_0000);
            let mip = v as u32;
            let (x, y) = texel_index(u, v, mip);
            (f32::from(texel_byte(mip, x, y, 0)) / 255.0).to_bits()
        })
        .collect()
}

#[test]
fn translated_sample_fault_bits_match_the_compiler() {
    assert_eq!(xenora_shader::SAMPLE_FAULT_LOD, 1);
    assert_eq!(xenora_shader::SAMPLE_FAULT_COORDINATES, 2);
}

#[test]
fn translated_image_sample_l_runtime_lod_in_domain_matches_the_reference() {
    let code = runtime_lod_program();
    let compiled = compile(Architecture::Gfx1030, &code).expect("compile runtime-LOD sample");
    assert!(
        compiled
            .msl
            .contains("device atomic_uint* fault [[buffer(2)]]")
    );
    assert!(compiled.msl.contains("xenora_whole_level("));
    let input = runtime_lod_input();
    let wanted = runtime_lod_expected(&input);
    assert!(
        wanted
            .iter()
            .enumerate()
            .any(|(lane, &value)| { lane % 2 == 1 && (20..32).contains(&lane) && value != 0 }),
        "the in-domain case must sample mip 1 in some lane"
    );
    let oracle = xenora_ref::execute(&code, &input).expect("CPU reference");
    assert_eq!(oracle, wanted, "independent CPU oracle");
    let actual = context()
        .run_words_sampled(
            &compiled.msl,
            ENTRY_POINT,
            &input,
            &bound_texture(),
            &bound_sampler(),
        )
        .expect("in-domain runtime LOD must run");
    assert_eq!(actual, wanted, "GPU must match the CPU oracle");
}

#[test]
fn translated_image_sample_l_refuses_out_of_domain_runtime_lod() {
    let code = runtime_lod_program();
    let compiled = compile(Architecture::Gfx1030, &code).expect("compile runtime-LOD sample");
    let texture = bound_texture();
    let sampler = bound_sampler();
    let below_one = f32::from_bits(1.0f32.to_bits() - 1);
    // One active lane (25) carries the bad LOD; every other lane is in domain.
    for (lod, coordinates_bad) in [
        (0.5f32, false),    // fractional
        (1.5, false),       // fractional between the two levels
        (below_one, false), // the largest value below a whole level
        (2.0, false),       // whole but above the last level (1)
        (15.0, false),      // far above the last level
        (-1.0, false),      // negative
        (f32::NAN, true),   // NaN, also a NaN coordinate
        (f32::INFINITY, true),
        (f32::NEG_INFINITY, true),
    ] {
        let mut input = runtime_lod_input();
        set_v(&mut input, 25, lod);
        let reference = xenora_ref::execute(&code, &input)
            .expect_err("the CPU reference refuses an out-of-domain LOD");
        let error = context()
            .run_words_sampled(&compiled.msl, ENTRY_POINT, &input, &texture, &sampler)
            .expect_err("the GPU must refuse an out-of-domain LOD the reference refuses");
        assert_eq!(error.operation, "image sample domain", "lod {lod}: {error}");
        let text = error.to_string();
        assert!(
            text.contains("LOD is not a whole mip level"),
            "lod {lod}: {text}"
        );
        assert_eq!(
            text.contains("NaN or infinite"),
            coordinates_bad,
            "lod {lod}: {text}; reference: {reference:?}"
        );
    }
    // The same values in an inactive lane (15) do not affect the dispatch.
    for lod in [0.5f32, 2.0, -1.0, f32::NAN] {
        let mut input = runtime_lod_input();
        set_v(&mut input, 15, lod);
        let wanted = runtime_lod_expected(&input);
        assert_eq!(
            xenora_ref::execute(&code, &input).expect("inactive lane is ignored"),
            wanted
        );
        let actual = context()
            .run_words_sampled(&compiled.msl, ENTRY_POINT, &input, &texture, &sampler)
            .expect("an inactive lane's LOD must not refuse the dispatch");
        assert_eq!(actual, wanted, "inactive lod {lod}");
    }
    // The device keeps working after a refused dispatch.
    let input = runtime_lod_input();
    let actual = context()
        .run_words_sampled(&compiled.msl, ENTRY_POINT, &input, &texture, &sampler)
        .expect("a later in-domain dispatch runs");
    assert_eq!(actual, runtime_lod_expected(&input));
    println!(
        "device={} refused fractional, out-of-range and non-finite runtime LODs",
        context().device_name()
    );
}

#[test]
fn translated_image_sample_lz_refuses_non_finite_coordinates() {
    let compiled = compile(Architecture::Gfx1030, LZ_DMASK1).expect("compile LZ sample");
    assert!(compiled.msl.contains("xenora_finite("));
    assert!(
        !compiled.msl.contains("xenora_whole_level(v"),
        "LZ has no explicit LOD to check"
    );
    let texture = bound_texture();
    let sampler = bound_sampler();
    for (lane, u, v) in [
        (20, f32::NAN, 0.125f32),
        (21, 0.125, f32::NAN),
        (22, f32::INFINITY, 0.125),
        (23, 0.125, f32::NEG_INFINITY),
    ] {
        let mut input = input();
        input[lane] = packed(u, v);
        xenora_ref::execute(LZ_DMASK1, &input)
            .expect_err("the CPU reference refuses a non-finite coordinate");
        let error = context()
            .run_words_sampled(&compiled.msl, ENTRY_POINT, &input, &texture, &sampler)
            .expect_err("the GPU must refuse a non-finite coordinate");
        assert_eq!(error.operation, "image sample domain");
        let text = error.to_string();
        assert!(text.contains("NaN or infinite"), "({u}, {v}): {text}");
        assert!(!text.contains("LOD"), "LZ reports no LOD fault: {text}");
    }
}
