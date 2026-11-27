// SPDX-License-Identifier: GPL-3.0-or-later
//! Every-bit checks of the MIMG T# and S# validators (AMD RDNA 2 ISA guide,
//! Tables 45 and 46, printed pages 75-77).
//!
//! These tests build descriptors from the guide's bit ranges with their own
//! helpers, flip every bit of a valid descriptor one at a time, and state the
//! expected outcome for each bit from the guide, not from the implementation.

use xenora_ref::texture_sampling::{
    Rgba8TextureDesc, SamplingError, lod_to_mip, parse_resource_descriptor, sample_rgba8_2d,
    validate_sampler_descriptor,
};

fn put(packed: &mut u128, lo: u32, hi: u32, value: u128) {
    let mask = ((1u128 << (hi - lo + 1)) - 1) << lo;
    *packed = (*packed & !mask) | ((value << lo) & mask);
}

fn split(packed: u128) -> [u32; 4] {
    [0, 1, 2, 3].map(|word| (packed >> (32 * word)) as u32)
}

/// A valid 4x4, two-level, linear RGBA8 unorm 2D T# at byte offset 256.
fn valid_resource() -> u128 {
    let mut t = 0u128;
    put(&mut t, 0, 39, 1); // base address bits 47:8 -> byte 256
    put(&mut t, 52, 60, 56); // 8_8_8_8_UNORM
    put(&mut t, 62, 77, 3); // width - 1
    put(&mut t, 78, 93, 3); // height - 1
    put(&mut t, 95, 95, 1); // resource level: set to 1
    put(&mut t, 96, 98, 4); // dst_sel_x = R
    put(&mut t, 99, 101, 5); // dst_sel_y = G
    put(&mut t, 102, 104, 6); // dst_sel_z = B
    put(&mut t, 105, 107, 7); // dst_sel_w = A
    put(&mut t, 112, 115, 1); // last level 1 (base level 0)
    put(&mut t, 124, 127, 9); // SQ_RSRC_IMG_2D
    t
}

fn resource(packed: u128) -> [u32; 8] {
    let low = split(packed);
    [low[0], low[1], low[2], low[3], 0, 0, 0, 0]
}

/// A valid S#: clamp-to-last-texel on x/y/z, point filters, point mip filter,
/// min LOD 0 and max LOD 1.0 (4.8 fixed point).
fn valid_sampler() -> u128 {
    let mut s = 0u128;
    put(&mut s, 0, 2, 2);
    put(&mut s, 3, 5, 2);
    put(&mut s, 6, 8, 2);
    put(&mut s, 44, 55, 0x100);
    put(&mut s, 90, 91, 1);
    s
}

#[test]
fn the_valid_descriptors_are_accepted() {
    let (desc, base) = parse_resource_descriptor(resource(valid_resource())).unwrap();
    assert_eq!(base, 256);
    assert_eq!(desc.mips, 2);
    validate_sampler_descriptor(split(valid_sampler()), 1).unwrap();
}

#[test]
fn every_resource_bit_has_the_documented_outcome() {
    use SamplingError::*;
    for bit in 0..128u32 {
        let words = resource(valid_resource() ^ (1u128 << bit));
        let result = parse_resource_descriptor(words);
        match bit {
            // Base address: still valid, at the new 256-byte unit.
            0..=39 => {
                let (_, base) = result.unwrap_or_else(|e| panic!("bit {bit}: {e:?}"));
                assert_eq!(base, ((1u128 ^ (1u128 << bit)) << 8) as usize, "bit {bit}");
            }
            40..=51 => assert_eq!(result.unwrap_err(), InvalidResourceMinLod, "bit {bit}"),
            52..=60 => assert_eq!(result.unwrap_err(), InvalidResourceFormat, "bit {bit}"),
            61 | 94 | 121..=123 => assert_eq!(
                result.unwrap_err(),
                ResourceUnsupportedBits { mask: 1u128 << bit },
                "bit {bit}"
            ),
            // Width-1 and height-1: the low 14 bits give another legal extent.
            62..=75 | 78..=91 => {
                result.unwrap_or_else(|e| panic!("bit {bit}: {e:?}"));
            }
            // The two MSBs of width-1 and height-1 must be zero.
            76 | 77 | 92 | 93 => {
                assert_eq!(result.unwrap_err(), InvalidResourceExtent, "bit {bit}")
            }
            95..=107 => assert_eq!(result.unwrap_err(), InvalidResourceLevel, "bit {bit}"),
            // Base level must be 0; a base above the last level is a range error.
            108..=111 => assert!(
                result.is_err(),
                "bit {bit}: base level {}",
                1 << (bit - 108)
            ),
            // Last level 1 -> 0 is a single legal level.
            112 => assert_eq!(result.unwrap().0.mips, 1, "bit {bit}"),
            // Last level 3, 5 or 9 needs more levels than a 4x4 texture has.
            113..=115 => assert!(result.is_err(), "bit {bit}"),
            116..=120 => assert_eq!(result.unwrap_err(), UnsupportedSwizzleMode, "bit {bit}"),
            124..=127 => assert_eq!(result.unwrap_err(), InvalidResourceType, "bit {bit}"),
            _ => unreachable!(),
        }
    }
}

#[test]
fn every_extension_word_bit_is_refused() {
    for word in 4..8 {
        for bit in 0..32 {
            let mut words = resource(valid_resource());
            words[word] = 1 << bit;
            assert_eq!(
                parse_resource_descriptor(words).unwrap_err(),
                SamplingError::ResourceExtensionNonzero,
                "word {word} bit {bit}"
            );
        }
    }
}

#[test]
fn every_sampler_bit_has_the_documented_outcome() {
    use SamplingError::*;
    for bit in 0..128u32 {
        let result = validate_sampler_descriptor(split(valid_sampler() ^ (1u128 << bit)), 1);
        match bit {
            0..=8 | 15 => assert_eq!(result.unwrap_err(), SamplerAddressing, "bit {bit}"),
            32..=43 => assert_eq!(result.unwrap_err(), SamplerLodRange, "bit {bit}"),
            // Max LOD 0x100: clearing bit 8 of the field drops it to 0 < 1.0.
            52 => assert_eq!(result.unwrap_err(), SamplerLodRange, "bit {bit}"),
            // Any other max LOD bit only raises the maximum above 1.0.
            44..=55 => result.unwrap_or_else(|e| panic!("bit {bit}: {e:?}")),
            84..=89 => assert_eq!(result.unwrap_err(), SamplerFilter, "bit {bit}"),
            // Mip filter 1 -> 0 (none) or 3: only point (1) selects a level.
            90 | 91 => assert_eq!(result.unwrap_err(), SamplerMipFilter, "bit {bit}"),
            // Everything else, including force degamma (20), trunc coord (27),
            // LOD bias (64-77) and LOD bias sec (78-83), must be zero.
            _ => assert_eq!(
                result.unwrap_err(),
                SamplerUnsupportedBits { mask: 1u128 << bit },
                "bit {bit}"
            ),
        }
    }
}

#[test]
fn the_named_hardware_affecting_sampler_bits_are_refused() {
    for (name, bit) in [("force degamma", 20), ("trunc coord", 27), ("bit 19", 19)] {
        assert_eq!(
            validate_sampler_descriptor(split(valid_sampler() | (1u128 << bit)), 1).unwrap_err(),
            SamplingError::SamplerUnsupportedBits { mask: 1u128 << bit },
            "{name}"
        );
    }
    for lo in [64u32, 76, 78] {
        let mut s = valid_sampler();
        put(&mut s, lo, lo, 1);
        assert!(
            matches!(
                validate_sampler_descriptor(split(s), 1).unwrap_err(),
                SamplingError::SamplerUnsupportedBits { .. }
            ),
            "LOD bias bit {lo}"
        );
    }
}

#[test]
fn max_lod_must_cover_the_last_level() {
    let mut s = valid_sampler();
    put(&mut s, 44, 55, 0x1ff);
    assert_eq!(
        validate_sampler_descriptor(split(s), 2).unwrap_err(),
        SamplingError::SamplerLodRange
    );
    put(&mut s, 44, 55, 0x200);
    validate_sampler_descriptor(split(s), 2).unwrap();
}

#[test]
fn explicit_lod_names_an_exact_level_or_is_refused() {
    let desc = Rgba8TextureDesc::new_packed(4, 4, 2).unwrap();
    for (lod, mip) in [(0.0f32, 0), (-0.0, 0), (1.0, 1)] {
        assert_eq!(lod_to_mip(lod.to_bits(), &desc).unwrap(), mip, "lod {lod}");
    }
    let just_above_one = f32::from_bits(1.0f32.to_bits() + 1);
    for lod in [0.5f32, just_above_one, 2.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            lod_to_mip(lod.to_bits(), &desc).unwrap_err(),
            SamplingError::InvalidLod,
            "lod {lod}"
        );
    }
}

/// Texel byte = (y << 4) | (x << 2) | channel on a 4x4 single-level texture.
#[test]
fn point_sampling_picks_the_texel_at_each_edge() {
    let mut data = vec![0u8; 64];
    for y in 0..4 {
        for x in 0..4 {
            for channel in 0..4 {
                data[(y * 4 + x) * 4 + channel] = ((y << 4) | (x << 2) | channel) as u8;
            }
        }
    }
    let desc = Rgba8TextureDesc::new_packed(4, 4, 1).unwrap();
    let below_half = f32::from_bits(0.5f32.to_bits() - 1);
    let below_one = f32::from_bits(1.0f32.to_bits() - 1);
    // (u, expected column): floor(u * 4) clamped to [0, 3].
    for (u, column) in [
        (-0.0f32, 0u8),
        (-1.0e-30, 0),
        (0.0, 0),
        (0.25, 1),
        (below_half, 1),
        (0.5, 2),
        (below_one, 3),
        (1.0, 3),
        (1.0e30, 3),
    ] {
        let bytes = sample_rgba8_2d(&data, &desc, 0, u, 0.0).unwrap();
        assert_eq!(bytes, [0, 1, 2, 3].map(|c| (column << 2) | c), "u {u}");
        let bytes = sample_rgba8_2d(&data, &desc, 0, 0.0, u).unwrap();
        assert_eq!(bytes, [0, 1, 2, 3].map(|c| (column << 4) | c), "v {u}");
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 0, bad, 0.0).unwrap_err(),
            SamplingError::NonFiniteCoordinates
        );
    }
}
