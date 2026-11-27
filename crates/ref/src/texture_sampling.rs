// SPDX-License-Identifier: GPL-3.0-or-later

//! Independent pure-Rust CPU reference for deterministic RGBA8 2D texture sampling.
//!
//! # Coordinate Mapping and Filtering
//!
//! Sampling operates on normalized `(u, v)` coordinates using nearest-neighbor filtering
//! and clamp-to-edge addressing:
//!
//! 1. Non-finite coordinates (`NaN`, `+Inf`, `-Inf`) are rejected with [`SamplingError::NonFiniteCoordinates`].
//! 2. Coordinates are clamped to the unit interval:
//!    `u_clamped = u.clamp(0.0, 1.0)`
//!    `v_clamped = v.clamp(0.0, 1.0)`
//! 3. For a mip level with dimensions `(W, H)`:
//!    `x = min(floor(u_clamped * W), W - 1)`
//!    `y = min(floor(v_clamped * H), H - 1)`
//!    At exact texel boundaries where `u * W` is an integer, standard IEEE 754 `floor`
//!    deterministically assigns the boundary to the right/lower texel (for example, with `W = 2`,
//!    `u = 0.5` maps to `x = 1`), while `u = 1.0` maps to `x = 1` via edge clamping.

use std::fmt;

/// Bytes per RGBA8 texel.
pub const RGBA8_TEXEL_BYTES: usize = 4;

/// Errors produced during CPU texture sampling reference execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingError {
    /// Texture width or height is zero or exceeds limits.
    InvalidDimensions,
    /// Mip count is zero or exceeds maximum possible for dimensions.
    InvalidMipCount,
    /// Specified mip level index is out of bounds.
    MipOutOfBounds,
    /// Row pitch is smaller than tight row bytes (`width * 4`) or unaligned.
    InvalidRowPitch,
    /// Buffer offset causes an overflow or unaligned read.
    InvalidOffset,
    /// Underlying data buffer is smaller than required by descriptor layout.
    BufferTooSmall,
    /// Input coordinates contained NaN or infinity.
    NonFiniteCoordinates,
    /// Resource descriptor type is not the bounded 2D image.
    InvalidResourceType,
    /// Resource descriptor data format is not RGBA8 unorm.
    InvalidResourceFormat,
    /// Resource descriptor level, channel select or base level is unsupported.
    InvalidResourceLevel,
    /// Resource descriptor width-1 or height-1 sets a reserved high bit.
    InvalidResourceExtent,
    /// Resource descriptor swizzle/tiling mode is not linear (zero).
    UnsupportedSwizzleMode,
    /// Resource descriptor minimum LOD is nonzero.
    InvalidResourceMinLod,
    /// Resource descriptor last level is below its base level.
    InvalidResourceLevelRange,
    /// Resource descriptor extension words (the four SGPRs past the low four)
    /// are not all zero.
    ResourceExtensionNonzero,
    /// Resource descriptor sets bits outside the supported set. The mask holds
    /// exactly the set bits that are not supported.
    ResourceUnsupportedBits { mask: u128 },
    /// Sampler addressing is not clamp-to-last-texel on x, y and z with
    /// normalized coordinates.
    SamplerAddressing,
    /// Sampler xy or z filtering is not point filtering.
    SamplerFilter,
    /// Sampler mip filtering is not point filtering.
    SamplerMipFilter,
    /// Sampler minimum LOD is nonzero or its maximum LOD is below the last
    /// level of the resource.
    SamplerLodRange,
    /// Sampler descriptor sets bits outside the supported set. The mask holds
    /// exactly the set bits that are not supported.
    SamplerUnsupportedBits { mask: u128 },
    /// Explicit level of detail is not a representable integer mip level.
    InvalidLod,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "invalid texture dimensions"),
            Self::InvalidMipCount => write!(f, "invalid mip count"),
            Self::MipOutOfBounds => write!(f, "selected mip index is out of bounds"),
            Self::InvalidRowPitch => write!(f, "invalid row pitch for mip level"),
            Self::InvalidOffset => write!(f, "invalid buffer offset"),
            Self::BufferTooSmall => write!(f, "buffer is too small for texture layout"),
            Self::NonFiniteCoordinates => write!(f, "non-finite texture coordinates"),
            Self::InvalidResourceType => write!(f, "resource descriptor is not a 2D image"),
            Self::InvalidResourceFormat => write!(f, "resource descriptor is not RGBA8 unorm"),
            Self::InvalidResourceLevel => write!(
                f,
                "unsupported resource level, channel select or base level"
            ),
            Self::InvalidResourceExtent => {
                write!(f, "resource width or height sets a reserved high bit")
            }
            Self::UnsupportedSwizzleMode => write!(f, "resource swizzle mode is not linear"),
            Self::InvalidResourceMinLod => write!(f, "resource minimum LOD is nonzero"),
            Self::InvalidResourceLevelRange => write!(f, "resource last level is below base level"),
            Self::ResourceExtensionNonzero => {
                write!(f, "nonzero 256-bit resource extension words")
            }
            Self::ResourceUnsupportedBits { mask } => {
                write!(f, "resource descriptor sets unsupported bits 0x{mask:x}")
            }
            Self::SamplerAddressing => write!(
                f,
                "sampler addressing is not clamp-to-last-texel with normalized coordinates"
            ),
            Self::SamplerFilter => write!(f, "sampler filtering is not point filtering"),
            Self::SamplerMipFilter => write!(f, "sampler mip filtering is not point filtering"),
            Self::SamplerLodRange => {
                write!(
                    f,
                    "sampler LOD range does not cover the resource mip levels"
                )
            }
            Self::SamplerUnsupportedBits { mask } => {
                write!(f, "sampler sets unsupported bits 0x{mask:x}")
            }
            Self::InvalidLod => write!(f, "explicit LOD is not a representable integer mip level"),
        }
    }
}

impl std::error::Error for SamplingError {}

/// Mip-level layout specification for 2D RGBA8 sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipLayout {
    pub width: usize,
    pub height: usize,
    pub offset: usize,
    pub row_pitch: usize,
}

/// Description of a 2D RGBA8 texture with an explicit mip chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8TextureDesc {
    pub width: usize,
    pub height: usize,
    pub mips: u32,
    pub mip_layouts: Vec<MipLayout>,
}

impl Rgba8TextureDesc {
    /// Create a descriptor with tightly packed row-major mip levels consecutively arranged.
    pub fn new_packed(width: usize, height: usize, mips: u32) -> Result<Self, SamplingError> {
        let max_mips = max_mip_levels(width, height)?;
        if mips == 0 || mips > max_mips {
            return Err(SamplingError::InvalidMipCount);
        }

        let mut mip_layouts = Vec::with_capacity(mips as usize);
        let mut current_offset = 0usize;

        for mip in 0..mips {
            let mip_w = (width >> mip).max(1);
            let mip_h = (height >> mip).max(1);
            let row_pitch = mip_w
                .checked_mul(RGBA8_TEXEL_BYTES)
                .ok_or(SamplingError::InvalidDimensions)?;
            let level_bytes = row_pitch
                .checked_mul(mip_h)
                .ok_or(SamplingError::InvalidDimensions)?;

            mip_layouts.push(MipLayout {
                width: mip_w,
                height: mip_h,
                offset: current_offset,
                row_pitch,
            });

            current_offset = current_offset
                .checked_add(level_bytes)
                .ok_or(SamplingError::InvalidDimensions)?;
        }

        Ok(Self {
            width,
            height,
            mips,
            mip_layouts,
        })
    }

    /// Validates internal consistency and returns the total required byte size.
    pub fn required_size(&self) -> Result<usize, SamplingError> {
        let max_mips = max_mip_levels(self.width, self.height)?;
        if self.mips == 0 || self.mips > max_mips || self.mip_layouts.len() != self.mips as usize {
            return Err(SamplingError::InvalidMipCount);
        }

        let mut total_end = 0usize;
        for (i, layout) in self.mip_layouts.iter().enumerate() {
            let expected_w = (self.width >> i).max(1);
            let expected_h = (self.height >> i).max(1);
            if layout.width != expected_w || layout.height != expected_h {
                return Err(SamplingError::InvalidDimensions);
            }

            let tight_pitch = layout
                .width
                .checked_mul(RGBA8_TEXEL_BYTES)
                .ok_or(SamplingError::InvalidDimensions)?;
            if layout.row_pitch < tight_pitch {
                return Err(SamplingError::InvalidRowPitch);
            }

            let level_size = layout
                .row_pitch
                .checked_mul(layout.height)
                .ok_or(SamplingError::InvalidDimensions)?;
            let level_end = layout
                .offset
                .checked_add(level_size)
                .ok_or(SamplingError::InvalidOffset)?;

            total_end = total_end.max(level_end);
        }

        Ok(total_end)
    }
}

/// Calculate maximum allowable mip levels for a given width and height.
///
/// For arbitrary positive dimensions, the maximum mip count is:
/// `floor(log2(max(width, height))) + 1`
pub fn max_mip_levels(width: usize, height: usize) -> Result<u32, SamplingError> {
    if width == 0 || height == 0 {
        return Err(SamplingError::InvalidDimensions);
    }
    let max_dim = width.max(height);
    let levels = usize::BITS - max_dim.leading_zeros();
    Ok(levels)
}

/// Sample an RGBA8 texel deterministically from raw texture bytes using nearest-neighbor
/// filtering, clamp-to-edge addressing and explicit mip selection.
pub fn sample_rgba8_2d(
    data: &[u8],
    desc: &Rgba8TextureDesc,
    mip: u32,
    u: f32,
    v: f32,
) -> Result<[u8; 4], SamplingError> {
    if !u.is_finite() || !v.is_finite() {
        return Err(SamplingError::NonFiniteCoordinates);
    }

    let required = desc.required_size()?;
    if data.len() < required {
        return Err(SamplingError::BufferTooSmall);
    }

    if mip >= desc.mips {
        return Err(SamplingError::MipOutOfBounds);
    }

    let layout = desc
        .mip_layouts
        .get(mip as usize)
        .ok_or(SamplingError::MipOutOfBounds)?;

    // Clamp to edge [0.0, 1.0]
    let u_clamped = u.clamp(0.0, 1.0);
    let v_clamped = v.clamp(0.0, 1.0);

    // Nearest-neighbor discrete texel coordinates:
    // floor(coord * dim) clamped to [0, dim - 1]
    let tex_x = ((u_clamped as f64 * layout.width as f64).floor() as usize).min(layout.width - 1);
    let tex_y = ((v_clamped as f64 * layout.height as f64).floor() as usize).min(layout.height - 1);

    let row_start = layout
        .offset
        .checked_add(
            tex_y
                .checked_mul(layout.row_pitch)
                .ok_or(SamplingError::InvalidOffset)?,
        )
        .ok_or(SamplingError::InvalidOffset)?;

    let texel_offset = row_start
        .checked_add(
            tex_x
                .checked_mul(RGBA8_TEXEL_BYTES)
                .ok_or(SamplingError::InvalidOffset)?,
        )
        .ok_or(SamplingError::InvalidOffset)?;

    let bytes = data
        .get(texel_offset..texel_offset + RGBA8_TEXEL_BYTES)
        .ok_or(SamplingError::BufferTooSmall)?;

    Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// The RDNA 2 image resource descriptor (T#) type value for a 2D texture.
pub const T_TYPE_2D: u32 = 9;
/// The RDNA 2 data format value for `8_8_8_8_UNORM` (RGBA8 unorm).
pub const T_FORMAT_RGBA8_UNORM: u32 = 56;
/// The sampler clamp mode for clamp-to-last-texel (guide Table 46).
pub const S_CLAMP_LAST_TEXEL: u32 = 2;

fn pack_words(words: [u32; 4]) -> u128 {
    let mut value = 0u128;
    for (index, word) in words.into_iter().enumerate() {
        value |= u128::from(word) << (32 * index);
    }
    value
}

fn field(packed: u128, lo: u32, hi: u32) -> u32 {
    ((packed >> lo) & ((1u128 << (hi - lo + 1)) - 1)) as u32
}

/// The same extraction as `field` for a field wider than 32 bits.
fn wide_field(packed: u128, lo: u32, hi: u32) -> u128 {
    (packed >> lo) & ((1u128 << (hi - lo + 1)) - 1)
}

/// The resource descriptor bits the bounded scope does not interpret: the
/// reserved bit 61, the reserved bit 94 and the BC swizzle field 123:121.
/// The result holds exactly the set bits among those positions.
fn resource_unsupported_bits(packed: u128) -> u128 {
    let mut unsupported = 0u128;
    for (lo, hi) in [(61u32, 61u32), (94, 94), (121, 123)] {
        unsupported |= u128::from(field(packed, lo, hi)) << lo;
    }
    unsupported
}

/// The sampler descriptor bits the bounded scope interprets: the three clamp
/// modes 8:0, the unnormalized-coordinate flag 15, the LOD range 55:32 and
/// the filter fields 91:84. Every other bit must be zero.
fn sampler_supported_mask() -> u128 {
    let clamp = (1u128 << 9) - 1; // 8:0
    let unnormalized = 1u128 << 15; // 15
    let lod_range = ((1u128 << 24) - 1) << 32; // 55:32
    let filters = ((1u128 << 8) - 1) << 84; // 91:84
    clamp | unnormalized | lod_range | filters
}

/// Parse and validate a 128-bit image resource descriptor (T#, guide Table 45,
/// printed page 75) held in four consecutive SGPRs, low word first.
///
/// Returns the packed mip-chain description and the base byte offset of the
/// texture data in the synthetic input buffer. The base-address field (39:0)
/// represents address bits 47:8, so the byte offset is `field << 8` and is
/// therefore always 256-byte aligned. The bounded scope accepts only a linear
/// (unswizzled) 2D RGBA8 unorm resource with an identity RGBA channel select,
/// a zero minimum LOD and base level zero; the 256-bit extension words (a
/// further four SGPRs) must be zero.
///
/// # Refused bits
///
/// Every bit of the descriptor is either checked by this function or required
/// to be zero. The bits that carry no meaning for the bounded scope are the
/// reserved bit 61, the reserved bit 94 and the BC swizzle bits 123:121; any
/// of them set is refused with [`SamplingError::ResourceUnsupportedBits`] and
/// a mask of exactly the set bits. A nonzero extension word is refused with
/// [`SamplingError::ResourceExtensionNonzero`].
pub fn parse_resource_descriptor(
    words: [u32; 8],
) -> Result<(Rgba8TextureDesc, usize), SamplingError> {
    // The four extension SGPRs (words 4..8) hold no information for a plain
    // 2D texture, so every one of their bits must be zero (guide Table 45,
    // printed page 75).
    if words[4..].iter().any(|&word| word != 0) {
        return Err(SamplingError::ResourceExtensionNonzero);
    }
    let packed = pack_words([words[0], words[1], words[2], words[3]]);
    // 127:124 type, 120:116 swizzle mode, 60:52 format, 51:40 minimum LOD,
    // 95 resource level, 96-107 channel select, 108-111 base level,
    // 112-115 last level, 62-77 width-1, 78-93 height-1, 39:0 base address
    // (address bits 47:8, a byte offset into the synthetic input buffer).
    if field(packed, 124, 127) != T_TYPE_2D {
        return Err(SamplingError::InvalidResourceType);
    }
    if field(packed, 52, 60) != T_FORMAT_RGBA8_UNORM {
        return Err(SamplingError::InvalidResourceFormat);
    }
    // The 2 most significant bits of width-1 and height-1 must be zero
    // (guide Table 45, printed page 75).
    let width1 = field(packed, 62, 77);
    let height1 = field(packed, 78, 93);
    if width1 & 0xC000 != 0 || height1 & 0xC000 != 0 {
        return Err(SamplingError::InvalidResourceExtent);
    }
    if field(packed, 116, 120) != 0 {
        return Err(SamplingError::UnsupportedSwizzleMode);
    }
    if field(packed, 40, 51) != 0 {
        return Err(SamplingError::InvalidResourceMinLod);
    }
    let base_level = field(packed, 108, 111);
    let last_level = field(packed, 112, 115);
    if last_level < base_level {
        return Err(SamplingError::InvalidResourceLevelRange);
    }
    let levels_ok = field(packed, 95, 95) == 1
        && [
            field(packed, 96, 98),
            field(packed, 99, 101),
            field(packed, 102, 104),
            field(packed, 105, 107),
        ] == [4, 5, 6, 7]
        && base_level == 0;
    if !levels_ok {
        return Err(SamplingError::InvalidResourceLevel);
    }
    // The bits the bounded scope does not interpret must be zero: the reserved
    // bit 61, the reserved bit 94 and the BC swizzle field 123:121.
    let unsupported = resource_unsupported_bits(packed);
    if unsupported != 0 {
        return Err(SamplingError::ResourceUnsupportedBits { mask: unsupported });
    }
    let width = usize::try_from(width1 + 1).map_err(|_| SamplingError::InvalidDimensions)?;
    let height = usize::try_from(height1 + 1).map_err(|_| SamplingError::InvalidDimensions)?;
    let mips = last_level - base_level + 1;
    // The base-address field is 40 bits wide, so it is read with a wide
    // extraction: a narrow one would drop the eight high address bits.
    let base = usize::try_from(wide_field(packed, 0, 39))
        .ok()
        .and_then(|address| address.checked_mul(256))
        .ok_or(SamplingError::InvalidOffset)?;
    let desc = Rgba8TextureDesc::new_packed(width, height, mips)?;
    Ok((desc, base))
}

/// Validate a 128-bit sampler descriptor (S#, guide Table 46, printed page 76)
/// held in four consecutive SGPRs, low word first, for point filtering,
/// clamp-to-last-texel addressing with normalized coordinates and an explicit
/// mip level.
///
/// `last_level` is the resource descriptor's last level, used to check the
/// sampler's 4.8 fixed-point maximum LOD. A mip filter of none (0) is refused
/// because it disables mipmapping and could not select the explicit level.
///
/// # Refused bits
///
/// Every bit of the descriptor is either checked by this function or required
/// to be zero. The bits the bounded scope does not interpret are the LOD bias
/// 75:64, the LOD bias high bits 77:76 and the LOD bias sec 83:78; the force
/// degamma bit 20; the trunc coord bit 27; bit 19; the anisotropy, depth
/// compare, filter mode, performance and border colour fields; and the
/// reserved bits. Any of them set is refused with
/// [`SamplingError::SamplerUnsupportedBits`] and a mask of exactly the set
/// bits. The remaining refusals are separate variants: the clamp modes and
/// the unnormalized-coordinate flag give [`SamplingError::SamplerAddressing`],
/// the xy and z filters give [`SamplingError::SamplerFilter`], the mip filter
/// gives [`SamplingError::SamplerMipFilter`] and the LOD range gives
/// [`SamplingError::SamplerLodRange`].
pub fn validate_sampler_descriptor(words: [u32; 4], last_level: u32) -> Result<(), SamplingError> {
    let packed = pack_words(words);
    // 2:0 / 5:3 / 8:6 clamp x/y/z are ClampLastTexel (2) and z matches x;
    // bit 15 forces unnormalized coordinates (0).
    if !(field(packed, 0, 2) == S_CLAMP_LAST_TEXEL
        && field(packed, 3, 5) == S_CLAMP_LAST_TEXEL
        && field(packed, 6, 8) == field(packed, 0, 2)
        && field(packed, 15, 15) == 0)
    {
        return Err(SamplingError::SamplerAddressing);
    }
    // 85:84 and 87:86 select point xy filtering and 89:88 point z filtering.
    if !(field(packed, 84, 85) == 0 && field(packed, 86, 87) == 0 && field(packed, 88, 89) == 0) {
        return Err(SamplingError::SamplerFilter);
    }
    // 91:90 must be point mip filtering (1) so an explicit LOD selects a level.
    if field(packed, 90, 91) != 1 {
        return Err(SamplingError::SamplerMipFilter);
    }
    // 43:32 minimum LOD is zero; 55:44 maximum LOD is at least the last level
    // in 4.8 fixed point.
    if !(field(packed, 32, 43) == 0 && field(packed, 44, 55) >= last_level << 8) {
        return Err(SamplingError::SamplerLodRange);
    }
    // Every bit the checks above do not interpret must be zero.
    let unsupported = packed & !sampler_supported_mask();
    if unsupported != 0 {
        return Err(SamplingError::SamplerUnsupportedBits { mask: unsupported });
    }
    Ok(())
}

/// Convert an explicit level of detail to the integer mip level it names.
///
/// Point mip filtering selects a single mip level, so the LOD must be an exact
/// integer within `[0, last_level]`.
pub fn lod_to_mip(lod_bits: u32, desc: &Rgba8TextureDesc) -> Result<u32, SamplingError> {
    let lod = f32::from_bits(lod_bits);
    if !lod.is_finite() || lod < 0.0 || lod.fract() != 0.0 {
        return Err(SamplingError::InvalidLod);
    }
    let mip = lod as u32;
    if mip >= desc.mips {
        return Err(SamplingError::InvalidLod);
    }
    Ok(mip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_four_distinct_texels_in_2x2() {
        // 2x2 RGBA8 image:
        // Top-left: [255, 0, 0, 255] (Red)
        // Top-right: [0, 255, 0, 255] (Green)
        // Bottom-left: [0, 0, 255, 255] (Blue)
        // Bottom-right: [255, 255, 0, 255] (Yellow)
        let pixels: [u8; 16] = [
            255, 0, 0, 255, // (0, 0)
            0, 255, 0, 255, // (1, 0)
            0, 0, 255, 255, // (0, 1)
            255, 255, 0, 255, // (1, 1)
        ];
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();

        // Sample centers of each texel:
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.25, 0.25).unwrap(),
            [255, 0, 0, 255]
        );
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.75, 0.25).unwrap(),
            [0, 255, 0, 255]
        );
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.25, 0.75).unwrap(),
            [0, 0, 255, 255]
        );
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.75, 0.75).unwrap(),
            [255, 255, 0, 255]
        );
    }

    #[test]
    fn test_normalized_uv_sampling() {
        let pixels: [u8; 16] = [
            10, 10, 10, 255, // (0, 0)
            20, 20, 20, 255, // (1, 0)
            30, 30, 30, 255, // (0, 1)
            40, 40, 40, 255, // (1, 1)
        ];
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();

        // Within (0, 0) region [0.0, 0.5)
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.1, 0.1).unwrap(),
            [10, 10, 10, 255]
        );
        // Within (1, 1) region [0.5, 1.0]
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.9, 0.9).unwrap(),
            [40, 40, 40, 255]
        );
    }

    #[test]
    fn test_clamp_to_edge() {
        let pixels: [u8; 16] = [
            1, 1, 1, 255, 2, 2, 2, 255, // row 0
            3, 3, 3, 255, 4, 4, 4, 255, // row 1
        ];
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();

        // Negative coordinates clamp to 0.0 -> (0, 0)
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, -5.0, -10.0).unwrap(),
            [1, 1, 1, 255]
        );
        // Coordinates > 1.0 clamp to 1.0 -> (1, 1)
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 10.0, 50.0).unwrap(),
            [4, 4, 4, 255]
        );
        // Mixed clamping
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, -1.0, 2.0).unwrap(),
            [3, 3, 3, 255]
        );
    }

    #[test]
    fn test_boundary_coordinates() {
        let pixels: [u8; 16] = [
            11, 11, 11, 255, 22, 22, 22, 255, 33, 33, 33, 255, 44, 44, 44, 255,
        ];
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();

        // Exactly 0.0 -> texel 0
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.0, 0.0).unwrap(),
            [11, 11, 11, 255]
        );
        // Exactly 0.5 -> 0.5 * 2 = 1.0 -> texel 1
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.5, 0.0).unwrap(),
            [22, 22, 22, 255]
        );
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 0.0, 0.5).unwrap(),
            [33, 33, 33, 255]
        );
        // Exactly 1.0 -> clamped to texel 1
        assert_eq!(
            sample_rgba8_2d(&pixels, &desc, 0, 1.0, 1.0).unwrap(),
            [44, 44, 44, 255]
        );
    }

    #[test]
    fn test_multiple_mip_levels() {
        // Mip 0: 2x2 (16 bytes), Mip 1: 1x1 (4 bytes)
        let mut data = vec![0u8; 20];
        data[0..16].copy_from_slice(&[1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255]);
        data[16..20].copy_from_slice(&[99, 99, 99, 255]);

        let desc = Rgba8TextureDesc::new_packed(2, 2, 2).unwrap();
        assert_eq!(desc.mip_layouts.len(), 2);
        assert_eq!(desc.mip_layouts[0].offset, 0);
        assert_eq!(desc.mip_layouts[1].offset, 16);

        // Mip 0 samples the 2x2 texels
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 0, 0.75, 0.75).unwrap(),
            [4, 0, 0, 255]
        );
        // Mip 1 samples the single 1x1 texel across all UVs
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 1, 0.1, 0.1).unwrap(),
            [99, 99, 99, 255]
        );
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 1, 0.9, 0.9).unwrap(),
            [99, 99, 99, 255]
        );
    }

    #[test]
    fn test_invalid_dimensions() {
        assert_eq!(
            Rgba8TextureDesc::new_packed(0, 4, 1).unwrap_err(),
            SamplingError::InvalidDimensions
        );
        assert_eq!(
            Rgba8TextureDesc::new_packed(4, 0, 1).unwrap_err(),
            SamplingError::InvalidDimensions
        );
        assert_eq!(
            Rgba8TextureDesc::new_packed(4, 4, 0).unwrap_err(),
            SamplingError::InvalidMipCount
        );
        // 4x4 has at most 3 mips (4x4, 2x2, 1x1)
        assert_eq!(
            Rgba8TextureDesc::new_packed(4, 4, 4).unwrap_err(),
            SamplingError::InvalidMipCount
        );
    }

    #[test]
    fn test_non_power_of_two_max_mips() {
        // For 3x3, max(3, 3) = 3. floor(log2(3)) + 1 = 1 + 1 = 2 mips (3x3 and 1x1).
        assert_eq!(max_mip_levels(3, 3).unwrap(), 2);
        // Valid 2-mip descriptor
        let desc = Rgba8TextureDesc::new_packed(3, 3, 2).unwrap();
        assert_eq!(desc.mip_layouts.len(), 2);
        assert_eq!(desc.mip_layouts[0].width, 3);
        assert_eq!(desc.mip_layouts[0].height, 3);
        assert_eq!(desc.mip_layouts[1].width, 1);
        assert_eq!(desc.mip_layouts[1].height, 1);
        // 3 mips is invalid for 3x3
        assert_eq!(
            Rgba8TextureDesc::new_packed(3, 3, 3).unwrap_err(),
            SamplingError::InvalidMipCount
        );
    }

    #[test]
    fn test_invalid_pitches() {
        let desc = Rgba8TextureDesc {
            width: 4,
            height: 4,
            mips: 1,
            mip_layouts: vec![MipLayout {
                width: 4,
                height: 4,
                offset: 0,
                row_pitch: 12, // Tight pitch is 4 * 4 = 16, so 12 is invalid!
            }],
        };
        let dummy = vec![0u8; 64];
        assert_eq!(
            sample_rgba8_2d(&dummy, &desc, 0, 0.5, 0.5).unwrap_err(),
            SamplingError::InvalidRowPitch
        );
    }

    #[test]
    fn test_truncated_data() {
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();
        // 2x2 requires 16 bytes; provide only 15
        let short_data = vec![0u8; 15];
        assert_eq!(
            sample_rgba8_2d(&short_data, &desc, 0, 0.0, 0.0).unwrap_err(),
            SamplingError::BufferTooSmall
        );
    }

    #[test]
    fn test_invalid_mip_indices() {
        let data = vec![0u8; 16];
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 1, 0.0, 0.0).unwrap_err(),
            SamplingError::MipOutOfBounds
        );
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 5, 0.0, 0.0).unwrap_err(),
            SamplingError::MipOutOfBounds
        );
    }

    #[test]
    fn test_non_finite_coordinates() {
        let data = vec![0u8; 16];
        let desc = Rgba8TextureDesc::new_packed(2, 2, 1).unwrap();

        assert_eq!(
            sample_rgba8_2d(&data, &desc, 0, f32::NAN, 0.0).unwrap_err(),
            SamplingError::NonFiniteCoordinates
        );
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 0, 0.0, f32::NAN).unwrap_err(),
            SamplingError::NonFiniteCoordinates
        );
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 0, f32::INFINITY, 0.5).unwrap_err(),
            SamplingError::NonFiniteCoordinates
        );
        assert_eq!(
            sample_rgba8_2d(&data, &desc, 0, 0.5, f32::NEG_INFINITY).unwrap_err(),
            SamplingError::NonFiniteCoordinates
        );
    }

    /// Build a valid T#: linear 2D RGBA8 unorm, identity channel select, base
    /// level zero, `mips` levels and a 40-bit base-address field `base`.
    fn resource_words(width: u32, height: u32, mips: u32, base: u32) -> [u32; 8] {
        let width1 = width - 1;
        let height1 = height - 1;
        [
            base,
            (T_FORMAT_RGBA8_UNORM << 20) | ((width1 & 3) << 30),
            ((width1 >> 2) & 0x3fff) | (height1 << 14) | (1 << 31),
            0x0fac | ((mips - 1) << 16) | (T_TYPE_2D << 28),
            0,
            0,
            0,
            0,
        ]
    }

    /// Build an S# with the given clamp modes, mip filter, LOD range (4.8 fixed
    /// point) and LOD bias.
    fn sampler_words(
        clamp_x: u32,
        clamp_y: u32,
        clamp_z: u32,
        mip_filter: u32,
        min_lod: u32,
        max_lod: u32,
        bias: u32,
    ) -> [u32; 4] {
        [
            clamp_x | (clamp_y << 3) | (clamp_z << 6),
            (min_lod & 0xfff) | ((max_lod & 0xfff) << 12),
            (bias & 0xfff) | (((bias >> 12) & 3) << 12) | (mip_filter << 26),
            0,
        ]
    }

    /// Hand-derived point-sampling results for a 4x4 mip 0 and a 2x2 mip 1
    /// whose texel byte is `(level << 6) | (y << 4) | (x << 2) | channel`.
    ///
    /// The texel of a case is `clamp(floor(u * width), 0, width - 1)` and
    /// `clamp(floor(v * height), 0, height - 1)` of the mip the LOD names, so:
    ///
    /// | u | v | lod | texel | bytes (R, G, B, A) |
    /// | --- | --- | --- | --- | --- |
    /// | 0.125 | 0.125 | 0 | (0, 0) | 0, 1, 2, 3 |
    /// | 0.5 | 0.5 | 0 | (2, 2) | 40, 41, 42, 43 |
    /// | -0.3 | 0.9 | 0 | (0, 3) | 48, 49, 50, 51 |
    /// | 1.7 | 1.2 | 0 | (3, 3) | 60, 61, 62, 63 |
    /// | 0.75 | 0.25 | 1 | (1, 0) | 68, 69, 70, 71 |
    /// | 0.25 | 0.75 | 1 | (0, 1) | 80, 81, 82, 83 |
    ///
    /// The last two cases name the 2x2 mip 1, whose extent is 2, so
    /// `0.75 * 2 = 1.5` floors to the second column and `0.25 * 2 = 0.5` floors
    /// to the first. The third and fourth cases clamp to the border texels.
    #[test]
    fn hand_derived_point_sampling_results() {
        let mut data = vec![0u8; 80];
        for level in 0..2u32 {
            let extent = 4usize >> level;
            for y in 0..extent {
                for x in 0..extent {
                    let offset = (level as usize) * 64 + (y * extent + x) * 4;
                    for channel in 0..4 {
                        data[offset + channel] =
                            ((level << 6) | ((y as u32) << 4) | ((x as u32) << 2) | channel as u32)
                                as u8;
                    }
                }
            }
        }
        let desc = Rgba8TextureDesc::new_packed(4, 4, 2).unwrap();
        // (u, v, mip, RGBA bytes, RGBA binary32 bits of byte / 255), worked
        // out by hand.
        type Case = (f32, f32, u32, [u8; 4], [u32; 4]);
        let cases: [Case; 6] = [
            (
                0.125,
                0.125,
                0,
                [0, 1, 2, 3],
                [0x0000_0000, 0x3b80_8081, 0x3c00_8081, 0x3c40_c0c1],
            ),
            (
                0.5,
                0.5,
                0,
                [40, 41, 42, 43],
                [0x3e20_a0a1, 0x3e24_a4a5, 0x3e28_a8a9, 0x3e2c_acad],
            ),
            (
                -0.3,
                0.9,
                0,
                [48, 49, 50, 51],
                [0x3e40_c0c1, 0x3e44_c4c5, 0x3e48_c8c9, 0x3e4c_cccd],
            ),
            (
                1.7,
                1.2,
                0,
                [60, 61, 62, 63],
                [0x3e70_f0f1, 0x3e74_f4f5, 0x3e78_f8f9, 0x3e7c_fcfd],
            ),
            (
                0.75,
                0.25,
                1,
                [68, 69, 70, 71],
                [0x3e88_8889, 0x3e8a_8a8b, 0x3e8c_8c8d, 0x3e8e_8e8f],
            ),
            (
                0.25,
                0.75,
                1,
                [80, 81, 82, 83],
                [0x3ea0_a0a1, 0x3ea2_a2a3, 0x3ea4_a4a5, 0x3ea6_a6a7],
            ),
        ];
        for (u, v, mip, bytes, bits) in cases {
            let texel = sample_rgba8_2d(&data, &desc, mip, u, v).expect("sample");
            assert_eq!(texel, bytes, "the texel of ({u}, {v}) at mip {mip}");
            for channel in 0..4 {
                assert_eq!(
                    (f32::from(texel[channel]) / 255.0).to_bits(),
                    bits[channel],
                    "the {channel} channel of ({u}, {v}) at mip {mip}"
                );
            }
        }
    }

    #[test]
    fn resource_descriptor_base_field_is_a_256_byte_unit() {
        let (desc, base) = parse_resource_descriptor(resource_words(4, 4, 2, 1)).unwrap();
        assert_eq!((desc.width, desc.height, desc.mips), (4, 4, 2));
        assert_eq!(base, 0x100);
        let (_, base) = parse_resource_descriptor(resource_words(4, 4, 2, 0)).unwrap();
        assert_eq!(base, 0);
        let (_, base) = parse_resource_descriptor(resource_words(4, 4, 2, 7)).unwrap();
        assert_eq!(base, 7 * 256);
    }

    #[test]
    fn resource_descriptor_rejects_width_or_height_high_bits() {
        let mut words = resource_words(4, 4, 2, 1);
        words[2] |= 1 << 12; // width-1 bit 14 is reserved and must be zero.
        assert_eq!(
            parse_resource_descriptor(words).unwrap_err(),
            SamplingError::InvalidResourceExtent
        );
        let mut words = resource_words(4, 4, 2, 1);
        words[2] |= 1 << 28; // height-1 bit 14 is reserved and must be zero.
        assert_eq!(
            parse_resource_descriptor(words).unwrap_err(),
            SamplingError::InvalidResourceExtent
        );
    }

    #[test]
    fn resource_descriptor_rejects_nonlinear_swizzle() {
        let mut words = resource_words(4, 4, 2, 1);
        words[3] |= 1 << 20; // SW mode 116:120 must be zero (linear).
        assert_eq!(
            parse_resource_descriptor(words).unwrap_err(),
            SamplingError::UnsupportedSwizzleMode
        );
    }

    #[test]
    fn resource_descriptor_rejects_nonzero_min_lod() {
        let mut words = resource_words(4, 4, 2, 1);
        words[1] |= 1 << 8; // minimimum LOD 51:40.
        assert_eq!(
            parse_resource_descriptor(words).unwrap_err(),
            SamplingError::InvalidResourceMinLod
        );
    }

    #[test]
    fn resource_descriptor_rejects_last_level_below_base_level() {
        let mut words = resource_words(4, 4, 2, 1);
        // Base level 1, last level 0: the range is inverted.
        words[3] = (words[3] & !(0xf << 16)) | (1 << 12);
        assert_eq!(
            parse_resource_descriptor(words).unwrap_err(),
            SamplingError::InvalidResourceLevelRange
        );
    }

    #[test]
    fn resource_descriptor_refuses_nonzero_extension_words() {
        for word in 4..8 {
            let mut words = resource_words(4, 4, 2, 1);
            words[word] = 1;
            assert_eq!(
                parse_resource_descriptor(words).unwrap_err(),
                SamplingError::ResourceExtensionNonzero,
                "word {word}"
            );
        }
    }

    #[test]
    fn resource_descriptor_refuses_the_bits_the_scope_does_not_interpret() {
        // The reserved bit 61, the reserved bit 94 and the BC swizzle field
        // 123:121 carry no meaning for the bounded scope.
        for bit in [61u32, 94, 121, 122, 123] {
            let mut words = resource_words(4, 4, 2, 1);
            words[(bit / 32) as usize] |= 1 << (bit % 32);
            assert_eq!(
                parse_resource_descriptor(words).unwrap_err(),
                SamplingError::ResourceUnsupportedBits { mask: 1u128 << bit },
                "bit {bit}"
            );
        }
    }

    #[test]
    fn sampler_descriptor_accepts_valid_point_clamp_state() {
        // 2 mips: the last level is 1, so the maximum LOD must be at least
        // 1.0 in 4.8 fixed point (0x100).
        validate_sampler_descriptor(sampler_words(2, 2, 2, 1, 0, 0x100, 0), 1).unwrap();
    }

    #[test]
    fn sampler_descriptor_refuses_mip_filter_none() {
        assert_eq!(
            validate_sampler_descriptor(sampler_words(2, 2, 2, 0, 0, 0x100, 0), 1).unwrap_err(),
            SamplingError::SamplerMipFilter
        );
    }

    #[test]
    fn sampler_descriptor_refuses_nonzero_z_filter() {
        let mut words = sampler_words(2, 2, 2, 1, 0, 0x100, 0);
        words[2] |= 1 << 24; // z filter 89:88 = point (0) only.
        assert_eq!(
            validate_sampler_descriptor(words, 1).unwrap_err(),
            SamplingError::SamplerFilter
        );
    }

    #[test]
    fn sampler_descriptor_refuses_clamp_z_different_from_clamp_x() {
        assert_eq!(
            validate_sampler_descriptor(sampler_words(2, 2, 0, 1, 0, 0x100, 0), 1).unwrap_err(),
            SamplingError::SamplerAddressing
        );
    }

    #[test]
    fn sampler_descriptor_refuses_nonzero_lod_bias() {
        assert_eq!(
            validate_sampler_descriptor(sampler_words(2, 2, 2, 1, 0, 0x100, 1), 1).unwrap_err(),
            SamplingError::SamplerUnsupportedBits { mask: 1u128 << 64 }
        );
        let mut words = sampler_words(2, 2, 2, 1, 0, 0x100, 0);
        words[2] |= 1 << 12; // LOD bias high bit (76:77).
        assert_eq!(
            validate_sampler_descriptor(words, 1).unwrap_err(),
            SamplingError::SamplerUnsupportedBits { mask: 1u128 << 76 }
        );
    }

    #[test]
    fn sampler_descriptor_refuses_nonzero_min_lod() {
        assert_eq!(
            validate_sampler_descriptor(sampler_words(2, 2, 2, 1, 1, 0x100, 0), 1).unwrap_err(),
            SamplingError::SamplerLodRange
        );
    }

    #[test]
    fn sampler_descriptor_refuses_max_lod_below_last_level() {
        // Last level 1 needs a maximum LOD of at least 0x100 (1.0).
        assert_eq!(
            validate_sampler_descriptor(sampler_words(2, 2, 2, 1, 0, 0x0ff, 0), 1).unwrap_err(),
            SamplingError::SamplerLodRange
        );
        validate_sampler_descriptor(sampler_words(2, 2, 2, 1, 0, 0x100, 0), 1).unwrap();
    }

    #[test]
    fn sampler_descriptor_refuses_the_bits_the_scope_does_not_interpret() {
        // Force degamma (20), trunc coord (27) and bit 19 change what the
        // hardware samples, so they must be refused rather than ignored.
        for bit in [19u32, 20, 27] {
            let mut words = sampler_words(2, 2, 2, 1, 0, 0x100, 0);
            words[(bit / 32) as usize] |= 1 << (bit % 32);
            assert_eq!(
                validate_sampler_descriptor(words, 1).unwrap_err(),
                SamplingError::SamplerUnsupportedBits { mask: 1u128 << bit },
                "bit {bit}"
            );
        }
        // The LOD bias sec field 83:78 is refused as well.
        let mut words = sampler_words(2, 2, 2, 1, 0, 0x100, 0);
        words[2] |= 1 << 14; // LOD bias sec 83:78.
        assert_eq!(
            validate_sampler_descriptor(words, 1).unwrap_err(),
            SamplingError::SamplerUnsupportedBits { mask: 1u128 << 78 }
        );
    }
}
