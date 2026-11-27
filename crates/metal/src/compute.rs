use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::fmt;
use std::mem::size_of_val;
use std::ptr::{NonNull, copy_nonoverlapping};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use objc2::Message;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLGPUFamily, MTLLibrary,
    MTLMathFloatingPointFunctions, MTLMathMode, MTLOrigin, MTLPixelFormat, MTLRegion,
    MTLResourceOptions, MTLSamplerState, MTLShaderValidation, MTLSharedEvent, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use xenora_texture::{
    AddressMode, Bc1Upload, BcAlphaFormat, BcAlphaUpload, BcChannelFormat, BcChannelUpload,
    FilterMode, TiledBc1,
};

#[path = "render.rs"]
mod render;

pub use render::{
    BlendDraw, BlendFactor, BlendOperation, BlendPass, BlendState, ColorWriteMask, DepthCompare,
    DepthDraw, DepthFill, DepthOutput, DepthPass, DepthState, IndexBuffer, IndexedDraw,
    IndexedPass,
};

const MAX_THREADS_PER_GROUP: usize = 64;
const MAX_GROUPS: usize = 64;
const MAX_GRID_WORDS: usize = MAX_GROUPS * MAX_THREADS_PER_GROUP;
const MAX_SOURCE_BYTES: usize = 64 * 1024;
const MAX_IN_FLIGHT: usize = 16;
const MAX_CACHED_PIPELINES: usize = 16;
const MAX_POOLED_BUFFERS: usize = MAX_IN_FLIGHT * 2;
const MAX_POOLED_BUFFER_BYTES: usize = 1024 * 1024;

type ComputePipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type SharedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

struct CachedPipeline {
    source: Box<str>,
    entry: Box<str>,
    shader_validation: bool,
    pipeline: ComputePipeline,
}

struct IdleBuffer {
    byte_len: usize,
    buffer: SharedBuffer,
}

// SAFETY: Idle buffers are owned by a mutex, never accessed by an in-flight
// command, and copied only while that mutex is held. Retain/release and shared
// contents access may occur on the thread that next checks the buffer out.
unsafe impl Send for IdleBuffer {}
unsafe impl Sync for IdleBuffer {}

#[derive(Default)]
struct BufferPool {
    idle: Vec<IdleBuffer>,
    idle_bytes: usize,
    allocations: usize,
}

impl BufferPool {
    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        words: &[u32],
    ) -> Result<SharedBuffer, MetalError> {
        let byte_len = size_of_val(words);
        if let Some(index) = self.idle.iter().position(|idle| idle.byte_len == byte_len) {
            let idle = self.idle.swap_remove(index);
            self.idle_bytes -= idle.byte_len;
            // SAFETY: This shared buffer is idle and has exactly byte_len bytes.
            // The source slice is initialized and disjoint from Metal storage.
            unsafe {
                copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    idle.buffer.contents().cast::<u8>().as_ptr(),
                    byte_len,
                );
            }
            return Ok(idle.buffer);
        }

        let bytes = NonNull::from(words).cast::<c_void>();
        // SAFETY: Metal reads exactly byte_len bytes from the live, initialized
        // slice and makes its own copy within this call.
        let buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    bytes,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    MetalError::new("newBufferWithBytes", "shared buffer allocation failed")
                })?
        };
        self.allocations += 1;
        Ok(buffer)
    }

    fn release(&mut self, buffer: SharedBuffer) {
        let byte_len = buffer.length();
        if self.idle.len() >= MAX_POOLED_BUFFERS
            || byte_len > MAX_POOLED_BUFFER_BYTES.saturating_sub(self.idle_bytes)
        {
            return;
        }
        self.idle_bytes += byte_len;
        self.idle.push(IdleBuffer { byte_len, buffer });
    }
}

const BC_READBACK: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_bc(texture2d<float, access::read> image [[texture(0)]],
                    device uint *output [[buffer(1)]],
                    uint index [[thread_position_in_grid]]) {
    uint width = image.get_width();
    uint4 channels = uint4(round(image.read(uint2(index % width, index / width)) * 255.0f));
    output[index] = channels.r | (channels.g << 8u) | (channels.b << 16u) | (channels.a << 24u);
}
";

// Signed formats return -1.0..=1.0; each byte holds the channel times 127.
const BC_SNORM_READBACK: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_bc_snorm(texture2d<float, access::read> image [[texture(0)]],
                          device uint *output [[buffer(1)]],
                          uint index [[thread_position_in_grid]]) {
    uint width = image.get_width();
    float4 texel = clamp(image.read(uint2(index % width, index / width)), -1.0f, 1.0f);
    uint4 bytes = uint4(int4(round(texel * 127.0f))) & 0xffu;
    output[index] = bytes.r | (bytes.g << 8u) | (bytes.b << 16u) | (bytes.a << 24u);
}
";

enum ComputeInput<'a> {
    Words(&'a [u32]),
    Texture(&'a ProtocolObject<dyn MTLTexture>, usize),
    SampledTexture {
        texture: &'a ProtocolObject<dyn MTLTexture>,
        sampler: &'a ProtocolObject<dyn MTLSamplerState>,
        coords: &'a [u32],
    },
    /// A translated gfx1030 kernel that samples a bound texture: the input
    /// words go to `[[buffer(0)]]`, output to `[[buffer(1)]]`, and the texture
    /// and sampler to `[[texture(0)]]` and `[[sampler(0)]]`, and the one-word
    /// sample-domain fault flag to `[[buffer(2)]]`.
    WordsSampled {
        words: &'a [u32],
        texture: &'a ProtocolObject<dyn MTLTexture>,
        sampler: &'a ProtocolObject<dyn MTLSamplerState>,
        fault: &'a ProtocolObject<dyn MTLBuffer>,
    },
}

impl ComputeInput<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Words(words) => words.len(),
            Self::Texture(_, pixels) => *pixels,
            Self::SampledTexture { coords, .. } => coords.len() / 2,
            Self::WordsSampled { words, .. } => words.len(),
        }
    }
}

/// Host-provided backing bytes for one synthetic guest resource.
pub struct GuestBuffer<'a> {
    pub format: u32,
    pub bytes: &'a [u8],
}

impl GuestBuffer<'_> {
    /// The synthetic ABI's format identifier for little-endian 32-bit words.
    pub const U32_LE: u32 = 1;
}

/// The synthetic guest's selection of an input range from a resource table.
#[derive(Debug, Clone, Copy)]
pub struct GuestBufferView {
    /// Zero-based index in the supplied resource table.
    pub handle: usize,
    /// Input slot in the fixture ABI. Only slot zero accepts guest data.
    pub binding: u32,
    pub byte_offset: usize,
    pub byte_len: usize,
}

/// Failure at a named operation in the native Metal backend.
#[derive(Debug)]
pub struct MetalError {
    pub operation: &'static str,
    pub detail: String,
}

impl MetalError {
    /// Operation name of the error returned when the device lacks BC texture
    /// compression, which is a hardware capability, not an implementation fault.
    pub const BC_UNAVAILABLE: &'static str = "supportsBCTextureCompression";

    /// Whether the device lacks a required capability, as opposed to a failure
    /// in Xenora's validation or Metal execution.
    #[must_use]
    pub fn is_capability_unavailable(&self) -> bool {
        self.operation == Self::BC_UNAVAILABLE
    }

    pub(crate) fn new(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            operation,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.operation, self.detail)
    }
}

impl std::error::Error for MetalError {}

/// One compiled-and-dispatched kernel with its measured stage timings.
///
/// Durations come from monotonic host clocks for the compiler stages and from
/// the command buffer's GPU start and end timestamps for `gpu_seconds`. They
/// measure a single submission and are not a benchmark distribution.
#[derive(Debug, Clone)]
pub struct RunProfile {
    /// Host library compilation time, or zero when reusing a compute pipeline.
    pub library_seconds: f64,
    /// Host pipeline creation time, or zero when reusing a compute pipeline.
    pub pipeline_seconds: f64,
    pub gpu_seconds: f64,
    /// Host wall time from the start of setup until wait observes completion.
    pub host_seconds: f64,
    /// Host time to acquire and initialize the input and output shared buffers,
    /// including any new Metal allocations.
    pub alloc_seconds: f64,
    /// Host time to create the command buffer and encode the dispatch.
    pub encode_seconds: f64,
    /// Host time spent in `commit`.
    pub commit_seconds: f64,
    /// Host time spent in `waitUntilCompleted`.
    pub wait_seconds: f64,
    /// Host time to copy the output words back; not part of `host_seconds`.
    pub readback_seconds: f64,
    pub output: Vec<u32>,
}

/// A native Metal device and its command queue for the fixture ABI.
pub struct NativeCompute {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    submissions: Mutex<SubmissionTable>,
    // Idle shared buffers are recycled only after their command reaches a terminal state.
    buffer_pool: Mutex<BufferPool>,
    // Least recently used first. In-flight submissions retain their own reference.
    compute_pipelines: Mutex<VecDeque<CachedPipeline>>,
}

/// A submitted dispatch whose result can be collected by its issuing context.
#[must_use]
pub struct Submission<'a> {
    context: &'a NativeCompute,
    id: u64,
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        let mut table = self.context.submissions();
        if let Some(entry) = table.entries.get_mut(&self.id) {
            if entry.resources.is_some() {
                entry.abandoned = true;
            } else {
                table.entries.remove(&self.id);
            }
        }
    }
}

#[doc(hidden)]
pub struct SubmissionGate {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl SubmissionGate {
    pub fn open(&self) {
        self.event.setSignaledValue(1);
    }
}

impl Drop for SubmissionGate {
    fn drop(&mut self) {
        self.open();
    }
}

#[derive(Default)]
struct SubmissionTable {
    next_id: u64,
    entries: BTreeMap<u64, SubmissionEntry>,
}

impl SubmissionTable {
    fn in_flight(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.resources.is_some())
            .count()
    }

    fn check_capacity(&self) -> Result<(), MetalError> {
        if self.in_flight() >= MAX_IN_FLIGHT {
            return Err(MetalError::new(
                "in-flight limit",
                format!("at most {MAX_IN_FLIGHT} compute submissions may be in flight"),
            ));
        }
        Ok(())
    }
}

struct SubmissionEntry {
    resources: Option<InFlightResources>,
    result: Result<RunProfile, MetalError>,
    host_start: Instant,
    abandoned: bool,
    waiting: bool,
}

struct RetiredBuffers {
    input: Option<SharedBuffer>,
    output: SharedBuffer,
}

/// Owns every resource referenced by the unretained command buffer until
/// Metal reports Completed or Error.
struct InFlightResources {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _pipeline: ComputePipeline,
    _input: Option<SharedBuffer>,
    output: SharedBuffer,
    _texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    _sampler: Option<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    _fault: Option<SharedBuffer>,
    gate: Option<Retained<ProtocolObject<dyn MTLSharedEvent>>>,
    word_count: usize,
}

// SAFETY: Metal permits command buffers to move between threads with exclusive
// access. The table mutex and waiting flag keep sweeps off a command during its
// wait. Readback uses private shared storage after GPU completion. Retained
// textures and input buffers are only kept alive by the table.
unsafe impl Send for InFlightResources {}

impl SubmissionEntry {
    fn retire(&mut self, status: MTLCommandBufferStatus) -> Option<RetiredBuffers> {
        if !is_terminal(status) {
            return None;
        }
        let resources = self.resources.take()?;
        if let Err(error) = check_command_status(&resources.command, status) {
            self.result = Err(error);
            return Some(RetiredBuffers {
                input: resources._input,
                output: resources.output,
            });
        }
        if let Ok(profile) = &mut self.result {
            profile.gpu_seconds = resources.command.GPUEndTime() - resources.command.GPUStartTime();
            profile.host_seconds = self.host_start.elapsed().as_secs_f64();
            let readback_start = Instant::now();
            let byte_len = resources.word_count * size_of::<u32>();
            // SAFETY: The initialized shared output is distinct from the host
            // vector and the GPU has finished writing to it. Both ranges hold
            // byte_len bytes and all u32 bit patterns are valid.
            unsafe {
                copy_nonoverlapping(
                    resources.output.contents().cast::<u8>().as_ptr(),
                    profile.output.as_mut_ptr().cast::<u8>(),
                    byte_len,
                );
            }
            profile.readback_seconds = readback_start.elapsed().as_secs_f64();
        }
        Some(RetiredBuffers {
            input: resources._input,
            output: resources.output,
        })
    }
}

impl Drop for NativeCompute {
    fn drop(&mut self) {
        // Tokens can be abandoned (or forgotten) while their work is pending.
        // Open all remaining gates before draining the queue.
        let table = self
            .submissions
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        for resources in table
            .entries
            .values()
            .filter_map(|entry| entry.resources.as_ref())
        {
            if let Some(gate) = &resources.gate {
                gate.setSignaledValue(1);
            }
        }
        for (_, entry) in std::mem::take(&mut table.entries) {
            if let Some(resources) = entry.resources {
                resources.command.waitUntilCompleted();
                if !is_terminal(resources.command.status()) {
                    // Preserve the resources if Metal unexpectedly returns before completion.
                    std::mem::forget(resources);
                }
            }
        }
    }
}

impl NativeCompute {
    /// Open the default Apple GPU and create a command queue.
    ///
    /// # Errors
    ///
    /// Returns an error if discovery, the capability check, or queue creation fails.
    pub fn new() -> Result<Self, MetalError> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().ok_or_else(|| {
                MetalError::new("MTLCreateSystemDefaultDevice", "no Metal device")
            })?;
            // Apple family 7 documents nonuniform threadgroups, which this ABI needs.
            if !device.supportsFamily(MTLGPUFamily::Apple7) {
                return Err(MetalError::new(
                    "supportsFamily",
                    "Apple family 7 capabilities are required",
                ));
            }
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| MetalError::new("newCommandQueue", "queue creation failed"))?;
            Ok(Self {
                device,
                queue,
                submissions: Mutex::default(),
                buffer_pool: Mutex::default(),
                compute_pipelines: Mutex::default(),
            })
        })
    }

    fn submissions(&self) -> MutexGuard<'_, SubmissionTable> {
        self.submissions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn retire_completed(&self) {
        let retired = {
            let mut table = self.submissions();
            let mut retired = Vec::new();
            for entry in table.entries.values_mut() {
                if !entry.waiting
                    && let Some(resources) = &entry.resources
                    && let Some(buffers) = entry.retire(resources.command.status())
                {
                    retired.push(buffers);
                }
            }
            table
                .entries
                .retain(|_, entry| !entry.abandoned || entry.resources.is_some());
            retired
        };
        self.recycle_buffers(retired);
    }

    fn recycle_buffers(&self, retired: Vec<RetiredBuffers>) {
        let mut pool = self
            .buffer_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for buffers in retired {
            if let Some(input) = buffers.input {
                pool.release(input);
            }
            pool.release(buffers.output);
        }
    }

    fn begin_submission(&self) -> Instant {
        let host_start = Instant::now();
        self.retire_completed();
        host_start
    }

    #[doc(hidden)]
    pub fn in_flight(&self) -> usize {
        self.submissions().in_flight()
    }

    #[doc(hidden)]
    pub fn tracked(&self) -> usize {
        self.submissions().entries.len()
    }

    #[doc(hidden)]
    pub fn buffer_allocations(&self) -> usize {
        self.buffer_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .allocations
    }

    #[doc(hidden)]
    pub fn idle_buffer_count(&self) -> usize {
        self.buffer_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .idle
            .len()
    }

    #[doc(hidden)]
    pub fn idle_buffer_bytes(&self) -> usize {
        self.buffer_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .idle_bytes
    }

    /// Report the selected device's registered name.
    pub fn device_name(&self) -> String {
        autoreleasepool(|_| self.device.name().to_string())
    }

    /// Access the underlying native Metal device protocol object.
    #[inline]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Access the underlying native Metal command queue protocol object.
    #[inline]
    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Read an original synthetic BC1 subresource with a native Metal texture.
    ///
    /// The fixed compute kernel returns row-major RGBA8 texels. Each returned
    /// word stores red in the least significant byte and alpha in the highest.
    ///
    /// # Errors
    ///
    /// Returns an error if the subresource is invalid, exceeds the 4096-texel
    /// limit, BC texture compression is unavailable, or Metal execution fails.
    pub fn read_tiled_bc1(
        &self,
        tiled: &TiledBc1<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let (width, height) = tiled
            .mip_extent(mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let pixels = self.validate_bc_extent(width, height, "BC1")?;
        let blocks = tiled
            .linearize(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        self.read_bc_blocks(
            MTLPixelFormat::BC1_RGBA,
            &blocks,
            width.div_ceil(4) * 8,
            width,
            height,
            pixels,
        )
    }

    /// Sample a native Metal RGBA8 texture using a native Metal sampler on Apple Silicon GPU.
    ///
    /// Accepts normalized float UV coordinates.
    /// Executes an authored compute fixture sampling the texture at `[[texture(0)]]` using
    /// the sampler at `[[sampler(0)]]`, reading coordinates from `[[buffer(0)]]`, and writing
    /// row-major RGBA8 texels to `[[buffer(1)]]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of sample points exceeds maximum allowable limits,
    /// or if Metal compilation, dispatch or completion fails.
    pub fn sample_rgba8(
        &self,
        texture: &crate::texture_pipeline::MetalRgba8Texture,
        sampler: &crate::sampler::MetalSampler,
        uv_coords: &[[f32; 2]],
    ) -> Result<Vec<u32>, MetalError> {
        const SAMPLER_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void sample_rgba8(texture2d<float, access::sample> tex [[texture(0)]],
                         sampler s [[sampler(0)]],
                         device const float2 *coords [[buffer(0)]],
                         device uint *output [[buffer(1)]],
                         uint id [[thread_position_in_grid]]) {
    float4 color = tex.sample(s, coords[id], level(0.0f));
    uint4 channels = uint4(round(color * 255.0f));
    output[id] = channels.r | (channels.g << 8u) | (channels.b << 16u) | (channels.a << 24u);
}
"#;
        if uv_coords.is_empty() || uv_coords.len() > MAX_GRID_WORDS {
            return Err(MetalError::new(
                "sample bounds",
                format!(
                    "expected 1..={MAX_GRID_WORDS} sample points, got {}",
                    uv_coords.len()
                ),
            ));
        }

        let words: Vec<u32> = uv_coords
            .iter()
            .flat_map(|[u, v]| [u.to_bits(), v.to_bits()])
            .collect();

        validate_source(SAMPLER_KERNEL, &["sample_rgba8"])?;
        autoreleasepool(|_| {
            self.run(
                SAMPLER_KERNEL,
                "sample_rgba8",
                ComputeInput::SampledTexture {
                    texture: texture.raw(),
                    sampler: sampler.raw(),
                    coords: &words,
                },
            )
            .map(|profile| profile.output)
        })
    }

    /// Read a selected subresource from a validated pitched BC1 upload.
    ///
    /// Output matches [`Self::read_tiled_bc1`]. Metal receives the upload's
    /// block-row stride and reads only the selected level's logical blocks.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or mip, out-of-policy size,
    /// unavailable BC compression, or Metal setup or execution failure.
    pub fn read_bc1_upload(
        &self,
        upload: &Bc1Upload<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let level = upload
            .subresource(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let pixels = self.validate_bc_extent(level.width, level.height, "BC1")?;
        self.read_bc_blocks(
            MTLPixelFormat::BC1_RGBA,
            level.bytes,
            level.row_pitch,
            level.width,
            level.height,
            pixels,
        )
    }

    /// Read a selected subresource from a validated pitched BC2 or BC3 upload.
    ///
    /// The original 16-byte blocks go to a native `BC2_RGBA` or `BC3_RGBA`
    /// texture, and the output has the layout of [`Self::read_bc1_upload`].
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or mip, out-of-policy size,
    /// unavailable BC compression, or Metal setup or execution failure.
    pub fn read_bc_alpha_upload(
        &self,
        upload: &BcAlphaUpload<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let level = upload
            .subresource(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let (format, name) = match upload.format() {
            BcAlphaFormat::Bc2 => (MTLPixelFormat::BC2_RGBA, "BC2"),
            BcAlphaFormat::Bc3 => (MTLPixelFormat::BC3_RGBA, "BC3"),
        };
        let pixels = self.validate_bc_extent(level.width, level.height, name)?;
        self.read_bc_blocks(
            format,
            level.bytes,
            level.row_pitch,
            level.width,
            level.height,
            pixels,
        )
    }

    /// Read a selected subresource from a validated pitched BC4 or BC5 upload.
    ///
    /// The original blocks go to a native `BC4_R*` or `BC5_RG*` texture.
    /// Unsigned output has the layout of [`Self::read_bc1_upload`]. Signed
    /// output stores each channel, times 127 and rounded, as a two's-complement
    /// byte in the same order. Metal fills missing green and blue with 0 and
    /// alpha with 1.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid layer or mip, out-of-policy size,
    /// unavailable BC compression (operation `supportsBCTextureCompression`),
    /// or Metal setup or execution failure.
    pub fn read_bc_channel_upload(
        &self,
        upload: &BcChannelUpload<'_>,
        layer: usize,
        mip: u32,
    ) -> Result<Vec<u32>, MetalError> {
        let level = upload
            .subresource(layer, mip)
            .map_err(|error| MetalError::new("texture subresource", format!("{error:?}")))?;
        let (format, name) = match upload.format() {
            BcChannelFormat::Bc4Unorm => (MTLPixelFormat::BC4_RUnorm, "BC4 UNORM"),
            BcChannelFormat::Bc4Snorm => (MTLPixelFormat::BC4_RSnorm, "BC4 SNORM"),
            BcChannelFormat::Bc5Unorm => (MTLPixelFormat::BC5_RGUnorm, "BC5 UNORM"),
            BcChannelFormat::Bc5Snorm => (MTLPixelFormat::BC5_RGSnorm, "BC5 SNORM"),
        };
        let pixels = self.validate_bc_extent(level.width, level.height, name)?;
        self.read_bc_blocks(
            format,
            level.bytes,
            level.row_pitch,
            level.width,
            level.height,
            pixels,
        )
    }

    fn validate_bc_extent(
        &self,
        width: usize,
        height: usize,
        name: &str,
    ) -> Result<usize, MetalError> {
        let pixels = width
            .checked_mul(height)
            .filter(|pixels| (1..=MAX_GRID_WORDS).contains(pixels))
            .ok_or_else(|| MetalError::new("texture bounds", "expected 1..=4096 texels"))?;
        if !self.device.supportsBCTextureCompression() {
            return Err(MetalError::new(
                MetalError::BC_UNAVAILABLE,
                format!("{name} is unavailable"),
            ));
        }
        Ok(pixels)
    }

    fn read_bc_blocks(
        &self,
        format: MTLPixelFormat,
        blocks: &[u8],
        row_pitch: usize,
        width: usize,
        height: usize,
        pixels: usize,
    ) -> Result<Vec<u32>, MetalError> {
        autoreleasepool(|_| {
            // SAFETY: The positive bounded dimensions and supported BC format
            // describe a 2D texture with just its base level.
            let descriptor = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    format, width, height, false,
                )
            };
            descriptor.setStorageMode(MTLStorageMode::Shared);
            descriptor.setUsage(MTLTextureUsage::ShaderRead);
            let texture = self
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or_else(|| {
                    MetalError::new("newTextureWithDescriptor", "BC allocation failed")
                })?;
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width,
                    height,
                    depth: 1,
                },
            };
            // Apple documents a zero stride for a single compressed block row.
            let bytes_per_row = if height <= 4 { 0 } else { row_pitch };
            // SAFETY: The region spans the whole level, including any partial
            // edge block. Both callers validate that the input holds all rows
            // of complete blocks of this format at the supplied, block-aligned
            // stride.
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    NonNull::from(blocks).cast(),
                    bytes_per_row,
                );
            }
            let (source, entry) =
                if format == MTLPixelFormat::BC4_RSnorm || format == MTLPixelFormat::BC5_RGSnorm {
                    (BC_SNORM_READBACK, "read_bc_snorm")
                } else {
                    (BC_READBACK, "read_bc")
                };
            self.run(source, entry, ComputeInput::Texture(&texture, pixels))
                .map(|profile| profile.output)
        })
    }

    /// Run trusted MSL with read-only words at buffer 0 and output words at buffer 1.
    ///
    /// The kernel must use one invocation per word and restrict its memory accesses
    /// to these bindings and ranges. Run this inside a child process with a wall
    /// deadline. The backend caps resources and dispatch size but does not verify
    /// the MSL, its control flow, or its termination.
    ///
    /// # Errors
    ///
    /// Returns an error for out-of-policy input sizes, compilation, resource setup,
    /// dispatch or completion failure.
    pub fn run_words(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
    ) -> Result<Vec<u32>, MetalError> {
        self.run_words_profiled(source, entry, input)
            .map(|profile| profile.output)
    }

    /// Run trusted MSL that also samples a bound RGBA8 texture and sampler.
    ///
    /// The input words reach `[[buffer(0)]]`, the output `[[buffer(1)]]`, and
    /// the validated texture and sampler `[[texture(0)]]` and `[[sampler(0)]]`.
    /// The bounds, completion and trusted-kernel requirements of
    /// [`Self::run_words`] apply. The kernel also takes a zeroed fault word at
    /// `[[buffer(2)]]`; a nonzero word after completion means an active lane
    /// sampled outside the supported domain, and the output is discarded.
    ///
    /// # Errors
    ///
    /// Returns an error for out-of-policy input sizes, compilation, resource
    /// setup, dispatch or completion failure, and when an active lane's
    /// explicit LOD is not a whole mip level or a coordinate is NaN or
    /// infinite (the cases the CPU reference refuses).
    pub fn run_words_sampled(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
        texture: &crate::texture_pipeline::MetalRgba8Texture,
        sampler: &crate::sampler::MetalSampler,
    ) -> Result<Vec<u32>, MetalError> {
        validate_words(source, entry, input)?;
        validate_image_binding(input, texture, sampler)?;
        autoreleasepool(|_| {
            let fault = self.new_buffer(&[0])?;
            let output = self
                .run_sized(
                    source,
                    entry,
                    ComputeInput::WordsSampled {
                        words: input,
                        texture: texture.raw(),
                        sampler: sampler.raw(),
                        fault: &fault,
                    },
                    None,
                )?
                .output;
            // SAFETY: run_sized returned after the command buffer completed, so
            // the GPU no longer writes the live, one-word shared buffer.
            let flags = unsafe { fault.contents().cast::<u32>().as_ptr().read() };
            sample_domain_result(flags)?;
            Ok(output)
        })
    }

    /// Resolve a synthetic guest input view and run the trusted fixture kernel.
    ///
    /// The validated word range is copied into its own Metal buffer at slot zero.
    /// Slot one receives a separate output buffer with the same number of words.
    /// The trusted-kernel requirements of [`Self::run_words`] also apply here.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported bindings or formats, invalid handles or
    /// ranges, or any error from [`Self::run_words`].
    pub fn run_guest_words(
        &self,
        source: &str,
        entry: &str,
        resources: &[GuestBuffer<'_>],
        view: GuestBufferView,
    ) -> Result<Vec<u32>, MetalError> {
        if view.binding != 0 {
            return Err(MetalError::new(
                "guest buffer binding",
                "expected input slot 0",
            ));
        }
        let buffer = resources
            .get(view.handle)
            .ok_or_else(|| MetalError::new("guest buffer handle", "handle is not in the table"))?;
        if buffer.format != GuestBuffer::U32_LE {
            return Err(MetalError::new("guest buffer format", "expected U32_LE"));
        }
        let word_size = size_of::<u32>();
        if !view.byte_offset.is_multiple_of(word_size)
            || !view.byte_len.is_multiple_of(word_size)
            || !(1..=MAX_GRID_WORDS).contains(&(view.byte_len / word_size))
        {
            return Err(MetalError::new(
                "guest buffer bounds",
                format!("expected an aligned range of 1..={MAX_GRID_WORDS} words"),
            ));
        }
        let bytes = view
            .byte_offset
            .checked_add(view.byte_len)
            .and_then(|end| buffer.bytes.get(view.byte_offset..end))
            .ok_or_else(|| MetalError::new("guest buffer bounds", "range exceeds the resource"))?;
        let words: Vec<_> = bytes
            .as_chunks::<{ size_of::<u32>() }>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect();
        self.run_words(source, entry, &words)
    }

    /// Run trusted MSL and report the measured compilation and execution stages.
    ///
    /// Applies the same bounds and completion checks as [`Self::run_words`].
    ///
    /// # Errors
    ///
    /// Returns an error for out-of-policy input sizes, compilation, resource setup,
    /// dispatch or completion failure.
    pub fn run_words_profiled(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
    ) -> Result<RunProfile, MetalError> {
        self.run_profiled_words(source, entry, input, None)
    }

    /// Submit trusted MSL with read-only words at buffer 0 and output words at buffer 1.
    ///
    /// The trusted-kernel requirements of [`Self::run_words`] apply.
    /// Use [`Self::wait`] to collect the result.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input or source, compilation or dispatch
    /// setup failure, or a full in-flight table.
    pub fn submit_words(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
    ) -> Result<Submission<'_>, MetalError> {
        autoreleasepool(|_| {
            let host_start = self.begin_submission();
            validate_words(source, entry, input)?;
            self.submit_sized(
                source,
                entry,
                ComputeInput::Words(input),
                None,
                None,
                host_start,
            )
        })
    }

    #[doc(hidden)]
    pub fn submit_words_gated(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
    ) -> Result<(Submission<'_>, SubmissionGate), MetalError> {
        autoreleasepool(|_| {
            let host_start = self.begin_submission();
            validate_words(source, entry, input)?;
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| MetalError::new("newSharedEvent", "gate creation failed"))?;
            let submission = self.submit_sized(
                source,
                entry,
                ComputeInput::Words(input),
                None,
                Some(&event),
                host_start,
            )?;
            Ok((submission, SubmissionGate { event }))
        })
    }

    /// Wait for this context's submission and return its readback and timings.
    ///
    /// # Errors
    ///
    /// Returns an error for a token from another context, an unknown submission,
    /// or a Metal completion failure.
    pub fn wait(&self, submission: Submission<'_>) -> Result<RunProfile, MetalError> {
        if !std::ptr::eq(self, submission.context) {
            return Err(MetalError::new(
                "submission",
                "token belongs to another context",
            ));
        }
        autoreleasepool(|_| {
            let id = submission.id;
            std::mem::forget(submission);
            let result = self.take_result(id);
            self.retire_completed();
            result
        })
    }

    fn take_result(&self, id: u64) -> Result<RunProfile, MetalError> {
        let unknown = || MetalError::new("submission", "unknown submission id");
        let command = {
            let mut table = self.submissions();
            let entry = table.entries.get_mut(&id).ok_or_else(unknown)?;
            entry.resources.as_ref().map(|resources| {
                entry.waiting = true;
                resources.command.clone()
            })
        };
        let wait_seconds = command.map_or(0.0, |command| {
            // Keep the table available to other submitters while Metal waits.
            let start = Instant::now();
            command.waitUntilCompleted();
            start.elapsed().as_secs_f64()
        });

        let (result, host_seconds, retired) = {
            let mut table = self.submissions();
            let entry = table.entries.get_mut(&id).ok_or_else(unknown)?;
            entry.waiting = false;
            let mut retired = None;
            let host_seconds = if let Some(resources) = &entry.resources {
                let status = resources.command.status();
                if !is_terminal(status) {
                    entry.abandoned = true;
                    check_command_status(&resources.command, status)?;
                }
                retired = entry.retire(status);
                None
            } else {
                Some(entry.host_start.elapsed().as_secs_f64())
            };
            let result = table.entries.remove(&id).ok_or_else(unknown)?.result;
            (result, host_seconds, retired)
        };
        if let Some(buffers) = retired {
            self.recycle_buffers(vec![buffers]);
        }
        let mut profile = result?;
        profile.wait_seconds = wait_seconds;
        if let Some(host_seconds) = host_seconds {
            profile.host_seconds = host_seconds;
        }
        Ok(profile)
    }

    /// Run trusted MSL with an explicit one-dimensional threadgroup width.
    ///
    /// The grid still contains one invocation per input word. The requested width
    /// is passed to Metal even when the grid is smaller or ends in a partial group.
    /// The trusted-kernel requirements of [`Self::run_words`] apply.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::run_words_profiled`], with dispatch
    /// bounds evaluated using this width. Zero and widths above the device or
    /// pipeline limit are rejected.
    pub fn run_words_with_group_width(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
        width: usize,
    ) -> Result<RunProfile, MetalError> {
        self.run_profiled_words(source, entry, input, Some(width))
    }

    /// Find or compile the kernel and report the smaller of the device and pipeline width limits.
    ///
    /// Input size and threadgroup count are checked separately at dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error if source validation, compilation or pipeline setup fails.
    pub fn max_group_width(&self, source: &str, entry: &str) -> Result<usize, MetalError> {
        validate_source(source, &[entry])?;
        autoreleasepool(|_| {
            let (pipeline, _, _) = self.prepare_compute_pipeline(source, entry)?;
            Ok(self.pipeline_group_width_limit(&pipeline))
        })
    }

    fn run_profiled_words(
        &self,
        source: &str,
        entry: &str,
        input: &[u32],
        group_width: Option<usize>,
    ) -> Result<RunProfile, MetalError> {
        validate_words(source, entry, input)?;
        autoreleasepool(|_| self.run_sized(source, entry, ComputeInput::Words(input), group_width))
    }

    fn run(
        &self,
        source: &str,
        entry: &str,
        input: ComputeInput<'_>,
    ) -> Result<RunProfile, MetalError> {
        self.run_sized(source, entry, input, None)
    }

    /// [`Self::run`] with an optional explicit threadgroup width; `None` keeps
    /// the default sizing.
    fn run_sized(
        &self,
        source: &str,
        entry: &str,
        input: ComputeInput<'_>,
        group_width: Option<usize>,
    ) -> Result<RunProfile, MetalError> {
        let host_start = self.begin_submission();
        self.wait(self.submit_sized(source, entry, input, group_width, None, host_start)?)
    }

    fn submit_sized(
        &self,
        source: &str,
        entry: &str,
        input: ComputeInput<'_>,
        group_width: Option<usize>,
        gate: Option<&ProtocolObject<dyn MTLSharedEvent>>,
        host_start: Instant,
    ) -> Result<Submission<'_>, MetalError> {
        self.submissions().check_capacity()?;
        let (pipeline, library_seconds, pipeline_seconds) =
            self.prepare_compute_pipeline(source, entry)?;

        let limit = self.pipeline_group_width_limit(&pipeline);
        let width = match group_width {
            Some(width) if (1..=limit).contains(&width) => width,
            Some(width) => {
                return Err(MetalError::new(
                    "threadgroup width",
                    format!("expected a nonzero width at most {limit}, got {width}"),
                ));
            }
            None => MAX_THREADS_PER_GROUP.min(input.len()).min(limit),
        };
        if width == 0 || input.len().div_ceil(width) > MAX_GROUPS {
            return Err(MetalError::new(
                "dispatch bounds",
                "device or pipeline would require too many groups",
            ));
        }

        let byte_len = input.len() * size_of::<u32>();
        if byte_len > self.device.maxBufferLength() {
            return Err(MetalError::new(
                "buffer bounds",
                "fixture exceeds device buffer limit",
            ));
        }
        let alloc_start = Instant::now();
        let mut pool = self
            .buffer_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut input_buffer = match input {
            ComputeInput::Words(words) => Some(pool.acquire(&self.device, words)?),
            ComputeInput::SampledTexture { coords, .. } => {
                Some(pool.acquire(&self.device, coords)?)
            }
            ComputeInput::WordsSampled { words, .. } => Some(pool.acquire(&self.device, words)?),
            ComputeInput::Texture(..) => None,
        };
        let words = vec![0xA5A5_A5A5; input.len()];
        let output_buffer = match pool.acquire(&self.device, &words) {
            Ok(buffer) => buffer,
            Err(error) => {
                if let Some(input) = input_buffer.take() {
                    pool.release(input);
                }
                return Err(error);
            }
        };
        let alloc_seconds = alloc_start.elapsed().as_secs_f64();
        drop(pool);
        let mut table = self.submissions();
        if let Err(error) = table.check_capacity() {
            drop(table);
            self.recycle_buffers(vec![RetiredBuffers {
                input: input_buffer,
                output: output_buffer,
            }]);
            return Err(error);
        }
        let id = table.next_id;
        table.next_id = match id.checked_add(1) {
            Some(next) => next,
            None => {
                drop(table);
                self.recycle_buffers(vec![RetiredBuffers {
                    input: input_buffer,
                    output: output_buffer,
                }]);
                return Err(MetalError::new("submission", "submission ids exhausted"));
            }
        };
        let encode_start = Instant::now();
        let command = match self.queue.commandBufferWithUnretainedReferences() {
            Some(command) => command,
            None => {
                drop(table);
                self.recycle_buffers(vec![RetiredBuffers {
                    input: input_buffer,
                    output: output_buffer,
                }]);
                return Err(MetalError::new(
                    "commandBuffer",
                    "command buffer creation failed",
                ));
            }
        };
        if let Some(gate) = gate {
            command.encodeWaitForEvent_value(gate.as_ref(), 1);
        }
        let encoder = match command.computeCommandEncoder() {
            Some(encoder) => encoder,
            None => {
                drop(table);
                self.recycle_buffers(vec![RetiredBuffers {
                    input: input_buffer,
                    output: output_buffer,
                }]);
                return Err(MetalError::new(
                    "computeCommandEncoder",
                    "compute encoder creation failed",
                ));
            }
        };
        encoder.setComputePipelineState(&pipeline);
        // SAFETY: All resources are live, initialized, and were allocated by
        // this device. The kernel uses one input at index zero and writes the
        // separate output buffer at index one. Both buffer offsets are zero.
        unsafe {
            match input {
                ComputeInput::Texture(texture, _) => {
                    encoder.setTexture_atIndex(Some(texture), 0);
                }
                ComputeInput::SampledTexture {
                    texture, sampler, ..
                } => {
                    encoder.setTexture_atIndex(Some(texture), 0);
                    encoder.setSamplerState_atIndex(Some(sampler), 0);
                }
                ComputeInput::WordsSampled {
                    texture,
                    sampler,
                    fault,
                    ..
                } => {
                    encoder.setTexture_atIndex(Some(texture), 0);
                    encoder.setSamplerState_atIndex(Some(sampler), 0);
                    encoder.setBuffer_offset_atIndex(Some(fault), 0, 2);
                }
                ComputeInput::Words(..) => {}
            }
            if let Some(buffer) = input_buffer.as_deref() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, 0);
            }
            encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(size(input.len()), size(width));
        encoder.endEncoding();
        let encode_seconds = encode_start.elapsed().as_secs_f64();
        let (texture, sampler, fault) = match input {
            ComputeInput::Texture(texture, _) => (Some(texture.retain()), None, None),
            ComputeInput::SampledTexture {
                texture, sampler, ..
            } => (Some(texture.retain()), Some(sampler.retain()), None),
            ComputeInput::WordsSampled {
                texture,
                sampler,
                fault,
                ..
            } => (
                Some(texture.retain()),
                Some(sampler.retain()),
                Some(fault.retain()),
            ),
            ComputeInput::Words(_) => (None, None, None),
        };
        let resources = InFlightResources {
            command,
            _pipeline: pipeline,
            _input: input_buffer,
            output: output_buffer,
            _texture: texture,
            _sampler: sampler,
            _fault: fault,
            gate: gate.map(Message::retain),
            word_count: input.len(),
        };
        let commit_start = Instant::now();
        resources.command.commit();
        let commit_seconds = commit_start.elapsed().as_secs_f64();
        table.entries.insert(
            id,
            SubmissionEntry {
                resources: Some(resources),
                result: Ok(RunProfile {
                    library_seconds,
                    pipeline_seconds,
                    gpu_seconds: 0.0,
                    host_seconds: 0.0,
                    alloc_seconds,
                    encode_seconds,
                    commit_seconds,
                    wait_seconds: 0.0,
                    readback_seconds: 0.0,
                    output: words,
                }),
                host_start,
                abandoned: false,
                waiting: false,
            },
        );
        Ok(Submission { context: self, id })
    }

    fn compile_library(
        &self,
        source: &str,
        options: &MTLCompileOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        self.device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(options))
            .map_err(|error| {
                MetalError::new(
                    "newLibraryWithSource",
                    error.localizedDescription().to_string(),
                )
            })
    }

    fn prepare_compute_pipeline(
        &self,
        source: &str,
        entry: &str,
    ) -> Result<(ComputePipeline, f64, f64), MetalError> {
        let shader_validation = std::env::var("MTL_SHADER_VALIDATION").as_deref() == Ok("1");
        // Device identity is fixed by this context. All misses use the same
        // compile options (safe math, precise functions, default language,
        // no macros) and unspecialized functions. Any configurable options
        // must become part of the key before they are used here.
        let mut cache = self
            .compute_pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(index) = cache.iter().position(|cached| {
            cached.source.as_ref() == source
                && cached.entry.as_ref() == entry
                && cached.shader_validation == shader_validation
        }) {
            let cached = cache.remove(index).expect("position in the locked cache");
            let pipeline = cached.pipeline.clone();
            cache.push_back(cached);
            return Ok((pipeline, 0.0, 0.0));
        }

        // Serialize cache misses in this context so concurrent requests for
        // one key share the same build. The submission table is unlocked here.
        let options = compile_options()?;
        let library_start = Instant::now();
        let library = self.compile_library(source, &options)?;
        let library_seconds = library_start.elapsed().as_secs_f64();
        let function = find_function(&library, entry)?;
        let pipeline_start = Instant::now();
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| {
                MetalError::new(
                    "newComputePipelineStateWithFunction",
                    error.localizedDescription().to_string(),
                )
            })?;
        let pipeline_seconds = pipeline_start.elapsed().as_secs_f64();
        if shader_validation {
            verify_shader_validation(&pipeline)?;
        }

        // Bound owned key text to 1 MiB. A valid oversized key can run
        // uncached, so this budget does not alter source or entry validation.
        if entry.len() <= MAX_SOURCE_BYTES.saturating_sub(source.len()) {
            if cache.len() == MAX_CACHED_PIPELINES {
                cache.pop_front();
            }
            cache.push_back(CachedPipeline {
                source: source.into(),
                entry: entry.into(),
                shader_validation,
                pipeline: pipeline.clone(),
            });
        }
        Ok((pipeline, library_seconds, pipeline_seconds))
    }

    fn pipeline_group_width_limit(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> usize {
        self.device
            .maxThreadsPerThreadgroup()
            .width
            .min(pipeline.maxTotalThreadsPerThreadgroup())
    }

    fn new_buffer(
        &self,
        words: &[u32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let bytes = NonNull::from(words).cast::<c_void>();
        // SAFETY: Metal reads exactly size_of_val(words) bytes from the live,
        // initialized slice and makes its own copy within this call.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                bytes,
                size_of_val(words),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| MetalError::new("newBufferWithBytes", "shared buffer allocation failed"))
    }
}

/// The guard bits a translated sample kernel sets in its fault word. They are
/// `xenora_shader::SAMPLE_FAULT_LOD` and `SAMPLE_FAULT_COORDINATES`; the metal
/// crate does not depend on the shader crate, so the values are repeated here
/// and pinned by the native sample tests.
const SAMPLE_FAULT_LOD: u32 = 1;
const SAMPLE_FAULT_COORDINATES: u32 = 2;

/// Refuse a dispatch whose fault word reports an out-of-domain active lane.
fn sample_domain_result(flags: u32) -> Result<(), MetalError> {
    if flags == 0 {
        return Ok(());
    }
    let mut reasons = Vec::new();
    if flags & SAMPLE_FAULT_LOD != 0 {
        reasons.push("an active lane's explicit LOD is not a whole mip level");
    }
    if flags & SAMPLE_FAULT_COORDINATES != 0 {
        reasons.push("an active lane's texture coordinate is NaN or infinite");
    }
    if flags & !(SAMPLE_FAULT_LOD | SAMPLE_FAULT_COORDINATES) != 0 {
        reasons.push("the kernel set an unknown fault bit");
    }
    Err(MetalError::new("image sample domain", reasons.join("; ")))
}

/// Validate the T# and S# the synthetic input carries against the bound texture
/// and sampler state before a translated sample runs.
///
/// The descriptor words are parsed with the `xenora-ref` reference functions so
/// the GPU path and the CPU oracle agree on what a valid descriptor is. The
/// crate graph permits the dependency (the reference crate has no dependencies
/// of its own); the shared copy alternative was not needed.
fn validate_image_binding(
    input: &[u32],
    texture: &crate::texture_pipeline::MetalRgba8Texture,
    sampler: &crate::sampler::MetalSampler,
) -> Result<(), MetalError> {
    use xenora_ref::texture_sampling;

    let resource: [u32; 8] = input
        .get(0..8)
        .ok_or_else(|| {
            MetalError::new(
                "image resource descriptor",
                "input has fewer than eight descriptor words",
            )
        })?
        .try_into()
        .expect("a slice of length eight");
    let sampler_words: [u32; 4] = input
        .get(8..12)
        .ok_or_else(|| {
            MetalError::new(
                "image sampler descriptor",
                "input has fewer than twelve descriptor words",
            )
        })?
        .try_into()
        .expect("a slice of length four");
    let (desc, _base) = texture_sampling::parse_resource_descriptor(resource)
        .map_err(|error| MetalError::new("image resource descriptor", error.to_string()))?;
    let last_level = desc.mips - 1;
    texture_sampling::validate_sampler_descriptor(sampler_words, last_level)
        .map_err(|error| MetalError::new("image sampler descriptor", error.to_string()))?;

    let bound = texture.desc();
    if (desc.width, desc.height, desc.mips) != (bound.width, bound.height, bound.mips) {
        return Err(MetalError::new(
            "image resource descriptor",
            "descriptor disagrees with the bound texture width, height or mip count",
        ));
    }

    let state = sampler.desc();
    let nearest = |filter: FilterMode| matches!(filter, FilterMode::Nearest);
    let clamp = |mode: AddressMode| matches!(mode, AddressMode::ClampToEdge);
    if !nearest(state.min_filter)
        || !nearest(state.mag_filter)
        || !nearest(state.mip_filter)
        || !clamp(state.address_u)
        || !clamp(state.address_v)
        || state.min_lod != 0.0
        || state.max_lod < last_level as f32
    {
        return Err(MetalError::new(
            "image sampler descriptor",
            "descriptor disagrees with the bound sampler state",
        ));
    }
    Ok(())
}

fn validate_words(source: &str, entry: &str, input: &[u32]) -> Result<(), MetalError> {
    if !(1..=MAX_GRID_WORDS).contains(&input.len()) {
        return Err(MetalError::new(
            "dispatch bounds",
            format!("expected 1..={MAX_GRID_WORDS} words, got {}", input.len()),
        ));
    }
    validate_source(source, &[entry])
}

fn validate_source(source: &str, entries: &[&str]) -> Result<(), MetalError> {
    if source.is_empty()
        || source.len() > MAX_SOURCE_BYTES
        || entries.iter().any(|entry| entry.is_empty())
    {
        return Err(MetalError::new(
            "source bounds",
            "expected nonempty entries and source of at most 64 KiB",
        ));
    }
    Ok(())
}

pub(crate) fn compile_options() -> Result<Retained<MTLCompileOptions>, MetalError> {
    let options = MTLCompileOptions::new();
    if !options.respondsToSelector(sel!(setMathMode:))
        || !options.respondsToSelector(sel!(setMathFloatingPointFunctions:))
    {
        return Err(MetalError::new(
            "MTLCompileOptions",
            "this Metal runtime lacks explicit safe and precise math options",
        ));
    }
    options.setMathMode(MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    Ok(options)
}

fn find_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, MetalError> {
    library
        .newFunctionWithName(&NSString::from_str(entry))
        .ok_or_else(|| MetalError::new("newFunctionWithName", format!("missing entry {entry}")))
}

fn complete_command(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command.commit();
    wait_command(command)
}

/// Block until a committed command buffer finishes and check its status.
fn wait_command(command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    command.waitUntilCompleted();
    check_command_status(command, command.status())
}

fn is_terminal(status: MTLCommandBufferStatus) -> bool {
    matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}

fn check_command_status(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    status: MTLCommandBufferStatus,
) -> Result<(), MetalError> {
    if status != MTLCommandBufferStatus::Completed {
        let error = command.error().map_or_else(
            || format!("unexpected command status {status:?}"),
            |error| error.localizedDescription().to_string(),
        );
        return Err(MetalError::new("waitUntilCompleted", error));
    }
    Ok(())
}

fn verify_shader_validation(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> Result<(), MetalError> {
    if !pipeline.respondsToSelector(sel!(shaderValidation)) {
        return Err(MetalError::new(
            "shaderValidation",
            "the runtime lacks the pipeline state query",
        ));
    }
    // SAFETY: The selector exists and this newly created pipeline is private
    // to the current thread. The non-atomic property is read before encoding.
    if unsafe { pipeline.shaderValidation() } != MTLShaderValidation::Enabled {
        return Err(MetalError::new(
            "shaderValidation",
            "shader validation requested but inactive for the pipeline",
        ));
    }
    Ok(())
}

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}
