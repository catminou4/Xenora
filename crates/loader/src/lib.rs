// SPDX-License-Identifier: GPL-3.0-or-later

//! Xenora Loader crate.
//!
//! Provides PE32+ parsing, isolated guest virtual memory, bounded x86-64 CPU
//! execution bootstrap, and read-only binary triage.

pub mod arm64;
pub mod cpu;
pub mod dbt;
pub(crate) mod exceptions;
pub mod export_table;
pub mod graphics;
pub mod import_table;
pub mod imports;
pub mod jit;
pub mod memory;
pub mod module_graph;
pub mod output_path;
pub mod pe;
pub mod probe;
pub mod reloc;
pub mod runtime;
#[rustfmt::skip]
pub mod runtime_table;
pub mod scheduler;
pub mod services;
pub mod softfloat;
pub mod startup;
pub mod tasks;
pub mod tls;
pub mod triage;
pub mod x86_avx;
pub mod x86_decode;
pub mod x86_exec;
pub mod x86_float;
pub mod x86_simd;
pub mod x86_strings;
#[rustfmt::skip]
pub mod x86_table;

pub use arm64::Arm64Assembler;
pub use cpu::{CpuState, ExecutionResult, ExitReason, Interpreter, RFlags, Register};
pub use dbt::{DbtEngine, DbtMetrics};
pub use graphics::{
    BackendRenderer, CommandListType, ExecutionOutcome, FramePresentation, GraphicsBackendId,
    GraphicsCommand, GraphicsCommandAllocator, GraphicsCommandList, GraphicsCommandQueue,
    GraphicsContext, GraphicsError, GraphicsHandle, GraphicsResource, ResourceDesc,
    ResourceDimension, ResourceStates,
};
pub use imports::{ImportError, ImportRegistry, ServiceDescriptor, ServiceHandler, Win64Args};
pub use jit::{JitError, JitMemory};
pub use memory::{GuestMemory, MemoryError, MemoryPermissions};
pub use pe::{LoadedImage, ParsedPe, PeError, PeSection};
pub use probe::{
    ImportedSymbol, InstructionRecord, ProbeConfig, ProbeReport, RelocationCensus, TlsCensus,
    probe_pe_bytes, run_probe_and_export,
};
pub use reloc::{
    IMAGE_REL_BASED_ABSOLUTE, IMAGE_REL_BASED_DIR64, RelocationError, RelocationReport,
    apply_relocations,
};
pub use services::{Clock, MockClock, ServicesState, SystemClock, register_standard_services};
pub use tls::{TlsDirectory, TlsError, execute_tls_callbacks, parse_tls_directory, setup_tls};
pub use triage::{TriageReport, triage_bytes, triage_file};
