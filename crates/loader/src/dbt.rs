// SPDX-License-Identifier: GPL-3.0-or-later

//! Bounded x86-64 to ARM64 Dynamic Binary Translation (DBT) engine for Xenora.
//!
//! Compiles decoded x86-64 basic blocks to native Apple Silicon ARM64 machine code,
//! executes them in a W^X code cache, and manages host-service dispatch and fallbacks.

use crate::arm64::{
    Arm64Assembler, Condition, SP, X0, X1, X2, X3, X4, X5, X6, X9, X10, X11, X12, X13, X14, X15,
    X19, X20, X21, X29, X30, XZR,
};
use crate::cpu::{CpuState, ExecutionResult, ExitReason, RFlags, Register};
use crate::jit::{JitError, JitMemory};
use crate::memory::GuestMemory;
use crate::x86_decode::{Encoding, Insn, MemOperand, RegOperand, RmOperand, Segment, decode};
use crate::x86_exec;
use crate::x86_table::{FORMS, Op};
use std::collections::HashMap;

const MAX_BLOCK_INSTRUCTIONS: usize = 16;

/// Registered guest host-service handler function.
pub type HostServiceFn =
    Box<dyn Fn(&mut CpuState, &mut GuestMemory) -> Result<(), ExitReason> + Send + Sync>;

/// Diagnostic execution counters for DBT runtime profiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DbtMetrics {
    pub blocks_compiled: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    /// Cached blocks discarded because the guest bytes they were translated
    /// from changed or became unreadable. Each also counts as a cache miss.
    pub blocks_invalidated: usize,
    pub native_bytes_emitted: usize,
    pub dbt_dispatches: usize,
    pub instructions_executed: usize,
    /// Instructions run by the shared Rust semantics instead of ARM64 code.
    pub interpreter_fallbacks: usize,
    pub service_calls: usize,
}

/// Counters for direct block linking.
///
/// They are kept apart from [`DbtMetrics`], whose fields other tools list one
/// by one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkMetrics {
    /// Links the dispatcher installed from a static successor of a block to
    /// the full block it selected next for that address.
    pub links_installed: usize,
    /// Block transitions made by following a link in native code, without
    /// returning to the dispatcher.
    pub links_followed: usize,
    /// Exits at a static successor that returned to the dispatcher instead:
    /// no link yet, a target not verified in the current epoch, too little
    /// budget left, or a chain that had to stop.
    pub link_fallbacks: usize,
    /// Times the verification epoch advanced. Each makes every link target
    /// unverified until the dispatcher selects and checks it again.
    pub epochs: usize,
    /// Link targets retired because their guest bytes changed. A retired
    /// target is never entered through a link again.
    pub targets_retired: usize,
    /// Guest code bytes the dispatcher compared to confirm its cache hits.
    /// A followed link skips this check.
    pub bytes_verified: usize,
}

/// The verification epoch of a retired link target. Live epochs start at 1.
const RETIRED_EPOCH: u64 = 0;

/// The epoch a chain takes once no further link may be followed. No target
/// is ever verified in it.
const STOPPED_EPOCH: u64 = u64::MAX;

/// Where a link enters a translated full block. Read by the exit stubs of the
/// blocks linked to it.
#[repr(C)]
struct LinkTarget {
    /// The engine epoch in which the dispatcher last selected this block for
    /// `start` and found its guest bytes unchanged, or [`RETIRED_EPOCH`].
    verified_epoch: u64,
    /// Guest address of the block's first instruction.
    start: u64,
    /// Host address of the block's first instruction after the prologue.
    body: u64,
}

/// One static successor of a block: the guest address control reaches and the
/// target installed for it, or 0.
#[repr(C)]
struct LinkSlot {
    target: u64,
    rip: u64,
}

const TARGET_EPOCH_OFFSET: u32 = std::mem::offset_of!(LinkTarget, verified_epoch) as u32;
const TARGET_START_OFFSET: u32 = std::mem::offset_of!(LinkTarget, start) as u32;
const TARGET_BODY_OFFSET: u32 = std::mem::offset_of!(LinkTarget, body) as u32;
const SLOT_TARGET_OFFSET: u32 = std::mem::offset_of!(LinkSlot, target) as u32;

/// The link records of one translated full block.
struct BlockLinks {
    target: LinkTarget,
    /// A block has at most two static successors, both sides of a `Jcc`.
    slots: [LinkSlot; 2],
}

/// Owner of every [`BlockLinks`]. Emitted code embeds their host addresses and
/// slots point at them, so each lives at a fixed address until the engine,
/// with its code cache, is dropped. They are allocated in chunks, so
/// translating a block does not allocate.
#[derive(Default)]
struct LinkArena {
    chunks: Vec<*mut [BlockLinks]>,
    /// Records handed out from the last chunk.
    used: usize,
}

const LINK_CHUNK: usize = 64;

impl LinkArena {
    fn allocate(&mut self, start: u64) -> *mut BlockLinks {
        if self.chunks.is_empty() || self.used == LINK_CHUNK {
            let chunk: Box<[BlockLinks]> = (0..LINK_CHUNK)
                .map(|_| BlockLinks {
                    target: LinkTarget {
                        verified_epoch: RETIRED_EPOCH,
                        start: 0,
                        body: 0,
                    },
                    slots: [
                        LinkSlot { target: 0, rip: 0 },
                        LinkSlot { target: 0, rip: 0 },
                    ],
                })
                .collect();
            self.chunks.push(Box::into_raw(chunk));
            self.used = 0;
        }
        let chunk = self.chunks[self.chunks.len() - 1];
        // SAFETY: `used` < LINK_CHUNK, so the record is inside the chunk,
        // and no other pointer to it has been handed out.
        unsafe {
            let links = chunk.cast::<BlockLinks>().add(self.used);
            (*links).target.start = start;
            self.used += 1;
            links
        }
    }
}

impl Drop for LinkArena {
    fn drop(&mut self) {
        for chunk in self.chunks.drain(..) {
            // SAFETY: each chunk came from `Box::into_raw` and is freed once.
            drop(unsafe { Box::from_raw(chunk) });
        }
    }
}

/// A translated block and the guest bytes it was translated from.
///
/// A cached block is only valid while those bytes are still executable at the
/// same address. Guest stores into code (Intel SDM Vol. 3A 11.6) and a different
/// address space at the same address both change them.
struct CachedBlock {
    code: *const u8,
    start: u64,
    source: Box<[u8]>,
    /// Link records of a full block; null for a budget-limited one.
    links: *mut BlockLinks,
}

/// A block the dispatcher selected: its entry point and, for a full block,
/// its link records.
struct SelectedBlock {
    code: *const u8,
    links: *mut BlockLinks,
}

impl CachedBlock {
    fn is_current(&self, memory: &GuestMemory) -> bool {
        self.source.iter().enumerate().all(|(offset, &byte)| {
            let address = self.start.wrapping_add(offset as u64);
            matches!(memory.fetch_instruction_byte(address), Ok(current) if current == byte)
        })
    }
}

/// Dynamic Binary Translator and native execution runtime.
///
/// # Direct block linking
///
/// A full block whose last instruction has static successors (the next
/// instruction after a block that ends at the instruction quota, the target
/// of a relative `JMP` or `CALL`, both sides of a `Jcc`) ends with an exit
/// stub. The stub may jump straight to the translated block of that successor
/// instead of returning to the dispatcher. Links are words in host data
/// memory, read by the stub; emitted code is never rewritten.
///
/// The stub follows a link only when all of these hold:
///
/// - the target block starts at the guest address control reaches;
/// - in the current verification epoch, the dispatcher selected that block
///   for its address after its trampoline and service checks, and either
///   translated it or found its guest bytes unchanged;
/// - after this block, enough of the budget is left that the dispatcher would
///   also select a full block: [`MAX_BLOCK_INSTRUCTIONS`] for [`Self::run`],
///   one instruction for a slice;
/// - nothing in the chain recorded an exit or a fault, stored into the range
///   of guest addresses that cached blocks were translated from, or ran an
///   instruction through the shared semantics, which may write anywhere.
///
/// The epoch advances at the start of every run, since the host may change
/// guest memory or pass another address space between runs, and within a run
/// after every service call, trampoline dispatch and fault dispatch, and after
/// any block that ended a chain for one of the reasons in the last point. A
/// block translated again because its bytes changed has its link target
/// retired for good. Every other transition goes through the dispatcher, which
/// keeps its lookup and byte check.
pub struct DbtEngine {
    jit: JitMemory,
    block_cache: HashMap<u64, CachedBlock>,
    limited_block_cache: HashMap<(u64, usize), CachedBlock>,
    assembler: Arm64Assembler,
    /// Decoded instructions that emitted helper calls point into.
    decoded_blocks: Vec<Box<[Insn]>>,
    links: LinkArena,
    /// Current verification epoch; see the type documentation.
    epoch: u64,
    /// Guest addresses that every cached block was translated from lie in
    /// `code_start..code_end`. Empty while nothing is cached.
    code_start: u64,
    code_end: u64,
    linking: bool,
    services: HashMap<u64, HostServiceFn>,
    pub import_registry: Option<crate::imports::ImportRegistry>,
    pub metrics: DbtMetrics,
    pub link_metrics: LinkMetrics,
    /// Exit status of a guest that ended its process on this engine.
    exited: Option<u32>,
}

impl DbtEngine {
    /// Create a new DBT engine with specified JIT cache capacity (e.g. 1MB).
    pub fn new(capacity: usize) -> Result<Self, JitError> {
        Ok(Self {
            jit: JitMemory::new(capacity)?,
            block_cache: HashMap::new(),
            limited_block_cache: HashMap::new(),
            assembler: Arm64Assembler::new(),
            decoded_blocks: Vec::new(),
            links: LinkArena::default(),
            epoch: RETIRED_EPOCH,
            code_start: u64::MAX,
            code_end: 0,
            linking: true,
            services: HashMap::new(),
            import_registry: None,
            metrics: DbtMetrics::default(),
            link_metrics: LinkMetrics::default(),
            exited: None,
        })
    }

    /// Enable or disable direct block linking; it is enabled by default.
    /// Disabled, every block returns to the dispatcher, as before linking
    /// existed, and no link is installed or followed.
    pub fn set_block_linking(&mut self, enabled: bool) {
        self.linking = enabled;
    }

    /// Make every link target unverified until the dispatcher selects and
    /// checks it again.
    fn advance_epoch(&mut self) {
        // A u64 does not wrap back to RETIRED_EPOCH within any real run.
        self.epoch += 1;
        self.link_metrics.epochs += 1;
    }

    /// Attach an import registry for automatic IAT trampoline dispatch.
    pub fn set_import_registry(&mut self, registry: crate::imports::ImportRegistry) {
        self.import_registry = Some(registry);
    }

    /// Register a host service at a synthetic guest virtual address.
    pub fn register_service<F>(&mut self, address: u64, handler: F)
    where
        F: Fn(&mut CpuState, &mut GuestMemory) -> Result<(), ExitReason> + Send + Sync + 'static,
    {
        self.services.insert(address, Box::new(handler));
    }

    /// Execute guest code starting at CPU RIP until HLT, budget exhaustion, or fault.
    ///
    /// Once the guest has ended its process, this engine executes nothing
    /// further: every later call reports the same exit.
    pub fn run(
        &mut self,
        cpu: &mut CpuState,
        memory: &mut GuestMemory,
        max_instructions: usize,
    ) -> ExecutionResult {
        self.run_bounded(cpu, memory, max_instructions, true)
    }

    /// As [`Self::run`], but stop at the first block boundary that reaches the
    /// budget. Repeated slices reuse whole blocks instead of compiling a
    /// shortened one at every cutoff. The budget may be exceeded by less than
    /// one block.
    pub(crate) fn run_slice(
        &mut self,
        cpu: &mut CpuState,
        memory: &mut GuestMemory,
        max_instructions: usize,
    ) -> ExecutionResult {
        self.run_bounded(cpu, memory, max_instructions, false)
    }

    /// Both entry points go through the process-exit latch: once the guest
    /// process has exited, no guest instruction runs again, whichever guest
    /// thread the caller tries to run.
    fn run_bounded(
        &mut self,
        cpu: &mut CpuState,
        memory: &mut GuestMemory,
        max_instructions: usize,
        exact: bool,
    ) -> ExecutionResult {
        if let Some(status) = self.exited {
            return ExecutionResult {
                instructions_executed: 0,
                final_state: cpu.clone(),
                exit_reason: ExitReason::ProcessExit(status),
            };
        }
        let result = self.execute(cpu, memory, max_instructions, exact);
        if let ExitReason::ProcessExit(status) = result.exit_reason {
            self.exited = Some(status);
        }
        result
    }

    fn execute(
        &mut self,
        cpu: &mut CpuState,
        memory: &mut GuestMemory,
        max_instructions: usize,
        exact: bool,
    ) -> ExecutionResult {
        let mut instructions_count = 0;
        // The host may have changed guest memory since the last run, or
        // passed another address space.
        self.advance_epoch();
        // The slot a block's exit stub could not follow, for the block the
        // dispatcher selects next.
        let mut missed_slot: *mut LinkSlot = std::ptr::null_mut();

        while instructions_count < max_instructions {
            // Check if current RIP is an attached import registry trampoline
            if let Some(registry) = &self.import_registry
                && registry.is_service_trampoline(cpu.rip)
            {
                self.metrics.service_calls += 1;
                let dispatched = registry.dispatch(cpu.rip, cpu, memory);
                // A service may change guest memory, its mappings and the
                // trampolines themselves.
                self.advance_epoch();
                missed_slot = std::ptr::null_mut();
                if let Err(reason) = dispatched {
                    return ExecutionResult {
                        instructions_executed: instructions_count,
                        final_state: cpu.clone(),
                        exit_reason: reason,
                    };
                }
                instructions_count += 1;
                continue;
            }

            // Check if current RIP is a registered host service
            if let Some(service) = self.services.get(&cpu.rip) {
                self.metrics.service_calls += 1;
                let served = service(cpu, memory);
                self.advance_epoch();
                missed_slot = std::ptr::null_mut();
                if let Err(reason) = served {
                    return ExecutionResult {
                        instructions_executed: instructions_count,
                        final_state: cpu.clone(),
                        exit_reason: reason,
                    };
                }
                instructions_count += 1;
                continue;
            }

            // Look up or translate block at cpu.rip
            let block_limit = if exact {
                (max_instructions - instructions_count).min(MAX_BLOCK_INSTRUCTIONS)
            } else {
                MAX_BLOCK_INSTRUCTIONS
            };
            let mut block = self.get_or_compile_block(cpu.rip, memory, block_limit);
            // A decode failure can belong to a later instruction in the block.
            // Execute the current instruction alone so earlier effects retire
            // and an earlier data fault takes precedence over speculative fetch.
            if block_limit > 1
                && matches!(
                    block,
                    Err(ExitReason::IllegalInstruction(_)
                        | ExitReason::MemoryFault(_)
                        | ExitReason::GeneralProtection)
                )
            {
                block = self.get_or_compile_block(cpu.rip, memory, 1);
            }
            let selected = match block {
                Ok(selected) => selected,
                Err(reason) => {
                    // A fetch or decode fault of a guest instruction may be
                    // caught by the guest's own exception handlers.
                    let dispatched = crate::exceptions::dispatch_fault(reason, true, cpu, memory);
                    self.advance_epoch();
                    missed_slot = std::ptr::null_mut();
                    let exit = match dispatched {
                        Ok(()) => {
                            instructions_count += 1;
                            continue;
                        }
                        Err(stop) => stop,
                    };
                    return ExecutionResult {
                        instructions_executed: instructions_count,
                        final_state: cpu.clone(),
                        exit_reason: exit,
                    };
                }
            };
            let slot = std::mem::replace(&mut missed_slot, std::ptr::null_mut());
            self.install_link(slot, &selected, cpu.rip);

            // Execute native ARM64 translated block!
            // Signature: extern "C" fn(cpu_state_ptr: *mut u64, guest_memory_ctx: *mut (GuestMemory, ExitReason)) -> u32
            // Returns: number of guest instructions executed in the block
            // that returned. Each link followed before it used up the
            // length of the block it left from `BlockContext::headroom`.
            type BlockFn = unsafe extern "C" fn(*mut u64, *mut BlockContext) -> u32;
            let block_fn: BlockFn = unsafe { std::mem::transmute(selected.code) };

            // After a block, the dispatcher would select a full block again
            // only with a whole block of budget left for `run`, and with any
            // budget left for a slice.
            let remaining = max_instructions - instructions_count;
            let headroom = match (self.linking, exact) {
                (false, _) => 0,
                (true, true) => remaining.saturating_sub(MAX_BLOCK_INSTRUCTIONS),
                (true, false) => remaining - 1,
            } as u64;
            let mut ctx = BlockContext {
                memory_ptr: memory as *mut GuestMemory,
                faulted: 0,
                epoch: self.epoch,
                headroom,
                links_followed: 0,
                missed_slot: 0,
                code_start: self.code_start,
                code_end: self.code_end,
                exit_reason: None,
                helper_instructions: 0,
            };

            self.metrics.dbt_dispatches += 1;
            let cpu_raw_ptr = cpu as *mut CpuState as *mut u64;
            let insns_in_block = unsafe { block_fn(cpu_raw_ptr, &mut ctx) };
            let executed = (headroom - ctx.headroom) as usize + insns_in_block as usize;
            instructions_count += executed;
            self.metrics.instructions_executed += executed;
            self.metrics.interpreter_fallbacks += ctx.helper_instructions as usize;
            self.link_metrics.links_followed += ctx.links_followed as usize;
            if ctx.epoch != self.epoch || ctx.exit_reason.is_some() {
                // Guest code may have changed, or a fault dispatch may write
                // guest memory next.
                self.advance_epoch();
            }

            if let Some(exit) = ctx.exit_reason {
                // A guest instruction's fault may be caught by the guest's
                // own exception handlers; RIP is at that instruction.
                let exit = match crate::exceptions::dispatch_fault(exit, false, cpu, memory) {
                    Ok(()) => continue,
                    Err(stop) => stop,
                };
                return ExecutionResult {
                    instructions_executed: instructions_count,
                    final_state: cpu.clone(),
                    exit_reason: exit,
                };
            }
            if ctx.missed_slot != 0 {
                self.link_metrics.link_fallbacks += 1;
                missed_slot = ctx.missed_slot as *mut LinkSlot;
            }
        }

        ExecutionResult {
            instructions_executed: instructions_count,
            final_state: cpu.clone(),
            exit_reason: ExitReason::StepLimitExceeded,
        }
    }

    /// Point `slot`, which a block's exit stub could not follow, at the block
    /// the dispatcher then selected, when that is a full block for the
    /// slot's own successor address.
    fn install_link(&mut self, slot: *mut LinkSlot, selected: &SelectedBlock, rip: u64) {
        if slot.is_null() || selected.links.is_null() || !self.linking {
            return;
        }
        // SAFETY: slots and targets are owned by `self.links` and outlive
        // this call; no translated code runs while the dispatcher holds them.
        unsafe {
            if (*slot).rip != rip || (*selected.links).target.start != rip {
                return;
            }
            let target = std::ptr::addr_of!((*selected.links).target) as u64;
            if (*slot).target != target {
                (*slot).target = target;
                self.link_metrics.links_installed += 1;
            }
        }
    }

    fn get_or_compile_block(
        &mut self,
        rip: u64,
        memory: &GuestMemory,
        block_limit: usize,
    ) -> Result<SelectedBlock, ExitReason> {
        if block_limit == MAX_BLOCK_INSTRUCTIONS {
            if let Some(cached) = self.block_cache.get(&rip) {
                if cached.is_current(memory) {
                    self.metrics.cache_hits += 1;
                    self.link_metrics.bytes_verified += cached.source.len();
                    if !cached.links.is_null() {
                        // SAFETY: owned by `self.links`; no translated code
                        // runs while the dispatcher selects a block.
                        unsafe { (*cached.links).target.verified_epoch = self.epoch };
                    }
                    return Ok(SelectedBlock {
                        code: cached.code,
                        links: cached.links,
                    });
                }
                // Blocks linked to the old translation fall back to the
                // dispatcher from now on. A stale block stays cached when its
                // new translation fails, so it is retired only once.
                // SAFETY: as above.
                if !cached.links.is_null()
                    && unsafe { (*cached.links).target.verified_epoch } != RETIRED_EPOCH
                {
                    unsafe { (*cached.links).target.verified_epoch = RETIRED_EPOCH };
                    self.link_metrics.targets_retired += 1;
                }
                self.metrics.blocks_invalidated += 1;
            }
            self.metrics.cache_misses += 1;
            let compiled = self.compile_block(rip, memory, block_limit)?;
            let selected = SelectedBlock {
                code: compiled.code,
                links: compiled.links,
            };
            self.block_cache.insert(rip, compiled);
            return Ok(selected);
        }

        // Keep budget-limited blocks separate so they cannot shorten a later run.
        let key = (rip, block_limit);
        if let Some(cached) = self.limited_block_cache.get(&key) {
            if cached.is_current(memory) {
                self.metrics.cache_hits += 1;
                self.link_metrics.bytes_verified += cached.source.len();
                return Ok(SelectedBlock {
                    code: cached.code,
                    links: std::ptr::null_mut(),
                });
            }
            self.metrics.blocks_invalidated += 1;
        }
        self.metrics.cache_misses += 1;
        let compiled = self.compile_block(rip, memory, block_limit)?;
        let selected = SelectedBlock {
            code: compiled.code,
            links: std::ptr::null_mut(),
        };
        self.limited_block_cache.insert(key, compiled);
        Ok(selected)
    }

    fn compile_block(
        &mut self,
        start_rip: u64,
        memory: &GuestMemory,
        block_limit: usize,
    ) -> Result<CachedBlock, ExitReason> {
        // Decode the whole block first: lowering needs to know whether a later
        // instruction observes the flags an earlier one produces.
        let mut decoded = Vec::with_capacity(block_limit);
        let mut rip = start_rip;
        while decoded.len() < block_limit {
            let insn = decode(rip, |address| memory.fetch_instruction_byte(address))
                .map_err(ExitReason::from)?;
            if !is_translatable(&insn) {
                return Err(ExitReason::IllegalInstruction(0x05));
            }
            rip = insn.next_rip();
            decoded.push(insn);
            if insn.op.is_terminator() {
                break;
            }
        }
        // Emitted code holds pointers to these instructions for helper calls,
        // so the allocation lives as long as the engine's code cache.
        let insns = decoded.into_boxed_slice();

        // Reuse the emission buffer across cold block translations.
        let asm = &mut self.assembler;
        asm.clear();

        // Function Prologue:
        // Inputs: X0 = &mut CpuState.gpr[0], X1 = &mut BlockContext
        // We preserve host callee-saved registers X19, X20, X21, X29 (FP), X30 (LR)
        asm.stp_pre_64(X29, X30, SP, -48);
        asm.str_uoff_64(X19, SP, 16);
        asm.str_uoff_64(X20, SP, 24);
        asm.str_uoff_64(X21, SP, 32);

        // Keep CPU GPR pointer in callee-saved X19
        asm.mov_reg_64(X19, X0);
        // Keep BlockContext pointer in callee-saved X20
        asm.mov_reg_64(X20, X1);
        // A link enters here, in the frame and with the X19 and X20 of the
        // first block of its chain. Every block has the same prologue and
        // epilogue, so any of them can return from that frame.
        let body = asm.current_offset();

        let mut emitter = Emitter {
            asm,
            fault_guards: Vec::new(),
        };
        for (index, insn) in insns.iter().enumerate() {
            let flags = flags_live_after(&insns, index);
            if is_native(insn) {
                emitter.lower(insn, index, flags);
            } else {
                emitter.call_semantics(insn, index);
            }
        }
        let Emitter { asm, fault_guards } = emitter;

        if !insns.last().is_some_and(|insn| insn.op.is_terminator()) {
            // End of basic block quota, update RIP.
            asm.load_imm_64(X9, rip);
            asm.str_uoff_64(X9, X19, RIP_OFFSET);
        }

        // Only full blocks link, from and to: a budget-limited block is
        // selected only when less than a full block of budget is left. Every
        // full block can be a target, whatever its own last instruction.
        let links = if block_limit == MAX_BLOCK_INSTRUCTIONS {
            let links = self.links.allocate(start_rip);
            let (successors, count) = static_successors(&insns, rip);
            if count > 0 {
                emit_link_exit(asm, links, &successors[..count], insns.len() as u64);
            }
            links
        } else {
            std::ptr::null_mut()
        };

        // Return number of guest instructions executed in this block in X0
        asm.load_imm_64(X0, insns.len() as u64);

        // Epilogue: faulting paths jump here with X0 already set to the number
        // of instructions that completed before the fault.
        let epilogue = asm.current_offset();
        for index in fault_guards {
            asm.patch_b(index, epilogue as i32 - index as i32);
        }
        asm.ldr_uoff_64(X19, SP, 16);
        asm.ldr_uoff_64(X20, SP, 24);
        asm.ldr_uoff_64(X21, SP, 32);
        asm.ldp_post_64(X29, X30, SP, 48);
        asm.ret();

        let emitted = self
            .jit
            .emit_block(asm.instructions())
            .map_err(|_| ExitReason::JitCacheExhausted)?;
        self.metrics.blocks_compiled += 1;
        self.metrics.native_bytes_emitted += asm.instructions().len() * 4;
        // Decoding just read every one of these bytes, so they are readable.
        let source = (0..rip.wrapping_sub(start_rip))
            .map(|offset| memory.fetch_instruction_byte(start_rip.wrapping_add(offset)))
            .collect::<Result<Box<[u8]>, _>>()
            .map_err(ExitReason::MemoryFault)?;
        self.decoded_blocks.push(insns);
        // Stores into these bytes must end a chain from now on.
        self.code_start = self.code_start.min(start_rip);
        self.code_end = self.code_end.max(rip);
        if !links.is_null() {
            // SAFETY: owned by `self.links`; the emitted code is not running.
            unsafe {
                (*links).target.body = emitted as u64 + body as u64 * 4;
                (*links).target.verified_epoch = self.epoch;
            }
        }
        Ok(CachedBlock {
            code: emitted,
            start: start_rip,
            source,
            links,
        })
    }
}

/// Guest addresses a block's last instruction passes control to that are
/// known when it is translated, in the order its exit stub tests them, and
/// how many there are. `next` is the address after the block.
fn static_successors(insns: &[Insn], next: u64) -> ([u64; 2], usize) {
    let Some(last) = insns.last() else {
        return ([0; 2], 0);
    };
    if !last.op.is_terminator() {
        // The block ended at the instruction quota.
        return ([next, 0], 1);
    }
    if !is_native(last) || last.encoding != Encoding::Rel {
        // RET, indirect branches and terminators run by the shared
        // semantics go through the dispatcher.
        return ([0; 2], 0);
    }
    match last.op {
        Op::Jmp | Op::Call => ([last.imm, 0], 1),
        Op::Jcc if last.imm == next => ([next, 0], 1),
        Op::Jcc => ([last.imm, next], 2),
        _ => ([0; 2], 0),
    }
}

/// Emit the exit stub of a full block with static successors. It follows the
/// link of the successor that the guest RIP names when the conditions in the
/// [`DbtEngine`] documentation hold. Otherwise it records that slot for the
/// dispatcher and falls through to the block's ordinary return of `len`.
fn emit_link_exit(asm: &mut Arm64Assembler, links: *mut BlockLinks, successors: &[u64], len: u64) {
    // SAFETY: `links` is live and nothing else refers to its slots yet.
    let slots = unsafe { &mut (*links).slots };
    asm.ldr_uoff_64(X9, X19, RIP_OFFSET);
    // With two successors, the first (a Jcc's target) is tested and the
    // second is taken otherwise. The start check below still requires the
    // target to begin at the guest RIP.
    let mut to_check = None;
    for (index, (&rip, slot)) in successors.iter().zip(slots.iter_mut()).enumerate() {
        slot.rip = rip;
        let address = std::ptr::from_mut(slot) as u64;
        if index + 1 < successors.len() {
            asm.load_imm_64(X10, rip);
            asm.cmp_reg_64(X9, X10);
            let mismatch = asm.current_offset();
            asm.b_cond(Condition::Ne, 0);
            asm.load_imm_64(X11, address);
            to_check = Some(asm.current_offset());
            asm.b(0);
            let next = asm.current_offset();
            asm.patch_b_cond(mismatch, Condition::Ne, next as i32 - mismatch as i32);
        } else {
            asm.load_imm_64(X11, address);
        }
    }
    // X9 = guest RIP, X11 = the slot of that successor.
    let check = asm.current_offset();
    if let Some(branch) = to_check {
        asm.patch_b(branch, check as i32 - branch as i32);
    }
    asm.ldr_uoff_64(X12, X11, SLOT_TARGET_OFFSET);
    let no_target = asm.current_offset();
    asm.cbz_64(X12, 0);
    // A stopped chain has STOPPED_EPOCH, in which no target is verified.
    asm.ldr_uoff_64(X13, X12, TARGET_EPOCH_OFFSET);
    asm.ldr_uoff_64(X14, X20, BLOCK_CTX_EPOCH_OFFSET);
    asm.cmp_reg_64(X13, X14);
    let unverified = asm.current_offset();
    asm.b_cond(Condition::Ne, 0);
    asm.ldr_uoff_64(X13, X12, TARGET_START_OFFSET);
    asm.cmp_reg_64(X13, X9);
    let elsewhere = asm.current_offset();
    asm.b_cond(Condition::Ne, 0);
    // headroom - len, refusing to wrap.
    asm.ldr_uoff_64(X13, X20, BLOCK_CTX_HEADROOM_OFFSET);
    asm.load_imm_64(X14, len);
    asm.subs_reg_64(X13, X13, X14);
    let over_budget = asm.current_offset();
    asm.b_cond(Condition::Cc, 0);
    // Follow the link: charge this block, then enter the target.
    asm.str_uoff_64(X13, X20, BLOCK_CTX_HEADROOM_OFFSET);
    asm.ldr_uoff_64(X13, X20, BLOCK_CTX_LINKS_FOLLOWED_OFFSET);
    asm.add_imm_64(X13, X13, 1);
    asm.str_uoff_64(X13, X20, BLOCK_CTX_LINKS_FOLLOWED_OFFSET);
    asm.ldr_uoff_64(X13, X12, TARGET_BODY_OFFSET);
    asm.br(X13);
    let exit = asm.current_offset();
    asm.patch_cbz_64(no_target, X12, exit as i32 - no_target as i32);
    for (branch, condition) in [
        (unverified, Condition::Ne),
        (elsewhere, Condition::Ne),
        (over_budget, Condition::Cc),
    ] {
        asm.patch_b_cond(branch, condition, exit as i32 - branch as i32);
    }
    asm.str_uoff_64(X11, X20, BLOCK_CTX_MISSED_SLOT_OFFSET);
}

const RIP_OFFSET: u32 = std::mem::offset_of!(CpuState, rip) as u32;
const FS_BASE_OFFSET: u32 = std::mem::offset_of!(CpuState, fs_base) as u32;
const GS_BASE_OFFSET: u32 = std::mem::offset_of!(CpuState, gs_base) as u32;
const FLAGS_OFFSET: u32 = std::mem::offset_of!(CpuState, flags) as u32;
const CF: u32 = FLAGS_OFFSET + std::mem::offset_of!(RFlags, cf) as u32;
const ZF: u32 = FLAGS_OFFSET + std::mem::offset_of!(RFlags, zf) as u32;
const SF: u32 = FLAGS_OFFSET + std::mem::offset_of!(RFlags, sf) as u32;
const OF: u32 = FLAGS_OFFSET + std::mem::offset_of!(RFlags, of) as u32;
const PF: u32 = FLAGS_OFFSET + std::mem::offset_of!(RFlags, pf) as u32;
const AF: u32 = FLAGS_OFFSET + std::mem::offset_of!(RFlags, af) as u32;
const ALL_FLAGS: u8 = 0x3F;

const NEON_EOR: u32 = 0x6e20_1c00; // EOR Vd.16B, Vn.16B, Vm.16B
const NEON_BSL: u32 = 0x6e60_1c00; // BSL Vd.16B, Vn.16B, Vm.16B
// SXTL and UXTL of the low lanes; the lane width in bits goes at bit 16.
const NEON_SXTL: u32 = 0x0f00_a400;
const NEON_UXTL: u32 = 0x2f00_a400;

fn xmm_offset(index: u8) -> u32 {
    std::mem::offset_of!(CpuState, xmm) as u32 + u32::from(index) * 16
}

fn reg_offset(index: u8) -> u32 {
    u32::from(index) * 8
}

/// Whether the DBT runs an instruction at all. It dispatches guest services only
/// through registered imports, so a raw SYSCALL is refused.
pub fn is_translatable(insn: &Insn) -> bool {
    insn.op != Op::Syscall
}

/// Whether an instruction is lowered to ARM64. Every other decoded instruction
/// calls the shared semantics and is counted in
/// [`DbtMetrics::interpreter_fallbacks`].
pub fn is_native(insn: &Insn) -> bool {
    // Locked read-modify-writes stay on the shared semantics, the one place
    // atomicity has to be provided once guest threads run concurrently.
    // So do VEX-encoded forms, whose operation may be one that is lowered.
    if insn.lock || insn.vex.is_some() {
        return false;
    }
    let wide = matches!(insn.size, 4 | 8);
    // The bit scans work on any register or memory source, and the bit test
    // needs the index in a register.
    let scanned = matches!(insn.size, 2 | 4 | 8);
    match insn.op {
        Op::Add | Op::Sub | Op::Cmp | Op::And | Op::Or | Op::Xor | Op::Test => wide,
        // Register and immediate forms only; a memory destination would have
        // to restore the guest carry again after the store helper call.
        Op::Adc | Op::Sbb => wide && insn.memory().is_none(),
        Op::Inc | Op::Dec | Op::Neg | Op::Not => wide,
        Op::Cmov => wide,
        Op::Bsf | Op::Bsr => scanned,
        // Only the `0F A3` form has the bit index in a register. The `0F BA`
        // group form takes an immediate index and may write memory, so it
        // stays on the shared semantics.
        Op::Bt => scanned && insn.encoding == Encoding::RmReg,
        Op::Pop => insn.encoding == Encoding::OpReg,
        Op::Mov
        | Op::Movzx
        | Op::Movsx
        | Op::Movsxd
        | Op::Lea
        | Op::Bswap
        | Op::Setcc
        | Op::Push
        | Op::Call
        | Op::Jmp
        | Op::Jcc
        | Op::Ret
        | Op::Nop
        | Op::Hlt => true,
        Op::Movaps
        | Op::Movups
        | Op::Movdqa
        | Op::Movdqu
        | Op::Movntdq
        | Op::Movntdqa
        | Op::Movntps
        | Op::Pblendvb
        | Op::Blendvps
        | Op::Blendvpd
        | Op::Movd
        | Op::Movq
        | Op::Movss
        | Op::Movsd
        | Op::Movlps
        | Op::Movhps => true,
        op if insn.encoding == Encoding::XmmRm => {
            neon_operation(op).is_some() || neon_widening(op).is_some()
        }
        Op::Psllw
        | Op::Pslld
        | Op::Psllq
        | Op::Psrlw
        | Op::Psrld
        | Op::Psrlq
        | Op::Psraw
        | Op::Psrad => insn.encoding == Encoding::RmImmXmm,
        _ => false,
    }
}

/// NEON base word for a packed operation with identical lane semantics, and
/// whether its sources are swapped (BIC is `Vn & !Vm`, PANDN is `!dst & src`).
#[rustfmt::skip]
fn neon_operation(op: Op) -> Option<(u32, bool)> {
    Some(match op {
        Op::Pand => (0x4e20_1c00, false), // and.16b
        Op::Por => (0x4ea0_1c00, false), // orr.16b
        Op::Pxor => (0x6e20_1c00, false), // eor.16b
        Op::Pandn => (0x4e60_1c00, true), // bic.16b
        Op::Paddb => (0x4e20_8400, false), // add.16b
        Op::Paddw => (0x4e60_8400, false), // add.8h
        Op::Paddd => (0x4ea0_8400, false), // add.4s
        Op::Paddq => (0x4ee0_8400, false), // add.2d
        Op::Psubb => (0x6e20_8400, false), // sub.16b
        Op::Psubw => (0x6e60_8400, false), // sub.8h
        Op::Psubd => (0x6ea0_8400, false), // sub.4s
        Op::Psubq => (0x6ee0_8400, false), // sub.2d
        Op::Paddsb => (0x4e20_0c00, false), // sqadd.16b
        Op::Paddsw => (0x4e60_0c00, false), // sqadd.8h
        Op::Psubsb => (0x4e20_2c00, false), // sqsub.16b
        Op::Psubsw => (0x4e60_2c00, false), // sqsub.8h
        Op::Paddusb => (0x6e20_0c00, false), // uqadd.16b
        Op::Paddusw => (0x6e60_0c00, false), // uqadd.8h
        Op::Psubusb => (0x6e20_2c00, false), // uqsub.16b
        Op::Psubusw => (0x6e60_2c00, false), // uqsub.8h
        Op::Pcmpeqb => (0x6e20_8c00, false), // cmeq.16b
        Op::Pcmpeqw => (0x6e60_8c00, false), // cmeq.8h
        Op::Pcmpeqd => (0x6ea0_8c00, false), // cmeq.4s
        Op::Pcmpgtb => (0x4e20_3400, false), // cmgt.16b
        Op::Pcmpgtw => (0x4e60_3400, false), // cmgt.8h
        Op::Pcmpgtd => (0x4ea0_3400, false), // cmgt.4s
        Op::Pminub => (0x6e20_6c00, false), // umin.16b
        Op::Pmaxub => (0x6e20_6400, false), // umax.16b
        Op::Pminsw => (0x4e60_6c00, false), // smin.8h
        Op::Pmaxsw => (0x4e60_6400, false), // smax.8h
        Op::Pavgb => (0x6e20_1400, false), // urhadd.16b
        Op::Pavgw => (0x6e60_1400, false), // urhadd.8h
        Op::Pmullw => (0x4e60_9c00, false), // mul.8h
        Op::Pminsb => (0x4e20_6c00, false), // smin.16b
        Op::Pminsd => (0x4ea0_6c00, false), // smin.4s
        Op::Pminuw => (0x6e60_6c00, false), // umin.8h
        Op::Pminud => (0x6ea0_6c00, false), // umin.4s
        Op::Pmaxsb => (0x4e20_6400, false), // smax.16b
        Op::Pmaxsd => (0x4ea0_6400, false), // smax.4s
        Op::Pmaxuw => (0x6e60_6400, false), // umax.8h
        Op::Pmaxud => (0x6ea0_6400, false), // umax.4s
        Op::Pmulld => (0x4ea0_9c00, false), // mul.4s
        Op::Pcmpeqq => (0x6ee0_8c00, false), // cmeq.2d
        Op::Pcmpgtq => (0x4ee0_3400, false), // cmgt.2d
        Op::Punpcklbw => (0x4e00_3800, false), // zip1.16b
        Op::Punpcklwd => (0x4e40_3800, false), // zip1.8h
        Op::Punpckldq => (0x4e80_3800, false), // zip1.4s
        Op::Punpcklqdq => (0x4ec0_3800, false), // zip1.2d
        Op::Punpckhbw => (0x4e00_7800, false), // zip2.16b
        Op::Punpckhwd => (0x4e40_7800, false), // zip2.8h
        Op::Punpckhdq => (0x4e80_7800, false), // zip2.4s
        Op::Punpckhqdq => (0x4ec0_7800, false), // zip2.2d
        _ => return None,
    })
}

/// Whether a PMOVSX or PMOVZX extends the sign, and the lane widths to pass
/// through, each doubling by one SXTL or UXTL.
fn neon_widening(op: Op) -> Option<(bool, &'static [u32])> {
    Some(match op {
        Op::Pmovsxbw => (true, &[8]),
        Op::Pmovsxbd => (true, &[8, 16]),
        Op::Pmovsxbq => (true, &[8, 16, 32]),
        Op::Pmovsxwd => (true, &[16]),
        Op::Pmovsxwq => (true, &[16, 32]),
        Op::Pmovsxdq => (true, &[32]),
        Op::Pmovzxbw => (false, &[8]),
        Op::Pmovzxbd => (false, &[8, 16]),
        Op::Pmovzxbq => (false, &[8, 16, 32]),
        Op::Pmovzxwd => (false, &[16]),
        Op::Pmovzxwq => (false, &[16, 32]),
        Op::Pmovzxdq => (false, &[32]),
        _ => return None,
    })
}

/// Whether the flags produced by `insns[index]` can be observed. They cannot if
/// a later instruction in the block overwrites all of them before anything reads
/// them or could leave the block.
fn flags_live_after(insns: &[Insn], index: usize) -> bool {
    for insn in &insns[index + 1..] {
        let can_exit = insn.op.is_terminator()
            || insn.memory().is_some()
            || matches!(insn.op, Op::Push | Op::Pop | Op::Leave)
            || insn.op.can_fault();
        if can_exit || insn.op.flags_read() != 0 {
            return true;
        }
        if insn.op.flags_written() == ALL_FLAGS {
            return false;
        }
    }
    true
}

/// Emits one block. X19 holds the CPU state and X20 the block context; X21 is
/// the only guest value kept across helper calls.
struct Emitter<'a> {
    asm: &'a mut Arm64Assembler,
    fault_guards: Vec<usize>,
}

impl Emitter<'_> {
    fn call_helper(&mut self, helper: u64) {
        self.asm.mov_reg_64(X0, X20);
        self.asm.load_imm_64(X4, helper);
        self.asm.blr(X4);
    }

    fn guard(&mut self, insn: &Insn, completed: usize) {
        emit_fault_guard(self.asm, &mut self.fault_guards, completed, insn.rip);
    }

    /// Execute through the shared semantics.
    fn call_semantics(&mut self, insn: &Insn, completed: usize) {
        self.asm.mov_reg_64(X1, X19);
        self.asm
            .load_imm_64(X2, std::ptr::from_ref(insn) as usize as u64);
        self.call_helper(execute_helper as *const () as u64);
        self.guard(insn, completed);
    }

    /// Load a register operand into `xd`, with AH/CH/DH/BH in bits 7:0. Bits
    /// above the operand width are not cleared.
    fn load_reg(&mut self, xd: u8, reg: RegOperand) {
        self.asm.ldr_uoff_64(xd, X19, reg_offset(reg.index));
        if reg.high {
            self.asm.lsr_imm_64(xd, xd, 8);
        }
    }

    /// Store `xs` (not X13 or X14) with x86 partial-register rules.
    fn store_reg(&mut self, reg: RegOperand, size: u8, xs: u8) {
        let offset = reg_offset(reg.index);
        match size {
            8 => self.asm.str_uoff_64(xs, X19, offset),
            4 => {
                self.asm.mov_reg_32(X13, xs);
                self.asm.str_uoff_64(X13, X19, offset);
            }
            2 => {
                self.asm.uxth_32(X13, xs);
                self.asm.ldr_uoff_64(X14, X19, offset);
                self.asm.load_imm_64(X15, !0xFFFF);
                self.asm.and_reg_64(X14, X14, X15);
                self.asm.orr_reg_64(X13, X13, X14);
                self.asm.str_uoff_64(X13, X19, offset);
            }
            _ => self.asm.strb_uoff(xs, X19, offset + u32::from(reg.high)),
        }
    }

    /// Zero-extend the low `size` bytes of `xd` to the full 64-bit register.
    ///
    /// An operation narrower than the register reads only the low bytes of its
    /// source, so the bits above them must not take part in a bit scan.
    fn zero_extend(&mut self, xd: u8, size: u8) {
        match size {
            8 => {}
            4 => self.asm.mov_reg_32(xd, xd),
            _ => self.asm.uxth_32(xd, xd),
        }
    }

    /// Effective address into X1, clobbering X9.
    fn effective_address(&mut self, operand: &MemOperand, with_segment: bool) {
        let asm = &mut *self.asm;
        match operand.base {
            Some(base) => {
                asm.ldr_uoff_64(X1, X19, reg_offset(base));
                let disp = operand.disp;
                if (1..=4095).contains(&disp) {
                    asm.add_imm_64(X1, X1, disp as u16);
                } else if (-4095..=-1).contains(&disp) {
                    asm.sub_imm_64(X1, X1, -disp as u16);
                } else if disp != 0 {
                    asm.load_imm_64(X9, disp as u64);
                    asm.add_reg_64(X1, X1, X9);
                }
            }
            None => asm.load_imm_64(X1, operand.disp as u64),
        }
        if let Some((index, scale)) = operand.index {
            asm.ldr_uoff_64(X9, X19, reg_offset(index));
            if scale > 0 {
                asm.add_reg_shifted_64(X1, X1, X9, scale);
            } else {
                asm.add_reg_64(X1, X1, X9);
            }
        }
        if with_segment && let Some(segment) = operand.segment {
            let offset = match segment {
                Segment::Fs => FS_BASE_OFFSET,
                Segment::Gs => GS_BASE_OFFSET,
            };
            asm.ldr_uoff_64(X9, X19, offset);
            asm.add_reg_64(X1, X1, X9);
        }
    }

    /// Load the ModRM operand into `xd`. A register is loaded whole; a memory
    /// operand is zero-extended.
    fn load_rm(&mut self, xd: u8, insn: &Insn, completed: usize) {
        match insn.rm {
            Some(RmOperand::Reg(reg)) => self.load_reg(xd, reg),
            Some(RmOperand::Mem(operand)) => {
                self.effective_address(&operand, true);
                self.call_helper(load_helper(insn.rm_size));
                self.guard(insn, completed);
                self.asm.mov_reg_64(xd, X0);
            }
            None => {}
        }
    }

    /// Store `xs` to the ModRM operand. A memory store leaves no other effect
    /// behind if it faults, so callers commit registers and flags afterwards.
    fn store_rm(&mut self, insn: &Insn, xs: u8, completed: usize) {
        match insn.rm {
            Some(RmOperand::Reg(reg)) => self.store_reg(reg, insn.rm_size, xs),
            Some(RmOperand::Mem(operand)) => {
                self.asm.mov_reg_64(X2, xs);
                self.effective_address(&operand, true);
                self.call_helper(store_helper(insn.rm_size));
                self.guard(insn, completed);
            }
            None => {}
        }
    }

    /// X11 = X9 op X10, setting host NZCV only for the arithmetic operations.
    /// ADC and SBB also read the guest CF and clobber X12 and X13.
    fn compute(&mut self, op: Op, wide: bool) {
        let asm = &mut *self.asm;
        if matches!(op, Op::Adc | Op::Sbb) {
            // ADCS adds the host C, so C must equal the guest CF. SBCS
            // subtracts NOT C, while x86 SBB subtracts the borrow CF, so there
            // C must be its inverse. A compare sets C when it does not borrow.
            asm.ldrb_uoff(X12, X19, CF);
            if op == Op::Adc {
                asm.load_imm_64(X13, 1);
                asm.cmp_reg_32(X12, X13); // C = CF >= 1
            } else {
                asm.cmp_reg_32(XZR, X12); // C = 0 >= CF
            }
        }
        match (op, wide) {
            (Op::Adc, true) => asm.adcs_reg_64(X11, X9, X10),
            (Op::Adc, false) => asm.adcs_reg_32(X11, X9, X10),
            (Op::Sbb, true) => asm.sbcs_reg_64(X11, X9, X10),
            (Op::Sbb, false) => asm.sbcs_reg_32(X11, X9, X10),
            (Op::Add | Op::Inc, true) => asm.adds_reg_64(X11, X9, X10),
            (Op::Add | Op::Inc, false) => asm.adds_reg_32(X11, X9, X10),
            (Op::And | Op::Test, true) => asm.ands_reg_64(X11, X9, X10),
            (Op::And | Op::Test, false) => asm.ands_reg_32(X11, X9, X10),
            (Op::Or, true) => asm.orr_reg_64(X11, X9, X10),
            (Op::Or, false) => asm.orr_reg_32(X11, X9, X10),
            (Op::Xor, _) => {
                asm.eor_reg_64(X11, X9, X10);
                if !wide {
                    asm.mov_reg_32(X11, X11);
                }
            }
            (_, true) => asm.subs_reg_64(X11, X9, X10),
            (_, false) => asm.subs_reg_32(X11, X9, X10),
        }
    }

    /// Store the guest flags of `X11 = X9 op X10` after [`Self::compute`].
    fn store_flags(&mut self, op: Op, wide: bool) {
        let asm = &mut *self.asm;
        let logical = matches!(op, Op::And | Op::Or | Op::Xor | Op::Test);
        if logical {
            if wide {
                asm.tst_reg_64(X11, X11);
            } else {
                asm.tst_reg_32(X11, X11);
            }
            asm.strb_uoff(XZR, X19, CF);
            asm.strb_uoff(XZR, X19, OF);
        } else {
            match op {
                // INC and DEC leave CF unchanged.
                Op::Inc | Op::Dec => {}
                Op::Add | Op::Adc => {
                    asm.cset_32(X12, Condition::Cs);
                    asm.strb_uoff(X12, X19, CF);
                }
                // ARM C is set when a subtraction does not borrow; x86 CF is
                // the borrow itself.
                _ => {
                    asm.cset_32(X12, Condition::Cc);
                    asm.strb_uoff(X12, X19, CF);
                }
            }
            asm.cset_32(X12, Condition::Vs);
            asm.strb_uoff(X12, X19, OF);
        }
        asm.cset_32(X12, Condition::Eq);
        asm.strb_uoff(X12, X19, ZF);
        asm.cset_32(X12, Condition::Mi);
        asm.strb_uoff(X12, X19, SF);
        if !logical {
            // AF is the carry into bit 4: bit 4 of a ^ b ^ result.
            asm.eor_lsr_32(X12, X9, X10, 0);
            asm.eor_lsr_32(X12, X12, X11, 0);
            asm.ubfx_bit_32(X12, X12, 4);
            asm.strb_uoff(X12, X19, AF);
        }
        // PF is set when the low result byte has an even number of ones.
        asm.eor_lsr_32(X12, X11, X11, 4);
        asm.eor_lsr_32(X12, X12, X12, 2);
        asm.eor_lsr_32(X12, X12, X12, 1);
        asm.load_imm_64(X13, 1);
        asm.bic_reg_32(X12, X13, X12);
        asm.strb_uoff(X12, X19, PF);
    }

    /// W12 = 1 if the guest condition holds, else 0. Clobbers X13 and X14.
    fn load_condition(&mut self, code: u8) {
        let asm = &mut *self.asm;
        match code >> 1 {
            0 => asm.ldrb_uoff(X12, X19, OF),
            1 => asm.ldrb_uoff(X12, X19, CF),
            2 => asm.ldrb_uoff(X12, X19, ZF),
            3 => {
                asm.ldrb_uoff(X12, X19, CF);
                asm.ldrb_uoff(X13, X19, ZF);
                asm.orr_reg_32(X12, X12, X13);
            }
            4 => asm.ldrb_uoff(X12, X19, SF),
            5 => asm.ldrb_uoff(X12, X19, PF),
            other => {
                asm.ldrb_uoff(X12, X19, SF);
                asm.ldrb_uoff(X13, X19, OF);
                asm.eor_lsr_32(X12, X12, X13, 0);
                if other == 7 {
                    asm.ldrb_uoff(X13, X19, ZF);
                    asm.orr_reg_32(X12, X12, X13);
                }
            }
        }
        if code & 1 != 0 {
            asm.load_imm_64(X13, 1);
            asm.eor_lsr_32(X12, X12, X13, 0);
        }
    }

    /// Push `X2` on the guest stack, committing RSP only after the store.
    fn push(&mut self, insn: &Insn, completed: usize) {
        self.asm
            .ldr_uoff_64(X1, X19, reg_offset(Register::Rsp as u8));
        self.call_helper(push_helper as *const () as u64);
        self.guard(insn, completed);
        self.asm
            .str_uoff_64(X0, X19, reg_offset(Register::Rsp as u8));
    }

    /// Read the guest stack top into X0.
    fn read_stack(&mut self, insn: &Insn, completed: usize) {
        self.asm
            .ldr_uoff_64(X1, X19, reg_offset(Register::Rsp as u8));
        self.call_helper(load_helper(8));
        self.guard(insn, completed);
    }

    fn release_stack(&mut self, bytes: u64) {
        let offset = reg_offset(Register::Rsp as u8);
        self.asm.ldr_uoff_64(X10, X19, offset);
        self.asm.load_imm_64(X9, bytes);
        self.asm.add_reg_64(X10, X10, X9);
        self.asm.str_uoff_64(X10, X19, offset);
    }

    /// Branch target into `xd`.
    fn load_target(&mut self, xd: u8, insn: &Insn, completed: usize) {
        if insn.encoding == Encoding::Rel {
            self.asm.load_imm_64(xd, insn.imm);
        } else {
            self.load_rm(xd, insn, completed);
        }
    }

    /// Source operand of a two-operand ALU form into X10; memory sources of
    /// `reg, r/m` forms are loaded by the caller.
    fn load_source(&mut self, insn: &Insn) {
        match insn.encoding {
            Encoding::RmReg => self.load_reg(X10, insn.reg),
            _ => {
                let imm = match insn.op {
                    Op::Inc | Op::Dec => 1,
                    _ => insn.imm,
                };
                self.asm.load_imm_64(X10, imm);
            }
        }
    }

    fn alu(&mut self, insn: &Insn, completed: usize, flags: bool) {
        let wide = insn.size == 8;
        let writes = !matches!(insn.op, Op::Cmp | Op::Test);
        match insn.encoding {
            Encoding::RegRm => {
                self.load_rm(X10, insn, completed);
                self.load_reg(X9, insn.reg);
                self.compute(insn.op, wide);
                if flags {
                    self.store_flags(insn.op, wide);
                }
                if writes {
                    self.store_reg(insn.reg, insn.size, X11);
                }
            }
            Encoding::AccImm => {
                self.load_reg(X9, insn.reg);
                self.load_source(insn);
                self.compute(insn.op, wide);
                if flags {
                    self.store_flags(insn.op, wide);
                }
                if writes {
                    self.store_reg(insn.reg, insn.size, X11);
                }
            }
            _ => {
                self.load_rm(X9, insn, completed);
                self.load_source(insn);
                if insn.memory().is_some() && writes {
                    // Keep the old value across the store, then recompute the
                    // flags so a faulting store leaves them unchanged.
                    self.asm.mov_reg_64(X21, X9);
                    self.compute(insn.op, wide);
                    self.store_rm(insn, X11, completed);
                    if flags {
                        self.asm.mov_reg_64(X9, X21);
                        self.load_source(insn);
                        self.compute(insn.op, wide);
                        self.store_flags(insn.op, wide);
                    }
                    return;
                }
                self.compute(insn.op, wide);
                if flags {
                    self.store_flags(insn.op, wide);
                }
                if writes {
                    self.store_rm(insn, X11, completed);
                }
            }
        }
    }

    /// Load a 16-byte memory operand into X0 (low) and X1 (high).
    fn load_wide(&mut self, insn: &Insn, operand: &MemOperand, completed: usize) {
        self.effective_address(operand, true);
        let helper = if FORMS[usize::from(insn.form)].unaligned {
            read_u128_helper as *const ()
        } else {
            read_aligned_u128_helper as *const ()
        };
        self.call_helper(helper as u64);
        self.guard(insn, completed);
    }

    /// Store both halves of a guest XMM register.
    fn store_halves(&mut self, index: u8, low: u8, high: u8) {
        self.asm.str_uoff_64(low, X19, xmm_offset(index));
        self.asm.str_uoff_64(high, X19, xmm_offset(index) + 8);
    }

    fn vector(&mut self, insn: &Insn, completed: usize) {
        let reg = insn.reg.index;
        match (insn.op, insn.encoding) {
            (Op::Movd, Encoding::XmmGpr) => {
                self.load_rm(X9, insn, completed);
                if insn.size == 4 {
                    self.asm.mov_reg_32(X9, X9);
                }
                self.store_halves(reg, X9, XZR);
            }
            (Op::Movd, _) => {
                self.asm.ldr_uoff_64(X10, X19, xmm_offset(reg));
                self.store_rm(insn, X10, completed);
            }
            (_, Encoding::RmXmm) => self.vector_store(insn, completed),
            (
                Op::Movaps
                | Op::Movups
                | Op::Movdqa
                | Op::Movdqu
                | Op::Movntdqa
                | Op::Movq
                | Op::Movss
                | Op::Movsd
                | Op::Movlps
                | Op::Movhps,
                _,
            ) => self.vector_load(insn, completed),
            (Op::Pblendvb | Op::Blendvps | Op::Blendvpd, _) => {
                let lane = match insn.op {
                    Op::Pblendvb => 8,
                    Op::Blendvps => 32,
                    _ => 64,
                };
                self.load_vector_source(insn, completed);
                let asm = &mut *self.asm;
                asm.ldr_q_uoff(0, X19, xmm_offset(reg));
                asm.ldr_q_uoff(2, X19, xmm_offset(0));
                // The sign of each lane of XMM0 selects the source.
                asm.sshr_vector(2, 2, lane, lane - 1);
                asm.neon3(NEON_BSL, 2, 1, 0);
                asm.str_q_uoff(2, X19, xmm_offset(reg));
            }
            (op, _) if neon_widening(op).is_some() => {
                let Some((signed, steps)) = neon_widening(op) else {
                    return;
                };
                self.load_vector_source(insn, completed);
                let extend = if signed { NEON_SXTL } else { NEON_UXTL };
                for lane in steps {
                    self.asm.neon3(extend | lane << 16, 1, 1, 0);
                }
                self.asm.str_q_uoff(1, X19, xmm_offset(reg));
            }
            (op, Encoding::RmImmXmm) => {
                let Some(RmOperand::Reg(target)) = insn.rm else {
                    return;
                };
                let offset = xmm_offset(target.index);
                let lane = match op {
                    Op::Psllw | Op::Psrlw | Op::Psraw => 16,
                    Op::Pslld | Op::Psrld | Op::Psrad => 32,
                    _ => 64,
                };
                let count = insn.imm.min(u64::from(lane)) as u32;
                if count == 0 {
                    return;
                }
                let asm = &mut *self.asm;
                asm.ldr_q_uoff(0, X19, offset);
                match op {
                    Op::Psllw | Op::Pslld | Op::Psllq if count == lane => {
                        asm.neon3(NEON_EOR, 0, 0, 0);
                    }
                    Op::Psllw | Op::Pslld | Op::Psllq => asm.shl_vector(0, 0, lane, count),
                    Op::Psraw | Op::Psrad => asm.sshr_vector(0, 0, lane, count),
                    _ => asm.ushr_vector(0, 0, lane, count),
                }
                asm.str_q_uoff(0, X19, offset);
            }
            (op, _) => {
                let Some((base, swapped)) = neon_operation(op) else {
                    return;
                };
                self.load_vector_source(insn, completed);
                let asm = &mut *self.asm;
                asm.ldr_q_uoff(0, X19, xmm_offset(reg));
                if swapped {
                    asm.neon3(base, 0, 1, 0);
                } else {
                    asm.neon3(base, 0, 0, 1);
                }
                asm.str_q_uoff(0, X19, xmm_offset(reg));
            }
        }
    }

    /// The XMM or memory source in V1, with zeros above a narrow memory operand.
    fn load_vector_source(&mut self, insn: &Insn, completed: usize) {
        match insn.rm {
            Some(RmOperand::Mem(operand)) if insn.rm_size == 16 => {
                self.load_wide(insn, &operand, completed);
                self.asm.fmov_d_from_x(1, X0);
                self.asm.ins_d1_from_x(1, X1);
            }
            Some(RmOperand::Mem(_)) => {
                self.load_rm(X9, insn, completed);
                self.asm.fmov_d_from_x(1, X9);
            }
            Some(RmOperand::Reg(source)) => {
                self.asm.ldr_q_uoff(1, X19, xmm_offset(source.index));
            }
            None => {}
        }
    }

    fn vector_load(&mut self, insn: &Insn, completed: usize) {
        let reg = insn.reg.index;
        let target = xmm_offset(reg);
        match insn.rm {
            Some(RmOperand::Mem(operand)) if insn.rm_size == 16 => {
                self.load_wide(insn, &operand, completed);
                self.store_halves(reg, X0, X1);
            }
            Some(RmOperand::Mem(_)) => {
                self.load_rm(X9, insn, completed);
                match insn.op {
                    Op::Movlps => self.asm.str_uoff_64(X9, X19, target),
                    Op::Movhps => self.asm.str_uoff_64(X9, X19, target + 8),
                    _ => self.store_halves(reg, X9, XZR),
                }
            }
            Some(RmOperand::Reg(source)) => {
                let from = xmm_offset(source.index);
                let asm = &mut *self.asm;
                match insn.op {
                    Op::Movss => {
                        asm.ldr_uoff_32(X9, X19, from);
                        asm.str_uoff_32(X9, X19, target);
                    }
                    Op::Movsd => {
                        asm.ldr_uoff_64(X9, X19, from);
                        asm.str_uoff_64(X9, X19, target);
                    }
                    Op::Movq => {
                        asm.ldr_uoff_64(X9, X19, from);
                        self.store_halves(reg, X9, XZR);
                    }
                    // MOVHLPS and MOVLHPS.
                    Op::Movlps => {
                        asm.ldr_uoff_64(X9, X19, from + 8);
                        asm.str_uoff_64(X9, X19, target);
                    }
                    Op::Movhps => {
                        asm.ldr_uoff_64(X9, X19, from);
                        asm.str_uoff_64(X9, X19, target + 8);
                    }
                    _ => {
                        asm.ldr_q_uoff(0, X19, from);
                        asm.str_q_uoff(0, X19, target);
                    }
                }
            }
            None => {}
        }
    }

    fn vector_store(&mut self, insn: &Insn, completed: usize) {
        let from = xmm_offset(insn.reg.index);
        match insn.rm {
            Some(RmOperand::Mem(operand)) if insn.rm_size == 16 => {
                self.asm.ldr_uoff_64(X2, X19, from);
                self.asm.ldr_uoff_64(X3, X19, from + 8);
                self.effective_address(&operand, true);
                let helper = if FORMS[usize::from(insn.form)].unaligned {
                    write_u128_helper as *const ()
                } else {
                    write_aligned_u128_helper as *const ()
                };
                self.call_helper(helper as u64);
                self.guard(insn, completed);
            }
            Some(RmOperand::Mem(_)) => {
                let half = if insn.op == Op::Movhps { 8 } else { 0 };
                self.asm.ldr_uoff_64(X10, X19, from + half);
                self.store_rm(insn, X10, completed);
            }
            Some(RmOperand::Reg(target)) => {
                let to = xmm_offset(target.index);
                match insn.op {
                    Op::Movss => {
                        self.asm.ldr_uoff_32(X9, X19, from);
                        self.asm.str_uoff_32(X9, X19, to);
                    }
                    Op::Movsd => {
                        self.asm.ldr_uoff_64(X9, X19, from);
                        self.asm.str_uoff_64(X9, X19, to);
                    }
                    Op::Movq => {
                        self.asm.ldr_uoff_64(X9, X19, from);
                        self.store_halves(target.index, X9, XZR);
                    }
                    _ => {
                        self.asm.ldr_q_uoff(0, X19, from);
                        self.asm.str_q_uoff(0, X19, to);
                    }
                }
            }
            None => {}
        }
    }

    /// Lower an instruction accepted by [`is_native`].
    fn lower(&mut self, insn: &Insn, completed: usize, flags: bool) {
        match insn.op {
            Op::Nop => {}
            _ if insn.op.is_vector() => self.vector(insn, completed),
            Op::Neg => {
                self.load_rm(X10, insn, completed);
                self.asm.mov_reg_64(X21, X10);
                self.asm.mov_reg_64(X9, XZR);
                self.compute(Op::Sub, insn.size == 8);
                self.store_rm(insn, X11, completed);
                if flags {
                    self.asm.mov_reg_64(X10, X21);
                    self.asm.mov_reg_64(X9, XZR);
                    self.compute(Op::Sub, insn.size == 8);
                    self.store_flags(Op::Sub, insn.size == 8);
                }
            }
            Op::Not => {
                self.load_rm(X9, insn, completed);
                self.asm.mvn_64(X11, X9);
                self.store_rm(insn, X11, completed);
            }
            Op::Mov => match insn.encoding {
                Encoding::RmReg => {
                    self.load_reg(X10, insn.reg);
                    self.store_rm(insn, X10, completed);
                }
                Encoding::RegRm => {
                    self.load_rm(X10, insn, completed);
                    self.store_reg(insn.reg, insn.size, X10);
                }
                Encoding::OpRegImm => {
                    self.asm.load_imm_64(X10, insn.imm);
                    self.store_reg(insn.reg, insn.size, X10);
                }
                _ => {
                    self.asm.load_imm_64(X10, insn.imm);
                    self.store_rm(insn, X10, completed);
                }
            },
            Op::Movzx | Op::Movsx | Op::Movsxd => {
                self.load_rm(X9, insn, completed);
                match (insn.op, insn.rm_size) {
                    (Op::Movzx, 1) => self.asm.uxtb_32(X10, X9),
                    (Op::Movzx, _) => self.asm.uxth_32(X10, X9),
                    (_, 1) => self.asm.sxtb_64(X10, X9),
                    (_, 2) => self.asm.sxth_64(X10, X9),
                    _ => self.asm.sxtw_64(X10, X9),
                }
                self.store_reg(insn.reg, insn.size, X10);
            }
            Op::Lea => {
                if let Some(operand) = insn.memory() {
                    self.effective_address(&operand, false);
                    self.store_reg(insn.reg, insn.size, X1);
                }
            }
            Op::Bswap => {
                self.load_reg(X9, insn.reg);
                self.asm.rev(X10, X9, insn.size == 8);
                self.store_reg(insn.reg, insn.size, X10);
            }
            Op::Setcc => {
                self.load_condition(insn.condition);
                self.store_rm(insn, X12, completed);
            }
            Op::Bsf | Op::Bsr => {
                // BSF and BSR write the index of the lowest or the highest set
                // bit of the source into the destination and set the zero flag
                // when the source is zero (Intel SDM Volume 2B, "BSF" and
                // "BSR"). A zero source leaves the destination undefined, so
                // the destination keeps its value, which is the same rule the
                // interpreter and the reference apply.
                self.load_rm(X9, insn, completed);
                self.zero_extend(X9, insn.size);
                match insn.op {
                    // The lowest set bit is the highest set bit of the
                    // reversed value, so its index is the leading zero count
                    // of the reversed value.
                    Op::Bsf => {
                        self.asm.rbit_64(X10, X9);
                        self.asm.clz_64(X10, X10);
                    }
                    // The highest set bit sits at 63 minus the leading zero
                    // count.
                    _ => {
                        self.asm.clz_64(X10, X9);
                        self.asm.movz_64(X13, 63, 0);
                        self.asm.sub_reg_64(X10, X13, X10);
                    }
                }
                self.load_reg(X11, insn.reg);
                self.asm.tst_reg_64(X9, X9);
                // CSEL takes the first operand when the condition holds, so
                // the index is the value that replaces the destination.
                self.asm.csel_64(X11, X10, X11, Condition::Ne);
                match insn.size {
                    // A 16-bit destination keeps the bits above it, so the
                    // register is merged rather than replaced.
                    2 => self.store_reg(insn.reg, insn.size, X11),
                    // A 32-bit or 64-bit destination is written in full. The
                    // index has no bits above the operand width, so this
                    // matches the interpreter's 32-bit write, and a zero
                    // source keeps the whole register unchanged.
                    _ => self.asm.str_uoff_64(X11, X19, reg_offset(insn.reg.index)),
                }
                if flags {
                    // The zero flag is set when the source is zero. The other
                    // flags are undefined and keep their value.
                    self.asm.cset_32(X12, Condition::Eq);
                    self.asm.strb_uoff(X12, X19, ZF);
                }
            }
            Op::Bt => {
                // BT sets the carry flag from the tested bit and leaves the
                // other flags undefined (Intel SDM Volume 2B, "BT"). The bit
                // index comes from the register operand and is masked to the
                // operand width.
                self.load_rm(X9, insn, completed);
                self.zero_extend(X9, insn.size);
                self.load_reg(X10, insn.reg);
                self.asm.load_imm_64(X13, u64::from(8 * insn.size - 1));
                self.asm.and_reg_64(X10, X10, X13);
                self.asm.load_imm_64(X11, 1);
                self.asm.lslv_64(X11, X11, X10);
                self.asm.tst_reg_64(X9, X11);
                if flags {
                    self.asm.cset_32(X12, Condition::Ne);
                    self.asm.strb_uoff(X12, X19, CF);
                }
            }
            Op::Cmov => {
                self.load_rm(X10, insn, completed);
                self.load_reg(X9, insn.reg);
                self.load_condition(insn.condition);
                self.asm.tst_reg_32(X12, X12);
                self.asm.csel_64(X11, X10, X9, Condition::Ne);
                self.store_reg(insn.reg, insn.size, X11);
            }
            Op::Push => {
                match insn.encoding {
                    Encoding::OpReg => self.load_reg(X2, insn.reg),
                    Encoding::Imm => self.asm.load_imm_64(X2, insn.imm),
                    _ => self.load_rm(X2, insn, completed),
                }
                self.push(insn, completed);
            }
            Op::Pop => {
                self.read_stack(insn, completed);
                // POP RSP keeps the popped value, so it is written last.
                self.asm.mov_reg_64(X21, X0);
                self.release_stack(8);
                self.store_reg(insn.reg, 8, X21);
            }
            Op::Call => {
                // The target is read before the return address is pushed.
                self.load_target(X21, insn, completed);
                self.asm.load_imm_64(X2, insn.next_rip());
                self.push(insn, completed);
                self.asm.str_uoff_64(X21, X19, RIP_OFFSET);
            }
            Op::Jmp => {
                self.load_target(X9, insn, completed);
                self.asm.str_uoff_64(X9, X19, RIP_OFFSET);
            }
            Op::Jcc => {
                self.load_condition(insn.condition);
                self.asm.load_imm_64(X9, insn.next_rip());
                self.asm.load_imm_64(X10, insn.imm);
                self.asm.tst_reg_32(X12, X12);
                self.asm.csel_64(X9, X10, X9, Condition::Ne);
                self.asm.str_uoff_64(X9, X19, RIP_OFFSET);
            }
            Op::Ret => {
                self.read_stack(insn, completed);
                self.asm.str_uoff_64(X0, X19, RIP_OFFSET);
                self.release_stack(8u64.wrapping_add(insn.imm));
            }
            Op::Hlt => {
                self.asm.load_imm_64(X9, insn.next_rip());
                self.asm.str_uoff_64(X9, X19, RIP_OFFSET);
                self.asm.load_imm_64(X1, 1); // 1 = Halt
                self.call_helper(set_exit_reason_helper as *const () as u64);
            }
            _ => self.alu(insn, completed, flags),
        }
    }
}

/// Emit the post-helper fault check for a translated memory access.
///
/// The helper records a fault in the block context instead of unwinding through
/// the native frame. This guard lets translated code observe that flag, restore
/// nothing (no architectural effect may have been committed yet), report the
/// number of fully executed instructions, point guest RIP at the faulting
/// instruction, and jump to the shared epilogue.
fn emit_fault_guard(
    asm: &mut Arm64Assembler,
    guards: &mut Vec<usize>,
    executed_before: usize,
    fault_rip: u64,
) {
    asm.ldr_uoff_64(X5, X20, BLOCK_CTX_FAULTED_OFFSET);
    let skip = asm.current_offset();
    asm.cbz_64(X5, 0);
    asm.load_imm_64(X6, fault_rip);
    asm.str_uoff_64(X6, X19, 16 * 8); // Guest RIP lives at byte offset 128.
    asm.load_imm_64(X0, executed_before as u64);
    guards.push(asm.current_offset());
    asm.b(0);
    let resume = asm.current_offset();
    asm.patch_cbz_64(skip, X5, resume as i32 - skip as i32);
}

/// State shared by the dispatcher and the blocks of one native call: the
/// block it enters and every block reached from it through links.
#[repr(C)]
struct BlockContext {
    memory_ptr: *mut GuestMemory,
    /// Set to 1 by a memory helper when a guest access faults. Translated code
    /// tests this word before committing any further architectural effect.
    faulted: u64,
    /// The engine's verification epoch for this call, or [`STOPPED_EPOCH`]
    /// once no further link may be followed: an exit or fault is recorded, a
    /// store may have written translated guest code, or the shared semantics
    /// ran. The dispatcher then advances the engine's epoch.
    epoch: u64,
    /// Instructions the running block and the blocks after it may still use
    /// beyond what the dispatcher needs left to select a full block. Following
    /// a link charges the length of the block it leaves. 0 while linking is
    /// disabled.
    headroom: u64,
    links_followed: u64,
    /// The [`LinkSlot`] the last exit stub could not follow, or 0.
    missed_slot: u64,
    /// See [`DbtEngine::code_start`].
    code_start: u64,
    code_end: u64,
    exit_reason: Option<ExitReason>,
    /// Instructions executed through [`execute_helper`].
    helper_instructions: u64,
}

const BLOCK_CTX_FAULTED_OFFSET: u32 = 8;
const _: () = assert!(std::mem::offset_of!(BlockContext, faulted) == 8);
const BLOCK_CTX_EPOCH_OFFSET: u32 = std::mem::offset_of!(BlockContext, epoch) as u32;
const BLOCK_CTX_HEADROOM_OFFSET: u32 = std::mem::offset_of!(BlockContext, headroom) as u32;
const BLOCK_CTX_LINKS_FOLLOWED_OFFSET: u32 =
    std::mem::offset_of!(BlockContext, links_followed) as u32;
const BLOCK_CTX_MISSED_SLOT_OFFSET: u32 = std::mem::offset_of!(BlockContext, missed_slot) as u32;

fn check_guest_address(ctx: *mut BlockContext, address: u64, size: u8) -> bool {
    if x86_exec::canonical_access(address, size) {
        return true;
    }
    fault(ctx, ExitReason::GeneralProtection);
    false
}

/// End the chain before a guest store that may write bytes some cached block
/// was translated from. The range is conservative: it also covers data
/// between translated blocks.
fn note_store(ctx: *mut BlockContext, address: u64, size: u64) {
    unsafe {
        if address < (*ctx).code_end && address.saturating_add(size) > (*ctx).code_start {
            (*ctx).epoch = STOPPED_EPOCH;
        }
    }
}

extern "C" fn write_u64_helper(ctx: *mut BlockContext, addr: u64, val: u64) {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return;
        }
        if !check_guest_address(ctx, addr, 8) {
            return;
        }
        note_store(ctx, addr, 8);
        let mem = &mut *(*ctx).memory_ptr;
        if let Err(e) = mem.write_u64(addr, val) {
            fault(ctx, ExitReason::MemoryFault(e));
        }
    }
}

extern "C" fn read_u64_helper(ctx: *mut BlockContext, addr: u64) -> u64 {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return 0;
        }
        if !check_guest_address(ctx, addr, 8) {
            return 0;
        }
        let mem = &*(*ctx).memory_ptr;
        match mem.read_u64(addr) {
            Ok(v) => v,
            Err(e) => {
                fault(ctx, ExitReason::MemoryFault(e));
                0
            }
        }
    }
}

/// Host address of the store helper for an operand of `size` bytes.
fn store_helper(size: u8) -> u64 {
    let helper = match size {
        1 => write_u8_helper as *const (),
        2 => write_u16_helper as *const (),
        4 => write_u32_helper as *const (),
        _ => write_u64_helper as *const (),
    };
    helper as u64
}

/// Host address of the zero-extending load helper for `size` bytes.
fn load_helper(size: u8) -> u64 {
    let helper = match size {
        1 => read_u8_helper as *const (),
        2 => read_u16_helper as *const (),
        4 => read_u32_helper as *const (),
        _ => read_u64_helper as *const (),
    };
    helper as u64
}

fn fault(ctx: *mut BlockContext, reason: ExitReason) {
    unsafe {
        (*ctx).faulted = 1;
        record_exit(ctx, reason);
    }
}

/// Record an exit that ends the native call, so no link is followed after it.
fn record_exit(ctx: *mut BlockContext, reason: ExitReason) {
    unsafe {
        (*ctx).epoch = STOPPED_EPOCH;
        (*ctx).exit_reason = Some(reason);
    }
}

/// Read 16 bytes at any alignment; returned in X0 (low) and X1 (high).
extern "C" fn read_u128_helper(ctx: *mut BlockContext, addr: u64) -> u128 {
    if !check_guest_address(ctx, addr, 16) {
        return 0;
    }
    let memory = unsafe { &*(*ctx).memory_ptr };
    let mut bytes = [0; 16];
    if let Err(error) = memory.read_bytes(addr, &mut bytes) {
        fault(ctx, ExitReason::MemoryFault(error));
    }
    u128::from_le_bytes(bytes)
}

extern "C" fn read_aligned_u128_helper(ctx: *mut BlockContext, addr: u64) -> u128 {
    if !addr.is_multiple_of(16) {
        fault(ctx, ExitReason::GeneralProtection);
        return 0;
    }
    read_u128_helper(ctx, addr)
}

/// Write 16 bytes at any alignment; `value` arrives in X2 (low) and X3 (high).
extern "C" fn write_u128_helper(ctx: *mut BlockContext, addr: u64, value: u128) {
    if !check_guest_address(ctx, addr, 16) {
        return;
    }
    note_store(ctx, addr, 16);
    let memory = unsafe { &mut *(*ctx).memory_ptr };
    if let Err(error) = memory.write_bytes(addr, &value.to_le_bytes()) {
        fault(ctx, ExitReason::MemoryFault(error));
    }
}

extern "C" fn write_aligned_u128_helper(ctx: *mut BlockContext, addr: u64, value: u128) {
    if !addr.is_multiple_of(16) {
        fault(ctx, ExitReason::GeneralProtection);
        return;
    }
    write_u128_helper(ctx, addr, value);
}

/// Execute one decoded instruction through the shared semantics.
extern "C" fn execute_helper(ctx: *mut BlockContext, cpu: *mut CpuState, insn: *const Insn) {
    unsafe {
        (*ctx).helper_instructions += 1;
        // The shared semantics may store anywhere, translated code included,
        // without passing through `note_store`.
        (*ctx).epoch = STOPPED_EPOCH;
        match x86_exec::execute(&mut *cpu, &mut *(*ctx).memory_ptr, &*insn) {
            Ok(None) => {}
            Ok(Some(reason)) => record_exit(ctx, reason),
            Err(reason) => fault(ctx, reason),
        }
    }
}

/// Push `value` below `sp` and return the new stack pointer.
extern "C" fn push_helper(ctx: *mut BlockContext, sp: u64, value: u64) -> u64 {
    let Some(address) = sp.checked_sub(8) else {
        fault(ctx, ExitReason::StackUnderflow);
        return sp;
    };
    write_u64_helper(ctx, address, value);
    address
}

extern "C" fn write_u16_helper(ctx: *mut BlockContext, addr: u64, val: u64) {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return;
        }
        if !check_guest_address(ctx, addr, 2) {
            return;
        }
        note_store(ctx, addr, 2);
        let mem = &mut *(*ctx).memory_ptr;
        if let Err(e) = mem.write_u16(addr, val as u16) {
            fault(ctx, ExitReason::MemoryFault(e));
        }
    }
}

extern "C" fn write_u32_helper(ctx: *mut BlockContext, addr: u64, val: u64) {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return;
        }
        if !check_guest_address(ctx, addr, 4) {
            return;
        }
        note_store(ctx, addr, 4);
        let mem = &mut *(*ctx).memory_ptr;
        if let Err(e) = mem.write_u32(addr, val as u32) {
            fault(ctx, ExitReason::MemoryFault(e));
        }
    }
}

extern "C" fn read_u32_helper(ctx: *mut BlockContext, addr: u64) -> u64 {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return 0;
        }
        if !check_guest_address(ctx, addr, 4) {
            return 0;
        }
        let mem = &*(*ctx).memory_ptr;
        match mem.read_u32(addr) {
            Ok(v) => u64::from(v),
            Err(e) => {
                fault(ctx, ExitReason::MemoryFault(e));
                0
            }
        }
    }
}

extern "C" fn write_u8_helper(ctx: *mut BlockContext, addr: u64, val: u64) {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return;
        }
        if !check_guest_address(ctx, addr, 1) {
            return;
        }
        note_store(ctx, addr, 1);
        let mem = &mut *(*ctx).memory_ptr;
        if let Err(e) = mem.write_u8(addr, val as u8) {
            fault(ctx, ExitReason::MemoryFault(e));
        }
    }
}

extern "C" fn read_u8_helper(ctx: *mut BlockContext, addr: u64) -> u64 {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return 0;
        }
        if !check_guest_address(ctx, addr, 1) {
            return 0;
        }
        let mem = &*(*ctx).memory_ptr;
        match mem.read_u8(addr) {
            Ok(v) => v as u64,
            Err(e) => {
                fault(ctx, ExitReason::MemoryFault(e));
                0
            }
        }
    }
}

extern "C" fn read_u16_helper(ctx: *mut BlockContext, addr: u64) -> u64 {
    unsafe {
        if (*ctx).exit_reason.is_some() {
            return 0;
        }
        if !check_guest_address(ctx, addr, 2) {
            return 0;
        }
        let mem = &*(*ctx).memory_ptr;
        match mem.read_u16(addr) {
            Ok(v) => v as u64,
            Err(e) => {
                fault(ctx, ExitReason::MemoryFault(e));
                0
            }
        }
    }
}

extern "C" fn set_exit_reason_helper(ctx: *mut BlockContext, code: u64) {
    unsafe {
        if code == 1 && (*ctx).exit_reason.is_none() {
            record_exit(ctx, ExitReason::Halt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryError, MemoryPermissions};

    const ENTRY: u64 = 0x1000;

    fn fixture(code: &[u8]) -> (DbtEngine, CpuState, GuestMemory) {
        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(ENTRY, code.len(), MemoryPermissions::READ_EXECUTE, code)
            .unwrap();
        (
            DbtEngine::new(64 * 1024).unwrap(),
            CpuState::new(ENTRY, 0),
            memory,
        )
    }

    /// `mov eax, imm32; hlt`, with the immediate at `ENTRY + 1`.
    fn mov_eax(imm: u8) -> [u8; 6] {
        [0xB8, imm, 0x00, 0x00, 0x00, 0xF4]
    }

    fn writable_fixture(code: &[u8]) -> (DbtEngine, GuestMemory) {
        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(
                ENTRY,
                code.len(),
                MemoryPermissions::READ_WRITE_EXECUTE,
                code,
            )
            .unwrap();
        (DbtEngine::new(64 * 1024).unwrap(), memory)
    }

    fn rax_after_run(engine: &mut DbtEngine, memory: &mut GuestMemory, limit: usize) -> u64 {
        let mut cpu = CpuState::new(ENTRY, 0);
        let result = engine.run(&mut cpu, memory, limit);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        cpu.get(Register::Rax)
    }

    #[test]
    fn unchanged_code_stays_cached() {
        let (mut engine, mut memory) = writable_fixture(&mov_eax(1));
        assert_eq!(rax_after_run(&mut engine, &mut memory, 16), 1);
        assert_eq!(rax_after_run(&mut engine, &mut memory, 16), 1);
        assert_eq!(engine.metrics.blocks_compiled, 1);
        assert_eq!(engine.metrics.cache_hits, 1);
        assert_eq!(engine.metrics.blocks_invalidated, 0);
    }

    #[test]
    fn rewritten_code_is_translated_again() {
        let (mut engine, mut memory) = writable_fixture(&mov_eax(1));
        assert_eq!(rax_after_run(&mut engine, &mut memory, 16), 1);
        memory.write_bytes(ENTRY + 1, &[2]).unwrap();
        assert_eq!(rax_after_run(&mut engine, &mut memory, 16), 2);
        assert_eq!(engine.metrics.blocks_compiled, 2);
        assert_eq!(engine.metrics.blocks_invalidated, 1);
        // The new translation is cached in turn.
        assert_eq!(rax_after_run(&mut engine, &mut memory, 16), 2);
        assert_eq!(engine.metrics.blocks_compiled, 2);
    }

    #[test]
    fn rewriting_the_last_byte_of_a_block_is_detected() {
        let (mut engine, mut memory) = writable_fixture(&mov_eax(1));
        assert_eq!(rax_after_run(&mut engine, &mut memory, 16), 1);
        // HLT becomes NOP, so the block now runs off the end of the mapping.
        memory.write_bytes(ENTRY + 5, &[0x90]).unwrap();
        let mut cpu = CpuState::new(ENTRY, 0);
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert!(matches!(result.exit_reason, ExitReason::MemoryFault(_)));
        assert_eq!(engine.metrics.blocks_invalidated, 1);
    }

    #[test]
    fn budget_limited_block_is_checked_against_current_bytes() {
        let (mut engine, mut memory) = writable_fixture(&mov_eax(1));
        for expected in [1, 3] {
            let mut cpu = CpuState::new(ENTRY, 0);
            let result = engine.run(&mut cpu, &mut memory, 1);
            assert_eq!(result.exit_reason, ExitReason::StepLimitExceeded);
            assert_eq!(cpu.get(Register::Rax), expected);
            memory.write_bytes(ENTRY + 1, &[3]).unwrap();
        }
        assert_eq!(engine.metrics.blocks_invalidated, 1);
    }

    #[test]
    fn another_address_space_does_not_reuse_a_block() {
        let (mut engine, mut first) = writable_fixture(&mov_eax(1));
        let (_, mut second) = writable_fixture(&mov_eax(7));
        assert_eq!(rax_after_run(&mut engine, &mut first, 16), 1);
        assert_eq!(rax_after_run(&mut engine, &mut second, 16), 7);

        let mut unmapped = GuestMemory::new();
        let mut cpu = CpuState::new(ENTRY, 0);
        let result = engine.run(&mut cpu, &mut unmapped, 16);
        assert!(matches!(result.exit_reason, ExitReason::MemoryFault(_)));
    }

    #[test]
    fn cached_block_obeys_remaining_budget() {
        let code = [
            0x48, 0x83, 0xC0, 0x80, // add rax, -128
            0x48, 0x83, 0xE8, 0xFF, // sub rax, -1
            0x48, 0x83, 0xE8, 0x02, // sub rax, 2
            0xF4, // hlt
        ];
        let (mut engine, initial, mut memory) = fixture(&code);

        let whole = engine.run(&mut initial.clone(), &mut memory, 16);
        assert_eq!(whole.instructions_executed, 4);
        assert_eq!(whole.exit_reason, ExitReason::Halt);
        assert_eq!(whole.final_state.get(Register::Rax), (-129_i64) as u64);

        for _ in 0..2 {
            let mut cpu = initial.clone();
            let limited = engine.run(&mut cpu, &mut memory, 2);
            assert_eq!(limited.instructions_executed, 2);
            assert_eq!(limited.exit_reason, ExitReason::StepLimitExceeded);
            assert_eq!(cpu.rip, ENTRY + 8);
            assert_eq!(cpu.get(Register::Rax), (-127_i64) as u64);

            let resumed = engine.run(&mut cpu, &mut memory, 2);
            assert_eq!(resumed.instructions_executed, 2);
            assert_eq!(resumed.exit_reason, ExitReason::Halt);
            assert_eq!(resumed.final_state, whole.final_state);
        }
        assert_eq!(engine.metrics.blocks_compiled, 3);
        assert_eq!(engine.metrics.cache_misses, 3);
        assert_eq!(engine.metrics.cache_hits, 2);
    }

    #[test]
    fn budget_does_not_decode_a_later_unsupported_opcode() {
        let code = [
            0x48, 0x83, 0xC0, 0x01, // add rax, 1
            0xCC, // int3 (unsupported)
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        let limited = engine.run(&mut cpu, &mut memory, 1);
        assert_eq!(limited.exit_reason, ExitReason::StepLimitExceeded);
        assert_eq!(limited.instructions_executed, 1);
        assert_eq!(cpu.rip, ENTRY + 4);
        assert_eq!(cpu.get(Register::Rax), 1);
        let unsupported = engine.run(&mut cpu, &mut memory, 1);
        assert_eq!(
            unsupported.exit_reason,
            ExitReason::IllegalInstruction(0xCC)
        );
        assert_eq!(unsupported.instructions_executed, 0);
    }

    #[test]
    fn completed_store_survives_the_next_fault_or_budget_boundary() {
        for native in [false, true] {
            for (tail, fault) in [
                (&[0xcc][..], ExitReason::IllegalInstruction(0xcc)),
                (
                    &[0xb8, 1][..],
                    ExitReason::MemoryFault(MemoryError::UnmappedAddress(ENTRY + 5)),
                ),
                (
                    &[0x48, 0x8b, 0x02, 0xf4][..],
                    ExitReason::MemoryFault(MemoryError::UnmappedAddress(0x4000)),
                ),
            ] {
                for budget in [0, 1, 16] {
                    // mov [rcx], rax; then an illegal, truncated or faulting instruction.
                    let code = [&[0x48, 0x89, 0x01][..], tail].concat();
                    let (mut engine, mut cpu, mut memory) = fixture(&code);
                    memory
                        .map_region(0x3000, 8, MemoryPermissions::READ_WRITE)
                        .unwrap();
                    cpu.set(Register::Rax, 0x1234);
                    cpu.set(Register::Rcx, 0x3000);
                    cpu.set(Register::Rdx, 0x4000);
                    let mut expected = cpu.clone();
                    if budget > 0 {
                        expected.rip += 3;
                    }
                    let result = if native {
                        engine.run(&mut cpu, &mut memory, budget)
                    } else {
                        crate::cpu::Interpreter::new(cpu, &mut memory, budget).run()
                    };
                    assert_eq!(result.instructions_executed, budget.min(1));
                    assert_eq!(
                        result.exit_reason,
                        if budget <= 1 {
                            ExitReason::StepLimitExceeded
                        } else {
                            fault
                        }
                    );
                    assert_eq!(result.final_state, expected);
                    assert_eq!(
                        memory.read_u64(0x3000).unwrap(),
                        if budget == 0 { 0 } else { 0x1234 }
                    );
                }
            }
        }
    }

    #[test]
    fn hlt_preserves_first_memory_fault() {
        let code = [
            0x48, 0x8B, 0x01, // mov rax, [rcx]
            0x48, 0x8B, 0x02, // mov rax, [rdx]
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        cpu.set(Register::Rcx, 0x2000);
        cpu.set(Register::Rdx, 0x3000);
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(
            result.exit_reason,
            ExitReason::MemoryFault(MemoryError::UnmappedAddress(0x2000))
        );
    }

    #[test]
    fn no_base_sib_computes_displacement_without_base() {
        // mov rax, [0x2000]; hlt
        let code = [
            0x48, 0x8B, 0x04, 0x25, 0x00, 0x20, 0, 0,    // mov rax, [0x2000]
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        memory
            .map_region(0x2000, 16, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory.write_u64(0x2000, 0x1122_3344_5566_7788).unwrap();
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(cpu.get(Register::Rax), 0x1122_3344_5566_7788);
    }

    #[test]
    fn sib_store_executes_through_the_native_dbt() {
        // mov [rsp + 0x20], rbx; hlt. The SIB store form that the static
        // probe used to report as a startup blocker.
        let code = [
            0x48, 0x89, 0x5c, 0x24, 0x20, // mov [rsp + 0x20], rbx
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        const STACK: u64 = 0x4000;
        const STORED_ADDR: u64 = 0x4020;
        const STORED_VALUE: u64 = 0x1122_3344_5566_7788;
        cpu.set(Register::Rsp, STACK);
        cpu.set(Register::Rbx, STORED_VALUE);
        memory
            .map_region(STACK, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(memory.read_u64(STORED_ADDR).unwrap(), STORED_VALUE);
    }

    /// A 32-bit MOVSX destination holds a 32-bit result, so the upper half of
    /// the register must be cleared.
    ///
    /// The encodings are `movsbl eax, bl` (0F BE C3) and
    /// `movsbq rdx, bl` (48 0F BE D3), as produced by LLVM 23.1.6
    /// `llvm-mc -show-encoding`. With `bl = 0xFF` the 32-bit destination holds
    /// 0x00000000ffffffff while the 64-bit destination holds the sign
    /// extension across all 64 bits.
    ///
    /// Per Intel SDM Volume 1, "Address Calculations in 64-Bit Mode": "32-bit
    /// operands generate a 32-bit result, zero-extended to a 64-bit result in
    /// the destination general-purpose register."
    #[test]
    fn movsx_to_a_32_bit_destination_clears_the_upper_half() {
        // mov rbx, 0xffffffffffffffff (48 BB <imm64>)
        // movsbl eax, bl (0F BE C3)
        // movsbq rdx, bl (48 0F BE D3)
        let mut code = vec![0x48, 0xBB];
        code.extend_from_slice(&u64::MAX.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0xBE, 0xC3]);
        code.extend_from_slice(&[0x48, 0x0F, 0xBE, 0xD3]);
        code.push(0xF4); // hlt

        let (mut engine, mut cpu, mut memory) = fixture(&code);
        cpu.set(Register::Rbx, u64::MAX);
        let result = engine.run(&mut cpu, &mut memory, 16);

        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(result.final_state.get(Register::Rax), 0x0000_0000_ffff_ffff);
        assert_eq!(result.final_state.get(Register::Rdx), 0xffff_ffff_ffff_ffff);
    }

    /// A 32-bit MOVSX destination from a 16-bit source also holds a 32-bit
    /// result.
    ///
    /// The encodings are `movswl eax, bx` (0F BF C3) and
    /// `movswq rdx, bx` (48 0F BF D3), as produced by LLVM 23.1.6
    /// `llvm-mc -show-encoding`.
    #[test]
    fn movsx_from_a_word_to_a_32_bit_destination_clears_the_upper_half() {
        // mov rbx, 0xffffffffffffffff (48 BB <imm64>)
        // movswl eax, bx (0F BF C3)
        // movswq rdx, bx (48 0F BF D3)
        let mut code = vec![0x48, 0xBB];
        code.extend_from_slice(&u64::MAX.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0xBF, 0xC3]);
        code.extend_from_slice(&[0x48, 0x0F, 0xBF, 0xD3]);
        code.push(0xF4); // hlt

        let (mut engine, mut cpu, mut memory) = fixture(&code);
        cpu.set(Register::Rbx, u64::MAX);
        let result = engine.run(&mut cpu, &mut memory, 16);

        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(result.final_state.get(Register::Rax), 0x0000_0000_ffff_ffff);
        assert_eq!(result.final_state.get(Register::Rdx), 0xffff_ffff_ffff_ffff);
    }

    /// The same 32-bit destination rule applies to the memory source forms.
    ///
    /// The encodings are `movsbl eax, byte [rdi]` (0F BE 07) and
    /// `movsbq rdx, byte [rdi]` (48 0F BE 17), as produced by LLVM 23.1.6
    /// `llvm-mc -show-encoding`.
    #[test]
    fn movsx_from_memory_to_a_32_bit_destination_clears_the_upper_half() {
        // mov rdi, 0x4000 (48 BF <imm64>)
        // movsbl eax, byte [rdi] (0F BE 07)
        // movsbq rdx, byte [rdi] (48 0F BE 17)
        let mut code = vec![0x48, 0xBF];
        code.extend_from_slice(&0x4000u64.to_le_bytes());
        code.extend_from_slice(&[0x0F, 0xBE, 0x07]);
        code.extend_from_slice(&[0x48, 0x0F, 0xBE, 0x17]);
        code.push(0xF4); // hlt

        let (mut engine, mut cpu, mut memory) = fixture(&code);
        memory
            .map_region(0x4000, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory.write_u8(0x4000, 0xFF).unwrap();
        let result = engine.run(&mut cpu, &mut memory, 16);

        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(result.final_state.get(Register::Rax), 0x0000_0000_ffff_ffff);
        assert_eq!(result.final_state.get(Register::Rdx), 0xffff_ffff_ffff_ffff);
    }

    #[test]
    fn indirect_call_target_fault_leaves_guest_stack_unchanged() {
        // call qword ptr [rip + 0x1000]; hlt. The IAT slot is outside the
        // mapped code region, so the read must fault before the push.
        let code = [0xFF, 0x15, 0x00, 0x10, 0x00, 0x00, 0xF4];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        memory
            .map_region(0x3000, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        cpu.set(Register::Rsp, 0x3080);
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(
            result.exit_reason,
            ExitReason::MemoryFault(MemoryError::UnmappedAddress(ENTRY + 0x1006))
        );
        assert_eq!(cpu.get(Register::Rsp), 0x3080);
    }

    #[test]
    fn indirect_register_call_preserves_stack_and_target() {
        // mov rax, 0x2000; call rax; hlt. 0x2000 is unmapped in this fixture,
        // so the translated call must fault after pushing and leave RIP at the
        // trampoline rather than executing translated bytes at an unknown address.
        let code = [
            0x48, 0xB8, 0x00, 0x20, 0, 0, 0, 0, 0, 0, // mov rax, 0x2000
            0xFF, 0xD0, // call rax
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        memory
            .map_region(0x3000, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        cpu.set(Register::Rsp, 0x3080);
        let result = engine.run(&mut cpu, &mut memory, 16);
        // The call itself succeeds; the next dispatch faults on unmapped code.
        assert!(
            matches!(result.exit_reason, ExitReason::MemoryFault(_)),
            "unexpected exit: {:?}",
            result.exit_reason
        );
        assert_eq!(cpu.get(Register::Rsp), 0x3078);
        assert_eq!(memory.read_u64(0x3078).unwrap(), ENTRY + 12);
    }

    #[test]
    fn exhausted_code_cache_reports_a_distinct_exit_reason() {
        const BLOCKS: usize = 128;
        const ADDS: usize = 16;
        let block_len = ADDS * 4 + 5;
        let mut code = vec![0u8; BLOCKS * block_len];
        for block in 0..BLOCKS {
            let start = block * block_len;
            for add in 0..ADDS {
                let at = start + add * 4;
                code[at..at + 4].copy_from_slice(&[0x48, 0x83, 0xC0, 0x01]);
            }
            let at = start + ADDS * 4;
            if block + 1 == BLOCKS {
                code[at] = 0xF4;
            } else {
                let next = (block + 1) * block_len;
                let rel = next as i32 - (at + 5) as i32;
                code[at] = 0xE9;
                code[at + 1..at + 5].copy_from_slice(&rel.to_le_bytes());
            }
        }
        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(ENTRY, code.len(), MemoryPermissions::READ_EXECUTE, &code)
            .unwrap();
        let mut engine = DbtEngine::new(16 * 1024).unwrap();
        let mut cpu = CpuState::new(ENTRY, 0);
        let result = engine.run(&mut cpu, &mut memory, BLOCKS * ADDS * 2);
        assert_eq!(result.exit_reason, ExitReason::JitCacheExhausted);
        assert!(engine.metrics.blocks_compiled < BLOCKS);
    }

    #[test]
    fn translated_blocks_balance_a_48_byte_frame() {
        let code = [0x48, 0x83, 0xC0, 0x01, 0xF4]; // add rax, 1; hlt
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        let emitted = engine.assembler.instructions();
        assert_eq!(emitted[0], 0xa9bd7bfd); // stp x29, x30, [sp, #-48]!
        assert_eq!(emitted[emitted.len() - 2], 0xa8c37bfd); // ldp x29, x30, [sp], #48
        assert_eq!(*emitted.last().unwrap(), 0xd65f03c0); // ret
    }

    #[test]
    fn faulting_push_leaves_rsp_and_guest_memory_unchanged() {
        // push rax with RSP aimed at an unmapped page. x86 raises the fault
        // without decrementing RSP or writing the stack.
        let code = [0x50, 0xF4]; // push rax; hlt
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        cpu.set(Register::Rax, 0x1234_5678);
        cpu.set(Register::Rsp, 0x9000);
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(
            result.exit_reason,
            ExitReason::MemoryFault(MemoryError::UnmappedAddress(0x8FF8))
        );
        assert_eq!(cpu.get(Register::Rsp), 0x9000);
        assert_eq!(cpu.rip, ENTRY);
        assert_eq!(result.instructions_executed, 0);
    }

    #[test]
    fn mid_block_fault_reports_only_completed_instructions() {
        let code = [
            0x48, 0x83, 0xC0, 0x01, // add rax, 1
            0x48, 0x8B, 0x01, // mov rax, [rcx] (unmapped)
            0x48, 0x83, 0xC0, 0x01, // add rax, 1 (must not run)
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        cpu.set(Register::Rcx, 0x9000);
        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(
            result.exit_reason,
            ExitReason::MemoryFault(MemoryError::UnmappedAddress(0x9000))
        );
        assert_eq!(result.instructions_executed, 1);
        assert_eq!(cpu.get(Register::Rax), 1);
        assert_eq!(cpu.rip, ENTRY + 4);
    }

    #[test]
    fn rip_relative_immediate_store_uses_end_of_instruction() {
        let code = [
            0x48, 0xC7, 0x05, 0xFA, 0x0F, 0, 0, 0xEF, 0xFF, 0xFF,
            0xFF, // mov [rip+0xffa], -17
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        memory
            .map_region(0x2000, 16, MemoryPermissions::READ_WRITE)
            .unwrap();
        let result = engine.run(&mut cpu, &mut memory, 2);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        let mut stored = [0; 16];
        memory.read_bytes(0x2000, &mut stored).unwrap();
        assert_eq!(
            stored,
            [
                0, 0, 0, 0, 0, 0xEF, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0
            ]
        );
    }

    // FS/GS segment override regression tests from #280 (#72), restored after a
    // merge-conflict resolution dropped them.
    /// A segment-relative store and load through the FS base.
    ///
    /// The encodings are `mov [fs:0x8], rbx` (64 48 89 1C 25 08 00 00 00)
    /// and `mov rdx, [fs:0x8]` (64 48 8B 14 25 08 00 00 00), as produced by
    /// LLVM 23.1.6 `llvm-mc -show-encoding`. The SIB byte 0x25 selects "no
    /// index" and "no base", so the address is the FS base plus the
    /// displacement 8.
    ///
    /// Per Intel SDM Volume 3A section 3.4.4 the linear address is
    /// `(FS or GS).base + index + displacement`, so the store must land at
    /// `FS_BASE + 8`.
    #[test]
    fn fs_segment_override_adds_the_fs_base_to_the_address() {
        let code = [
            0x64, 0x48, 0x89, 0x1C, 0x25, 0x08, 0x00, 0x00, 0x00, // mov [fs:0x8], rbx
            0x64, 0x48, 0x8B, 0x14, 0x25, 0x08, 0x00, 0x00, 0x00, // mov rdx, [fs:0x8]
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        const FS_BASE: u64 = 0x7000;
        const GS_BASE: u64 = 0x8000;
        const STORED_VALUE: u64 = 0x1122_3344_5566_7788;
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        memory
            .map_region(FS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .map_region(GS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();

        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        // The store landed at the FS base plus the displacement, not at the
        // bare displacement and not at the GS base. The bare displacement
        // address 8 is not mapped at all, so reaching it would have faulted.
        assert_eq!(memory.read_u64(FS_BASE + 8).unwrap(), STORED_VALUE);
        assert_eq!(memory.read_u64(GS_BASE + 8).unwrap(), 0);
        // The load read back the same location.
        assert_eq!(result.final_state.get(Register::Rdx), STORED_VALUE);
    }

    /// A RIP-relative segment-relative store adds the segment base too.
    ///
    /// The encoding is `mov [fs:(%rip)], rbx` (64 48 89 1D 00 00 00 00), as
    /// produced by LLVM 23.1.6 `llvm-mc -show-encoding`. The displacement is
    /// zero, so the address is the address of the next instruction plus the
    /// FS base.
    ///
    /// The segment base is added to the effective address for every addressing
    /// form, so this must land at `FS_BASE + ENTRY + 8` and not at the bare
    /// RIP-relative address.
    #[test]
    fn rip_relative_segment_override_adds_the_segment_base() {
        let code = [
            0x64, 0x48, 0x89, 0x1D, 0x00, 0x00, 0x00, 0x00, // mov [fs:(%rip)], rbx
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        const FS_BASE: u64 = 0x7000;
        const STORED_VALUE: u64 = 0x1122_3344_5566_7788;
        cpu.fs_base = FS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        // The segment-adjusted address is FS_BASE + ENTRY + 8, so the region
        // has to reach that far.
        memory
            .map_region(FS_BASE, 0x2000, MemoryPermissions::READ_WRITE)
            .unwrap();

        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        // The instruction is 8 bytes long, so the next instruction is at
        // ENTRY + 8 and the displacement of zero makes that the offset.
        assert_eq!(memory.read_u64(FS_BASE + ENTRY + 8).unwrap(), STORED_VALUE);
    }

    /// The two execution paths must agree on segment-relative addressing.
    ///
    /// The native DBT translator and the software interpreter are separate
    /// implementations of the same guest semantics, so a program that uses a
    /// segment override must produce the same result and the same memory
    /// writes under both.
    ///
    /// The encodings are `mov [fs:0x8], rbx` (64 48 89 1C 25 08 00 00 00),
    /// `mov [gs:0x10], rbx` (65 48 89 1C 25 10 00 00 00),
    /// `mov rdx, [fs:0x8]` (64 48 8B 14 25 08 00 00 00) and
    /// `mov rsi, [gs:0x10]` (65 48 8B 34 25 10 00 00 00), as produced by
    /// LLVM 23.1.6 `llvm-mc -show-encoding`.
    #[test]
    fn interpreter_and_native_dbt_agree_on_segment_relative_addressing() {
        const CODE: [u8; 37] = [
            0x64, 0x48, 0x89, 0x1C, 0x25, 0x08, 0x00, 0x00, 0x00, // mov [fs:0x8], rbx
            0x65, 0x48, 0x89, 0x1C, 0x25, 0x10, 0x00, 0x00, 0x00, // mov [gs:0x10], rbx
            0x64, 0x48, 0x8B, 0x14, 0x25, 0x08, 0x00, 0x00, 0x00, // mov rdx, [fs:0x8]
            0x65, 0x48, 0x8B, 0x34, 0x25, 0x10, 0x00, 0x00, 0x00, // mov rsi, [gs:0x10]
            0xF4, // hlt
        ];
        const FS_BASE: u64 = 0x7000;
        const GS_BASE: u64 = 0x8000;
        const STORED_VALUE: u64 = 0x1122_3344_5566_7788;

        // The native DBT path.
        let (mut engine, mut cpu, mut memory) = fixture(&CODE);
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        memory
            .map_region(FS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .map_region(GS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        let native = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(native.exit_reason, ExitReason::Halt);
        let native_memory = memory;

        // The software interpreter path over the same program.
        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(ENTRY, CODE.len(), MemoryPermissions::READ_EXECUTE, &CODE)
            .unwrap();
        memory
            .map_region(FS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .map_region(GS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut cpu = CpuState::new(ENTRY, 0);
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        let mut interp = crate::cpu::Interpreter::new(cpu, &mut memory, 16);
        let interpreted = interp.run();

        assert_eq!(interpreted.exit_reason, native.exit_reason);
        assert_eq!(interpreted.final_state, native.final_state);
        assert_eq!(
            memory.read_u64(FS_BASE + 8).unwrap(),
            native.final_state.get(Register::Rbx)
        );
        assert_eq!(
            memory.read_u64(GS_BASE + 0x10).unwrap(),
            native.final_state.get(Register::Rbx)
        );
        // The bytes each path left in guest memory must be identical, not only the
        // two words checked above.
        for base in [FS_BASE, GS_BASE] {
            assert_eq!(
                region_bytes(&native_memory, base, 0x100),
                region_bytes(&memory, base, 0x100),
                "segment region at {base:#x}"
            );
        }
    }

    fn region_bytes(memory: &GuestMemory, base: u64, len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len];
        memory.read_bytes(base, &mut bytes).unwrap();
        bytes
    }

    /// FS and GS RIP-relative loads and stores with positive and negative
    /// displacements. The effective address is the segment base plus the address of
    /// the next instruction plus the signed disp32 (AMD64 APM vol. 1, RIP-relative
    /// addressing; FS/GS overrides still apply in 64-bit mode). Each expected address
    /// below is worked out by hand from that rule.
    const RIP_CODE: [u8; 33] = [
        // 0x1000: mov fs:[rip - 0x100], rbx   -> FS + 0x1008 - 0x100 = FS + 0x0F08
        0x64, 0x48, 0x89, 0x1D, 0x00, 0xFF, 0xFF, 0xFF,
        // 0x1008: mov gs:[rip + 0x40], rbx    -> GS + 0x1010 + 0x40 = GS + 0x1050
        0x65, 0x48, 0x89, 0x1D, 0x40, 0x00, 0x00, 0x00,
        // 0x1010: mov rdx, fs:[rip - 0x28]    -> FS + 0x1018 - 0x28 = FS + 0x0FF0
        0x64, 0x48, 0x8B, 0x15, 0xD8, 0xFF, 0xFF, 0xFF,
        // 0x1018: mov rsi, gs:[rip + 0x20]    -> GS + 0x1020 + 0x20 = GS + 0x1040
        0x65, 0x48, 0x8B, 0x35, 0x20, 0x00, 0x00, 0x00, // 0x1020: hlt
        0xF4,
    ];
    const RIP_FS: u64 = 0x10_0000;
    const RIP_GS: u64 = 0x20_0000;
    const RIP_STORED: u64 = 0x0102_0304_0506_0708;
    const RIP_FS_SEED: u64 = 0xA1A2_A3A4_A5A6_A7A8;
    const RIP_GS_SEED: u64 = 0xB1B2_B3B4_B5B6_B7B8;

    fn rip_setup(cpu: &mut CpuState, memory: &mut GuestMemory) {
        cpu.fs_base = RIP_FS;
        cpu.gs_base = RIP_GS;
        cpu.set(Register::Rbx, RIP_STORED);
        for base in [RIP_FS, RIP_GS] {
            memory
                .map_region(base, 0x2000, MemoryPermissions::READ_WRITE)
                .unwrap();
        }
        memory.write_u64(RIP_FS + 0x0FF0, RIP_FS_SEED).unwrap();
        memory.write_u64(RIP_GS + 0x1040, RIP_GS_SEED).unwrap();
    }

    fn check_rip_results(state: &CpuState, memory: &GuestMemory, path: &str) {
        assert_eq!(
            memory.read_u64(RIP_FS + 0x0F08).unwrap(),
            RIP_STORED,
            "{path}: FS store, negative disp"
        );
        assert_eq!(
            memory.read_u64(RIP_GS + 0x1050).unwrap(),
            RIP_STORED,
            "{path}: GS store, positive disp"
        );
        assert_eq!(
            state.get(Register::Rdx),
            RIP_FS_SEED,
            "{path}: FS load, negative disp"
        );
        assert_eq!(
            state.get(Register::Rsi),
            RIP_GS_SEED,
            "{path}: GS load, positive disp"
        );
    }

    #[test]
    fn fs_and_gs_rip_relative_loads_and_stores_use_the_segment_base() {
        let (mut engine, mut cpu, mut memory) = fixture(&RIP_CODE);
        rip_setup(&mut cpu, &mut memory);
        let native = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(native.exit_reason, ExitReason::Halt);
        check_rip_results(&native.final_state, &memory, "native DBT");
    }

    #[test]
    fn interpreter_and_native_dbt_agree_on_rip_relative_segment_accesses() {
        let (mut engine, mut cpu, mut native_memory) = fixture(&RIP_CODE);
        rip_setup(&mut cpu, &mut native_memory);
        let native = engine.run(&mut cpu, &mut native_memory, 16);

        let (_, mut cpu, mut memory) = fixture(&RIP_CODE);
        rip_setup(&mut cpu, &mut memory);
        let interpreted = crate::cpu::Interpreter::new(cpu, &mut memory, 16).run();

        assert_eq!(interpreted.exit_reason, native.exit_reason);
        assert_eq!(interpreted.final_state, native.final_state);
        check_rip_results(&interpreted.final_state, &memory, "interpreter");
        for base in [RIP_FS, RIP_GS] {
            assert_eq!(
                region_bytes(&native_memory, base, 0x2000),
                region_bytes(&memory, base, 0x2000),
                "segment region at {base:#x}"
            );
        }
    }

    #[test]
    fn rip_relative_segment_faults_report_the_segment_adjusted_address() {
        // One instruction per case, with nothing mapped at either segment base.
        let cases: [(&[u8], u64); 4] = [
            // mov fs:[rip - 0x10], rbx   -> FS + 0x1008 - 0x10
            (
                &[0x64, 0x48, 0x89, 0x1D, 0xF0, 0xFF, 0xFF, 0xFF, 0xF4],
                RIP_FS + 0x0FF8,
            ),
            // mov gs:[rip + 0x30], rbx   -> GS + 0x1008 + 0x30
            (
                &[0x65, 0x48, 0x89, 0x1D, 0x30, 0x00, 0x00, 0x00, 0xF4],
                RIP_GS + 0x1038,
            ),
            // mov rdx, fs:[rip - 0x800]  -> FS + 0x1008 - 0x800
            (
                &[0x64, 0x48, 0x8B, 0x15, 0x00, 0xF8, 0xFF, 0xFF, 0xF4],
                RIP_FS + 0x0808,
            ),
            // mov rsi, gs:[rip + 0x8]    -> GS + 0x1008 + 0x8
            (
                &[0x65, 0x48, 0x8B, 0x35, 0x08, 0x00, 0x00, 0x00, 0xF4],
                RIP_GS + 0x1010,
            ),
        ];
        for (code, fault) in cases {
            let expected = ExitReason::MemoryFault(MemoryError::UnmappedAddress(fault));
            let (mut engine, mut cpu, mut memory) = fixture(code);
            cpu.fs_base = RIP_FS;
            cpu.gs_base = RIP_GS;
            let native = engine.run(&mut cpu, &mut memory, 16);
            assert_eq!(
                native.exit_reason, expected,
                "native DBT, fault at {fault:#x}"
            );

            let (_, mut cpu, mut memory) = fixture(code);
            cpu.fs_base = RIP_FS;
            cpu.gs_base = RIP_GS;
            let interpreted = crate::cpu::Interpreter::new(cpu, &mut memory, 16).run();
            assert_eq!(
                interpreted.exit_reason, expected,
                "interpreter, fault at {fault:#x}"
            );
        }
    }

    /// Both execution paths must agree when an instruction carries an FS and
    /// a GS override together.
    ///
    /// The Intel SDM states that "the effect of redundant prefixes (more than
    /// one prefix from a group) is undefined", and FS and GS are in the same
    /// prefix group. The architecture therefore does not define which base
    /// applies, so this test pins the local rule instead: both paths prefer
    /// GS, and they must agree with each other.
    ///
    /// The encoding is `mov [fs:0x8], rbx` preceded by a GS override
    /// (65 64 48 89 1C 25 08 00 00 00).
    #[test]
    fn redundant_fs_and_gs_overrides_are_handled_the_same_by_both_paths() {
        const CODE: [u8; 11] = [
            0x65, 0x64, 0x48, 0x89, 0x1C, 0x25, 0x08, 0x00, 0x00, 0x00, // mov [fs:0x8], rbx
            0xF4, // hlt
        ];
        const FS_BASE: u64 = 0x7000;
        const GS_BASE: u64 = 0x8000;
        const STORED_VALUE: u64 = 0x1122_3344_5566_7788;

        let (mut engine, mut cpu, mut memory) = fixture(&CODE);
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        memory
            .map_region(FS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .map_region(GS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        let native = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(native.exit_reason, ExitReason::Halt);

        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(ENTRY, CODE.len(), MemoryPermissions::READ_EXECUTE, &CODE)
            .unwrap();
        memory
            .map_region(FS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .map_region(GS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut cpu = CpuState::new(ENTRY, 0);
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        let mut interp = crate::cpu::Interpreter::new(cpu, &mut memory, 16);
        let interpreted = interp.run();

        assert_eq!(interpreted.exit_reason, native.exit_reason);
        assert_eq!(interpreted.final_state, native.final_state);
        // The local rule prefers GS, so the store landed at the GS base.
        assert_eq!(memory.read_u64(GS_BASE + 8).unwrap(), STORED_VALUE);
        assert_eq!(memory.read_u64(FS_BASE + 8).unwrap(), 0);
    }

    /// The GS segment override adds the GS base, and distinct FS and GS bases
    /// must not be confused with each other.
    ///
    /// The encodings are `mov [gs:0x8], rbx` (65 48 89 1C 25 08 00 00 00)
    /// and `mov rdx, [gs:0x8]` (65 48 8B 14 25 08 00 00 00), as produced by
    /// LLVM 23.1.6 `llvm-mc -show-encoding`.
    #[test]
    fn gs_segment_override_adds_the_gs_base_to_the_address() {
        let code = [
            0x65, 0x48, 0x89, 0x1C, 0x25, 0x08, 0x00, 0x00, 0x00, // mov [gs:0x8], rbx
            0x65, 0x48, 0x8B, 0x14, 0x25, 0x08, 0x00, 0x00, 0x00, // mov rdx, [gs:0x8]
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        const FS_BASE: u64 = 0x7000;
        const GS_BASE: u64 = 0x8000;
        const STORED_VALUE: u64 = 0x1122_3344_5566_7788;
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        cpu.set(Register::Rbx, STORED_VALUE);
        memory
            .map_region(FS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .map_region(GS_BASE, 0x100, MemoryPermissions::READ_WRITE)
            .unwrap();

        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(memory.read_u64(GS_BASE + 8).unwrap(), STORED_VALUE);
        // The FS base must be untouched.
        assert_eq!(memory.read_u64(FS_BASE + 8).unwrap(), 0);
        assert_eq!(result.final_state.get(Register::Rdx), STORED_VALUE);
    }

    /// A segment-relative access to an unmapped address must fault, and the
    /// fault must report the segment-adjusted address.
    ///
    /// The encoding is `mov [fs:0x8], rbx` (64 48 89 1C 25 08 00 00 00).
    /// The FS base points outside every mapped region, so the store faults at
    /// `FS_BASE + 8`.
    #[test]
    fn segment_relative_access_to_an_unmapped_address_faults() {
        let code = [
            0x64, 0x48, 0x89, 0x1C, 0x25, 0x08, 0x00, 0x00, 0x00, // mov [fs:0x8], rbx
            0xF4, // hlt
        ];
        let (mut engine, mut cpu, mut memory) = fixture(&code);
        const FS_BASE: u64 = 0x7000;
        cpu.fs_base = FS_BASE;
        cpu.set(Register::Rbx, 0x1122_3344_5566_7788);
        // No region is mapped at FS_BASE, so the segment-adjusted address
        // FS_BASE + 8 is unmapped.

        let result = engine.run(&mut cpu, &mut memory, 16);
        assert_eq!(
            result.exit_reason,
            ExitReason::MemoryFault(MemoryError::UnmappedAddress(FS_BASE + 8))
        );
    }

    /// Intel SDM Volume 1, section 3.4.3.1: for `SUB` and `CMP`, `CF` is the
    /// borrow out of the most significant bit. `JB` is `CF = 1`, `JAE` is
    /// `CF = 0`, `JBE` is `CF = 1 or ZF = 1` and `JA` is `CF = 0 and ZF = 0`.
    ///
    /// ARM64 reports the inverse: `SUBS` sets `C` when the subtraction does
    /// *not* borrow, and its `HI`/`LS` conditions are defined on that same
    /// inverted carry. These four x86 conditions therefore cannot be taken
    /// from an ARM64 condition code without translating the borrow first,
    /// which makes them the ones to pin at the boundaries.
    /// One unsigned condition: the low nibble pair as encoded in the `0x70`
    /// block, and the rule that decides it from CF and ZF.
    type UnsignedCondition = (u8, fn(bool, bool) -> bool);

    const UNSIGNED_CONDITIONS: [UnsignedCondition; 4] = [
        (0x72, |cf, _zf| cf),        // JB, JC, JNAE
        (0x73, |cf, _zf| !cf),       // JAE, JNB, JNC
        (0x76, |cf, zf| cf || zf),   // JBE, JNA
        (0x77, |cf, zf| !cf && !zf), // JA, JNBE
    ];

    /// Build `<op> eax, ebx` (or `rax, rbx`) followed by the short or near form
    /// of `condition`. The not-taken path writes 1 to `ECX` and halts, the
    /// taken path writes 2. A near branch targets offset 140, which a `disp8`
    /// cannot reach, so the two encodings are exercised separately.
    fn unsigned_branch_program(op: [u8; 2], condition: u8, wide: bool, near: bool) -> Vec<u8> {
        let mut code = Vec::new();
        if wide {
            code.push(0x48); // REX.W
        }
        code.extend_from_slice(&op); // cmp/sub eax, ebx
        let jcc = code.len();
        if near {
            // The near form is `0F 8x`, where x is the low nibble of the
            // short form's opcode.
            code.extend_from_slice(&[0x0F, 0x80 | (condition & 0x0F), 0, 0, 0, 0]);
        } else {
            code.extend_from_slice(&[condition, 0]);
        }
        code.extend_from_slice(&[0xB9, 1, 0, 0, 0]); // mov ecx, 1
        code.push(0xF4); // hlt
        let target = if near { 140 } else { code.len() };
        while code.len() < target {
            code.push(0x90); // nop
        }
        code.extend_from_slice(&[0xB9, 2, 0, 0, 0]); // mov ecx, 2
        code.push(0xF4); // hlt
        let next = jcc + if near { 6 } else { 2 };
        let displacement = target as i64 - next as i64;
        if near {
            code[jcc + 2..jcc + 6].copy_from_slice(&(displacement as i32).to_le_bytes());
        } else {
            code[jcc + 1] = displacement as i8 as u8;
        }
        code
    }

    /// Run one branch program and return `(ecx, cf, zf)`.
    fn run_unsigned_branch(code: &[u8], left: u64, right: u64) -> (u64, bool, bool) {
        let (mut engine, mut cpu, mut memory) = fixture(code);
        cpu.set(Register::Rax, left);
        cpu.set(Register::Rbx, right);
        let result = engine.run(&mut cpu, &mut memory, 32);
        assert_eq!(result.exit_reason, ExitReason::Halt);
        // Either the two-instruction taken path or the two-instruction
        // not-taken path runs after the compare and the branch.
        assert_eq!(result.instructions_executed, 4);
        // The compare, the branch and the halting move all run as translated
        // ARM64, not through the shared interpreter.
        assert_eq!(engine.metrics.interpreter_fallbacks, 0);
        (cpu.get(Register::Rcx), cpu.flags.cf, cpu.flags.zf)
    }

    /// The four unsigned conditions, at both ends of the 32-bit and 64-bit
    /// ranges, in the short and the near encoding. Expected flags and
    /// outcomes come from the SDM rules above, not from the lowering.
    #[test]
    fn unsigned_branch_conditions_follow_the_borrow_at_every_boundary() {
        const CMP: [u8; 2] = [0x39, 0xD8];
        let cases: &[(bool, u64, u64)] = &[
            // 32-bit: zero, one, equal operands, both ends of the range, and
            // the pair where the sign bit flips but the unsigned order holds.
            (false, 0, 0),
            (false, 0, 1),
            (false, 1, 0),
            (false, 0xFFFF_FFFF, 0xFFFF_FFFF),
            (false, 0xFFFF_FFFF, 0),
            (false, 0, 0xFFFF_FFFF),
            (false, 0x7FFF_FFFF, 0x8000_0000),
            (false, 0x8000_0000, 0x7FFF_FFFF),
            // The same boundaries at 64 bits, plus a subtraction that clears
            // the sign bit without borrowing.
            (true, 0, 0),
            (true, 0, 1),
            (true, 1, 0),
            (true, u64::MAX, u64::MAX),
            (true, u64::MAX, 0),
            (true, 0, u64::MAX),
            (true, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000),
            (true, 0x8000_0000_0000_0000, 0x7FFF_FFFF_FFFF_FFFF),
            (true, 0x8000_0000_0000_0000, 1),
        ];

        for &(wide, left, right) in cases {
            let mask = if wide { u64::MAX } else { u64::from(u32::MAX) };
            let (left, right) = (left & mask, right & mask);
            // SDM: CF is the borrow out of the most significant bit, ZF is a
            // zero result. Both follow from the operands alone.
            let (cf, zf) = (left < right, left == right);
            for &(condition, holds) in &UNSIGNED_CONDITIONS {
                let taken = holds(cf, zf);
                for &near in &[false, true] {
                    let code = unsigned_branch_program(CMP, condition, wide, near);
                    let (ecx, got_cf, got_zf) = run_unsigned_branch(&code, left, right);
                    let case = format!(
                        "{condition:#04x} {:#x} {:#x} {}",
                        left,
                        right,
                        if near { "near" } else { "short" }
                    );
                    assert_eq!(got_cf, cf, "{case}: CF");
                    assert_eq!(got_zf, zf, "{case}: ZF");
                    assert_eq!(ecx, if taken { 2 } else { 1 }, "{case}: branch outcome");
                }
            }
        }
    }

    /// `SUB` is the other producer of the carry these conditions read, and the
    /// one where the ARM64 carry is inverted: `0 - 1` borrows, while
    /// `0x8000_0000_0000_0000 - 1` does not borrow even though it clears the
    /// sign bit.
    #[test]
    fn subtract_borrow_into_and_out_of_the_high_bit_feeds_the_unsigned_branches() {
        const SUB: [u8; 2] = [0x29, 0xD8];
        let cases: &[(bool, u64, u64)] = &[
            (false, 0, 1),
            (false, 0x8000_0000, 1),
            (true, 0, 1),
            (true, 0x8000_0000_0000_0000, 1),
        ];

        for &(wide, left, right) in cases {
            let mask = if wide { u64::MAX } else { u64::from(u32::MAX) };
            let (left, right) = (left & mask, right & mask);
            let (cf, zf) = (left < right, left == right);
            for &(condition, holds) in &UNSIGNED_CONDITIONS {
                let taken = holds(cf, zf);
                for &near in &[false, true] {
                    let code = unsigned_branch_program(SUB, condition, wide, near);
                    let (ecx, got_cf, got_zf) = run_unsigned_branch(&code, left, right);
                    let case = format!(
                        "{condition:#04x} {:#x} {:#x} {}",
                        left,
                        right,
                        if near { "near" } else { "short" }
                    );
                    assert_eq!(got_cf, cf, "{case}: CF");
                    assert_eq!(got_zf, zf, "{case}: ZF");
                    assert_eq!(ecx, if taken { 2 } else { 1 }, "{case}: branch outcome");
                }
            }
        }
    }

    /// The carry these conditions read is produced in one translated block and
    /// consumed in the next. A lowering that left it in the host's condition
    /// flags would pass the single-block cases above and fail here.
    #[test]
    fn unsigned_branch_reads_flags_stored_by_a_previous_block() {
        const CMP: [u8; 2] = [0x39, 0xD8];
        let cases: &[(bool, u64, u64)] = &[
            (false, 0, 0),                    // CF = 0, ZF = 1
            (false, 0, 1),                    // CF = 1, ZF = 0
            (false, 0xFFFF_FFFF, 0),          // CF = 0, ZF = 0
            (true, 0x8000_0000_0000_0000, 1), // CF = 0 although the sign bit clears
        ];

        for &(wide, left, right) in cases {
            let (cf, zf) = (left < right, left == right);
            for &(condition, holds) in &UNSIGNED_CONDITIONS {
                let taken = holds(cf, zf);
                let code = unsigned_branch_program(CMP, condition, wide, false);
                let (mut engine, mut cpu, mut memory) = fixture(&code);
                cpu.set(Register::Rax, left);
                cpu.set(Register::Rbx, right);

                // One instruction per run: the compare is translated as a
                // one-instruction block, so the branch is compiled separately
                // and can only read the flags the first block stored.
                let first = engine.run(&mut cpu, &mut memory, 1);
                assert_eq!(first.exit_reason, ExitReason::StepLimitExceeded);
                assert_eq!(cpu.rip, ENTRY + if wide { 3 } else { 2 });
                assert_eq!((cpu.flags.cf, cpu.flags.zf), (cf, zf));

                let second = engine.run(&mut cpu, &mut memory, 8);
                assert_eq!(second.exit_reason, ExitReason::Halt);
                assert_eq!(second.instructions_executed, 3);
                assert_eq!(cpu.get(Register::Rcx), if taken { 2 } else { 1 });
                assert_eq!(engine.metrics.interpreter_fallbacks, 0);
            }
        }
    }

    // Direct block linking. Every program below is written for these tests;
    // each runs on the interpreter and on fresh engines with and without links.

    const LINK_DATA: u64 = 0x8000;
    const LINK_DATA_LEN: usize = 0x40;
    const LINK_STACK_TOP: u64 = 0x3_0000;

    /// `code` at ENTRY, a small data region and a stack, all outside the code.
    fn link_memory(code: &[u8], permissions: MemoryPermissions) -> GuestMemory {
        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(ENTRY, code.len(), permissions, code)
            .unwrap();
        let data: Vec<u8> = (0..LINK_DATA_LEN as u8).collect();
        memory
            .map_region_with_data(
                LINK_DATA,
                LINK_DATA_LEN,
                MemoryPermissions::READ_WRITE,
                &data,
            )
            .unwrap();
        memory
            .map_region(
                LINK_STACK_TOP - 0x1000,
                0x1000,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        memory
    }

    /// One engine's run: its result, the data and stack bytes, and the link
    /// counters (all zero for the interpreter).
    #[derive(Debug, PartialEq)]
    struct Outcome {
        exit_reason: ExitReason,
        instructions_executed: usize,
        final_state: CpuState,
        data: Vec<u8>,
        stack: Vec<u8>,
    }

    fn outcome(result: ExecutionResult, memory: &GuestMemory) -> Outcome {
        Outcome {
            exit_reason: result.exit_reason,
            instructions_executed: result.instructions_executed,
            final_state: result.final_state,
            data: region_bytes(memory, LINK_DATA, LINK_DATA_LEN),
            stack: region_bytes(memory, LINK_STACK_TOP - 0x1000, 0x1000),
        }
    }

    fn link_cpu() -> CpuState {
        CpuState::new(ENTRY, LINK_STACK_TOP)
    }

    fn interpreted(code: &[u8], permissions: MemoryPermissions, budget: usize) -> Outcome {
        let mut memory = link_memory(code, permissions);
        let result = crate::cpu::Interpreter::new(link_cpu(), &mut memory, budget).run();
        outcome(result, &memory)
    }

    fn translated(
        code: &[u8],
        permissions: MemoryPermissions,
        budget: usize,
        linking: bool,
    ) -> (Outcome, DbtEngine) {
        let mut memory = link_memory(code, permissions);
        let mut engine = DbtEngine::new(256 * 1024).unwrap();
        engine.set_block_linking(linking);
        let result = engine.run(&mut link_cpu(), &mut memory, budget);
        (outcome(result, &memory), engine)
    }

    /// Run `code` three ways, check that they agree, and return the linked
    /// engine.
    fn agree_three_ways(code: &[u8], permissions: MemoryPermissions, budget: usize) -> DbtEngine {
        let expected = interpreted(code, permissions, budget);
        let (unlinked, unlinked_engine) = translated(code, permissions, budget, false);
        let (linked, linked_engine) = translated(code, permissions, budget, true);
        assert_eq!(unlinked, expected, "unlinked DBT, budget {budget}");
        assert_eq!(linked, expected, "linked DBT, budget {budget}");
        assert_eq!(unlinked_engine.link_metrics.links_followed, 0);
        assert_eq!(unlinked_engine.link_metrics.links_installed, 0);
        linked_engine
    }

    /// `rel8` from the end of an instruction at `next` to `target`.
    fn rel8(next: u64, target: u64) -> u8 {
        let offset = target.wrapping_sub(next) as i64;
        assert!((-128..=127).contains(&offset));
        offset as i8 as u8
    }

    /// mov ecx, count; xor eax, eax; L: add rax, 3; sub rcx, 1; jnz L; hlt.
    /// 2 + 3 * count + 1 instructions.
    fn counted_loop(count: u8) -> Vec<u8> {
        let mut code = vec![0xB9, count, 0, 0, 0, 0x31, 0xC0];
        let head = ENTRY + code.len() as u64;
        code.extend_from_slice(&[0x48, 0x83, 0xC0, 0x03, 0x48, 0x83, 0xE9, 0x01, 0x75]);
        let next = ENTRY + code.len() as u64 + 1;
        code.push(rel8(next, head));
        code.push(0xF4);
        code
    }

    #[test]
    fn linked_loop_agrees_with_the_interpreter_and_the_unlinked_dbt() {
        let code = counted_loop(200);
        let engine = agree_three_ways(&code, MemoryPermissions::READ_EXECUTE, 10_000);
        let links = engine.link_metrics;
        // The loop block links to itself once the dispatcher has installed
        // that link: every later iteration but the last two, whose budget
        // or exit leaves through the dispatcher, follows it.
        assert!(links.links_followed >= 190, "{links:?}");
        assert!(links.links_installed >= 1, "{links:?}");
        assert_eq!(engine.metrics.interpreter_fallbacks, 0);
        // Followed links skip the dispatcher, so it entered native code far
        // fewer times than blocks ran.
        assert!(engine.metrics.dbt_dispatches < 20, "{:?}", engine.metrics);
    }

    #[test]
    fn links_respect_every_exact_budget() {
        let code = counted_loop(30);
        let total = 2 + 3 * 30 + 1;
        for budget in 0..=total + 3 {
            let engine = agree_three_ways(&code, MemoryPermissions::READ_EXECUTE, budget);
            if budget > 48 {
                assert!(engine.link_metrics.links_followed > 0, "budget {budget}");
            }
        }
    }

    #[test]
    fn a_run_cut_by_the_budget_resumes_to_the_uncut_result() {
        let code = counted_loop(30);
        let mut memory = link_memory(&code, MemoryPermissions::READ_EXECUTE);
        let mut engine = DbtEngine::new(256 * 1024).unwrap();
        let whole = engine.run(&mut link_cpu(), &mut memory, 1_000);
        assert_eq!(whole.exit_reason, ExitReason::Halt);
        for cut in [1, 15, 16, 17, 31, 32, 33, 50, 64, 80] {
            let mut cpu = link_cpu();
            let first = engine.run(&mut cpu, &mut memory, cut);
            assert_eq!(first.exit_reason, ExitReason::StepLimitExceeded);
            assert_eq!(first.instructions_executed, cut);
            let rest = engine.run(&mut cpu, &mut memory, 1_000);
            assert_eq!(rest.exit_reason, ExitReason::Halt);
            assert_eq!(
                cut + rest.instructions_executed,
                whole.instructions_executed
            );
            assert_eq!(rest.final_state, whole.final_state);
        }
        assert!(engine.link_metrics.links_followed > 0);
    }

    #[test]
    fn linked_slices_stop_where_unlinked_slices_stop() {
        let code = counted_loop(30);
        for budget in 1..=100 {
            let mut results = [true, false].map(|linking| {
                let mut memory = link_memory(&code, MemoryPermissions::READ_EXECUTE);
                let mut engine = DbtEngine::new(256 * 1024).unwrap();
                engine.set_block_linking(linking);
                let result = engine.run_slice(&mut link_cpu(), &mut memory, budget);
                assert!(result.instructions_executed < budget + MAX_BLOCK_INSTRUCTIONS);
                (outcome(result, &memory), engine.link_metrics.links_followed)
            });
            assert_eq!(results[1].1, 0);
            // The loop block's link to itself is installed when the
            // dispatcher selects that block again, on the third pass, and
            // followed at its end, 5 + 3 + 3 instructions in.
            if budget > 11 {
                assert!(results[0].1 > 0, "budget {budget}");
            }
            results[0].1 = 0;
            assert_eq!(results[0], results[1], "budget {budget}");
        }
    }

    #[test]
    fn static_successors_cover_quota_jumps_calls_and_both_branch_sides() {
        fn successors(code: &[u8]) -> Vec<u64> {
            let fetch = |address: u64| {
                let offset = address.wrapping_sub(ENTRY) as usize;
                code.get(offset)
                    .copied()
                    .ok_or(MemoryError::UnmappedAddress(address))
            };
            let insn = decode(ENTRY, fetch).unwrap();
            let (successors, count) = static_successors(&[insn], insn.next_rip());
            successors[..count].to_vec()
        }
        let next = |len: u64| ENTRY + len;
        assert_eq!(successors(&[0x48, 0x83, 0xC0, 0x01]), [next(4)]); // add rax, 1
        assert_eq!(successors(&[0xEB, 0x10]), [next(2) + 0x10]); // jmp +16
        assert_eq!(successors(&[0x75, 0xF0]), [next(2) - 0x10, next(2)]); // jnz -16
        assert_eq!(successors(&[0x74, 0x00]), [next(2)]); // jz to the next insn
        assert_eq!(successors(&[0xE8, 0x20, 0, 0, 0]), [next(5) + 0x20]); // call +32
        assert!(successors(&[0xC3]).is_empty()); // ret
        assert!(successors(&[0xFF, 0xE0]).is_empty()); // jmp rax
        assert!(successors(&[0xFF, 0xD0]).is_empty()); // call rax
        assert!(successors(&[0xF4]).is_empty()); // hlt
    }

    #[test]
    fn quota_fallthrough_and_relative_calls_are_linked() {
        // mov ecx, 5
        // L: add rax, 1 (20 times; the loop body spans a block quota)
        //    call F; sub rcx, 1; jnz L; hlt
        // F: add rbx, 2; ret
        let mut code = vec![0xB9, 5, 0, 0, 0];
        let head = ENTRY + code.len() as u64;
        for _ in 0..20 {
            code.extend_from_slice(&[0x48, 0x83, 0xC0, 0x01]);
        }
        let call = code.len();
        code.extend_from_slice(&[0xE8, 0, 0, 0, 0, 0x48, 0x83, 0xE9, 0x01, 0x75]);
        let jnz_next = ENTRY + code.len() as u64 + 1;
        code.push(rel8(jnz_next, head));
        code.push(0xF4);
        let function = code.len() as u32;
        code[call + 1..call + 5].copy_from_slice(&(function - (call as u32 + 5)).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x83, 0xC3, 0x02, 0xC3]);

        let engine = agree_three_ways(&code, MemoryPermissions::READ_EXECUTE, 10_000);
        let expected = interpreted(&code, MemoryPermissions::READ_EXECUTE, 10_000);
        assert_eq!(expected.final_state.get(Register::Rax), 100);
        assert_eq!(expected.final_state.get(Register::Rbx), 10);
        // From the third iteration on, the quota fallthrough, the call and
        // the taken branch each follow a link; RET returns through the
        // dispatcher.
        assert!(
            engine.link_metrics.links_followed >= 9,
            "{:?}",
            engine.link_metrics
        );
    }

    #[test]
    fn stack_stores_outside_translated_code_keep_links() {
        // mov ecx, 10; L: push rcx; pop rbx; add rax, rbx; sub rcx, 1; jnz L; hlt
        let mut code = vec![0xB9, 10, 0, 0, 0];
        let head = ENTRY + code.len() as u64;
        code.extend_from_slice(&[0x51, 0x5B, 0x48, 0x01, 0xD8, 0x48, 0x83, 0xE9, 0x01, 0x75]);
        let next = ENTRY + code.len() as u64 + 1;
        code.push(rel8(next, head));
        code.push(0xF4);
        let engine = agree_three_ways(&code, MemoryPermissions::READ_EXECUTE, 1_000);
        // The first pass runs in the entry block, the second installs the
        // loop block's link to itself, and passes 4 to 10 enter through it.
        assert_eq!(
            engine.link_metrics.links_followed, 7,
            "{:?}",
            engine.link_metrics
        );
    }

    #[test]
    fn shared_semantics_instruction_ends_every_chain() {
        // mov ecx, 10; L: add al, 1; sub rcx, 1; jnz L; hlt. The 8-bit ADD
        // runs through the shared semantics, which may store anywhere.
        let mut code = vec![0xB9, 10, 0, 0, 0];
        let head = ENTRY + code.len() as u64;
        code.extend_from_slice(&[0x04, 0x01, 0x48, 0x83, 0xE9, 0x01, 0x75]);
        let next = ENTRY + code.len() as u64 + 1;
        code.push(rel8(next, head));
        code.push(0xF4);
        let engine = agree_three_ways(&code, MemoryPermissions::READ_EXECUTE, 1_000);
        assert_eq!(engine.metrics.interpreter_fallbacks, 10);
        assert_eq!(engine.link_metrics.links_followed, 0);
    }

    #[test]
    fn store_into_a_linked_block_is_translated_again() {
        // mov ecx, 4; mov edi, IMM; xor eax, eax
        // B: add rax, imm8 (at IMM); jmp C
        // C: mov [rdi], cl; sub rcx, 1; jnz B; hlt
        // Each pass rewrites B's immediate with the counter, then branches
        // back to B, which the dispatcher verified in the current epoch.
        let b = ENTRY + 12;
        let mut code = vec![0xB9, 4, 0, 0, 0, 0xBF];
        code.extend_from_slice(&((b + 3) as u32).to_le_bytes());
        code.extend_from_slice(&[0x31, 0xC0, 0x48, 0x83, 0xC0, 0x01, 0xEB, 0x00]);
        code.extend_from_slice(&[0x88, 0x0F, 0x48, 0x83, 0xE9, 0x01, 0x75]);
        let next = ENTRY + code.len() as u64 + 1;
        code.push(rel8(next, b));
        code.push(0xF4);

        let rwx = MemoryPermissions::READ_WRITE_EXECUTE;
        let engine = agree_three_ways(&code, rwx, 1_000);
        let expected = interpreted(&code, rwx, 1_000);
        // 1 + 4 + 3 + 2: every pass used the immediate the previous one wrote.
        assert_eq!(expected.final_state.get(Register::Rax), 10);
        assert!(
            engine.link_metrics.targets_retired >= 2,
            "{:?}",
            engine.link_metrics
        );
    }

    #[test]
    fn link_into_another_address_space_is_not_followed() {
        // mov ecx, 10; xor eax, eax; jmp L; L: add rax, K; sub rcx, 1; jnz L; hlt
        let program = |k: u8| {
            let mut code = vec![0xB9, 10, 0, 0, 0, 0x31, 0xC0, 0xEB, 0x00];
            let head = ENTRY + code.len() as u64;
            code.extend_from_slice(&[0x48, 0x83, 0xC0, k, 0x48, 0x83, 0xE9, 0x01, 0x75]);
            let next = ENTRY + code.len() as u64 + 1;
            code.push(rel8(next, head));
            code.push(0xF4);
            code
        };
        let mut engine = DbtEngine::new(256 * 1024).unwrap();
        for k in [1, 5, 1] {
            let code = program(k);
            let mut memory = link_memory(&code, MemoryPermissions::READ_EXECUTE);
            let followed = engine.link_metrics.links_followed;
            let result = engine.run(&mut link_cpu(), &mut memory, 1_000);
            assert_eq!(
                outcome(result, &memory),
                interpreted(&code, MemoryPermissions::READ_EXECUTE, 1_000)
            );
            assert_eq!(memory.read_u8(ENTRY + 12).unwrap(), k);
            assert!(engine.link_metrics.links_followed > followed);
        }
        // The entry block is the same in both programs; its link to the loop
        // block was refused, and the loop block was translated again, twice.
        assert_eq!(engine.link_metrics.targets_retired, 2);
    }

    #[test]
    fn host_write_between_runs_is_seen_through_a_link() {
        // mov ecx, 50; A: add rax, 1; jmp B; B: add rbx, K; sub rcx, 1; jnz A; hlt
        let code = [
            0xB9, 50, 0, 0, 0, // mov ecx, 50
            0x48, 0x83, 0xC0, 0x01, // A: add rax, 1
            0xEB, 0x00, // jmp B
            0x48, 0x83, 0xC3, 0x02, // B: add rbx, K
            0x48, 0x83, 0xE9, 0x01, // sub rcx, 1
            0x75, 0xF0, // jnz A
            0xF4, // hlt
        ];
        let immediate = ENTRY + 14;
        let rwx = MemoryPermissions::READ_WRITE_EXECUTE;
        // 3 entry instructions, then B and 10 passes of A and B: the run
        // stops at A, whose full block links to B.
        let cut = 3 + 3 + 10 * 5;

        let mut memory = link_memory(&code, rwx);
        let first = crate::cpu::Interpreter::new(link_cpu(), &mut memory, cut).run();
        memory.write_bytes(immediate, &[7]).unwrap();
        let second = crate::cpu::Interpreter::new(first.final_state, &mut memory, 1_000).run();
        let expected = outcome(second, &memory);

        let mut memory = link_memory(&code, rwx);
        let mut engine = DbtEngine::new(256 * 1024).unwrap();
        let mut cpu = link_cpu();
        let first = engine.run(&mut cpu, &mut memory, cut);
        assert_eq!(first.exit_reason, ExitReason::StepLimitExceeded);
        assert_eq!(cpu.rip, ENTRY + 5);
        assert!(engine.link_metrics.links_followed > 0);
        // The host rewrites B while the engine is between runs.
        memory.write_bytes(immediate, &[7]).unwrap();
        let second = engine.run(&mut cpu, &mut memory, 1_000);
        assert_eq!(outcome(second, &memory), expected);
        assert_eq!(engine.link_metrics.targets_retired, 1);
    }

    #[test]
    fn service_registered_at_a_linked_target_runs() {
        // mov ecx, 3; L: add rax, 1; jmp S; S: sub rcx, 1; jnz L; hlt
        let code = [
            0xB9, 3, 0, 0, 0, // mov ecx, 3
            0x48, 0x83, 0xC0, 0x01, // L: add rax, 1
            0xEB, 0x00, // jmp S
            0x48, 0x83, 0xE9, 0x01, // S: sub rcx, 1
            0x75, 0xF4, // jnz L
            0xF4, // hlt
        ];
        let service = ENTRY + 11;
        let hlt = ENTRY + 17;
        let mut memory = link_memory(&code, MemoryPermissions::READ_EXECUTE);
        let mut engine = DbtEngine::new(256 * 1024).unwrap();
        let first = engine.run(&mut link_cpu(), &mut memory, 1_000);
        assert_eq!(first.final_state.get(Register::Rax), 3);
        assert!(engine.link_metrics.links_followed > 0);

        engine.register_service(service, move |cpu, _| {
            cpu.set(Register::Rbx, 0x5E);
            cpu.rip = hlt;
            Ok(())
        });
        let second = engine.run(&mut link_cpu(), &mut memory, 1_000);
        assert_eq!(second.exit_reason, ExitReason::Halt);
        assert_eq!(second.final_state.get(Register::Rax), 1);
        assert_eq!(second.final_state.get(Register::Rbx), 0x5E);
        assert_eq!(engine.metrics.service_calls, 1);
    }

    #[test]
    fn service_that_rewrites_code_ends_the_epoch() {
        // mov ecx, 5; B: add rbx, K; call SVC; R: sub rcx, 1; jnz B; hlt.
        // The service returns to R and increments K, the immediate of B,
        // which R's link enters next.
        const SERVICE: u64 = 0x1100;
        let mut code = vec![0xB9, 5, 0, 0, 0, 0x48, 0x83, 0xC3, 0x01, 0xE8];
        let call_next = ENTRY + code.len() as u64 + 4;
        code.extend_from_slice(&(SERVICE.wrapping_sub(call_next) as u32).to_le_bytes());
        code.extend_from_slice(&[0x48, 0x83, 0xE9, 0x01, 0x75]);
        let next = ENTRY + code.len() as u64 + 1;
        code.push(rel8(next, ENTRY + 5));
        code.push(0xF4);
        let immediate = ENTRY + 8;

        for linking in [true, false] {
            let mut memory = link_memory(&code, MemoryPermissions::READ_WRITE_EXECUTE);
            let mut engine = DbtEngine::new(256 * 1024).unwrap();
            engine.set_block_linking(linking);
            engine.register_service(SERVICE, move |cpu, memory| {
                let rsp = cpu.get(Register::Rsp);
                cpu.rip = memory.read_u64(rsp).map_err(ExitReason::MemoryFault)?;
                cpu.set(Register::Rsp, rsp + 8);
                let k = memory.read_u8(immediate).map_err(ExitReason::MemoryFault)?;
                memory
                    .write_bytes(immediate, &[k + 1])
                    .map_err(ExitReason::MemoryFault)?;
                Ok(())
            });
            let result = engine.run(&mut link_cpu(), &mut memory, 1_000);
            assert_eq!(result.exit_reason, ExitReason::Halt);
            // 1 + 2 + 3 + 4 + 5: every pass saw the immediate the service wrote.
            assert_eq!(
                result.final_state.get(Register::Rbx),
                15,
                "linking {linking}"
            );
            assert_eq!(engine.metrics.service_calls, 5);
        }
    }

    #[test]
    fn fault_inside_a_chain_reports_the_completed_instructions() {
        // mov esi, LINK_DATA; L: mov rax, [rsi]; add rsi, 8; jmp L. The ninth
        // load leaves the data region.
        let code = [
            0xBE, 0x00, 0x80, 0x00, 0x00, // mov esi, 0x8000
            0x48, 0x8B, 0x06, // L: mov rax, [rsi]
            0x48, 0x83, 0xC6, 0x08, // add rsi, 8
            0xEB, 0xF7, // jmp L
        ];
        let engine = agree_three_ways(&code, MemoryPermissions::READ_EXECUTE, 1_000);
        let expected = interpreted(&code, MemoryPermissions::READ_EXECUTE, 1_000);
        assert!(matches!(expected.exit_reason, ExitReason::MemoryFault(_)));
        assert_eq!(expected.instructions_executed, 1 + 8 * 3);
        assert_eq!(expected.final_state.rip, ENTRY + 5);
        assert!(
            engine.link_metrics.links_followed >= 6,
            "{:?}",
            engine.link_metrics
        );
    }

    #[test]
    fn retranslation_does_not_rewrite_emitted_code() {
        // Linking only writes host data: a warm run emits nothing and keeps
        // the code cache's committed size.
        let code = counted_loop(50);
        let mut memory = link_memory(&code, MemoryPermissions::READ_EXECUTE);
        let mut engine = DbtEngine::new(256 * 1024).unwrap();
        engine.run(&mut link_cpu(), &mut memory, 1_000);
        let (used, emitted) = (engine.jit.used(), engine.metrics.native_bytes_emitted);
        let followed = engine.link_metrics.links_followed;
        engine.run(&mut link_cpu(), &mut memory, 1_000);
        assert_eq!(engine.jit.used(), used);
        assert_eq!(engine.metrics.native_bytes_emitted, emitted);
        assert!(engine.link_metrics.links_followed > followed);
    }
}
