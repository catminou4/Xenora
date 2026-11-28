// SPDX-License-Identifier: GPL-3.0-or-later

//! Minimal x86-64 guest CPU register state and bounded instruction interpreter.
//!
//! Executes real x86-64 opcodes in software against isolated `GuestMemory`.
//! Never executes guest x86-64 code natively on host ARM64.

use crate::memory::{GuestMemory, MemoryError};
use crate::x86_decode::{DecodeError, decode};
use crate::x86_exec;
use std::fmt;

/// 64-bit General Purpose Register identifiers in x86-64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Register {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Register {
    pub fn from_index(index: u8) -> Self {
        match index & 0xF {
            0 => Self::Rax,
            1 => Self::Rcx,
            2 => Self::Rdx,
            3 => Self::Rbx,
            4 => Self::Rsp,
            5 => Self::Rbp,
            6 => Self::Rsi,
            7 => Self::Rdi,
            8 => Self::R8,
            9 => Self::R9,
            10 => Self::R10,
            11 => Self::R11,
            12 => Self::R12,
            13 => Self::R13,
            14 => Self::R14,
            15 => Self::R15,
            _ => unreachable!(),
        }
    }
}

/// RFLAGS status bits tracked during guest execution.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RFlags {
    pub cf: bool, // Carry flag
    pub zf: bool, // Zero flag
    pub sf: bool, // Sign flag
    pub of: bool, // Overflow flag
    pub pf: bool, // Parity flag
    pub af: bool, // Auxiliary carry flag
}

/// SSE control and status register. Reset state masks every exception and
/// rounds to nearest (Intel SDM Vol. 1, 10.2.3).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mxcsr(pub u32);

impl Mxcsr {
    pub const RESET: u32 = 0x1F80;
    /// Bits that exist; loading any other bit raises #GP.
    pub const WRITABLE: u32 = 0xFFFF;
}

impl Default for Mxcsr {
    fn default() -> Self {
        Self(Self::RESET)
    }
}

/// x87 FPU control, status and tag words.
///
/// FNINIT is the only instruction that writes them and nothing reads them yet.
/// The x87 data registers and every other x87 instruction are not modelled.
/// The host's floating-point state is never involved.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X87State {
    pub control: u16,
    pub status: u16,
    pub tag: u16,
}

impl X87State {
    /// What FINIT and FNINIT set (Intel SDM Vol. 2A, "FINIT/FNINIT"; Vol. 1,
    /// 8.1.3 to 8.1.7): every exception masked, 64-bit precision, round to
    /// nearest; no flag, top of stack 0; every register tagged empty.
    pub const INIT: Self = Self {
        control: 0x037F,
        status: 0,
        tag: 0xFFFF,
    };
}

impl Default for X87State {
    fn default() -> Self {
        Self::INIT
    }
}

/// x86-64 CPU state. Native DBT accesses the general registers and RIP by offset.
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuState {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub flags: RFlags,
    pub gs_base: u64,
    pub fs_base: u64,
    /// XMM0-XMM15, lane 0 in the low bits.
    pub xmm: [u128; 16],
    pub mxcsr: Mxcsr,
    /// Direction flag: string instructions step RSI and RDI down when set.
    pub df: bool,
    /// Starts as FNINIT leaves it. The value an operating system gives a new
    /// thread is not modelled.
    pub x87: X87State,
    /// Bits 255:128 of YMM0-YMM15. Legacy SSE instructions never change them.
    pub ymm_high: [u128; 16],
    /// While a repeated CMPS or SCAS is part-way through, its address and the
    /// status flags it began with. A fault on a later element restores those
    /// flags (Intel SDM Vol. 2B, REP/REPE/REPZ/REPNE/REPNZ). Each element is a
    /// step of its own, so this is not architectural state. It is kept here so
    /// that where a step budget splits a repeat cannot change its result.
    pub repeat_flags: Option<(u64, RFlags)>,
}

impl CpuState {
    pub fn new(entry_point: u64, stack_pointer: u64) -> Self {
        let mut gpr = [0u64; 16];
        gpr[Register::Rsp as usize] = stack_pointer;
        Self {
            gpr,
            rip: entry_point,
            flags: RFlags::default(),
            gs_base: 0,
            fs_base: 0,
            xmm: [0; 16],
            mxcsr: Mxcsr::default(),
            df: false,
            x87: X87State::INIT,
            ymm_high: [0; 16],
            repeat_flags: None,
        }
    }

    pub fn get(&self, reg: Register) -> u64 {
        self.gpr[reg as usize]
    }

    pub fn set(&mut self, reg: Register, val: u64) {
        self.gpr[reg as usize] = val;
    }
}

/// CPUID leaf 1, EDX: the time-stamp counter and RDTSC.
pub const CPUID_TSC: u32 = 1 << 4;
/// CPUID leaf 1, EDX: CMPXCHG8B.
pub const CPUID_CX8: u32 = 1 << 8;
/// CPUID leaf 1, EDX: CMOVcc.
pub const CPUID_CMOV: u32 = 1 << 15;
/// CPUID leaf 1, ECX: CMPXCHG16B.
pub const CPUID_CMPXCHG16B: u32 = 1 << 13;
/// CPUID leaf 1, ECX: POPCNT.
pub const CPUID_POPCNT: u32 = 1 << 23;
/// CPUID leaf 0x8000_0001, ECX: LAHF and SAHF in 64-bit mode.
pub const CPUID_LAHF_SAHF: u32 = 1;
/// CPUID leaf 0x8000_0001, ECX: LZCNT.
pub const CPUID_LZCNT: u32 = 1 << 5;
/// CPUID leaf 0x8000_0001, EDX: 64-bit mode.
pub const CPUID_LONG_MODE: u32 = 1 << 29;
/// CPUID leaf 7 subleaf 0, EBX: bit-manipulation instruction set 1.
pub const CPUID7_BMI1: u32 = 1 << 3;
/// CPUID leaf 7 subleaf 0, EBX: bit-manipulation instruction set 2.
pub const CPUID7_BMI2: u32 = 1 << 8;

/// What the guest's CPUID returns in EAX, EBX, ECX and EDX.
///
/// The table is fixed and never reads the host. It names a synthetic
/// processor, not an Intel or AMD product, and sets a feature bit only when
/// every instruction of that feature is implemented. A leaf that is not
/// listed returns zeros.
pub fn cpuid(leaf: u32, subleaf: u32) -> [u32; 4] {
    let word = |text: &[u8; 4]| u32::from_le_bytes(*text);
    match leaf {
        // Highest basic leaf, then "Xenora guest" in EBX, EDX, ECX.
        0 => [7, word(b"Xeno"), word(b"uest"), word(b"ra g")],
        // No family, model or stepping: this is no shipped processor.
        1 => [
            0,
            0,
            CPUID_CMPXCHG16B | CPUID_POPCNT,
            CPUID_TSC | CPUID_CX8 | CPUID_CMOV,
        ],
        // Leaves 2 to 6 carry no content: leaf 2's legacy cache and TLB
        // descriptors are zeros, which probing software reads as unknown
        // geometry; leaf 3's processor serial number never existed after the
        // Pentium III; leaf 4 zeros end the deterministic-cache enumeration
        // at its first subleaf, reporting no geometry for a machine that has
        // no fixed one; leaves 5 and 6 report no monitor or thermal features.
        // Nothing is overclaimed, and a guest that needs one of those facts
        // cannot act on a positive answer that is not there.
        //
        // Leaf 7 subleaf 0: EAX is the highest supported subleaf, zero here.
        // EBX reports BMI1 and BMI2, whose full instruction lists execute in
        // both engines; AVX2 stays clear while gathers, masked moves and
        // VPMOVMSKB at 256 bits are refused, and every other bit names a
        // feature with no implementation at all. Any other subleaf is
        // undefined space and reads as zeros.
        7 => {
            if subleaf == 0 {
                [0, CPUID7_BMI1 | CPUID7_BMI2, 0, 0]
            } else {
                [0; 4]
            }
        }
        // Highest extended leaf.
        0x8000_0000 => [0x8000_0001, 0, 0, 0],
        0x8000_0001 => [0, 0, CPUID_LAHF_SAHF | CPUID_LZCNT, CPUID_LONG_MODE],
        _ => [0; 4],
    }
}

/// Termination reason for guest execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    Halt,
    Syscall(u64),
    StepLimitExceeded,
    JitCacheExhausted,
    MemoryFault(MemoryError),
    IllegalInstruction(u8),
    StackUnderflow,
    /// Division by zero or a quotient too large for its destination (#DE).
    DivideError,
    /// A non-canonical data address, misaligned 16-byte operand or reserved MXCSR bit (#GP).
    GeneralProtection,
    /// An unmasked SSE floating-point exception (#XM); MXCSR holds its flags.
    SimdException,
    /// A host service changed a register the Win64 ABI requires it to preserve.
    ServiceAbiViolation,
    /// The guest ended its own process with this exit status. Only the guest
    /// stops; the host keeps running.
    ProcessExit(u32),
    /// A service has to wait; the same call runs again later.
    GuestBlocked,
    /// A service completed and asked the scheduler for a turn.
    GuestYield,
    /// The running guest thread ended itself.
    GuestThreadExited,
    /// Guest code returned to the sentinel the host placed under a TLS
    /// callback or a thread's start routine. RIP is left on the sentinel.
    CallbackReturned,
    /// Exception dispatch or unwinding stopped: malformed or unknown unwind
    /// data, an invalid handler disposition, a target frame that is not on
    /// the stack, or a request the dispatcher does not model.
    InvalidUnwind,
}

impl fmt::Display for ExitReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Halt => write!(f, "guest executed HLT"),
            Self::Syscall(nr) => write!(f, "guest executed SYSCALL ({nr})"),
            Self::StepLimitExceeded => write!(f, "execution step budget exhausted"),
            Self::JitCacheExhausted => write!(f, "native JIT code cache exhausted"),
            Self::MemoryFault(e) => write!(f, "memory fault: {e}"),
            Self::IllegalInstruction(op) => write!(f, "illegal or unsupported opcode: 0x{op:02x}"),
            Self::StackUnderflow => write!(f, "stack underflow"),
            Self::DivideError => write!(f, "guest divide error"),
            Self::GeneralProtection => write!(f, "guest general-protection fault"),
            Self::SimdException => write!(f, "guest SIMD floating-point exception"),
            Self::ProcessExit(status) => write!(f, "guest process exited with status {status}"),
            Self::CallbackReturned => write!(f, "guest callback returned to the host"),
            Self::InvalidUnwind => write!(f, "guest exception dispatch or unwind failed"),
            Self::ServiceAbiViolation => {
                write!(f, "host service changed a nonvolatile guest register")
            }
            Self::GuestBlocked => write!(f, "guest thread is waiting"),
            Self::GuestYield => write!(f, "guest thread gave up its turn"),
            Self::GuestThreadExited => write!(f, "guest thread exited"),
        }
    }
}

impl std::error::Error for ExitReason {}

#[cfg(test)]
mod layout_tests {
    use super::{CpuState, RFlags, Register, X87State};

    /// Intel SDM Vol. 2A, "FINIT/FNINIT", Operation.
    #[test]
    fn a_new_guest_starts_with_the_x87_words_fninit_sets() {
        let init = X87State {
            control: 0x037F,
            status: 0,
            tag: 0xFFFF,
        };
        assert_eq!(X87State::INIT, init);
        assert_eq!(X87State::default(), init);
        assert_eq!(CpuState::default().x87, init);
        assert_eq!(CpuState::new(0x1000, 0x2000).x87, init);
    }

    /// The native translator addresses guest registers and RIP by raw byte
    /// offset. Keep these assertions next to the struct so a layout change
    /// cannot silently corrupt translated blocks.
    #[test]
    fn native_translator_offsets_match_repr_c_layout() {
        assert_eq!(std::mem::offset_of!(CpuState, gpr), 0);
        assert_eq!(std::mem::offset_of!(CpuState, rip), 16 * 8);
        assert_eq!(std::mem::offset_of!(CpuState, flags), 16 * 8 + 8);
        assert_eq!(std::mem::offset_of!(CpuState, gs_base), 16 * 8 + 8 + 8);
        assert_eq!(std::mem::offset_of!(CpuState, fs_base), 16 * 8 + 8 + 8 + 8);
        assert_eq!(std::mem::size_of::<RFlags>(), 6);
        assert_eq!(std::mem::offset_of!(RFlags, cf), 0);
        assert_eq!(std::mem::offset_of!(RFlags, zf), 1);
        assert_eq!(std::mem::offset_of!(RFlags, sf), 2);
        assert_eq!(std::mem::offset_of!(RFlags, of), 3);
        assert_eq!(std::mem::offset_of!(RFlags, pf), 4);
        assert_eq!(std::mem::offset_of!(RFlags, af), 5);
        assert_eq!(std::mem::offset_of!(CpuState, xmm), 160);
        assert_eq!(std::mem::offset_of!(CpuState, mxcsr), 160 + 16 * 16);

        let cpu = CpuState::default();
        let base = std::ptr::addr_of!(cpu) as usize;
        for reg in [Register::Rax, Register::Rsp, Register::R15] {
            assert_eq!(
                base + (reg as usize) * 8,
                std::ptr::addr_of!(cpu.gpr[reg as usize]) as usize
            );
        }
        assert_eq!(base + 16 * 8, std::ptr::addr_of!(cpu.rip) as usize);
        assert_eq!(base + 16 * 8 + 8, std::ptr::addr_of!(cpu.flags) as usize);
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::{CpuState, ExitReason, Interpreter, Register};
    use crate::memory::{GuestMemory, MemoryPermissions};

    const CODE_BASE: u64 = 0x1000;
    const CODE_SIZE: usize = 0x1000;
    const STACK_BASE: u64 = 0x8000_0000;
    const STACK_SIZE: usize = 0x10000;
    const FS_BASE: u64 = 0x9000;
    const GS_BASE: u64 = 0xA000;

    /// A guest memory image with executable code, a stack and distinct FS and
    /// GS bases.
    fn memory_with_code(code: &[u8]) -> GuestMemory {
        let mut memory = GuestMemory::new();
        memory
            .map_region_with_data(
                CODE_BASE,
                CODE_SIZE,
                MemoryPermissions::READ_WRITE_EXECUTE,
                code,
            )
            .expect("code region");
        memory
            .map_region(STACK_BASE, STACK_SIZE, MemoryPermissions::READ_WRITE)
            .expect("stack region");
        // The segment bases are distinct so a swapped prefix is visible.
        memory
            .map_region(FS_BASE, 0x1000, MemoryPermissions::READ_WRITE)
            .expect("fs region");
        memory
            .map_region(GS_BASE, 0x1000, MemoryPermissions::READ_WRITE)
            .expect("gs region");
        memory
    }

    /// Run `code` on the interpreter and return the final state and exit reason.
    fn run(code: &[u8]) -> (CpuState, ExitReason) {
        let mut memory = memory_with_code(code);
        let cpu = CpuState::new(CODE_BASE, STACK_BASE + 0x8000);
        let mut interp = Interpreter::new(cpu, &mut memory, 64);
        let result = interp.run();
        (result.final_state, result.exit_reason)
    }

    /// `mov r8, imm64`: 49 B8 <imm64>.
    fn mov_r8_imm64(value: u64) -> Vec<u8> {
        let mut code = vec![0x49, 0xB8];
        code.extend_from_slice(&value.to_le_bytes());
        code
    }

    /// `mov rbx, imm64`: 48 BB <imm64>.
    fn mov_rbx_imm64(value: u64) -> Vec<u8> {
        let mut code = vec![0x48, 0xBB];
        code.extend_from_slice(&value.to_le_bytes());
        code
    }

    /// `mov rax, imm64`: 48 B8 <imm64>.
    fn mov_rax_imm64(value: u64) -> Vec<u8> {
        let mut code = vec![0x48, 0xB8];
        code.extend_from_slice(&value.to_le_bytes());
        code
    }

    /// REX.B extends the ModRM `rm` field, so `rm=0` selects r8.
    ///
    /// Encoding 49 89 D8 is `mov r8, rbx`.
    #[test]
    fn rex_b_extends_the_rm_field_to_r8() {
        let mut code = mov_rbx_imm64(0x1122_3344_5566_7788);
        code.extend_from_slice(&[0x49, 0x89, 0xD8]); // mov r8, rbx
        code.push(0xF4); // hlt

        let (state, reason) = run(&code);
        assert_eq!(reason, ExitReason::Halt);
        assert_eq!(state.get(Register::Rbx), 0x1122_3344_5566_7788);
        assert_eq!(state.get(Register::R8), 0x1122_3344_5566_7788);
        // No prefix in this encoding selects r9.
        assert_eq!(state.get(Register::R9), 0);
    }

    /// REX.R extends the ModRM `reg` field, so `reg=0` selects r8.
    ///
    /// Encoding 4C 89 C3 is `mov rbx, r8`.
    #[test]
    fn rex_r_extends_the_reg_field_to_r8() {
        let mut code = mov_r8_imm64(0x1122_3344_5566_7788);
        code.extend_from_slice(&[0x4C, 0x89, 0xC3]); // mov rbx, r8
        code.push(0xF4); // hlt

        let (state, reason) = run(&code);
        assert_eq!(reason, ExitReason::Halt);
        assert_eq!(state.get(Register::R8), 0x1122_3344_5566_7788);
        assert_eq!(state.get(Register::Rbx), 0x1122_3344_5566_7788);
        // No prefix in this encoding selects r9.
        assert_eq!(state.get(Register::R9), 0);
    }

    /// REX.R and REX.B extend independent fields, so the destination r9 and
    /// the source r12 are chosen by different prefix bits.
    ///
    /// Encoding 4D 89 E1 is `mov r9, r12`.
    #[test]
    fn rex_r_and_rex_b_extend_independent_fields() {
        let mut code = mov_r8_imm64(0x1122_3344_5566_7788);
        code.extend_from_slice(&[0x4D, 0x89, 0xE1]); // mov r9, r12
        code.push(0xF4); // hlt

        let (state, reason) = run(&code);
        assert_eq!(reason, ExitReason::Halt);
        // r9 receives r12, which is zero here, and not the value in r8.
        assert_eq!(state.get(Register::R9), 0);
        assert_eq!(state.get(Register::R12), 0);
        assert_eq!(state.get(Register::R8), 0x1122_3344_5566_7788);
    }

    /// Without REX.W the move is 32-bit wide, so the destination keeps only the
    /// low half of the source and its upper half is cleared.
    ///
    /// Encoding 89 D8 is `mov eax, ebx`.
    #[test]
    fn a_move_without_rex_w_is_32_bit_wide() {
        let mut code = mov_rbx_imm64(0x1122_3344_5566_7788);
        code.extend_from_slice(&[0x89, 0xD8]); // mov eax, ebx
        code.push(0xF4); // hlt

        let (state, reason) = run(&code);
        assert_eq!(reason, ExitReason::Halt);
        // Only the low 32 bits of ebx survive, and rax's upper half is cleared.
        assert_eq!(state.get(Register::Rax), 0x5566_7788);
        assert_eq!(state.get(Register::Rbx), 0x1122_3344_5566_7788);
    }

    /// The FS segment override adds the FS base to the effective address.
    ///
    /// Encoding 64 48 89 04 25 08 00 00 00 is `mov fs:[0x8], rax`.
    #[test]
    fn the_fs_prefix_adds_the_fs_base() {
        let mut code = mov_rax_imm64(0x1122_3344_5566_7788);
        code.extend_from_slice(&[0x64, 0x48, 0x89, 0x04, 0x25, 0x08, 0x00, 0x00, 0x00]);
        code.push(0xF4); // hlt

        let mut memory = memory_with_code(&code);
        let mut cpu = CpuState::new(CODE_BASE, STACK_BASE + 0x8000);
        cpu.fs_base = FS_BASE;
        cpu.gs_base = GS_BASE;
        let mut interp = Interpreter::new(cpu, &mut memory, 64);
        let result = interp.run();
        assert_eq!(result.exit_reason, ExitReason::Halt);
        assert_eq!(
            memory.read_u64(FS_BASE + 0x8).expect("fs read"),
            0x1122_3344_5566_7788
        );
        // The GS base is untouched: the GS-prefixed store did not happen.
        assert_eq!(memory.read_u64(GS_BASE + 0x8).expect("gs read"), 0);
    }
}

/// Result returned after running guest execution.
#[derive(Debug)]
pub struct ExecutionResult {
    pub instructions_executed: usize,
    pub final_state: CpuState,
    pub exit_reason: ExitReason,
}

/// Bounded software interpreter for x86-64 instructions.
pub struct Interpreter<'a> {
    pub cpu: CpuState,
    pub memory: &'a mut GuestMemory,
    pub max_instructions: usize,
    pub instructions_executed: usize,
}

impl<'a> Interpreter<'a> {
    pub fn new(cpu: CpuState, memory: &'a mut GuestMemory, max_instructions: usize) -> Self {
        Self {
            cpu,
            memory,
            max_instructions,
            instructions_executed: 0,
        }
    }

    /// Run interpreter until halt, error, or instruction budget limit.
    pub fn run(&mut self) -> ExecutionResult {
        while self.instructions_executed < self.max_instructions {
            match self.step() {
                Ok(None) => {
                    self.instructions_executed += 1;
                }
                Ok(Some(reason)) => {
                    self.instructions_executed += 1;
                    return ExecutionResult {
                        instructions_executed: self.instructions_executed,
                        final_state: self.cpu.clone(),
                        exit_reason: reason,
                    };
                }
                Err(reason) => {
                    return ExecutionResult {
                        instructions_executed: self.instructions_executed,
                        final_state: self.cpu.clone(),
                        exit_reason: reason,
                    };
                }
            }
        }

        ExecutionResult {
            instructions_executed: self.instructions_executed,
            final_state: self.cpu.clone(),
            exit_reason: ExitReason::StepLimitExceeded,
        }
    }

    /// Single-step execution. Returns Ok(Some(ExitReason)) on clean exit
    /// (HLT/SYSCALL). An error leaves the guest state unchanged.
    pub fn step(&mut self) -> Result<Option<ExitReason>, ExitReason> {
        let memory = &*self.memory;
        let insn = decode(self.cpu.rip, |address| {
            memory.fetch_instruction_byte(address)
        })
        .map_err(ExitReason::from)?;
        x86_exec::execute(&mut self.cpu, self.memory, &insn)
    }
}

impl From<DecodeError<MemoryError>> for ExitReason {
    fn from(error: DecodeError<MemoryError>) -> Self {
        match error {
            DecodeError::Fetch(fault) => Self::MemoryFault(fault),
            DecodeError::Unsupported(byte) => Self::IllegalInstruction(byte),
            // Intel SDM Vol. 3A, 7.15: exceeding the 15-byte limit is #GP(0).
            DecodeError::TooLong => Self::GeneralProtection,
        }
    }
}
