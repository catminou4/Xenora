// SPDX-License-Identifier: GPL-3.0-or-later

//! Guest thread scheduler tests.
//!
//! The workloads are original C programs compiled for x86-64 Windows (see
//! `threads/compile.py`), bound through a real import table and executed as
//! translated guest code. Expected values are computed here or stated by the
//! public Win32 documentation, not taken from the scheduler.

use std::collections::BTreeMap;
use std::sync::Arc;
use xenora_loader::cpu::{CpuState, ExitReason, Register};
use xenora_loader::dbt::{DbtEngine, DbtMetrics};
use xenora_loader::imports::ImportRegistry;
use xenora_loader::memory::{GuestMemory, MemoryPermissions};
use xenora_loader::pe::ParsedPe;
use xenora_loader::scheduler::{DEFAULT_QUANTUM, Outcome, ScheduleReport, Scheduler};
use xenora_loader::services::{Clock, MockClock, ServicesState, register_standard_services};

const BASE: u64 = 0x1_4000_0000;
const TEXT_RVA: u32 = 0x1000;
const TEXT_RAW: usize = 0x200;
const TEXT_LEN: usize = 0x6000;
const IDATA_RVA: u32 = 0x8000;
const IDATA_RAW: usize = TEXT_RAW + TEXT_LEN;
const IDATA_LEN: usize = 0x2000;
const STACK: u64 = 0x8_0000;
const STACK_LEN: usize = 0x1_0000;
const TEB: u64 = 0x7_0000;
const SHARED: u64 = 0x5_0000;
const COUNTER: u64 = SHARED + 256;
/// First guest thread slot: guard page, stack, guard page, TEB.
const FIRST_SLOT: u64 = 0x6_0000_0000;
const BUDGET: usize = 20_000_000;
const MAIN_ID: u32 = 0x4104;
/// Host services of the test programs: advance the clock by RCX milliseconds,
/// and try to run a second scheduler.
const ADVANCE: u64 = 0x6_0000;
const NESTED: u64 = 0x6_0010;

fn return_to_guest(cpu: &mut CpuState, memory: &mut GuestMemory) -> Result<(), ExitReason> {
    let rsp = cpu.get(Register::Rsp);
    cpu.rip = memory.read_u64(rsp).map_err(ExitReason::MemoryFault)?;
    cpu.set(Register::Rsp, rsp + 8);
    Ok(())
}

struct Program {
    code: Vec<u8>,
    entries: BTreeMap<&'static str, usize>,
    imports: Vec<(&'static str, usize)>,
}

fn program(main: &str) -> Program {
    let mut program = Program {
        code: Vec::new(),
        entries: BTreeMap::new(),
        imports: Vec::new(),
    };
    let listing = if main.starts_with("main_fls_") {
        include_str!("threads/fls.txt")
    } else if main.starts_with("main_tq_") {
        include_str!("threads/taskqueue.txt")
    } else {
        include_str!("threads/program.txt")
    };
    for line in listing.lines() {
        let parts: Vec<&str> = line.split(' ').collect();
        match parts[0] {
            "entry" => {
                program.entries.insert(parts[1], parts[2].parse().unwrap());
            }
            "import" => program.imports.push((parts[1], parts[2].parse().unwrap())),
            _ => program.code.extend(
                (0..parts[1].len())
                    .step_by(2)
                    .map(|at| u8::from_str_radix(&parts[1][at..at + 2], 16).unwrap()),
            ),
        }
    }
    program
}

/// Build a PE image whose entry point is `call main; hlt`.
fn image(main: &str) -> Vec<u8> {
    let program = program(main);
    let mut pe = vec![0u8; IDATA_RAW + IDATA_LEN];
    fn put(pe: &mut [u8], at: usize, bytes: &[u8]) {
        pe[at..at + bytes.len()].copy_from_slice(bytes);
    }
    let start = program.code.len().next_multiple_of(16);
    assert!(start + 16 <= TEXT_LEN);
    put(&mut pe, 0, b"MZ");
    put(&mut pe, 0x3C, &0x80u32.to_le_bytes());
    put(&mut pe, 0x80, b"PE\0\0");
    put(&mut pe, 0x84, &0x8664u16.to_le_bytes());
    put(&mut pe, 0x86, &2u16.to_le_bytes());
    put(&mut pe, 0x94, &240u16.to_le_bytes());
    let optional = 0x98;
    put(&mut pe, optional, &0x20Bu16.to_le_bytes());
    put(
        &mut pe,
        optional + 16,
        &(TEXT_RVA + start as u32).to_le_bytes(),
    );
    put(&mut pe, optional + 24, &BASE.to_le_bytes());
    put(&mut pe, optional + 32, &0x1000u32.to_le_bytes());
    put(&mut pe, optional + 36, &0x200u32.to_le_bytes());
    put(&mut pe, optional + 56, &0x10000u32.to_le_bytes());
    put(&mut pe, optional + 60, &0x200u32.to_le_bytes());
    put(&mut pe, optional + 108, &16u32.to_le_bytes());
    put(&mut pe, optional + 120, &IDATA_RVA.to_le_bytes());
    for (index, (name, rva, raw, len, flags)) in [
        (b".text", TEXT_RVA, TEXT_RAW, TEXT_LEN, 0x6000_0020u32),
        (b".idat", IDATA_RVA, IDATA_RAW, IDATA_LEN, 0xC000_0040),
    ]
    .into_iter()
    .enumerate()
    {
        let section = optional + 240 + 40 * index;
        put(&mut pe, section, name);
        put(&mut pe, section + 8, &(len as u32).to_le_bytes());
        put(&mut pe, section + 12, &rva.to_le_bytes());
        put(&mut pe, section + 16, &(len as u32).to_le_bytes());
        put(&mut pe, section + 20, &(raw as u32).to_le_bytes());
        put(&mut pe, section + 36, &flags.to_le_bytes());
    }
    put(&mut pe, TEXT_RAW, &program.code);
    let displacement = program.entries[main] as i32 - (start as i32 + 5);
    put(&mut pe, TEXT_RAW + start, &[0xE8]);
    put(&mut pe, TEXT_RAW + start + 1, &displacement.to_le_bytes());
    put(&mut pe, TEXT_RAW + start + 5, &[0xF4]);

    let file = |rva: u32| IDATA_RAW + (rva - IDATA_RVA) as usize;
    let mut names: Vec<&str> = program.imports.iter().map(|import| import.0).collect();
    names.sort_unstable();
    names.dedup();
    let implicit_fls_exit = main == "main_fls_return";
    if implicit_fls_exit {
        names.retain(|name| *name != "ExitThread");
    }
    let descriptor_size = 20 * (names.len() as u32 + 1);
    put(&mut pe, optional + 124, &descriptor_size.to_le_bytes());
    let mut next = IDATA_RVA + descriptor_size;
    let mut take = |len: usize| {
        let rva = next;
        next = (next + len as u32 + 7) & !7;
        rva
    };
    let mut slots = BTreeMap::new();
    if implicit_fls_exit {
        // Relocations in the unused explicit path still point to an IAT
        // slot, but this image never imports or binds ExitThread.
        slots.insert("ExitThread", take(8));
    }
    for (index, name) in names.iter().enumerate() {
        let module: &[u8] = if *name == "_errno" {
            b"api-ms-win-crt-runtime-l1-1-0.dll"
        } else if name.starts_with("XTaskQueue") {
            b"xgameruntime.dll"
        } else {
            b"KERNEL32.dll"
        };
        let module_rva = take(module.len() + 1);
        put(&mut pe, file(module_rva), module);
        let hint = take(name.len() + 3);
        put(&mut pe, file(hint) + 2, name.as_bytes());
        let (lookup, slot) = (take(16), take(16));
        put(&mut pe, file(lookup), &u64::from(hint).to_le_bytes());
        put(&mut pe, file(slot), &u64::from(hint).to_le_bytes());
        let descriptor = file(IDATA_RVA) + 20 * index;
        put(&mut pe, descriptor, &lookup.to_le_bytes());
        put(&mut pe, descriptor + 12, &module_rva.to_le_bytes());
        put(&mut pe, descriptor + 16, &slot.to_le_bytes());
        slots.insert(*name, slot);
    }
    for (name, offset) in &program.imports {
        let displacement = slots[name].wrapping_sub(TEXT_RVA + *offset as u32 + 4);
        put(&mut pe, TEXT_RAW + offset, &displacement.to_le_bytes());
    }
    pe
}

struct Run {
    report: ScheduleReport,
    memory: GuestMemory,
    clock: Arc<MockClock>,
    metrics: DbtMetrics,
    state: Arc<ServicesState>,
}

impl Run {
    fn out(&self, index: u64) -> u64 {
        self.memory.read_u64(SHARED + 8 * index).unwrap()
    }

    fn outs(&self, count: u64) -> Vec<u64> {
        (0..count).map(|index| self.out(index)).collect()
    }

    fn halted(self) -> Self {
        assert_eq!(self.report.outcome, Outcome::Halted);
        self
    }
}

struct Guest {
    memory: GuestMemory,
    engine: DbtEngine,
    cpu: CpuState,
    state: Arc<ServicesState>,
    clock: Arc<MockClock>,
}

fn guest(main: &str) -> Guest {
    let bytes = image(main);
    let pe = ParsedPe::parse(&bytes).unwrap();
    let mut loaded = pe.load_into_memory(&bytes).unwrap();
    let memory = &mut loaded.memory;
    for (base, len) in [(SHARED, 0x1000), (STACK, STACK_LEN), (TEB, 0x2000)] {
        memory
            .map_region(base, len, MemoryPermissions::READ_WRITE)
            .unwrap();
    }
    let clock = Arc::new(MockClock::new(1_000_000));
    let state = Arc::new(ServicesState::new(clock.clone()));
    let mut registry = ImportRegistry::new();
    register_standard_services(&mut registry, Arc::clone(&state));
    registry.bind_imports(&pe, &bytes, BASE, memory).unwrap();
    let mut engine = DbtEngine::new(1024 * 1024).unwrap();
    engine.set_import_registry(registry);
    let time = clock.clone();
    engine.register_service(ADVANCE, move |cpu, memory| {
        time.advance(cpu.get(Register::Rcx) * 10_000);
        return_to_guest(cpu, memory)
    });
    let mut cpu = CpuState::new(loaded.entry_point, STACK + STACK_LEN as u64 - 0x100);
    cpu.gs_base = TEB;
    Guest {
        memory: loaded.memory,
        engine,
        cpu,
        state,
        clock,
    }
}

fn run_with(main: &str, quantum: usize, budget: usize) -> Run {
    let mut guest = guest(main);
    let report = Scheduler::new(Arc::clone(&guest.state), quantum)
        .run(&mut guest.engine, &mut guest.memory, &guest.cpu, budget)
        .unwrap();
    Run {
        report,
        memory: guest.memory,
        clock: guest.clock,
        metrics: guest.engine.metrics,
        state: guest.state,
    }
}

fn run(main: &str) -> Run {
    run_with(main, DEFAULT_QUANTUM, BUDGET)
}

const QUANTA: [usize; 6] = [1, 2, 7, 64, 1024, 1_000_000];
const FAILED: u64 = u32::MAX as u64;

fn fibonacci(n: u32) -> u64 {
    (0..n)
        .fold((0u64, 1u64), |(a, b), _| (b, a.wrapping_add(b)))
        .0
}

fn collatz_steps(mut value: u64) -> u64 {
    let mut steps = 0;
    while value != 1 {
        value = if value % 2 == 1 {
            3 * value + 1
        } else {
            value / 2
        };
        steps += 1;
    }
    steps
}

#[test]
fn threads_run_independent_functions_and_report_exit_codes() {
    for quantum in QUANTA {
        let run = run_with("main_independent", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(7),
            [fibonacci(80), collatz_steps(97), 0, 11, 22, 1, 1],
            "quantum {quantum}"
        );
        assert_eq!(run.report.peak_threads, 3);
    }
}

#[test]
fn atomic_and_locked_counters_lose_no_update() {
    for quantum in QUANTA {
        for main in ["main_atomic", "main_mutex"] {
            let run = run_with(main, quantum, BUDGET).halted();
            assert_eq!(run.outs(2), [0, 800], "{main} quantum {quantum}");
            assert_eq!(run.report.peak_threads, 5);
        }
    }
}

#[test]
fn unsynchronized_counters_do_lose_updates() {
    // Shows that the threads really interleave: the locked version above has
    // the same read, yield, write sequence.
    let run = run("main_racy").halted();
    assert!(run.out(1) < 800, "{}", run.out(1));
}

#[test]
fn a_blocked_thread_resumes_when_it_is_signaled() {
    let expected = (1..=50u64).fold(0u64, |sum, value| sum.wrapping_mul(31).wrapping_add(value));
    for quantum in QUANTA {
        let run = run_with("main_producer", quantum, BUDGET).halted();
        assert_eq!(run.outs(2), [expected, 1], "quantum {quantum}");
    }
}

#[test]
fn one_signal_releases_the_oldest_waiter_only() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_fifo", quantum, BUDGET).halted();
        // Nobody ran before the signal; the setter cannot take it back.
        assert_eq!(
            [run.out(8), run.out(9), run.out(10)],
            [0, 258, 1],
            "quantum {quantum}"
        );
        assert_eq!(run.outs(4), [10, 11, 12, 13], "quantum {quantum}");
    }
}

#[test]
fn timeouts_follow_the_clock_exactly() {
    let run = run("main_timeout").halted();
    assert_eq!(run.outs(3), [258, 258, 0]);
    // 250 ms in 100 ns units, and no further sleep.
    assert_eq!(run.clock.performance_count(), 2_500_000);
    let run = run_with("main_signal_before_timeout", 1024, BUDGET).halted();
    assert_eq!(run.outs(2), [0, 258]);
    assert_eq!(run.clock.performance_count(), 400_000 + 1_000_000);
}

#[test]
fn thread_local_state_is_isolated() {
    for quantum in QUANTA {
        let run = run_with("main_isolated", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(7),
            [1, 1, 1, 1, 1234, 0xAAAA, 1],
            "quantum {quantum}"
        );
    }
}

#[test]
fn exit_and_termination_follow_the_documented_results() {
    let run = run("main_lifecycle").halted();
    assert_eq!(
        run.outs(16),
        [
            0, 0xC0DE, // ExitThread
            259, 258, // STILL_ACTIVE, WAIT_TIMEOUT
            1, 0, 77, // TerminateThread, signaled, its code
            0, 5, 77, // already terminated: ERROR_ACCESS_DENIED, code kept
            1, FAILED, 6, // closed: WAIT_FAILED, ERROR_INVALID_HANDLE
            0, 0, 0, // OpenThread fails; nothing ran after ExitThread
        ]
    );
}

#[test]
fn suspended_threads_do_not_run() {
    let run = run("main_suspend");
    // The program ends by suspending itself, with nobody left to resume it.
    assert_eq!(
        run.report.outcome,
        Outcome::Deadlock {
            blocked: vec![],
            suspended: vec![MAIN_ID]
        }
    );
    assert_eq!(
        run.outs(15),
        [
            0, 1, 2, // created suspended, then suspended again, then resumed once
            258, 1, 0, 1, // still suspended; resumed; ended; ran
            FAILED, 5, // an ended thread cannot be suspended
            1, 0, // resumed and joined through an opened handle
            0, 5, 0, // which may not query or terminate
            0, // main never resumed itself
        ]
    );
}

#[test]
fn hundreds_of_threads_leave_nothing_behind() {
    for quantum in [7, 1024] {
        let run = run_with("main_churn", quantum, BUDGET).halted();
        assert_eq!(run.outs(2), [600, 600], "quantum {quantum}");
        assert_eq!(run.report.peak_threads, 2, "one slot is reused");
        for offset in [0, 0x1000, 0x2_0000, 0xF0_0000] {
            assert!(
                run.memory.read_u8(FIRST_SLOT + offset).is_err(),
                "{offset:#x} is still mapped"
            );
            assert!(
                run.memory
                    .read_u8(FIRST_SLOT + 0x100_0000 + offset)
                    .is_err()
            );
        }
        assert_eq!(run.state.last_error(FIRST_SLOT + 0xF0_0000), 0);
    }
}

#[test]
fn thread_creation_is_bounded_and_validated() {
    let run = run("main_limit");
    // The main thread counts towards the limit of 64. The waiters never end.
    assert_eq!(run.outs(5), [63, 8, 0, 0, 87]);
    assert_eq!(run.report.outcome, Outcome::Halted);
    assert!(
        run.memory.read_u8(FIRST_SLOT + 0x1000).is_err(),
        "reclaimed at exit"
    );
}

#[test]
fn a_deadlock_is_reported_with_its_threads() {
    let run = run("main_deadlock");
    assert_eq!(
        run.report.outcome,
        Outcome::Deadlock {
            blocked: vec![MAIN_ID, MAIN_ID + 4],
            suspended: vec![]
        }
    );
}

#[test]
fn a_mutex_owned_by_an_ended_thread_is_reported_abandoned_once() {
    let run = run("main_abandoned").halted();
    // WAIT_ABANDONED_0 + 1, released, then an ordinary acquisition.
    assert_eq!(run.outs(3), [0x81, 1, 0]);
}

#[test]
fn stack_overflow_faults_in_the_guard_page_of_that_thread() {
    let run = run("main_overflow");
    let Outcome::Fault {
        thread,
        reason,
        rip,
    } = run.report.outcome
    else {
        panic!("{:?}", run.report.outcome);
    };
    assert_eq!(thread, MAIN_ID + 4);
    assert!((BASE..BASE + 0x8000).contains(&rip));
    let ExitReason::MemoryFault(error) = reason else {
        panic!("{reason:?}")
    };
    let text = format!("{error:?}");
    assert!(text.contains("PermissionViolation"), "{text}");
    // A 256 KiB stack with frames of at least 512 bytes.
    assert!((100..512).contains(&run.out(0)), "{}", run.out(0));
}

#[test]
fn a_wait_keeps_the_objects_it_was_given() {
    for quantum in QUANTA {
        let run = run_with("main_captured", quantum, BUDGET).halted();
        // Object 1, the semaphore, whose only permit the wait consumed.
        assert_eq!(run.outs(2), [1, 258], "quantum {quantum}");
    }
}

#[test]
fn the_process_ends_with_its_main_thread_but_not_with_exit_thread() {
    let run = run("main_outlived").halted();
    assert!(
        run.memory.read_u8(FIRST_SLOT + 0x1000).is_err(),
        "the spinning thread is gone"
    );
    let run = run_with("main_exits_first", 1024, BUDGET);
    assert_eq!(run.report.outcome, Outcome::AllThreadsExited);
    assert_eq!(run.outs(2), [258, 1]);
}

#[test]
fn scheduling_is_deterministic_and_bounded() {
    for main in [
        "main_racy",
        "main_mutex",
        "main_isolated",
        "main_churn",
        "main_section",
        "main_srw",
        "main_queue",
        "main_sleep",
    ] {
        for quantum in [1, 64, 1024] {
            let (first, second) = (
                run_with(main, quantum, BUDGET),
                run_with(main, quantum, BUDGET),
            );
            assert_eq!(first.report, second.report, "{main} quantum {quantum}");
            assert_eq!(first.outs(32), second.outs(32), "{main} quantum {quantum}");
        }
    }
    for budget in [0, 1, 100, 5000] {
        let run = run_with("main_atomic", 64, budget);
        assert_eq!(run.report.outcome, Outcome::BudgetExhausted);
        assert!(
            run.report.instructions <= budget + 64,
            "{}",
            run.report.instructions
        );
    }
}

#[test]
fn small_quanta_do_not_fill_the_code_cache() {
    // Compiled code must not depend on where turns happened to end.
    let blocks = |quantum| {
        run_with("main_independent", quantum, BUDGET)
            .metrics
            .blocks_compiled
    };
    let whole = blocks(1_000_000);
    for quantum in [1, 2, 3, 5, 7, 11, 13, 64] {
        assert_eq!(blocks(quantum), whole, "quantum {quantum}");
    }
}

#[test]
fn without_a_scheduler_nothing_pretends_to_create_threads() {
    let mut guest = guest("main_independent");
    let result = guest.engine.run(&mut guest.cpu, &mut guest.memory, BUDGET);
    assert_eq!(result.exit_reason, ExitReason::Halt);
    assert_eq!(
        guest.memory.read_u64(SHARED + 392).unwrap(),
        0,
        "CreateThread returned null"
    );
    assert_eq!(
        guest.memory.read_u64(SHARED + 16).unwrap(),
        FAILED,
        "the wait failed"
    );
    assert_eq!(guest.state.last_error(TEB), 6);

    let mut waiting = self::guest("main_timeout");
    waiting
        .engine
        .run(&mut waiting.cpu, &mut waiting.memory, BUDGET);
    assert_eq!(waiting.memory.read_u64(SHARED).unwrap(), FAILED);
    assert_eq!(waiting.clock.performance_count(), 0);
}

#[test]
fn a_finished_run_detaches_its_scheduler() {
    let mut guest = guest("main_timeout");
    for _ in 0..2 {
        let scheduler = Scheduler::new(Arc::clone(&guest.state), 1024);
        let report = scheduler
            .run(&mut guest.engine, &mut guest.memory, &guest.cpu, BUDGET)
            .unwrap();
        assert_eq!(report.outcome, Outcome::Halted);
        assert_ne!(report.main.get(Register::Rsp), 0);
    }
    assert_eq!(guest.memory.read_u64(COUNTER).unwrap(), 0);
}

#[test]
fn a_second_scheduler_cannot_attach_while_one_runs() {
    let mut guest = guest("main_nested");
    let state = Arc::clone(&guest.state);
    guest.engine.register_service(NESTED, move |cpu, memory| {
        let mut engine = DbtEngine::new(1024 * 1024).unwrap();
        let nested = Scheduler::new(Arc::clone(&state), 1024).run(&mut engine, memory, cpu, BUDGET);
        assert_eq!(nested.err(), Some(ExitReason::ServiceAbiViolation));
        memory
            .write_u64(SHARED, 1)
            .map_err(ExitReason::MemoryFault)?;
        return_to_guest(cpu, memory)
    });
    let report = Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &guest.cpu, BUDGET)
        .unwrap();
    // The refused attempt leaves the running scheduler attached.
    assert_eq!(report.outcome, Outcome::Halted);
    assert_eq!(guest.memory.read_u64(SHARED).unwrap(), 1);
    assert_eq!(guest.memory.read_u64(SHARED + 8).unwrap(), 0);
}

#[test]
fn a_signal_after_the_deadline_is_not_taken_by_the_expired_wait() {
    for quantum in QUANTA {
        let run = run_with("main_expired_wait", quantum, BUDGET).halted();
        // WAIT_TIMEOUT, the signaling thread ends, and the auto-reset signal is
        // still there once.
        assert_eq!(run.outs(4), [258, 0, 0, 258], "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 1_000_000);
    }
}

/// Run `main` until `budget` is spent.
fn first_part_of(main: &str, budget: usize) -> (Guest, ScheduleReport) {
    let mut guest = guest(main);
    let report = Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &guest.cpu, budget)
        .unwrap();
    (guest, report)
}

fn first_part(budget: usize) -> (Guest, ScheduleReport) {
    first_part_of("main_continued_wait", budget)
}

/// Smallest budget with which the wait has been called.
fn budget_that_blocks() -> usize {
    (1..200)
        .find(|&budget| first_part(budget).0.engine.metrics.service_calls >= 2)
        .unwrap()
}

#[test]
fn a_continued_run_keeps_the_pending_wait() {
    let blocks = budget_that_blocks();
    for budget in [blocks, blocks + 1] {
        let (mut guest, report) = first_part(budget);
        assert_eq!(report.outcome, Outcome::BudgetExhausted);
        guest.clock.advance(6_000_000);
        // Neither the timeout nor the objects are taken again.
        guest.memory.write_u64(SHARED + 8 * 49, 0xBAD).unwrap();
        let report = Scheduler::new(Arc::clone(&guest.state), 1024)
            .run(&mut guest.engine, &mut guest.memory, &report.main, BUDGET)
            .unwrap();
        assert_eq!(report.outcome, Outcome::Halted);
        assert_eq!(guest.memory.read_u64(SHARED).unwrap(), 258);
        assert_eq!(guest.clock.performance_count(), 10_000_000);
    }
}

#[test]
fn a_run_from_another_context_does_not_inherit_a_pending_wait() {
    let (mut guest, _) = first_part(budget_that_blocks());
    let report = Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &guest.cpu, BUDGET)
        .unwrap();
    assert_eq!(report.outcome, Outcome::Halted);
    guest.clock.advance(6_000_000);
    let begun = guest.clock.performance_count();
    Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &guest.cpu, BUDGET)
        .unwrap();
    assert_eq!(guest.clock.performance_count(), begun + 10_000_000);

    let (mut guest, _) = first_part(budget_that_blocks());
    guest.clock.advance(6_000_000);
    Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &guest.cpu, BUDGET)
        .unwrap();
    assert_eq!(guest.clock.performance_count(), 16_000_000);
}

#[test]
fn an_exhausted_run_neither_sleeps_nor_reports_a_deadlock() {
    let blocks = budget_that_blocks();
    for budget in 1..=blocks + 1 {
        let (guest, report) = first_part(budget);
        assert_eq!(report.outcome, Outcome::BudgetExhausted, "{budget}");
        assert_eq!(guest.clock.performance_count(), 0, "{budget}");
    }
    // One retry more: there is budget left for the result of the sleep.
    assert_eq!(
        first_part(blocks + 2).0.clock.performance_count(),
        10_000_000
    );
}

#[test]
fn a_thread_that_ends_with_its_wait_is_noticed_at_once() {
    for quantum in QUANTA {
        // Neither a deadlock nor the longer timeout of the thread waiting for it.
        let run = run_with("main_ends_in_a_wait", quantum, BUDGET).halted();
        assert_eq!(run.outs(3), [0, 0, 258], "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 1_000_000);
    }
}

#[test]
fn a_thread_that_ended_cannot_be_opened_by_its_id() {
    for quantum in QUANTA {
        let run = run_with("main_reopened", quantum, BUDGET);
        assert_eq!(run.report.outcome, Outcome::AllThreadsExited);
        // ERROR_INVALID_PARAMETER.
        assert_eq!(run.outs(4), [1, 1, 0, 87], "quantum {quantum}");
    }
}

#[test]
fn a_signaled_object_goes_to_the_waiting_thread() {
    for quantum in [256, 1024, 1_000_000] {
        let run = run_with("main_handoff", quantum, BUDGET).halted();
        // Not available again, and the waiter has not run when the call
        // returns: for a mutex, a semaphore and an event.
        assert_eq!(
            run.outs(11),
            [258, 0, 1, 258, 1, 2, 258, 2, 3, 0, 1],
            "quantum {quantum}"
        );
    }
}

#[test]
fn a_suspended_waiter_takes_nothing() {
    for quantum in QUANTA {
        let run = run_with("main_suspended_waiter", quantum, BUDGET).halted();
        assert_eq!(run.outs(4), [0, 0, 0, 1], "quantum {quantum}");
    }
}

#[test]
fn opened_handles_carry_only_their_rights() {
    let run = run("main_rights").halted();
    assert_eq!(
        run.outs(13),
        [
            FAILED, 5, 0, 5, 1, 259, // query only
            0, 1, 0, FAILED, 5, 0, // suspend only
            1,
        ]
    );
}

#[test]
fn a_privileged_halt_in_a_worker_is_a_fault_not_a_process_exit() {
    let run = run("main_worker_halts");
    assert!(
        matches!(
            run.report.outcome,
            Outcome::Fault { thread, reason: ExitReason::Halt, .. } if thread == MAIN_ID + 4
        ),
        "{:?}",
        run.report.outcome
    );
}

#[test]
fn thread_stacks_are_reused() {
    let run = run("main_reuse").halted();
    assert_eq!(run.memory.read_u64(COUNTER).unwrap(), 50);
}

#[test]
fn repeated_signals_do_not_starve_ready_threads() {
    for quantum in [1, 7, 64, 1024, 10_000] {
        let run = run_with("main_starve", quantum, 1_000_000).halted();
        assert_eq!(run.out(0), 1, "quantum {quantum}");
        // The main thread gave up its turn after about one quantum.
        assert!(run.out(1) <= quantum as u64, "{} at {quantum}", run.out(1));
    }
}

#[test]
fn locks_of_an_ended_thread_are_not_inherited() {
    for quantum in QUANTA {
        let run = run_with("main_orphaned_locks", quantum, BUDGET);
        // The second worker runs in the first one's slot. It owns nothing, and
        // completing an initialization it did not begin fails.
        assert_eq!(run.outs(5), [1, 1, 0, 0, 0], "quantum {quantum}");
        assert_eq!(run.out(7), 0);
        match run.report.outcome {
            Outcome::Halted => assert_eq!(run.out(5), 0),
            Outcome::Fault { thread, .. } => assert_eq!(thread, MAIN_ID + 8),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_reader_writer_lock_of_an_ended_thread_cannot_be_released_by_another() {
    for main in ["main_orphaned_exclusive", "main_orphaned_shared"] {
        let run = run(main);
        assert!(
            matches!(run.report.outcome, Outcome::Fault { thread, .. } if thread == MAIN_ID + 8),
            "{main}: {:?}",
            run.report.outcome
        );
        assert_eq!(run.memory.read_u32(SHARED + 548).unwrap(), 0, "{main}");
    }
    assert_eq!(run("main_orphaned_shared").outs(2), [1, 0]);
}

#[test]
fn pending_initializations_are_bounded() {
    let run = run_with("main_pending_bound", 1024, 200_000_000);
    assert!(
        matches!(run.report.outcome, Outcome::Fault { .. }),
        "{:?}",
        run.report.outcome
    );
    assert_eq!(run.memory.read_u64(COUNTER).unwrap(), 4096);
}

#[test]
fn a_worker_can_duplicate_its_own_pseudo_handle() {
    let run = run("main_duplicate").halted();
    assert_eq!(
        run.outs(9),
        [
            1, 1, 258, 1, // a real handle to the worker itself
            0, 50, // it has no priority to set
            0, 44, // signaled, with its exit code
            0,  // the main thread's priority is unchanged
        ]
    );
}

#[test]
fn a_small_commitment_keeps_the_default_reservation() {
    let run = run("main_small_commit").halted();
    assert_eq!(run.out(0), (1..=200).sum::<u64>());
    let slot = |index: u64| run.memory.read_u64(SHARED + 264 + 8 * index).unwrap();
    assert!(
        slot(0) - slot(1) > 100_000,
        "the recursion really used the stack"
    );
    // With the flag the size is the reservation, rounded up to 64 KiB.
    let run = self::run("main_small_reservation");
    assert!(
        matches!(run.report.outcome, Outcome::Fault { .. }),
        "{:?}",
        run.report.outcome
    );
}

#[test]
fn guest_allocations_cannot_block_or_free_thread_memory() {
    let run = run("main_area_taken").halted();
    assert_eq!(run.outs(3), [FIRST_SLOT, 1, 1]);
    let run = self::run("main_free_stack").halted();
    assert_eq!(run.outs(2), [0, 0]);
}

#[test]
fn threads_can_be_ended_in_every_state() {
    for quantum in [256, 1024, 1_000_000] {
        let run = run_with("main_terminations", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(11),
            [
                1, 60, 0, // blocked: ended, and its wait consumed nothing
                1, 0, 61, 0, // suspended, then by itself; neither ran on
                0x80, 0, // both mutexes abandoned under wait-all, then normal
                FAILED, 156, // the 128th suspension is refused
            ],
            "quantum {quantum}"
        );
    }
}

#[test]
fn normal_thread_exit_runs_fls_callbacks_before_join() {
    let worker = u64::from(MAIN_ID + 4);
    for quantum in QUANTA {
        for (main, code) in [("main_fls_return", 0xCAFE), ("main_fls_explicit", 0xBEEF)] {
            let run = run_with(main, quantum, BUDGET).halted();
            assert_eq!(
                run.outs(14),
                [
                    worker, 0, code, 0x12345, worker, 0x67890, worker, 2, 0, 0, 1, 1, 1, 2
                ],
                "{main}, quantum {quantum}"
            );
        }
    }
}

#[test]
fn terminating_a_thread_discards_fls_values_without_callbacks() {
    let run = run("main_fls_terminate").halted();
    assert_eq!(
        run.outs(14),
        [
            u64::from(MAIN_ID + 4),
            0,
            0xDEAD,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0
        ]
    );
}

#[test]
fn an_exit_from_fls_cleanup_ends_the_worker_without_restarting_cleanup() {
    let worker = u64::from(MAIN_ID + 4);
    let run = run("main_fls_reenter").halted();
    assert_eq!(
        run.outs(16),
        [
            worker, 0, 0xF15, 0x12345, worker, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0
        ]
    );
}

#[test]
fn termination_and_teardown_cancel_in_progress_fls_cleanup() {
    let worker = u64::from(MAIN_ID + 4);
    for quantum in [1, DEFAULT_QUANTUM] {
        let run = run_with("main_fls_cancel", quantum, BUDGET).halted();
        let next = worker + 4;
        assert_eq!(
            run.outs(20),
            [
                worker, 0, 0xDEAD, 0x12345, next, 0x67890, next, 0, 0, 1, 1, 1, 1, 2, 0x12345, 259,
                258, next, 0, 0xCAFE
            ]
        );
        assert_ne!(run.out(20), 0);
        assert_eq!(run.out(20), run.out(21), "worker stack must be reused");
        assert!(run.memory.read_u64(FIRST_SLOT + 0x1000).is_err());

        let run = run_with("main_fls_teardown", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(17),
            [
                worker, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x12345, 259, 258
            ]
        );
        assert!(run.memory.read_u64(FIRST_SLOT + 0x1000).is_err());
    }
}

#[test]
fn every_small_budget_stops_cleanly() {
    for main in [
        "main_limit",
        "main_mutex",
        "main_fifo",
        "main_section",
        "main_queue",
        "main_srw",
        "main_once",
    ] {
        for quantum in [1, 64, 1024] {
            for budget in (1..1500).step_by(29) {
                let run = run_with(main, quantum, budget);
                assert!(!matches!(run.report.outcome, Outcome::Deadlock { .. }));
                // One block and one pass of retries may follow the limit.
                assert!(
                    run.report.instructions < budget + 32,
                    "{main} {quantum} {budget}: {}",
                    run.report.instructions
                );
            }
        }
    }
}

#[test]
fn retrying_many_waiters_stays_within_the_budget() {
    // Up to 63 threads wait at once; a pass over all of them must stop.
    for budget in (1500..12_000).step_by(211) {
        let run = run_with("main_limit", 64, budget);
        assert!(!matches!(run.report.outcome, Outcome::Deadlock { .. }));
        assert!(
            run.report.instructions < budget + 32,
            "{budget}: {}",
            run.report.instructions
        );
    }
}

#[test]
fn a_run_that_stopped_on_its_budget_keeps_main_thread_state() {
    let mut guest = guest("main_keeps_state");
    let mut cpu = guest.cpu.clone();
    let mut runs = 0;
    loop {
        let report = Scheduler::new(Arc::clone(&guest.state), 64)
            .run(&mut guest.engine, &mut guest.memory, &cpu, 500)
            .unwrap();
        runs += 1;
        if report.outcome == Outcome::Halted {
            break;
        }
        assert_eq!(report.outcome, Outcome::BudgetExhausted);
        cpu = report.main;
    }
    assert!(runs > 10, "{runs}");
    assert_eq!(guest.memory.read_u64(SHARED).unwrap(), 88);
    assert_eq!(guest.memory.read_u64(SHARED + 8).unwrap(), 77);
}

#[test]
fn thread_ids_are_not_reused_by_a_later_run() {
    let mut first = guest("main_worker_halts");
    let mut threads = Vec::new();
    for _ in 0..3 {
        let report = Scheduler::new(Arc::clone(&first.state), 1024)
            .run(&mut first.engine, &mut first.memory, &first.cpu, BUDGET)
            .unwrap();
        let Outcome::Fault { thread, .. } = report.outcome else {
            panic!("{:?}", report.outcome);
        };
        threads.push(thread);
    }
    assert_eq!(threads, [MAIN_ID + 4, MAIN_ID + 8, MAIN_ID + 12]);
}

#[test]
fn service_calls_that_end_a_turn_are_counted() {
    for (main, calls) in [
        ("main_signals", 1000),
        ("main_independent", 5),
        ("main_mutex", 100),
        ("main_fifo", 10),
        ("main_churn", 100),
        ("main_exits_first", 3),
        ("main_section", 800),
        ("main_queue", 400),
        ("main_once", 10),
        ("main_sleep", 5),
    ] {
        for quantum in QUANTA {
            let run = run_with(main, quantum, BUDGET);
            assert!(run.metrics.service_calls >= calls, "{main} {quantum}");
            assert_eq!(
                run.report.instructions,
                run.metrics.instructions_executed + run.metrics.service_calls,
                "{main} quantum {quantum}"
            );
        }
    }
}

/// Stopped by an operation the runtime refuses, in the main thread.
fn refused(main: &str) -> Run {
    let run = run(main);
    assert!(
        matches!(
            run.report.outcome,
            Outcome::Fault {
                thread: MAIN_ID,
                reason: ExitReason::IllegalInstruction(0xFE),
                ..
            }
        ),
        "{:?}",
        run.report.outcome
    );
    run
}

#[test]
fn critical_sections_exclude_threads_that_yield_inside() {
    for quantum in QUANTA {
        let run = run_with("main_section", quantum, BUDGET).halted();
        assert_eq!(run.memory.read_u64(COUNTER).unwrap(), 400);
        // Every wait ended, nobody met another thread inside, and the section
        // is free: LockCount is -1 and it has no owner.
        assert_eq!(run.outs(4), [0, 0, FAILED, 0], "quantum {quantum}");
    }
}

#[test]
fn blocked_threads_enter_a_critical_section_in_order() {
    for quantum in QUANTA {
        let run = run_with("main_section_order", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [0, 0, 1, 2, 3, 0], "quantum {quantum}");
    }
    for quantum in [256, 1024, 1_000_000] {
        let run = run_with("main_section_handoff", quantum, BUDGET).halted();
        assert_eq!(run.outs(2), [0, 1], "quantum {quantum}");
    }
}

#[test]
fn a_lock_nobody_can_release_is_reported_as_a_deadlock() {
    for main in ["main_orphaned_section", "main_recursive_srw"] {
        let run = run(main);
        let blocked = vec![MAIN_ID];
        let suspended = Vec::new();
        assert_eq!(
            run.report.outcome,
            Outcome::Deadlock { blocked, suspended },
            "{main}"
        );
        assert_eq!(run.out(0), 1);
    }
}

#[test]
fn slim_locks_share_readers_and_exclude_writers() {
    for quantum in QUANTA {
        let run = run_with("main_srw", quantum, BUDGET).halted();
        assert_eq!(run.memory.read_u64(COUNTER).unwrap(), 100);
        assert_eq!((run.out(0), run.out(1)), (0, 0), "quantum {quantum}");
        assert!((2..=3).contains(&run.out(2)), "{}", run.out(2));
        assert_eq!(run.out(3), 1);
    }
}

#[test]
fn overlapping_readers_do_not_keep_a_writer_out() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_writer_progress", quantum, BUDGET).halted();
        // Without a queue the readers would run all 600 of their rounds.
        assert!((1..100).contains(&run.out(0)), "{}", run.out(0));
        assert_eq!(run.out(1), 0);
    }
}

#[test]
fn a_writer_that_cannot_run_does_not_hold_up_readers() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_waiting_writer", quantum, BUDGET).halted();
        assert_eq!(run.outs(8), [0, 0, 0, 1, 1, 0, 2, 0], "quantum {quantum}");
    }
}

#[test]
fn slim_lock_waiters_go_in_the_order_they_came() {
    for quantum in [256, 1024, 1_000_000] {
        let run = run_with("main_srw_order", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [0, 0, 1, 2, 3, 0], "quantum {quantum}");
        let run = run_with("main_woken_writer_place", quantum, BUDGET).halted();
        assert_eq!(run.outs(2), [1, 0], "quantum {quantum}");
        let run = run_with("main_woken_writer", quantum, BUDGET).halted();
        assert_eq!(run.outs(4), [0, 0, 0, 1], "quantum {quantum}");
    }
}

#[test]
fn condition_variables_lose_no_item() {
    for quantum in QUANTA {
        let run = run_with("main_queue", quantum, BUDGET).halted();
        assert_eq!(run.out(0), 0);
        // Two producers each add 1 to 60.
        assert_eq!(run.memory.read_u64(COUNTER).unwrap(), 2 * 60 * 61 / 2);
        assert_eq!(run.out(8), 0, "quantum {quantum}");
    }
}

#[test]
fn a_condition_sleep_that_times_out_holds_its_lock_again() {
    for quantum in QUANTA {
        let run = run_with("main_condition_timeout", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(10),
            [0, 1460, 1, 0, 0, 1460, 0, 0, 0, 0],
            "quantum {quantum}"
        );
        assert_eq!(run.clock.performance_count(), 1_000_000);
    }
}

#[test]
fn condition_variables_wake_the_oldest_sleeper_or_all() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_wake_order", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(10),
            [1, 0, 1, 2, 2, 0, 11, 31, 41, 51],
            "quantum {quantum}"
        );
    }
}

#[test]
fn a_woken_sleeper_queues_behind_threads_already_waiting_for_the_lock() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_woken_order", quantum, BUDGET).halted();
        assert_eq!(run.outs(3), [0, 2, 11], "quantum {quantum}");
    }
}

#[test]
fn a_wake_up_goes_to_a_sleep_that_has_not_timed_out() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_late_wake", quantum, BUDGET).halted();
        assert_eq!(run.outs(4), [0, 1460, 0, 21], "quantum {quantum}");
    }
}

#[test]
fn waiting_for_the_lock_is_not_part_of_the_timeout() {
    for quantum in QUANTA {
        let run = run_with("main_slow_reacquire", quantum, BUDGET).halted();
        assert_eq!((run.out(0), run.out(2)), (1, 258), "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 2_000_000);
        let run = run_with("main_slow_timeout", quantum, BUDGET).halted();
        assert_eq!(run.outs(3), [0, 1460, 258], "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 2_000_000);
    }
}

#[test]
fn a_condition_sleep_in_a_section_entered_twice_is_refused() {
    assert_eq!(refused("main_sleep_entered_twice").out(0), 1);
}

#[test]
fn shared_sleepers_wait_for_the_writer_that_woke_them() {
    for quantum in QUANTA {
        let run = run_with("main_shared_sleepers", quantum, BUDGET).halted();
        assert_eq!(run.outs(2), [0, 0], "quantum {quantum}");
        assert_eq!(run.memory.read_u64(COUNTER).unwrap(), 3);
    }
}

#[test]
fn one_time_initialization_runs_once_and_the_others_wait() {
    for quantum in QUANTA {
        let run = run_with("main_once", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [0x1230, 0x1230, 0x1230, 0x1230, 0, 1]);
        assert_eq!(run.clock.performance_count(), 50_000);
    }
    for quantum in [64, 1024, 1_000_000] {
        // The first attempt fails; one waiter runs the initializer again.
        let run = run_with("main_once_retried", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [0, 0x1240, 0x1240, 0x1240, 0, 2]);
        assert_eq!(run.clock.performance_count(), 100_000);
    }
}

#[test]
fn a_begun_initialization_holds_up_other_threads() {
    for quantum in QUANTA {
        let run = run_with("main_once_begin", quantum, BUDGET).halted();
        assert_eq!(run.outs(5), [1, 0, 0x440, 258, 1], "quantum {quantum}");
        // Failure hands the initialization to the waiter.
        let run = run_with("main_once_begin_failed", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [1, 1, 0, 1, 0, 0x880], "quantum {quantum}");
    }
}

#[test]
fn a_failed_initialization_goes_to_the_waiter_first() {
    for quantum in [256, 1024, 1_000_000] {
        let run = run_with("main_once_handed_over", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [1, 1, 0, 1, 0, 0x880], "quantum {quantum}");
        // The first thread fails and calls again at once, but the second one
        // was waiting.
        let run = run_with("main_once_callback_handed_over", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [0, 32, 32, 16, 32, 2], "quantum {quantum}");
    }
    assert_eq!(refused("main_once_begun_twice").out(0), 1);
}

#[test]
fn sleep_delays_only_the_calling_thread() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_sleep", quantum, BUDGET).halted();
        assert_eq!(run.outs(5), [0, 0, 10, 20, 30], "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 450_000);
    }
    for quantum in QUANTA {
        assert_eq!(
            run_with("main_sleep_zero", quantum, BUDGET).halted().out(0),
            1
        );
    }
    let run = run("main_long_sleep").halted();
    assert_eq!(run.clock.performance_count(), 150_000_000);
}

#[test]
fn an_infinite_sleep_ends_only_with_its_thread() {
    let run = run("main_sleep_forever");
    let blocked = vec![MAIN_ID];
    let suspended = Vec::new();
    assert_eq!(run.report.outcome, Outcome::Deadlock { blocked, suspended });
    assert_eq!(run.outs(2), [258, 0]);
    assert_eq!(run.clock.performance_count(), 400_000);
}

#[test]
fn a_continued_run_keeps_the_rest_of_a_sleep() {
    let blocks = (1..200)
        .find(|&budget| {
            let (guest, _) = first_part_of("main_continued_sleep", budget);
            guest.engine.metrics.service_calls >= 1
        })
        .unwrap();
    let (mut guest, report) = first_part_of("main_continued_sleep", blocks);
    assert_eq!(report.outcome, Outcome::BudgetExhausted);
    // One tick is left.
    guest.clock.advance(9_999_999);
    let report = Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &report.main, BUDGET)
        .unwrap();
    assert_eq!(report.outcome, Outcome::Halted);
    assert_eq!(guest.clock.performance_count(), 10_000_000);
}

#[test]
fn a_continued_run_keeps_a_condition_sleep() {
    let blocks = (1..200)
        .find(|&budget| {
            let (guest, _) = first_part_of("main_continued_condition", budget);
            guest.engine.metrics.service_calls >= 3
        })
        .unwrap();
    let (mut guest, report) = first_part_of("main_continued_condition", blocks);
    assert_eq!(report.outcome, Outcome::BudgetExhausted);
    guest.clock.advance(9_999_999);
    let report = Scheduler::new(Arc::clone(&guest.state), 1024)
        .run(&mut guest.engine, &mut guest.memory, &report.main, BUDGET)
        .unwrap();
    assert_eq!(report.outcome, Outcome::Halted);
    assert_eq!(guest.clock.performance_count(), 10_000_000);
    let out = |index: u64| guest.memory.read_u64(SHARED + 8 * index).unwrap();
    assert_eq!((out(0), out(1), out(2)), (0, 1460, 1));
}

#[test]
fn task_queue_ready_work_survives_long_and_infinite_timeouts() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_ready", quantum, BUDGET).halted();
        assert_eq!(run.outs(8), [1, 1, 1, 1, 1, 1, 1, 2], "quantum {quantum}");
    }
}

#[test]
fn a_waiting_task_queue_dispatch_does_not_block_other_guest_threads() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_cross_thread_submit", quantum, BUDGET).halted();
        // Dispatch returned true, the callback ran on the main thread, and the
        // worker's submit succeeded.
        assert_eq!(run.outs(4), [1, 1, 1, 1], "quantum {quantum}");
    }
}

#[test]
fn terminating_a_task_queue_wakes_an_infinite_dispatch_as_false() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_infinite_terminated", quantum, BUDGET).halted();
        assert_eq!(run.outs(4), [0, 1, 0, 1], "quantum {quantum}");
    }
}

#[test]
fn a_task_queue_dispatch_timeout_expires_without_dropping_the_queue() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_timeout", quantum, BUDGET).halted();
        assert_eq!(run.outs(5), [0, 1, 1, 1, 1], "quantum {quantum}");
        // The monotonic test clock starts at zero; the waiter sleeps exactly 25 ms.
        assert_eq!(run.clock.performance_count(), 250_000, "quantum {quantum}");
    }
}

#[test]
fn work_submitted_after_a_task_queue_deadline_stays_queued() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_late_submit", quantum, BUDGET).halted();
        assert_eq!(run.outs(5), [0, 1, 1, 1, 1], "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 500_000, "quantum {quantum}");
    }
}

#[test]
fn task_queue_termination_orders_ports_and_completion_callback() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_termination_order", quantum, BUDGET).halted();
        // Creation/submission/termination succeeded; the first completion poll is
        // gated, then work, completion and termination callbacks dispatch once.
        assert_eq!(
            run.outs(9),
            [1, 1, 1, 1, 0, 1, 1, 1, 0],
            "quantum {quantum}"
        );
        let low = |index| run.out(index) & 0xffff_ffff;
        assert_eq!((low(20), low(21), low(22)), (1, 2, 3), "quantum {quantum}");
        assert_eq!(run.out(20) >> 32, 1, "work cancellation");
        assert_eq!(run.out(21) >> 32, 1, "completion cancellation");
    }
}

#[test]
fn synchronous_task_queue_termination_waits_for_callback_return() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_termination_wait", quantum, BUDGET).halted();
        assert_eq!(run.outs(6), [1, 1, 1, 1, 3, 3], "quantum {quantum}");
    }
}

#[test]
fn task_queue_termination_follows_queue_object_ownership() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_termination_owner_scope", quantum, BUDGET).halted();
        assert_eq!(
            run.outs(11),
            [1, 1, 1, 1, 0x8000_4004, 1, 0, 1, 1, 1, 1],
            "quantum {quantum}"
        );
        assert_eq!(run.out(20) >> 32, 1, "duplicate handle cancellation");
        assert_eq!(run.out(21) >> 32, 0, "composite remains independent");
    }
}

#[test]
fn delayed_task_queue_work_uses_the_scheduler_clock() {
    for quantum in [64, 1024, 1_000_000] {
        let run = run_with("main_tq_delayed_infinite", quantum, BUDGET).halted();
        assert_eq!(run.outs(4), [1, 1, 1, 1], "quantum {quantum}");
        assert_eq!(run.clock.performance_count(), 250_000, "quantum {quantum}");
    }
}
