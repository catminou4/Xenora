// SPDX-License-Identifier: GPL-3.0-or-later

//! Handle ownership for the documented manual GDK task queue subset.

use super::ServicesState;
use super::events::{INFINITE, TICKS_PER_MILLI};
use super::threads::Pending;
use super::{ERROR_INVALID_HANDLE, ERROR_INVALID_PARAMETER, ERROR_NOT_SUPPORTED};
use crate::cpu::ExitReason;
use crate::imports::{
    CallbackStart, CallbackStep, CallbackTask, GuestCallback, ImportRegistry, ServiceStatus,
    Win64Args,
};
use crate::memory::GuestMemory;
use crate::runtime::HResult;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MODULE: &str = "xgameruntime.dll";
const MAX_HANDLES: usize = 64;
const MAX_CALLBACKS_PER_PORT: usize = 32;
const ERROR_OUTOFMEMORY: u32 = 14;
const MANUAL: u32 = 0;
const E_ABORT: u64 = 0x8000_4004;

#[derive(Default)]
pub(super) struct QueueHandles {
    last_id: u64,
    queues: BTreeMap<u64, Queue>,
    terminated: BTreeSet<u64>,
    terminations: BTreeMap<u64, QueueTermination>,
    pending: BTreeMap<u64, VecDeque<ScheduledCallback>>,
}

#[derive(Clone, Copy)]
struct Queue {
    id: u64,
    ports: [u64; 2],
}

#[derive(Default)]
struct QueueTermination {
    callback: Option<GuestCallback>,
    work_remaining: usize,
    completion_remaining: usize,
    callback_queued: bool,
    complete: bool,
}

struct ScheduledCallback {
    owner: u64,
    termination: bool,
    ready_at: Instant,
    ready_ticks: Option<u64>,
    callback: GuestCallback,
}

#[derive(Clone, Copy)]
struct ReadyCallback {
    owner: u64,
    port: u32,
    termination: bool,
    callback: GuestCallback,
}

impl QueueHandles {
    /// Task-queue handles the guest has open.
    pub(super) fn open_count(&self) -> usize {
        self.queues.len()
    }

    /// Process exit: every handle is closed and nothing stays scheduled.
    pub(super) fn close_all(&mut self) {
        self.queues.clear();
        self.terminated.clear();
        self.terminations.clear();
        self.pending.clear();
    }

    fn open(
        &mut self,
        duplicate: Option<Queue>,
        composite: Option<[u64; 2]>,
        output: u64,
        memory: &mut GuestMemory,
    ) -> Result<(), u32> {
        if output == 0 {
            return Err(ERROR_INVALID_PARAMETER);
        }
        if self.queues.len() == MAX_HANDLES {
            return Err(ERROR_OUTOFMEMORY);
        }
        let next = self
            .last_id
            .checked_add(1)
            .filter(|id| *id < (1 << 48))
            .ok_or(ERROR_OUTOFMEMORY)?;
        let handle = 0x5144_0000_0000_0000 | next;
        memory
            .write_u64(output, handle)
            .map_err(|_| ERROR_INVALID_PARAMETER)?;
        self.last_id = next;
        self.queues.insert(
            handle,
            duplicate.unwrap_or(Queue {
                id: handle,
                ports: composite
                    .unwrap_or([0x5157_0000_0000_0000 | next, 0x5143_0000_0000_0000 | next]),
            }),
        );
        Ok(())
    }

    fn close(&mut self, handle: u64) {
        if let Some(queue) = self.queues.remove(&handle) {
            if !self.queues.values().any(|live| live.id == queue.id) {
                self.terminated.remove(&queue.id);
                self.terminations.remove(&queue.id);
            }
            for port in queue.ports {
                if !self.queues.values().any(|live| live.ports.contains(&port)) {
                    self.pending.remove(&port);
                }
            }
        }
    }

    fn submit(
        &mut self,
        handle: u64,
        selector: u32,
        mut callback: ScheduledCallback,
    ) -> Result<(), u32> {
        let queue = self.queues.get(&handle).ok_or(ERROR_INVALID_HANDLE)?;
        callback.owner = queue.id;
        let port = queue
            .ports
            .get(selector as usize)
            .ok_or(ERROR_INVALID_PARAMETER)?;
        let queue = self.pending.entry(*port).or_default();
        if queue.len() >= MAX_CALLBACKS_PER_PORT {
            return Err(ERROR_OUTOFMEMORY);
        }
        let index = queue.partition_point(|item| match (item.ready_ticks, callback.ready_ticks) {
            (Some(item), Some(callback)) => item <= callback,
            _ => item.ready_at <= callback.ready_at,
        });
        queue.insert(index, callback);
        Ok(())
    }

    /// Queue the termination callback once both ports have returned every
    /// callback that was pending when termination started. Returns true when
    /// waiters have to be woken.
    fn maybe_queue_termination_callback(
        &mut self,
        owner: u64,
        now: Instant,
        now_ticks: Option<u64>,
    ) -> bool {
        let ready = self.terminations.get(&owner).is_some_and(|termination| {
            termination.work_remaining == 0
                && termination.completion_remaining == 0
                && !termination.complete
                && !termination.callback_queued
        });
        if !ready {
            return false;
        }
        let Some(queue) = self
            .queues
            .values()
            .find(|queue| queue.id == owner)
            .copied()
        else {
            if let Some(termination) = self.terminations.get_mut(&owner) {
                termination.complete = true;
            }
            return true;
        };
        let callback = self
            .terminations
            .get(&owner)
            .and_then(|termination| termination.callback);
        if let Some(callback) = callback {
            let item = ScheduledCallback {
                owner,
                termination: true,
                ready_at: now,
                ready_ticks: now_ticks,
                callback,
            };
            let port = queue.ports[1];
            let ready_at = item.ready_at;
            let ready_ticks = item.ready_ticks;
            let pending = self.pending.entry(port).or_default();
            // Append after callbacks already ready at this instant; later delayed
            // callbacks of other queues keep their own deadline order.
            let index =
                pending.partition_point(|existing| match (existing.ready_ticks, ready_ticks) {
                    (Some(existing), Some(new)) => existing <= new,
                    _ => existing.ready_at <= ready_at,
                });
            pending.insert(index, item);
            if let Some(termination) = self.terminations.get_mut(&owner) {
                termination.callback_queued = true;
            }
        } else if let Some(termination) = self.terminations.get_mut(&owner) {
            termination.complete = true;
        }
        true
    }

    /// Begin termination. Subsequent calls are idempotent so a synchronous
    /// waiter can retry its blocked service call without queueing another
    /// termination sequence.
    fn terminate(
        &mut self,
        handle: u64,
        context: u64,
        entry: u64,
        now: Instant,
        now_ticks: Option<u64>,
    ) -> Result<bool, u32> {
        let queue = *self.queues.get(&handle).ok_or(ERROR_INVALID_HANDLE)?;
        if self.terminated.contains(&queue.id) {
            return Ok(false);
        }
        let mut work_remaining = 0;
        let mut completion_remaining = 0;
        for (port_index, port) in queue.ports.into_iter().enumerate() {
            if let Some(pending) = self.pending.get_mut(&port) {
                for item in pending.iter_mut().filter(|item| item.owner == queue.id) {
                    if port_index == 0 {
                        work_remaining += 1;
                    } else {
                        completion_remaining += 1;
                    }
                    item.ready_at = item.ready_at.min(now);
                    if let (Some(ready), Some(now)) = (item.ready_ticks, now_ticks) {
                        item.ready_ticks = Some(ready.min(now));
                    }
                    item.callback.cancelled = true;
                }
                pending.make_contiguous().sort_by_key(|item| item.ready_at);
            }
        }
        self.terminated.insert(queue.id);
        self.terminations.insert(
            queue.id,
            QueueTermination {
                callback: (entry != 0).then_some(GuestCallback {
                    entry,
                    context,
                    cancelled: false,
                }),
                work_remaining,
                completion_remaining,
                callback_queued: false,
                complete: false,
            },
        );
        self.maybe_queue_termination_callback(queue.id, now, now_ticks);
        Ok(true)
    }

    /// A dispatched callback returned to the queue dispatcher. Counts advance
    /// here, not when the callback is popped, so a synchronous termination
    /// waiter cannot pass a still-running callback.
    fn callback_returned(
        &mut self,
        ready: ReadyCallback,
        now: Instant,
        now_ticks: Option<u64>,
    ) -> bool {
        if !self.terminated.contains(&ready.owner) {
            return false;
        }
        if ready.termination {
            if let Some(termination) = self.terminations.get_mut(&ready.owner) {
                termination.complete = true;
            }
            return true;
        }
        if let Some(termination) = self.terminations.get_mut(&ready.owner) {
            match ready.port {
                0 => termination.work_remaining = termination.work_remaining.saturating_sub(1),
                1 => {
                    termination.completion_remaining =
                        termination.completion_remaining.saturating_sub(1);
                }
                _ => {}
            }
        }
        self.maybe_queue_termination_callback(ready.owner, now, now_ticks)
    }

    fn termination_complete(&self, handle: u64) -> bool {
        self.queues.get(&handle).is_none_or(|queue| {
            self.terminations
                .get(&queue.id)
                .is_some_and(|termination| termination.complete)
        })
    }
}

enum DispatchPoll {
    Ready(ReadyCallback),
    Wait {
        wake_at: Option<Instant>,
        wake_ticks: Option<u64>,
    },
    Done,
}

/// Poll one manual queue port. Termination does not skip the queue's own
/// cancelled callbacks; those are delivered before the dispatch reports false.
fn poll_dispatch(
    queues: &mut QueueHandles,
    handle: u64,
    selector: u32,
    memory: &GuestMemory,
    clock_ticks: Option<u64>,
) -> Result<DispatchPoll, ExitReason> {
    let Some(&queue) = queues.queues.get(&handle) else {
        return Ok(DispatchPoll::Done);
    };
    let Some(&port) = queue.ports.get(selector as usize) else {
        return Ok(DispatchPoll::Done);
    };
    let terminated = queues.terminated.contains(&queue.id);
    let work_remaining = queues
        .terminations
        .get(&queue.id)
        .map_or(0, |termination| termination.work_remaining);
    let now = Instant::now();
    let mut wake_at = None;
    let mut wake_ticks = None;
    if let Some(pending) = queues.pending.get_mut(&port)
        && let Some((index, item)) = pending.iter().enumerate().find(|(_, item)| {
            if !(terminated && item.owner == queue.id) {
                return true;
            }
            item.termination || selector == 0 || work_remaining == 0
        })
    {
        let ready = match (clock_ticks, item.ready_ticks) {
            (Some(now), Some(ready)) => ready <= now,
            _ => item.ready_at <= now,
        };
        if ready {
            memory
                .fetch_instruction_byte(item.callback.entry)
                .map_err(ExitReason::MemoryFault)?;
            let item = pending.remove(index).expect("polled callback");
            return Ok(DispatchPoll::Ready(ReadyCallback {
                owner: item.owner,
                port: selector,
                termination: item.termination,
                callback: item.callback,
            }));
        }
        wake_at = Some(item.ready_at);
        wake_ticks = item.ready_ticks;
    }
    if terminated {
        let complete = queues
            .terminations
            .get(&queue.id)
            .is_some_and(|termination| termination.complete);
        if complete || selector == 0 {
            Ok(DispatchPoll::Done)
        } else {
            // A completion dispatcher cannot report completion while terminated
            // callbacks are still returning or the termination callback has yet
            // to be queued.
            Ok(DispatchPoll::Wait {
                wake_at,
                wake_ticks,
            })
        }
    } else {
        Ok(DispatchPoll::Wait {
            wake_at,
            wake_ticks,
        })
    }
}

fn clear_task_wait(state: &ServicesState, teb: u64) {
    let mut table = state.threads.lock().expect("guest thread table");
    if matches!(table.waits.get(&teb), Some(Pending::TaskQueue { .. })) {
        table.waits.remove(&teb);
    }
}

fn existing_task_wait(state: &ServicesState, teb: u64) -> Option<Option<u64>> {
    let table = state.threads.lock().expect("guest thread table");
    match table.waits.get(&teb) {
        Some(Pending::TaskQueue {
            timeout_deadline, ..
        }) => Some(*timeout_deadline),
        _ => None,
    }
}

fn initial_timeout_deadline(state: &ServicesState, timeout: u32) -> Option<u64> {
    (timeout != INFINITE).then(|| {
        state
            .clock
            .performance_count()
            .saturating_add(u64::from(timeout).saturating_mul(TICKS_PER_MILLI))
    })
}

fn update_task_wait(
    state: &ServicesState,
    teb: u64,
    timeout_deadline: Option<u64>,
    callback_deadline: Option<u64>,
) {
    let deadline = match (timeout_deadline, callback_deadline) {
        (Some(timeout), Some(callback)) => Some(timeout.min(callback)),
        (timeout, None) => timeout,
        (None, callback) => callback,
    };
    state
        .threads
        .lock()
        .expect("guest thread table")
        .waits
        .insert(
            teb,
            Pending::TaskQueue {
                timeout_deadline,
                deadline,
            },
        );
}

/// Scheduler-backed dispatch: the service call stays blocked on its trampoline,
/// so other guest threads continue to run and can submit work or terminate the
/// queue. The queue is polled again on every scheduler retry.
fn dispatch_scheduled(
    state: &ServicesState,
    args: &Win64Args,
    teb: u64,
    timeout: u32,
    memory: &GuestMemory,
) -> Result<Option<ReadyCallback>, ExitReason> {
    // A retry whose deadline has passed is over before the queue is polled:
    // work submitted after that deadline stays queued, matching the scheduler's
    // timed-wait ordering rule.
    let prior_wait = existing_task_wait(state, teb);
    if prior_wait.is_some_and(|deadline| {
        deadline.is_some_and(|deadline| state.clock.performance_count() >= deadline)
    }) {
        clear_task_wait(state, teb);
        return Ok(None);
    }
    let poll = {
        let mut queues = state.gdk_queues.lock().expect("GDK queue handle table");
        poll_dispatch(
            &mut queues,
            args.rcx,
            args.rdx as u32,
            memory,
            Some(state.clock.performance_count()),
        )?
    };
    match poll {
        DispatchPoll::Ready(callback) => {
            clear_task_wait(state, teb);
            Ok(Some(callback))
        }
        DispatchPoll::Done => {
            clear_task_wait(state, teb);
            Ok(None)
        }
        DispatchPoll::Wait { .. } if timeout == 0 => Ok(None),
        DispatchPoll::Wait { wake_ticks, .. } => {
            let timeout_deadline =
                prior_wait.unwrap_or_else(|| initial_timeout_deadline(state, timeout));
            update_task_wait(state, teb, timeout_deadline, wake_ticks);
            Err(ExitReason::GuestBlocked)
        }
    }
}

/// Direct service call without a scheduler. This is the only place a host wait
/// is appropriate: there are no other guest threads that a host sleep could
/// starve. INFINITE waits for a queue change, with timed-out delayed callbacks
/// as the only finite self-wake.
fn dispatch_host(
    state: &ServicesState,
    args: &Win64Args,
    timeout: u32,
    memory: &GuestMemory,
) -> Result<Option<ReadyCallback>, ExitReason> {
    let deadline = if timeout == INFINITE {
        None
    } else {
        Instant::now().checked_add(Duration::from_millis(u64::from(timeout)))
    };
    let mut queues = state.gdk_queues.lock().expect("GDK queue handle table");
    loop {
        let poll = poll_dispatch(&mut queues, args.rcx, args.rdx as u32, memory, None)?;
        match poll {
            DispatchPoll::Ready(callback) => return Ok(Some(callback)),
            DispatchPoll::Done => return Ok(None),
            DispatchPoll::Wait { wake_at, .. } => {
                if timeout == 0 {
                    return Ok(None);
                }
                let now = Instant::now();
                if deadline.is_some_and(|deadline| now >= deadline) {
                    return Ok(None);
                }
                let wake_at = match (deadline, wake_at) {
                    (Some(deadline), Some(callback)) => Some(deadline.min(callback)),
                    (Some(deadline), None) => Some(deadline),
                    (None, callback) => callback,
                };
                if let Some(wake_at) = wake_at {
                    let remaining = wake_at.saturating_duration_since(now);
                    if remaining.is_zero() {
                        continue;
                    }
                    (queues, _) = state
                        .gdk_queue_changed
                        .wait_timeout(queues, remaining)
                        .expect("GDK queue handle table");
                } else {
                    queues = state
                        .gdk_queue_changed
                        .wait(queues)
                        .expect("GDK queue handle table");
                }
            }
        }
    }
}

struct DispatchTask {
    state: Arc<ServicesState>,
    owner: u64,
    port: u32,
    termination: bool,
}

impl CallbackTask for DispatchTask {
    fn resume(&mut self, _: u64, _: u64, _: &mut GuestMemory) -> Result<CallbackStep, ExitReason> {
        let now = Instant::now();
        let now_ticks = self
            .state
            .hooks
            .is_attached()
            .then(|| self.state.clock.performance_count());
        let changed = {
            let mut queues = self
                .state
                .gdk_queues
                .lock()
                .expect("GDK queue handle table");
            queues.callback_returned(
                ReadyCallback {
                    owner: self.owner,
                    port: self.port,
                    termination: self.termination,
                    callback: GuestCallback {
                        entry: 0,
                        context: 0,
                        cancelled: false,
                    },
                },
                now,
                now_ticks,
            )
        };
        if changed {
            self.state.gdk_queue_changed.notify_all();
            self.state.hooks.signal();
        }
        Ok(CallbackStep::Return(1))
    }
}

fn dispatch_start(
    state: &Arc<ServicesState>,
    args: &Win64Args,
    teb: u64,
    memory: &mut GuestMemory,
) -> Result<CallbackStart, ExitReason> {
    let timeout = args.r8 as u32;
    let scheduled = state
        .threads
        .lock()
        .expect("guest thread table")
        .id_of(teb)
        .is_some();
    let ready = if scheduled {
        dispatch_scheduled(state, args, teb, timeout, memory)?
    } else {
        dispatch_host(state, args, timeout, memory)?
    };
    let Some(ready) = ready else {
        return Ok(CallbackStart::Return(0));
    };
    Ok(CallbackStart::Call {
        task: Box::new(DispatchTask {
            state: Arc::clone(state),
            owner: ready.owner,
            port: ready.port,
            termination: ready.termination,
        }),
        entry: ready.callback.entry,
        args: [
            ready.callback.context,
            u64::from(ready.callback.cancelled),
            0,
            0,
        ],
    })
}

fn guest_hresult(result: Result<(), u32>) -> u64 {
    match result {
        Ok(()) => HResult::OK.guest_return(),
        Err(error) => HResult::from_win32(error).guest_return(),
    }
}

pub(super) fn register(registry: &mut ImportRegistry, state: &Arc<ServicesState>) {
    let create = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "XTaskQueueCreate",
        ServiceStatus::Partial,
        move |args, _, memory| {
            if args.rcx as u32 != MANUAL || args.rdx as u32 != MANUAL {
                return Ok(guest_hresult(Err(ERROR_NOT_SUPPORTED)));
            }
            let mut queues = create.gdk_queues.lock().expect("GDK queue handle table");
            Ok(guest_hresult(queues.open(None, None, args.r8, memory)))
        },
    );

    let duplicate = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "XTaskQueueDuplicateHandle",
        ServiceStatus::Partial,
        move |args, _, memory| {
            let mut queues = duplicate.gdk_queues.lock().expect("GDK queue handle table");
            let Some(&object) = queues.queues.get(&args.rcx) else {
                return Ok(guest_hresult(Err(ERROR_INVALID_HANDLE)));
            };
            Ok(guest_hresult(queues.open(
                Some(object),
                None,
                args.rdx,
                memory,
            )))
        },
    );

    let close = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "XTaskQueueCloseHandle",
        ServiceStatus::Partial,
        move |args, _, _| {
            close
                .gdk_queues
                .lock()
                .expect("GDK queue handle table")
                .close(args.rcx);
            close.gdk_queue_changed.notify_all();
            Ok(0)
        },
    );

    let get = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "XTaskQueueGetPort",
        ServiceStatus::Partial,
        move |args, _, memory| {
            let result = (|| {
                let queues = get.gdk_queues.lock().expect("GDK queue handle table");
                let ports = queues.queues.get(&args.rcx).ok_or(ERROR_INVALID_HANDLE)?;
                let port = ports
                    .ports
                    .get(args.rdx as u32 as usize)
                    .ok_or(ERROR_INVALID_PARAMETER)?;
                if args.r8 == 0 {
                    return Err(ERROR_INVALID_PARAMETER);
                }
                memory
                    .write_u64(args.r8, *port)
                    .map_err(|_| ERROR_INVALID_PARAMETER)
            })();
            Ok(guest_hresult(result))
        },
    );

    let composite = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "XTaskQueueCreateComposite",
        ServiceStatus::Partial,
        move |args, _, memory| {
            let mut queues = composite.gdk_queues.lock().expect("GDK queue handle table");
            let ports = [args.rcx, args.rdx];
            if ports
                .iter()
                .any(|port| !queues.queues.values().any(|live| live.ports.contains(port)))
            {
                return Ok(guest_hresult(Err(ERROR_INVALID_HANDLE)));
            }
            Ok(guest_hresult(queues.open(
                None,
                Some(ports),
                args.r8,
                memory,
            )))
        },
    );

    for (symbol, delayed) in [
        ("XTaskQueueSubmitCallback", false),
        ("XTaskQueueSubmitDelayedCallback", true),
    ] {
        let submit = Arc::clone(state);
        registry.register_with_status(
            MODULE,
            symbol,
            ServiceStatus::Partial,
            move |args, _, memory| {
                let (delay, context, entry) = if delayed {
                    let Ok(entry) = args.get(4, memory) else {
                        return Ok(guest_hresult(Err(ERROR_INVALID_PARAMETER)));
                    };
                    (args.r8 as u32, args.r9, entry)
                } else {
                    (0, args.r8, args.r9)
                };
                if entry == 0 || memory.fetch_instruction_byte(entry).is_err() {
                    return Ok(guest_hresult(Err(ERROR_INVALID_PARAMETER)));
                }
                let mut queues = submit.gdk_queues.lock().expect("GDK queue handle table");
                if queues
                    .queues
                    .get(&args.rcx)
                    .is_some_and(|queue| queues.terminated.contains(&queue.id))
                {
                    return Ok(E_ABORT);
                }
                let ready_ticks = submit.hooks.is_attached().then(|| {
                    submit
                        .clock
                        .performance_count()
                        .saturating_add(u64::from(delay).saturating_mul(TICKS_PER_MILLI))
                });
                let callback = ScheduledCallback {
                    owner: 0,
                    termination: false,
                    ready_at: Instant::now() + Duration::from_millis(u64::from(delay)),
                    ready_ticks,
                    callback: GuestCallback {
                        entry,
                        context,
                        cancelled: false,
                    },
                };
                let result = queues.submit(args.rcx, args.rdx as u32, callback);
                if result.is_ok() {
                    submit.gdk_queue_changed.notify_all();
                }
                Ok(guest_hresult(result))
            },
        );
    }
    let terminate = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "XTaskQueueTerminate",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            let wait = args.rdx as u8 != 0;
            let (context, entry) = (args.r8, args.r9);
            if entry != 0 && memory.fetch_instruction_byte(entry).is_err() {
                return Ok(guest_hresult(Err(ERROR_INVALID_PARAMETER)));
            }
            let now = Instant::now();
            let now_ticks = terminate
                .hooks
                .is_attached()
                .then(|| terminate.clock.performance_count());
            let started = {
                let mut queues = terminate.gdk_queues.lock().expect("GDK queue handle table");
                queues.terminate(args.rcx, context, entry, now, now_ticks)
            };
            let started = match started {
                Ok(started) => started,
                Err(error) => return Ok(guest_hresult(Err(error))),
            };
            if started {
                terminate.gdk_queue_changed.notify_all();
                terminate.hooks.signal();
            }
            if !wait {
                return Ok(0);
            }
            if terminate.hooks.is_attached() {
                let complete = {
                    terminate
                        .gdk_queues
                        .lock()
                        .expect("GDK queue handle table")
                        .termination_complete(args.rcx)
                };
                if complete {
                    clear_task_wait(&terminate, cpu.gs_base);
                    return Ok(0);
                }
                update_task_wait(&terminate, cpu.gs_base, None, None);
                return Err(ExitReason::GuestBlocked);
            }
            let mut queues = terminate.gdk_queues.lock().expect("GDK queue handle table");
            while !queues.termination_complete(args.rcx) {
                queues = terminate
                    .gdk_queue_changed
                    .wait(queues)
                    .expect("GDK queue handle table");
            }
            Ok(0)
        },
    );
    let dispatch = Arc::clone(state);
    registry.register_callback_task(
        MODULE,
        "XTaskQueueDispatch",
        ServiceStatus::Partial,
        move |args, teb, memory, _| dispatch_start(&dispatch, args, teb, memory),
    );
}
