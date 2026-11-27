// SPDX-License-Identifier: GPL-3.0-or-later
//
// Original guest programs for scheduler-backed XTaskQueueDispatch tests.
// Evidence is stored in the fixed shared block mapped by runtime_threads.rs.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef void *HANDLE;
typedef u32 (*START)(void *);
typedef void (*QUEUE_CALLBACK)(void *, u32);

#define IMPORT __declspec(dllimport)
IMPORT HANDLE CreateThread(void *, u64, START, void *, u32, u32 *);
IMPORT u32 WaitForSingleObject(HANDLE, u32);
IMPORT int CloseHandle(HANDLE);
IMPORT u32 GetCurrentThreadId(void);
IMPORT int SwitchToThread(void);
IMPORT void Sleep(u32);
IMPORT u32 XTaskQueueCreate(u32, u32, HANDLE *);
IMPORT u32 XTaskQueueSubmitCallback(HANDLE, u32, void *, QUEUE_CALLBACK);
IMPORT u32 XTaskQueueSubmitDelayedCallback(HANDLE, u32, u32, void *, QUEUE_CALLBACK);
IMPORT u32 XTaskQueueDispatch(HANDLE, u32, u32);
IMPORT u32 XTaskQueueTerminate(HANDLE, u32, void *, void (*)(void *));
IMPORT u32 XTaskQueueDuplicateHandle(HANDLE, HANDLE *);
IMPORT u32 XTaskQueueGetPort(HANDLE, u32, HANDLE *);
IMPORT u32 XTaskQueueCreateComposite(HANDLE, HANDLE, HANDLE *);

#define INFINITE 0xFFFFFFFFu
#define THREAD(n) ((void *)(u64)(n))
struct shared {
    volatile u64 out[32];
    volatile u64 counter;
    volatile u64 queue;
    volatile u64 thread;
    volatile u64 main_thread;
};
#define S ((struct shared *)0x50000)

static HANDLE spawn(START start, u64 parameter) {
    return CreateThread(0, 0, start, THREAD(parameter), 0, 0);
}

static void join(HANDLE thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

static u32 create_queue(void) {
    return XTaskQueueCreate(0, 0, (HANDLE *)(void *)&S->queue) == 0;
}

// context is the output slot. Bit zero records that the callback ran on the
// dispatching thread; bit 32 would record cancellation.
static void record(void *context, u32 cancelled) {
    u64 slot = (u64)context;
    S->out[slot] = (GetCurrentThreadId() == (u32)S->main_thread) | ((u64)cancelled << 32);
    S->counter++;
}

// Ready work is dispatched even when the timeout is above the old 1000 ms
// implementation limit, including INFINITE.
void main_tq_ready(void) {
    S->main_thread = GetCurrentThreadId();
    S->out[0] = create_queue();
    S->out[1] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(8), record) == 0;
    S->out[2] = XTaskQueueDispatch((HANDLE)S->queue, 0, 1001);
    S->out[3] = S->out[8];
    S->out[4] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(9), record) == 0;
    S->out[5] = XTaskQueueDispatch((HANDLE)S->queue, 0, INFINITE);
    S->out[6] = S->out[9];
    S->out[7] = S->counter;
}

static u32 submitter(void *unused) {
    // Let the dispatcher reach its wait first even at large quanta.
    for (int i = 0; i < 20; i++) SwitchToThread();
    S->out[5] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(8), record) == 0;
    return 0;
}

// A dispatch waiting with INFINITE blocks only its own guest thread. The
// worker submits the callback and the blocked main thread receives it.
void main_tq_cross_thread_submit(void) {
    S->main_thread = GetCurrentThreadId();
    S->out[3] = create_queue();
    S->thread = (u64)spawn(submitter, 0);
    S->out[0] = XTaskQueueDispatch((HANDLE)S->queue, 0, INFINITE);
    S->out[1] = S->out[8];
    join((HANDLE)S->thread);
    S->out[2] = S->out[5];
}

static u32 terminator(void *unused) {
    // Let the dispatcher reach its INFINITE wait first even at large quanta.
    for (int i = 0; i < 20; i++) SwitchToThread();
    XTaskQueueTerminate((HANDLE)S->queue, 0, 0, 0);
    S->out[5] = 1;
    return 0;
}

// With no cancelled callback left to deliver, termination during an INFINITE
// wait makes the dispatch return false.
void main_tq_infinite_terminated(void) {
    S->out[3] = create_queue();
    S->thread = (u64)spawn(terminator, 0);
    S->out[0] = XTaskQueueDispatch((HANDLE)S->queue, 0, INFINITE);
    join((HANDLE)S->thread);
    S->out[1] = S->out[5];
}


static u32 late_submitter(void *unused) {
    Sleep(50);
    S->out[1] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(8), record) == 0;
    return 0;
}

// Work submitted only after a finite deadline does not satisfy the expired
// wait. It remains queued for a later dispatch.
void main_tq_late_submit(void) {
    S->main_thread = GetCurrentThreadId();
    S->out[4] = create_queue();
    S->thread = (u64)spawn(late_submitter, 0);
    S->out[0] = XTaskQueueDispatch((HANDLE)S->queue, 0, 25);
    join((HANDLE)S->thread);
    S->out[2] = XTaskQueueDispatch((HANDLE)S->queue, 0, 0);
    S->out[3] = S->out[8];
}



static void record_ordered(void *context, u32 cancelled) {
    u64 slot = (u64)context;
    S->out[slot] = ++S->counter | ((u64)cancelled << 32);
}

static void terminated(void *context) {
    u64 slot = (u64)context;
    S->out[slot] = ++S->counter;
}

// Work callbacks pending when termination starts finish before completion
// callbacks. The termination callback is then queued to the completion port.
void main_tq_termination_order(void) {
    S->out[0] = create_queue();
    S->out[1] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(10), record_ordered) == 0;
    S->out[2] = XTaskQueueSubmitCallback((HANDLE)S->queue, 1, THREAD(11), record_ordered) == 0;
    S->out[3] = XTaskQueueTerminate((HANDLE)S->queue, 0, THREAD(12), terminated) == 0;
    // A completion poll cannot pass the still-pending work cancellation.
    S->out[4] = XTaskQueueDispatch((HANDLE)S->queue, 1, 0);
    S->out[5] = XTaskQueueDispatch((HANDLE)S->queue, 0, 0);
    S->out[6] = XTaskQueueDispatch((HANDLE)S->queue, 1, 0);
    S->out[7] = XTaskQueueDispatch((HANDLE)S->queue, 1, 0);
    S->out[8] = XTaskQueueDispatch((HANDLE)S->queue, 1, 0);
    S->out[20] = S->out[10];
    S->out[21] = S->out[11];
    S->out[22] = S->out[12];
}

static u32 pump_both_ports(void *unused) {
    while (XTaskQueueDispatch((HANDLE)S->queue, 0, 0)) {}
    while (XTaskQueueDispatch((HANDLE)S->queue, 1, 0)) {}
    return 0;
}

// A synchronous terminate blocks only the calling guest thread. The worker
// dispatches both manual ports, and terminate returns after the termination
// callback itself has returned.
void main_tq_termination_wait(void) {
    S->out[0] = create_queue();
    S->out[1] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(10), record) == 0;
    S->out[2] = XTaskQueueSubmitCallback((HANDLE)S->queue, 1, THREAD(11), record) == 0;
    S->thread = (u64)spawn(pump_both_ports, 0);
    S->out[3] = XTaskQueueTerminate((HANDLE)S->queue, 1, THREAD(12), terminated) == 0;
    // The callback evidence is complete before the synchronous call returns.
    S->out[4] = S->counter;
    S->out[5] = S->out[12];
    join((HANDLE)S->thread);
}

// Duplicate handles name the same queue object, while a composite queue that
// shares a port remains a distinct owner and is not cancelled with it.
void main_tq_termination_owner_scope(void) {
    HANDLE duplicate = 0, port = 0, composite = 0;
    S->out[0] = create_queue();
    S->out[1] = XTaskQueueDuplicateHandle((HANDLE)S->queue, &duplicate) == 0;
    S->out[2] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(10), record) == 0;
    S->out[3] = XTaskQueueTerminate(duplicate, 0, 0, 0) == 0;
    S->out[4] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(11), record);
    S->out[5] = XTaskQueueDispatch((HANDLE)S->queue, 0, 0);
    S->out[20] = S->out[10];

    S->out[7] = XTaskQueueGetPort((HANDLE)S->queue, 0, &port) == 0;
    S->out[8] = XTaskQueueCreateComposite(port, port, &composite) == 0;
    S->out[9] = XTaskQueueSubmitCallback(composite, 0, THREAD(11), record) == 0;
    S->out[10] = XTaskQueueDispatch(composite, 0, 0);
    S->out[21] = S->out[11];
}

// Delayed work uses the scheduler clock too: an INFINITE dispatcher sleeps
// until the delay instead of deadlocking behind host time.
void main_tq_delayed_infinite(void) {
    S->main_thread = GetCurrentThreadId();
    S->out[3] = create_queue();
    S->out[0] = XTaskQueueSubmitDelayedCallback((HANDLE)S->queue, 0, 25, THREAD(8), record) == 0;
    S->out[1] = XTaskQueueDispatch((HANDLE)S->queue, 0, INFINITE);
    S->out[2] = S->out[8];
}

// A finite wait expires without dropping the queue. Afterwards immediate work
// is still available to a zero poll.
void main_tq_timeout(void) {
    S->main_thread = GetCurrentThreadId();
    S->out[3] = create_queue();
    S->out[0] = XTaskQueueDispatch((HANDLE)S->queue, 0, 25);
    S->out[1] = XTaskQueueSubmitCallback((HANDLE)S->queue, 0, THREAD(8), record) == 0;
    S->out[2] = XTaskQueueDispatch((HANDLE)S->queue, 0, 0);
    S->out[4] = S->out[8];
}
