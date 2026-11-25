// SPDX-License-Identifier: GPL-3.0-or-later

//! The C runtime's file streams: what `fopen` opens.
//!
//! A stream reads and writes a file on the in-memory `C:` drive that the
//! handle services in `files.rs` also use, as the C runtime sits on Win32
//! files on Windows. The rules are from the Microsoft Learn pages "fopen,
//! _wfopen", "fclose, _fcloseall", "fgets, fgetws" and "perror, _wperror".
//!
//! Modes: `r`, `w` or `a`, then any of `+`, `b` and `t`. In text mode, which
//! is the default as `_fmode` starts as `_O_TEXT`, "LF characters are
//! translated to CR-LF combinations on output", "CR-LF combinations are
//! translated into single line feeds on input", and "CTRL+Z is interpreted as
//! an end-of-file character on input".
//!
//! Nothing is buffered: a write reaches the drive at once. A position is the
//! byte offset in the file on the drive, in text mode too, so a position
//! that `_ftelli64` or `fgetpos` returned can be given back to `_fseeki64` or
//! `fsetpos` ("fseek, _fseeki64", "ftell, _ftelli64", "fgetpos", "fsetpos").

use super::GuestObject;
use super::ObjectState;
use super::files::DriveEntry;
use crate::memory::{GuestMemory, PROCESS_HEAP};
use std::collections::BTreeMap;

/// errno.h values.
pub(super) const ENOENT: u32 = 2;
pub(super) const EBADF: u32 = 9;
/// A wide character the C locale cannot write.
pub(super) const EILSEQ: u32 = 42;
pub(super) const ENOMEM: u32 = 12;
pub(super) const EACCES: u32 = 13;
pub(super) const EINVAL: u32 = 22;
pub(super) const EMFILE: u32 = 24;
pub(super) const ERANGE: u32 = 34;
pub(super) const ENOSPC: u32 = 28;

/// The most streams `fopen` keeps open at once.
pub(super) const MAX_OPEN_STREAMS: usize = 64;
/// `EEXIST`, for `_open` with `_O_CREAT | _O_EXCL`.
pub(super) const EEXIST: u32 = 17;
/// The most file descriptors, as the CRT's default table size is 8192.
pub(super) const MAX_DESCRIPTORS: u32 = 8192;
/// `CTRL+Z`, which ends the input of a text-mode stream.
const CTRL_Z: u8 = 0x1A;

/// The text `perror` prints for an `errno` value. The page gives the text for
/// a missing file in its example; the others are the usual C library wording
/// and are Xenora's choice.
pub(super) fn error_text(errno: u32) -> &'static str {
    match errno {
        0 => "No error",
        ENOENT => "No such file or directory",
        EBADF => "Bad file descriptor",
        ENOMEM => "Not enough space",
        EACCES => "Permission denied",
        EINVAL => "Invalid argument",
        EMFILE => "Too many open files",
        ENOSPC => "No space left on device",
        ERANGE => "Result too large",
        _ => "Unknown error",
    }
}

pub(super) struct Mode {
    readable: bool,
    writable: bool,
    /// Every write goes to the end of the file.
    append: bool,
    /// The file must already exist.
    existing: bool,
    /// The file's contents are destroyed on opening.
    truncate: bool,
    text: bool,
}

/// `r`, `w` or `a` first, then `+`, `b` and `t` in any order, each at most
/// once and not both `b` and `t`. Anything else is not a mode this runtime
/// opens, including the other characters the page lists.
pub(super) fn parse_mode(mode: &[u8]) -> Option<Mode> {
    let (&kind, rest) = mode.split_first()?;
    let (mut update, mut binary, mut text) = (false, false, false);
    for &byte in rest {
        let seen = match byte {
            b'+' => &mut update,
            b'b' => &mut binary,
            b't' => &mut text,
            _ => return None,
        };
        if std::mem::replace(seen, true) {
            return None;
        }
    }
    if binary && text {
        return None;
    }
    let (readable, writable, append, existing, truncate) = match kind {
        b'r' => (true, update, false, true, false),
        b'w' => (update, true, false, false, true),
        b'a' => (update, true, true, false, false),
        _ => return None,
    };
    Some(Mode {
        readable,
        writable,
        append,
        existing,
        truncate,
        text: !binary,
    })
}

/// The first io operation on a stream fixes its orientation, as the CRT
/// fixes it: byte or wide. The other kind then fails. Returns false for a
/// conflict.
fn orient(stream: &mut FileStream, wide: bool) -> bool {
    let wanted = if wide { 2 } else { 1 };
    if stream.orientation == 0 {
        stream.orientation = wanted;
    }
    stream.orientation == wanted
}

struct FileStream {
    path: String,
    /// Where the next read, or the next write of a stream that does not
    /// append, takes place.
    position: u64,
    readable: bool,
    writable: bool,
    append: bool,
    text: bool,
    /// `tmpfile` opened it or `_O_TEMPORARY` was passed: the file is
    /// deleted when the stream or its descriptor closes.
    temporary: bool,
    /// The end-of-file indicator, set when a read meets the end (or the
    /// CTRL+Z of a text stream) and cleared by `clearerr` or a seek.
    eof: bool,
    /// The error indicator, set when a write through the stream fails and
    /// cleared by `clearerr` — the write reports its `errno` at once too.
    error: bool,
    /// The C orientation: 0 unoriented, 1 byte, 2 wide. The first io
    /// operation sets it; the other kind of operation then fails.
    orientation: u8,
    /// Bytes `ungetc` pushed back, in push order: the last byte pushed is
    /// the first one a read hands out again, as the CRT stacks them.
    pushback: Vec<u8>,
}

/// A file descriptor, as `_open` and `_fileno` hand out. 0, 1 and 2 are
/// the standard streams and never enter this table.
enum Descriptor {
    /// `_open` opened it: the state is its own.
    Owned(FileStream),
    /// `_fileno` gave it out for a stream: the state is the stream's, as
    /// the CRT shares one file between a stream and its descriptor.
    Shared(u64),
    /// `_dup` or `_dup2` made it: the state is another descriptor's, as the
    /// CRT shares the file position across the duplication.
    Alias(u32),
    /// The read end of an anonymous pipe, by the pipe's identifier.
    PipeRead(u32),
    /// The write end of one.
    PipeWrite(u32),
    /// `_open_osfhandle` made it: the state is a kernel object's, as the
    /// CRT takes ownership of the handle it wraps.
    Handle(u64),
}

/// An anonymous pipe's shared state: the bytes written and not yet read,
/// and whether each end is still open. The end of the pipe is only end of
/// file once the write end has closed.
pub(super) struct PipeState {
    buffer: std::collections::VecDeque<u8>,
    write_open: bool,
}

#[derive(Default)]
pub(super) struct FileStreams {
    /// Open streams by the guest address of their `FILE` object.
    open: BTreeMap<u64, FileStream>,
    /// The counter `tmpnam` and `tmpfile` names come from.
    temporary_counter: u32,
    /// Open file descriptors by their number, from 3.
    descriptors: BTreeMap<u32, Descriptor>,
    /// Anonymous pipes `_pipe` creates, by identifier.
    pipes: BTreeMap<u32, PipeState>,
    pipe_counter: u32,
}

impl FileStreams {
    pub(super) fn contains(&self, file: u64) -> bool {
        self.open.contains_key(&file)
    }

    /// `fopen`. Returns the guest address of a new `FILE` object, or the
    /// `errno` value of the failure.
    pub(super) fn open(
        &mut self,
        memory: &mut GuestMemory,
        drive: &mut ObjectState,
        name: &[u8],
        mode: &[u8],
    ) -> Result<u64, u32> {
        // An empty name or mode is an invalid parameter: neither is a mode
        // nor a name the drive accepts. Xenora's choice: so is any other mode
        // it does not open and any other name the drive rejects.
        let mode = parse_mode(mode).ok_or(EINVAL)?;
        let path = ObjectState::drive_path(name).ok_or(EINVAL)?;
        if self.open.len() >= MAX_OPEN_STREAMS {
            return Err(EMFILE);
        }
        let create = match drive.drive_entry(&path) {
            DriveEntry::Directory => return Err(EACCES),
            DriveEntry::File => false,
            DriveEntry::Missing { .. } if mode.existing => return Err(ENOENT),
            DriveEntry::Missing { parent: false } => return Err(ENOENT),
            DriveEntry::Missing { parent: true } => true,
        };
        // The FILE object is allocated before the drive is touched, so a
        // failed fopen neither creates a file nor destroys the contents of
        // one.
        let file = memory
            .map_heap_allocation(PROCESS_HEAP, 8)
            .map_err(|_| ENOMEM)?;
        if create && !drive.drive_create(&path) {
            memory.release_heap_allocation(PROCESS_HEAP, file);
            return Err(ENOSPC);
        }
        if mode.writable && drive.drive_is_readonly(&path) {
            memory.release_heap_allocation(PROCESS_HEAP, file);
            return Err(EACCES);
        }
        if mode.truncate {
            drive.drive_truncate(&path);
        }
        self.open.insert(
            file,
            FileStream {
                path,
                position: 0,
                readable: mode.readable,
                writable: mode.writable,
                append: mode.append,
                text: mode.text,
                temporary: false,
                eof: false,
                error: false,
                orientation: 0,
                pushback: Vec::new(),
            },
        );
        Ok(file)
    }

    /// `fclose`. False when `file` is not an open stream. A stream
    /// `tmpfile` opened takes its file with it.
    pub(super) fn close(
        &mut self,
        memory: &mut GuestMemory,
        drive: &mut ObjectState,
        file: u64,
    ) -> bool {
        let Some(stream) = self.open.remove(&file) else {
            return false;
        };
        if stream.temporary {
            drive.drive_remove(&stream.path);
        }
        self.drop_descriptors_of(file);
        memory.release_heap_allocation(PROCESS_HEAP, file);
        true
    }

    /// A name `tmpnam` and `tmpfile` can use: `tmp` and a decimal counter,
    /// never one the drive holds. The CRT counts from 1 for the lifetime of
    /// the process, and so does this.
    pub(super) fn temporary_name(&mut self, drive: &ObjectState) -> String {
        loop {
            self.temporary_counter += 1;
            let name = format!("tmp{}", self.temporary_counter);
            if matches!(drive.drive_entry(&name), DriveEntry::Missing { .. }) {
                return name;
            }
        }
    }

    /// `tmpfile`: a stream open for reading and writing in binary, on a
    /// file no other stream uses, deleted when the stream closes.
    pub(super) fn open_temporary(
        &mut self,
        memory: &mut GuestMemory,
        drive: &mut ObjectState,
    ) -> Result<u64, u32> {
        if self.open.len() >= MAX_OPEN_STREAMS {
            return Err(EMFILE);
        }
        let file = memory
            .map_heap_allocation(PROCESS_HEAP, 8)
            .map_err(|_| ENOMEM)?;
        let path = self.temporary_name(drive);
        if !drive.drive_create(&path) {
            memory.release_heap_allocation(PROCESS_HEAP, file);
            return Err(ENOSPC);
        }
        self.open.insert(
            file,
            FileStream {
                path,
                position: 0,
                readable: true,
                writable: true,
                append: false,
                text: false,
                temporary: true,
                eof: false,
                error: false,
                orientation: 0,
                pushback: Vec::new(),
            },
        );
        Ok(file)
    }

    /// Write to an open stream, in its mode. Returns the `errno` value of a
    /// failure, with nothing written.
    pub(super) fn write(
        &mut self,
        drive: &mut ObjectState,
        file: u64,
        bytes: &[u8],
    ) -> Result<(), u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        if !stream.writable {
            stream.error = true;
            return Err(EBADF);
        }
        if !orient(stream, false) {
            stream.error = true;
            return Err(EBADF);
        }
        self.write_bytes(drive, file, bytes)
    }

    /// The write itself, the orientation already taken care of.
    fn write_bytes(&mut self, drive: &mut ObjectState, file: u64, bytes: &[u8]) -> Result<(), u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        let stored: Vec<u8> = if stream.text {
            let mut stored = Vec::with_capacity(bytes.len());
            for &byte in bytes {
                if byte == b'\n' {
                    stored.push(b'\r');
                }
                stored.push(byte);
            }
            stored
        } else {
            bytes.to_vec()
        };
        // "All write operations occur at the end of the file."
        if stream.append {
            stream.position = drive.drive_file(&stream.path).ok_or(EBADF)?.len() as u64;
        }
        let outcome = match drive.drive_write(&stream.path, stream.position, &stored) {
            Some(true) => {
                stream.position += stored.len() as u64;
                Ok(())
            }
            Some(false) => Err(ENOSPC),
            // The file was deleted or moved while the stream was open.
            None => Err(EBADF),
        };
        if outcome.is_err() {
            stream.error = true;
        }
        outcome
    }

    /// `fgets`: up to `limit` characters, ending after a newline. `None` is
    /// end of file with nothing read.
    /// `fputws` on a file stream: every wide character converts in the C
    /// locale — one byte of the same value, or nothing when it is above
    /// 255, which fails the write. Returns false for an orientation
    /// conflict, as the CRT returns `WEOF` for one.
    pub(super) fn write_wide(
        &mut self,
        drive: &mut ObjectState,
        file: u64,
        wide: &[u16],
    ) -> Result<bool, u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        if !stream.writable {
            return Err(EBADF);
        }
        if !orient(stream, true) {
            return Ok(false);
        }
        let mut bytes = Vec::with_capacity(wide.len());
        for unit in wide {
            let Some(byte) = u8::try_from(*unit).ok() else {
                return Err(EILSEQ);
            };
            bytes.push(byte);
        }
        self.write_bytes(drive, file, &bytes)?;
        Ok(true)
    }

    /// `fgetws` on a file stream: bytes read as `fgets` reads them, every
    /// one zero-extended to its wide character. `None` is end of file with
    /// nothing read; the inner `Ok(false)` is an orientation conflict.
    pub(super) fn read_wide_line(
        &mut self,
        drive: &ObjectState,
        file: u64,
        limit: usize,
    ) -> Result<Option<Vec<u16>>, u32> {
        // Orient before delegating, so a conflict does not consume input.
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        // An orientation conflict reads nothing, exactly what end of file
        // reads: `fgetws` answers null to both.
        if !orient(stream, true) {
            return Ok(None);
        }
        if !stream.readable {
            return Err(EBADF);
        }
        let data = drive.drive_file(&stream.path).ok_or(EBADF)?;
        let mut at = usize::try_from(stream.position).map_or(data.len(), |at| at.min(data.len()));
        if limit == 0 {
            let ended = at == data.len() || (stream.text && data[at] == CTRL_Z);
            return Ok((!ended).then(Vec::new));
        }
        let mut line = Vec::new();
        while line.len() < limit && at < data.len() {
            let mut byte = data[at];
            if stream.text {
                if byte == CTRL_Z {
                    break;
                }
                if byte == b'\r' && data.get(at + 1) == Some(&b'\n') {
                    at += 1;
                    byte = b'\n';
                }
            }
            at += 1;
            line.push(u16::from(byte));
            if byte == b'\n' {
                break;
            }
        }
        stream.position = at as u64;
        Ok((!line.is_empty()).then_some(line))
    }

    /// The position of a stream: where its next read takes place, or its
    /// next write unless it appends, less the bytes ungetc stacked — the
    /// position indicator steps back one for each byte pushed, never below
    /// zero (ISO C 7.21.7.10).
    pub(super) fn tell(&self, file: u64) -> Result<u64, u32> {
        self.open
            .get(&file)
            .map(|stream| stream.position.saturating_sub(stream.pushback.len() as u64))
            .ok_or(EBADF)
    }

    /// Move a stream to `offset` from the start (`origin` 0), the current
    /// position (1) or the end of the file (2). A position before the start
    /// is `EINVAL`; one past the end is allowed, and a write there fills the
    /// gap with zeros. A stream whose file was deleted fails with `EBADF`.
    pub(super) fn seek(
        &mut self,
        drive: &ObjectState,
        file: u64,
        offset: i64,
        origin: u32,
    ) -> Result<(), u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        let base = match origin {
            0 => 0,
            1 => stream.position,
            2 => drive.drive_file(&stream.path).ok_or(EBADF)?.len() as u64,
            _ => return Err(EINVAL),
        };
        let target = i128::from(base) + i128::from(offset);
        if !(0..=i128::from(i64::MAX)).contains(&target) {
            return Err(EINVAL);
        }
        drive.drive_file(&stream.path).ok_or(EBADF)?;
        stream.position = target as u64;
        // A successful seek or fsetpos clears the end-of-file indicator
        // (ISO C 7.21.9.2 and 7.21.9.3) and undoes the pushback, as a seek
        // forgets the bytes ungetc stacked (ISO C 7.21.7.10).
        stream.eof = false;
        stream.pushback.clear();
        Ok(())
    }

    /// Raw bytes from a stream's file, for `_read`. Text mode translates
    /// CR LF into one line feed and ends at CTRL+Z, exactly as a line read
    /// does.
    pub(super) fn read_bytes(
        &mut self,
        drive: &ObjectState,
        file: u64,
        limit: usize,
    ) -> Result<Vec<u8>, u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        if !stream.readable {
            return Err(EBADF);
        }
        let data = drive.drive_file(&stream.path).ok_or(EBADF)?;
        let mut at = usize::try_from(stream.position).map_or(data.len(), |at| at.min(data.len()));
        let start = at;
        let mut bytes = Vec::new();
        // Bytes ungetc pushed come first, the last pushed first, and the
        // position climbs back over them as they are consumed.
        while bytes.len() < limit && !stream.pushback.is_empty() {
            bytes.push(stream.pushback.pop().expect("pushback byte"));
        }
        while bytes.len() < limit && at < data.len() {
            let mut byte = data[at];
            if stream.text {
                if byte == CTRL_Z {
                    break;
                }
                if byte == b'\r' && data.get(at + 1) == Some(&b'\n') {
                    at += 1;
                    byte = b'\n';
                }
            }
            at += 1;
            bytes.push(byte);
        }
        // The position advances by the bytes read, from where it was: a read
        // past the end reads nothing and leaves it there, as POSIX leaves it.
        stream.position += (at - start) as u64;
        Ok(bytes)
    }

    /// Whether a descriptor number is open, for `_isatty`.
    pub(super) fn has_descriptor(&self, fd: u32) -> bool {
        self.descriptors.contains_key(&fd)
    }

    /// `ungetc`: one byte back onto the stream, the last pushed the first
    /// re-read. The end-of-file indicator clears, as the standard says it
    /// does. The CRT promises a single byte of pushback; this machine's
    /// streams keep stacking, which the standard allows an implementation
    /// to do.
    pub(super) fn ungetc_byte(&mut self, file: u64, byte: u8) -> Result<(), u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        if !stream.readable || !orient(stream, false) {
            return Err(EBADF);
        }
        stream.pushback.push(byte);
        stream.eof = false;
        Ok(())
    }

    /// The descriptor of a stream, for `_fileno`: the one it already has,
    /// or a new one that shares the stream's state, as the CRT's streams
    /// and descriptors share a file.
    pub(super) fn descriptor_of(&mut self, file: u64) -> Result<u32, u32> {
        if !self.open.contains_key(&file) {
            return Err(EBADF);
        }
        if let Some((&fd, _)) = self
            .descriptors
            .iter()
            .find(|(_, entry)| matches!(entry, Descriptor::Shared(shared) if *shared == file))
        {
            return Ok(fd);
        }
        self.insert_descriptor(Descriptor::Shared(file))
    }

    /// The lowest free descriptor from 3, or `EMFILE` when the table is as
    /// full as the CRT's default allows.
    /// Whether one more descriptor fits, so a failing open can refuse
    /// before it creates or truncates anything.
    fn has_descriptor_room(&self) -> bool {
        let mut fd = 3;
        while self.descriptors.contains_key(&fd) {
            fd += 1;
            if fd >= MAX_DESCRIPTORS {
                return false;
            }
        }
        true
    }

    fn insert_descriptor(&mut self, entry: Descriptor) -> Result<u32, u32> {
        let mut fd = 3;
        while self.descriptors.contains_key(&fd) {
            fd += 1;
            if fd >= MAX_DESCRIPTORS {
                return Err(EMFILE);
            }
        }
        self.descriptors.insert(fd, entry);
        Ok(fd)
    }

    /// The stream state a descriptor works on: its own for one `_open`
    /// made, or the stream's for one `_fileno` gave out.
    /// The descriptor a duplication chain ends at: itself for one that is
    /// no alias. The chain is short by construction — `_dup2` rewires — so
    /// the bound never acts in practice.
    fn descriptor_terminal(&self, fd: u32) -> Result<u32, u32> {
        let mut at = fd;
        for _ in 0..MAX_DESCRIPTORS {
            match self.descriptors.get(&at).ok_or(EBADF)? {
                Descriptor::Alias(next) => at = *next,
                _ => return Ok(at),
            }
        }
        Err(EBADF)
    }

    fn descriptor_stream(&mut self, fd: u32) -> Result<&mut FileStream, u32> {
        let fd = self.descriptor_terminal(fd)?;
        match self.descriptors.get(&fd).ok_or(EBADF)? {
            Descriptor::Shared(file) => {
                let file = *file;
                self.open.get_mut(&file).ok_or(EBADF)
            }
            Descriptor::Owned(_) => match self.descriptors.get_mut(&fd) {
                Some(Descriptor::Owned(stream)) => Ok(stream),
                _ => Err(EBADF),
            },
            _ => Err(EBADF),
        }
    }

    /// The path and length a `_fstat64` reports, of the file a descriptor
    /// has open. A removed file reports the length it would have had, as the
    /// descriptor still has it open: zero, the drive carries no content for
    /// it anymore.
    pub(super) fn descriptor_stat(&self, drive: &ObjectState, fd: u32) -> Result<u64, u32> {
        let fd = self.descriptor_terminal(fd)?;
        let stream = match self.descriptors.get(&fd).ok_or(EBADF)? {
            Descriptor::Shared(file) => self.open.get(file).ok_or(EBADF)?,
            Descriptor::Owned(stream) => stream,
            _ => return Err(EBADF),
        };
        let _ = stream;
        Ok(drive
            .drive_file(&stream.path)
            .map_or(0, |data| data.len() as u64))
    }

    /// `_fdopen` wraps a descriptor in a stream: the descriptor's state moves
    /// to the open-stream table and the descriptor becomes the stream's own,
    /// exactly the arrangement `_fileno` starts from. Closing the stream then
    /// takes the descriptor with it, as the CRT's `fclose` does. A descriptor
    /// of a standard stream cannot move: `EINVAL`, Xenora's choice.
    pub(super) fn fdopen_stream(&mut self, memory: &mut GuestMemory, fd: u32) -> Result<u64, u32> {
        let fd = self.descriptor_terminal(fd)?;
        match self.descriptors.get(&fd) {
            Some(Descriptor::Owned(_)) => {}
            _ => return Err(EINVAL),
        }
        if self.open.len() >= MAX_OPEN_STREAMS {
            return Err(EMFILE);
        }
        let file = memory
            .map_heap_allocation(PROCESS_HEAP, 8)
            .map_err(|_| ENOMEM)?;
        let Some(Descriptor::Owned(stream)) = self.descriptors.remove(&fd) else {
            memory.release_heap_allocation(PROCESS_HEAP, file);
            return Err(EINVAL);
        };
        self.open.insert(file, stream);
        self.descriptors.insert(fd, Descriptor::Shared(file));
        Ok(file)
    }

    /// The pipe identifier and whether it is the write end, for a descriptor
    /// of one — so the service layer routes the operation before the stream
    /// paths, which pipes are no part of.
    pub(super) fn pipe_end(&self, fd: u32) -> Result<Option<(u32, bool)>, u32> {
        let fd = self.descriptor_terminal(fd)?;
        Ok(match self.descriptors.get(&fd) {
            Some(Descriptor::PipeRead(id)) => Some((*id, false)),
            Some(Descriptor::PipeWrite(id)) => Some((*id, true)),
            _ => None,
        })
    }

    /// `_pipe`: the read end and the write end, sharing the one queue as
    /// the CRT's anonymous pipe shares it. The size and mode arguments do
    /// not act: the queue grows as the writes ask.
    pub(super) fn pipe_create(&mut self) -> Result<(u32, u32), u32> {
        let mut room = 0;
        for fd in 3..MAX_DESCRIPTORS {
            if !self.descriptors.contains_key(&fd) {
                room += 1;
                if room == 2 {
                    break;
                }
            }
        }
        if room < 2 {
            return Err(EMFILE);
        }
        self.pipe_counter += 1;
        let id = self.pipe_counter;
        self.pipes.insert(
            id,
            PipeState {
                buffer: std::collections::VecDeque::new(),
                write_open: true,
            },
        );
        let read = self.insert_descriptor(Descriptor::PipeRead(id))?;
        let write = self.insert_descriptor(Descriptor::PipeWrite(id))?;
        Ok((read, write))
    }

    /// Read end bytes: what the queue holds, up to the limit. The empty
    /// queue is end of file once the write end closed, and `Ok(None)` while
    /// it has not — a read there would block, which the hosted runtime does
    /// not model.
    pub(super) fn pipe_read(&mut self, id: u32, limit: usize) -> Result<Option<Vec<u8>>, u32> {
        let pipe = self.pipes.get_mut(&id).ok_or(EBADF)?;
        let mut bytes = Vec::new();
        while bytes.len() < limit {
            let Some(byte) = pipe.buffer.pop_front() else {
                break;
            };
            bytes.push(byte);
        }
        if bytes.is_empty() && pipe.write_open {
            return Ok(None);
        }
        Ok(Some(bytes))
    }

    /// Write end bytes: the whole write lands in the queue.
    pub(super) fn pipe_write(&mut self, id: u32, bytes: &[u8]) -> Result<usize, u32> {
        let pipe = self.pipes.get_mut(&id).ok_or(EBADF)?;
        pipe.buffer.extend(bytes.iter().copied());
        Ok(bytes.len())
    }

    /// `_dup`: the lowest free descriptor sharing the file position, as the
    /// CRT's duplication shares it.
    pub(super) fn descriptor_duplicate(&mut self, fd: u32) -> Result<u32, u32> {
        self.descriptor_terminal(fd)?;
        if !self.has_descriptor_room() {
            return Err(EMFILE);
        }
        self.insert_descriptor(Descriptor::Alias(fd))
    }

    /// `_dup2`: `target` becomes the duplication, closed first as the CRT
    /// closes it, unless it already is `fd`.
    pub(super) fn descriptor_duplicate2(
        &mut self,
        memory: &mut GuestMemory,
        drive: &mut ObjectState,
        fd: u32,
        target: u32,
    ) -> Result<u32, u32> {
        self.descriptor_terminal(fd)?;
        if target >= MAX_DESCRIPTORS {
            return Err(EBADF);
        }
        if target == fd {
            return Ok(target);
        }
        if self.descriptors.contains_key(&target) {
            self.descriptor_close(memory, drive, target);
        }
        self.descriptors.insert(target, Descriptor::Alias(fd));
        Ok(target)
    }

    /// `_open`. The flags are the `<fcntl.h>` values: the access mode in the
    /// low two bits, `_O_APPEND` 8, `_O_CREAT` 0x100, `_O_TRUNC` 0x200,
    /// `_O_EXCL` 0x400, `_O_TEXT` 0x4000, `_O_BINARY` 0x8000. The wide
    /// text modes stop the guest before this; the hint bits are accepted
    /// and ignored.
    pub(super) fn open_descriptor(
        &mut self,
        drive: &mut ObjectState,
        name: &[u8],
        flags: u32,
    ) -> Result<u32, u32> {
        let path = ObjectState::drive_path(name).ok_or(EINVAL)?;
        let (readable, writable) = match flags & 3 {
            0 => (true, false),
            1 => (false, true),
            2 => (true, true),
            _ => return Err(EINVAL),
        };
        let append = flags & 0x8 != 0;
        let create = flags & 0x100 != 0;
        let truncate = flags & 0x200 != 0;
        let exclusive = flags & 0x400 != 0;
        // Text unless binary, as `_fmode` starts as `_O_TEXT`.
        let text = flags & 0x8000 == 0;
        if !writable && truncate {
            return Err(EINVAL);
        }
        let create = match drive.drive_entry(&path) {
            DriveEntry::Directory => return Err(EACCES),
            DriveEntry::File => {
                if create && exclusive {
                    return Err(EEXIST);
                }
                false
            }
            DriveEntry::Missing { .. } if !create => return Err(ENOENT),
            DriveEntry::Missing { parent: false } => return Err(ENOENT),
            DriveEntry::Missing { parent: true } => true,
        };
        // A full table fails before anything is created or truncated: the
        // EMFILE of the CRT leaves the file as it was.
        if !self.has_descriptor_room() {
            return Err(EMFILE);
        }
        if create && !drive.drive_create(&path) {
            return Err(ENOSPC);
        }
        if truncate {
            drive.drive_truncate(&path);
        }
        let mut stream = FileStream {
            path,
            position: 0,
            readable,
            writable,
            append,
            text,
            temporary: flags & 0x40 != 0,
            eof: false,
            error: false,
            orientation: 0,
            pushback: Vec::new(),
        };
        if append && let Some(data) = drive.drive_file(&stream.path) {
            stream.position = data.len() as u64;
        }
        self.insert_descriptor(Descriptor::Owned(stream))
    }

    /// `_read`. The bytes, or the `errno` value. An empty read at the end
    /// is no error.
    pub(super) fn descriptor_read(
        &mut self,
        drive: &ObjectState,
        fd: u32,
        limit: usize,
    ) -> Result<Vec<u8>, u32> {
        // A shared descriptor reads through the stream, indicators included.
        if let Some(Descriptor::Shared(file)) = self.descriptors.get(&fd) {
            let file = *file;
            return self.read_bytes(drive, file, limit);
        }
        // A handle descriptor drains the kernel pipe it wraps.
        if let Some(Descriptor::Handle(handle)) = self.descriptors.get(&fd) {
            return handle_read(drive, *handle, limit);
        }
        let stream = self.descriptor_stream(fd)?;
        if !stream.readable {
            return Err(EBADF);
        }
        let position_path = stream.path.clone();
        let data = drive.drive_file(&position_path).ok_or(EBADF)?;
        let mut at = usize::try_from(stream.position).map_or(data.len(), |at| at.min(data.len()));
        let start = at;
        let mut bytes = Vec::new();
        while bytes.len() < limit && at < data.len() {
            let mut byte = data[at];
            if stream.text {
                if byte == CTRL_Z {
                    break;
                }
                if byte == b'\r' && data.get(at + 1) == Some(&b'\n') {
                    at += 1;
                    byte = b'\n';
                }
            }
            at += 1;
            bytes.push(byte);
        }
        // The position advances by the bytes read, from where it was: a read
        // past the end reads nothing and leaves it there, as POSIX leaves it.
        stream.position += (at - start) as u64;
        Ok(bytes)
    }

    /// `_write`. Nothing is written when the drive has no room.
    pub(super) fn descriptor_write(
        &mut self,
        drive: &mut ObjectState,
        fd: u32,
        bytes: &[u8],
    ) -> Result<(), u32> {
        if let Some(Descriptor::Shared(file)) = self.descriptors.get(&fd) {
            let file = *file;
            return self.write(drive, file, bytes);
        }
        if let Some(Descriptor::Handle(handle)) = self.descriptors.get(&fd) {
            return handle_write(drive, *handle, bytes);
        }
        let path = self.descriptor_stream(fd)?.path.clone();
        let stream = self.descriptor_stream(fd)?;
        if !stream.writable {
            return Err(EBADF);
        }
        let mut stored: Vec<u8> = Vec::new();
        if stream.text {
            for &byte in bytes {
                if byte == b'\n' {
                    stored.push(b'\r');
                }
                stored.push(byte);
            }
        } else {
            stored.extend_from_slice(bytes);
        }
        let position = if stream.append {
            drive.drive_file(&path).map(|data| data.len() as u64)
        } else {
            Some(stream.position)
        };
        let Some(position) = position else {
            return Err(EBADF);
        };
        // A zero-byte write touches nothing: it must not extend the file to
        // the cursor, as a write of zero bytes has no effect.
        if !stored.is_empty() {
            let written = drive.drive_write(&path, position, &stored).ok_or(EBADF)?;
            if !written {
                return Err(ENOSPC);
            }
        }
        stream.position = position + stored.len() as u64;
        Ok(())
    }

    /// `_lseek`. A position before the start, one that no 32-bit `long` can
    /// hold, or an unknown origin is -1 with `EINVAL`. A seek clears the
    /// end-of-file indicator of a shared stream, as `_fseeki64` does.
    pub(super) fn descriptor_seek(
        &mut self,
        drive: &ObjectState,
        fd: u32,
        offset: i32,
        origin: u32,
    ) -> Result<i64, u32> {
        if origin > 2 {
            return Err(EINVAL);
        }
        // A shared descriptor seeks the stream.
        if let Some(Descriptor::Shared(file)) = self.descriptors.get(&fd) {
            let file = *file;
            self.seek(drive, file, i64::from(offset), origin)?;
            return self.tell(file).map(|position| position as i64);
        }
        if matches!(self.descriptors.get(&fd), Some(Descriptor::Handle(_))) {
            // A pipe has no position to move, as the CRT answers for one.
            return Err(EINVAL);
        }
        let path = self.descriptor_stream(fd)?.path.clone();
        let stream = self.descriptor_stream(fd)?;
        let base = match origin {
            0 => 0,
            1 => stream.position,
            2 => drive.drive_file(&path).ok_or(EBADF)?.len() as u64,
            _ => return Err(EINVAL),
        };
        let target = i128::from(base) + i128::from(offset);
        if !(0..=i128::from(i32::MAX)).contains(&target) {
            return Err(EINVAL);
        }
        drive.drive_file(&stream.path).ok_or(EBADF)?;
        stream.position = target as u64;
        Ok(target as i64)
    }

    /// `_lseeki64`. What `_lseek` does, with a 64-bit offset and positions
    /// no 32-bit `long` need hold: the file model's positions are the
    /// drive's bytes, so the limit is the signed range itself.
    pub(super) fn descriptor_seek64(
        &mut self,
        drive: &ObjectState,
        fd: u32,
        offset: i64,
        origin: u32,
    ) -> Result<i64, u32> {
        if origin > 2 {
            return Err(EINVAL);
        }
        if let Some(Descriptor::Shared(file)) = self.descriptors.get(&fd) {
            let file = *file;
            self.seek(drive, file, offset, origin)?;
            return self.tell(file).map(|position| position as i64);
        }
        if matches!(self.descriptors.get(&fd), Some(Descriptor::Handle(_))) {
            // A pipe has no position to move, as the CRT answers for one.
            return Err(EINVAL);
        }
        let path = self.descriptor_stream(fd)?.path.clone();
        let stream = self.descriptor_stream(fd)?;
        let base = match origin {
            0 => 0,
            1 => stream.position,
            2 => drive.drive_file(&path).ok_or(EBADF)?.len() as u64,
            _ => return Err(EINVAL),
        };
        let target = i128::from(base) + i128::from(offset);
        if !(0..=i128::from(i64::MAX)).contains(&target) {
            return Err(EINVAL);
        }
        drive.drive_file(&stream.path).ok_or(EBADF)?;
        stream.position = target as u64;
        Ok(target as i64)
    }

    /// `_setmode`: the text or binary translation of the descriptor's
    /// stream, answering the previous one. The modes are the `<fcntl.h>`
    /// values `_O_TEXT` 0x4000 and `_O_BINARY` 0x8000; anything else is
    /// `EINVAL`, the wide forms included — the hosted runtime's streams are
    /// byte-oriented, as `_open` already refuses them.
    pub(super) fn descriptor_set_mode(&mut self, fd: u32, mode: u32) -> Result<u32, u32> {
        if mode != 0x4000 && mode != 0x8000 {
            return Err(EINVAL);
        }
        let stream = self.descriptor_stream(fd)?;
        let previous = if stream.text { 0x4000 } else { 0x8000 };
        stream.text = mode == 0x4000;
        Ok(previous)
    }

    /// `_close`. Closing a shared descriptor closes the underlying file, so
    /// the stream it came from fails afterwards, as it does in the CRT.
    /// `_filbuf`: one byte read off the stream, as the legacy macro
    /// support reads it — the streams here are unbuffered, so the byte is
    /// the next one, or nothing at the end.
    pub(super) fn read_one(&mut self, drive: &ObjectState, file: u64) -> Result<Option<u8>, u32> {
        let bytes = self.read_bytes(drive, file, 1)?;
        Ok(bytes.first().copied())
    }

    /// `_flsbuf`: one byte written onto the stream. `EOF` for a stream that
    /// cannot take it, as the write failure answers.
    pub(super) fn write_one(
        &mut self,
        drive: &mut ObjectState,
        file: u64,
        byte: u8,
    ) -> Result<(), u32> {
        self.write(drive, file, &[byte])
    }

    /// `_open_osfhandle`: the descriptor wrapping a kernel object handle,
    /// which the CRT takes ownership of. `EBADF` for a handle that is no
    /// pipe end — the only kernel object this interop knows so far.
    pub(super) fn descriptor_open_handle(
        &mut self,
        drive: &ObjectState,
        handle: u64,
    ) -> Result<u32, u32> {
        if !matches!(
            drive.handles.get(&handle),
            Some(GuestObject::Pipe { .. } | GuestObject::File(_))
        ) {
            return Err(EBADF);
        }
        self.insert_descriptor(Descriptor::Handle(handle))
    }

    /// `_get_osfhandle`: the kernel handle a descriptor wraps. The CRT owns
    /// one for the `_open_osfhandle` descriptors only; its own streams and
    /// files have no kernel object behind them, so the answer is -1 with
    /// `EBADF` there, as the hosted runtime answers for a bad descriptor.
    pub(super) fn descriptor_handle(&self, fd: u32) -> Result<u64, u32> {
        match self.descriptors.get(&fd) {
            Some(Descriptor::Handle(handle)) => Ok(*handle),
            _ => Err(EBADF),
        }
    }

    pub(super) fn descriptor_close(
        &mut self,
        memory: &mut GuestMemory,
        drive: &mut ObjectState,
        fd: u32,
    ) -> bool {
        match self.descriptors.remove(&fd) {
            Some(Descriptor::Owned(stream)) => {
                // A duplication of this descriptor becomes the owner, as the
                // file stays open while any descriptor of it does; the rest
                // of the chain rewires to it.
                if let Some((&heir, _)) = self
                    .descriptors
                    .iter()
                    .find(|(_, entry)| matches!(entry, Descriptor::Alias(of) if *of == fd))
                {
                    self.descriptors.insert(heir, Descriptor::Owned(stream));
                    let rewire: Vec<u32> = self
                        .descriptors
                        .iter()
                        .filter_map(|(&other, entry)| {
                            (matches!(entry, Descriptor::Alias(of) if *of == fd) && other != heir)
                                .then_some(other)
                        })
                        .collect();
                    for other in rewire {
                        self.descriptors.insert(other, Descriptor::Alias(heir));
                    }
                } else if stream.temporary {
                    drive.drive_remove(&stream.path);
                }
                true
            }
            Some(Descriptor::Handle(handle)) => {
                // The CRT owns the handle now: closing the descriptor closes
                // it, and a write end closing marks the pipe's end of stream,
                // as CloseHandle does.
                if let Some(GuestObject::Pipe { state, write }) = drive.handles.get(&handle)
                    && *write
                {
                    state.lock().expect("pipe state").write_open = false;
                }
                drive.handles.remove(&handle);
                true
            }
            Some(Descriptor::Alias(_)) => true,
            Some(Descriptor::PipeRead(id)) => {
                // The pipe itself goes once nothing references it: no other
                // descriptor of it, and the write end closed.
                if !self
                    .descriptors
                    .values()
                    .any(|entry| matches!(entry, Descriptor::PipeWrite(other) | Descriptor::PipeRead(other) if *other == id))
                {
                    self.pipes.remove(&id);
                }
                true
            }
            Some(Descriptor::PipeWrite(id)) => {
                if let Some(pipe) = self.pipes.get_mut(&id) {
                    pipe.write_open = false;
                }
                if !self
                    .descriptors
                    .values()
                    .any(|entry| matches!(entry, Descriptor::PipeWrite(other) | Descriptor::PipeRead(other) if *other == id))
                {
                    self.pipes.remove(&id);
                }
                true
            }
            Some(Descriptor::Shared(file)) => {
                self.open.remove(&file);
                memory.release_heap_allocation(PROCESS_HEAP, file);
                true
            }
            None => false,
        }
    }

    /// Drop a stream's shared descriptors when the stream itself is closed,
    /// so a later `_fileno` of a recycled stream address cannot pick them up.
    pub(super) fn drop_descriptors_of(&mut self, file: u64) {
        self.descriptors
            .retain(|_, entry| !matches!(entry, Descriptor::Shared(shared) if *shared == file));
    }

    /// The end-of-file indicator of a stream.
    pub(super) fn end_of_file(&self, file: u64) -> bool {
        self.open.get(&file).is_some_and(|stream| stream.eof)
    }

    /// Set the end-of-file indicator, as the read functions do when they
    /// meet the end: `fread` and `fgetc` share `read_bytes`, which is too
    /// low a level to know what the caller asked for.
    pub(super) fn note_end_of_file(&mut self, file: u64) {
        if let Some(stream) = self.open.get_mut(&file) {
            stream.eof = true;
        }
    }

    /// The error indicator of a stream, set when a write through it fails.
    pub(super) fn error_flag(&self, file: u64) -> bool {
        self.open.get(&file).is_some_and(|stream| stream.error)
    }

    /// `clearerr`: both indicators off. False when `file` is not open.
    pub(super) fn clear_errors(&mut self, file: u64) -> bool {
        let Some(stream) = self.open.get_mut(&file) else {
            return false;
        };
        stream.eof = false;
        stream.error = false;
        true
    }

    pub(super) fn read_line(
        &mut self,
        drive: &ObjectState,
        file: u64,
        limit: usize,
    ) -> Result<Option<Vec<u8>>, u32> {
        let stream = self.open.get_mut(&file).ok_or(EBADF)?;
        if !stream.readable {
            return Err(EBADF);
        }
        if !orient(stream, false) {
            return Err(EBADF);
        }
        let data = drive.drive_file(&stream.path).ok_or(EBADF)?;
        let mut at = usize::try_from(stream.position).map_or(data.len(), |at| at.min(data.len()));
        let ended = at == data.len() || (stream.text && data[at] == CTRL_Z);
        if limit == 0 {
            // No room to read a character: an empty line unless the stream
            // has ended.
            return Ok((!ended).then(Vec::new));
        }
        let mut line = Vec::new();
        while line.len() < limit && at < data.len() {
            let mut byte = data[at];
            if stream.text {
                if byte == CTRL_Z {
                    break;
                }
                if byte == b'\r' && data.get(at + 1) == Some(&b'\n') {
                    at += 1;
                    byte = b'\n';
                }
            }
            at += 1;
            line.push(byte);
            if byte == b'\n' {
                break;
            }
        }
        // The position advances by the bytes read, from where it was: a read
        // past the end reads nothing and leaves it there, as POSIX leaves it.
        stream.position = at as u64;
        if line.is_empty() {
            // End of file: the indicator is set, as ISO C says of the read
            // functions.
            stream.eof = true;
            return Ok(None);
        }
        Ok(Some(line))
    }
}

/// The pipe cap of the kernel pipes, so a CRT write plays by the same
/// rules as a `WriteFile` one.
const PIPE_CAP: usize = 16 * 1024 * 1024;

/// The access bits of a kernel file object, as CreateFile takes them.
const FILE_READ: u32 = 0x8000_0000;
const FILE_WRITE: u32 = 0x4000_0000;

/// `_read` of a handle descriptor: the pipe drains, the oldest byte first.
/// An empty pipe with its write end still open cannot block — the guest is
/// the only thread that could fill it — so the read answers what is there,
/// and the end-of-file emptiness is honest only once the write end closes.
fn handle_read(drive: &super::ObjectState, handle: u64, limit: usize) -> Result<Vec<u8>, u32> {
    match drive.handles.get(&handle) {
        Some(GuestObject::Pipe { state, write }) => {
            if *write {
                return Err(EACCES);
            }
            let mut pipe = state.lock().expect("pipe state");
            let mut bytes = Vec::new();
            while bytes.len() < limit {
                let Some(byte) = pipe.buffer.pop_front() else {
                    break;
                };
                bytes.push(byte);
            }
            Ok(bytes)
        }
        Some(GuestObject::File(file)) => {
            use std::sync::atomic::Ordering;
            if file.access & FILE_READ == 0 {
                return Err(EACCES);
            }
            let data = drive.drive_file(file.path()).ok_or(EBADF)?;
            let position = file.position().load(Ordering::Relaxed);
            let at = usize::try_from(position).map_or(data.len(), |at| at.min(data.len()));
            let taken = limit.min(data.len() - at);
            let bytes = data[at..at + taken].to_vec();
            file.position()
                .store(position + taken as u64, Ordering::Relaxed);
            Ok(bytes)
        }
        _ => Err(EBADF),
    }
}

/// `_write` of one: the pipe fills, within the same cap the kernel applies.
fn handle_write(drive: &mut super::ObjectState, handle: u64, bytes: &[u8]) -> Result<(), u32> {
    match drive.handles.get(&handle) {
        Some(GuestObject::Pipe { state, write }) => {
            if !*write {
                return Err(EACCES);
            }
            let mut pipe = state.lock().expect("pipe state");
            if pipe.buffer.len() + bytes.len() > PIPE_CAP {
                return Err(ENOSPC);
            }
            pipe.buffer.extend(bytes);
            Ok(())
        }
        Some(GuestObject::File(_)) => {
            use std::sync::atomic::Ordering;
            // The object lookup ends before the drive write, which borrows
            // the state again: the path and the position cell are the file's.
            let Some(GuestObject::File(file)) = drive.handles.get(&handle) else {
                return Err(EBADF);
            };
            if file.access & FILE_WRITE == 0 {
                return Err(EACCES);
            }
            let path = file.path().to_owned();
            let cell = std::sync::Arc::clone(file.position());
            let position = cell.load(Ordering::Relaxed);
            if !bytes.is_empty() {
                let written = drive.drive_write(&path, position, bytes).ok_or(EBADF)?;
                if !written {
                    return Err(ENOSPC);
                }
            }
            cell.store(position + bytes.len() as u64, Ordering::Relaxed);
            Ok(())
        }
        _ => Err(EBADF),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_mode;

    /// The six access types of the `fopen` page, each alone and with `t` or
    /// `b`, and what the page says of each.
    #[test]
    fn modes_follow_the_fopen_page() {
        // (mode, readable, writable, append, must exist, truncates)
        for (mode, readable, writable, append, existing, truncate) in [
            ("r", true, false, false, true, false),
            ("w", false, true, false, false, true),
            ("a", false, true, true, false, false),
            ("r+", true, true, false, true, false),
            ("w+", true, true, false, false, true),
            ("a+", true, true, true, false, false),
        ] {
            for (suffix, text) in [("", true), ("t", true), ("b", false)] {
                let mut spellings = vec![format!("{mode}{suffix}")];
                if mode.len() == 2 && !suffix.is_empty() {
                    // "rb+" is the same mode as "r+b".
                    spellings.push(format!("{}{suffix}+", &mode[..1]));
                }
                for spelling in spellings {
                    let parsed = parse_mode(spelling.as_bytes())
                        .unwrap_or_else(|| panic!("{spelling} is a mode"));
                    assert_eq!(
                        (
                            parsed.readable,
                            parsed.writable,
                            parsed.append,
                            parsed.existing,
                            parsed.truncate,
                            parsed.text
                        ),
                        (readable, writable, append, existing, truncate, text),
                        "{spelling}"
                    );
                }
            }
        }
        // "If t or b is prefixed to the argument, the function fails." The
        // other mode characters the page lists are not opened here, and
        // neither is anything that is not a mode.
        for mode in [
            "",
            "t",
            "b",
            "tr",
            "br",
            "+r",
            "x",
            "rw",
            "wr",
            "ra",
            "rbt",
            "rtb",
            "r++",
            "rbb",
            "wtt",
            "wx",
            "w+x",
            "rc",
            "rn",
            "rN",
            "rS",
            "rR",
            "wT",
            "wD",
            "r,ccs=UTF-8",
            "R",
            "W",
            "r ",
            " r",
            "rb+t",
        ] {
            assert!(parse_mode(mode.as_bytes()).is_none(), "{mode:?}");
        }
    }
}
