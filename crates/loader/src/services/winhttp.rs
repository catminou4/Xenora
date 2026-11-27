// SPDX-License-Identifier: GPL-3.0-or-later

//! A bounded synchronous subset of the publicly documented WinHTTP API.
//!
//! The object layer models sessions, connections and requests. The transport
//! layer performs one cleartext HTTP/1.1 request/response exchange per request
//! with `Connection: close`. TLS, chunked coding, asynchronous callbacks,
//! cancellation, redirection, authentication, proxying, streaming writes and
//! connection reuse remain separate unsupported slices.

use super::{ERROR_INVALID_HANDLE, ERROR_INVALID_PARAMETER, ERROR_NOT_SUPPORTED, ServicesState};
use crate::imports::{ImportRegistry, ServiceStatus};
use crate::memory::GuestMemory;
use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MODULE: &str = "winhttp.dll";
const MAX_HANDLES: usize = 256;
const MAX_WIDE_UNITS: usize = 512;
const MAX_ACCEPT_TYPES: usize = 32;
const ERROR_OUTOFMEMORY: u32 = 14;
const ERROR_WINHTTP_INVALID_URL: u32 = 12005;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const ERROR_WINHTTP_TIMEOUT: u32 = 12002;
const ERROR_WINHTTP_NAME_NOT_RESOLVED: u32 = 12007;
const ERROR_WINHTTP_INCORRECT_HANDLE_TYPE: u32 = 12018;
const ERROR_WINHTTP_INCORRECT_HANDLE_STATE: u32 = 12019;
const ERROR_WINHTTP_CANNOT_CONNECT: u32 = 12029;
const ERROR_WINHTTP_CONNECTION_ERROR: u32 = 12030;
const ERROR_WINHTTP_INVALID_SERVER_RESPONSE: u32 = 12152;
const ERROR_WINHTTP_HEADER_SIZE_OVERFLOW: u32 = 12182;
const WINHTTP_QUERY_CONTENT_TYPE: u32 = 1;
const WINHTTP_QUERY_CONTENT_LENGTH: u32 = 5;
const WINHTTP_QUERY_STATUS_CODE: u32 = 19;
const WINHTTP_QUERY_STATUS_TEXT: u32 = 20;
const WINHTTP_QUERY_SERVER: u32 = 37;
const WINHTTP_QUERY_FLAG_NUMBER: u32 = 0x2000_0000;
const ERROR_WINHTTP_HEADER_NOT_FOUND: u32 = 12150;
const MAX_ADDITIONAL_HEADER_UNITS: usize = 4096;
const MAX_HEADER_LINE_BYTES: usize = 8192;
const MAX_RESPONSE_HEAD_BYTES: usize = 64 * 1024;
const MAX_READ_CHUNK_BYTES: usize = 64 * 1024;
const MAX_OPTIONAL_BODY_BYTES: u32 = 16 * 1024 * 1024;

const WINHTTP_ACCESS_TYPE_DEFAULT_PROXY: u32 = 0;
const WINHTTP_ACCESS_TYPE_NO_PROXY: u32 = 1;
const WINHTTP_FLAG_ASYNC: u32 = 0x1000_0000;
const WINHTTP_FLAG_SECURE: u32 = 0x0080_0000;
const DOCUMENTED_REQUEST_FLAGS: u32 = 0x0000_01cc | WINHTTP_FLAG_SECURE;

enum HandleKind {
    Session {
        agent: Option<String>,
    },
    Connection {
        agent: Option<String>,
        server: String,
        port: u32,
    },
    Request {
        agent: Option<String>,
        server: String,
        port: u32,
        verb: String,
        object_name: String,
        version: String,
        referrer: Option<String>,
        accept_types: Vec<String>,
        secure: bool,
        transport: Arc<Mutex<TransportState>>,
    },
}

struct RequestSnapshot {
    agent: Option<String>,
    server: String,
    port: u32,
    verb: String,
    object_name: String,
    version: String,
    referrer: Option<String>,
    accept_types: Vec<String>,
    secure: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TransportPhase {
    New,
    Busy,
    Sent,
    Ready,
    Failed,
}

#[derive(Clone, Copy)]
enum BodyKind {
    Empty,
    Fixed(u64),
    UntilClose { eof: bool },
    Chunked,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChunkedPosition {
    SizeLine,
    Data,
    DataCrLf,
    Trailers,
    Eof,
}

struct ChunkedState {
    position: ChunkedPosition,
    chunk_remaining: u64,
    queued: VecDeque<u8>,
    trailer_bytes: usize,
}

impl ChunkedState {
    fn new() -> Self {
        Self {
            position: ChunkedPosition::SizeLine,
            chunk_remaining: 0,
            queued: VecDeque::new(),
            trailer_bytes: 0,
        }
    }
}

struct ResponseHead {
    status: u32,
    status_text: String,
    headers: Vec<(String, String)>,
}

struct TransportState {
    phase: TransportPhase,
    verb: String,
    stream: Option<BufReader<TcpStream>>,
    response: Option<ResponseHead>,
    body: BodyKind,
    chunked: ChunkedState,
}

impl TransportState {
    fn new() -> Self {
        Self {
            phase: TransportPhase::New,
            verb: String::new(),
            stream: None,
            response: None,
            body: BodyKind::Empty,
            chunked: ChunkedState::new(),
        }
    }
}

/// The WinHTTP handle table. Handles are opaque guest values; the table owns
/// immutable snapshots of every string, so later guest-memory writes cannot
/// mutate an existing object.
#[derive(Default)]
pub(super) struct WinHttpHandles {
    next: u64,
    handles: BTreeMap<u64, HandleKind>,
}

impl WinHttpHandles {
    pub(super) fn open_count(&self) -> usize {
        self.handles.len()
    }

    pub(super) fn close_all(&mut self) {
        self.handles.clear();
    }

    fn insert(&mut self, kind: HandleKind) -> Result<u64, u32> {
        if self.handles.len() >= MAX_HANDLES {
            return Err(ERROR_OUTOFMEMORY);
        }
        self.next = self
            .next
            .checked_add(1)
            .filter(|next| *next < (1 << 32))
            .ok_or(ERROR_OUTOFMEMORY)?;
        let handle = 0x5748_0000_0000_0000 | self.next;
        self.handles.insert(handle, kind);
        Ok(handle)
    }

    fn kind(&self, handle: u64) -> Option<&HandleKind> {
        self.handles.get(&handle)
    }

    fn close(&mut self, handle: u64) -> bool {
        self.handles.remove(&handle).is_some()
    }
}

fn read_wide_string(memory: &GuestMemory, address: u64, limit: usize) -> Result<String, u32> {
    if address == 0 {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let mut units = Vec::new();
    for index in 0..limit {
        let unit_address = address
            .checked_add(2 * index as u64)
            .ok_or(ERROR_INVALID_PARAMETER)?;
        let unit = memory
            .read_u16(unit_address)
            .map_err(|_| ERROR_INVALID_PARAMETER)?;
        if unit == 0 {
            return Ok(String::from_utf16_lossy(&units));
        }
        units.push(unit);
    }
    Err(ERROR_INVALID_PARAMETER)
}

fn read_optional_wide_string(memory: &GuestMemory, address: u64) -> Result<Option<String>, u32> {
    if address == 0 {
        Ok(None)
    } else {
        read_wide_string(memory, address, MAX_WIDE_UNITS).map(Some)
    }
}

fn valid_token(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| (0x21..=0x7e).contains(&byte) && byte != b' ' && byte != b'\t')
}

fn valid_server_name(text: &str) -> bool {
    !text.is_empty()
        && text.is_ascii()
        && text.bytes().all(|byte| byte > 0x20 && byte != 0x7f)
        && !text
            .bytes()
            .any(|byte| matches!(byte, b'/' | b'\\' | b'?' | b'#'))
}

fn valid_header_name(text: &str) -> bool {
    !text.is_empty()
        && text.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn valid_header_value(text: &str) -> bool {
    text.bytes()
        .all(|byte| byte == b'\t' || (0x20..=0x7e).contains(&byte))
}

fn parse_additional_headers(text: &str) -> Result<Vec<(String, String)>, u32> {
    let mut headers = Vec::new();
    if text.is_empty() {
        return Ok(headers);
    }
    if !text.ends_with("\r\n") {
        return Err(ERROR_INVALID_PARAMETER);
    }
    for line in text.split_terminator("\r\n") {
        if line.is_empty() || line.len() > MAX_HEADER_LINE_BYTES {
            return Err(ERROR_INVALID_PARAMETER);
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(ERROR_INVALID_PARAMETER);
        };
        let value = value.trim_matches(|byte| byte == ' ' || byte == '\t');
        if !valid_header_name(name) || !valid_header_value(value) {
            return Err(ERROR_INVALID_PARAMETER);
        }
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "host" | "content-length" | "connection" | "transfer-encoding"
        ) {
            return Err(ERROR_NOT_SUPPORTED);
        }
        headers.push((name.to_owned(), value.to_owned()));
    }
    Ok(headers)
}

fn read_exact_wide_string(memory: &GuestMemory, address: u64, units: usize) -> Result<String, u32> {
    if address == 0 || units > MAX_ADDITIONAL_HEADER_UNITS {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let mut wide = Vec::with_capacity(units);
    for index in 0..units {
        let unit = memory
            .read_u16(address + 2 * index as u64)
            .map_err(|_| ERROR_INVALID_PARAMETER)?;
        wide.push(unit);
    }
    Ok(String::from_utf16_lossy(&wide))
}

fn read_additional_headers(
    memory: &GuestMemory,
    address: u64,
    units: u32,
) -> Result<Vec<(String, String)>, u32> {
    if address == 0 {
        return if units == 0 {
            Ok(Vec::new())
        } else {
            Err(ERROR_INVALID_PARAMETER)
        };
    }
    let text = if units == u32::MAX {
        read_wide_string(memory, address, MAX_ADDITIONAL_HEADER_UNITS)?
    } else {
        read_exact_wide_string(memory, address, units as usize)?
    };
    if !text.is_ascii() {
        return Err(ERROR_INVALID_PARAMETER);
    }
    parse_additional_headers(&text)
}

fn request_path(object_name: &str) -> Result<String, u32> {
    if object_name.is_empty() {
        return Ok("/".to_owned());
    }
    if object_name.starts_with('/')
        && object_name.is_ascii()
        && object_name.bytes().all(|byte| byte > 0x20 && byte != 0x7f)
    {
        Ok(object_name.to_owned())
    } else {
        Err(ERROR_WINHTTP_INVALID_URL)
    }
}

fn effective_port(port: u32, secure: bool) -> u16 {
    if port != 0 {
        port as u16
    } else if secure {
        443
    } else {
        80
    }
}

fn build_request_wire(
    snapshot: &RequestSnapshot,
    additional_headers: &[(String, String)],
    body: &[u8],
    total_length: u32,
) -> Result<Vec<u8>, u32> {
    let path = request_path(&snapshot.object_name)?;
    let port = effective_port(snapshot.port, snapshot.secure);
    let default_port = if snapshot.secure { 443 } else { 80 };
    let host = if port == default_port {
        snapshot.server.clone()
    } else {
        format!("{}:{}", snapshot.server, port)
    };
    let mut request = format!(
        "{} {} {}\r\nHost: {}\r\n",
        snapshot.verb, path, snapshot.version, host
    );
    if let Some(agent) = &snapshot.agent
        && !agent.is_empty()
    {
        if !valid_header_value(agent) {
            return Err(ERROR_INVALID_PARAMETER);
        }
        request.push_str("User-Agent: ");
        request.push_str(agent);
        request.push_str("\r\n");
    }
    if !snapshot.accept_types.is_empty() {
        if snapshot
            .accept_types
            .iter()
            .any(|accept| !valid_header_value(accept))
        {
            return Err(ERROR_INVALID_PARAMETER);
        }
        request.push_str("Accept: ");
        request.push_str(&snapshot.accept_types.join(", "));
        request.push_str("\r\n");
    }
    if let Some(referrer) = &snapshot.referrer {
        if !valid_header_value(referrer) {
            return Err(ERROR_INVALID_PARAMETER);
        }
        request.push_str("Referer: ");
        request.push_str(referrer);
        request.push_str("\r\n");
    }
    for (name, value) in additional_headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    if total_length != 0 {
        request.push_str("Content-Length: ");
        request.push_str(&total_length.to_string());
        request.push_str("\r\n");
    }
    request.push_str("Connection: close\r\n\r\n");
    let mut wire = request.into_bytes();
    wire.extend_from_slice(body);
    Ok(wire)
}

fn request_parts(
    handles: &WinHttpHandles,
    handle: u64,
) -> Result<(RequestSnapshot, Arc<Mutex<TransportState>>), u32> {
    match handles.kind(handle) {
        Some(HandleKind::Request {
            agent,
            server,
            port,
            verb,
            object_name,
            version,
            referrer,
            accept_types,
            secure,
            transport,
            ..
        }) => Ok((
            RequestSnapshot {
                agent: agent.clone(),
                server: server.clone(),
                port: *port,
                verb: verb.clone(),
                object_name: object_name.clone(),
                version: version.clone(),
                referrer: referrer.clone(),
                accept_types: accept_types.clone(),
                secure: *secure,
            },
            Arc::clone(transport),
        )),
        Some(_) => Err(ERROR_WINHTTP_INCORRECT_HANDLE_TYPE),
        None => Err(ERROR_INVALID_HANDLE),
    }
}

fn request_transport(
    handles: &WinHttpHandles,
    handle: u64,
) -> Result<Arc<Mutex<TransportState>>, u32> {
    match handles.kind(handle) {
        Some(HandleKind::Request { transport, .. }) => Ok(Arc::clone(transport)),
        Some(_) => Err(ERROR_WINHTTP_INCORRECT_HANDLE_TYPE),
        None => Err(ERROR_INVALID_HANDLE),
    }
}

fn connect_tcp(server: &str, port: u16) -> Result<TcpStream, u32> {
    let addresses = (server, port)
        .to_socket_addrs()
        .map_err(|_| ERROR_WINHTTP_NAME_NOT_RESOLVED)?;
    let mut saw_timeout = false;
    let mut saw_address = false;
    for address in addresses {
        saw_address = true;
        match TcpStream::connect_timeout(&address, Duration::from_secs(60)) {
            Ok(stream) => {
                stream
                    .set_read_timeout(Some(Duration::from_secs(30)))
                    .map_err(io_winhttp_error)?;
                stream
                    .set_write_timeout(Some(Duration::from_secs(30)))
                    .map_err(io_winhttp_error)?;
                return Ok(stream);
            }
            Err(error) => {
                saw_timeout |= matches!(
                    error.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                );
            }
        }
    }
    if !saw_address {
        Err(ERROR_WINHTTP_NAME_NOT_RESOLVED)
    } else if saw_timeout {
        Err(ERROR_WINHTTP_TIMEOUT)
    } else {
        Err(ERROR_WINHTTP_CANNOT_CONNECT)
    }
}

fn io_winhttp_error(error: std::io::Error) -> u32 {
    if matches!(
        error.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    ) {
        ERROR_WINHTTP_TIMEOUT
    } else {
        ERROR_WINHTTP_CONNECTION_ERROR
    }
}

fn fail_transport(transport: &Mutex<TransportState>) {
    transport.lock().expect("WinHTTP transport").phase = TransportPhase::Failed;
}

fn read_crlf_line(reader: &mut BufReader<TcpStream>) -> Result<Vec<u8>, u32> {
    let mut line = Vec::new();
    loop {
        if line.len() >= MAX_HEADER_LINE_BYTES {
            return Err(ERROR_WINHTTP_HEADER_SIZE_OVERFLOW);
        }
        let mut byte = [0];
        reader.read_exact(&mut byte).map_err(io_winhttp_error)?;
        if byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
    }
    if line.pop() != Some(b'\r') {
        return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
    }
    Ok(line)
}

fn parse_one_response_head(reader: &mut BufReader<TcpStream>) -> Result<ResponseHead, u32> {
    let status_line = read_crlf_line(reader)?;
    let mut total = status_line.len() + 2;
    let status_text =
        String::from_utf8(status_line).map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
    let mut parts = status_text.splitn(3, ' ');
    let version = parts.next().ok_or(ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
    let status = parts.next().ok_or(ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
    let status_text = parts.next().unwrap_or("").to_owned();
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        || status.len() != 3
        || !status.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
    }
    let status = status
        .parse::<u32>()
        .map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
    let mut headers = Vec::new();
    loop {
        let line = read_crlf_line(reader)?;
        total += line.len() + 2;
        if total > MAX_RESPONSE_HEAD_BYTES {
            return Err(ERROR_WINHTTP_HEADER_SIZE_OVERFLOW);
        }
        if line.is_empty() {
            break;
        }
        let line = String::from_utf8(line).map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
        let Some((name, value)) = line.split_once(':') else {
            return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
        };
        let value = value.trim_matches(|byte| byte == ' ' || byte == '\t');
        if !valid_header_name(name) || !valid_header_value(value) {
            return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
        }
        headers.push((name.to_owned(), value.to_owned()));
    }
    Ok(ResponseHead {
        status,
        status_text,
        headers,
    })
}

fn parse_response(
    reader: &mut BufReader<TcpStream>,
    verb: &str,
) -> Result<(ResponseHead, BodyKind), u32> {
    let head = loop {
        let head = parse_one_response_head(reader)?;
        if !(100..200).contains(&head.status) || head.status == 101 {
            break head;
        }
    };
    let no_body = verb.eq_ignore_ascii_case("HEAD")
        || (100..200).contains(&head.status)
        || head.status == 204
        || head.status == 304;
    let body = if no_body {
        BodyKind::Empty
    } else if head
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("transfer-encoding"))
    {
        let transfer_encodings = head
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("transfer-encoding"))
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        if transfer_encodings.len() != 1 {
            return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
        }
        if !transfer_encodings[0].eq_ignore_ascii_case("chunked") {
            return Err(ERROR_NOT_SUPPORTED);
        }
        BodyKind::Chunked
    } else {
        let mut length = None;
        for (name, value) in &head.headers {
            if name.eq_ignore_ascii_case("content-length") {
                let parsed = value
                    .parse::<u64>()
                    .map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
                if let Some(previous) = length {
                    if previous != parsed {
                        return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
                    }
                } else {
                    length = Some(parsed);
                }
            }
        }
        if let Some(length) = length {
            BodyKind::Fixed(length)
        } else {
            BodyKind::UntilClose { eof: false }
        }
    };
    Ok((head, body))
}

fn fill_chunked_queue(
    reader: &mut BufReader<TcpStream>,
    state: &mut ChunkedState,
    target: usize,
) -> Result<(), u32> {
    while state.queued.len() < target {
        match state.position {
            ChunkedPosition::Eof => break,
            ChunkedPosition::SizeLine => {
                let line = read_crlf_line(reader)?;
                let text =
                    String::from_utf8(line).map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
                let size_text = text.split(';').next().unwrap_or("").trim();
                if size_text.is_empty()
                    || size_text.len() > 16
                    || !size_text.bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
                }
                let size = u64::from_str_radix(size_text, 16)
                    .map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
                state.chunk_remaining = size;
                state.position = if size == 0 {
                    ChunkedPosition::Trailers
                } else {
                    ChunkedPosition::Data
                };
            }
            ChunkedPosition::Data => {
                let wanted = (target - state.queued.len())
                    .min(state.chunk_remaining as usize)
                    .min(MAX_READ_CHUNK_BYTES);
                if wanted == 0 {
                    continue;
                }
                let mut bytes = vec![0; wanted];
                let count = reader.read(&mut bytes).map_err(io_winhttp_error)?;
                if count == 0 {
                    return Err(ERROR_WINHTTP_CONNECTION_ERROR);
                }
                bytes.truncate(count);
                state.queued.extend(bytes);
                state.chunk_remaining -= count as u64;
                if state.chunk_remaining == 0 {
                    state.position = ChunkedPosition::DataCrLf;
                }
                break;
            }
            ChunkedPosition::DataCrLf => {
                let mut crlf = [0; 2];
                reader.read_exact(&mut crlf).map_err(io_winhttp_error)?;
                if &crlf != b"\r\n" {
                    return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
                }
                state.position = ChunkedPosition::SizeLine;
            }
            ChunkedPosition::Trailers => loop {
                let line = read_crlf_line(reader)?;
                state.trailer_bytes += line.len() + 2;
                if state.trailer_bytes > MAX_RESPONSE_HEAD_BYTES {
                    return Err(ERROR_WINHTTP_HEADER_SIZE_OVERFLOW);
                }
                if line.is_empty() {
                    state.position = ChunkedPosition::Eof;
                    break;
                }
                let line =
                    String::from_utf8(line).map_err(|_| ERROR_WINHTTP_INVALID_SERVER_RESPONSE)?;
                let Some((name, value)) = line.split_once(':') else {
                    return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
                };
                let value = value.trim_matches(|byte| byte == ' ' || byte == '\t');
                if !valid_header_name(name) || !valid_header_value(value) {
                    return Err(ERROR_WINHTTP_INVALID_SERVER_RESPONSE);
                }
            },
        }
    }
    Ok(())
}

fn close_result(state: &ServicesState, teb: u64, closed: bool) -> u64 {
    if closed {
        1
    } else {
        state.fail(teb, ERROR_INVALID_HANDLE, 0).unwrap()
    }
}

/// Register the implemented handle-lifecycle subset. Every function remains
/// partial because documented proxy, asynchronous and transport-related flag
/// combinations are intentionally refused rather than silently accepted.
pub(super) fn register(registry: &mut ImportRegistry, state: &Arc<ServicesState>) {
    let open = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpOpen",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            let agent = match read_optional_wide_string(memory, args.rcx) {
                Ok(value) => value,
                Err(error) => return open.fail(cpu.gs_base, error, 0),
            };
            let access_type = args.rdx as u32;
            match access_type {
                WINHTTP_ACCESS_TYPE_DEFAULT_PROXY | WINHTTP_ACCESS_TYPE_NO_PROXY => {}
                3 | 4 => return open.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0),
                _ => return open.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0),
            }
            // Proxy strings are inputs to WINHTTP_ACCESS_TYPE_NAMED_PROXY.
            // That access type is refused above; direct and default modes do
            // not consume them, so they are deliberately not dereferenced.
            let flags = args.get(4, memory).map_err(memory_fault)? as u32;
            if flags == WINHTTP_FLAG_ASYNC {
                return open.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0);
            }
            if flags != 0 {
                return open.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            let mut handles = open.winhttp.lock().expect("WinHTTP handle table");
            match handles.insert(HandleKind::Session { agent }) {
                Ok(handle) => Ok(handle),
                Err(error) => open.fail(cpu.gs_base, error, 0),
            }
        },
    );

    let connect = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpConnect",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            if args.r9 != 0 {
                return connect.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            let server = match read_wide_string(memory, args.rdx, MAX_WIDE_UNITS) {
                Ok(server) if valid_server_name(&server) => server,
                Ok(_) | Err(_) => {
                    return connect.fail(cpu.gs_base, ERROR_WINHTTP_INVALID_URL, 0);
                }
            };
            // INTERNET_PORT is a WORD. As in the Win64 ABI, only the low
            // sixteen bits of the register carry the argument.
            let port = u32::from(args.r8 as u16);
            let mut handles = connect.winhttp.lock().expect("WinHTTP handle table");
            let agent = match handles.kind(args.rcx) {
                Some(HandleKind::Session { agent, .. }) => agent.clone(),
                Some(_) => {
                    return connect.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_TYPE, 0);
                }
                None => return connect.fail(cpu.gs_base, ERROR_INVALID_HANDLE, 0),
            };
            match handles.insert(HandleKind::Connection {
                agent,
                server,
                port,
            }) {
                Ok(handle) => Ok(handle),
                Err(error) => connect.fail(cpu.gs_base, error, 0),
            }
        },
    );

    let request = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpOpenRequest",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            let verb = match read_optional_wide_string(memory, args.rdx) {
                Ok(Some(verb)) if valid_token(&verb) => verb,
                Ok(Some(_)) | Err(_) => {
                    return request.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
                }
                Ok(None) => "GET".to_owned(),
            };
            let object_name = match read_optional_wide_string(memory, args.r8) {
                Ok(Some(object)) => object,
                // Microsoft's examples pass NULL successfully but do not
                // specify an expansion. Retain the absent target honestly;
                // a transport slice must define wire behavior separately.
                Ok(None) => String::new(),
                Err(_) => return request.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0),
            };
            let version = match read_optional_wide_string(memory, args.r9) {
                Ok(Some(version)) if valid_token(&version) => version,
                Ok(Some(_)) | Err(_) => {
                    return request.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
                }
                Ok(None) => "HTTP/1.1".to_owned(),
            };
            let referrer_ptr = args.get(4, memory).map_err(memory_fault)?;
            let referrer = match read_optional_wide_string(memory, referrer_ptr) {
                Ok(value) => value,
                Err(error) => return request.fail(cpu.gs_base, error, 0),
            };
            let accept_ptr = args.get(5, memory).map_err(memory_fault)?;
            let mut accept_types = Vec::new();
            if accept_ptr != 0 {
                let mut terminated = false;
                for index in 0..MAX_ACCEPT_TYPES {
                    let pointer = memory
                        .read_u64(accept_ptr + 8 * index as u64)
                        .map_err(|_| ERROR_INVALID_PARAMETER);
                    let pointer = match pointer {
                        Ok(pointer) => pointer,
                        Err(error) => return request.fail(cpu.gs_base, error, 0),
                    };
                    if pointer == 0 {
                        terminated = true;
                        break;
                    }
                    let Ok(text) = read_wide_string(memory, pointer, MAX_WIDE_UNITS) else {
                        return request.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
                    };
                    accept_types.push(text);
                }
                if !terminated {
                    return request.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
                }
            }
            let flags = args.get(6, memory).map_err(memory_fault)? as u32;
            if flags & !DOCUMENTED_REQUEST_FLAGS != 0 {
                return request.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            if flags & !WINHTTP_FLAG_SECURE != 0 {
                return request.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0);
            }
            let mut handles = request.winhttp.lock().expect("WinHTTP handle table");
            let (agent, server, port) = match handles.kind(args.rcx) {
                Some(HandleKind::Connection {
                    agent,
                    server,
                    port,
                    ..
                }) => (agent.clone(), server.clone(), *port),
                Some(_) => {
                    return request.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_TYPE, 0);
                }
                None => return request.fail(cpu.gs_base, ERROR_INVALID_HANDLE, 0),
            };
            match handles.insert(HandleKind::Request {
                agent,
                server,
                port,
                verb,
                object_name,
                version,
                referrer,
                accept_types,
                secure: flags & WINHTTP_FLAG_SECURE != 0,
                transport: Arc::new(Mutex::new(TransportState::new())),
            }) {
                Ok(handle) => Ok(handle),
                Err(error) => request.fail(cpu.gs_base, error, 0),
            }
        },
    );

    let send = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpSendRequest",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            let (snapshot, transport) = {
                let handles = send.winhttp.lock().expect("WinHTTP handle table");
                match request_parts(&handles, args.rcx) {
                    Ok(parts) => parts,
                    Err(error) => return send.fail(cpu.gs_base, error, 0),
                }
            };
            if snapshot.secure {
                return send.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0);
            }
            let additional = match read_additional_headers(memory, args.rdx, args.r8 as u32) {
                Ok(headers) => headers,
                Err(error) => return send.fail(cpu.gs_base, error, 0),
            };
            let optional_length = args.get(4, memory).map_err(memory_fault)? as u32;
            let total_length = args.get(5, memory).map_err(memory_fault)? as u32;
            let _context = args.get(6, memory).map_err(memory_fault)?;
            if total_length < optional_length {
                return send.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            if total_length != optional_length {
                return send.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0);
            }
            if optional_length > MAX_OPTIONAL_BODY_BYTES {
                return send.fail(cpu.gs_base, ERROR_OUTOFMEMORY, 0);
            }
            let mut body = vec![0; optional_length as usize];
            if optional_length != 0 {
                if args.r9 == 0 {
                    return send.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
                }
                if memory.read_bytes(args.r9, &mut body).is_err() {
                    return send.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
                }
            }
            let wire = match build_request_wire(&snapshot, &additional, &body, total_length) {
                Ok(wire) => wire,
                Err(error) => return send.fail(cpu.gs_base, error, 0),
            };
            {
                let mut transport_guard = transport.lock().expect("WinHTTP transport");
                if transport_guard.phase != TransportPhase::New {
                    return send.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_STATE, 0);
                }
                transport_guard.phase = TransportPhase::Busy;
            }
            let port = effective_port(snapshot.port, snapshot.secure);
            let result = connect_tcp(&snapshot.server, port).and_then(|mut stream| {
                stream.write_all(&wire).map_err(io_winhttp_error)?;
                Ok(stream)
            });
            match result {
                Ok(stream) => {
                    let mut transport_guard = transport.lock().expect("WinHTTP transport");
                    transport_guard.phase = TransportPhase::Sent;
                    transport_guard.verb = snapshot.verb;
                    transport_guard.stream = Some(BufReader::new(stream));
                    Ok(1)
                }
                Err(error) => {
                    fail_transport(&transport);
                    send.fail(cpu.gs_base, error, 0)
                }
            }
        },
    );

    let receive = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpReceiveResponse",
        ServiceStatus::Partial,
        move |args, cpu, _| {
            if args.rdx != 0 {
                return receive.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            let transport = {
                let handles = receive.winhttp.lock().expect("WinHTTP handle table");
                match request_transport(&handles, args.rcx) {
                    Ok(transport) => transport,
                    Err(error) => return receive.fail(cpu.gs_base, error, 0),
                }
            };
            let mut transport_guard = transport.lock().expect("WinHTTP transport");
            if transport_guard.phase != TransportPhase::Sent {
                return receive.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_STATE, 0);
            }
            let verb = transport_guard.verb.clone();
            let result = match transport_guard.stream.as_mut() {
                Some(reader) => parse_response(reader, &verb),
                None => Err(ERROR_WINHTTP_INCORRECT_HANDLE_STATE),
            };
            match result {
                Ok((head, body)) => {
                    transport_guard.response = Some(head);
                    transport_guard.body = body;
                    transport_guard.phase = TransportPhase::Ready;
                    Ok(1)
                }
                Err(error) => {
                    transport_guard.phase = TransportPhase::Failed;
                    receive.fail(cpu.gs_base, error, 0)
                }
            }
        },
    );

    let query_available = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpQueryDataAvailable",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            if args.rdx == 0 {
                return query_available.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            let transport = {
                let handles = query_available
                    .winhttp
                    .lock()
                    .expect("WinHTTP handle table");
                match request_transport(&handles, args.rcx) {
                    Ok(transport) => transport,
                    Err(error) => return query_available.fail(cpu.gs_base, error, 0),
                }
            };
            let mut transport_guard = transport.lock().expect("WinHTTP transport");
            if transport_guard.phase != TransportPhase::Ready {
                return query_available.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_STATE, 0);
            }
            let body = transport_guard.body;
            let mut mark_eof = false;
            let TransportState {
                stream, chunked, ..
            } = &mut *transport_guard;
            let result = match stream.as_mut() {
                Some(reader) => match body {
                    BodyKind::Empty | BodyKind::Fixed(0) => Ok(0_usize),
                    BodyKind::Fixed(remaining) => match reader.fill_buf() {
                        Ok([]) => Err(ERROR_WINHTTP_CONNECTION_ERROR),
                        Ok(buffer) => Ok(buffer.len().min(remaining as usize)),
                        Err(error) => Err(io_winhttp_error(error)),
                    },
                    BodyKind::UntilClose { eof: true } => Ok(0_usize),
                    BodyKind::UntilClose { eof: false } => match reader.fill_buf() {
                        Ok(buffer) => {
                            if buffer.is_empty() {
                                mark_eof = true;
                            }
                            Ok(buffer.len())
                        }
                        Err(error) => Err(io_winhttp_error(error)),
                    },
                    BodyKind::Chunked => fill_chunked_queue(reader, chunked, MAX_READ_CHUNK_BYTES)
                        .map(|()| chunked.queued.len()),
                },
                None => Err(ERROR_WINHTTP_INCORRECT_HANDLE_STATE),
            };
            if mark_eof {
                transport_guard.body = BodyKind::UntilClose { eof: true };
            }
            match result {
                Ok(available) => {
                    memory
                        .write_u32(args.rdx, available as u32)
                        .map_err(memory_fault)?;
                    Ok(1)
                }
                Err(error) => {
                    transport_guard.phase = TransportPhase::Failed;
                    query_available.fail(cpu.gs_base, error, 0)
                }
            }
        },
    );

    let read_data = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpReadData",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            if args.r9 == 0 {
                return read_data.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            memory.write_u32(args.r9, 0).map_err(memory_fault)?;
            let wanted = args.r8 as u32;
            if wanted != 0 {
                memory
                    .check_write(args.rdx, wanted as usize)
                    .map_err(memory_fault)?;
            }
            let transport = {
                let handles = read_data.winhttp.lock().expect("WinHTTP handle table");
                match request_transport(&handles, args.rcx) {
                    Ok(transport) => transport,
                    Err(error) => return read_data.fail(cpu.gs_base, error, 0),
                }
            };
            let mut transport_guard = transport.lock().expect("WinHTTP transport");
            if transport_guard.phase != TransportPhase::Ready {
                return read_data.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_STATE, 0);
            }
            let body = transport_guard.body;
            let limit = match body {
                BodyKind::Empty | BodyKind::Fixed(0) => 0_usize,
                BodyKind::Fixed(remaining) => (remaining as usize)
                    .min(wanted as usize)
                    .min(MAX_READ_CHUNK_BYTES),
                BodyKind::UntilClose { eof: true } => 0_usize,
                BodyKind::UntilClose { eof: false } => (wanted as usize).min(MAX_READ_CHUNK_BYTES),
                BodyKind::Chunked => (wanted as usize).min(MAX_READ_CHUNK_BYTES),
            };
            let mut bytes = vec![0; limit];
            let result = if limit == 0 {
                Ok(0_usize)
            } else {
                let TransportState {
                    stream, chunked, ..
                } = &mut *transport_guard;
                match stream.as_mut() {
                    Some(reader) if matches!(body, BodyKind::Chunked) => {
                        fill_chunked_queue(reader, chunked, limit).map(|()| {
                            let mut count = 0;
                            while count < limit {
                                let Some(byte) = chunked.queued.pop_front() else {
                                    break;
                                };
                                bytes[count] = byte;
                                count += 1;
                            }
                            count
                        })
                    }
                    Some(reader) => reader.read(&mut bytes).map_err(io_winhttp_error),
                    None => Err(ERROR_WINHTTP_INCORRECT_HANDLE_STATE),
                }
            };
            match result {
                Ok(count) => {
                    match body {
                        BodyKind::Fixed(remaining) => {
                            if count == 0 && remaining != 0 {
                                transport_guard.phase = TransportPhase::Failed;
                                return read_data.fail(
                                    cpu.gs_base,
                                    ERROR_WINHTTP_CONNECTION_ERROR,
                                    0,
                                );
                            }
                            transport_guard.body = BodyKind::Fixed(remaining - count as u64);
                        }
                        BodyKind::UntilClose { eof: false } if count == 0 => {
                            transport_guard.body = BodyKind::UntilClose { eof: true };
                        }
                        _ => {}
                    }
                    bytes.truncate(count);
                    memory.write_bytes(args.rdx, &bytes).map_err(memory_fault)?;
                    memory
                        .write_u32(args.r9, count as u32)
                        .map_err(memory_fault)?;
                    Ok(1)
                }
                Err(error) => {
                    transport_guard.phase = TransportPhase::Failed;
                    read_data.fail(cpu.gs_base, error, 0)
                }
            }
        },
    );

    let query_headers = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpQueryHeaders",
        ServiceStatus::Partial,
        move |args, cpu, memory| {
            let info_level = args.rdx as u32;
            let length_ptr = args.get(4, memory).map_err(memory_fault)?;
            let index_ptr = args.get(5, memory).map_err(memory_fault)?;
            if args.r8 != 0 || length_ptr == 0 {
                return query_headers.fail(cpu.gs_base, ERROR_INVALID_PARAMETER, 0);
            }
            if index_ptr != 0 {
                return query_headers.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0);
            }
            let numeric_status =
                info_level == (WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER);
            let numeric_length =
                info_level == (WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER);
            let textual_status = info_level == WINHTTP_QUERY_STATUS_CODE;
            let textual_status_text = info_level == WINHTTP_QUERY_STATUS_TEXT;
            let textual_content_type = info_level == WINHTTP_QUERY_CONTENT_TYPE;
            let textual_content_length = info_level == WINHTTP_QUERY_CONTENT_LENGTH;
            let textual_server = info_level == WINHTTP_QUERY_SERVER;
            if !(numeric_status
                || numeric_length
                || textual_status
                || textual_status_text
                || textual_content_type
                || textual_content_length
                || textual_server)
            {
                return query_headers.fail(cpu.gs_base, ERROR_NOT_SUPPORTED, 0);
            }
            let transport = {
                let handles = query_headers.winhttp.lock().expect("WinHTTP handle table");
                match request_transport(&handles, args.rcx) {
                    Ok(transport) => transport,
                    Err(error) => return query_headers.fail(cpu.gs_base, error, 0),
                }
            };
            let transport_guard = transport.lock().expect("WinHTTP transport");
            if transport_guard.phase != TransportPhase::Ready {
                return query_headers.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_STATE, 0);
            }
            let Some(response) = &transport_guard.response else {
                return query_headers.fail(cpu.gs_base, ERROR_WINHTTP_INCORRECT_HANDLE_STATE, 0);
            };
            let named_header = if textual_content_type {
                Some("content-type")
            } else if textual_content_length || numeric_length {
                Some("content-length")
            } else if textual_server {
                Some("server")
            } else {
                None
            };
            let header_value = named_header.map(|wanted| {
                response
                    .headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(wanted))
                    .map(|(_, value)| value.as_str())
            });
            if matches!(header_value, Some(None)) {
                return query_headers.fail(cpu.gs_base, ERROR_WINHTTP_HEADER_NOT_FOUND, 0);
            }
            let (contents, success_length) = if numeric_status {
                (response.status.to_le_bytes().to_vec(), 4_u32)
            } else if numeric_length {
                let value = header_value
                    .and_then(|value| value)
                    .expect("content-length");
                let Ok(value) = value.parse::<u32>() else {
                    return query_headers.fail(
                        cpu.gs_base,
                        ERROR_WINHTTP_INVALID_SERVER_RESPONSE,
                        0,
                    );
                };
                (value.to_le_bytes().to_vec(), 4_u32)
            } else {
                let text = if textual_status {
                    response.status.to_string()
                } else if textual_status_text {
                    response.status_text.clone()
                } else {
                    header_value
                        .and_then(|value| value)
                        .expect("named header")
                        .to_owned()
                };
                let unit_count = text.encode_utf16().count();
                let mut bytes = Vec::with_capacity((unit_count + 1) * 2);
                for unit in text.encode_utf16().chain([0]) {
                    bytes.extend(unit.to_le_bytes());
                }
                (bytes, (unit_count * 2) as u32)
            };
            let required = contents.len() as u32;
            let capacity = memory.read_u32(length_ptr).map_err(memory_fault)?;
            memory
                .write_u32(length_ptr, required)
                .map_err(memory_fault)?;
            if args.r9 == 0 || capacity < required {
                return query_headers.fail(cpu.gs_base, ERROR_INSUFFICIENT_BUFFER, 0);
            }
            memory
                .write_bytes(args.r9, &contents)
                .map_err(memory_fault)?;
            memory
                .write_u32(length_ptr, success_length)
                .map_err(memory_fault)?;
            Ok(1)
        },
    );

    let close = Arc::clone(state);
    registry.register_with_status(
        MODULE,
        "WinHttpCloseHandle",
        ServiceStatus::Partial,
        move |args, cpu, _| {
            let mut handles = close.winhttp.lock().expect("WinHTTP handle table");
            Ok(close_result(&close, cpu.gs_base, handles.close(args.rcx)))
        },
    );
}

// The service handlers return `Result<_, ExitReason>`; keeping the conversion
// in one place avoids exposing memory faults as successful handle values.
fn memory_fault(error: crate::memory::MemoryError) -> crate::cpu::ExitReason {
    crate::cpu::ExitReason::MemoryFault(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuState, Register};
    use crate::import_table::ImportName;
    use crate::imports::Win64Args;
    use crate::memory::MemoryPermissions;
    use crate::services::MockClock;

    fn write_wide(memory: &mut GuestMemory, address: u64, text: &str) {
        for (index, unit) in text.encode_utf16().chain([0]).enumerate() {
            memory.write_u16(address + 2 * index as u64, unit).unwrap();
        }
    }

    fn read_test_wide(memory: &GuestMemory, address: u64, byte_len: u32) -> String {
        let units = (0..byte_len / 2)
            .map(|index| memory.read_u16(address + 2 * u64::from(index)).unwrap())
            .collect::<Vec<_>>();
        String::from_utf16(&units).unwrap()
    }

    struct Fixture {
        registry: ImportRegistry,
        state: Arc<ServicesState>,
        memory: GuestMemory,
    }

    impl Fixture {
        fn new() -> Self {
            let mut registry = ImportRegistry::new();
            let state = Arc::new(ServicesState::new(Arc::new(MockClock::new(0))));
            register(&mut registry, &state);
            let mut memory = GuestMemory::new();
            memory
                .map_region(0x1000, 0x8000, MemoryPermissions::READ_WRITE)
                .unwrap();
            write_wide(&mut memory, 0x2000, "XenoraTest/1.0");
            write_wide(&mut memory, 0x2100, "example.test");
            write_wide(&mut memory, 0x2200, "GET");
            write_wide(&mut memory, 0x2300, "/fixture");
            write_wide(&mut memory, 0x2400, "HTTP/1.1");
            Self {
                registry,
                state,
                memory,
            }
        }

        fn call(&mut self, symbol: &str, args: &[u64]) -> u64 {
            let mut cpu = CpuState::default();
            for (register, value) in [Register::Rcx, Register::Rdx, Register::R8, Register::R9]
                .into_iter()
                .zip(args.iter().copied())
            {
                cpu.set(register, value);
            }
            cpu.set(Register::Rsp, 0x7000);
            for (index, value) in args.iter().copied().enumerate().skip(4) {
                self.memory
                    .write_u64(0x7000 + 0x28 + (index as u64 - 4) * 8, value)
                    .unwrap();
            }
            let descriptor = self
                .registry
                .resolve(MODULE, &ImportName::Name(symbol.to_owned()))
                .unwrap();
            (descriptor.handler)(&Win64Args::new(&cpu), &mut cpu, &mut self.memory).unwrap()
        }

        fn open_session(&mut self) -> u64 {
            self.call("WinHttpOpen", &[0x2000, 1, 0, 0, 0])
        }

        fn open_connection(&mut self) -> (u64, u64) {
            let session = self.open_session();
            let connection = self.call("WinHttpConnect", &[session, 0x2100, 80, 0]);
            (session, connection)
        }
    }

    #[test]
    fn session_connection_request_and_close_have_owner_lifetimes() {
        let mut fixture = Fixture::new();
        let (session, connection) = fixture.open_connection();
        assert_ne!(session, 0);
        assert_ne!(connection, 0);
        assert_ne!(session, connection);
        write_wide(&mut fixture.memory, 0x2500, "text/plain");
        fixture.memory.write_u64(0x2580, 0x2500).unwrap();
        fixture.memory.write_u64(0x2588, 0).unwrap();
        let request = fixture.call(
            "WinHttpOpenRequest",
            &[connection, 0x2200, 0x2300, 0x2400, 0, 0x2580, 0],
        );
        assert_ne!(request, 0);
        assert_eq!(fixture.state.open_handle_count(), 3);
        assert_eq!(fixture.call("WinHttpCloseHandle", &[request]), 1);
        assert_eq!(fixture.call("WinHttpCloseHandle", &[request]), 0);
        assert_eq!(fixture.state.last_error(0), ERROR_INVALID_HANDLE);
        assert_eq!(fixture.call("WinHttpCloseHandle", &[connection]), 1);
        assert_eq!(fixture.call("WinHttpCloseHandle", &[session]), 1);
        assert_eq!(fixture.state.open_handle_count(), 0);
    }

    #[test]
    fn defaults_and_secure_flag_are_retained_without_transport() {
        let mut fixture = Fixture::new();
        let (_session, connection) = fixture.open_connection();
        let request = fixture.call(
            "WinHttpOpenRequest",
            &[connection, 0, 0, 0, 0, 0, WINHTTP_FLAG_SECURE as u64],
        );
        assert_ne!(request, 0);
        let handles = fixture.state.winhttp.lock().unwrap();
        assert!(matches!(
            handles.kind(request),
            Some(HandleKind::Request {
                verb,
                object_name,
                version,
                secure: true,
                ..
            }) if verb == "GET" && object_name.is_empty() && version == "HTTP/1.1"
        ));
    }

    #[test]
    fn invalid_parents_flags_strings_and_reserved_values_fail_honestly() {
        let mut fixture = Fixture::new();
        assert_eq!(fixture.call("WinHttpConnect", &[1234, 0x2100, 80, 0]), 0);
        assert_eq!(fixture.state.last_error(0), ERROR_INVALID_HANDLE);
        let wrong_parent = fixture.open_session();
        assert_eq!(
            fixture.call("WinHttpOpenRequest", &[wrong_parent, 0, 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INCORRECT_HANDLE_TYPE
        );
        assert_eq!(fixture.call("WinHttpCloseHandle", &[wrong_parent]), 1);
        assert_eq!(fixture.call("WinHttpOpen", &[0x2000, 4, 0, 0, 0]), 0);
        assert_eq!(fixture.state.last_error(0), ERROR_NOT_SUPPORTED);
        assert_eq!(
            fixture.call("WinHttpOpen", &[0x2000, 1, 0, 0, WINHTTP_FLAG_ASYNC as u64]),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_NOT_SUPPORTED);
        let (_session, connection) = fixture.open_connection();
        assert_eq!(
            fixture.call(
                "WinHttpOpenRequest",
                &[connection, 0x2200, 0x2300, 0x2400, 0, 0, 4]
            ),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_NOT_SUPPORTED);
        fixture.memory.write_u16(0x2100, 0).unwrap();
        assert_eq!(fixture.call("WinHttpConnect", &[1234, 0x2100, 80, 0]), 0);
        assert_eq!(fixture.state.last_error(0), ERROR_WINHTTP_INVALID_URL);
    }

    #[test]
    fn process_exit_closes_the_winhttp_handle_table() {
        let mut fixture = Fixture::new();
        let (_session, connection) = fixture.open_connection();
        assert_ne!(connection, 0);
        assert_eq!(fixture.state.open_handle_count(), 2);
        assert_eq!(fixture.state.exit_process(7), 7);
        assert_eq!(fixture.state.open_handle_count(), 0);
    }

    #[test]
    fn internet_port_uses_only_the_low_sixteen_bits() {
        let mut fixture = Fixture::new();
        let session = fixture.open_session();
        let connection = fixture.call("WinHttpConnect", &[session, 0x2100, 0x1_0000_0050, 0]);
        assert_ne!(connection, 0);
        let handles = fixture.state.winhttp.lock().unwrap();
        assert!(matches!(
            handles.kind(connection),
            Some(HandleKind::Connection { port: 80, .. })
        ));
    }

    #[test]
    fn accept_types_are_copied_out_of_guest_memory() {
        let mut fixture = Fixture::new();
        write_wide(&mut fixture.memory, 0x2500, "text/plain");
        fixture.memory.write_u64(0x2600, 0x2500).unwrap();
        fixture.memory.write_u64(0x2608, 0).unwrap();
        let (_session, connection) = fixture.open_connection();
        let request = fixture.call("WinHttpOpenRequest", &[connection, 0, 0, 0, 0, 0x2600, 0]);
        assert_ne!(request, 0);
        write_wide(&mut fixture.memory, 0x2500, "text/html");
        let handles = fixture.state.winhttp.lock().unwrap();
        assert!(matches!(
            handles.kind(request),
            Some(HandleKind::Request { accept_types, .. }) if accept_types == &vec!["text/plain".to_owned()]
        ));
    }

    #[test]
    fn closing_a_parent_does_not_mutate_an_existing_request_snapshot() {
        let mut fixture = Fixture::new();
        let (session, connection) = fixture.open_connection();
        let request = fixture.call("WinHttpOpenRequest", &[connection, 0, 0, 0, 0, 0, 0]);
        assert_ne!(request, 0);
        assert_eq!(fixture.call("WinHttpCloseHandle", &[connection]), 1);
        assert_eq!(fixture.call("WinHttpCloseHandle", &[session]), 1);
        let handles = fixture.state.winhttp.lock().unwrap();
        assert!(matches!(
            handles.kind(request),
            Some(HandleKind::Request {
                server,
                port: 80,
                ..
            }) if server == "example.test"
        ));
    }

    fn serve_once(
        response: &'static [u8],
    ) -> (u16, Arc<Mutex<Vec<u8>>>, std::thread::JoinHandle<()>) {
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_thread = Arc::clone(&captured);
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            loop {
                let count = stream.read(&mut buffer).unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
                let complete = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .is_some_and(|header_end| {
                        let head = String::from_utf8_lossy(&request[..header_end]);
                        let body_length = head
                            .split("\r\n")
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        request.len() >= header_end + 4 + body_length
                    });
                if complete {
                    break;
                }
                assert!(request.len() < 16 * 1024);
            }
            *captured_thread.lock().unwrap() = request;
            stream.write_all(response).unwrap();
        });
        (port, captured, thread)
    }

    fn write_loopback_host(fixture: &mut Fixture) {
        write_wide(&mut fixture.memory, 0x2100, "127.0.0.1");
    }

    fn open_loopback_request(fixture: &mut Fixture, port: u16) -> u64 {
        write_loopback_host(fixture);
        let session = fixture.open_session();
        let connection = fixture.call("WinHttpConnect", &[session, 0x2100, port as u64, 0]);
        assert_ne!(connection, 0);
        write_wide(&mut fixture.memory, 0x2500, "text/plain");
        fixture.memory.write_u64(0x2580, 0x2500).unwrap();
        fixture.memory.write_u64(0x2588, 0).unwrap();
        let request = fixture.call(
            "WinHttpOpenRequest",
            &[connection, 0x2200, 0x2300, 0x2400, 0, 0x2580, 0],
        );
        assert_ne!(request, 0);
        request
    }

    #[test]
    fn synchronous_plaintext_get_reads_status_and_response_body() {
        let (port, captured, server) = serve_once(
            b"HTTP/1.1 203 Non-Authoritative\r\nContent-Length: 11\r\nContent-Type: text/plain\r\nX-Server: fixture\r\n\r\nhello WORLD",
        );
        let mut fixture = Fixture::new();
        let request = open_loopback_request(&mut fixture, port);
        write_wide(&mut fixture.memory, 0x2700, "X-Fixture: yes\r\n");
        assert_eq!(
            fixture.call(
                "WinHttpSendRequest",
                &[request, 0x2700, u32::MAX as u64, 0, 0, 0, 0xabc]
            ),
            1
        );
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 1);
        fixture.memory.write_u32(0x2908, 2).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    (WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER) as u64,
                    0,
                    0x2900,
                    0x2908,
                    0,
                ]
            ),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_INSUFFICIENT_BUFFER);
        assert_eq!(fixture.memory.read_u32(0x2908).unwrap(), 4);
        fixture.memory.write_u32(0x2908, 4).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    (WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER) as u64,
                    0,
                    0x2900,
                    0x2908,
                    0,
                ]
            ),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2900).unwrap(), 203);

        fixture.memory.write_u32(0x2928, 256).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    WINHTTP_QUERY_STATUS_TEXT as u64,
                    0,
                    0x3000,
                    0x2928,
                    0
                ]
            ),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2928).unwrap(), 34);
        assert_eq!(
            read_test_wide(&fixture.memory, 0x3000, 34),
            "Non-Authoritative"
        );

        fixture.memory.write_u32(0x2938, 4).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    (WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER) as u64,
                    0,
                    0x3020,
                    0x2938,
                    0,
                ]
            ),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x3020).unwrap(), 11);

        fixture.memory.write_u32(0x2948, 256).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    WINHTTP_QUERY_CONTENT_TYPE as u64,
                    0,
                    0x3040,
                    0x2948,
                    0
                ]
            ),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2948).unwrap(), 20);
        assert_eq!(read_test_wide(&fixture.memory, 0x3040, 20), "text/plain");

        fixture.memory.write_u32(0x2958, 256).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[request, WINHTTP_QUERY_SERVER as u64, 0, 0x3080, 0x2958, 0]
            ),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_WINHTTP_HEADER_NOT_FOUND);
        assert_eq!(
            fixture.call("WinHttpQueryDataAvailable", &[request, 0x2910]),
            1
        );
        let available = fixture.memory.read_u32(0x2910).unwrap();
        assert!((1..=11).contains(&available));
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a00, 5, 0x2914]),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2914).unwrap(), 5);
        let mut first = [0; 5];
        fixture.memory.read_bytes(0x2a00, &mut first).unwrap();
        assert_eq!(&first, b"hello");
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a10, 64, 0x2914]),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2914).unwrap(), 6);
        let mut rest = [0; 6];
        fixture.memory.read_bytes(0x2a10, &mut rest).unwrap();
        assert_eq!(&rest, b" WORLD");
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a20, 8, 0x2914]),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2914).unwrap(), 0);
        server.join().unwrap();
        let wire = captured.lock().unwrap().clone();
        let wire = String::from_utf8(wire).unwrap();
        assert!(wire.starts_with("GET /fixture HTTP/1.1\r\n"));
        assert!(wire.contains(&format!("Host: 127.0.0.1:{port}\r\n")));
        assert!(wire.contains("User-Agent: XenoraTest/1.0\r\n"));
        assert!(wire.contains("Accept: text/plain\r\n"));
        assert!(wire.contains("X-Fixture: yes\r\n"));
        assert!(wire.contains("Connection: close\r\n\r\n"));
    }

    #[test]
    fn optional_request_data_is_copied_into_the_wire_body() {
        let (port, captured, server) =
            serve_once(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n");
        let mut fixture = Fixture::new();
        write_wide(&mut fixture.memory, 0x2200, "POST");
        let request = open_loopback_request(&mut fixture, port);
        fixture.memory.write_bytes(0x2800, b"data").unwrap();
        assert_eq!(
            fixture.call("WinHttpSendRequest", &[request, 0, 0, 0x2800, 4, 4, 0]),
            1
        );
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 1);
        server.join().unwrap();
        let wire = captured.lock().unwrap().clone();
        let text = String::from_utf8_lossy(&wire);
        assert!(text.starts_with("POST /fixture HTTP/1.1\r\n"));
        assert!(text.contains("Content-Length: 4\r\n"));
        assert!(wire.ends_with(b"data"));
    }

    #[test]
    fn transport_state_is_enforced_before_network_use() {
        let mut fixture = Fixture::new();
        let (_session, connection) = fixture.open_connection();
        let request = fixture.call("WinHttpOpenRequest", &[connection, 0, 0, 0, 0, 0, 0]);
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 0);
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INCORRECT_HANDLE_STATE
        );
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a00, 1, 0x2914]),
            0
        );
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INCORRECT_HANDLE_STATE
        );
        fixture.memory.write_u32(0x2908, 4).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    (WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER) as u64,
                    0,
                    0x2900,
                    0x2908,
                    0,
                ]
            ),
            0
        );
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INCORRECT_HANDLE_STATE
        );
    }

    #[test]
    fn chunked_response_is_decoded_incrementally() {
        let (port, _captured, server) = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6;extension=yes\r\n WORLD\r\n0\r\nX-Checksum: yes\r\n\r\n",
        );
        let mut fixture = Fixture::new();
        let request = open_loopback_request(&mut fixture, port);
        assert_eq!(
            fixture.call("WinHttpSendRequest", &[request, 0, 0, 0, 0, 0, 0]),
            1
        );
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 1);
        fixture.memory.write_u32(0x2958, 4).unwrap();
        assert_eq!(
            fixture.call(
                "WinHttpQueryHeaders",
                &[
                    request,
                    (WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER) as u64,
                    0,
                    0x3020,
                    0x2958,
                    0,
                ]
            ),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_WINHTTP_HEADER_NOT_FOUND);
        assert_eq!(
            fixture.call("WinHttpQueryDataAvailable", &[request, 0x2910]),
            1
        );
        assert!(fixture.memory.read_u32(0x2910).unwrap() > 0);
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a00, 4, 0x2914]),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2914).unwrap(), 4);
        let mut first = [0; 4];
        fixture.memory.read_bytes(0x2a00, &mut first).unwrap();
        assert_eq!(&first, b"hell");
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a10, 32, 0x2914]),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2914).unwrap(), 7);
        let mut rest = [0; 7];
        fixture.memory.read_bytes(0x2a10, &mut rest).unwrap();
        assert_eq!(&rest, b"o WORLD");
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a20, 8, 0x2914]),
            1
        );
        assert_eq!(fixture.memory.read_u32(0x2914).unwrap(), 0);
        server.join().unwrap();
    }

    #[test]
    fn malformed_chunked_response_fails_during_body_read() {
        let (port, _captured, server) = serve_once(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nnot-hex\r\nhello\r\n0\r\n\r\n",
        );
        let mut fixture = Fixture::new();
        let request = open_loopback_request(&mut fixture, port);
        assert_eq!(
            fixture.call("WinHttpSendRequest", &[request, 0, 0, 0, 0, 0, 0]),
            1
        );
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 1);
        assert_eq!(
            fixture.call("WinHttpReadData", &[request, 0x2a00, 8, 0x2914]),
            0
        );
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INVALID_SERVER_RESPONSE
        );
        server.join().unwrap();
    }

    #[test]
    fn malformed_response_fails_without_reporting_a_status() {
        let (port, _captured, server) = serve_once(b"this is not HTTP\r\n\r\n");
        let mut fixture = Fixture::new();
        let request = open_loopback_request(&mut fixture, port);
        assert_eq!(
            fixture.call("WinHttpSendRequest", &[request, 0, 0, 0, 0, 0, 0]),
            1
        );
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 0);
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INVALID_SERVER_RESPONSE
        );
        server.join().unwrap();
    }

    #[test]
    fn connection_failure_is_reported_without_a_response_state() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut fixture = Fixture::new();
        let request = open_loopback_request(&mut fixture, port);
        assert_eq!(
            fixture.call("WinHttpSendRequest", &[request, 0, 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_WINHTTP_CANNOT_CONNECT);
        assert_eq!(fixture.call("WinHttpReceiveResponse", &[request, 0]), 0);
        assert_eq!(
            fixture.state.last_error(0),
            ERROR_WINHTTP_INCORRECT_HANDLE_STATE
        );
    }

    #[test]
    fn plaintext_secure_requests_are_refused_before_transport() {
        let mut fixture = Fixture::new();
        write_loopback_host(&mut fixture);
        let session = fixture.open_session();
        let connection = fixture.call("WinHttpConnect", &[session, 0x2100, 443, 0]);
        let request = fixture.call(
            "WinHttpOpenRequest",
            &[connection, 0, 0, 0, 0, 0, WINHTTP_FLAG_SECURE as u64],
        );
        assert_eq!(
            fixture.call("WinHttpSendRequest", &[request, 0, 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(fixture.state.last_error(0), ERROR_NOT_SUPPORTED);
    }

    #[test]
    fn concurrent_creation_is_allowed_on_the_same_parent_handles() {
        use std::sync::{Arc, Barrier};

        let state = Arc::new(ServicesState::new(Arc::new(MockClock::new(0))));
        let mut fixture = Fixture {
            registry: ImportRegistry::new(),
            state: Arc::clone(&state),
            memory: GuestMemory::new(),
        };
        register(&mut fixture.registry, &fixture.state);
        fixture
            .memory
            .map_region(0x1000, 0x8000, MemoryPermissions::READ_WRITE)
            .unwrap();
        write_wide(&mut fixture.memory, 0x2000, "agent");
        write_wide(&mut fixture.memory, 0x2100, "example.test");
        let session = fixture.open_session();
        let connection = fixture.call("WinHttpConnect", &[session, 0x2100, 80, 0]);
        assert_ne!(session, 0);
        assert_ne!(connection, 0);

        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();
        for use_connection in [false, true] {
            for _ in 0..4 {
                let state = Arc::clone(&state);
                let barrier = Arc::clone(&barrier);
                threads.push(std::thread::spawn(move || {
                    let mut registry = ImportRegistry::new();
                    register(&mut registry, &state);
                    let mut memory = GuestMemory::new();
                    memory
                        .map_region(0x1000, 0x8000, MemoryPermissions::READ_WRITE)
                        .unwrap();
                    write_wide(&mut memory, 0x2100, "example.test");
                    barrier.wait();
                    let parent = if use_connection { connection } else { session };
                    let symbol = if use_connection {
                        "WinHttpOpenRequest"
                    } else {
                        "WinHttpConnect"
                    };
                    let arguments = if use_connection {
                        [parent, 0, 0, 0, 0, 0, 0]
                    } else {
                        [parent, 0x2100, 80, 0, 0, 0, 0]
                    };
                    let mut cpu = CpuState::default();
                    for (register, value) in
                        [Register::Rcx, Register::Rdx, Register::R8, Register::R9]
                            .into_iter()
                            .zip(arguments.iter().copied())
                    {
                        cpu.set(register, value);
                    }
                    cpu.set(Register::Rsp, 0x7000);
                    for (index, value) in arguments.iter().copied().enumerate().skip(4) {
                        memory
                            .write_u64(0x7000 + 0x28 + (index as u64 - 4) * 8, value)
                            .unwrap();
                    }
                    let descriptor = registry
                        .resolve(MODULE, &ImportName::Name(symbol.to_owned()))
                        .unwrap();
                    let handle =
                        (descriptor.handler)(&Win64Args::new(&cpu), &mut cpu, &mut memory).unwrap();
                    assert_ne!(handle, 0);
                }));
            }
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(state.open_handle_count(), 10);
    }

    #[test]
    fn concurrent_services_share_one_thread_safe_handle_table() {
        use std::sync::{Arc, Barrier};

        let state = Arc::new(ServicesState::new(Arc::new(MockClock::new(0))));
        let barrier = Arc::new(Barrier::new(4));
        let mut threads = Vec::new();
        for _ in 0..4 {
            let state = Arc::clone(&state);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                let mut fixture = Fixture {
                    registry: ImportRegistry::new(),
                    state,
                    memory: GuestMemory::new(),
                };
                register(&mut fixture.registry, &fixture.state);
                fixture
                    .memory
                    .map_region(0x1000, 0x8000, MemoryPermissions::READ_WRITE)
                    .unwrap();
                write_wide(&mut fixture.memory, 0x2000, "agent");
                write_wide(&mut fixture.memory, 0x2100, "example.test");
                barrier.wait();
                let session = fixture.open_session();
                let connection = fixture.call("WinHttpConnect", &[session, 0x2100, 80, 0]);
                assert_ne!(session, 0);
                assert_ne!(connection, 0);
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(state.open_handle_count(), 8);
        let handles = state.winhttp.lock().unwrap();
        assert_eq!(handles.open_count(), 8);
    }
}
