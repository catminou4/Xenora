// SPDX-License-Identifier: GPL-3.0-or-later

//! Synthetic interpreter and DBT timings.
//!
//! Set `XENORA_DBT_BENCH_JSON=<path>` to also write the results as one JSON
//! document (format version 1). The text output does not change. Values the
//! benchmark cannot determine, such as the git revision outside a checkout,
//! are written as `null`. The benchmark cannot see build settings such as LTO,
//! so `XENORA_DBT_BENCH_BUILD_LABEL=<text>` records a label of your choosing
//! for them.
//!
//! Set `XENORA_DBT_BENCH_WORKLOAD` to `add-chain` (the default),
//! `branch-loop` or `checked-memory` to select the original synthetic guest
//! workload. Set `XENORA_DBT_BENCH_COMPARE=<baseline.json>,<candidate.json>` to compare
//! two such documents instead of running the benchmark. The comparison is
//! refused, with exit status 1, unless both runs used the same host, workload,
//! build settings and sample counts.
//!
//! The benchmark target disables Cargo's test harness. To run the percentile
//! unit tests, build with
//! `cargo rustc --locked -p xenora-loader --bench dbt_bench --message-format=json -- --test`
//! and run the executable named by the benchmark's compiler-artifact message.

use std::hint::black_box;
use std::time::{Duration, Instant};
use xenora_loader::cpu::{CpuState, ExecutionResult, ExitReason, Interpreter, Register};
use xenora_loader::dbt::{DbtEngine, DbtMetrics};
use xenora_loader::memory::MemoryPermissions;
use xenora_loader::pe::{
    IMAGE_FILE_MACHINE_AMD64, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, PE32_PLUS_MAGIC, ParsedPe,
};

fn build_pe(code: &[u8]) -> Vec<u8> {
    let mut pe = Vec::new();
    pe.extend_from_slice(b"MZ");
    pe.resize(0x3C, 0);
    let e_lfanew: u32 = 0x80;
    pe.extend_from_slice(&e_lfanew.to_le_bytes());
    pe.resize(0x80, 0);
    pe.extend_from_slice(b"PE\0\0");
    pe.extend_from_slice(&IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    let num_sections: u16 = 1;
    pe.extend_from_slice(&num_sections.to_le_bytes());
    pe.extend_from_slice(&0x67300000u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    let opt_hdr_size: u16 = 112;
    pe.extend_from_slice(&opt_hdr_size.to_le_bytes());
    pe.extend_from_slice(&0x0022u16.to_le_bytes());
    pe.extend_from_slice(&PE32_PLUS_MAGIC.to_le_bytes());
    pe.push(14);
    pe.push(0);
    pe.extend_from_slice(&(code.len() as u32).to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    let entry_point_rva: u32 = 0x1000;
    pe.extend_from_slice(&entry_point_rva.to_le_bytes());
    pe.extend_from_slice(&0x1000u32.to_le_bytes());
    let image_base: u64 = 0x0000000140000000;
    pe.extend_from_slice(&image_base.to_le_bytes());
    pe.extend_from_slice(&0x1000u32.to_le_bytes());
    pe.extend_from_slice(&0x200u32.to_le_bytes());
    pe.extend_from_slice(&6u16.to_le_bytes());
    pe.extend_from_slice(&0u16.to_le_bytes());
    pe.extend_from_slice(&0u16.to_le_bytes());
    pe.extend_from_slice(&0u16.to_le_bytes());
    pe.extend_from_slice(&6u16.to_le_bytes());
    pe.extend_from_slice(&0u16.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    let size_of_image = 0x1000 + ((code.len() as u32 + 0xfff) & !0xfff);
    pe.extend_from_slice(&size_of_image.to_le_bytes());
    pe.extend_from_slice(&0x200u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    pe.extend_from_slice(&3u16.to_le_bytes());
    pe.extend_from_slice(&0x8160u16.to_le_bytes());
    pe.extend_from_slice(&0x100000u64.to_le_bytes());
    pe.extend_from_slice(&0x1000u64.to_le_bytes());
    pe.extend_from_slice(&0x100000u64.to_le_bytes());
    pe.extend_from_slice(&0x1000u64.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    let mut sec_name = [0u8; 8];
    sec_name[..5].copy_from_slice(b".text");
    pe.extend_from_slice(&sec_name);
    pe.extend_from_slice(&(code.len() as u32).to_le_bytes());
    pe.extend_from_slice(&0x1000u32.to_le_bytes());
    let raw_size = (code.len() as u32 + 0x1ff) & !0x1ff;
    pe.extend_from_slice(&raw_size.to_le_bytes());
    pe.extend_from_slice(&0x200u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    pe.extend_from_slice(&0u32.to_le_bytes());
    pe.extend_from_slice(&0u16.to_le_bytes());
    pe.extend_from_slice(&0u16.to_le_bytes());
    pe.extend_from_slice(&(IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE).to_le_bytes());
    pe.resize(0x200, 0);
    pe.extend_from_slice(code);
    pe.resize(0x200 + raw_size as usize, 0);
    pe
}

const ADDS: usize = 10_000;
const BRANCH_ITERATIONS: usize = 5_000;
const MEMORY_ITERATIONS: usize = 2_000;
const BUDGET: usize = 20_000;
const STACK_TOP: u64 = 0x2000_8000;
const DATA_BASE: u64 = 0x2000_4000;

/// One original synthetic guest workload. The instruction count includes HLT.
#[derive(Debug)]
struct Workload {
    name: &'static str,
    code: Vec<u8>,
    instructions: usize,
    final_rax: u64,
}

fn add_chain() -> Workload {
    let mut code = vec![0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0]; // mov rax, 0
    for _ in 0..ADDS {
        code.extend_from_slice(&[0x48, 0x83, 0xC0, 0x01]); // add rax, 1
    }
    code.push(0xF4); // hlt
    Workload {
        name: "add-chain",
        code,
        instructions: ADDS + 2,
        final_rax: ADDS as u64,
    }
}

fn branch_loop() -> Workload {
    let mut code = vec![0x48, 0xB8];
    code.extend_from_slice(&(BRANCH_ITERATIONS as u64).to_le_bytes()); // mov rax, iterations
    let sub_at = code.len();
    code.extend_from_slice(&[0x48, 0x83, 0xE8, 0x01]); // sub rax, 1
    let after_jump = code.len() + 2;
    let displacement = (sub_at as isize - after_jump as isize) as i8;
    code.extend_from_slice(&[0x75, displacement as u8]); // jne the subtraction
    code.push(0xF4); // hlt
    Workload {
        name: "branch-loop",
        code,
        instructions: 1 + BRANCH_ITERATIONS * 2 + 1,
        final_rax: 0,
    }
}

fn checked_memory() -> Workload {
    let mut code = vec![0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0]; // mov rax, 0
    code.extend_from_slice(&[0x48, 0xB9]); // mov rcx, DATA_BASE
    code.extend_from_slice(&DATA_BASE.to_le_bytes());
    for _ in 0..MEMORY_ITERATIONS {
        code.extend_from_slice(&[0x89, 0x01]); // mov [rcx], eax
        code.extend_from_slice(&[0x8B, 0x01]); // mov eax, [rcx]
        code.extend_from_slice(&[0x83, 0xC0, 0x01]); // add eax, 1
        code.extend_from_slice(&[0x48, 0x83, 0xC1, 0x04]); // add rcx, 4
    }
    code.push(0xF4); // hlt
    Workload {
        name: "checked-memory",
        code,
        instructions: 2 + MEMORY_ITERATIONS * 4 + 1,
        final_rax: MEMORY_ITERATIONS as u64,
    }
}

fn workload(name: Option<&str>) -> Result<Workload, String> {
    match name.unwrap_or("add-chain") {
        "add-chain" => Ok(add_chain()),
        "branch-loop" => Ok(branch_loop()),
        "checked-memory" => Ok(checked_memory()),
        other => Err(format!(
            "unknown workload `{other}`; expected add-chain, branch-loop or checked-memory"
        )),
    }
}

fn check(result: ExecutionResult, workload: &Workload) {
    assert_eq!(result.exit_reason, ExitReason::Halt);
    assert_eq!(result.instructions_executed, workload.instructions);
    assert_eq!(result.final_state.get(Register::Rax), workload.final_rax);
}

fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    // Interpolate between adjacent values at rank (n - 1) * fraction.
    let rank = (sorted.len() - 1) as f64 * fraction;
    let lower = sorted[rank.floor() as usize];
    let upper = sorted[rank.ceil() as usize];
    lower + (upper - lower).mul_f64(rank.fract())
}

/// Order statistics of one timing series, shared by the text and JSON reports.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Summary {
    samples: usize,
    median: Duration,
    min: Duration,
    max: Duration,
    p95: Duration,
    p99: Duration,
    iqr: Duration,
    mips: f64,
}

fn summarize(samples: &mut [Duration], guest_instructions: usize) -> Summary {
    samples.sort_unstable();
    let median = percentile(samples, 0.50);
    Summary {
        samples: samples.len(),
        median,
        min: samples[0],
        max: samples[samples.len() - 1],
        p95: percentile(samples, 0.95),
        p99: percentile(samples, 0.99),
        iqr: percentile(samples, 0.75) - percentile(samples, 0.25),
        mips: guest_instructions as f64 / median.as_secs_f64() / 1_000_000.0,
    }
}

fn report_line(label: &str, s: &Summary) -> String {
    format!(
        "{label}: {} samples, median {:?}, min {:?}, max {:?}, p95 {:?}, p99 {:?}, IQR {:?}, median {:.2} MIPS",
        s.samples, s.median, s.min, s.max, s.p95, s.p99, s.iqr, s.mips
    )
}

/// Facts about the run that are not timings. `None` means the benchmark could not
/// determine the value.
struct RunInfo {
    unix_time: Option<u64>,
    cpu: Option<String>,
    os: &'static str,
    os_version: Option<String>,
    arch: &'static str,
    revision: Option<String>,
    dirty: Option<bool>,
    debug_assertions: bool,
    build_label: Option<String>,
    guest_code_bytes: usize,
    warmup: usize,
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn run_info(guest_code_bytes: usize, warmup: usize) -> RunInfo {
    let non_empty = |value: Option<String>| value.filter(|v| !v.is_empty());
    RunInfo {
        unix_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs()),
        cpu: non_empty(command_output(
            "sysctl",
            &["-n", "machdep.cpu.brand_string"],
        )),
        os: std::env::consts::OS,
        os_version: non_empty(command_output("sw_vers", &["-productVersion"])),
        arch: std::env::consts::ARCH,
        revision: non_empty(command_output("git", &["rev-parse", "HEAD"])),
        dirty: command_output("git", &["status", "--porcelain", "--untracked-files=no"])
            .map(|status| !status.is_empty()),
        debug_assertions: cfg!(debug_assertions),
        build_label: non_empty(std::env::var("XENORA_DBT_BENCH_BUILD_LABEL").ok()),
        guest_code_bytes,
        warmup,
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_option<T>(value: Option<T>, render: impl FnOnce(T) -> String) -> String {
    value.map_or_else(|| "null".to_owned(), render)
}

/// JSON has no infinity or NaN, so a non-finite rate (a zero median) is `null`.
fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_owned()
    }
}

fn json_object(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields
        .iter()
        .map(|(key, value)| format!("{}:{value}", json_string(key)))
        .collect();
    format!("{{{}}}", body.join(","))
}

fn json_summary(s: &Summary) -> String {
    json_object(&[
        ("samples", s.samples.to_string()),
        ("median_ns", s.median.as_nanos().to_string()),
        ("min_ns", s.min.as_nanos().to_string()),
        ("max_ns", s.max.as_nanos().to_string()),
        ("p95_ns", s.p95.as_nanos().to_string()),
        ("p99_ns", s.p99.as_nanos().to_string()),
        ("iqr_ns", s.iqr.as_nanos().to_string()),
        ("median_mips", json_f64(s.mips)),
    ])
}

fn json_document(
    info: &RunInfo,
    workload: &Workload,
    interpreter: &Summary,
    cold: &Summary,
    warm: &Summary,
    metrics: &DbtMetrics,
) -> String {
    let string = |value: &String| json_string(value);
    json_object(&[
        ("format_version", "1".to_owned()),
        ("benchmark", json_string("dbt_bench")),
        ("unix_time", json_option(info.unix_time, |t| t.to_string())),
        (
            "host",
            json_object(&[
                ("cpu", json_option(info.cpu.as_ref(), string)),
                ("os", json_string(info.os)),
                ("os_version", json_option(info.os_version.as_ref(), string)),
                ("arch", json_string(info.arch)),
            ]),
        ),
        (
            "source",
            json_object(&[
                ("revision", json_option(info.revision.as_ref(), string)),
                ("dirty", json_option(info.dirty, |d| d.to_string())),
            ]),
        ),
        (
            "build",
            json_object(&[
                ("debug_assertions", info.debug_assertions.to_string()),
                ("label", json_option(info.build_label.as_ref(), string)),
            ]),
        ),
        (
            "workload",
            json_object(&[
                ("name", json_string(workload.name)),
                ("guest_instructions", workload.instructions.to_string()),
                ("guest_code_bytes", info.guest_code_bytes.to_string()),
                ("instruction_budget", BUDGET.to_string()),
                ("warmup_rounds", info.warmup.to_string()),
            ]),
        ),
        (
            "timings",
            json_object(&[
                ("interpreter", json_summary(interpreter)),
                ("cold_dbt", json_summary(cold)),
                ("warm_dbt", json_summary(warm)),
            ]),
        ),
        (
            "cold_dbt_metrics",
            json_object(&[
                ("blocks_compiled", metrics.blocks_compiled.to_string()),
                ("cache_hits", metrics.cache_hits.to_string()),
                ("cache_misses", metrics.cache_misses.to_string()),
                (
                    "native_bytes_emitted",
                    metrics.native_bytes_emitted.to_string(),
                ),
                ("dbt_dispatches", metrics.dbt_dispatches.to_string()),
                (
                    "instructions_executed",
                    metrics.instructions_executed.to_string(),
                ),
                (
                    "interpreter_fallbacks",
                    metrics.interpreter_fallbacks.to_string(),
                ),
                ("service_calls", metrics.service_calls.to_string()),
            ]),
        ),
    ])
}

/// A parsed JSON value. Numbers keep their source text, so integers compare exactly.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The value at a dotted path such as `host.cpu`, if every step is an object key.
    fn get(&self, path: &str) -> Option<&Json> {
        path.split('.').try_fold(self, |value, key| match value {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        })
    }
}

const MAX_JSON_DEPTH: usize = 32;

/// Reads one JSON document (RFC 8259) and rejects anything after it.
fn parse_json(text: &str) -> Result<Json, String> {
    let mut reader = JsonReader {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = reader.value(0)?;
    reader.skip_whitespace();
    if reader.pos != reader.bytes.len() {
        return reader.error("unexpected data after the document");
    }
    Ok(value)
}

struct JsonReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl JsonReader<'_> {
    fn error<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            self.error(&format!("expected '{}'", byte as char))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_JSON_DEPTH {
            return self.error("nesting too deep");
        }
        self.skip_whitespace();
        match self.peek() {
            None => self.error("unexpected end of input"),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.error("unexpected character"),
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            self.error("invalid literal")
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return self.error("expected ',' or ']'"),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(fields));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            if fields.iter().any(|(k, _)| *k == key) {
                return self.error(&format!("duplicate key `{key}`"));
            }
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.value(depth + 1)?;
            fields.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(fields));
                }
                _ => return self.error("expected ',' or '}'"),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return self.error("invalid number"),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if self.digits() == 0 {
                return self.error("invalid number");
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return self.error("invalid number");
            }
        }
        // Only ASCII digits and signs were consumed.
        Ok(Json::Number(
            String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned(),
        ))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self.bytes.get(self.pos..self.pos + 4);
        let value = digits
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok());
        match value {
            Some(value) if digits.is_some_and(|d| d.iter().all(u8::is_ascii_hexdigit)) => {
                self.pos += 4;
                Ok(value)
            }
            _ => self.error("invalid \\u escape"),
        }
    }

    fn unicode_escape(&mut self) -> Result<char, String> {
        let first = self.hex4()?;
        let code = match first {
            0xD800..=0xDBFF => {
                if !self.bytes[self.pos..].starts_with(b"\\u") {
                    return self.error("unpaired surrogate");
                }
                self.pos += 2;
                let second = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&second) {
                    return self.error("unpaired surrogate");
                }
                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
            }
            0xDC00..=0xDFFF => return self.error("unpaired surrogate"),
            code => code,
        };
        char::from_u32(code).map_or_else(|| self.error("invalid code point"), Ok)
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            // Copy the run up to the next quote, backslash or control byte. Those
            // are ASCII, so the run never ends inside a multi-byte character.
            let start = self.pos;
            while matches!(self.peek(), Some(b) if b != b'"' && b != b'\\' && b >= 0x20) {
                self.pos += 1;
            }
            match std::str::from_utf8(&self.bytes[start..self.pos]) {
                Ok(run) => out.push_str(run),
                Err(_) => return self.error("invalid UTF-8"),
            }
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let escaped = self.peek();
                    self.pos += 1;
                    out.push(match escaped {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => self.unicode_escape()?,
                        _ => {
                            self.pos -= 1;
                            return self.error("invalid escape");
                        }
                    });
                }
                Some(_) => return self.error("control character in string"),
                None => return self.error("unterminated string"),
            }
        }
    }
}

/// Fields that must be equal for two runs to be compared. A field listed in
/// `OPTIONAL_FIELDS` counts as `null` when a document does not have it.
const MATCHING_FIELDS: [&str; 15] = [
    "format_version",
    "benchmark",
    "host.cpu",
    "host.os",
    "host.arch",
    "build.debug_assertions",
    "build.label",
    "workload.name",
    "workload.guest_instructions",
    "workload.guest_code_bytes",
    "workload.instruction_budget",
    "workload.warmup_rounds",
    "timings.interpreter.samples",
    "timings.cold_dbt.samples",
    "timings.warm_dbt.samples",
];

/// Older version 1 documents predate the build label.
const OPTIONAL_FIELDS: [&str; 1] = ["build.label"];

/// Host fields that must be known: two unknown CPUs are not the same CPU.
const KNOWN_FIELDS: [&str; 3] = ["host.cpu", "host.os", "host.arch"];

const SERIES: [(&str, &str); 3] = [
    ("interpreter", "Interpreter"),
    ("cold_dbt", "Cold DBT"),
    ("warm_dbt", "Warm DBT"),
];

const COUNTERS: [&str; 8] = [
    "blocks_compiled",
    "cache_hits",
    "cache_misses",
    "native_bytes_emitted",
    "dbt_dispatches",
    "instructions_executed",
    "interpreter_fallbacks",
    "service_calls",
];

fn field<'a>(doc: &'a Json, path: &str) -> Result<&'a Json, String> {
    match doc.get(path) {
        Some(value) => Ok(value),
        None if OPTIONAL_FIELDS.contains(&path) => Ok(&Json::Null),
        None => Err(format!("missing field `{path}`")),
    }
}

fn integer(doc: &Json, path: &str) -> Result<u128, String> {
    match field(doc, path)? {
        Json::Number(text) => text
            .parse()
            .map_err(|_| format!("`{path}` is not a non-negative integer")),
        _ => Err(format!("`{path}` is not a number")),
    }
}

fn text(doc: &Json, path: &str) -> String {
    match doc.get(path) {
        Some(Json::String(value)) => value.clone(),
        Some(Json::Bool(value)) => value.to_string(),
        Some(Json::Number(value)) => value.clone(),
        // The build label is optional, so its absence is not an unknown value.
        _ if OPTIONAL_FIELDS.contains(&path) => "none".to_owned(),
        _ => "unknown".to_owned(),
    }
}

fn duration(ns: u128) -> Duration {
    Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX))
}

fn change(before: u128, after: u128) -> String {
    if before == 0 {
        return "n/a".to_owned();
    }
    format!(
        "{:+.2}%",
        (after as f64 - before as f64) / before as f64 * 100.0
    )
}

fn source(doc: &Json) -> String {
    let dirty = match doc.get("source.dirty") {
        Some(Json::Bool(true)) => " with local changes",
        Some(Json::Bool(false)) => "",
        _ => ", local changes unknown",
    };
    format!("{}{dirty}", text(doc, "source.revision"))
}

/// Compares two exported documents. Returns the report lines, or why the runs
/// cannot be compared.
fn compare_documents(baseline: &Json, candidate: &Json) -> Result<Vec<String>, String> {
    for doc in [baseline, candidate] {
        if field(doc, "format_version")? != &Json::Number("1".to_owned()) {
            return Err("only format version 1 is supported".to_owned());
        }
    }
    for path in KNOWN_FIELDS {
        for (name, doc) in [("baseline", baseline), ("candidate", candidate)] {
            if field(doc, path)? == &Json::Null {
                return Err(format!(
                    "`{path}` is unknown in the {name}, so the hosts cannot be shown to match"
                ));
            }
        }
    }
    for path in MATCHING_FIELDS {
        if field(baseline, path)? != field(candidate, path)? {
            return Err(format!(
                "`{path}` differs: baseline {}, candidate {}",
                text(baseline, path),
                text(candidate, path)
            ));
        }
    }

    let mut lines = vec![
        format!("Baseline:  {}", source(baseline)),
        format!("Candidate: {}", source(candidate)),
        format!(
            "Host: {}, {} {}; debug assertions {}; build label {}",
            text(baseline, "host.cpu"),
            text(baseline, "host.os"),
            text(baseline, "host.arch"),
            text(baseline, "build.debug_assertions"),
            text(baseline, "build.label"),
        ),
    ];
    if field(baseline, "host.os_version")? != field(candidate, "host.os_version")? {
        lines.push(format!(
            "Note: OS version differs: baseline {}, candidate {}",
            text(baseline, "host.os_version"),
            text(candidate, "host.os_version")
        ));
    }
    for (key, label) in SERIES {
        let path = |stat: &str| format!("timings.{key}.{stat}");
        let before = integer(baseline, &path("median_ns"))?;
        let after = integer(candidate, &path("median_ns"))?;
        let spread = integer(baseline, &path("iqr_ns"))?.max(integer(candidate, &path("iqr_ns"))?);
        let verdict = if before.abs_diff(after) > spread {
            "larger than"
        } else {
            "within"
        };
        lines.push(format!(
            "{label}: median {:?} -> {:?} ({}), {verdict} the larger IQR {:?}",
            duration(before),
            duration(after),
            change(before, after),
            duration(spread)
        ));
    }
    let mut counters_changed = false;
    for counter in COUNTERS {
        let path = format!("cold_dbt_metrics.{counter}");
        let (before, after) = (integer(baseline, &path)?, integer(candidate, &path)?);
        if before != after {
            counters_changed = true;
            lines.push(format!(
                "Cold DBT {counter}: {before} -> {after} ({})",
                change(before, after)
            ));
        }
    }
    if !counters_changed {
        lines.push("Cold DBT counters are identical".to_owned());
    }
    Ok(lines)
}

fn compare_files(spec: &str) -> Result<Vec<String>, String> {
    let (baseline, candidate) = spec
        .split_once(',')
        .ok_or("expected <baseline.json>,<candidate.json>")?;
    let read = |path: &str| {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        parse_json(&text).map_err(|e| format!("{path}: {e}"))
    };
    compare_documents(&read(baseline)?, &read(candidate)?)
}

fn main() {
    if let Ok(spec) = std::env::var("XENORA_DBT_BENCH_COMPARE") {
        match compare_files(&spec) {
            Ok(lines) => lines.iter().for_each(|line| println!("{line}")),
            Err(reason) => {
                eprintln!("dbt_bench: cannot compare: {reason}");
                std::process::exit(1);
            }
        }
        return;
    }

    let sample_count = std::env::var("XENORA_DBT_BENCH_SAMPLES")
        .map_or(101, |value| value.parse::<usize>().expect("sample count"));
    assert!(sample_count > 0);
    let warmup_count = std::env::var("XENORA_DBT_BENCH_WARMUP")
        .map_or(10, |value| value.parse::<usize>().expect("warmup count"));

    let workload = match workload(std::env::var("XENORA_DBT_BENCH_WORKLOAD").as_deref().ok()) {
        Ok(workload) => workload,
        Err(reason) => {
            eprintln!("dbt_bench: {reason}");
            std::process::exit(2);
        }
    };

    let pe_bytes = build_pe(&workload.code);
    let parsed = ParsedPe::parse(&pe_bytes).unwrap();
    let mut loaded = parsed.load_into_memory(&pe_bytes).unwrap();
    loaded
        .memory
        .map_region(0x2000_0000, 0x10000, MemoryPermissions::READ_WRITE)
        .unwrap();
    let initial_cpu = CpuState::new(loaded.entry_point, STACK_TOP);
    let mut interpreter = Vec::with_capacity(sample_count);
    let mut cold = Vec::with_capacity(sample_count);
    let mut warm = Vec::with_capacity(sample_count);
    let mut cold_metrics = None;

    for record in
        std::iter::repeat_n(false, warmup_count).chain(std::iter::repeat_n(true, sample_count))
    {
        let mut interp = Interpreter::new(initial_cpu.clone(), &mut loaded.memory, BUDGET);
        let start = Instant::now();
        let result = black_box(interp.run());
        let interpreter_time = start.elapsed();
        check(result, &workload);

        let mut dbt = DbtEngine::new(4 * 1024 * 1024).unwrap();
        let mut cpu = initial_cpu.clone();
        let start = Instant::now();
        let result = black_box(dbt.run(&mut cpu, &mut loaded.memory, BUDGET));
        let cold_time = start.elapsed();
        check(result, &workload);
        assert!(dbt.metrics.blocks_compiled > 0);
        assert_eq!(dbt.metrics.cache_misses, dbt.metrics.blocks_compiled);
        let metrics = dbt.metrics;

        let mut cpu = initial_cpu.clone();
        let start = Instant::now();
        let result = black_box(dbt.run(&mut cpu, &mut loaded.memory, BUDGET));
        let warm_time = start.elapsed();
        check(result, &workload);
        assert_eq!(dbt.metrics.blocks_compiled, metrics.blocks_compiled);
        assert!(dbt.metrics.cache_hits > metrics.cache_hits);
        assert_eq!(dbt.metrics.cache_misses, metrics.cache_misses);

        if record {
            interpreter.push(interpreter_time);
            cold.push(cold_time);
            warm.push(warm_time);
            cold_metrics = Some(metrics);
        }
    }

    println!(
        "{} workload: {} guest instructions per sample, final RAX={}",
        workload.name, workload.instructions, workload.final_rax
    );
    println!("{warmup_count} warmup rounds excluded (XENORA_DBT_BENCH_WARMUP, default 10)");
    println!(
        "Cold DBT includes translation with a fresh engine; warm DBT reuses that engine's cache."
    );
    println!(
        "Percentiles describe this synthetic workload on this host; tail estimates vary with sample count."
    );
    println!("Even sample counts use the midpoint of the two middle values for the median.");
    let interpreter = summarize(&mut interpreter, workload.instructions);
    let cold = summarize(&mut cold, workload.instructions);
    let warm = summarize(&mut warm, workload.instructions);
    println!("{}", report_line("Interpreter", &interpreter));
    println!("{}", report_line("Cold DBT", &cold));
    println!("{}", report_line("Warm DBT", &warm));
    let metrics = cold_metrics.unwrap();
    println!(
        "Cold DBT compiled {} blocks and emitted {} native bytes in {} dispatches",
        metrics.blocks_compiled, metrics.native_bytes_emitted, metrics.dbt_dispatches
    );

    if let Some(path) = std::env::var_os("XENORA_DBT_BENCH_JSON") {
        let info = run_info(workload.code.len(), warmup_count);
        let document = json_document(&info, &workload, &interpreter, &cold, &warm, &metrics);
        std::fs::write(&path, document + "\n").expect("write XENORA_DBT_BENCH_JSON");
        println!("JSON results written to {}", path.to_string_lossy());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn single_sample_has_no_spread() {
        use super::{Duration, percentile};

        let samples = [Duration::from_micros(50)];
        for fraction in [0.25, 0.50, 0.75, 0.95, 0.99] {
            assert_eq!(percentile(&samples, fraction), samples[0]);
        }
    }

    #[test]
    fn percentiles_interpolate_across_even_samples() {
        use super::{Duration, percentile};

        let samples = [10, 20, 30, 100].map(Duration::from_micros);
        assert_eq!(percentile(&samples, 0.50), Duration::from_micros(25));
        assert_eq!(percentile(&samples, 0.95), Duration::from_nanos(89_500));
        assert_eq!(percentile(&samples, 0.99), Duration::from_nanos(97_900));
        assert_eq!(
            percentile(&samples, 0.75) - percentile(&samples, 0.25),
            Duration::from_nanos(30_000)
        );
    }

    #[test]
    fn median_uses_the_middle_of_odd_samples() {
        use super::{Duration, percentile};

        let samples = [10, 20, 100].map(Duration::from_micros);
        assert_eq!(percentile(&samples, 0.50), Duration::from_micros(20));
    }

    #[test]
    fn json_strings_escape_quotes_backslashes_and_controls() {
        use super::json_string;

        assert_eq!(json_string("plain"), r#""plain""#);
        assert_eq!(json_string(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(json_string("l1\nl2\r\t"), r#""l1\nl2\r\t""#);
        assert_eq!(json_string("\u{1}\u{1f}"), r#""\u0001\u001f""#);
        // Non-ASCII text needs no escape in UTF-8 JSON.
        assert_eq!(json_string("Zażółć"), "\"Zażółć\"");
    }

    #[test]
    fn unknown_values_and_non_finite_rates_are_null() {
        use super::{json_f64, json_option};

        assert_eq!(json_option(None::<u64>, |v| v.to_string()), "null");
        assert_eq!(json_option(Some(7_u64), |v| v.to_string()), "7");
        assert_eq!(json_f64(f64::INFINITY), "null");
        assert_eq!(json_f64(f64::NAN), "null");
        assert_eq!(json_f64(12.5), "12.5");
    }

    #[test]
    fn document_has_every_field_with_the_summary_values() {
        fn fixed_summary(median_us: u64) -> super::Summary {
            use super::Duration;
            super::Summary {
                samples: 3,
                median: Duration::from_micros(median_us),
                min: Duration::from_micros(10),
                max: Duration::from_micros(100),
                p95: Duration::from_nanos(92_000),
                p99: Duration::from_nanos(98_400),
                iqr: Duration::from_micros(45),
                mips: 500.25,
            }
        }

        fn fixed_info() -> super::RunInfo {
            super::RunInfo {
                unix_time: Some(1_700_000_000),
                cpu: Some("Test \"CPU\"\\1".to_owned()),
                os: "macos",
                os_version: None,
                arch: "aarch64",
                revision: Some("0123abcd".to_owned()),
                dirty: Some(false),
                debug_assertions: false,
                build_label: Some("fat-lto".to_owned()),
                guest_code_bytes: 40_011,
                warmup: 10,
            }
        }

        use super::{DbtMetrics, add_chain, json_document};

        let metrics = DbtMetrics {
            blocks_compiled: 626,
            blocks_invalidated: 0,
            cache_hits: 0,
            cache_misses: 626,
            native_bytes_emitted: 245_168,
            dbt_dispatches: 626,
            instructions_executed: 10_002,
            interpreter_fallbacks: 1,
            service_calls: 0,
        };
        let workload = add_chain();
        let document = json_document(
            &fixed_info(),
            &workload,
            &fixed_summary(20),
            &fixed_summary(30),
            &fixed_summary(40),
            &metrics,
        );
        let summary = |median: &str| {
            format!(
                r#"{{"samples":3,"median_ns":{median},"min_ns":10000,"max_ns":100000,"p95_ns":92000,"p99_ns":98400,"iqr_ns":45000,"median_mips":500.25}}"#
            )
        };
        let expected = format!(
            concat!(
                r#"{{"format_version":1,"benchmark":"dbt_bench","unix_time":1700000000,"#,
                r#""host":{{"cpu":"Test \"CPU\"\\1","os":"macos","os_version":null,"arch":"aarch64"}},"#,
                r#""source":{{"revision":"0123abcd","dirty":false}},"#,
                r#""build":{{"debug_assertions":false,"label":"fat-lto"}},"#,
                r#""workload":{{"name":"add-chain","guest_instructions":{},"guest_code_bytes":40011,"instruction_budget":20000,"warmup_rounds":10}},"#,
                r#""timings":{{"interpreter":{},"cold_dbt":{},"warm_dbt":{}}},"#,
                r#""cold_dbt_metrics":{{"blocks_compiled":626,"cache_hits":0,"cache_misses":626,"native_bytes_emitted":245168,"dbt_dispatches":626,"instructions_executed":10002,"interpreter_fallbacks":1,"service_calls":0}}}}"#,
            ),
            workload.instructions,
            summary("20000"),
            summary("30000"),
            summary("40000"),
        );
        assert_eq!(document, expected);
    }

    #[test]
    fn text_report_keeps_its_format() {
        fn fixed_summary(median_us: u64) -> super::Summary {
            use super::Duration;
            super::Summary {
                samples: 3,
                median: Duration::from_micros(median_us),
                min: Duration::from_micros(10),
                max: Duration::from_micros(100),
                p95: Duration::from_nanos(92_000),
                p99: Duration::from_nanos(98_400),
                iqr: Duration::from_micros(45),
                mips: 500.25,
            }
        }

        use super::report_line;

        assert_eq!(
            report_line("Cold DBT", &fixed_summary(20)),
            "Cold DBT: 3 samples, median 20µs, min 10µs, max 100µs, p95 92µs, p99 98.4µs, IQR 45µs, median 500.25 MIPS"
        );
    }

    #[test]
    fn summary_matches_the_percentile_helper() {
        use super::{Duration, summarize};

        let mut samples = [100, 10, 30, 20].map(Duration::from_micros);
        let s = summarize(&mut samples, 10_002);
        assert_eq!(s.samples, 4);
        assert_eq!(s.median, Duration::from_micros(25));
        assert_eq!(s.min, Duration::from_micros(10));
        assert_eq!(s.max, Duration::from_micros(100));
        assert_eq!(s.p95, Duration::from_nanos(89_500));
        assert_eq!(s.p99, Duration::from_nanos(97_900));
        assert_eq!(s.iqr, Duration::from_micros(30));
        assert_eq!(s.mips, 10_002_f64 / 25e-6 / 1e6);
    }

    // `cargo bench` builds this target with `--cfg test` but without `--test`,
    // so this module is compiled with its #[test] functions removed and shared
    // helpers look unused there.
    #[allow(dead_code)]
    fn comparable_document() -> super::Json {
        super::parse_json(concat!(
            r#"{"format_version":1,"benchmark":"dbt_bench","unix_time":1,"#,
            r#""host":{"cpu":"Apple M5","os":"macos","os_version":"27.2.1","arch":"aarch64"},"#,
            r#""source":{"revision":"aaaa","dirty":false},"#,
            r#""build":{"debug_assertions":false,"label":null},"#,
            r#""workload":{"name":"add_rax_chain","guest_instructions":10002,"guest_code_bytes":40011,"instruction_budget":20000,"warmup_rounds":10},"#,
            r#""timings":{"#,
            r#""interpreter":{"samples":101,"median_ns":900000,"min_ns":1,"max_ns":2,"p95_ns":1,"p99_ns":1,"iqr_ns":20000,"median_mips":11.1},"#,
            r#""cold_dbt":{"samples":101,"median_ns":1100000,"min_ns":1,"max_ns":2,"p95_ns":1,"p99_ns":1,"iqr_ns":40000,"median_mips":9.1},"#,
            r#""warm_dbt":{"samples":101,"median_ns":22000,"min_ns":1,"max_ns":2,"p95_ns":1,"p99_ns":1,"iqr_ns":500,"median_mips":454.6}},"#,
            r#""cold_dbt_metrics":{"blocks_compiled":626,"cache_hits":0,"cache_misses":626,"native_bytes_emitted":245168,"dbt_dispatches":626,"instructions_executed":10002,"interpreter_fallbacks":0,"service_calls":0}}"#,
        ))
        .expect("fixture parses")
    }

    /// Replaces the value at a dotted path, which must exist except for its last key.
    #[allow(dead_code)]
    fn set(doc: &mut super::Json, path: &str, value: super::Json) {
        let (parent, key) = path.rsplit_once('.').map_or(("", path), |(p, k)| (p, k));
        let mut node = doc;
        for step in parent.split('.').filter(|s| !s.is_empty()) {
            let super::Json::Object(fields) = node else {
                panic!("{step} is not in an object")
            };
            node = &mut fields.iter_mut().find(|(k, _)| k == step).expect(step).1;
        }
        let super::Json::Object(fields) = node else {
            panic!("{path} has no parent object")
        };
        match fields.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value,
            None => fields.push((key.to_owned(), value)),
        }
    }

    #[test]
    fn reader_accepts_json_values() {
        use super::{Json, parse_json};

        let doc = parse_json(
            r#" { "a" : [1, -0.5, 2e3, 3E-2, true, false, null, {}, []], "s": "q\"b\\s\/n\n\té😀 Zażółć" } "#,
        )
        .unwrap();
        assert_eq!(
            doc.get("a"),
            Some(&Json::Array(vec![
                Json::Number("1".to_owned()),
                Json::Number("-0.5".to_owned()),
                Json::Number("2e3".to_owned()),
                Json::Number("3E-2".to_owned()),
                Json::Bool(true),
                Json::Bool(false),
                Json::Null,
                Json::Object(vec![]),
                Json::Array(vec![]),
            ]))
        );
        assert_eq!(
            doc.get("s"),
            Some(&Json::String("q\"b\\s/n\n\té😀 Zażółć".to_owned()))
        );
        assert_eq!(doc.get("a.b"), None);
        assert_eq!(doc.get("missing"), None);
    }

    #[test]
    fn reader_rejects_malformed_json() {
        use super::parse_json;

        let deep = "[".repeat(40) + &"]".repeat(40);
        for bad in [
            "",
            "{",
            r#"{"a":1,}"#,
            r#"{"a" 1}"#,
            r#"{"a":1,"a":2}"#,
            "[1 2]",
            "01",
            "1.",
            "-",
            "1e",
            "+1",
            "tru",
            r#""open"#,
            "\"tab\tinside\"",
            r#""\x""#,
            r#""\u12g4""#,
            r#""\u+123""#,
            r#""\ud83d""#,
            r#""\ude00""#,
            r#""\ud83dA""#,
            "{} {}",
            deep.as_str(),
        ] {
            assert!(parse_json(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn exported_document_reads_back() {
        use super::{DbtMetrics, Duration, Json, RunInfo, Summary, json_document, parse_json};

        let summary = Summary {
            samples: 7,
            median: Duration::from_nanos(1_234_567),
            min: Duration::from_nanos(1),
            max: Duration::from_nanos(9),
            p95: Duration::from_nanos(8),
            p99: Duration::from_nanos(9),
            iqr: Duration::from_nanos(3),
            mips: 8.25,
        };
        let info = RunInfo {
            unix_time: None,
            cpu: Some("CPU \"x\" \\ \u{1}".to_owned()),
            os: "macos",
            os_version: None,
            arch: "aarch64",
            revision: None,
            dirty: Some(true),
            debug_assertions: true,
            build_label: Some("é label".to_owned()),
            guest_code_bytes: 5,
            warmup: 2,
        };
        let metrics = DbtMetrics {
            native_bytes_emitted: 42,
            ..DbtMetrics::default()
        };
        let workload = super::branch_loop();
        let doc = parse_json(&json_document(
            &info, &workload, &summary, &summary, &summary, &metrics,
        ))
        .unwrap();
        let string = |s: &str| Some(Json::String(s.to_owned()));
        assert_eq!(doc.get("host.cpu").cloned(), string("CPU \"x\" \\ \u{1}"));
        assert_eq!(doc.get("build.label").cloned(), string("é label"));
        assert_eq!(doc.get("host.os_version"), Some(&Json::Null));
        assert_eq!(doc.get("source.dirty"), Some(&Json::Bool(true)));
        assert_eq!(
            doc.get("timings.cold_dbt.median_ns"),
            Some(&Json::Number("1234567".to_owned()))
        );
        assert_eq!(
            doc.get("cold_dbt_metrics.native_bytes_emitted"),
            Some(&Json::Number("42".to_owned()))
        );
    }

    #[test]
    fn comparison_reports_changes_against_the_spread() {
        use super::{Json, compare_documents};

        let baseline = comparable_document();
        let mut candidate = comparable_document();
        set(
            &mut candidate,
            "source.revision",
            Json::String("bbbb".to_owned()),
        );
        set(&mut candidate, "source.dirty", Json::Bool(true));
        // 900 µs -> 846 µs is 54 µs, beyond the larger IQR of 20 µs.
        set(
            &mut candidate,
            "timings.interpreter.median_ns",
            Json::Number("846000".to_owned()),
        );
        // 1.1 ms -> 1.12 ms is 20 µs, within the larger IQR of 45 µs.
        set(
            &mut candidate,
            "timings.cold_dbt.median_ns",
            Json::Number("1120000".to_owned()),
        );
        set(
            &mut candidate,
            "timings.cold_dbt.iqr_ns",
            Json::Number("45000".to_owned()),
        );
        set(
            &mut candidate,
            "cold_dbt_metrics.native_bytes_emitted",
            Json::Number("290184".to_owned()),
        );
        set(
            &mut candidate,
            "host.os_version",
            Json::String("27.3".to_owned()),
        );

        let lines = compare_documents(&baseline, &candidate).unwrap();
        assert_eq!(
            lines,
            [
                "Baseline:  aaaa",
                "Candidate: bbbb with local changes",
                "Host: Apple M5, macos aarch64; debug assertions false; build label none",
                "Note: OS version differs: baseline 27.2.1, candidate 27.3",
                "Interpreter: median 900µs -> 846µs (-6.00%), larger than the larger IQR 20µs",
                "Cold DBT: median 1.1ms -> 1.12ms (+1.82%), within the larger IQR 45µs",
                "Warm DBT: median 22µs -> 22µs (+0.00%), within the larger IQR 500ns",
                "Cold DBT native_bytes_emitted: 245168 -> 290184 (+18.36%)",
            ]
        );

        let same = compare_documents(&baseline, &baseline).unwrap();
        assert_eq!(same.last().unwrap(), "Cold DBT counters are identical");
    }

    #[test]
    fn comparison_rejects_every_mismatching_field() {
        use super::{Json, compare_documents};

        // Written out independently of MATCHING_FIELDS, so dropping a field
        // there fails this test.
        for path in [
            "format_version",
            "benchmark",
            "host.cpu",
            "host.os",
            "host.arch",
            "build.debug_assertions",
            "build.label",
            "workload.name",
            "workload.guest_instructions",
            "workload.guest_code_bytes",
            "workload.instruction_budget",
            "workload.warmup_rounds",
            "timings.interpreter.samples",
            "timings.cold_dbt.samples",
            "timings.warm_dbt.samples",
        ] {
            let mut candidate = comparable_document();
            let other = match path {
                "build.debug_assertions" => Json::Bool(true),
                "build.label" => Json::String("fat-lto".to_owned()),
                // A valid but different version is refused as unsupported first.
                "format_version" => Json::Number("2".to_owned()),
                p if p.contains("samples")
                    || p.starts_with("workload.") && p != "workload.name" =>
                {
                    Json::Number("7".to_owned())
                }
                _ => Json::String("other".to_owned()),
            };
            set(&mut candidate, path, other);
            let error = compare_documents(&comparable_document(), &candidate).expect_err(path);
            assert!(
                error.contains(path) || path == "format_version" && error.contains("version"),
                "{path}: {error}"
            );
        }
    }

    #[test]
    fn comparison_needs_known_hosts_and_matching_labels() {
        use super::{Json, compare_documents};

        for path in ["host.cpu", "host.os", "host.arch"] {
            let mut unknown = comparable_document();
            set(&mut unknown, path, Json::Null);
            let error = compare_documents(&unknown, &unknown).expect_err(path);
            assert!(error.contains("unknown in the baseline"), "{error}");
        }

        // A document written before the label existed counts as having no label.
        let mut old = comparable_document();
        let Json::Object(fields) = &mut old else {
            unreachable!()
        };
        let build = &mut fields.iter_mut().find(|(k, _)| k == "build").unwrap().1;
        let Json::Object(build) = build else {
            unreachable!()
        };
        build.retain(|(k, _)| k != "label");
        assert!(compare_documents(&old, &comparable_document()).is_ok());

        let mut labelled = comparable_document();
        set(
            &mut labelled,
            "build.label",
            Json::String("fat-lto".to_owned()),
        );
        assert!(compare_documents(&old, &labelled).is_err());
        assert!(compare_documents(&labelled, &labelled).is_ok());

        let mut missing = comparable_document();
        let Json::Object(fields) = &mut missing else {
            unreachable!()
        };
        fields.retain(|(k, _)| k != "workload");
        let error = compare_documents(&comparable_document(), &missing).unwrap_err();
        assert!(error.contains("missing field `workload.name`"), "{error}");
    }

    #[test]
    fn workload_layouts_match_their_independent_instruction_counts() {
        use super::{
            ADDS, BRANCH_ITERATIONS, MEMORY_ITERATIONS, add_chain, branch_loop, checked_memory,
        };

        let add = add_chain();
        assert_eq!(add.name, "add-chain");
        assert_eq!(add.code.len(), 10 + ADDS * 4 + 1);
        assert_eq!(add.instructions, ADDS + 2);
        assert_eq!(add.final_rax, ADDS as u64);
        assert_eq!(add.code[..2], [0x48, 0xB8]);
        assert_eq!(add.code.last(), Some(&0xF4));

        let branch = branch_loop();
        assert_eq!(branch.name, "branch-loop");
        assert_eq!(branch.code.len(), 10 + 4 + 2 + 1);
        assert_eq!(branch.instructions, 1 + BRANCH_ITERATIONS * 2 + 1);
        assert_eq!(branch.final_rax, 0);
        assert_eq!(branch.code[10..14], [0x48, 0x83, 0xE8, 0x01]);
        // The jump ends at byte 16 and byte 10 is six bytes earlier.
        assert_eq!(branch.code[14..16], [0x75, 0xFA]);

        let memory = checked_memory();
        assert_eq!(memory.name, "checked-memory");
        assert_eq!(memory.code.len(), 10 + 10 + MEMORY_ITERATIONS * 11 + 1);
        assert_eq!(memory.instructions, 2 + MEMORY_ITERATIONS * 4 + 1);
        assert_eq!(memory.final_rax, MEMORY_ITERATIONS as u64);
        assert_eq!(memory.code[10..12], [0x48, 0xB9]);
        assert!(MEMORY_ITERATIONS * 4 <= 0x4000);
    }

    #[test]
    fn workload_selection_is_exact_and_reports_unknown_names() {
        use super::workload;

        assert_eq!(workload(None).unwrap().name, "add-chain");
        assert_eq!(workload(Some("branch-loop")).unwrap().name, "branch-loop");
        assert_eq!(
            workload(Some("checked-memory")).unwrap().name,
            "checked-memory"
        );
        let error = workload(Some("branches")).unwrap_err();
        assert!(error.contains("unknown workload"), "{error}");
        assert!(error.contains("branch-loop"), "{error}");
    }
}
