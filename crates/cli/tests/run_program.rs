// SPDX-License-Identifier: GPL-3.0-or-later

//! `xenora run` on an executable: the real binary, real compiled C programs,
//! and what a shell sees of them.
//!
//! The programs are the Gate A ones, built from source for each test process
//! by `tests/compat_fixtures/build-gate-a2.sh` and `build-gate-a1.sh`; what
//! they compute is checked by `crates/loader/tests/gate_a2.rs` and
//! `gate_a1.rs`. These tests check the command around them, so every
//! expectation here follows from the command line the test itself wrote: the
//! arguments it passed, the variant it chose and the digest it gave. Digests
//! come from `shasum`, not from the code under test.
//!
//! A missing toolchain is a failure, never a skip. The default tests need no GPU.
//! The opt-in structured report test for an embedded fixture renders on Metal.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// The line the Gate A2 program's `atexit` function prints when every check
/// passed. `gate_a2.rs` derives the count.
const ALL_PASSED: &str = "atexit: 76 checks passed\r\n";
/// The exit status of the Gate A1 program, as `gate_a1.rs` has it.
const A1_STATUS: u32 = 0x00A1_A1A1;

fn build(script: &str, folder: &str, lock: &'static OnceLock<PathBuf>) -> &'static Path {
    lock.get_or_init(|| {
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{folder}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let script = format!(
            "{}/../../tests/compat_fixtures/{script}",
            env!("CARGO_MANIFEST_DIR")
        );
        let build = Command::new("sh")
            .arg(&script)
            .arg(&out)
            .output()
            .unwrap_or_else(|error| panic!("run {script}: {error}"));
        assert!(
            build.status.success(),
            "{script} failed; the hosted program needs mingw-w64 and the freestanding one \
             a clang and an LLD:\n{}{}",
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );
        out
    })
}

fn hosted(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-gate-a2.sh", "run-program-a2", &OUT).join(format!("{name}.exe"))
}

fn crt_tmp(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-tmp.sh", "run-program-crt-tmp", &OUT).join(format!("{name}.exe"))
}

fn crt_fds(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-fds.sh", "run-program-crt-fds", &OUT).join(format!("{name}.exe"))
}

fn crt_clock(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-clock.sh", "run-program-crt-clock", &OUT).join(format!("{name}.exe"))
}

fn crt_wide(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-wide.sh", "run-program-crt-wide", &OUT).join(format!("{name}.exe"))
}

fn crt_files(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-files.sh", "run-program-crt-files", &OUT).join(format!("{name}.exe"))
}

fn crt_wstream(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-wstream.sh", "run-program-crt-wstream", &OUT).join(format!("{name}.exe"))
}

fn crt_math(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-math.sh", "run-program-crt-math", &OUT).join(format!("{name}.exe"))
}

fn crt_swprintf(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-swprintf.sh", "run-program-crt-swprintf", &OUT).join(format!("{name}.exe"))
}

fn freestanding(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-gate-a1.sh", "run-program-a1", &OUT).join(format!("{name}.exe"))
}

fn crt_sprintf(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-sprintf.sh", "run-program-crt-sprintf", &OUT).join(format!("{name}.exe"))
}

fn crt_codepages(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-codepages.sh", "run-program-crt-codepages", &OUT).join(format!("{name}.exe"))
}

fn crt_locale(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-locale.sh", "run-program-crt-locale", &OUT).join(format!("{name}.exe"))
}

fn crt_compare(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-compare.sh", "run-program-crt-compare", &OUT).join(format!("{name}.exe"))
}

fn crt_console(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-console.sh", "run-program-crt-console", &OUT).join(format!("{name}.exe"))
}

fn crt_winfiles(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-winfiles.sh", "run-program-crt-winfiles", &OUT).join(format!("{name}.exe"))
}

fn crt_winmemory(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-winmemory.sh", "run-program-crt-winmemory", &OUT).join(format!("{name}.exe"))
}

fn crt_winsys(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-winsys.sh", "run-program-crt-winsys", &OUT).join(format!("{name}.exe"))
}

fn crt_winlocks(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-winlocks.sh", "run-program-crt-winlocks", &OUT).join(format!("{name}.exe"))
}

fn crt_crtutils(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-crtutils.sh", "run-program-crt-crtutils", &OUT).join(format!("{name}.exe"))
}

/// The SHA-256 of a file, from the host's `shasum`.
fn digest(path: &Path) -> String {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .expect("run shasum");
    assert!(output.status.success(), "shasum {path:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    let digest = text.split_whitespace().next().unwrap().to_owned();
    assert_eq!(digest.len(), 64, "{text:?}");
    digest
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn xenora(arguments: &[&str]) -> Run {
    let Output {
        status,
        stdout,
        stderr,
    } = Command::new(env!("CARGO_BIN_EXE_xenora"))
        .arg("run")
        .args(arguments)
        .output()
        .expect("run xenora");
    Run {
        code: status.code(),
        stdout: String::from_utf8(stdout).expect("ASCII output"),
        stderr: String::from_utf8(stderr).expect("ASCII output"),
    }
}

/// Run a file as a program, with options of `xenora run` and the program's
/// own arguments.
fn run_program(path: &Path, options: &[&str], arguments: &[&str]) -> Run {
    let mut all = vec!["--pe", path.to_str().unwrap(), "--experimental"];
    all.extend(options);
    all.push("--");
    all.extend(arguments);
    xenora(&all)
}

/// Run a program with bytes piped to its standard input, the way a user
/// does in a shell.
fn run_program_with_input(path: &Path, arguments: &[&str], input: &[u8]) -> Run {
    use std::io::Write as _;
    use std::process::Stdio;
    let mut all = vec!["--pe", path.to_str().unwrap(), "--experimental", "--"];
    all.extend(arguments);
    let mut child = Command::new(env!("CARGO_BIN_EXE_xenora"))
        .arg("run")
        .args(&all)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn xenora");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(input)
        .expect("write input");
    let output = child.wait_with_output().expect("run xenora");
    Run {
        code: output.status.code(),
        stdout: String::from_utf8(output.stdout).expect("ASCII output"),
        stderr: String::from_utf8(output.stderr).expect("ASCII output"),
    }
}

fn crt_secure(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-secure.sh", "run-program-crt-secure", &OUT).join(format!("{name}.exe"))
}

fn crt_stdin(name: &str) -> PathBuf {
    static OUT: OnceLock<PathBuf> = OnceLock::new();
    build("build-crt-stdin.sh", "run-program-crt-stdin", &OUT).join(format!("{name}.exe"))
}

/// The fixed input the fixture expects: two short lines in CR LF, one line
/// longer than its buffer, and a last line with no newline.
const STDIN_INPUT: &[u8] = b"alpha\r\nbeta\r\n0123456789\r\nlast";

#[test]
fn a_program_reading_lines_from_standard_input_counts_them_with_input_and_without() {
    for level in ["O0", "O2"] {
        let path = crt_stdin(&format!("crt-stdin-{level}"));
        let run = run_program_with_input(&path, &[], STDIN_INPUT);
        assert_eq!(run.code, Some(0), "err:{}", run.stderr);
        assert!(
            run.stdout.contains("crt_stdin: 7 of 7 checks passed"),
            "{}",
            run.stdout
        );
        let run = run_program_with_input(&path, &["empty"], b"");
        assert_eq!(run.code, Some(0), "{}", run.stderr);
        assert!(
            run.stdout.contains("crt_stdin: 5 of 5 checks passed"),
            "{}",
            run.stdout
        );
    }
}

#[test]
fn a_wrong_line_count_is_detected_and_input_beyond_the_limit_is_refused() {
    let path = crt_stdin("crt-stdin-break");
    let run = run_program_with_input(&path, &[], STDIN_INPUT);
    assert_eq!(run.code, Some(7), "out:{} err:{}", run.stdout, run.stderr);
    assert!(
        run.stdout.contains("check 7 failed: control"),
        "{}",
        run.stdout
    );
    // One byte more than the limit is refused before the program starts:
    // none of its output appears.
    let input = vec![b'x'; 1_048_577];
    let run = run_program_with_input(&crt_stdin("crt-stdin-O2"), &[], &input);
    assert_ne!(run.code, Some(0));
    assert!(
        run.stderr.contains("exceeds the 1048576 byte limit"),
        "{}",
        run.stderr
    );
    assert!(!run.stdout.contains("crt_stdin"), "{}", run.stdout);
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(format!(
        "{}/../../fixtures/pe/{name}.exe",
        env!("CARGO_MANIFEST_DIR")
    ))
}

fn scratch(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{}-{name}", std::process::id()))
}

/// What the Gate A2 program prints first: its arguments, as it received them.
fn echoed(name: &str, program: &[&str]) -> String {
    let mut text = format!(
        "gate a2: {} arguments\r\n  argv[0] = \"{name}.exe\"\r\n",
        program.len() + 1
    );
    for (index, argument) in program.iter().enumerate() {
        text.push_str(&format!("  argv[{}] = \"{argument}\"\r\n", index + 1));
    }
    text
}

#[test]
fn a_changed_fixture_runs_as_a_program_and_not_as_the_fixture() {
    // The same length as an embedded fixture, so only the comparison of the
    // bytes can tell them apart. It gets no draw service and no fixture
    // result: its import is unresolved and it never reaches its entry point.
    let refused = "run failed: unresolved import dependency: xenora.dll!fixture_draw\n";
    for name in ["trusted-runner", "runtime-services"] {
        let mut bytes = std::fs::read(fixture(name)).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        let changed = scratch(&format!("{name}-changed.exe"));
        std::fs::write(&changed, bytes).unwrap();
        let run = xenora(&["--pe", changed.to_str().unwrap(), "--experimental"]);
        std::fs::remove_file(&changed).unwrap();
        assert_eq!(run.code, Some(125), "{name}: {}", run.stderr);
        assert_eq!(run.stdout, "", "{name}");
        assert_eq!(run.stderr, refused, "{name}");
    }
    // An embedded fixture given program arguments is run as a program too.
    let run = run_program(&fixture("trusted-runner"), &[], &["argument"]);
    assert_eq!(run.code, Some(125), "{}", run.stderr);
    assert_eq!(run.stdout, "");
    assert_eq!(run.stderr, refused);
}

#[test]
fn a_file_that_is_not_an_executable_is_a_failure() {
    let text = scratch("hello.txt");
    std::fs::write(&text, "hello\n").unwrap();
    let empty = scratch("empty.exe");
    std::fs::write(&empty, "").unwrap();
    let missing = scratch("missing.exe");
    // A real executable, but of the host: Mach-O, not PE.
    let host = PathBuf::from(env!("CARGO_BIN_EXE_xenora"));
    let folder = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    for file in [&text, &empty, &missing, &host, &folder] {
        let run = xenora(&["--pe", file.to_str().unwrap(), "--experimental"]);
        assert_eq!(run.code, Some(125), "{file:?}: {}", run.stderr);
        assert_eq!(run.stdout, "", "{file:?}");
        assert!(
            run.stderr.starts_with("run failed: "),
            "{file:?}: {}",
            run.stderr
        );
        assert_eq!(run.stderr.lines().count(), 1, "{file:?}: {}", run.stderr);
    }
    std::fs::remove_file(text).unwrap();
    std::fs::remove_file(empty).unwrap();
}

#[test]
fn an_import_nobody_provides_is_refused_before_the_entry_point() {
    let run = run_program(&freestanding("gate-a1-missing"), &[], &[]);
    assert_eq!(run.code, Some(125), "{}", run.stderr);
    assert_eq!(run.stdout, "");
    assert!(
        run.stderr
            .starts_with("run failed: unresolved import dependency: "),
        "{}",
        run.stderr
    );
}

#[test]
fn any_executable_runs_with_its_arguments_output_and_exit_status() {
    for level in ["O0", "O1", "O2", "Os"] {
        let name = format!("gate-a2-{level}");
        let program = ["2026", "xenora"];
        let run = run_program(&hosted(&name), &[], &program);
        assert_eq!(run.code, Some(0), "{name}: {}", run.stderr);
        assert_eq!(run.stderr, "", "{name}");
        assert!(
            run.stdout.starts_with(&echoed(&name, &program)),
            "{name}: {:?}",
            run.stdout
        );
        assert!(run.stdout.ends_with(ALL_PASSED), "{name}: {:?}", run.stdout);
    }
}

#[test]
fn program_arguments_arrive_as_they_were_written() {
    // A space, an empty-looking option, the separator itself and words that
    // are options of `xenora run`: after `--` they are the program's.
    for label in ["two words", "--trace", "--", "--sha256", "a\"b"] {
        let program = ["7", label];
        let run = run_program(&hosted("gate-a2-O1"), &[], &program);
        // The program prints a quote in an argument as it is.
        assert!(
            run.stdout.starts_with(&echoed("gate-a2-O1", &program)),
            "{label:?}: {:?}",
            run.stdout
        );
        assert_eq!(run.code, Some(0), "{label:?}: {}", run.stderr);
        assert!(run.stdout.ends_with(ALL_PASSED), "{label:?}");
    }
}

#[test]
fn a_program_using_math_h_gets_its_results_from_the_runtime() {
    // crt_math.c checks each of the 49 functions of #468 against values it
    // carries, and exits with the number of the first check that failed.
    for name in ["crt-math-O0", "crt-math-O2"] {
        let run = run_program(&crt_math(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(
            run.stdout, "crt_math: 110 of 110 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_math("crt-math-break"), &[], &[]);
    assert_eq!(run.code, Some(67), "{}", run.stderr);
    assert_eq!(
        run.stdout,
        "check 67 failed: atan2\r\ncrt_math: 109 of 110 checks passed\r\n"
    );
}

#[test]
fn a_program_moving_through_files_and_renaming_them_passes_its_checks() {
    for name in ["crt-files-O0", "crt-files-O2"] {
        let run = run_program(&crt_files(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(
            run.stdout, "crt_files: 28 of 28 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_files("crt-files-break"), &[], &[]);
    assert_eq!(run.code, Some(17), "{}", run.stderr);
    assert_eq!(
        run.stdout,
        "check 17 failed: text position counts the CR\r\ncrt_files: 27 of 28 checks passed\r\n"
    );
}

#[test]
fn a_program_using_temporary_files_passes_its_checks() {
    for name in ["crt-tmp-O0", "crt-tmp-O2"] {
        let run = run_program(&crt_tmp(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(run.stdout, "crt_tmp: 9 of 9 checks passed\r\n", "{name}");
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_tmp("crt-tmp-break"), &[], &[]);
    assert_eq!(run.code, Some(10), "{}", run.stderr);
    assert!(
        run.stdout.contains("check 10 failed: control"),
        "{}",
        run.stdout
    );
}

#[test]
fn a_program_using_file_descriptors_passes_its_checks() {
    for name in ["crt-fds-O0", "crt-fds-O2"] {
        let run = run_program(&crt_fds(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(run.stdout, "crt_fds: 35 of 35 checks passed\r\n", "{name}");
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_fds("crt-fds-break"), &[], &[]);
    assert_eq!(run.code, Some(36), "{}", run.stderr);
    assert!(
        run.stdout.contains("check 36 failed: control"),
        "{}",
        run.stdout
    );
}

#[test]
fn a_program_using_fopen_s_and_stream_indicators_passes_its_checks() {
    for name in ["crt-secure-O0", "crt-secure-O2"] {
        let run = run_program(&crt_secure(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(
            run.stdout, "crt_secure: 16 of 16 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_secure("crt-secure-break"), &[], &[]);
    assert_eq!(run.code, Some(17), "{}", run.stderr);
    assert!(
        run.stdout.contains("check 17 failed: control"),
        "{}",
        run.stdout
    );
}

#[test]
fn a_program_using_the_c_clock_passes_its_checks() {
    for name in ["crt-clock-O0", "crt-clock-O2"] {
        let run = run_program(&crt_clock(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(
            run.stdout, "crt_clock: 18 of 18 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_clock("crt-clock-break"), &[], &[]);
    assert_eq!(run.code, Some(19), "{}", run.stderr);
    assert!(
        run.stdout.contains("check 19 failed: control"),
        "{}",
        run.stdout
    );
}

#[test]
fn a_program_using_wide_conversions_passes_its_checks() {
    for name in ["crt-wide-O0", "crt-wide-O2"] {
        let run = run_program(&crt_wide(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(run.stdout, "crt_wide: 17 of 17 checks passed\r\n", "{name}");
        assert_eq!(run.stderr, "", "{name}");
    }
    let run = run_program(&crt_wide("crt-wide-break"), &[], &[]);
    assert_eq!(run.code, Some(18), "{}", run.stderr);
    assert!(
        run.stdout.contains("check 18 failed: control"),
        "{}",
        run.stdout
    );
}

#[test]
fn a_program_reading_and_writing_wide_stream_io_passes_its_checks() {
    for name in ["crt-wstream-O0", "crt-wstream-O2"] {
        let run = run_program(&crt_wstream(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        // `putwchar` oriented stdout wide: its byte is the program's only
        // standard output; the diagnostics are on stderr.
        assert_eq!(run.stdout, "z", "{name}");
        assert_eq!(run.stderr, "crt-wstream: 17/17 checks passed\r\n", "{name}");
    }
    let run = run_program(&crt_wstream("crt-wstream-break"), &[], &[]);
    assert_eq!(run.code, Some(2), "{}", run.stderr);
    assert_eq!(
        run.stderr,
        "check 2 failed: fputws\r\ncrt-wstream: 16/17 checks passed\r\n"
    );
}

#[test]
fn a_program_formatting_wide_strings_in_buffers_passes_its_checks() {
    for name in ["crt-swprintf-O0", "crt-swprintf-O2"] {
        let run = run_program(&crt_swprintf(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {}{}", run.stdout, run.stderr);
        assert_eq!(run.stderr, "crt-swprintf: 9/9 checks passed\r\n", "{name}");
    }
    let run = run_program(&crt_swprintf("crt-swprintf-break"), &[], &[]);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert_eq!(
        run.stderr,
        "check 1 failed: integer conversion\r\ncrt-swprintf: 8/9 checks passed\r\n"
    );
}

#[test]
fn the_exit_status_is_the_programs_own() {
    // (variant, program arguments, exit status, standard error)
    let cases: [(&str, &[&str], i32, &str); 6] = [
        ("gate-a2-exit", &["2026", "xenora"], 42, ""),
        ("gate-a2-abort", &["2026", "xenora"], 3, ""),
        (
            "gate-a2-break",
            &["2026", "xenora"],
            22,
            "gate a2: check 22 failed\r\n",
        ),
        // The program's own checks on its arguments: too few, a seed that is
        // not a number, and an empty label, which must arrive empty.
        ("gate-a2-O1", &[], 11, "gate a2: check 11 failed\r\n"),
        (
            "gate-a2-O1",
            &["5x", "label"],
            13,
            "gate a2: check 13 failed\r\n",
        ),
        ("gate-a2-O1", &["5", ""], 14, "gate a2: check 14 failed\r\n"),
    ];
    for (name, program, status, errors) in cases {
        let run = run_program(&hosted(name), &[], program);
        let what = format!("{name} {program:?}");
        assert_eq!(run.code, Some(status), "{what}: {}", run.stderr);
        assert_eq!(run.stderr, errors, "{what}");
        assert!(
            run.stdout.starts_with(&echoed(name, program)),
            "{what}: {:?}",
            run.stdout
        );
        assert!(!run.stdout.contains("76 checks passed"), "{what}");
    }
}

#[test]
fn a_status_wider_than_eight_bits_is_reported_in_full() {
    // A host exit code has eight bits. The Gate A1 program exits with a
    // 32-bit status and prints nothing.
    let run = run_program(&freestanding("gate-a1-O2"), &[], &[]);
    assert_eq!(run.code, Some(0xA1), "{}", run.stderr);
    assert_eq!(run.stdout, "");
    assert_eq!(
        run.stderr,
        format!("guest exit status {A1_STATUS} ({A1_STATUS:#x}); host exit code 161\n")
    );
}

#[test]
fn a_digest_pins_the_file_that_runs() {
    let path = hosted("gate-a2-O2");
    let own = digest(&path);
    // Another file's digest, one digit off, and a file changed after it was
    // hashed.
    let other = digest(&hosted("gate-a2-O1"));
    let last = if own.ends_with('0') { "1" } else { "0" };
    let near = format!("{}{last}", &own[..63]);
    let changed = path.with_file_name("gate-a2-changed.exe");
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&changed, bytes).unwrap();
    for (file, given) in [(&path, &other), (&path, &near), (&changed, &own)] {
        let run = run_program(file, &["--sha256", given], &["2026", "xenora"]);
        assert_eq!(run.code, Some(125), "{file:?}: {}", run.stderr);
        assert_eq!(run.stdout, "", "{file:?}");
        assert_eq!(
            run.stderr, "run failed: execution denied: the file's SHA-256 is not the one given\n",
            "{file:?}"
        );
    }
    // The right digest runs, in either case of its letters.
    for given in [own.clone(), own.to_uppercase()] {
        let run = run_program(&path, &["--sha256", &given], &["2026", "xenora"]);
        assert_eq!(run.code, Some(0), "{}", run.stderr);
        assert!(run.stdout.ends_with(ALL_PASSED), "{:?}", run.stdout);
    }
}

#[test]
fn a_step_limit_that_is_too_small_is_a_failure_not_an_exit() {
    let path = hosted("gate-a2-O0");
    let program = ["2026", "xenora"];
    let whole = run_program(&path, &[], &program);
    assert_eq!(whole.code, Some(0), "{}", whole.stderr);
    let run = run_program(&path, &["--max-steps", "5000"], &program);
    assert_eq!(run.code, Some(125), "{}", run.stderr);
    assert_eq!(
        run.stderr,
        "run failed: the program stopped without exiting after 5002 entry steps: \
         StepLimitExceeded\n"
    );
    // What it had written by then is shown, and is the start of the whole.
    assert!(run.stdout.len() < whole.stdout.len());
    assert!(whole.stdout.starts_with(&run.stdout), "{:?}", run.stdout);
    assert!(!run.stdout.contains("checks passed"));
}

#[test]
fn the_trace_goes_to_standard_error_and_leaves_the_output_alone() {
    let path = hosted("gate-a2-O2");
    let arguments = ["2026", "xenora"];
    let plain = run_program(&path, &[], &arguments);
    let own = digest(&path);
    // (options, the policy and the limits the first trace line names)
    let cases: [(&[&str], &str); 3] = [
        (
            &["--trace"],
            "policy=any entry_step_limit=none wall_limit_ms=15000\n",
        ),
        (
            &["--trace", "--sha256", &own],
            "policy=sha256 entry_step_limit=none wall_limit_ms=15000\n",
        ),
        (
            &["--trace", "--max-steps", "9000000", "--wall-seconds", "20"],
            "policy=any entry_step_limit=9000000 wall_limit_ms=20000\n",
        ),
    ];
    for (options, limits) in cases {
        let traced = run_program(&path, options, &arguments);
        assert_eq!(traced.code, Some(0), "{}", traced.stderr);
        assert_eq!(traced.stdout, plain.stdout);
        assert!(
            traced
                .stderr
                .starts_with(&format!("trace program=gate-a2-O2.exe {limits}")),
            "{}",
            traced.stderr
        );
        assert!(
            traced.stderr.contains("exit=ProcessExit(0)"),
            "{}",
            traced.stderr
        );
    }
}

/// The entry steps the trace of a run reports.
fn entry_steps(trace: &str) -> usize {
    let line = trace
        .lines()
        .find(|line| line.starts_with("trace stage=exit "))
        .unwrap_or_else(|| panic!("no exit line in {trace:?}"));
    let steps = line.split("entry_steps=").nth(1).unwrap();
    steps.split(' ').next().unwrap().parse().unwrap()
}

#[test]
fn a_program_has_no_step_limit_unless_one_is_given() {
    // The spin variant counts to its first argument: three guest steps a turn.
    let path = hosted("gate-a2-spin");
    let arguments = ["500000", "x"];
    let printed = format!(
        "{}gate a2: 500000 turns\r\n",
        echoed("gate-a2-spin", &arguments)
    );
    // More than 1,000,000 steps, the limit a program once had by default.
    let run = run_program(&path, &["--trace"], &arguments);
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(run.stdout.starts_with(&printed), "{:?}", run.stdout);
    let needed = entry_steps(&run.stderr);
    assert!(needed > 1_500_000, "{needed}");
    // A limit that is given holds: half of what the program needs stops it
    // inside its loop. One more than the count runs (the loader reports the
    // limit when the count is used up exactly at the exit), and so does the
    // largest number there is.
    // The scheduler counts the two dispatch trampolines of the main thread
    // as instructions, which direct execution never had: the report is the
    // budget plus those two.
    let short = (needed / 2).to_string();
    let run = run_program(&path, &["--max-steps", &short], &arguments);
    assert_eq!(run.code, Some(125), "{}", run.stderr);
    let reported = needed / 2 + 2;
    assert_eq!(
        run.stderr,
        format!(
            "gate a2: counting to 500000\r\nrun failed: the program stopped without exiting \
             after {reported} entry steps: StepLimitExceeded\n"
        )
    );
    assert_eq!(run.stdout, echoed("gate-a2-spin", &arguments));
    for enough in [(needed + 1).to_string(), usize::MAX.to_string()] {
        let run = run_program(&path, &["--max-steps", &enough], &arguments);
        assert_eq!(run.code, Some(0), "{enough}: {}", run.stderr);
        assert!(run.stdout.starts_with(&printed), "{enough}");
    }
}

#[test]
fn a_program_that_never_ends_is_stopped_at_the_wall_deadline() {
    // Given 0 the spin variant never ends. What it wrote first, to standard
    // output and to standard error, is shown.
    let path = hosted("gate-a2-spin");
    let arguments = ["0", "for ever"];
    // (options, whether the trace's lines come first on standard error)
    for (options, traced) in [
        (&["--wall-seconds", "1"][..], false),
        (&["--wall-seconds", "2", "--trace"], true),
    ] {
        let seconds: u64 = options[1].parse().unwrap();
        let started = std::time::Instant::now();
        let run = run_program(&path, options, &arguments);
        let took = started.elapsed();
        assert_eq!(run.code, Some(124), "{options:?}: {}", run.stderr);
        assert_eq!(
            run.stdout,
            echoed("gate-a2-spin", &arguments),
            "{options:?}"
        );
        // The program's own line on standard error, then the runner's.
        let message = format!(
            "gate a2: counting to 0\r\nrun failed: the program exceeded the {seconds} second \
             wall deadline\n"
        );
        if traced {
            assert!(run.stderr.ends_with(&message), "{}", run.stderr);
            assert!(!run.stderr.contains("stage=exit"), "{}", run.stderr);
        } else {
            assert_eq!(run.stderr, message);
        }
        // Not before the deadline. That the output is there shows the worker
        // stopped itself: a worker the supervisor has to stop shows nothing.
        assert!(took >= std::time::Duration::from_secs(seconds), "{took:?}");
    }
}

#[test]
fn bad_command_lines_are_usage_errors() {
    let path = hosted("gate-a2-O2");
    let path = path.to_str().unwrap();
    let own = digest(Path::new(path));
    let short = &own[..63];
    let not_hex = format!("{}g", &own[..63]);
    let signed = format!("+{}", &own[..63]);
    let embedded = fixture("trusted-runner");
    let embedded = embedded.to_str().unwrap();
    // (the command line, what the message names)
    let cases: [(&[&str], &str); 15] = [
        (
            &["--pe", path, "--experimental", "--sha256", short],
            "--sha256 needs",
        ),
        (
            &["--pe", path, "--experimental", "--sha256", &not_hex],
            "--sha256 needs",
        ),
        (
            &["--pe", path, "--experimental", "--sha256", &signed],
            "--sha256 needs",
        ),
        (
            &["--pe", path, "--experimental", "--sha256"],
            "--sha256 needs",
        ),
        (&["--pe", path], "run requires --experimental"),
        (&["--experimental"], "missing --pe path"),
        // The Metal read-out and the worker's own flag are not a program's.
        (
            &["--pe", path, "--experimental", "--metal-hud"],
            "--metal-hud applies",
        ),
        (
            &["--pe", path, "--experimental", "--program"],
            "unknown or repeated",
        ),
        // A program's step limit has no upper bound; a fixture's has. The
        // fixture is refused here, before it would draw.
        (
            &["--pe", path, "--experimental", "--max-steps", "0"],
            "--max-steps must be at least 1",
        ),
        (
            &["--pe", path, "--experimental", "--max-steps", "many"],
            "--max-steps needs a number",
        ),
        (
            &["--pe", path, "--experimental", "--wall-seconds", "0"],
            "--wall-seconds must be in 1..=3600",
        ),
        (
            &["--pe", path, "--experimental", "--wall-seconds", "3601"],
            "--wall-seconds must be in 1..=3600",
        ),
        (
            &["--pe", path, "--experimental", "--wall-seconds", "1.5"],
            "--wall-seconds needs a number",
        ),
        // A fixture keeps its own deadline. It is refused here, before it
        // would draw.
        (
            &["--pe", embedded, "--experimental", "--wall-seconds", "5"],
            "--wall-seconds applies to a program",
        ),
        (
            &["--pe", embedded, "--experimental", "--max-steps", "10001"],
            "--max-steps must be in 1..=10000",
        ),
    ];
    for (arguments, message) in cases {
        let run = xenora(arguments);
        assert_eq!(run.code, Some(2), "{arguments:?}: {}", run.stderr);
        assert_eq!(run.stdout, "", "{arguments:?}");
        assert!(
            run.stderr.starts_with(message),
            "{arguments:?}: {}",
            run.stderr
        );
        assert!(run.stderr.contains("usage: xenora run"), "{arguments:?}");
    }
}

fn startup_report(run: &Run) -> &str {
    let reports: Vec<_> = run
        .stderr
        .lines()
        .filter_map(|line| line.strip_prefix("xenora-startup "))
        .collect();
    assert_eq!(reports.len(), 1, "{}", run.stderr);
    reports[0]
}

#[test]
fn startup_report_is_ordered_and_preserves_the_guest_status() {
    let path = freestanding("gate-a1-O2");
    let plain = run_program(&path, &[], &[]);
    let run = run_program(&path, &["--startup-report"], &[]);
    assert_eq!(run.code, plain.code);
    assert_eq!(run.stdout, plain.stdout);
    assert!(!plain.stderr.contains("xenora-startup "));
    let report = startup_report(&run);
    assert_eq!(
        report,
        concat!(
            r#"{"schema_version":1,"events":[{"stage":"parsed","completed":true},"#,
            r#"{"stage":"mapped","completed":true},{"stage":"relocated","completed":true},"#,
            r#"{"stage":"imports_bound","completed":true},{"stage":"tls_completed","completed":true},"#,
            r#"{"stage":"entry_started","completed":false},{"stage":"entry_returned","completed":true}],"#,
            r#""last_completed_stage":"entry_returned","protocol":"complete","outcome":"#,
            r#"{"kind":"guest_exit","guest_status":10592673,"worker_code":161}}"#,
        ),
    );
}

#[test]
fn startup_report_keeps_import_failure_context_and_the_relocated_stage() {
    let run = run_program(&fixture("delay-imports"), &["--startup-report"], &[]);
    assert_eq!(run.code, Some(125));
    let report = startup_report(&run);
    assert!(
        report.contains(r#""last_completed_stage":"relocated""#),
        "{report}"
    );
    assert!(
        report.contains(r#""kind":"known_failure","phase":"startup""#),
        "{report}"
    );
    assert!(
        report.contains("fixture-ordinary.dll!OrdinaryByName"),
        "{report}"
    );
    assert!(!report.contains("entry_started"), "{report}");
}

#[test]
fn startup_report_preserves_a_real_worker_deadline_and_guest_output() {
    let arguments = ["0"];
    let run = run_program(
        &hosted("gate-a2-spin"),
        &["--wall-seconds", "1", "--startup-report"],
        &arguments,
    );
    assert_eq!(run.code, Some(124));
    assert!(
        run.stdout.starts_with(&echoed("gate-a2-spin", &arguments)),
        "{}",
        run.stdout
    );
    let report = startup_report(&run);
    assert!(
        report.contains(r#"{"stage":"entry_started","completed":false}"#),
        "{report}"
    );
    assert!(
        report.contains(r#""last_completed_stage":"tls_completed""#),
        "{report}"
    );
    assert!(
        report.contains(r#""kind":"wall_timeout","worker_code":124"#),
        "{report}"
    );
}

#[test]
fn startup_report_keeps_the_admission_failure_outside_the_loader_stages() {
    let run = run_program(
        &fixture("trusted-runner"),
        &["--sha256", &"0".repeat(64), "--startup-report"],
        &[],
    );
    assert_eq!(run.code, Some(125));
    let report = startup_report(&run);
    assert!(
        report.contains(r#""events":[],"last_completed_stage":null"#),
        "{report}"
    );
    assert!(
        report.contains(r#""kind":"known_failure","phase":"admission""#),
        "{report}"
    );
    assert!(report.contains("SHA-256 is not the one given"), "{report}");
}

#[test]
#[ignore = "requires native Metal; run with API and shader validation enabled"]
fn startup_report_ends_in_verified_metal_frame() {
    let path = fixture("trusted-runner");
    let run = xenora(&[
        "--pe",
        path.to_str().unwrap(),
        "--experimental",
        "--startup-report",
    ]);
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(run.stdout.contains("verified_pixels=12"), "{}", run.stdout);
    let report = startup_report(&run);
    assert!(
        report.contains(r#"{"stage":"framebuffer_verified","completed":true}"#),
        "{report}"
    );
    assert!(
        report.contains(r#""last_completed_stage":"framebuffer_verified","protocol":"complete""#),
        "{report}"
    );
    assert!(
        report.contains(r#""kind":"verified","worker_code":0"#),
        "{report}"
    );
}

#[test]
fn startup_report_preserves_a_guest_exit_from_tls_before_entry() {
    let path = freestanding("gate-a1-tls-exit");
    let run = run_program(&path, &["--startup-report"], &[]);
    assert_eq!(run.code, Some(73), "{}", run.stderr);
    assert!(run.stdout.is_empty());
    assert_eq!(
        startup_report(&run),
        concat!(
            r#"{"schema_version":1,"events":[{"stage":"parsed","completed":true},"#,
            r#"{"stage":"mapped","completed":true},{"stage":"relocated","completed":true},"#,
            r#"{"stage":"imports_bound","completed":true},{"stage":"tls_completed","completed":true}],"#,
            r#""last_completed_stage":"tls_completed","protocol":"complete","outcome":"#,
            r#"{"kind":"guest_exit","guest_status":73,"worker_code":73}}"#,
        ),
    );
}

#[test]
fn a_program_using_the_buffer_forms_of_printf_runs_and_detects_a_wrong_result() {
    for name in ["crt-sprintf-O0", "crt-sprintf-O2"] {
        let run = run_program(&crt_sprintf(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(
            run.stdout, "!\r\ncrt_sprintf: 31 of 31 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_sprintf("crt-sprintf-break"), &[], &[]);
    assert_eq!(run.code, Some(31), "control: {:?}", run.stdout);
    assert_eq!(
        run.stdout,
        "!\r\ncheck 31 failed: control\r\ncrt_sprintf: 30 of 31 checks passed\r\n"
    );
}

#[test]
fn a_program_using_the_encoding_conversions_runs_and_detects_a_wrong_result() {
    for name in ["crt-codepages-O0", "crt-codepages-O2"] {
        let run = run_program(&crt_codepages(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(
            run.stdout, "crt-codepages: 23/23 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_codepages("crt-codepages-break"), &[], &[]);
    assert_eq!(run.code, Some(24), "control: {:?}", run.stdout);
    assert_eq!(
        run.stdout,
        "check 24 failed: the control fails\r\ncrt-codepages: 1/24 checks failed, first 24\r\n"
    );
}

#[test]
fn a_program_using_the_locale_identity_runs_and_detects_a_wrong_result() {
    for name in ["crt-locale-O0", "crt-locale-O2"] {
        let run = run_program(&crt_locale(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(run.stdout, "crt-locale: 18/18 checks passed\r\n", "{name}");
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_locale("crt-locale-break"), &[], &[]);
    assert_eq!(run.code, Some(19), "control: {:?}", run.stdout);
    assert_eq!(
        run.stdout,
        "check 19 failed: the control fails\r\ncrt-locale: 1/19 checks failed, first 19\r\n"
    );
}

#[test]
fn a_program_comparing_strings_runs_and_detects_a_wrong_result() {
    for name in ["crt-compare-O0", "crt-compare-O2"] {
        let run = run_program(&crt_compare(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(run.stdout, "crt-compare: 26/26 checks passed\r\n", "{name}");
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_compare("crt-compare-break"), &[], &[]);
    assert_eq!(run.code, Some(27), "control: {:?}", run.stdout);
    assert_eq!(
        run.stdout,
        "check 27 failed: the control fails\r\ncrt-compare: 1/27 checks failed, first 27\r\n"
    );
}

#[test]
fn a_program_using_the_console_runs_and_detects_a_wrong_result() {
    for name in ["crt-console-O0", "crt-console-O2"] {
        let run = run_program(&crt_console(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert!(
            run.stdout.contains("crt-console: 20/20 checks passed\r\n"),
            "{name}: {:?}",
            run.stdout
        );
        // printf and WriteConsole share one capture, in call order.
        // WriteConsole writes its bytes raw: LF, not the CRT's CR LF.
        let narrow = run.stdout.find("narrow\n").expect("narrow write");
        let wide = run.stdout.find("wide\n").expect("wide write");
        assert!(narrow < wide, "{name}: {:?}", run.stdout);
        assert!(run.stderr.contains("to-error"), "{name}: {:?}", run.stderr);
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_console("crt-console-break"), &[], &[]);
    assert_eq!(run.code, Some(21), "control: {:?}", run.stdout);
}

#[test]
fn a_program_using_the_windows_file_services_runs_and_detects_a_wrong_result() {
    for name in ["crt-winfiles-O0", "crt-winfiles-O2"] {
        let run = run_program(&crt_winfiles(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(
            run.stdout, "crt-winfiles: 25/25 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_winfiles("crt-winfiles-break"), &[], &[]);
    assert_eq!(run.code, Some(26), "control: {:?}", run.stdout);
}

#[test]
fn a_program_using_the_legacy_memory_services_runs_and_detects_a_wrong_result() {
    for name in ["crt-winmemory-O0", "crt-winmemory-O2"] {
        let run = run_program(&crt_winmemory(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(
            run.stdout, "crt-winmemory: 16/16 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_winmemory("crt-winmemory-break"), &[], &[]);
    assert_eq!(run.code, Some(17), "control: {:?}", run.stdout);
}

#[test]
fn a_program_using_the_error_environment_and_system_services_runs() {
    for name in ["crt-winsys-O0", "crt-winsys-O2"] {
        let run = run_program(&crt_winsys(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(run.stdout, "crt-winsys: 25/25 checks passed\r\n", "{name}");
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_winsys("crt-winsys-break"), &[], &[]);
    assert_eq!(run.code, Some(26), "control: {:?}", run.stdout);
}

#[test]
fn a_program_using_byte_range_locks_and_the_file_mode_services_runs() {
    for name in ["crt-winlocks-O0", "crt-winlocks-O2"] {
        let run = run_program(&crt_winlocks(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(
            run.stdout, "crt-winlocks: 51/51 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_winlocks("crt-winlocks-break"), &[], &[]);
    assert_eq!(run.code, Some(52), "control: {:?}", run.stdout);
}

#[test]
fn a_program_using_the_small_crt_utilities_runs() {
    for name in ["crt-crtutils-O0", "crt-crtutils-O2"] {
        let run = run_program(&crt_crtutils(name), &[], &[]);
        assert_eq!(run.code, Some(0), "{name}: {:?}", run.stdout);
        assert_eq!(
            run.stdout, "crt-crtutils: 23/23 checks passed\r\n",
            "{name}"
        );
        assert_eq!(run.stderr, "", "{name}");
    }
    // The control must fail its last check: the exit code is its number.
    let run = run_program(&crt_crtutils("crt-crtutils-break"), &[], &[]);
    assert_eq!(run.code, Some(24), "control: {:?}", run.stdout);
}
