// SPDX-License-Identifier: GPL-3.0-or-later

//! Formatted input: the engine behind `sscanf`, `scanf` and `fscanf` (#605).
//!
//! A mingw-w64 program reaches the narrow `scanf` family through two entry
//! points of the Universal CRT: `__stdio_common_vsscanf` for a string and
//! `__stdio_common_vfscanf` for a stream (corecrt_stdio_config.h, stdio.h).
//! The rules here are those of ISO C 7.21.6.2 (C17), with the extensions
//! Microsoft documents in "scanf Width Specification" and "scanf Type Field
//! Characters":
//!
//! - A white-space directive reads input up to the first character that is
//!   not white space, which stays unread, and never fails. Any other
//!   ordinary character must be the next input character; a different one
//!   stays unread and the directive fails.
//! - A conversion skips white space first, except `[`, `c` and `n`. Its input
//!   item is "the longest sequence of input characters which does not exceed
//!   any specified field width and which is, or is a prefix of, a matching
//!   input sequence". The character after it stays unread. When the item is
//!   only a prefix, such as `0x` for `%x` or `1e` for `%f`, the directive
//!   fails and the item stays consumed: a stream has one character of
//!   push-back.
//! - An input item of length zero is a matching failure, unless the input
//!   ended, which is an input failure. The result is the number of items
//!   assigned, or EOF when an input failure comes before the first conversion
//!   has completed. `%n` stores the characters read so far, but "no argument
//!   is converted", so it is not counted and, like `%%`, does not end the
//!   window for EOF. A conversion with `*` does: ISO C's wording and
//!   Microsoft's ("found in the first attempt to read a character") both
//!   give the count, 0, when only suppressed conversions completed. The host
//!   C libraries Xenora was compared with return EOF there.
//!
//! Integers are those of `strtol` and `strtoul` in the base the type names:
//! `d` 10, `i` from the prefix, `o` 8, `u` 10, `x` 16 with an optional `0x`.
//! An `int` and a `long` are 32 bits, `long long`, `intmax_t`, `size_t` and
//! `ptrdiff_t` 64. A negative input of an unsigned conversion is negated in
//! the type `strtoul` or `strtoull` returns, as ISO C says. `p` takes
//! hexadecimal digits with an optional `0x`, as Microsoft's table says, and
//! no sign, which is Xenora's choice where ISO C leaves the form to the
//! implementation. Floating-point input is the subject sequence of `strtod`:
//! decimal, hexadecimal, an infinity or a NaN, rounded correctly to nearest,
//! ties to even, into a `float`, or a `double` for `l` and `L` (Microsoft:
//! "long double (same as double)").
//!
//! Where ISO C leaves the result undefined and Microsoft documents nothing,
//! the guest stops with the runtime-limit fault instead of receiving a
//! guessed value:
//!
//! - a converted integer the destination cannot represent;
//! - `%n` with `*` or a width, a width of zero, a size the type does not take,
//!   `I` and `I32`, an unknown type and an unterminated `[`;
//! - a `-` in a scanset right after a range, as in `[a-c-e]`;
//! - a `%c` field that the input ends inside, after its first character;
//! - in the secure forms, a field that does not fit its buffer: Microsoft
//!   says only that "nothing is written to the buffer";
//! - wide conversions (`C`, `S`, `lc`, `ls`, `l[`), which narrow `scanf`
//!   would have to convert through a locale;
//! - an input item longer than 1 MiB.
//!
//! The NaN payload of `nan(...)` is not kept: the result is the default
//! quiet NaN with the input's sign, which is Xenora's choice where ISO C
//! leaves the meaning of the sequence to the implementation. A decimal or
//! hexadecimal number too large or too small for the type becomes the
//! correctly rounded infinity or zero, and `errno` is not set: ISO C does
//! not say that `scanf` sets it, and Microsoft's pages say nothing.
//!
//! None of this has been compared with Windows.

use super::crt_files::EINVAL;
use super::crt_streams::{Failed, Outcome, c_string};
use crate::cpu::ExitReason;
use crate::memory::{GuestMemory, MemoryError};

/// The exit for a request the runtime does not model.
const RUNTIME_LIMIT: ExitReason = ExitReason::IllegalInstruction(0xFE);
/// The longest format string the services read, in bytes.
const MAX_FORMAT: u64 = 1 << 20;
/// Maximum bytes read by the modern string adapter, plus its terminator probe.
const MAX_STRING_INPUT: u64 = 1 << 20;
/// The longest input item, in characters.
const MAX_ITEM: u64 = 1 << 20;
/// `EOF` (stdio.h).
pub(super) const EOF: i32 = -1;

/// Option bits of `__stdio_common_vsscanf` and `__stdio_common_vfscanf`
/// (corecrt_stdio_config.h). mingw-w64 passes the legacy wide-specifier bit
/// from `sscanf`, `scanf` and `fscanf`, and the secure bit alone from
/// `sscanf_s` and `scanf_s`. The wide-specifier bit concerns what `%s` and
/// `%c` mean in the wide functions and changes nothing here. The merged
/// string adapter also accepts the legacy msvcrt compatibility bit. Its
/// distinctions are not modelled by this narrow scanner.
const SECURE_CRT: u64 = 0x1;
const LEGACY_WIDE_SPECIFIERS: u64 = 0x2;
const LEGACY_MSVCRT_COMPATIBILITY: u64 = 0x4;
const KNOWN_OPTIONS: u64 = SECURE_CRT | LEGACY_WIDE_SPECIFIERS | LEGACY_MSVCRT_COMPATIBILITY;

/// Where the engine reads input characters.
pub(super) trait Source {
    /// The next character, without taking it, or `None` at the end. Once it
    /// has returned `None` it returns `None` again.
    fn peek(&mut self) -> Result<Option<u8>, MemoryError>;
    /// Take the character `peek` returned.
    fn take(&mut self);
}

/// Where the engine finds its arguments and puts what it converts.
pub(super) trait Target {
    /// The next 8-byte slot of the guest's `va_list`.
    fn argument(&mut self) -> Result<u64, MemoryError>;
    /// Assign the bytes of one converted item to guest memory at `address`.
    fn store(&mut self, address: u64, bytes: Vec<u8>);
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ScanError {
    /// Not modelled, or undefined and undocumented: the guest stops.
    Unsupported,
    Memory(MemoryError),
}

impl From<MemoryError> for ScanError {
    fn from(error: MemoryError) -> Self {
        Self::Memory(error)
    }
}

/// What one call did.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) struct Scanned {
    /// The function result: the number of items assigned, or `EOF`.
    pub(super) result: i32,
    /// How many input characters the call consumed. The rest stays unread.
    pub(super) consumed: u64,
    /// Whether a read met the end of the input, which sets a stream's
    /// end-of-file indicator.
    pub(super) met_end: bool,
}

/// Why a directive failed.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Failure {
    Input,
    Matching,
}

/// The C locale's `isspace`: blank, and tab through carriage return.
fn is_space(byte: u8) -> bool {
    byte == b' ' || (0x09..=0x0D).contains(&byte)
}

struct Reader<'s, S> {
    source: &'s mut S,
    consumed: u64,
    met_end: bool,
}

impl<'s, S: Source> Reader<'s, S> {
    fn peek(&mut self) -> Result<Option<u8>, ScanError> {
        let next = self.source.peek()?;
        if next.is_none() {
            self.met_end = true;
        }
        Ok(next)
    }

    fn take(&mut self) {
        self.source.take();
        self.consumed += 1;
    }

    fn skip_space(&mut self) -> Result<(), ScanError> {
        while let Some(byte) = self.peek()? {
            if !is_space(byte) {
                break;
            }
            self.take();
        }
        Ok(())
    }

    /// The input item of one conversion: at most `width` characters.
    fn field(&mut self, width: Option<u64>) -> Field<'_, 's, S> {
        Field {
            reader: self,
            left: width.unwrap_or(u64::MAX),
            length: 0,
        }
    }
}

struct Field<'r, 's, S> {
    reader: &'r mut Reader<'s, S>,
    left: u64,
    length: u64,
}

impl<S: Source> Field<'_, '_, S> {
    /// The next character of the field, or `None` at its width or at the
    /// end of the input.
    fn peek(&mut self) -> Result<Option<u8>, ScanError> {
        if self.left == 0 {
            Ok(None)
        } else {
            self.reader.peek()
        }
    }

    /// Take the next character when `accept` says it extends the item.
    fn take_if(&mut self, accept: impl Fn(u8) -> bool) -> Result<Option<u8>, ScanError> {
        match self.peek()? {
            Some(byte) if accept(byte) => {
                if self.length == MAX_ITEM {
                    return Err(ScanError::Unsupported);
                }
                self.reader.take();
                self.left -= 1;
                self.length += 1;
                Ok(Some(byte))
            }
            _ => Ok(None),
        }
    }

    /// Take the characters of `word`, in either case. False when the input
    /// stops matching it; the characters that matched stay consumed.
    fn word(&mut self, word: &[u8]) -> Result<bool, ScanError> {
        for expected in word {
            if self
                .take_if(|byte| byte.eq_ignore_ascii_case(expected))?
                .is_none()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The failure of an item that is not a matching sequence. A peek that
    /// met the end stays at the end, so an empty item with the end met is
    /// one the end prevented.
    fn failure(&self) -> Failure {
        if self.length == 0 && self.reader.met_end {
            Failure::Input
        } else {
            Failure::Matching
        }
    }
}

/// A length modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Size {
    Default,
    /// `hh`
    Char,
    /// `h`
    Short,
    /// `l`
    Long,
    /// `ll`
    LongLong,
    /// `j`, `z`, `t` and `I64`: 64 bits on x64.
    Wide64,
    /// `L`
    LongDouble,
}

impl Size {
    /// The bytes of an integer destination, and the width in bits of the
    /// type `strtol` or `strtoul` returns for this size.
    fn integer(self) -> Result<(usize, u32), ScanError> {
        match self {
            Size::Char => Ok((1, 32)),
            Size::Short => Ok((2, 32)),
            Size::Default | Size::Long => Ok((4, 32)),
            Size::LongLong | Size::Wide64 => Ok((8, 64)),
            Size::LongDouble => Err(ScanError::Unsupported),
        }
    }
}

struct Spec {
    suppress: bool,
    width: Option<u64>,
    size: Size,
    kind: u8,
    /// The scanset of `[`: which bytes it accepts.
    set: Option<Box<[bool; 256]>>,
}

fn parse_spec(format: &[u8], at: &mut usize) -> Result<Spec, ScanError> {
    let next = |at: usize| format.get(at).copied();
    let suppress = next(*at) == Some(b'*');
    if suppress {
        *at += 1;
    }
    let mut width = None;
    while let Some(digit) = next(*at).filter(u8::is_ascii_digit) {
        let value: u64 = width.unwrap_or(0);
        width = Some(
            value
                .saturating_mul(10)
                .saturating_add(u64::from(digit - b'0')),
        );
        *at += 1;
    }
    if width == Some(0) {
        return Err(ScanError::Unsupported);
    }
    let size = match (next(*at), next(*at + 1), next(*at + 2)) {
        (Some(b'h'), Some(b'h'), _) => (Size::Char, 2),
        (Some(b'h'), ..) => (Size::Short, 1),
        (Some(b'l'), Some(b'l'), _) => (Size::LongLong, 2),
        (Some(b'l'), ..) => (Size::Long, 1),
        (Some(b'j' | b'z' | b't'), ..) => (Size::Wide64, 1),
        (Some(b'I'), Some(b'6'), Some(b'4')) => (Size::Wide64, 3),
        // `I` and `I32` are not on Microsoft's scanf pages.
        (Some(b'I'), ..) => return Err(ScanError::Unsupported),
        (Some(b'L'), ..) => (Size::LongDouble, 1),
        _ => (Size::Default, 0),
    };
    *at += size.1;
    let size = size.0;
    let kind = next(*at).ok_or(ScanError::Unsupported)?;
    *at += 1;
    let set = if kind == b'[' {
        Some(parse_scanset(format, at)?)
    } else {
        None
    };
    Ok(Spec {
        suppress,
        width,
        size,
        kind,
        set,
    })
}

/// The scanset after `[`, through its `]`. A `]` first, after an optional
/// `^`, is a member (ISO C). A `-` first or last is a member; between two
/// characters it is a range in either order: "Both %[a-z] and %[z-a] are
/// interpreted as equivalent" (Microsoft).
fn parse_scanset(format: &[u8], at: &mut usize) -> Result<Box<[bool; 256]>, ScanError> {
    let mut set = Box::new([false; 256]);
    let negate = format.get(*at) == Some(&b'^');
    if negate {
        *at += 1;
    }
    let mut first = true;
    // The last member, when it may start a range.
    let mut previous: Option<u8> = None;
    loop {
        let byte = *format.get(*at).ok_or(ScanError::Unsupported)?;
        if byte == b']' && !first {
            *at += 1;
            break;
        }
        let last = format.get(*at + 1) == Some(&b']');
        if byte == b'-' && !first && !last {
            let (Some(low), Some(&high)) = (previous, format.get(*at + 1)) else {
                // After a range, or before the end of the format: what that
                // means is not documented.
                return Err(ScanError::Unsupported);
            };
            let (low, high) = (low.min(high), low.max(high));
            for member in low..=high {
                set[usize::from(member)] = true;
            }
            previous = None;
            *at += 2;
        } else {
            set[usize::from(byte)] = true;
            previous = Some(byte);
            *at += 1;
        }
        first = false;
    }
    if negate {
        for member in set.iter_mut() {
            *member = !*member;
        }
    }
    Ok(set)
}

/// What a successful directive did.
struct Done {
    /// A conversion completed, which `%%` and `%n` are not.
    conversion: bool,
    /// It assigned an item that the result counts.
    counted: bool,
}

const CONVERTED: Done = Done {
    conversion: true,
    counted: true,
};
/// A conversion with `*`: it completes and assigns nothing.
const UNCOUNTED: Done = Done {
    conversion: true,
    counted: false,
};
/// `%%` and `%n`, which convert nothing.
const NO_CONVERSION: Done = Done {
    conversion: false,
    counted: false,
};

/// Run `format` over the input. Items are assigned through `target` as
/// their conversions complete.
fn scan_source<S: Source, T: Target>(
    format: &[u8],
    source: &mut S,
    target: &mut T,
    secure: bool,
) -> Result<Scanned, ScanError> {
    let mut reader = Reader {
        source,
        consumed: 0,
        met_end: false,
    };
    let mut assigned = 0i32;
    let mut converted = false;
    let mut at = 0;
    let failure = loop {
        let Some(&byte) = format.get(at) else {
            break None;
        };
        if is_space(byte) {
            while format.get(at).copied().is_some_and(is_space) {
                at += 1;
            }
            reader.skip_space()?;
            continue;
        }
        at += 1;
        if byte != b'%' {
            match reader.peek()? {
                None => break Some(Failure::Input),
                Some(next) if next == byte => reader.take(),
                Some(_) => break Some(Failure::Matching),
            }
            continue;
        }
        let spec = parse_spec(format, &mut at)?;
        match convert(&mut reader, target, &spec, secure)? {
            Ok(done) => {
                converted |= done.conversion;
                if done.counted {
                    assigned += 1;
                }
            }
            Err(failure) => break Some(failure),
        }
    };
    let result = match failure {
        Some(Failure::Input) if !converted => EOF,
        _ => assigned,
    };
    Ok(Scanned {
        result,
        consumed: reader.consumed,
        met_end: reader.met_end,
    })
}

/// The destination of a character conversion, and its size in the secure
/// forms, which follows the pointer as an `unsigned` ("The size parameter is
/// of type unsigned, not size_t").
fn character_destination<T: Target>(
    target: &mut T,
    secure: bool,
) -> Result<(u64, Option<u64>), ScanError> {
    let address = target.argument()?;
    let size = if secure {
        Some(target.argument()? & 0xFFFF_FFFF)
    } else {
        None
    };
    Ok((address, size))
}

fn convert<S: Source, T: Target>(
    reader: &mut Reader<'_, S>,
    target: &mut T,
    spec: &Spec,
    secure: bool,
) -> Result<Result<Done, Failure>, ScanError> {
    match spec.kind {
        b'%' => {
            // "The complete conversion specification shall be %%."
            if spec.suppress || spec.width.is_some() || spec.size != Size::Default {
                return Err(ScanError::Unsupported);
            }
            reader.skip_space()?;
            Ok(match reader.peek()? {
                None => Err(Failure::Input),
                Some(b'%') => {
                    reader.take();
                    Ok(NO_CONVERSION)
                }
                Some(_) => Err(Failure::Matching),
            })
        }
        b'n' => {
            if spec.suppress || spec.width.is_some() {
                return Err(ScanError::Unsupported);
            }
            let (bytes, _) = spec.size.integer()?;
            let value = signed_bits(false, Some(reader.consumed), bytes)?;
            let address = target.argument()?;
            target.store(address, value.to_le_bytes()[..bytes].to_vec());
            Ok(Ok(NO_CONVERSION))
        }
        b'c' | b's' | b'[' => characters(reader, target, spec, secure),
        b'd' | b'i' | b'u' | b'o' | b'x' | b'X' | b'p' => integer(reader, target, spec),
        b'a' | b'A' | b'e' | b'E' | b'f' | b'F' | b'g' | b'G' => floating(reader, target, spec),
        // `C` and `S` are wide in the narrow functions, and anything else
        // has no meaning.
        _ => Err(ScanError::Unsupported),
    }
}

fn characters<S: Source, T: Target>(
    reader: &mut Reader<'_, S>,
    target: &mut T,
    spec: &Spec,
    secure: bool,
) -> Result<Result<Done, Failure>, ScanError> {
    // `h` is "single-byte"; `l` is wide; other sizes have no meaning here.
    if !matches!(spec.size, Size::Default | Size::Short) {
        return Err(ScanError::Unsupported);
    }
    let destination = if spec.suppress {
        None
    } else {
        Some(character_destination(target, secure)?)
    };
    let mut bytes = Vec::new();
    let keep = destination.is_some();
    let terminated = spec.kind != b'c';
    match spec.kind {
        b'c' => {
            // "The width indicates the exact number of characters", one by
            // default, with no terminator.
            let width = spec.width.unwrap_or(1);
            if let Some((_, Some(size))) = destination
                && width > size
            {
                return Err(ScanError::Unsupported);
            }
            let mut field = reader.field(Some(width));
            while let Some(byte) = field.take_if(|_| true)? {
                if keep {
                    bytes.push(byte);
                }
            }
            if field.length < width {
                if field.length == 0 {
                    return Ok(Err(Failure::Input));
                }
                return Err(ScanError::Unsupported);
            }
        }
        b's' => {
            reader.skip_space()?;
            let mut field = reader.field(spec.width);
            while let Some(byte) = field.take_if(|byte| !is_space(byte))? {
                if keep {
                    bytes.push(byte);
                }
            }
            if field.length == 0 {
                return Ok(Err(field.failure()));
            }
        }
        _ => {
            let set = spec.set.as_deref().expect("a scanset");
            let mut field = reader.field(spec.width);
            while let Some(byte) = field.take_if(|byte| set[usize::from(byte)])? {
                if keep {
                    bytes.push(byte);
                }
            }
            if field.length == 0 {
                return Ok(Err(field.failure()));
            }
        }
    }
    let Some((address, size)) = destination else {
        return Ok(Ok(UNCOUNTED));
    };
    if terminated {
        bytes.push(0);
    }
    if size.is_some_and(|size| bytes.len() as u64 > size) {
        return Err(ScanError::Unsupported);
    }
    target.store(address, bytes);
    Ok(Ok(CONVERTED))
}

fn digit_value(byte: u8) -> Option<u32> {
    char::from(byte).to_digit(16)
}

/// Read an integer item in `base`, 0 for the prefix rule of `%i`. The sign
/// and the magnitude, `None` when it exceeds 64 bits.
fn integer_item<S: Source>(
    field: &mut Field<'_, '_, S>,
    base: u32,
    signed_input: bool,
) -> Result<Result<(bool, Option<u64>), Failure>, ScanError> {
    let mut negative = false;
    if signed_input && let Some(sign) = field.take_if(|byte| byte == b'+' || byte == b'-')? {
        negative = sign == b'-';
    }
    let mut base = base;
    let mut digits = false;
    if base == 16 || base == 0 {
        if field.take_if(|byte| byte == b'0')?.is_some() {
            digits = true;
            if field
                .take_if(|byte| byte == b'x' || byte == b'X')?
                .is_some()
            {
                // `0x` is only a prefix: digits must follow.
                base = 16;
                digits = false;
            } else if base == 0 {
                base = 8;
            }
        } else if base == 0 {
            base = 10;
        }
    }
    let mut magnitude = Some(0u64);
    while let Some(byte) =
        field.take_if(|byte| digit_value(byte).is_some_and(|value| value < base))?
    {
        let value = u64::from(digit_value(byte).expect("a digit"));
        digits = true;
        magnitude = magnitude
            .and_then(|magnitude| magnitude.checked_mul(u64::from(base)))
            .and_then(|magnitude| magnitude.checked_add(value));
    }
    if !digits {
        return Ok(Err(field.failure()));
    }
    Ok(Ok((negative, magnitude)))
}

/// The bits of a signed value in `bytes` bytes, or none when it does not
/// fit.
fn signed_bits(negative: bool, magnitude: Option<u64>, bytes: usize) -> Result<u64, ScanError> {
    let magnitude = i128::from(magnitude.ok_or(ScanError::Unsupported)?);
    let value = if negative { -magnitude } else { magnitude };
    let bits = 8 * bytes as u32;
    let (low, high) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
    if !(low..=high).contains(&value) {
        return Err(ScanError::Unsupported);
    }
    Ok(value as u64)
}

/// The bits of an unsigned conversion: the magnitude must fit the type
/// `strtoul` or `strtoull` returns, `width` bits, a negative input is
/// negated in it, and the result must fit the destination.
fn unsigned_bits(
    negative: bool,
    magnitude: Option<u64>,
    bytes: usize,
    width: u32,
) -> Result<u64, ScanError> {
    let magnitude = magnitude.ok_or(ScanError::Unsupported)?;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if magnitude > mask {
        return Err(ScanError::Unsupported);
    }
    let value = if negative {
        magnitude.wrapping_neg() & mask
    } else {
        magnitude
    };
    if bytes < 8 && value >> (8 * bytes) != 0 {
        return Err(ScanError::Unsupported);
    }
    Ok(value)
}

fn integer<S: Source, T: Target>(
    reader: &mut Reader<'_, S>,
    target: &mut T,
    spec: &Spec,
) -> Result<Result<Done, Failure>, ScanError> {
    let (base, signed, signed_input) = match spec.kind {
        b'd' => (10, true, true),
        b'i' => (0, true, true),
        b'u' => (10, false, true),
        b'o' => (8, false, true),
        b'x' | b'X' => (16, false, true),
        _ => {
            if spec.size != Size::Default {
                return Err(ScanError::Unsupported);
            }
            (16, false, false)
        }
    };
    let (bytes, width) = if spec.kind == b'p' {
        (8, 64)
    } else {
        spec.size.integer()?
    };
    reader.skip_space()?;
    let mut field = reader.field(spec.width);
    let (negative, magnitude) = match integer_item(&mut field, base, signed_input)? {
        Ok(item) => item,
        Err(failure) => return Ok(Err(failure)),
    };
    if spec.suppress {
        return Ok(Ok(UNCOUNTED));
    }
    let value = if signed {
        signed_bits(negative, magnitude, bytes)?
    } else {
        unsigned_bits(negative, magnitude, bytes, width)?
    };
    let address = target.argument()?;
    target.store(address, value.to_le_bytes()[..bytes].to_vec());
    Ok(Ok(CONVERTED))
}

/// A binary floating-point format of IEEE 754: its width, its precision in
/// bits, counting the implicit one, and its exponent range.
#[derive(Clone, Copy)]
struct Binary {
    width: u32,
    precision: i64,
    min_exponent: i64,
    max_exponent: i64,
}

const BINARY32: Binary = Binary {
    width: 32,
    precision: 24,
    min_exponent: -126,
    max_exponent: 127,
};
const BINARY64: Binary = Binary {
    width: 64,
    precision: 53,
    min_exponent: -1022,
    max_exponent: 1023,
};

impl Binary {
    fn sign(self) -> u64 {
        1 << (self.width - 1)
    }

    fn infinity(self) -> u64 {
        ((2 * self.max_exponent + 1) as u64) << (self.precision - 1)
    }

    /// The default quiet NaN: all exponent bits and the top fraction bit.
    fn nan(self) -> u64 {
        self.infinity() | 1 << (self.precision - 2)
    }
}

/// The bits of `digits` (hexadecimal digit values) with the last
/// `fraction_digits` after the point, times two to the `exponent`, rounded
/// to nearest with ties to even.
fn hexadecimal_bits(digits: &[u8], fraction_digits: usize, exponent: i64, format: Binary) -> u64 {
    let Some(first) = digits.iter().position(|&digit| digit != 0) else {
        return 0;
    };
    let significant = &digits[first..];
    let kept = significant.len().min(16);
    let mut mantissa = 0u64;
    for &digit in &significant[..kept] {
        mantissa = mantissa << 4 | u64::from(digit);
    }
    // The digits that did not fit only matter as a sticky bit below the
    // last kept one.
    let sticky = significant[kept..].iter().any(|&digit| digit != 0);
    let dropped = (significant.len() - kept) as i64;
    // The value is mantissa * 2^scale.
    let scale = exponent
        .saturating_sub(4 * fraction_digits as i64)
        .saturating_add(4 * dropped);
    let zeros = mantissa.leading_zeros();
    let mantissa = mantissa << zeros;
    // With the top bit at 63, the unbiased exponent of the leading one.
    let leading = scale.saturating_sub(i64::from(zeros)).saturating_add(63);
    if leading > format.max_exponent {
        return format.infinity();
    }
    let below = format.min_exponent.saturating_sub(leading).max(0);
    let exponent = leading.max(format.min_exponent);
    let shift = (64 - format.precision).saturating_add(below);
    if shift > 64 {
        // Less than half the smallest subnormal.
        return 0;
    }
    let wide = u128::from(mantissa);
    let quotient = (wide >> shift) as u64;
    let rest = wide & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    let up = rest > half || (rest == half && (sticky || quotient & 1 == 1));
    let bias = format.max_exponent;
    // Adding the rounded significand, implicit one included, carries into
    // the exponent field when rounding overflows it, up to infinity.
    (((exponent + bias - 1) as u64) << (format.precision - 1)) + quotient + u64::from(up)
}

enum FloatItem {
    /// The decimal text without its sign, in the grammar Rust's parser takes.
    Decimal(Vec<u8>),
    Hexadecimal {
        digits: Vec<u8>,
        fraction_digits: usize,
        exponent: i64,
    },
    Infinity,
    NotANumber,
}

/// Read the subject sequence of `strtod`: the sign, and what follows it.
fn float_item<S: Source>(
    field: &mut Field<'_, '_, S>,
) -> Result<Result<(bool, FloatItem), Failure>, ScanError> {
    let negative = field.take_if(|byte| byte == b'+' || byte == b'-')? == Some(b'-');
    match field.peek()? {
        Some(b'i' | b'I') => {
            if !field.word(b"inf")? {
                return Ok(Err(Failure::Matching));
            }
            if field
                .peek()?
                .is_some_and(|byte| byte.eq_ignore_ascii_case(&b'i'))
                && !field.word(b"inity")?
            {
                return Ok(Err(Failure::Matching));
            }
            return Ok(Ok((negative, FloatItem::Infinity)));
        }
        Some(b'n' | b'N') => {
            if !field.word(b"nan")? {
                return Ok(Err(Failure::Matching));
            }
            if field.take_if(|byte| byte == b'(')?.is_some() {
                while field
                    .take_if(|byte| byte.is_ascii_alphanumeric() || byte == b'_')?
                    .is_some()
                {}
                if field.take_if(|byte| byte == b')')?.is_none() {
                    return Ok(Err(Failure::Matching));
                }
            }
            return Ok(Ok((negative, FloatItem::NotANumber)));
        }
        _ => {}
    }
    let mut text = Vec::new();
    let mut digits = 0usize;
    if field.take_if(|byte| byte == b'0')?.is_some() {
        if field
            .take_if(|byte| byte == b'x' || byte == b'X')?
            .is_some()
        {
            return hexadecimal_item(field).map(|item| item.map(|item| (negative, item)));
        }
        text.push(b'0');
        digits = 1;
    }
    while let Some(byte) = field.take_if(|byte| byte.is_ascii_digit())? {
        text.push(byte);
        digits += 1;
    }
    if field.take_if(|byte| byte == b'.')?.is_some() {
        text.push(b'.');
        while let Some(byte) = field.take_if(|byte| byte.is_ascii_digit())? {
            text.push(byte);
            digits += 1;
        }
    }
    if digits == 0 {
        return Ok(Err(field.failure()));
    }
    if field
        .take_if(|byte| byte == b'e' || byte == b'E')?
        .is_some()
    {
        text.push(b'e');
        if let Some(sign) = field.take_if(|byte| byte == b'+' || byte == b'-')? {
            text.push(sign);
        }
        let mut exponent_digits = 0;
        while let Some(byte) = field.take_if(|byte| byte.is_ascii_digit())? {
            text.push(byte);
            exponent_digits += 1;
        }
        if exponent_digits == 0 {
            return Ok(Err(Failure::Matching));
        }
    }
    Ok(Ok((negative, FloatItem::Decimal(text))))
}

/// The rest of a hexadecimal subject sequence, after its `0x`.
fn hexadecimal_item<S: Source>(
    field: &mut Field<'_, '_, S>,
) -> Result<Result<FloatItem, Failure>, ScanError> {
    let mut digits = Vec::new();
    let mut fraction_digits = 0;
    while let Some(byte) = field.take_if(|byte| byte.is_ascii_hexdigit())? {
        digits.push(digit_value(byte).expect("a digit") as u8);
    }
    if field.take_if(|byte| byte == b'.')?.is_some() {
        while let Some(byte) = field.take_if(|byte| byte.is_ascii_hexdigit())? {
            digits.push(digit_value(byte).expect("a digit") as u8);
            fraction_digits += 1;
        }
    }
    if digits.is_empty() {
        return Ok(Err(Failure::Matching));
    }
    let mut exponent = 0i64;
    if field
        .take_if(|byte| byte == b'p' || byte == b'P')?
        .is_some()
    {
        let negative = field.take_if(|byte| byte == b'+' || byte == b'-')? == Some(b'-');
        let mut exponent_digits = 0;
        while let Some(byte) = field.take_if(|byte| byte.is_ascii_digit())? {
            exponent = exponent
                .saturating_mul(10)
                .saturating_add(i64::from(byte - b'0'));
            exponent_digits += 1;
        }
        if exponent_digits == 0 {
            return Ok(Err(Failure::Matching));
        }
        // Far beyond any exponent a double has, and far from overflow.
        exponent = exponent.min(1 << 40);
        if negative {
            exponent = -exponent;
        }
    }
    Ok(Ok(FloatItem::Hexadecimal {
        digits,
        fraction_digits,
        exponent,
    }))
}

fn floating<S: Source, T: Target>(
    reader: &mut Reader<'_, S>,
    target: &mut T,
    spec: &Spec,
) -> Result<Result<Done, Failure>, ScanError> {
    let (format, bytes) = match spec.size {
        Size::Default => (BINARY32, 4),
        Size::Long | Size::LongDouble => (BINARY64, 8),
        _ => return Err(ScanError::Unsupported),
    };
    reader.skip_space()?;
    let mut field = reader.field(spec.width);
    let (negative, item) = match float_item(&mut field)? {
        Ok(item) => item,
        Err(failure) => return Ok(Err(failure)),
    };
    if spec.suppress {
        return Ok(Ok(UNCOUNTED));
    }
    let magnitude = match item {
        FloatItem::Infinity => format.infinity(),
        FloatItem::NotANumber => format.nan(),
        FloatItem::Hexadecimal {
            digits,
            fraction_digits,
            exponent,
        } => hexadecimal_bits(&digits, fraction_digits, exponent, format),
        FloatItem::Decimal(text) => {
            let text = std::str::from_utf8(&text).expect("ASCII digits");
            // Rust's parser rounds correctly to the type it is asked for;
            // parsing a float directly avoids rounding twice.
            if bytes == 4 {
                u64::from(
                    text.parse::<f32>()
                        .expect("a decimal subject sequence")
                        .to_bits(),
                )
            } else {
                text.parse::<f64>()
                    .expect("a decimal subject sequence")
                    .to_bits()
            }
        }
    };
    let value = if negative {
        magnitude | format.sign()
    } else {
        magnitude
    };
    let address = target.argument()?;
    target.store(address, value.to_le_bytes()[..bytes].to_vec());
    Ok(Ok(CONVERTED))
}

/// The input of `__stdio_common_vsscanf`: the guest string, up to its
/// terminator or `limit` characters, read only as far as the conversions
/// need it.
struct GuestString<'m> {
    memory: &'m GuestMemory,
    address: u64,
    at: u64,
    limit: u64,
}

impl Source for GuestString<'_> {
    fn peek(&mut self) -> Result<Option<u8>, MemoryError> {
        if self.at == self.limit {
            return Ok(None);
        }
        let address = self
            .address
            .checked_add(self.at)
            .ok_or(MemoryError::Overflow)?;
        match self.memory.read_u8(address)? {
            0 => {
                // The terminator ends the input for good.
                self.limit = self.at;
                Ok(None)
            }
            byte => Ok(Some(byte)),
        }
    }

    fn take(&mut self) {
        self.at += 1;
    }
}

/// `CTRL+Z`, which ends the input of a text-mode stream ("fopen, _wfopen").
const CTRL_Z: u8 = 0x1A;

/// A stream's bytes from `at`, which is where the next read starts and,
/// once a scan has returned, the stream's new position. In text mode a CR
/// LF pair is read as one line feed and `CTRL+Z` ends the input, as the
/// `fopen` page says; a CR alone stays. The standard input is already
/// translated when it is supplied, so it is read with `text` false.
pub(super) struct Bytes<'b> {
    pub(super) bytes: &'b [u8],
    pub(super) at: usize,
    pub(super) text: bool,
}

impl Bytes<'_> {
    /// A CR LF pair at `at` in text mode.
    fn pair(&self) -> bool {
        self.text
            && self.bytes.get(self.at) == Some(&b'\r')
            && self.bytes.get(self.at + 1) == Some(&b'\n')
    }
}

impl Source for Bytes<'_> {
    fn peek(&mut self) -> Result<Option<u8>, MemoryError> {
        Ok(match self.bytes.get(self.at) {
            Some(&CTRL_Z) if self.text => None,
            Some(_) if self.pair() => Some(b'\n'),
            byte => byte.copied(),
        })
    }

    fn take(&mut self) {
        self.at += if self.pair() { 2 } else { 1 };
    }
}

/// Arguments from the guest's `va_list`, one 8-byte slot each, and the
/// assignments, which are written once the input has been read. The guest
/// cannot observe the difference: it is not running while the service is.
struct Deferred<'m> {
    memory: &'m GuestMemory,
    next: u64,
    stores: Vec<(u64, Vec<u8>)>,
}

impl Target for Deferred<'_> {
    fn argument(&mut self) -> Result<u64, MemoryError> {
        let slot = self.memory.read_u64(self.next)?;
        self.next = self.next.checked_add(8).ok_or(MemoryError::Overflow)?;
        Ok(slot)
    }

    fn store(&mut self, address: u64, bytes: Vec<u8>) {
        self.stores.push((address, bytes));
    }
}

/// One call's format and its options, checked before any input is read.
pub(super) struct Request {
    format: Vec<u8>,
    secure: bool,
}

/// What a scan left to do: the assignments to write, in order, and its
/// outcome.
pub(super) struct Pending {
    stores: Vec<(u64, Vec<u8>)>,
    outcome: Result<Scanned, ScanError>,
}

impl Request {
    /// The format string, after the option bits and the locale. A locale
    /// object is not modelled.
    pub(super) fn read(
        memory: &GuestMemory,
        options: u64,
        format: u64,
        locale: u64,
    ) -> Result<Self, ExitReason> {
        if options & !KNOWN_OPTIONS != 0 || locale != 0 {
            return Err(RUNTIME_LIMIT);
        }
        let mut text = Vec::new();
        for offset in 0..=MAX_FORMAT {
            let address = format
                .checked_add(offset)
                .ok_or(ExitReason::MemoryFault(MemoryError::Overflow))?;
            match memory.read_u8(address).map_err(ExitReason::MemoryFault)? {
                0 => {
                    return Ok(Self {
                        format: text,
                        secure: options & SECURE_CRT != 0,
                    });
                }
                byte => text.push(byte),
            }
        }
        Err(RUNTIME_LIMIT)
    }

    /// Scan `source` with the arguments of the `va_list` at `arguments`.
    pub(super) fn scan(
        &self,
        memory: &GuestMemory,
        source: &mut impl Source,
        arguments: u64,
    ) -> Pending {
        let mut target = Deferred {
            memory,
            next: arguments,
            stores: Vec::new(),
        };
        let outcome = scan_source(&self.format, source, &mut target, self.secure);
        Pending {
            stores: target.stores,
            outcome,
        }
    }
}

impl Pending {
    /// Write the assignments and give the outcome. Assignments made before a
    /// fault or an unmodelled conversion are kept, as they would be on a
    /// machine where each completes in turn.
    pub(super) fn finish(self, memory: &mut GuestMemory) -> Result<Scanned, ExitReason> {
        self.write(memory).map_err(|error| match error {
            ScanError::Unsupported => RUNTIME_LIMIT,
            ScanError::Memory(error) => ExitReason::MemoryFault(error),
        })
    }

    fn write(self, memory: &mut GuestMemory) -> Result<Scanned, ScanError> {
        for (address, bytes) in self.stores {
            memory.write_bytes(address, &bytes)?;
        }
        self.outcome
    }
}

/// Read a bounded input or format string for the legacy variadic `sscanf`.
pub(super) fn scan_input_string(memory: &GuestMemory, address: u64) -> Result<Vec<u8>, ExitReason> {
    c_string(memory, address, MAX_ITEM + 1).map_err(ExitReason::MemoryFault)
}

/// Count the argument pointers the legacy variadic adapter needs. A percent
/// inside a scanset is part of the set, and suppression takes no argument.
pub(super) fn count_assignments(format: &[u8]) -> usize {
    let mut at = 0;
    let mut count = 0;
    while let Some(&byte) = format.get(at) {
        at += 1;
        if byte == b'%' {
            let Ok(spec) = parse_spec(format, &mut at) else {
                break;
            };
            if spec.kind != b'%' && !spec.suppress {
                count += 1;
            }
        }
    }
    count
}

/// Run the same engine for the legacy variadic `sscanf` adapter, whose
/// argument pointers have already been collected from the calling convention.
pub(super) fn scan(
    memory: &mut GuestMemory,
    input: Vec<u8>,
    format: &[u8],
    slot: impl FnMut() -> Result<u64, MemoryError>,
) -> Result<i32, ScanError> {
    struct VariadicTarget<F> {
        slot: F,
        stores: Vec<(u64, Vec<u8>)>,
    }

    impl<F: FnMut() -> Result<u64, MemoryError>> Target for VariadicTarget<F> {
        fn argument(&mut self) -> Result<u64, MemoryError> {
            (self.slot)()
        }

        fn store(&mut self, address: u64, bytes: Vec<u8>) {
            self.stores.push((address, bytes));
        }
    }

    let mut source = Bytes {
        bytes: &input,
        at: 0,
        text: false,
    };
    let mut target = VariadicTarget {
        slot,
        stores: Vec::new(),
    };
    let outcome = scan_source(format, &mut source, &mut target, false);
    let pending = Pending {
        stores: target.stores,
        outcome,
    };
    Ok(pending.write(memory)?.result)
}

/// `__stdio_common_vsscanf`: scan the string at `buffer`, at most `count`
/// characters; mingw-w64's `sscanf` passes `SIZE_MAX`. A null buffer or
/// format returns `EOF` with `EINVAL`, as Microsoft's `sscanf_s` page says
/// for when execution continues after the invalid parameter handler.
pub(super) fn scan_string(
    memory: &mut GuestMemory,
    options: u64,
    buffer: u64,
    count: u64,
    format: u64,
    locale: u64,
    arguments: u64,
) -> Outcome<i32> {
    if buffer == 0 || format == 0 {
        return Ok(Err(Failed(EOF, EINVAL)));
    }
    let request = Request::read(memory, options, format, locale)?;
    let shared: &GuestMemory = memory;
    let mut source = GuestString {
        memory: shared,
        address: buffer,
        at: 0,
        limit: if count == u64::MAX {
            MAX_STRING_INPUT + 1
        } else {
            count.min(MAX_STRING_INPUT)
        },
    };
    let pending = request.scan(shared, &mut source, arguments);
    Ok(Ok(pending.finish(memory)?.result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_scans_keep_the_total_input_bound_across_whitespace() {
        use crate::memory::MemoryPermissions;

        const FORMAT: u64 = 0x1000;
        const INPUT: u64 = 0x2000;
        let mut memory = GuestMemory::new();
        let mut bytes = vec![b' '; (1 << 20) + 3];
        bytes[(1 << 20) + 1..].copy_from_slice(b"7\0");
        memory
            .map_region_with_data(INPUT, bytes.len(), MemoryPermissions::READ_WRITE, &bytes)
            .unwrap();
        memory
            .map_region_with_data(FORMAT, 3, MemoryPermissions::READ_WRITE, b"%d\0")
            .unwrap();
        for count in [(1 << 20) + 3, u64::MAX] {
            let result = scan_string(&mut memory, 0, INPUT, count, FORMAT, 0, 0).unwrap();
            assert!(matches!(result, Ok(EOF)), "input count {count}");
        }
    }

    struct Slots {
        slots: Vec<u64>,
        next: usize,
        stores: Vec<(u64, Vec<u8>)>,
    }

    impl Target for Slots {
        fn argument(&mut self) -> Result<u64, MemoryError> {
            let slot = *self.slots.get(self.next).expect("enough arguments");
            self.next += 1;
            Ok(slot)
        }

        fn store(&mut self, address: u64, bytes: Vec<u8>) {
            self.stores.push((address, bytes));
        }
    }

    /// The assignments of one call, in order: address and bytes.
    type Stores = Vec<(u64, Vec<u8>)>;

    /// Scan `input` with the given slots as the `va_list`.
    fn run_with(
        format: &str,
        input: &[u8],
        slots: &[u64],
        secure: bool,
    ) -> Result<(Scanned, Stores), ScanError> {
        let mut source = Bytes {
            bytes: input,
            at: 0,
            text: false,
        };
        let mut target = Slots {
            slots: slots.to_vec(),
            next: 0,
            stores: Vec::new(),
        };
        let scanned = scan_source(format.as_bytes(), &mut source, &mut target, secure)?;
        assert_eq!(scanned.consumed, source.at as u64);
        Ok((scanned, target.stores))
    }

    /// Scan `input` with the addresses 1 to 6 as the destinations.
    fn run(format: &str, input: &str) -> Result<(Scanned, Stores), ScanError> {
        run_with(format, input.as_bytes(), &[1, 2, 3, 4, 5, 6], false)
    }

    fn result(format: &str, input: &str) -> i32 {
        run(format, input).unwrap().0.result
    }

    fn stored(format: &str, input: &str) -> Vec<Vec<u8>> {
        run(format, input)
            .unwrap()
            .1
            .into_iter()
            .map(|(_, bytes)| bytes)
            .collect()
    }

    fn int(value: i32) -> Vec<u8> {
        value.to_le_bytes().to_vec()
    }

    fn f32_bits(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    fn f64_bits(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    /// The host C library's `sscanf`, on cases whose result ISO C defines
    /// and the host follows. `tests/scanf/reference.py` wrote the file and
    /// says how each line is laid out and what it leaves out.
    #[test]
    fn results_match_the_host_c_library_on_defined_cases() {
        let vectors = include_str!("../../tests/scanf/vectors.txt");
        let unhex = |text: &str| -> Vec<u8> {
            (0..text.len() / 2)
                .map(|at| u8::from_str_radix(&text[2 * at..2 * at + 2], 16).unwrap())
                .collect()
        };
        let mut cases = 0;
        let mut differences = Vec::new();
        for line in vectors.lines().filter(|line| !line.starts_with('#')) {
            let fields: Vec<&str> = line.split('\t').collect();
            let [format, input, result, stores] = fields[..] else {
                panic!("four fields: {line}");
            };
            let (format, input) = (unhex(format), unhex(input));
            let expected: Vec<Vec<u8>> = stores.split(',').map(unhex).collect();
            assert_eq!(expected.len(), 6, "{line}");
            let mut source = Bytes {
                bytes: &input,
                at: 0,
                text: false,
            };
            let mut target = Slots {
                slots: (1..=6).map(|index| index * 0x1000).collect(),
                next: 0,
                stores: Vec::new(),
            };
            let shown = format!(
                "{:?} on {:?}",
                String::from_utf8_lossy(&format),
                String::from_utf8_lossy(&input)
            );
            let scanned = match scan_source(&format, &mut source, &mut target, false) {
                Ok(scanned) => scanned,
                Err(error) => {
                    differences.push(format!("{shown}: {error:?}"));
                    continue;
                }
            };
            let mut buffers = [[0xAAu8; 256]; 6];
            for (address, bytes) in target.stores {
                let (index, offset) =
                    ((address / 0x1000 - 1) as usize, (address % 0x1000) as usize);
                buffers[index][offset..offset + bytes.len()].copy_from_slice(&bytes);
            }
            let got: Vec<Vec<u8>> = buffers
                .iter()
                .map(|buffer| {
                    let end = buffer
                        .iter()
                        .rposition(|&byte| byte != 0xAA)
                        .map_or(0, |at| at + 1);
                    buffer[..end].to_vec()
                })
                .collect();
            if scanned.result.to_string() != result || got != expected {
                differences.push(format!(
                    "{shown}: {} {:02x?}, host {result} {:02x?}",
                    scanned.result, got, expected
                ));
            }
            cases += 1;
        }
        assert!(
            differences.is_empty(),
            "{} differences:\n{}",
            differences.len(),
            differences.join("\n")
        );
        assert!(cases >= 4000, "{cases} cases");
    }

    #[test]
    fn eof_comes_only_from_an_input_failure_before_any_conversion() {
        // ISO C 7.21.6.2p16: EOF if an input failure occurs before the first
        // conversion (if any) has completed, else the items assigned.
        assert_eq!(result("%d", ""), EOF);
        assert_eq!(result("%d", "   \n"), EOF, "white space, then the end");
        assert_eq!(result("%d", "x"), 0, "a matching failure");
        assert_eq!(result("", ""), 0, "no directive reads anything");
        assert_eq!(result(" ", ""), 0, "a white-space directive never fails");
        assert_eq!(result("%d%d", "1"), 1);
        assert_eq!(result("a%d", "a"), EOF, "a literal is no conversion");
        assert_eq!(result("a%d", "b"), 0);
        assert_eq!(result("a", ""), EOF);
        assert_eq!(result("%%%d", "%"), EOF, "%% is no conversion");
        assert_eq!(result("%*d%d", "1"), 0, "a suppressed conversion completes");
        assert_eq!(result("%n%d", ""), EOF, "%n converts nothing");
        assert_eq!(result("%n%d", "x"), 0);
    }

    #[test]
    fn white_space_and_literals_leave_the_first_other_character_unread() {
        let (scanned, _) = run(" x", " \t\n\x0b\x0c\rxy").unwrap();
        assert_eq!(scanned.result, 0);
        assert_eq!(scanned.consumed, 7);
        assert!(!scanned.met_end);
        // A literal that differs stays unread.
        let (scanned, _) = run("ab", "ac").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 1));
        // A white-space directive that meets the end sets the indicator.
        let (scanned, _) = run("%d ", "5  ").unwrap();
        assert_eq!(
            (scanned.result, scanned.consumed, scanned.met_end),
            (1, 3, true)
        );
        // %% skips white space before its %.
        assert_eq!(result("%% %d", "  %7"), 1);
    }

    #[test]
    fn integers_follow_strtol_in_the_base_of_their_type() {
        assert_eq!(stored("%d", " -42"), [int(-42)]);
        assert_eq!(stored("%d", "+7"), [int(7)]);
        assert_eq!(stored("%i", "077"), [int(63)]);
        assert_eq!(stored("%i", "-0x1F"), [int(-31)]);
        assert_eq!(stored("%i", "09"), [int(0)], "octal stops at 9");
        assert_eq!(stored("%o", "777"), [int(511)]);
        assert_eq!(stored("%x", "0XfF"), [int(255)]);
        assert_eq!(stored("%X", "ff"), [int(255)]);
        assert_eq!(stored("%u", "4294967295"), [int(-1)]);
        assert_eq!(stored("%hhd", "-128"), [vec![0x80]]);
        assert_eq!(stored("%hu", "65535"), [vec![0xFF, 0xFF]]);
        assert_eq!(stored("%ld", "-2147483648"), [int(i32::MIN)]);
        assert_eq!(
            stored("%lld", "-9223372036854775808"),
            [i64::MIN.to_le_bytes().to_vec()]
        );
        assert_eq!(
            stored("%I64x", "ffffffffffffffff"),
            [u64::MAX.to_le_bytes().to_vec()]
        );
        for size in ["j", "z", "t"] {
            assert_eq!(
                stored(&format!("%{size}u"), "4294967296"),
                [(1u64 << 32).to_le_bytes().to_vec()],
                "{size}"
            );
        }
        assert_eq!(
            stored("%p", "0x0000000140001000"),
            [0x1_4000_1000u64.to_le_bytes().to_vec()]
        );
        assert_eq!(result("%p", "-1"), 0, "no sign for a pointer");
        // The widths count every character of the item, sign and prefix
        // included.
        assert_eq!(stored("%2d%d", "-123"), [int(-1), int(23)]);
        assert_eq!(stored("%1x%x", "0x5"), [int(0)], "x is no hex digit");
        assert_eq!(stored("%3x", "0x5f"), [int(5)]);
    }

    #[test]
    fn a_negative_unsigned_input_is_negated_in_the_strtoul_type() {
        assert_eq!(stored("%u", "-1"), [int(-1)]);
        assert_eq!(stored("%x", "-0x10"), [int(-16)]);
        assert_eq!(stored("%llu", "-1"), [u64::MAX.to_le_bytes().to_vec()]);
        assert_eq!(stored("%u", "-0"), [int(0)]);
        // strtoul returns 32 bits, which an unsigned short cannot hold.
        assert_eq!(run("%hu", "-1").unwrap_err(), ScanError::Unsupported);
        assert_eq!(run("%hhu", "256").unwrap_err(), ScanError::Unsupported);
    }

    #[test]
    fn values_the_destination_cannot_represent_stop_the_guest() {
        for (format, input) in [
            ("%d", "2147483648"),
            ("%d", "-2147483649"),
            ("%hhd", "128"),
            ("%hd", "-32769"),
            ("%u", "4294967296"),
            ("%lld", "9223372036854775808"),
            ("%llu", "18446744073709551616"),
            ("%i", "0x80000000"),
        ] {
            assert_eq!(
                run(format, input).unwrap_err(),
                ScanError::Unsupported,
                "{format} {input}"
            );
        }
        // A suppressed conversion assigns nothing, so nothing overflows.
        assert_eq!(result("%*d", "99999999999999999999999"), 0);
    }

    #[test]
    fn a_prefix_that_is_not_a_number_fails_and_stays_consumed() {
        // ISO C's longest-prefix rule: `0x` is a prefix of a hexadecimal
        // number, so it is the item, and it is not one.
        let (scanned, stores) = run("%x", "0xg").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 2));
        assert!(stores.is_empty());
        let (scanned, _) = run("%i", "0x").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 2));
        let (scanned, _) = run("%d", "-x").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 1));
        let (scanned, _) = run("%d", "-").unwrap();
        assert_eq!(
            (scanned.result, scanned.consumed),
            (0, 1),
            "not EOF: one character was read"
        );
        // The standard's own example: "100ergs" fails %f because 100e is
        // not a number, and "rgs" stays.
        let (scanned, _) = run("%f%s", "100ergs").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 4));
        let (scanned, _) = run("%f", "1e+").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 3));
        let (scanned, _) = run("%f", "infin").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 5));
        let (scanned, _) = run("%f", "nan(ab").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 6));
        let (scanned, _) = run("%f", "0x.p1").unwrap();
        assert_eq!(scanned.result, 0);
        let (scanned, _) = run("%f", "0x1p").unwrap();
        assert_eq!((scanned.result, scanned.consumed), (0, 4));
    }

    #[test]
    fn the_examples_of_the_standard_scan_as_it_says() {
        // ISO C 7.21.6.2, EXAMPLE 1: an int, a float and a string.
        let (scanned, stores) = run("%d%f%s", "25 54.32E-1 thompson").unwrap();
        assert_eq!(scanned.result, 3);
        assert_eq!(stores[0].1, int(25));
        assert_eq!(f32_bits(&stores[1].1), 5.432f32.to_bits());
        assert_eq!(stores[2].1, b"thompson\0");

        // EXAMPLE 2: widths, a suppressed field and a scanset.
        let (scanned, stores) = run("%2d%f%*d %[0123456789]", "56789 0123 56a72").unwrap();
        assert_eq!(scanned.result, 3);
        assert_eq!(stores[0].1, int(56));
        assert_eq!(f32_bits(&stores[1].1), 789.0f32.to_bits());
        assert_eq!(stores[2].1, b"56\0");
        assert_eq!(scanned.consumed, 13, "\"a72\" stays unread");

        // EXAMPLE 3, one call per line of its input.
        let format = "%f%20s of %20s";
        let one = |input: &str| run(format, input).unwrap();
        let (scanned, stores) = one("2 quarts of oil");
        assert_eq!(scanned.result, 3);
        assert_eq!(f32_bits(&stores[0].1), 2.0f32.to_bits());
        assert_eq!(
            (&stores[1].1[..], &stores[2].1[..]),
            (&b"quarts\0"[..], &b"oil\0"[..])
        );
        let (scanned, stores) = one("-12.8degrees Celsius");
        assert_eq!(scanned.result, 2);
        assert_eq!(f32_bits(&stores[0].1), (-12.8f32).to_bits());
        assert_eq!(stores[1].1, b"degrees\0");
        assert_eq!(one("lots of luck").0.result, 0);
        assert_eq!(one("10.0LBS of\ndirt").0.result, 3);
        assert_eq!(one("100ergs of energy").0.result, 0);
        assert_eq!(one("").0.result, EOF);
    }

    #[test]
    fn n_stores_the_characters_read_so_far_and_is_not_counted() {
        let (scanned, stores) = run("%d%n %n%c%hhn", "12   x").unwrap();
        assert_eq!(scanned.result, 2);
        assert_eq!(stores[0].1, int(12));
        assert_eq!(stores[1].1, int(2));
        assert_eq!(stores[2].1, int(5));
        assert_eq!(stores[3].1, b"x");
        assert_eq!(stores[4].1, [6]);
        assert_eq!(stored("%lln", ""), [0u64.to_le_bytes().to_vec()]);
        for format in ["%*n", "%2n", "%Ln"] {
            assert_eq!(
                run(format, "").unwrap_err(),
                ScanError::Unsupported,
                "{format}"
            );
        }
    }

    #[test]
    fn characters_strings_and_scansets_store_what_they_matched() {
        assert_eq!(
            stored("%c", " x"),
            [b" ".to_vec()],
            "no white space is skipped"
        );
        assert_eq!(stored("%3c", "abcd"), [b"abc".to_vec()], "no terminator");
        assert_eq!(stored("%hc", "z"), [b"z".to_vec()], "h is single-byte");
        assert_eq!(stored("%s", "  word next"), [b"word\0".to_vec()]);
        assert_eq!(
            stored("%3s%s", "abcdef"),
            [b"abc\0".to_vec(), b"def\0".to_vec()]
        );
        assert_eq!(stored("%[a-c]", "abcd"), [b"abc\0".to_vec()]);
        assert_eq!(
            stored("%[c-a]", "abcd"),
            [b"abc\0".to_vec()],
            "a range in either order"
        );
        assert_eq!(stored("%[]a]", "]a]b"), [b"]a]\0".to_vec()]);
        assert_eq!(stored("%[^]x]", "ab]"), [b"ab\0".to_vec()]);
        assert_eq!(stored("%[-a]", "-a-b"), [b"-a-\0".to_vec()]);
        assert_eq!(stored("%[a-]", "a-b"), [b"a-\0".to_vec()]);
        assert_eq!(
            stored("%[^\n]%c", "line one\nrest"),
            [b"line one\0".to_vec(), b"\n".to_vec()]
        );
        assert_eq!(stored("%2[0-9]", "12345"), [b"12\0".to_vec()]);
        // A scanset skips no white space, and an empty match fails.
        assert_eq!(result("%[a]", " a"), 0);
        assert_eq!(result("%[a]", ""), EOF);
        assert_eq!(result("%s", ""), EOF);
        assert_eq!(result("%c", ""), EOF);
        // Bytes above 0x7F are characters like any other.
        assert_eq!(
            stored("%s", "\u{e9}t\u{e9}"),
            [b"\xc3\xa9t\xc3\xa9\0".to_vec()]
        );
        // Suppressed fields read their input and take no argument.
        assert_eq!(
            stored("%*s %*c%*[a-z]%s", "skip !abc end"),
            [b"end\0".to_vec()]
        );
    }

    #[test]
    fn what_is_undefined_or_undocumented_stops_the_guest() {
        for (format, input) in [
            ("%0d", "1"),
            ("%Id", "1"),
            ("%I32d", "1"),
            ("%y", "1"),
            ("%", "1"),
            ("%[abc", "a"),
            ("%[a-c-e]", "a"),
            ("%lc", "a"),
            ("%ls", "a"),
            ("%l[a]", "a"),
            ("%C", "a"),
            ("%S", "a"),
            ("%hhc", "a"),
            ("%Ld", "1"),
            ("%hf", "1"),
            ("%llf", "1"),
            ("%lp", "1"),
            ("%5%", "%"),
            ("%*%", "%"),
            ("%3c", "ab"),
        ] {
            assert_eq!(
                run(format, input).unwrap_err(),
                ScanError::Unsupported,
                "{format:?} {input:?}"
            );
        }
    }

    #[test]
    fn the_secure_forms_take_an_unsigned_size_after_each_buffer() {
        let secure = |format: &str, input: &str, slots: &[u64]| {
            run_with(format, input.as_bytes(), slots, true)
        };
        // The size's upper half is not part of it.
        let (scanned, stores) =
            secure("%s %c", "abc d", &[0x10, 0xFFFF_FFFF_0000_0004, 0x20, 1]).unwrap();
        assert_eq!(scanned.result, 2);
        assert_eq!(stores, [(0x10, b"abc\0".to_vec()), (0x20, b"d".to_vec())]);
        assert_eq!(
            secure("%[a-z]", "abcd", &[0x10, 5]).unwrap().1,
            [(0x10, b"abcd\0".to_vec())]
        );
        // A field that does not fit: Microsoft documents only that nothing is
        // written, so the guest stops.
        assert_eq!(
            secure("%s", "abcd", &[0x10, 4]).unwrap_err(),
            ScanError::Unsupported
        );
        assert_eq!(
            secure("%2c", "ab", &[0x10, 1]).unwrap_err(),
            ScanError::Unsupported
        );
        assert_eq!(
            secure("%c", "a", &[0x10, 0]).unwrap_err(),
            ScanError::Unsupported
        );
        // A width keeps a long token within the buffer.
        assert_eq!(
            secure("%3s", "abcdef", &[0x10, 4]).unwrap().1,
            [(0x10, b"abc\0".to_vec())]
        );
        // Numbers take no size, and a suppressed field takes no argument.
        assert_eq!(secure("%*s%d", "x 5", &[0x30]).unwrap().1, [(0x30, int(5))]);
    }

    #[test]
    fn decimal_input_is_rounded_once_into_float_or_double() {
        let float = |input: &str| f32_bits(&stored("%f", input)[0]);
        let double = |input: &str| f64_bits(&stored("%lf", input)[0]);
        assert_eq!(float("0.1"), 0.1f32.to_bits());
        assert_eq!(double("0.1"), 0.1f64.to_bits());
        assert_eq!(f64_bits(&stored("%Lf", "0.1")[0]), 0.1f64.to_bits());
        for format in ["%e", "%E", "%F", "%g", "%G", "%a", "%A"] {
            assert_eq!(
                f32_bits(&stored(format, "1.5")[0]),
                1.5f32.to_bits(),
                "{format}"
            );
        }
        // 1 + 2^-24 + 10^-20 is a little above half way between two
        // floats. A double would round it to the tie 1 + 2^-24, and a float
        // from that to 1. Rounded once, it is 1 + 2^-23.
        let text = "1.000000059604644775400625";
        assert_eq!(float(text), 0x3F80_0001);
        assert_eq!(double("1e400"), f64::INFINITY.to_bits());
        assert_eq!(double("-1e400"), f64::NEG_INFINITY.to_bits());
        assert_eq!(double("1e-400"), 0);
        assert_eq!(double("-0"), (-0.0f64).to_bits());
        assert_eq!(double(".5"), 0.5f64.to_bits());
        assert_eq!(double("5."), 5.0f64.to_bits());
        assert_eq!(double("2.5e+3"), 2500.0f64.to_bits());
        assert_eq!(
            double("4.9406564584124654e-324"),
            1,
            "the smallest subnormal"
        );
        // Width cuts the item.
        assert_eq!(
            stored("%3lf%lf", "1.25"),
            [
                1.2f64.to_bits().to_le_bytes().to_vec(),
                5.0f64.to_bits().to_le_bytes().to_vec()
            ]
        );
    }

    #[test]
    fn infinities_and_nans_are_read_in_any_case() {
        let double = |input: &str| f64_bits(&stored("%lf", input)[0]);
        assert_eq!(double("inf"), f64::INFINITY.to_bits());
        assert_eq!(double("-INFINITY"), f64::NEG_INFINITY.to_bits());
        assert_eq!(double("InFiNiTy"), f64::INFINITY.to_bits());
        assert_eq!(double("nan"), 0x7FF8_0000_0000_0000);
        assert_eq!(double("-NaN(chars_09)"), 0xFFF8_0000_0000_0000);
        assert_eq!(f32_bits(&stored("%f", "nan")[0]), 0x7FC0_0000);
        assert_eq!(f32_bits(&stored("%f", "-inf")[0]), 0xFF80_0000);
        // "inf" followed by something else than "inity" is an infinity, and
        // the rest stays.
        let (scanned, stores) = run("%lf%s", "info").unwrap();
        assert_eq!(scanned.result, 2);
        assert_eq!(stores[1].1, b"o\0");
    }

    #[test]
    fn hexadecimal_input_is_rounded_to_nearest_even() {
        let double = |input: &str| f64_bits(&stored("%la", input)[0]);
        let float = |input: &str| f32_bits(&stored("%a", input)[0]);
        assert_eq!(double("0x1.8p3"), 12.0f64.to_bits());
        assert_eq!(double("0X.8P1"), 1.0f64.to_bits());
        assert_eq!(
            double("0x10"),
            16.0f64.to_bits(),
            "the exponent is optional"
        );
        assert_eq!(double("-0x0p0"), (-0.0f64).to_bits());
        assert_eq!(double("0x1p-1074"), 1);
        assert_eq!(
            double("0x1p-1075"),
            0,
            "half the smallest subnormal ties to zero"
        );
        assert_eq!(double("0x1.8p-1075"), 1, "more than half rounds up");
        assert_eq!(
            double("0x1.00000000000000000001p-1075"),
            1,
            "the sticky digits count"
        );
        assert_eq!(double("0x3p-1075"), 2, "1.5 subnormal units tie to even");
        assert_eq!(double("0x1p-1022"), 0x0010_0000_0000_0000);
        assert_eq!(
            double("0x0.fffffffffffff8p-1022"),
            0x0010_0000_0000_0000,
            "rounds up to the smallest normal"
        );
        assert_eq!(double("0x1.fffffffffffffp1023"), f64::MAX.to_bits());
        assert_eq!(double("0x1.fffffffffffff8p1023"), f64::INFINITY.to_bits());
        assert_eq!(double("0x1p1024"), f64::INFINITY.to_bits());
        assert_eq!(double("0x1p-99999999999999999999"), 0);
        assert_eq!(double("0x00000000000000000000001p0"), 1.0f64.to_bits());
        // More digits than the 64 bits kept: Rust rounds an integer to the
        // nearest double, ties to even, too.
        assert_eq!(
            double("0x123456789abcdef0123p0"),
            (0x123_4567_89ab_cdef_0123_u128 as f64).to_bits()
        );
        assert_eq!(float("0x1.000001p0"), 0x3F80_0000, "a tie to even, down");
        assert_eq!(float("0x1.000003p0"), 0x3F80_0002, "a tie to even, up");
        assert_eq!(float("0x1.0000011p0"), 0x3F80_0001);
        assert_eq!(float("0x1p-149"), 1);
        assert_eq!(float("0x1p-150"), 0);
        assert_eq!(float("0x1.fffffep127"), f32::MAX.to_bits());
        assert_eq!(float("0x1.ffffffp127"), f32::INFINITY.to_bits());
    }

    /// A text-mode stream: CR LF is one line feed, a CR alone stays, and
    /// CTRL+Z ends the input. The position counts bytes of the file.
    #[test]
    fn text_mode_reads_cr_lf_as_a_line_feed_and_stops_at_ctrl_z() {
        let text = |format: &str, input: &[u8]| {
            let mut source = Bytes {
                bytes: input,
                at: 0,
                text: true,
            };
            let mut target = Slots {
                slots: vec![1, 2, 3],
                next: 0,
                stores: Vec::new(),
            };
            let scanned = scan_source(format.as_bytes(), &mut source, &mut target, false).unwrap();
            let stores: Vec<Vec<u8>> = target.stores.into_iter().map(|(_, bytes)| bytes).collect();
            (scanned, stores, source.at)
        };
        let (scanned, stores, at) = text("%c%c%n", b"\r\nx");
        assert_eq!(scanned.result, 2);
        assert_eq!(stores, [b"\n".to_vec(), b"x".to_vec(), int(2)]);
        assert_eq!(at, 3, "two characters, three bytes");
        let (scanned, stores, at) = text("%[^\n]%c", b"a\rb\r\n");
        assert_eq!(scanned.result, 2);
        assert_eq!(stores, [b"a\rb\0".to_vec(), b"\n".to_vec()]);
        assert_eq!(at, 5);
        let (scanned, stores, at) = text("%d%d", b"12\x1a 34");
        assert_eq!(scanned.result, 1, "CTRL+Z is the end");
        assert!(scanned.met_end);
        assert_eq!(stores, [int(12)]);
        assert_eq!(at, 2, "CTRL+Z stays unread");
        let (scanned, _, at) = text("%d", b"\x1a5");
        assert_eq!(scanned.result, EOF);
        assert_eq!(at, 0);
        // A binary stream reads both bytes and CTRL+Z as characters.
        let mut source = Bytes {
            bytes: b"\r\n\x1a",
            at: 0,
            text: false,
        };
        let mut target = Slots {
            slots: vec![1],
            next: 0,
            stores: Vec::new(),
        };
        let scanned = scan_source(b"%3c", &mut source, &mut target, false).unwrap();
        assert_eq!(scanned.result, 1);
        assert_eq!(target.stores[0].1, b"\r\n\x1a");
    }
}
