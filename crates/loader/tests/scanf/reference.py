#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference results for the scanf engine, from the host C library.

  generate   write vectors.txt beside this script
  --check    regenerate in memory; exit 1 if the committed file differs

The cases are generated here, compiled into a small C program with the host
`clang`, and scanned by the host C library's `sscanf`. That library is an
implementation of ISO C 7.21.6.2 that shares no code with Xenora, so the file
is an independent answer for every case whose result ISO C defines and the
host follows.

Left out on purpose, because ISO C leaves the result undefined, the host and
Windows differ, or the host does not follow ISO C; `crt_scan.rs` tests these
by hand:

  - the `l` size of an integer (a long is 64 bits on the host and 32 on
    Windows), and `I64`, which the host does not know
  - a value the destination cannot represent, and a negative input of an
    unsigned conversion into a char or a short, which ISO C leaves undefined
  - an input item that is only a prefix of a number, such as `0x` for `%x`,
    `1e` for `%f` or `infin`: ISO C makes it a matching failure, while the
    host pushes characters back and converts the part before them
  - a `%c` field that the input ends inside, which the host assigns in part
  - a range written high to low in a scanset, which Microsoft reads as the
    same range and the host as three characters
  - `nan(...)`, whose meaning ISO C leaves to the implementation, and wide
    conversions, the secure forms, `%p`, `I` and `I32`, which the engine
    refuses or the host lacks

Each line of vectors.txt has four tab-separated fields:

  format    the format string, in hexadecimal
  input     the input string, in hexadecimal
  result    what sscanf returned, in decimal
  stores    six comma-separated fields, one per destination buffer: its
            bytes in hexadecimal up to the last one that is not 0xAA, the
            value every buffer held before the call

The destinations are the first six arguments after the format, each a
pointer to its own 256-byte buffer, which every input here fits.

Standard library only (Python 3.9+). Needs `clang` on PATH.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "vectors.txt"
MASK64 = (1 << 64) - 1
BUFFERS = 6


class Random:
    """xorshift64*, so the cases do not depend on the Python version."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        s = self.state
        s ^= s >> 12
        s ^= (s << 25) & MASK64
        s ^= s >> 27
        self.state = s
        return (s * 0x2545F4914F6CDD1D) & MASK64

    def below(self, bound):
        return self.next() % bound

    def pick(self, items):
        return items[self.below(len(items))]

    def chance(self, numerator, denominator):
        return self.below(denominator) < numerator


# size prefix -> bits of the destination; `l` is left out (see above)
SIZES = {"hh": 8, "h": 16, "": 32, "ll": 64, "j": 64, "z": 64, "t": 64}
SEPARATORS = [(" ", " "), (" ", "  "), (" ", "\t\n "), ("", " "), (",", ","), (" , ", ","),
              ("x", "x"), (";", ";")]


def integer_value(random, bits, signed):
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    kind = random.below(4)
    if kind == 0:
        return random.pick([low, high, 0, 1, -1 if signed else 2, high // 2, low // 2])
    if kind == 1:
        return low + random.below(min(high - low + 1, 1000))
    if kind == 2:
        return high - random.below(min(high - low + 1, 1000))
    return low + random.next() % (high - low + 1)


def integer_text(random, kind, value):
    """The input text of one integer item, in the base `kind` reads."""
    negative = value < 0
    magnitude = -value if negative else value
    detected = kind == "i"
    if detected:
        kind = random.pick("doxX")
        prefix = {"d": "", "o": "0", "x": "0x", "X": "0X"}[kind]
    else:
        prefix = ""
        if kind in "xX" and random.chance(1, 2):
            prefix = random.pick(["0x", "0X"])
        if kind == "o" and random.chance(1, 3):
            prefix = "0"
    if kind in "du":
        digits = str(magnitude)
    elif kind == "o":
        digits = format(magnitude, "o")
    else:
        digits = format(magnitude, "x")
        if random.chance(1, 2):
            digits = digits.upper()
    if prefix == "0" and digits == "0":
        prefix = ""
    sign = "-" if negative else random.pick(["", "", "", "+"])
    # Leading zeros would make a decimal number octal for %i.
    padded = prefix in ("0x", "0X") or (prefix == "" and not detected)
    zeros = "0" * random.pick([0, 0, 0, 1, 3]) if padded else ""
    return sign + prefix + zeros + digits


def complete_integer(text, kind):
    """Whether `text` is a whole matching sequence, not only a prefix of one."""
    body = text.lstrip("+-")
    if len(text) - len(body) > 1:
        return False
    if kind in "xXi" and body[:2].lower() == "0x":
        return len(body) > 2
    return len(body) > 0


def cut(random, text, complete):
    """A width for `text`: none, one at least as long, or one that stops
    inside it where `complete` says the part read is a whole item. What the
    width leaves is returned too: the format matches it as literal text, so
    no later conversion reads it."""
    if not random.chance(1, 3):
        return "", ""
    widths = [w for w in range(1, len(text) + 3) if complete(text[:w])]
    width = random.pick(widths)
    return str(width), text[width:]


def integer_spec(random):
    kind = random.pick("diuoxX")
    size = random.pick(list(SIZES))
    bits = SIZES[size]
    signed = kind in "di"
    value = integer_value(random, bits, signed)
    if not signed and bits >= 32 and random.chance(1, 5):
        # A negative input is negated in the type strtoul or strtoull
        # returns; it fits only for these sizes.
        value = -random.pick([1, 2, 255, 1 << 31, (1 << bits) - 1, random.next() % (1 << bits)])
    text = integer_text(random, kind, value)
    width, rest = cut(random, text, lambda part: complete_integer(part, kind))
    return f"%{width}{size}{kind}", text, rest


DECIMALS = ["0", "-0", "1", "-1", "0.5", ".5", "5.", "1.5", "0.1", "0.2", "0.3", "123.456",
            "-123.456", "1e10", "1E+10", "1e-10", "2.5e-3", "3.4028235e38", "3.4028236e38",
            "1.17549435e-38", "1.4e-45", "7e-46", "1e-50", "1.7976931348623157e308",
            "1.7976931348623159e308", "2.2250738585072014e-308", "4.9406564584124654e-324",
            "2.4703282292062327e-324", "2.4703282292062328e-324", "1e400", "-1e400", "1e-400",
            "0.000000000000000000000000000000000000000000001", "123456789012345678901234567890",
            "0.1000000000000000055511151231257827021181583404541015625",
            "1.000000059604644775390625", "1.0000000596046447753906250000001",
            "9007199254740993", "9007199254740992.5", "00001.25000", "+7.75", "8.", "100"]
SPECIALS = ["inf", "INF", "-inf", "+Inf", "infinity", "INFINITY", "-Infinity", "nan", "NAN",
            "NaN", "-nan", "+nan"]


def decimal_text(random):
    kind = random.below(4)
    if kind == 0:
        return random.pick(DECIMALS)
    digits = "".join(str(random.below(10)) for _ in range(1 + random.below(25)))
    point = random.below(len(digits) + 1)
    text = digits[:point] + "." + digits[point:] if random.chance(2, 3) else digits
    if random.chance(1, 2):
        text += random.pick("eE") + random.pick(["", "+", "-"]) + str(random.below(330))
    return random.pick(["", "", "-", "+"]) + text


def hexadecimal_text(random):
    digits = "".join(random.pick("0123456789abcdefABCDEF") for _ in range(1 + random.below(20)))
    if random.chance(1, 4):
        digits = "1" + "0" * random.below(16) + random.pick("0123456789abcdef")
    point = random.below(len(digits) + 1)
    text = digits[:point] + "." + digits[point:] if random.chance(2, 3) else digits
    if random.chance(3, 4):
        exponent = random.pick([0, 1, -1, 127, 128, -126, -149, -150, 1023, 1024, -1022, -1074,
                                -1075, -1076, random.below(2200) - 1100])
        text += random.pick("pP") + ("-" if exponent < 0 else random.pick(["", "+"])) + str(
            abs(exponent))
    return random.pick(["", "", "-", "+"]) + random.pick(["0x", "0X"]) + text


def complete_float(text):
    """Whether `text` is a whole strtod subject sequence, not only a prefix."""
    body = text.lstrip("+-")
    if len(text) - len(body) > 1:
        return False
    lower = body.lower()
    if lower[:1] in ("i", "n"):
        return lower in ("inf", "infinity", "nan")
    hexadecimal = lower.startswith("0x")
    if hexadecimal:
        body = lower[2:]
        mantissa, _, exponent = body.partition("p")
        marker = "p" in body
    else:
        mantissa, _, exponent = lower.partition("e")
        marker = "e" in lower
    if mantissa.replace(".", "", 1) == "" or mantissa.count(".") > 1:
        return False
    if marker:
        return exponent.lstrip("+-") != "" and len(exponent) - len(exponent.lstrip("+-")) <= 1
    return True


def float_spec(random):
    kind = random.pick("aAeEfFgG")
    size = random.pick(["", "", "l", "l", "L"])
    choice = random.below(10)
    if choice < 5:
        text = decimal_text(random)
    elif choice < 9:
        text = hexadecimal_text(random)
    else:
        text = random.pick(SPECIALS)
    width, rest = cut(random, text, complete_float)
    return f"%{width}{size}{kind}", text, rest


WORD_BYTES = [chr(c) for c in range(0x21, 0x7F)] + ["é"]


def word(random):
    return "".join(random.pick(WORD_BYTES) for _ in range(1 + random.below(12)))


def scanset(random):
    """A scanset and a member-rich input for it. A `-` is first or last, never
    next to a range or after a leading `]`, and ranges go strictly low to
    high (the host reads `a-a` as `a` and `b`)."""
    members = set()
    parts = []
    negate = random.chance(1, 4)
    if random.chance(1, 6):
        parts.append("]")
        members.add("]")
    elif random.chance(1, 6):
        parts.append("-")
        members.add("-")
    for _ in range(1 + random.below(3)):
        if random.chance(1, 2):
            low = random.pick("aAk0")
            high = chr(ord(low) + 1 + random.below(9))
            parts.append(f"{low}-{high}")
            members.update(chr(c) for c in range(ord(low), ord(high) + 1))
        else:
            character = random.pick("xyz!.?_+=")
            parts.append(character)
            members.add(character)
    if random.chance(1, 6) and "-" not in members:
        parts.append("-")
        members.add("-")
    text = "[" + ("^" if negate else "") + "".join(parts) + "]"
    pool = sorted(members) + ["q", "Q", "9", "#", " "]
    if negate:
        pool = [c for c in "QWERTYqwerty#@ 9" if c not in members] + sorted(members)
    item = "".join(random.pick(pool) for _ in range(1 + random.below(10)))
    return text, item


def character_spec(random):
    choice = random.below(4)
    if choice == 0:
        text = word(random)
        width, rest = cut(random, text, lambda part: part != "")
        return f"%{width}s", text, rest
    if choice == 1:
        text = word(random)
        width = random.below(len(text)) + 1
        shown = str(width) if width > 1 or random.chance(1, 2) else ""
        return f"%{shown}c", text, text[width:]
    set_text, item = scanset(random)
    width, rest = cut(random, item, lambda part: part != "")
    return f"%{width}{set_text}", item, rest


def literal(text):
    """Format text that matches `text`: a `%` is written `%%`."""
    return text.replace("%", "%%")


def case(random):
    """One format and input. Conversions are joined by separators, and the
    input may stop early, right after an item, to exercise input failures
    and EOF.

    Nothing may extend a numeric item into a prefix that is not a number,
    which the host would treat otherwise (see above): the input stops only
    after a whole item, no `x` follows a number, what a width leaves is
    matched as literal text, and the tail after the last item starts with a
    character no number continues with. A `c` or `[` conversion, which skips
    no white space, is never preceded by a separator that only the input
    has."""
    format_parts, input_parts = [], []
    boundaries = [0]
    if random.chance(1, 4):
        text = random.pick(["x=", "[", " ", "%", "\t", "id:"])
        format_parts.append(literal(text))
        input_parts.append(text)
        boundaries.append(len("".join(input_parts)))
    numeric = False
    for index in range(1 + random.below(5)):
        choice = random.below(10)
        if choice < 4:
            spec, text, rest = integer_spec(random)
        elif choice < 7:
            spec, text, rest = float_spec(random)
        elif choice < 9:
            spec, text, rest = character_spec(random)
        else:
            spec, text = random.pick([("%n", ""), ("%hhn", ""), ("%lln", ""), ("%%", "%")])
            rest = ""
        if index:
            in_format, in_input = random.pick(SEPARATORS)
            while (numeric and in_format == "x") or (
                    in_format == "" and spec[-1] in "c]"):
                in_format, in_input = random.pick(SEPARATORS)
            format_parts.append(in_format)
            input_parts.append(in_input if in_format.strip() == "" else in_format)
        numeric = choice < 7 and rest == ""
        if random.chance(1, 8) and spec not in ("%n", "%hhn", "%lln", "%%"):
            spec = "%*" + spec[1:]
        format_parts.append(spec + literal(rest))
        input_parts.append(text)
        boundaries.append(len("".join(input_parts)))
    text = "".join(input_parts)
    if random.chance(1, 6):
        text = text[: random.pick(boundaries)]
    elif random.chance(1, 4):
        text += random.pick([" tail", "\n", "#", "#end", "  "])
    if random.chance(1, 5) and not format_parts[0].startswith(("%c", "%[")) and not (
            format_parts[0][:1] == "%" and format_parts[0][1:2].isdigit()
            and format_parts[0].lstrip("%0123456789")[:1] in ("c", "[")):
        text = random.pick([" ", "\n", "  "]) + text
    return "".join(format_parts).encode("latin-1"), text.encode("utf-8")


def cases():
    random = Random(0x5CA2F00D5EED1234)
    return [case(random) for _ in range(4400)]


def c_literal(data):
    return '"' + "".join(f"\\{byte:03o}" for byte in data) + '"'


def reference(all_cases):
    lines = ["#include <stdio.h>", "#include <string.h>",
             f"static unsigned char b[{BUFFERS}][256];",
             "static void reset(void) { memset(b, 0xAA, sizeof b); }",
             "static void emit(int r) {",
             "    printf(\"%d\\t\", r);",
             f"    for (int i = 0; i < {BUFFERS}; i++) {{",
             "        int last = 255;",
             "        while (last >= 0 && b[i][last] == 0xAA) last--;",
             "        for (int j = 0; j <= last; j++) printf(\"%02x\", b[i][j]);",
             f"        putchar(i + 1 < {BUFFERS} ? ',' : '\\n');",
             "    }",
             "}",
             "int main(void) {"]
    pointers = ", ".join(f"b[{i}]" for i in range(BUFFERS))
    for format_bytes, input_bytes in all_cases:
        lines.append(f"    reset(); emit(sscanf({c_literal(input_bytes)}, "
                     f"{c_literal(format_bytes)}, {pointers}));")
    lines += ["    return 0;", "}"]
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory, "reference.c"), Path(directory, "reference")
        source.write_text("\n".join(lines) + "\n")
        subprocess.run(["clang", "-w", "-O0", source, "-o", binary], check=True)
        output = subprocess.run([binary], check=True, capture_output=True, text=True).stdout
    results = output.splitlines()
    if len(results) != len(all_cases) or any(len(bytes(i)) > 200 for _, i in all_cases):
        sys.exit("the reference program did not scan every case")
    return results


def main(argv):
    if len(argv) != 2 or argv[1] not in ("generate", "--check"):
        sys.exit("usage: reference.py generate|--check")
    all_cases = cases()
    results = reference(all_cases)
    # The host returns EOF when only suppressed conversions completed before
    # an input failure. ISO C ("before the first conversion ... has
    # completed") and Microsoft's description ("found in the first attempt
    # to read a character") both give the count, which crt_scan.rs tests by
    # hand, so those cases are left out.
    kept = [(case, result) for case, result in zip(all_cases, results)
            if not (b"%*" in case[0] and result.startswith("-1\t"))]
    all_cases, results = [case for case, _ in kept], [result for _, result in kept]
    header = ["# Generated by crates/loader/tests/scanf/reference.py from the host C library's",
              "# sscanf. Do not edit. Fields: format, input, result, stores; see that script."]
    rows = [f"{format_bytes.hex()}\t{input_bytes.hex()}\t{result}"
            for (format_bytes, input_bytes), result in zip(all_cases, results)]
    content = "\n".join(header + rows) + "\n"
    if argv[1] == "generate":
        VECTORS.write_text(content)
    elif not VECTORS.exists() or VECTORS.read_text() != content:
        sys.exit("vectors.txt differs from the host C library's output; run reference.py generate")
    print(f"vectors.txt: {len(rows)} cases")


if __name__ == "__main__":
    main(sys.argv)
