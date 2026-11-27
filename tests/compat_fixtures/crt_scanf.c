// SPDX-License-Identifier: GPL-3.0-or-later
//
// A hosted C program for #605: formatted input with sscanf, sscanf_s,
// scanf, scanf_s, fscanf and fscanf_s. Each expected value follows from the
// input this program gives or writes and from ISO C 7.21.6.2 (C17), with
// the extensions Microsoft documents in "scanf Type Field Characters",
// "scanf Width Specification", "sscanf_s" and "fopen" (text mode). It prints
// how many checks passed and returns the number of the first that failed,
// or 0.
//
// The host pipes a fixed input to the standard input:
// "42 word\r\n3.25,tail\r\nend". After the text-mode replacement of CR LF
// by LF the stream is "42 word\n3.25,tail\nend".
#include <float.h>
#include <stdio.h>

static unsigned passed, total, first;

static void check(int ok, const char *what) {
    total++;
    if (ok) {
        passed++;
    } else {
        if (!first) first = total;
        printf("check %u failed: %s\n", total, what);
    }
}

static int same(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return *a == *b;
}

int main(void) {
    int a = 0, b = 0, c = 0, n = 0;
    unsigned u = 0, x = 0, y = 0;
    signed char small = 0;
    short half = 0;
    long long wide = 0;
    float f = 0, g = 0;
    double d = 0, e = 0;
    char s[16], t[16], r[16], ch = 0;

    // Integers, in the base each type names: %i takes the prefix.
    check(sscanf("  -17 0x1F 017 4294967295", "%d %i %i %u", &a, &b, &c, &u) == 4 && a == -17 &&
              b == 31 && c == 15 && u == 4294967295u,
          "d, i and u");
    check(sscanf("ff 0XAbc 777", "%x %X %o", &x, &y, &u) == 3 && x == 0xFF && y == 0xABC &&
              u == 0777,
          "x and o");
    check(sscanf("-5 300 -9000000000", "%hhd %hd %lld", &small, &half, &wide) == 3 &&
              small == -5 && half == 300 && wide == -9000000000LL,
          "hh, h and ll");
    // A width ends an item early; the rest is the next item's.
    check(sscanf("123456", "%2d%3d%d", &a, &b, &c) == 3 && a == 12 && b == 345 && c == 6,
          "widths");

    // Strings, characters and scansets.
    check(sscanf("hello world", "%s %3c", s, t) == 2 && same(s, "hello") && t[0] == 'w' &&
              t[1] == 'o' && t[2] == 'r',
          "s and c");
    check(sscanf("abc123,def", "%[a-z]%[0-9],%[^\n]", s, t, r) == 3 && same(s, "abc") &&
              same(t, "123") && same(r, "def"),
          "scansets");

    // Floating point: the compiler rounds each literal correctly too.
    check(sscanf("2.5 -1e-3 0x1.8p1 inf", "%f %lf %la %f", &f, &d, &e, &g) == 4 && f == 2.5f &&
              d == -1e-3 && e == 3.0 && g > FLT_MAX,
          "floating point");
    // 16777217 lies halfway between two floats and rounds to the even one.
    check(sscanf("0.1 16777217", "%lf %f", &d, &f) == 2 && d == 0.1 && f == 16777216.0f,
          "rounding to nearest");

    // %% matches a percent sign and %n stores the characters read so far;
    // neither is counted.
    check(sscanf("50% done", "%d%% %n", &a, &n) == 1 && a == 50 && n == 4, "%% and %n");
    // Suppressed conversions read but assign nothing.
    a = 0;
    check(sscanf("1 2 3", "%*d %d %*d", &a) == 1 && a == 2, "suppression");

    // Failures: a matching failure gives the count so far; an input failure
    // before the first conversion gives EOF.
    b = 77;
    check(sscanf("12 abc", "%d %d", &a, &b) == 1 && a == 12 && b == 77, "matching failure");
    check(sscanf("x", "%d", &a) == 0, "matching failure first");
    check(sscanf("", "%d", &a) == EOF && sscanf("   ", "%d", &a) == EOF, "input failure");
    check(sscanf("a-b", "a+%d", &a) == 0, "literal mismatch");

    // The secure forms take the size of each c, s and [ buffer.
    check(sscanf_s("abc x", "%s %c", s, (unsigned)sizeof s, &ch, 1u) == 2 && same(s, "abc") &&
              ch == 'x',
          "sscanf_s");
    check(sscanf_s("toolong", "%3s%[a-z]", s, 4u, t, (unsigned)sizeof t) == 2 && same(s, "too") &&
              same(t, "long"),
          "sscanf_s widths");

    // The standard input. A conversion leaves the character after its item
    // unread, so getchar sees the line feed.
#ifdef CRT_SCANF_BREAK
    // Control: the carriage return does not survive the replacement, so
    // this check fails.
    check(scanf("%d %s", &a, s) == 2 && a == 42 && same(s, "word") && getchar() == '\r',
          "stdin line feed left unread");
#else
    check(scanf("%d %s", &a, s) == 2 && a == 42 && same(s, "word") && getchar() == '\n',
          "stdin line feed left unread");
#endif
    check(scanf("%lf,%s", &d, s) == 2 && d == 3.25 && same(s, "tail"), "stdin literal");
    check(scanf_s("%s", s, (unsigned)sizeof s) == 1 && same(s, "end") && feof(stdin),
          "scanf_s to the end");
    check(scanf("%d", &a) == EOF, "stdin EOF");

    // A file in text mode: each line feed is written as CR LF, read back as
    // one line feed, and positions count both bytes.
    FILE *w = fopen("scan.txt", "w");
    check(w && fputs("7 8.5 name\n-3\n", w) >= 0 && fclose(w) == 0, "write file");
    FILE *in = fopen("scan.txt", "r");
    check(in && fscanf(in, "%d %f %s", &a, &f, s) == 3 && a == 7 && f == 8.5f &&
              same(s, "name") && _ftelli64(in) == 10,
          "fscanf text");
    check(in && fscanf(in, "%d", &a) == 1 && a == -3 && _ftelli64(in) == 14, "fscanf next line");
    check(in && fscanf(in, "%d", &a) == EOF && fclose(in) == 0, "fscanf EOF");

    // In binary mode the carriage return is a character.
    FILE *raw = fopen("scan.txt", "rb");
    check(raw && fscanf(raw, "%*s %*s %*s%c", &ch) == 1 && ch == '\r', "fscanf binary");
    check(raw && fscanf_s(raw, "%c%s", &ch, 1u, s, (unsigned)sizeof s) == 2 && ch == '\n' &&
              same(s, "-3"),
          "fscanf_s binary");
    check(raw && fclose(raw) == 0 && remove("scan.txt") == 0, "close and remove");

    printf("crt_scanf: %u of %u checks passed\n", passed, total);
    return (int)first;
}
