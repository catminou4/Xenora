// SPDX-License-Identifier: GPL-3.0-or-later
//
// Hosted checks for the kernel32 services a database engine such as sqlite3
// needs besides plain file I/O: byte-range locks, the file-API code page
// mode, free disk space, heap validation, view flushing and the process id.
// Built with mingw-w64, every call is an import. A wrong check prints its
// number and the program exits with it. With CRT_WINLOCKS_BREAK one
// expectation is wrong on purpose.

#include <direct.h>
#include <errno.h>
#include <io.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <windows.h>

static unsigned checks;
static unsigned failed;
static unsigned failures;

#define CHECK(note, expr)                                                \
    do {                                                                 \
        checks++;                                                        \
        if (!(expr)) {                                                   \
            if (!failed)                                                 \
                failed = checks;                                         \
            failures++;                                                  \
            printf("check %u failed: %s\n", checks, note);              \
        }                                                                \
    } while (0)

static BOOL lock_ex(HANDLE file, DWORD flags, DWORD offset, DWORD length) {
    OVERLAPPED overlapped = {0};
    overlapped.Offset = offset;
    return LockFileEx(file, flags | LOCKFILE_FAIL_IMMEDIATELY, 0, length, 0, &overlapped);
}

static BOOL unlock_ex(HANDLE file, DWORD offset, DWORD length) {
    OVERLAPPED overlapped = {0};
    overlapped.Offset = offset;
    return UnlockFileEx(file, 0, length, 0, &overlapped);
}

int main(void) {
    const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE;
    HANDLE first = CreateFileW(L"C:\\locks.db", GENERIC_READ | GENERIC_WRITE, share, NULL,
                               OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    HANDLE second = CreateFileW(L"C:\\locks.db", GENERIC_READ | GENERIC_WRITE, share, NULL,
                                OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    CHECK("open twice", first != INVALID_HANDLE_VALUE && second != INVALID_HANDLE_VALUE);

    // sqlite's pending, reserved and shared bytes sit at fixed offsets.
    CHECK("exclusive lock", LockFile(first, 1000, 0, 1, 0));
    CHECK("conflicting lock fails",
          !LockFile(second, 1000, 0, 1, 0) && GetLastError() == ERROR_LOCK_VIOLATION);
    CHECK("adjacent lock", LockFile(second, 1001, 0, 1, 0));
    CHECK("partial unlock fails",
          !UnlockFile(first, 1000, 0, 2, 0) && GetLastError() == ERROR_NOT_LOCKED);
    CHECK("unlock", UnlockFile(first, 1000, 0, 1, 0) && UnlockFile(second, 1001, 0, 1, 0));

    CHECK("shared lock", lock_ex(first, 0, 2048, 510));
    CHECK("second shared lock", lock_ex(second, 0, 2048, 510));
    CHECK("exclusive over shared fails",
          !lock_ex(second, LOCKFILE_EXCLUSIVE_LOCK, 2100, 1)
              && GetLastError() == ERROR_LOCK_VIOLATION);
    CHECK("exclusive beyond shared", lock_ex(second, LOCKFILE_EXCLUSIVE_LOCK, 2558, 1));
    CHECK("unlock ex", unlock_ex(first, 2048, 510) && unlock_ex(second, 2048, 510)
                           && unlock_ex(second, 2558, 1));
    CHECK("exclusive after release", lock_ex(first, LOCKFILE_EXCLUSIVE_LOCK, 2100, 1));
    CHECK("release", unlock_ex(first, 2100, 1));

    CHECK("file APIs start ANSI", AreFileApisANSI());
    SetFileApisToOEM();
    CHECK("file APIs switch to OEM", !AreFileApisANSI());
    SetFileApisToANSI();
    CHECK("file APIs switch back", AreFileApisANSI());

    DWORD sectors = 0, bytes = 0, free_clusters = 0, clusters = 0;
    CHECK("disk free space of the current root",
          GetDiskFreeSpaceA(NULL, &sectors, &bytes, &free_clusters, &clusters)
              && sectors * bytes > 0 && clusters > 0 && free_clusters <= clusters);

    HANDLE heap = GetProcessHeap();
    void *block = HeapAlloc(heap, 0, 64);
    CHECK("heap validates", block && HeapValidate(heap, 0, NULL));
    HeapFree(heap, 0, block);

    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, 4096, NULL);
    char *view = mapping ? MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0) : NULL;
    CHECK("flush a mapped view", view && FlushViewOfFile(view, 0));
    CHECK("flush an unknown base fails", !FlushViewOfFile((void *)0x10, 0));
    if (view)
        UnmapViewOfFile(view);
    if (mapping)
        CloseHandle(mapping);

    // The CRT directory scan is what sqlite's unix-style file control uses.
    CHECK("scan directory", CreateDirectoryW(L"C:\\scan", NULL));
    HANDLE created = CreateFileW(L"C:\\scan\\a.txt", GENERIC_WRITE, 0, NULL,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    CHECK("scan file a", created != INVALID_HANDLE_VALUE && CloseHandle(created));
    created = CreateFileW(L"C:\\scan\\b.txt", GENERIC_WRITE, 0, NULL,
                          CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    CHECK("scan file b", created != INVALID_HANDLE_VALUE && CloseHandle(created));

    struct _finddata64i32_t data;
    intptr_t scan = _findfirst64i32("c:\\scan\\*.txt", &data);
    CHECK("find first", scan != -1 && strcmp(data.name, "a.txt") == 0 && data.size == 0);
    errno = 0;
    CHECK("find next", scan != -1 && _findnext64i32(scan, &data) == 0
                                    && strcmp(data.name, "b.txt") == 0);
    CHECK("find exhausted", scan != -1 && _findnext64i32(scan, &data) == -1 && errno == ENOENT);
    CHECK("find close", scan != -1 && _findclose(scan) == 0);
    errno = 0;
    CHECK("find missing", _findfirst64i32("c:\\scan\\*.jpg", &data) == -1 && errno == ENOENT);

    char *absolute = _fullpath(NULL, "scan\\a.txt", 0);
    CHECK("full path allocates", absolute && strcmp(absolute, "c:\\scan\\a.txt") == 0);
    free(absolute);
    char small[8];
    errno = 0;
    CHECK("full path range", _fullpath(small, "scan\\a.txt", sizeof small) == NULL
                                 && errno == ERANGE);
    char exact[16];
    CHECK("full path exact",
          _fullpath(exact, "scan\\a.txt", sizeof exact) == exact
              && strcmp(exact, "c:\\scan\\a.txt") == 0);

    CHECK("process id", (DWORD)_getpid() == GetCurrentProcessId());
    CHECK("close", CloseHandle(first) && CloseHandle(second));

    // The wide CRT file operations sqlite uses for most of its file names.
    FILE *wide = _wfopen(L"c:\\wops.txt", L"w");
    CHECK("wfopen write", wide != NULL && fwrite("wide", 1, 4, wide) == 4 && fclose(wide) == 0);

    struct _stat64i32 st;
    memset(&st, 0, sizeof st);
    CHECK("wstat regular file",
          _wstat64i32(L"c:\\wops.txt", &st) == 0 && (st.st_mode & _S_IFREG) != 0
              && (st.st_mode & _S_IWRITE) != 0 && st.st_size == 4);
    CHECK("wstat missing fails", _wstat64i32(L"c:\\wops-missing.txt", &st) == -1);

    CHECK("access exists", _access("c:\\wops.txt", 0) == 0 && _waccess(L"c:\\wops.txt", 6) == 0);
    CHECK("access missing fails", _waccess(L"c:\\wops-missing.txt", 0) == -1);

    CHECK("chmod readonly", _wchmod(L"c:\\wops.txt", _S_IREAD) == 0);
    CHECK("readonly write access fails", _waccess(L"c:\\wops.txt", 2) == -1);
    CHECK("readonly write open fails", _wfopen(L"c:\\wops.txt", L"w") == NULL);
    CHECK("readonly unlink fails", _wunlink(L"c:\\wops.txt") == -1);
    errno = 0;
    memset(&st, 0, sizeof st);
    CHECK("readonly stat mode",
          _wstat64i32(L"c:\\wops.txt", &st) == 0 && (st.st_mode & _S_IWRITE) == 0);
    CHECK("chmod writable", _wchmod(L"c:\\wops.txt", _S_IREAD | _S_IWRITE) == 0);
    CHECK("unlink narrow", _unlink("c:\\wops.txt") == 0);
    CHECK("wstat unlinked fails", _wstat64i32(L"c:\\wops.txt", &st) == -1);

    CHECK("wmkdir", _wmkdir(L"c:\\wdir") == 0);
    CHECK("wmkdir existing fails", _wmkdir(L"c:\\wdir") == -1);
    FILE *inner = _wfopen(L"c:\\wdir\\in.txt", L"w");
    CHECK("file inside new directory", inner != NULL && fclose(inner) == 0);
    CHECK("wmkdir missing parent fails", _wmkdir(L"c:\\nope\\dir") == -1);
    CHECK("wunlink inner", _wunlink(L"c:\\wdir\\in.txt") == 0);

#ifdef CRT_WINLOCKS_BREAK
    CHECK("the control fails", sectors == 999);
#endif

    if (failures) {
        printf("crt-winlocks: %u/%u checks failed, first %u\n", failures, checks, failed);
        return failed;
    }
    printf("crt-winlocks: %u/%u checks passed\n", checks, checks);
    return 0;
}
