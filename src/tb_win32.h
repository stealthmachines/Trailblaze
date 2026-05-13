/*
 * src/tb_win32.h — Windows POSIX compatibility shim for TRAILBLAZE
 *
 * Covers: mmap/munmap, open/close/fstat, clock_gettime, strdup, mkdir,
 *         pthread_mutex (via CRITICAL_SECTION), dirent, __attribute__
 *
 * Include once per translation unit under #ifdef _WIN32 before any POSIX
 * headers.  Linux/macOS paths are completely unaffected — they never see
 * this file.
 *
 * Usage pattern (copy-paste into each .c file that includes POSIX headers):
 *
 *   #ifdef _WIN32
 *   #  include "../src/tb_win32.h"   // adjust relative path as needed
 *   #else
 *   #  include <sys/mman.h>
 *   #  include <sys/stat.h>
 *   #  include <unistd.h>
 *   #  include <fcntl.h>
 *   #  include <dirent.h>
 *   #  include <pthread.h>
 *   #endif
 */
#pragma once
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>         /* _open, _close, _read, _lseeki64 */
#include <fcntl.h>      /* _O_RDONLY, _O_BINARY             */
#include <sys/types.h>
#include <sys/stat.h>   /* _stat64, _fstat64                */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <direct.h>     /* _mkdir                           */

/* ── Suppress MSVC POSIX-name deprecation noise ─────────────────────────── */
#pragma warning(disable: 4996)

/* ── GCC attributes — no-op on MSVC ─────────────────────────────────────── */
#ifndef __attribute__
#  define __attribute__(x)
#endif

/* ── mmap / munmap ───────────────────────────────────────────────────────── */
#define PROT_READ   1
#define MAP_PRIVATE 2
#define MAP_FAILED  ((void*)(uintptr_t)(~0))

static inline void *mmap(void *addr, size_t len,
                          int prot, int flags,
                          int fd, long offset)
{
    (void)addr; (void)prot; (void)flags; (void)offset;
    HANDLE hfile = (HANDLE)_get_osfhandle(fd);
    if (hfile == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE hmap = CreateFileMappingA(hfile, NULL, PAGE_READONLY,
                                     (DWORD)(len >> 32),
                                     (DWORD)(len & 0xFFFFFFFFu),
                                     NULL);
    if (!hmap) return MAP_FAILED;
    void *p = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, len);
    CloseHandle(hmap);
    return p ? p : MAP_FAILED;
}

static inline int munmap(void *addr, size_t len)
{
    (void)len;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

/* ── POSIX open / close ──────────────────────────────────────────────────── */
#define open(path, flags)   _open((path), (flags) | _O_BINARY)
#define close(fd)           _close(fd)
#define O_RDONLY            _O_RDONLY

/* ── stat / fstat ────────────────────────────────────────────────────────── */
#define fstat(fd, st)       _fstat64((fd), (st))
typedef struct _stat64      tb_stat_t;
#define stat                _stat64

/* ── strdup ──────────────────────────────────────────────────────────────── */
#ifndef strdup
#  define strdup _strdup
#endif

/* ── mkdir (POSIX takes mode arg; Windows _mkdir doesn't) ───────────────── */
static inline int tb_mkdir(const char *path, int mode)
{
    (void)mode;
    return _mkdir(path);
}
#define mkdir(p, m) tb_mkdir((p), (m))

/* ── POSIX clock_gettime / struct timespec ───────────────────────────────── *
 * MSVC ships <time.h> with struct timespec since VS 2015 but not             *
 * clock_gettime.  Implement via QueryPerformanceCounter (QPC) for            *
 * CLOCK_MONOTONIC and GetSystemTimePreciseAsFileTime for CLOCK_REALTIME.     */

#ifndef CLOCK_REALTIME
#  define CLOCK_REALTIME  0
#endif
#ifndef CLOCK_MONOTONIC
#  define CLOCK_MONOTONIC 1
#endif

/* struct timespec is defined in <time.h> under MSVC (VS2015+) — no redef  */
typedef int clockid_t;

static inline int clock_gettime(clockid_t clk, struct timespec *tp)
{
    if (clk == CLOCK_MONOTONIC) {
        LARGE_INTEGER freq, cnt;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&cnt);
        tp->tv_sec  = (time_t)(cnt.QuadPart / freq.QuadPart);
        tp->tv_nsec = (long)(((cnt.QuadPart % freq.QuadPart) * 1000000000LL)
                             / freq.QuadPart);
    } else {
        /* CLOCK_REALTIME via FILETIME (100-ns ticks since 1601-01-01)
         * Use GetSystemTimeAsFileTime (available XP+; Precise variant is Win8+) */
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        ULARGE_INTEGER ul;
        ul.LowPart  = ft.dwLowDateTime;
        ul.HighPart = ft.dwHighDateTime;
        /* Convert to Unix epoch: 116444736000000000 100-ns ticks 1601→1970 */
        ul.QuadPart -= 116444736000000000ULL;
        tp->tv_sec  = (time_t)(ul.QuadPart / 10000000ULL);
        tp->tv_nsec = (long)((ul.QuadPart % 10000000ULL) * 100);
    }
    return 0;
}

/* ── pthread_mutex — thin wrapper over CRITICAL_SECTION ─────────────────── *
 * Only implements the subset used by tb_infer.c:                             *
 *   pthread_mutex_t, pthread_mutex_init/lock/unlock/destroy                  */

typedef CRITICAL_SECTION pthread_mutex_t;
typedef void             *pthread_mutexattr_t;   /* ignored               */

static inline int pthread_mutex_init(pthread_mutex_t *m,
                                     const pthread_mutexattr_t *attr)
{
    (void)attr;
    InitializeCriticalSection(m);
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *m)
{
    EnterCriticalSection(m);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *m)
{
    LeaveCriticalSection(m);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *m)
{
    DeleteCriticalSection(m);
    return 0;
}

/* ── pthread_t / pthread_create / pthread_join — for HTTP worker threads ── */
typedef HANDLE pthread_t;
typedef void  *pthread_attr_t;   /* ignored */

static inline int pthread_create(pthread_t *thr,
                                  const pthread_attr_t *attr,
                                  void *(*fn)(void *), void *arg)
{
    (void)attr;
    *thr = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(void*)fn,
                        arg, 0, NULL);
    return (*thr == NULL) ? -1 : 0;
}

static inline int pthread_join(pthread_t thr, void **retval)
{
    (void)retval;
    WaitForSingleObject(thr, INFINITE);
    CloseHandle(thr);
    return 0;
}

static inline void pthread_detach(pthread_t thr)
{
    CloseHandle(thr);
}

/* ── dirent.h — minimal directory iteration ─────────────────────────────── */
#ifndef _TB_DIRENT_DEFINED
#define _TB_DIRENT_DEFINED

#define DT_UNKNOWN 0
#define DT_REG     8
#define DT_DIR     4

struct dirent {
    char d_name[MAX_PATH];
    unsigned char d_type;
};

typedef struct {
    HANDLE          handle;
    WIN32_FIND_DATAA fdata;
    struct dirent   cur;
    int             first;
} DIR;

static inline DIR *opendir(const char *path)
{
    char pat[MAX_PATH];
    snprintf(pat, sizeof(pat), "%s\\*", path);
    DIR *d = (DIR *)calloc(1, sizeof(DIR));
    if (!d) return NULL;
    d->handle = FindFirstFileA(pat, &d->fdata);
    if (d->handle == INVALID_HANDLE_VALUE) { free(d); return NULL; }
    d->first = 1;
    return d;
}

static inline struct dirent *readdir(DIR *d)
{
    if (d->first) {
        d->first = 0;
    } else {
        if (!FindNextFileA(d->handle, &d->fdata)) return NULL;
    }
    strncpy(d->cur.d_name, d->fdata.cFileName, MAX_PATH - 1);
    d->cur.d_type = (d->fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                    ? DT_DIR : DT_REG;
    return &d->cur;
}

static inline int closedir(DIR *d)
{
    if (d) { FindClose(d->handle); free(d); }
    return 0;
}

#endif /* _TB_DIRENT_DEFINED */

/* ── Network sockets — winsock2 already included via windows.h ──────────── *
 * Winsock uses SOCKET (uintptr_t) not int fds.  The HTTP server in           *
 * tb_infer.c is guarded with typedef tb_socket_t so no changes needed.       */
#define ssize_t intptr_t

/* ── popen / pclose ──────────────────────────────────────────────────────── *
 * Windows provides _popen/_pclose with the same semantics.                   */
#ifndef popen
#  define popen  _popen
#  define pclose _pclose
#endif

/* ── 64-bit fseek/ftell — MSVC long is 32-bit even on 64-bit Windows ─────── *
 * fseek(f, offset, whence) fails silently for files > 2 GB on MSVC.          *
 * Use _fseeki64 / _ftelli64 instead.  We do NOT globally #define fseek       *
 * because it would break small-offset callers that pass int literals.         *
 * Callers that need large-file support should use tb_fseek64 / tb_ftell64.    */
#define tb_fseek64(f, off, whence)  _fseeki64((f), (__int64)(off), (whence))
#define tb_ftell64(f)               ((__int64)_ftelli64(f))

#endif /* _WIN32 */
