"""Host memory and disk figures for routing, window sizing and logging.

Uses ``ctypes`` on Windows and ``os.sysconf`` / ``/proc`` elsewhere, so no
extra dependency (psutil) is needed.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_GB = 1024**3


def _windows_memory_status() -> tuple[int, int] | None:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def memory_status() -> tuple[int, int]:
    """(total, available) physical RAM in bytes; (0, 0) when unknown."""
    try:
        if sys.platform == "win32":
            result = _windows_memory_status()
            if result is not None:
                return result
        else:
            page = os.sysconf("SC_PAGE_SIZE")
            total = page * os.sysconf("SC_PHYS_PAGES")
            try:
                avail = page * os.sysconf("SC_AVPHYS_PAGES")
            except (ValueError, OSError):
                avail = total
            return int(total), int(avail)
    except Exception:
        pass
    return 0, 0


def total_ram() -> int:
    return memory_status()[0]


def available_ram() -> int:
    return memory_status()[1]


def process_rss() -> int:
    """Resident set size of this process in bytes (0 when unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
            return 0
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def free_disk(path: str | Path) -> int:
    """Free bytes on the volume holding ``path`` (walks up to an existing dir)."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return 0


def gb(n: int | float | None) -> str:
    if n is None:
        return "?"
    return f"{n / _GB:.2f} GB"
