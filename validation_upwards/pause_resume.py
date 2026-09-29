"""Suspend or resume every BLADE-upwards pipeline process (Windows NtSuspendProcess / NtResumeProcess).

usage: pause_resume.py pause|resume
pause writes the PIDs to E:/BladeReactor_Upwards_analysis/paused_pids.json; resume reads them back.
"""
import ctypes
import json
import subprocess
import sys
from pathlib import Path

STATE = Path(r"E:\BladeReactor_Upwards_analysis\paused_pids.json")
PATTERN = "run_blade_upwards|analyze_blade|gap_calibration|event_velocity|qa_restrike|fourier_analysis|overnight_driver"
PROCESS_SUSPEND_RESUME = 0x0800
ntdll, k32 = ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32")


def find():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '%s' -and "
          "$_.ProcessId -ne %d } | ForEach-Object { \"$($_.ProcessId)`t$($_.Name)\" }" % (PATTERN, __import__('os').getpid()))
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    return [(int(p), n) for p, n in (ln.split("\t") for ln in out.split("\n") if "\t" in ln)]


def act(pid, fn):
    h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        return False
    rc = fn(h)
    k32.CloseHandle(h)
    return rc == 0


if sys.argv[1] == "pause":
    procs = find()
    ok = [(p, n) for p, n in procs if act(p, ntdll.NtSuspendProcess)]
    STATE.write_text(json.dumps(ok))
    print(f"suspended {len(ok)}/{len(procs)} processes:", ok)
else:
    procs = [tuple(x) for x in json.loads(STATE.read_text())]
    ok = [p for p, _ in procs if act(p, ntdll.NtResumeProcess)]
    print(f"resumed {len(ok)}/{len(procs)} processes")
