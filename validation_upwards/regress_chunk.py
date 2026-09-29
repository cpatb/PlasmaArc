"""Re-run one analyze chunk with the CURRENT code into a scratch dir and md5-compare measurements.csv.

usage: regress_chunk.py <analyze_cNN.log> <scratch_out_dir>
Takes the exact command line recorded in the log (first '>>>' line), swaps --out-dir.
"""
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

log, out = Path(sys.argv[1]), Path(sys.argv[2])
line = next(ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith(">>> "))
body = line[4:]
k = body.index(".py ") + 3                  # the script path has a space ("Plasma Arc"); logged unquoted
args = [body[:k]] + shlex.split(body[k:], posix=False)   # remaining paths have no spaces
i = args.index("--out-dir")
orig = Path(args[i + 1])
args[i + 1] = str(out)
r = subprocess.run([sys.executable, "-X", "utf8"] + args, capture_output=True, text=True)
if r.returncode:
    print(r.stdout[-2000:], r.stderr[-2000:])
    sys.exit(r.returncode)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


a, b = md5(orig / "measurements.csv"), md5(out / "measurements.csv")
print("orig", a, "\nnew ", b, "\nIDENTICAL" if a == b else "\nDIFFERENT")
