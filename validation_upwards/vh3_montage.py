"""V-H3 for one cine with EXACTLY the pipeline detector (run_blade_upwards constants): background,
N, S, thresholds, relative floor; random detected frames + random frames of any kind."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cine_frames as cf  # noqa: E402
import run_blade_upwards as rb  # noqa: E402

cine, out = sys.argv[1], Path(sys.argv[2])
seed = sys.argv[3] if len(sys.argv) > 3 else "5"
src = cf.open_cine(cine)
bg, N, info = cf.measure_background(src)
floor = max(N, info["smear_S"])
T_low, T_high = rb.LOW_K * floor, rb.HIGH_K * rb.LOW_K * floor
bgp = Path(tempfile.gettempdir()) / (Path(cine).stem + "_bg.npy")
np.save(bgp, bg)
print(Path(cine).stem, f"N {N:.0f} S {info['smear_S']:.0f} T {T_low:.0f}/{T_high:.0f} sat_frames {info['sat_frames']}")
for sel in ("detected", "all"):
    subprocess.run([sys.executable, "-X", "utf8", str(HERE / "validation_upwards" / "mask_montage.py"), cine,
                    f"{T_high}", str(out / f"VH3_{Path(cine).stem}_{sel}.png"), "--low", f"{T_low}",
                    "--low-frac", str(rb.LOW_FRAC), "--background", str(bgp), "--select", sel, "--n", "18",
                    "--pool", "3000", "--seed", seed, "--crop", "60,256,20,236"], check=True)
