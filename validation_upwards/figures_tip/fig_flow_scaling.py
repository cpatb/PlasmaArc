"""Leading-tip velocity against the nozzle exit velocity U, for the pure gases (Air, Argon, CO2).

U = Re * nu / d, with d = 0.6 mm (the '06mmNozzle' in every file name) and nu = mu/rho from CoolProp at 293.15 K and
101325 Pa. ASSUMPTIONS, stated in the caption:
- the Re labels are nozzle Reynolds numbers built on the same nu (the Air folder's note, 'multiply the Re by
  mu_Ar/mu_Air', implies Re is proportional to 1/mu at a set flow, i.e. Re = 4 mdot / (pi d mu), which is
  U d / nu);
- room-temperature, atmospheric properties. U reaches Mach 0.6, so the exit density, and with it U, is not exactly
  this; a Mach-dependent correction moves the slopes by a few hundredths (referee 2026-09-30). Within a gas U is
  proportional to Re, so the slopes carry no information beyond "rises with Re"; the between-gas comparison does.
The mixtures are left out: their viscosity needs a mixing rule, which is not taken from memory here.

(a) log-log, a power-law slope fitted to each of the four velocity estimators (range reported). (b) v / U on linear axes.
Air is drawn as labelled (filled) and with the folder note's x1.225 correction (open, not fitted).

usage: python fig_flow_scaling.py
"""
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

D_NOZZLE_M = 0.6e-3
T_K, P_PA = 293.15, 101325.0
PURE = {"Air": "Air", "Argon": "Argon", "CO2": "CO2"}
EST = ["med", "med_nz", "mean", "agg"]
NO_FIT = {"CO2"}   # see the loop: its low-Re points are the spot regime, not a power law


def props(fluid):
    mu = CP.PropsSI("V", "T", T_K, "P", P_PA, fluid)
    rho = CP.PropsSI("D", "T", T_K, "P", P_PA, fluid)
    return mu, rho, CP.PropsSI("A", "T", T_K, "P", P_PA, fluid)


def main():
    C.style()
    D = C.per_run()
    D = D[D["gas"].isin(PURE)].copy()
    pr = {g: props(f) for g, f in PURE.items()}
    note = pr["Argon"][0] / pr["Air"][0]
    print("properties at 293.15 K, 1 atm: " + "; ".join(f"{g} nu {m / r:.4e} m2/s, c {c:.1f} m/s"
                                                         for g, (m, r, c) in pr.items()) + f"; mu_Ar/mu_Air {note:.4f}")
    D["U"] = D["Re"] * D["gas"].map({g: m / r for g, (m, r, _) in pr.items()}) / D_NOZZLE_M
    D["Ma"] = D["U"] / D["gas"].map({g: c for g, (_, _, c) in pr.items()})
    V = C.velocity_estimators()
    V = V[V["gas"].isin(PURE)].copy()
    V["U"] = V["Re"] * V["gas"].map({g: m / r for g, (m, r, _) in pr.items()}) / D_NOZZLE_M
    V["mid"] = V[EST].mean(axis=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    rows, fits = [], []
    for gas in [g for g in C.GASES if g in PURE]:
        c = V[V["gas"] == gas].sort_values("Re")
        # the tip velocity depends on the estimator where the 5-frame velocity is quantised (C.velocity_estimators):
        # points = mean of the four estimators, bars = their range; a slope is fitted to EACH estimator and the range
        # of slopes is reported, so no estimator is chosen after the fact
        if gas in NO_FIT:
            slopes = []
            fits.append(f"{gas}: not fitted (spot regime at low Re)")
            lab = f"{C.LABEL[gas]} (not fitted)"
        else:
            slopes = [float(np.polyfit(np.log(c["U"]), np.log(c[e]), 1)[0]) for e in EST if (c[e] > 0).all()]
            fits.append(f"{gas}: slope {min(slopes):.2f}-{max(slopes):.2f} over {len(slopes)} estimators; v/U "
                        f"{100 * (c['lo'] / c['U']).min():.2f}-{100 * (c['hi'] / c['U']).max():.2f} %")
            lab = f"{C.LABEL[gas]} (slope {min(slopes):.2f}–{max(slopes):.2f})"
        for ax, s in ((a1, 1.0), (a2, 1.0 / c["U"])):
            ax.errorbar(c["U"], c["mid"] * s, yerr=[(c["mid"] - c["lo"]) * s, (c["hi"] - c["mid"]) * s],
                        fmt=C.MARK[gas], color=C.COLOR[gas], ms=4.5, capsize=2, label=lab)
        if gas == "Air":
            for ax, y in ((a1, c["mid"]), (a2, c["mid"] / (c["U"] * note))):
                ax.plot(c["U"] * note, y, C.MARK[gas], mfc="none", mec=C.COLOR[gas], ms=4.5,
                        label="Air, Re × μ$_{Ar}$/μ$_{Air}$" if ax is a1 else None)
        rows.append(c.assign(slope_min=min(slopes) if slopes else np.nan, slope_max=max(slopes) if slopes else np.nan))
    Ma_max = float(D["Ma"].max())
    print("\n".join(fits))
    a1.set_xscale("log"), a1.set_yscale("log")
    a1.set_xlabel("nozzle exit velocity U (m/s)"), a1.set_ylabel("tip velocity (m/s)")
    a2.set_xlabel("nozzle exit velocity U (m/s)"), a2.set_ylabel("tip velocity / U")
    a2.set_xlim(0, None), a2.set_ylim(0, None)
    a1.legend(fontsize=7, frameon=False, loc="upper left")
    for ax, t in zip((a1, a2), "ab"):
        ax.text(0.97, 0.05, f"({t})", transform=ax.transAxes, ha="right")
    fig.tight_layout()
    T = pd.concat(rows, ignore_index=True)
    C.save(fig, "fig_flow_scaling", T,
           f"validation_upwards/figures_tip/fig_flow_scaling.py; d {D_NOZZLE_M} m; CoolProp at {T_K} K {P_PA} Pa; "
           f"mu_Ar/mu_Air {note:.4f}; velocity = 4 estimators; inputs per_recording.csv, frames_cache.parquet",
           f"""# Tip velocity against nozzle exit velocity

(a) Tip velocity against the nozzle exit velocity U = Re ν / d (d = 0.6 mm, ν from CoolProp at 293 K, 1 atm),
log–log. Points: mean of four velocity estimators; bars: their range (the 5-frame velocity is quantised, so at low
speed the estimator matters; see fig_tip_kinematics). A power-law slope is fitted to each estimator separately and
the range is given. (b) The ratio v / U. Filled: Re as labelled. Open (Air only, not fitted): Re multiplied by
μ_Ar/μ_Air = {note:.3f}, the correction the Air folder's INFORMATION.txt says may apply. Assumptions, unverified: the
Re labels are U d / ν with these properties; U reaches Mach {Ma_max:.2f}, and a Mach-dependent correction would
change the slopes by a few hundredths; the nozzle-to-gap distance is unknown, so the fraction of U is partly set by
jet decay before the arc. Within a gas U is proportional to Re, so a slope only restates that the tip speeds up with
Re; the information is in the comparison BETWEEN gases. Only the pure gases are drawn (mixture viscosities need a
mixing rule). Fits: {'; '.join(fits)}.""")

if __name__ == "__main__":
    main()
