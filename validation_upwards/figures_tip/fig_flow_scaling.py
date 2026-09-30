"""Leading-tip velocity against the nozzle exit velocity U, for the pure gases (Air, Argon, CO2).

U = Re * nu / d, with d = 0.6 mm (the '06mmNozzle' in every file name) and nu = mu/rho from CoolProp at 293.15 K and
101325 Pa. ASSUMPTIONS, stated in the caption:
- the Re labels are nozzle Reynolds numbers built on the same nu (the Air folder's note, 'multiply the Re by
  mu_Ar/mu_Air', implies Re is proportional to 1/mu at a set flow, i.e. Re = 4 mdot / (pi d mu), which is
  U d / nu);
- room-temperature, atmospheric properties. U reaches Mach 0.6, so the exit density, and with it U, is not exactly
  this. A gas-dependent constant factor in U moves a gas's line but not its slope.
The mixtures are left out: their viscosity needs a mixing rule, which is not taken from memory here.

(a) log-log, with the fitted slope per gas (OLS on per-trial points, standard error). (b) v / U on linear axes.
Air is drawn as labelled (filled) and with the folder note's x1.225 correction (open, not fitted).

usage: python fig_flow_scaling.py
"""
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import _common as C

D_NOZZLE_M = 0.6e-3
T_K, P_PA = 293.15, 101325.0
PURE = {"Air": "Air", "Argon": "Argon", "CO2": "CO2"}
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
    D["v"] = D["local_vel_ms_med"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    rows, fits = [], []
    for gas in [g for g in C.GASES if g in PURE]:
        d = D[D["gas"] == gas]
        c = d.groupby("Re").agg(U=("U", "mean"), v=("v", "mean"), sd=("v", "std"), Ma=("Ma", "mean")).reset_index()
        if gas in NO_FIT:
            # the spot regime at low Re (complete-arc fraction 0.04-0.12, one recording with a median tip velocity of
            # exactly 0) is a different regime, not a point on a power law; no fit is drawn rather than drop points
            b = se = np.nan
            fits.append(f"{gas}: not fitted (spot regime at low Re; {int((d['v'] <= 0).sum())} recording(s) with "
                        f"median velocity 0)")
            lab = f"{C.LABEL[gas]} (not fitted)"
        else:
            X = sm.add_constant(np.log(d["U"].to_numpy()))
            f = sm.OLS(np.log(d["v"].to_numpy()), X).fit()
            b, se = f.params[1], f.bse[1]
            fits.append(f"{gas}: slope {b:.2f} ± {se:.2f} (n = {len(d)} recordings)")
            lab = f"{C.LABEL[gas]} (slope {b:.2f} ± {se:.2f})"
        for ax, y, e in ((a1, c["v"], c["sd"]), (a2, c["v"] / c["U"], c["sd"] / c["U"])):
            ax.errorbar(c["U"], y, yerr=e, fmt=C.MARK[gas], color=C.COLOR[gas], ms=4.5, capsize=2, label=lab)
        if gas not in NO_FIT:
            uu = np.linspace(d["U"].min(), d["U"].max(), 50)
            a1.plot(uu, np.exp(f.params[0]) * uu ** b, "-", color=C.COLOR[gas], lw=1)
        if gas == "Air":
            for ax, y in ((a1, c["v"]), (a2, c["v"] / (c["U"] * note))):
                ax.plot(c["U"] * note, y, C.MARK[gas], mfc="none", mec=C.COLOR[gas], ms=4.5,
                        label="Air, Re × μ$_{Ar}$/μ$_{Air}$" if ax is a1 else None)
        rows.append(c.assign(gas=gas, slope=b, slope_se=se))
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
           f"mu_Ar/mu_Air {note:.4f}; input aggregate_annotated/per_recording.csv",
           f"""# Tip velocity against nozzle exit velocity

(a) Median local tip velocity (mean ± sd over trials) against the nozzle exit velocity U = Re ν / d (d = 0.6 mm,
ν from CoolProp at 293 K, 1 atm), log–log, with the OLS power-law slope per gas (per-trial points; ± standard error).
(b) The ratio v / U. Filled: Re as labelled. Open (Air only, not fitted): Re multiplied by μ_Ar/μ_Air = {note:.3f},
the correction the Air folder's INFORMATION.txt says may apply. Assumptions: the Re labels are U d / ν with these
properties; room-temperature, atmospheric properties although U reaches Mach {D['Ma'].max():.2f}. A gas-dependent
constant factor in U would move a gas's points, not its slope. The CO₂ mixtures are omitted (a mixture viscosity
needs a mixing rule). Fits: {'; '.join(fits)}.""")


if __name__ == "__main__":
    main()
