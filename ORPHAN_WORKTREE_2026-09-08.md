# The orphan worktree is a stale copy, and rescuing it would reintroduce a bug

The Status card carried this as the project's last outstanding file:

> **One file is still uncommitted and untracked.**
> `…\.claude\worktrees\mode-transition-analysis\mode_transition_analysis.py` (400 lines,
> 2026-07-28) — the Markov / conditional-probability formalization, on an orphan
> `worktree-mode-transition-analysis` branch whose only commit is an empty
> "Initial commit".

**That line is stale, and acting on it would have been a regression.**

## What is actually true

`mode_transition_analysis.py` **is tracked in `master`** and has been since **2026-06-29**,
across two commits:

```
cb24728  feat(mode-dynamics): J-electrode mode-transition, initiation,
         segment-velocity & geometry analyses
689ae0d  feat(mode-dynamics): dead-zone valley Mode A/B cut + 10-segment
         detector primary
```

The tracked version is **511 lines** against the orphan's 400. It additionally imports
`arc_resets`, takes a `fname` argument selecting the detector, and defaults to
`npoint_restrikes.csv` — the 10-segment whole-arc detector its own docstring calls
*"the primary/more-trustworthy detector"* — where the orphan copy is hard-wired to the tip
detector `re_strikes_qa.csv`.

The orphan file is a **superseded snapshot**, not lost work.

## And it carries a bug that master already fixed

The orphan's `find_recordings` excludes only

```python
if "ARCHIVE" in csv or "reference_5000" in csv:
    continue
```

Master's adds the rule that actually settles it — an authoritative recording's results
directory is a **direct child** of the group root, so `rel.count("/") != 1` is a copy:

```python
rel = os.path.relpath(csv, root).replace("\\", "/")
if rel.count("/") != 1:
    continue
```

There are **45** `re_strikes_qa.csv` files under `Desktop\Plasma Arc` and **8 real
recordings**. The rest are `__pre_3p2_backup__`, `__pre_calib_rescale_backup__`,
`__pre_resegment_snapshot__`, `_FULL_BACKUP_pre_resegment_2026-06-26`,
`_PRE_ROBUSTNESS_REPROCESS_2026-06-29`, the figure package's `_tree`, and `ARCHIVE`.

**Measured by running both copies on the same tree**, on the tip detector so the two are
comparable:

| pooled, CO₂ (J-electrode) | orphan copy | master's rule |
|---|---|---|
| recordings discovered | 43 | **8** |
| `n_restrikes` | 11 096 | **1 821** |
| `p_B` | 0.6467 | **0.5014** |
| `P(B\|A)` | 0.6291 | **0.5856** |
| `P(B\|B)` | 0.6567 | **0.4176** |

Read the last two rows together. Under the orphan copy `P(B|B)` **>** `P(B|A)`, i.e. Mode B
looks **persistent**. Under master's rule `P(B|B)` **<** `P(B|A)`, i.e. Mode B is
**anti-persistent**. **The duplication does not merely inflate `n` — it reverses the sign of
the memory effect**, and a 6.1× inflated `n` would have made every χ² p-value look far more
significant than the data supports. Some of the duplicated copies also predate the 25 JUN
calibration rescale that moved every velocity in this study by 1.333×.

**The control**: the N₂ (blade) group is **identical under both copies** (`n_restrikes`
1411, `p_B` 0.52587, `P(B|A)` 0.46707, `P(B|B)` 0.58030), because its only duplicates sit
under `ARCHIVE/`, which the old filter already caught. The difference is confined to
exactly the rows that had duplicates.

## What was done

1. Nothing was rescued. The orphan file is superseded; copying it into `master` was
   attempted, caught by `git status` showing the path as **modified rather than
   untracked**, and reverted.
2. `test_mode_transition_discovery.py` pins master's exclusion rule, which had no test.
   It includes a **legal-input check** (a recording whose own name contains underscores
   must survive) and a **re-introduction check** that restores the orphan's permissive
   filter and asserts the discovery count jumps back to 43.
3. The Status card line is corrected.

## What is NOT claimed

- **No number is retracted.** Master's version is what has always been on disk, and it has
  the correct filter. The orphan copy appears never to have been run.
- **Nothing about which detector is right.** Master defaults to `npoint_restrikes.csv` and
  the comparison above is on `re_strikes_qa.csv` for both copies, precisely so the detector
  is held fixed while the discovery rule varies.
- **The statistics themselves are unreviewed.** Only the input set is checked here.
