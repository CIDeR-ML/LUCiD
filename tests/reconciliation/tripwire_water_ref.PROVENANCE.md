# `tripwire_water_ref.npz` — provenance

A reference re-captured without a recorded reason is a rubber stamp with a fresh date. This file
is the reason. **Append to it on every re-capture; do not overwrite.**

---

## 2026-09-24 — re-captured for the first-hit deposit

| | |
|---|---|
| device | **CPU**, pinned by `tripwire_capture.py` |
| host | Intel Xeon Gold 5118 (host of a `turing` node) |
| `scalar.q_l2` | 581.7201 → **540.6970** (−7.1%) |
| `scalar.q_sum` | 17215.2617 → **15670.7793** (−9.0%) |
| `wavelength.q_l2` | 771.2139 → **716.5498** |

### What moved it

The deposit gained three pieces of physics in one commit: an ahead gate (no charge on a sensor
whose closest approach lies behind the photon), a first-hit survival product (a photon deposits at
most once in total), and first-entry geometry (a ray threading several spheres stops at the first).
Each was isolated by capturing the full change and the full change minus that one piece, every tree
on the same node and CPU:

| tree | `scalar.q_sum` | `scalar.q_l2` |
|---|---|---|
| before the deposit physics | 17215.2188 | 581.7166 |
| full change | 15670.7793 | 540.6970 |
| full minus the cap | 16295.4922 | 547.7850 |
| full minus the gate | 16311.0723 | 565.8101 |
| full minus first-entry | 15666.7891 | 540.4186 |

Removing the gate or the cap alone restores about 3.9% of the charge each; first-entry is worth
0.03%. Those leave-one-out effects sum to about 7.4%, against 9.0% for the whole change: the gate and
the cap overlap, since both remove charge from the same over-counted photons. The "before" row
reproduces the previous reference's 581.72 to the host difference measured in the 2026-09-23 entry,
so nothing else moved.

The 400-node overlap table that landed just before does not reach this path: the tripwire runs the
step function (`temperature=None`), which never reads the table. The recapture on the final tree
reads 540.6970, identical to the attribution run made before the table changed.

### One side effect worth knowing

The Fisher information on `sensor_reflection_rate` roughly **doubles**, 49.7 → 102.1, while the
other columns fall 3–10% with the charge. Without the ahead gate, light reflected off a PMT was
deposited on sensors BEHIND the photon -- the tube it just left and its neighbours; 38,447 such
deposits in a 100k-photon sample leaving PMTs on SK_like -- which hid the reflection from the rest of
the detector. With the gate the reflected light reaches the rest of the
detector, and the calibration can see it.

The per-column tolerances from the previous entry were set from cross-host floors, which are a
property of the arithmetic rather than of the reference's values, so they are unchanged.

---

## 2026-09-23 — re-captured, and the tolerance repaired

| | |
|---|---|
| device | **CPU** — forced by `tripwire_capture.py` before importing jax |
| host | AMD (`milano`); see the portability section, which is why the host is recorded at all |
| `scalar.q_l2` | 581.5638 → **581.7201** (+0.027%) |
| `scalar.q_sum` | 17206.0586 → **17215.2617** |

### What justifies the new values

A CPU bisect over the commits post-dating the previous reference (`b93e2cc`), device held fixed:

| `lucid/` at | `scalar.q_l2` | |
|---|---|---|
| `b93e2cc` | 581.5645 | reproduces the OLD reference to 1.2e-6 |
| `ac8861c` = `b8c266e^` | 581.5645 | unchanged: everything in the window is inert |
| **`b8c266e`** | **581.7201** | ← the change |
| `86a1871` | 581.7201 | current main, three runs identical |

**`b8c266e` — "fix: scatter direction transform was transposed (`frame @` → `frame.T @`)" —
accounts for the entire drift.** `create_local_frame` returns basis vectors as ROWS, so the
local→world rotation of a sampled scatter direction is `frame.T @ local_dir`; the scattering paths
used `frame @ local_dir`, the inverse rotation, building the scatter cone about the wrong axis.

That is an intended physics correction, it is the only contributor, and the tripwire firing on it
was the instrument working. Accepting the new values means accepting that fix, not accepting an
unexplained drift.

Measuring `b8c266e^` is what makes this airtight, and it replaces an earlier attempt that read
584.4250 off `ec34a8d` and could not explain why that value never appears at HEAD. The reason is
that `ec34a8d` and `b93e2cc` are **siblings**: `git merge-base --is-ancestor` is false in both
directions, so a tree checked out at `ec34a8d` simply lacks `b93e2cc`'s `scalar_mix` default and
the number measured the missing default rather than the commit. A bisect step is a measurement of
a commit only when that commit is a descendant of the step before it.

### Why the device is pinned

**This digest is device-dependent at 7.8e-4, roughly 8× its own `1e-4` tolerance** — at HEAD,
`q_l2` is 581.7201 on CPU and 581.2667 on GPU.

`tripwire_capture.py` used to pin nothing. `tests/conftest.py` forces CPU and the pytest wrapper
passes `os.environ` into the capture subprocess, so the same reference silently measured **CPU
under pytest and GPU when run by hand**. The script now sets all three variables `conftest` sets,
itself, before importing jax.

This was not academic: an earlier attempt to diagnose the drift ran its bisect on GPU and concluded
the reference matched no device at any commit, i.e. that it was unreproducible. It was reproducible
all along, on the device it was captured on.

### Why `fisher_diag` now has one tolerance per column

The recapture alone did not make the test pass. `fisher_diag` still failed, and the diagnostic fact
is that **it failed against a reference captured at its own commit** (`b93e2cc`), while `q_l2`
reproduced there to 1.2e-6. That is not drift; it is a bound nothing can satisfy.

Two candidates, one refuted and one confirmed, both by measurement:

* **Thread count — refuted.** 1, 2, 4, 8, 16, 32 cores on one node: `q_l2` bit-identical at every
  count, `fisher_diag` spread ≤ 2e-5.
* **CPU architecture — confirmed.** Identical code and seeds, AMD (`milano`) vs the Intel Xeon Gold
  5118 host of a `turing` node, both on CPU:

  | column | cross-host relative difference |
  |---|---|
  | `mie_scatter_length` | 4.1e-4 |
  | `wall_reflection_rate` | 3.4e-4 |
  | `g` | 3.1e-4 |
  | `sensor_reflection_rate` | 1.3e-4 |
  | `scatter_length` | 3.8e-5 |
  | `absorption_length` | 2.3e-5 |
  | `qe` | 4.6e-6 |
  | `scalar.q_l2` | **6.0e-6** |

Four of seven columns exceed the old blanket `1e-4`. `fisher_diag` is `(adJ**2).sum` over 10764
sensors, and a weakly-determined column is built from tiny per-sensor entries, so its float32
relative error is far larger than a well-determined one's — the floor tracks column magnitude
almost monotonically. One bound for all seven is either too loose for `qe` or too tight for `mie`.

Each column now carries its own `rtol` at **at least 20× its measured floor**, rounded up to one
significant figure, which lands each between 20.6× and 26.3×. The norm assertions stay at `1e-4`,
where their floor is 6.0e-6, a 17× margin.

A single blanket `5e-3` was tried first and rejected on this same data: under `b8c266e`,
`sensor_reflection_rate` moves 4.3e-3 and would have passed. Note this is a claim about what the
two sets *detect on that data*, not about the bounds themselves — three per-column bounds (`g` and
`wall_reflection_rate` at 7e-3, `mie_scatter_length` at 1e-2) are looser than 5e-3. What matters is
that on the one real change available to test against, the per-column set catches a strict superset
of what the blanket one does.

**If a toolchain bump breaks this, re-measure the floor and widen from the measurement.** Do not
raise a bound to make a test pass. A jaxlib/XLA upgrade was never probed and is a bigger lever on
this floor than the CPU vendor. Capturing in float64 would remove the problem entirely, but it
stops the tripwire pinning the float32 path production runs, which is why it stays float32.

### How long it had been failing

Since `b8c266e` (2026-07-15). Unnoticed because `test_tripwire.py` is in `conftest.py`'s
`_SLOW_FILES`, so `pytest tests/` — what CI runs — never executed it. The one instrument built to
catch forward-model drift was excluded from the only place it would be checked automatically. See
the CI change that lands with this.
