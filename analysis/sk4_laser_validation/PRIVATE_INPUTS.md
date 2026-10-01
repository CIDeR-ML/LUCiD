# Private-input boundary

This validation package contains no SKDetSim source code and no files copied
from an SKDetSim installation.

The following inputs remain external and must not be committed to this public
repository:

- SKDetSim source, build products, job cards, logs, and executables;
- ZBS, HBOOK, ROOT, or converted event-level output;
- PMT connection tables and per-PMT QE tables;
- acrylic, photocathode, reflectance, or other SKDetSim optical lookup tables;
- derived portable copies of those lookup tables; and
- compact `.npz` response/deposition caches containing SKDetSim events.

The committed PNG figures and JSON files contain only plotted or aggregate
comparison results. The Python files in this directory are original LUCiD
simulation and comparison code. They accept collaboration inputs by path but
do not embed or redistribute them.

The local `.gitignore` blocks the known private/bulky input formats. Before
publishing updates, also inspect staged files with:

```bash
git diff --cached --name-only
git diff --cached --numstat
```
