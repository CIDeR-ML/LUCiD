"""Phase-0 reconciliation tripwire as a (slow) regression test. Subprocess-isolated to avoid
JAX jaxpr-cache collisions with the rest of the suite. Asserts the water-mode forward / AD-Fisher /
leaf-order references in tripwire_water_ref.npz still hold — the gate every merge phase must pass.
Regenerate the reference deliberately with CAPTURE=1 (see docs/RECONCILIATION_PLAN.md Phase 0)."""
import os, subprocess, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.slow
def test_water_mode_tripwire():
    ref = os.path.join(HERE, 'tripwire_water_ref.npz')
    assert os.path.exists(ref), 'tripwire reference missing — run tripwire_capture.py with CAPTURE=1'
    r = subprocess.run([sys.executable, os.path.join(HERE, 'tripwire_capture.py')],
                       capture_output=True, text=True, env={**os.environ, 'CAPTURE': '0'})
    assert r.returncode == 0, f'tripwire FAILED:\n{r.stdout}\n{r.stderr}'
    assert 'TRIPWIRE OK' in r.stdout, r.stdout
