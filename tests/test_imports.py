"""Import hygiene: every lucid module imports, and importing one costs what it should.

Three things are pinned here.

1. Every module under ``lucid/`` imports at all, and ``lucid.fitting``'s public surface resolves to
   the objects it advertises rather than to same-named submodules.
2. The **import-graph ratchet**: importing a package does not drag in a plotting library, a
   notebook library or a training library, and does not initialise a JAX backend. This is the test
   that keeps the fix fixed -- ``import lucid.simulation`` used to run
   ``lucid/siren/training/__init__.py`` and so import IPython, optax and matplotlib
   before a photon was propagated.
3. Nothing in ``lucid/`` sets JAX's global numerical policy: runs use JAX's default matmul
   precision with x64 off.

The ratchet runs each module in a **subprocess**: ``sys.modules`` is process-global, and by the
time pytest reaches this file the session has already imported half the tree, so an in-process
check would measure the session and not the module.
"""
import ast
import importlib
import json
import pkgutil
import subprocess
import sys
import textwrap

import pytest
from tests.helpers.paths import REPO


def _lucid_modules():
    """Every module under ``lucid/``, found on disk without importing anything.

    ``pkgutil.walk_packages`` would import each package to recurse, and it skips directories
    without an ``__init__.py`` -- which is all of ``lucid/production``. So walk the tree with
    ``pkgutil.iter_modules`` (which only lists) and recurse into every directory that holds
    Python, package or not.
    """
    def walk(path, prefix):
        for m in pkgutil.iter_modules([str(path)]):
            yield prefix + m.name
            if m.ispkg:
                yield from walk(path / m.name, f'{prefix}{m.name}.')
        for sub in sorted(path.iterdir()):
            if (sub.is_dir() and sub.name != '__pycache__' and not (sub / '__init__.py').exists()
                    and any(sub.rglob('*.py'))):
                yield prefix + sub.name                     # a namespace package
                yield from walk(sub, f'{prefix}{sub.name}.')

    return ['lucid'] + sorted(set(walk(REPO / 'lucid', 'lucid.')))


# Not importable by design, so not import-tested:
#   *.__main__                       a CLI entry point; importing it is running it
#   lucid.production.derived_views.* standalone scripts that read sys.argv and write files at
#                                    module scope (``python depile.py SRC DST IDX``)
_NOT_MODULES = ('lucid.production.derived_views.',)
MODULES = [m for m in _lucid_modules()
           if not m.endswith('.__main__') and not m.startswith(_NOT_MODULES)]

# Libraries no forward or fitting import may pull in. Dimensionless, by import name:
#   IPython     -- an undeclared dependency, and a notebook library on a batch path
#   matplotlib  -- likewise, and it picks a backend at import
#   plotly      -- the event display; four paper figures want it, no production run does
_NEVER = ('IPython', 'matplotlib', 'plotly')

# ...plus, for the forward model specifically, the training stack.
_NEVER_FORWARD = _NEVER + ('optax', 'pandas')

RATCHET = {
    'lucid': _NEVER_FORWARD,
    'lucid.simulation': _NEVER_FORWARD,
    'lucid.fitting': _NEVER,          # lucid.fitting.transforms is built on optax
}


def _run_in_fresh_interpreter(body):
    """Execute ``body`` in a fresh interpreter with the repo importable; return its stdout.

    Parameters
    ----------
    body : str
        Python source, dedented and indented for you, that raises to fail.

    Returns
    -------
    str
        The subprocess's stdout, whose last line is ``OK``.
    """
    src = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
{textwrap.indent(textwrap.dedent(body), ' ' * 8)}
        print('OK')
    """)
    r = subprocess.run([sys.executable, '-c', src], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f'STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr[-3000:]}'
    return r.stdout


def _import_in_subprocess(module):
    """Import ``module`` in a fresh interpreter; report what came with it.

    Parameters
    ----------
    module : str
        Dotted module name to import.

    Returns
    -------
    dict
        ``{'resident': [str], 'jax_backend': bool}`` -- the top-level third-party packages left in
        ``sys.modules`` after the import, and whether a JAX backend was initialised.
    """
    out = _run_in_fresh_interpreter(f"""
        import json
        __import__({module!r})
        resident = sorted(m for m in sys.modules if '.' not in m)
        backend = False
        if 'jax' in sys.modules:
            from jax._src import xla_bridge
            backend = bool(xla_bridge._backends)
        print(json.dumps({{'resident': resident, 'jax_backend': backend}}))
    """)
    return json.loads(out.strip().splitlines()[-2])


def test_the_module_walk_finds_the_whole_tree():
    """The list is generated, so check the generator: it must reach the namespace-package
    subtrees (no ``__init__.py``) that ``pkgutil.walk_packages`` misses, and not stop short."""
    assert len(MODULES) > 120, len(MODULES)
    for m in ('lucid.fitting.schur_gn', 'lucid.production.dataprod.fanout',
              'lucid.production.siren_inputs.generate_jobs', 'lucid.siren.training.trainer'):
        assert m in MODULES, m


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    """Every module under lucid/ imports."""
    importlib.import_module(module)


@pytest.mark.parametrize("module", sorted(RATCHET))
def test_import_pulls_in_no_plotting_or_training_library(module):
    """Importing a lucid package does not import a plotting, notebook or training library.

    The cost is paid by every batch job and every production run, for code none of them calls.
    """
    resident = set(_import_in_subprocess(module)['resident'])
    forbidden = sorted(resident.intersection(RATCHET[module]))
    assert not forbidden, (
        f'`import {module}` pulled in {forbidden}. Find the module-scope import that does it and '
        f'move it inside the function that needs it.')


@pytest.mark.parametrize("module", sorted(RATCHET))
def test_import_initialises_no_jax_backend(module):
    """Importing a lucid package initialises no JAX backend (G6, J3).

    ``jax.devices()`` and any ``jnp`` array construction build a backend. At import that costs a
    second, probes the GPU driver on a node that may be busy, and fixes device placement before a
    caller has said which device it wants.
    """
    assert not _import_in_subprocess(module)['jax_backend'], (
        f'`import {module}` initialised a JAX backend. The cause is a module-level jnp array, a '
        f'jax.devices() call, or a device= in a decorator.')


def test_every_public_name_resolves_to_what_it_claims_to_be():
    """A name in `__all__` must be the OBJECT it advertises, not merely present.

    A submodule with the same name as an exported function (e.g. `gauss_newton`) can shadow it, so
    `from lucid.fitting import gauss_newton` returns the module. The shadowing is silent and
    import-order dependent, and tests that import from the submodule path do not catch it.
    """
    import types
    import lucid.fitting as F

    for name in F.__all__:
        assert hasattr(F, name), f'{name} is in __all__ but not on the package'

    shadowed = [n for n in F.__all__
                if isinstance(getattr(F, n), types.ModuleType) and n != 'report']
    assert not shadowed, (
        f'{shadowed} resolve to MODULES, not to the objects __all__ advertises — a submodule of '
        f'the same name has shadowed them')

    # `damped_matrix` is deliberately NOT here: it is reached as
    # `lucid.fitting.transforms.damped_matrix` and is not re-exported from the package.
    for name in ('gauss_newton', 'gauss_newton_tx', 'calibrate', 'closure_data',
                 'fit_track', 'crb', 'profile_gains', 'neyman_residual'):
        assert callable(getattr(F, name)), f'lucid.fitting.{name} is not callable'


def test_lucid_siren_training_package_imports_nothing():
    """`lucid/siren/training/__init__.py` has no imports at all.

    It used to be on the forward path (the simulator imported the trained-model loader from it),
    so anything re-exported here was imported by every simulation run. Asserted on the source, not
    on `sys.modules`, so the reason a failure gives is
    the line that broke it.
    """
    src = (REPO / 'lucid' / 'siren' / 'training' / '__init__.py').read_text()
    offenders = [f'line {n.lineno}: {ast.unparse(n)}'
                 for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not offenders, (
        'lucid/siren/training/__init__.py must import nothing; it imports:\n  '
        + '\n  '.join(offenders))


def test_no_lucid_module_imports_the_siren_training_package_itself():
    """Nothing under `lucid/` may `import lucid.siren.training` as a package.

    Importing the package is how the training stack used to reach the forward path. Importing one
    of its modules (`...training.trainer`, `...training.dataset`) is fine and is what every
    caller does.
    """
    offenders = []
    for path in sorted((REPO / 'lucid').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == 'lucid.siren.training':
                offenders.append(f'{path.relative_to(REPO)}:{node.lineno}')
            if isinstance(node, ast.Import):
                offenders += [f'{path.relative_to(REPO)}:{node.lineno}'
                              for a in node.names if a.name == 'lucid.siren.training']
    assert not offenders, (
        'these import the lucid.siren.training PACKAGE; import one of its modules instead:\n  '
        + '\n  '.join(offenders))


def test_no_lucid_module_sets_the_precision_policy():
    """`jax_default_matmul_precision` and `jax_enable_x64` are written nowhere under `lucid/`.

    Every run uses JAX's default matmul precision (TF32 on an NVIDIA GPU, exact float32 on CPU,
    the working point the published numbers were captured at) with x64 off. A writer anywhere
    makes the policy depend on import order.
    """
    keys = ('jax_default_matmul_precision', 'jax_enable_x64')
    writers = []
    for path in sorted((REPO / 'lucid').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'update'):
                continue
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in keys:
                writers.append(f'{path.relative_to(REPO)}:{node.lineno} -> {node.args[0].value}')
    assert not writers, (
        'these set the JAX precision policy, which lucid leaves at JAX\'s default:\n  '
        + '\n  '.join(writers))


def test_the_pdg_table_imports_nothing():
    """`lucid/pdg.py` is read by the fiTQun fan-out on a submit host with no scientific stack."""
    src = (REPO / 'lucid' / 'pdg.py').read_text()
    assert not [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
