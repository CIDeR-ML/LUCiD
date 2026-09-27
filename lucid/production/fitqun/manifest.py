"""One manifest per detector recording every input a fiTQun tune is made of.

``names.py`` says what fiTQun *requires*; this says what we actually *produced*,
so the two together answer the only questions that matter when moving a tune to a
new detector or water model:

* what is still missing,
* what is in use but was not measured here (a value copied from another tune),
* where each file is, and what it has to be called once installed.

Every generator records its own product, so the manifest is written where the
knowledge is rather than reconstructed afterwards by a script that greps for
files. Consumers are deliberately dumb: the report tool plots whatever the
manifest lists, and the install step copies ``path`` to ``install_as`` without
knowing what any of it means.

``status`` is the reason this is worth having. A tune is never simply finished --
it carries borrowed files and untuned scalars for a while -- so the state of each
input is recorded as data instead of living in someone's head:

``adopted``
    Measured here and in use.
``borrowed``
    In use, but produced for a different detector or photosensor. The thing to
    replace next.
``measured_not_adopted``
    Measured here, the measurement is sound, and the result is deliberately not
    used -- e.g. the reference procedure does not apply to this optical model.
    ``note`` must say why, or the decision is lost.
``reference_default``
    A value taken from the reference tune with no measurement of our own.
``missing``
    fiTQun requires it and we have nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import names

SCHEMA = 1

#: Anything not ``adopted`` is a difference between this tune and a fully tuned
#: one, which is what ``verify`` reports and the report tool highlights.
STATUSES = ("adopted", "borrowed", "measured_not_adopted",
            "reference_default", "missing")


@dataclass
class Artifact:
    kind: str
    install_as: str
    status: str
    path: str | None = None
    value: float | None = None
    note: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str]:
        return (self.kind, self.install_as)

    def check(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}; got {self.status!r}")
        if self.path is None and self.value is None and self.status != "missing":
            raise ValueError(f"{self.kind}/{self.install_as}: needs a path or a value")
        if self.status == "measured_not_adopted" and not self.note:
            raise ValueError(
                f"{self.kind}/{self.install_as}: measured_not_adopted must say why "
                "in 'note', otherwise the decision is lost")


@dataclass
class Manifest:
    detector: str
    config: str
    pmt_type: str
    artifacts: list[Artifact] = field(default_factory=list)
    schema: int = SCHEMA

    def to_json(self) -> str:
        d = asdict(self)
        d["artifacts"].sort(key=lambda a: (a["kind"], a["install_as"]))
        return json.dumps(d, indent=2) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=Path(__file__).resolve().parents[3],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def load(path) -> Manifest:
    d = json.loads(Path(path).read_text())
    arts = [Artifact(**a) for a in d.pop("artifacts", [])]
    return Manifest(artifacts=arts, **d)


def save(m: Manifest, path) -> Path:
    """Write atomically -- generators record concurrently from batch jobs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(m.to_json())
    tmp.replace(path)
    return path


def record(manifest_path, *, detector: str, config: str, pmt_type: str,
           **fields) -> Artifact:
    """Add or replace one artifact, creating the manifest if needed.

    Named ``manifest_path`` rather than ``path`` because an artifact has a
    ``path`` of its own and the two are easy to confuse at the call site.
    """
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        m = load(manifest_path)
    else:
        m = Manifest(detector=detector, config=config, pmt_type=pmt_type)
    art = Artifact(**fields)
    art.check()
    art.provenance.setdefault("recorded", time.strftime("%Y-%m-%dT%H:%M:%S"))
    art.provenance.setdefault("commit", _git_commit())
    m.artifacts = [a for a in m.artifacts if a.key() != art.key()] + [art]
    save(m, manifest_path)
    return art


def verify(m: Manifest, *, pdgs=names.TUNED_PDGS, with_fitted_cprofile: bool = True,
           with_3d: bool = True) -> dict[str, list[str]]:
    """``{'missing': [...], 'borrowed': [...], ...}`` against the naming contract.

    ``required_files`` is the authority on what has to exist, so a file we
    produced but that fiTQun will never look for shows up as ``unused`` -- the
    usual cause being a name built by hand instead of through ``names.py``.
    """
    required = set(names.required_files(m.config, m.pmt_type, pdgs=pdgs,
                                        with_3d=with_3d,
                                        with_fitted_cprofile=with_fitted_cprofile))
    have = {a.install_as: a for a in m.artifacts if a.kind != "scalar"}
    out: dict[str, list[str]] = {s: [] for s in STATUSES}
    out["unused"] = sorted(set(have) - required)
    for name in sorted(required):
        a = have.get(name)
        out["missing" if a is None else a.status].append(name)
    for a in m.artifacts:
        if a.kind == "scalar" and a.status != "adopted":
            out[a.status].append(a.install_as)
    return {k: v for k, v in out.items() if v}


def install_plan(m: Manifest, const_dir) -> list[tuple[Path, Path]]:
    """``(source, destination)`` for every file input that is actually in use.

    ``measured_not_adopted`` and ``missing`` are skipped on purpose: the first is
    a measurement we chose not to use, the second has nothing to copy.
    """
    const_dir = Path(const_dir)
    return [(Path(a.path), const_dir / a.install_as)
            for a in m.artifacts
            if a.kind != "scalar" and a.path and a.status in ("adopted", "borrowed")]


def scalar_overrides(m: Manifest) -> dict[str, float]:
    """``{'fiTQun.<key><Det>': value}`` for the parameter override file."""
    return {a.install_as: a.value for a in m.artifacts
            if a.kind == "scalar" and a.value is not None}
