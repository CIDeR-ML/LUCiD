"""``python -m lucid.siren.training.validate``: a subcommand's --material/--particle override the
global flags only when given, wherever they appear on the command line."""
import sys

import pytest


class _Built(Exception):
    pass


@pytest.mark.parametrize('argv, want', [
    (['--material', 'ice', 'energy'], ('ice', 'muon')),
    (['energy', '--material', 'ice'], ('ice', 'muon')),
    (['energy'], ('water', 'muon')),
    (['--particle', 'electron', 'all', '--material', 'wbls'], ('wbls', 'electron')),
])
def test_material_and_particle_reach_the_validator(monkeypatch, argv, want):
    import lucid.siren.training.validate as validate
    got = {}

    def build(material, particle, kind):
        got['mp'] = (material, particle)
        raise _Built

    monkeypatch.setattr(validate, 'PhotonSimValidator', build)
    monkeypatch.setattr(sys, 'argv', ['validate'] + argv)
    with pytest.raises(_Built):
        validate.main()
    assert got['mp'] == want
