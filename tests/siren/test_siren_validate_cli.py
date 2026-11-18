"""``python -m lucid.siren.training.validate``: a subcommand's --material/--particle override the
global flags only when given, wherever they appear on the command line; the table the integral
study reads is given, never assumed."""
import sys

import pytest


class _Built(Exception):
    pass


@pytest.mark.parametrize('argv, want', [
    (['--material', 'ice', 'energy'], ('ice', 'muon')),
    (['energy', '--material', 'ice'], ('ice', 'muon')),
    (['energy'], ('water', 'muon')),
    (['--particle', 'electron', 'all', '--material', 'wbls', '--h5-path', 't.h5'],
     ('wbls', 'electron')),
])
def test_material_and_particle_reach_the_validator(monkeypatch, argv, want):
    import lucid.siren.training.validate as validate
    got = {}

    def build(material, particle, kind, h5_path):
        got['mp'] = (material, particle)
        raise _Built

    monkeypatch.setattr(validate, 'PhotonSimValidator', build)
    monkeypatch.setattr(sys, 'argv', ['validate'] + argv)
    with pytest.raises(_Built):
        validate.main()
    assert got['mp'] == want


@pytest.mark.parametrize('argv', [['integral'], ['all'], ['all', '--kind', 'cherenkov']])
def test_the_integral_study_needs_the_table(monkeypatch, capsys, argv):
    import lucid.siren.training.validate as validate
    monkeypatch.setattr(validate, 'PhotonSimValidator', lambda **kw: pytest.fail('built'))
    monkeypatch.setattr(sys, 'argv', ['validate'] + argv)
    with pytest.raises(SystemExit):
        validate.main()
    assert '--h5-path' in capsys.readouterr().err
