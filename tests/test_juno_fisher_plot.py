"""Tests for numerical-basis Fisher plotting."""

from scripts.plot_juno_synthetic_fisher import render


def test_fisher_renderer_writes_pdf(tmp_path):
    """A minimal full-rank Fisher result renders as vector PDF."""
    names = ["a", "b"]
    source = {
        "position_m": [0.0, 0.0, 0.0],
        "response_mass": 10.0,
        "jacobian_summary": {"l2_norm": [2.0, 1.0]},
    }
    result = {
        "schema": "lucid_per_source_numerical_basis_fisher_v1",
        "optimizer_coordinate_names": names,
        "sources": [source],
        "total": {
            "rank": 2,
            "condition_number": 2.0,
            "eigenvalues": [1.0, 2.0],
            "pseudoinverse_correlation": [[1.0, 0.2], [0.2, 1.0]],
        },
    }
    output = tmp_path / "fisher.pdf"
    render(result, output)
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1000
