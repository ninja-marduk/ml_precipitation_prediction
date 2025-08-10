from pathlib import Path
import json
import numpy as np
import pytest


def _repo_root(start: Path) -> Path:
    """Find repository root by locating a .git directory."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
    return start


def _advanced_spatial_meta_dir() -> Path:
    base = _repo_root(Path.cwd())
    return base / "models" / "output" / "advanced_spatial" / "meta_models"


@pytest.fixture(scope="session")
def meta_dir() -> Path:
    return _advanced_spatial_meta_dir()


def test_meta_exports_exist(meta_dir: Path) -> None:
    if not meta_dir.exists():
        pytest.skip(f"Meta models directory not found: {meta_dir}")

    subdirs = [d for d in meta_dir.iterdir() if d.is_dir()]
    if not subdirs:
        pytest.skip("No export subdirectories found under meta_models; run export step first")

    # Check each export directory contains core files
    found_valid = False
    for sub in sorted(subdirs):
        pred_f = sub / "predictions.npy"
        targ_f = sub / "targets.npy"
        # inputs and metadata are recommended but optional
        assert pred_f.exists(), f"Missing predictions.npy in {sub}"
        assert targ_f.exists(), f"Missing targets.npy in {sub}"
        found_valid = True
    assert found_valid, "No valid export directories found"


def test_meta_exports_shapes_and_values(meta_dir: Path) -> None:
    if not meta_dir.exists():
        pytest.skip(f"Meta models directory not found: {meta_dir}")

    subdirs = [d for d in meta_dir.iterdir() if d.is_dir()]
    if not subdirs:
        pytest.skip("No export subdirectories found under meta_models; run export step first")

    shapes = []
    for sub in sorted(subdirs):
        pred_f = sub / "predictions.npy"
        targ_f = sub / "targets.npy"
        if not pred_f.exists() or not targ_f.exists():
            continue

        pred = np.load(pred_f)
        targ = np.load(targ_f)

        # Must be 4D: (N, H, Y, X)
        assert pred.ndim == 4, f"predictions must be 4D, got {pred.shape} in {sub}"
        assert targ.ndim == 4, f"targets must be 4D, got {targ.shape} in {sub}"
        # Same shape between predictions and targets
        assert pred.shape == targ.shape, f"shape mismatch in {sub}: {pred.shape} vs {targ.shape}"
        # Non-empty
        assert pred.size > 0, f"empty predictions array in {sub}"
        # Finite values
        assert np.isfinite(pred).all(), f"non-finite values in predictions {sub}"
        assert np.isfinite(targ).all(), f"non-finite values in targets {sub}"
        shapes.append(pred.shape)

    if not shapes:
        pytest.skip("No valid predictions/targets found; ensure export step produced arrays")

    # Verify there exists a common minimal shape across exports
    N = min(s[0] for s in shapes)
    H = min(s[1] for s in shapes)
    Y = min(s[2] for s in shapes)
    X = min(s[3] for s in shapes)
    assert N > 0 and H > 0 and Y > 0 and X > 0, f"invalid common shape: {(N,H,Y,X)}"


def test_export_reports_if_present(meta_dir: Path) -> None:
    # Optional: validate smoke test report if the notebook generated it
    report_path = meta_dir / "export_smoke_test.json"
    if not report_path.exists():
        pytest.skip("export_smoke_test.json not found; run the smoke-test cell if you need this check")

    data = json.loads(report_path.read_text())
    assert "exports_checked" in data and data["exports_checked"] >= 1
    assert "valid_exports" in data and data["valid_exports"] >= 1
    assert isinstance(data.get("errors", []), list)


