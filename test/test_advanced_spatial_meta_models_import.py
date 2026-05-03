from pathlib import Path
import numpy as np
import pytest


def _repo_root(start: Path) -> Path:
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


def test_discovery_minimum(meta_dir: Path) -> None:
    if not meta_dir.exists():
        pytest.skip(f"Meta models directory not found: {meta_dir}")

    subdirs = [d for d in meta_dir.iterdir() if d.is_dir()]
    if not subdirs:
        pytest.skip("No export subdirectories found under meta_models; run export step first")

    # Build a minimal in-memory discovery: collect arrays and align shapes
    preds = {}
    shapes = []
    y_true = None

    for sub in sorted(subdirs):
        pred_f = sub / "predictions.npy"
        targ_f = sub / "targets.npy"
        if not pred_f.exists() or not targ_f.exists():
            continue
        pred = np.load(pred_f)
        targ = np.load(targ_f)
        if pred.ndim != 4 or targ.ndim != 4:
            continue
        preds[sub.name] = pred
        shapes.append(pred.shape)
        if y_true is None:
            y_true = targ

    if not preds:
        pytest.skip("No valid predictions discovered; ensure export step produced arrays")

    # Align to minimal common shape
    N = min(s[0] for s in shapes)
    H = min(s[1] for s in shapes)
    Y = min(s[2] for s in shapes)
    X = min(s[3] for s in shapes)
    assert N > 0 and H > 0 and Y > 0 and X > 0

    # Slice and verify
    for k, arr in preds.items():
        arr_c = arr[:N, :H, :Y, :X]
        assert arr_c.shape == (N, H, Y, X)
        assert np.isfinite(arr_c).all()

    yt_c = y_true[:N, :H, :Y, :X]
    assert yt_c.shape == (N, H, Y, X)
    assert np.isfinite(yt_c).all()


def test_tiny_stack_fit(meta_dir: Path) -> None:
    # Optional tiny fit to ensure arrays are consumable
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        pytest.skip("scikit-learn not available in environment")

    if not meta_dir.exists():
        pytest.skip(f"Meta models directory not found: {meta_dir}")

    subdirs = [d for d in meta_dir.iterdir() if d.is_dir()]
    if len(subdirs) < 1:
        pytest.skip("No export subdirectories found under meta_models; run export step first")

    # Collect one or two models for stacking
    arrays = []
    targets = None
    for sub in sorted(subdirs):
        pred_f = sub / "predictions.npy"
        targ_f = sub / "targets.npy"
        if pred_f.exists() and targ_f.exists():
            arrays.append(np.load(pred_f))
            if targets is None:
                targets = np.load(targ_f)
        if len(arrays) >= 2:
            break

    if len(arrays) < 1 or targets is None:
        pytest.skip("Insufficient arrays for tiny stacking test")

    # Align shapes
    N = min(a.shape[0] for a in arrays)
    H = min(a.shape[1] for a in arrays)
    Y = min(a.shape[2] for a in arrays)
    X = min(a.shape[3] for a in arrays)
    arrays = [a[:N, :H, :Y, :X] for a in arrays]
    targets = targets[:N, :H, :Y, :X]

    # Horizon 0 only, flatten spatial
    X_feat = np.concatenate([a[:, 0].reshape(N, -1) for a in arrays], axis=1)
    y_vec = targets[:, 0].reshape(N, -1).mean(axis=1)

    # Very small model for speed
    rf = RandomForestRegressor(n_estimators=10, random_state=0)
    rf.fit(X_feat[: min(16, N)], y_vec[: min(16, N)])
    yp = rf.predict(X_feat[: min(16, N)])
    assert np.isfinite(yp).all()


