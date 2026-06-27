from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

ridge_data = np.load(DATA_DIR / "ridge_100d_dataset.npz")

RIDGE_100D_X = ridge_data["X"]
RIDGE_100D_Y = ridge_data["y"]

assert RIDGE_100D_X.shape == (100, 100)
assert RIDGE_100D_Y.shape == (100,)
#
# Quadratic / concentric-rings binary classification dataset
# Generated once and saved to app/data/quadratic_classification.npz
QUAD_DATA_FILE = DATA_DIR / "quadratic_classification.npz"

if QUAD_DATA_FILE.exists():
    quad_data = np.load(QUAD_DATA_FILE)
    QUAD_X = quad_data["X"]
    QUAD_Y = quad_data["y"]
else:
    # Create a simple two-ring (concentric) dataset that's not linearly separable.
    rng = np.random.default_rng(seed=12345)
    n_per_ring = 250
    # inner ring
    angles0 = rng.random(n_per_ring) * 2.0 * np.pi
    r0 = rng.normal(loc=0.8, scale=0.12, size=n_per_ring)
    X0 = np.column_stack((r0 * np.cos(angles0), r0 * np.sin(angles0)))
    y0 = np.zeros(n_per_ring, dtype=int)

    # outer ring
    angles1 = rng.random(n_per_ring) * 2.0 * np.pi
    r1 = rng.normal(loc=1.8, scale=0.12, size=n_per_ring)
    X1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    y1 = np.ones(n_per_ring, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # ensure data dir exists and save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(QUAD_DATA_FILE, X=X, y=y)
    QUAD_X = X
    QUAD_Y = y

def get_dataset(problem_id: str):
    """
    Return (X, y) for dataset-backed problems.
    Currently supports 'quadratic_binary'.
    """
    if problem_id == "quadratic_binary":
        return QUAD_X, QUAD_Y
    raise KeyError(f"No dataset for problem_id: {problem_id}")
