from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

ridge_data = np.load(DATA_DIR / "ridge_100d_dataset.npz")

RIDGE_100D_X = ridge_data["X"]
RIDGE_100D_Y = ridge_data["y"]

assert RIDGE_100D_X.shape == (100, 100)
assert RIDGE_100D_Y.shape == (100,)
