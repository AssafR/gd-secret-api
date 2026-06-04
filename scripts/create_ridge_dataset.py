from pathlib import Path
import numpy as np

rng = np.random.default_rng(42)

n_samples = 100
n_features = 100

X = rng.normal(size=(n_samples, n_features))

true_w = rng.normal(size=n_features)
noise = rng.normal(scale=0.5, size=n_samples)

y = X @ true_w + noise

output_path = Path("app/data/ridge_100d_dataset.npz")
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(output_path, X=X, y=y)

print(f"Saved dataset to {output_path}")