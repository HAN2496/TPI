import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

print("=" * 60)
print("Testing StratifiedKFold with n_splits=1")
print("=" * 60)

try:
    skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
    print("OK: n_splits=1 works!")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Testing StratifiedKFold with n_splits=2")
print("=" * 60)

try:
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
    print("OK: n_splits=2 works!")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Comparing StratifiedKFold(n_splits=5) vs train_test_split")
print("=" * 60)

# StratifiedKFold first fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    if fold_idx == 0:
        kfold_train_idx = sorted(train_idx)
        kfold_test_idx = sorted(test_idx)
        print(f"StratifiedKFold Fold 0:")
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        print(f"  Train indices[:10]: {kfold_train_idx[:10]}")
        print(f"  Test indices[:10]:  {kfold_test_idx[:10]}")
        break

# train_test_split
train_idx, test_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
)
train_idx_sorted = sorted(train_idx)
test_idx_sorted = sorted(test_idx)

print(f"\ntrain_test_split (test_size=0.2):")
print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
print(f"  Train indices[:10]: {train_idx_sorted[:10]}")
print(f"  Test indices[:10]:  {test_idx_sorted[:10]}")

print(f"\nAre they the same? {kfold_train_idx == train_idx_sorted and kfold_test_idx == test_idx_sorted}")
