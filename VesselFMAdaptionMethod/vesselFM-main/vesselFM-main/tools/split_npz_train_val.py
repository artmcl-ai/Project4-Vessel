import os
import pathlib
import random
import argparse

def main(
    all_dir="data/nnunet_npz_all",
    train_dir="data/nnunet_npz_train",
    val_dir="data/nnunet_npz_val",
    val_fraction=0.2,
    seed=42,
):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    all_dir = (repo_root / all_dir).resolve()
    train_dir = (repo_root / train_dir).resolve()
    val_dir = (repo_root / val_dir).resolve()

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(all_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {all_dir}")

    random.seed(seed)
    random.shuffle(npz_files)

    n_total = len(npz_files)
    n_val = max(1, int(round(n_total * val_fraction)))
    val_files = npz_files[:n_val]
    train_files = npz_files[n_val:]

    print(f"Total cases: {n_total}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    def symlink(src, dst_dir):
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    # Create symlinks
    for f in train_files:
        symlink(f, train_dir)
    for f in val_files:
        symlink(f, val_dir)

    print(f"Train .npz in {train_dir}")
    print(f"Val   .npz in {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_dir", default="data/nnunet_npz_all")
    parser.add_argument("--train_dir", default="data/nnunet_npz_train")
    parser.add_argument("--val_dir", default="data/nnunet_npz_val")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        all_dir=args.all_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
