"""Generate CATH Homology-disjoint train/val/test splits.

Algorithm: Greedy family assignment.
  1. Group all proteins by CATH Homology level.
  2. Sort groups by size (largest first).
  3. Greedily assign groups to train until train >= 85%.
  4. Assign remaining largest groups to val until val >= 5%.
  5. Rest → test.

Ensures: same CATH Homology level never appears in >1 split.

Usage:
    python dspl/phase2_crossmodal/generate_cath_disjoint_split.py \
        --cath-file data/cath_annotations.json \
        --original-split-dir data/splits/adaptability \
        --output-dir data/splits/adaptability_cath \
        --train-ratio 0.85 --val-ratio 0.05 --seed 42
"""
import os, sys, json, argparse, random
from collections import defaultdict, Counter

ROOT = os.environ.get('DSPL_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, ROOT)


def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cath-file", default="data/cath_annotations.json")
    parser.add_argument("--original-split-dir", default="data/splits/adaptability")
    parser.add_argument("--output-dir", default="data/splits/adaptability_cath")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-group-size", type=int, default=5,
                       help="Groups with fewer proteins than this: merge into train (too small to isolate)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load CATH annotations
    with open(args.cath_file) as f:
        cath_map = json.load(f)
    print(f"Loaded CATH annotations: {len(cath_map)} proteins")

    # Load original split IDs (to maintain the same protein universe)
    orig_train = set(load_split(os.path.join(args.original_split_dir, "train.txt")))
    orig_val = set(load_split(os.path.join(args.original_split_dir, "val.txt")))
    orig_test = set(load_split(os.path.join(args.original_split_dir, "test.txt")))
    all_orig = orig_train | orig_val | orig_test
    print(f"Original universe: {len(all_orig)} proteins (train={len(orig_train)}, val={len(orig_val)}, test={len(orig_test)})")

    # Filter: only use proteins that have valid CATH annotations
    valid_pdbs = {p for p in all_orig if p in cath_map and cath_map[p] not in ("NO_CATH",) and not cath_map[p].startswith("ERROR")}
    no_cath = all_orig - valid_pdbs
    if no_cath:
        print(f"\n⚠ {len(no_cath)} proteins have no valid CATH annotation.")
        print(f"  They will be merged into train (most conservative).")
        print(f"  First 10: {list(no_cath)[:10]}")

    # Group by CATH Homology (only valid ones)
    homology_groups = defaultdict(list)
    for pdb in valid_pdbs:
        h = cath_map[pdb]
        homology_groups[h].append(pdb)

    # Shuffle within each group
    for h in homology_groups:
        random.shuffle(homology_groups[h])

    n_groups = len(homology_groups)
    total_proteins = len(all_orig)
    print(f"\nCATH Homology groups: {n_groups}")
    print(f"Target: train≥{args.train_ratio:.0%}, val≥{args.val_ratio:.0%}, test=rest")

    # Merge tiny groups into a "miscellaneous" pool (too small to have statistical meaning)
    merged = []
    tiny_pool = []
    for h, pdbs in homology_groups.items():
        if len(pdbs) < args.min_group_size:
            tiny_pool.extend(pdbs)
        else:
            merged.append((h, pdbs))

    if tiny_pool:
        merged.append(("MISCELLANEOUS", tiny_pool))
    print(f"After merging groups < {args.min_group_size}: {len(merged)} groups (tiny pool: {len(tiny_pool)} proteins)")

    # Sort groups by size descending
    merged.sort(key=lambda x: -len(x[1]))

    # Greedy assignment
    train_pool = list(no_cath)  # start with no-CATH proteins in train
    val_pool = []
    test_pool = []

    total = len(all_orig)
    target_train = int(total * args.train_ratio)
    target_val = int(total * args.val_ratio)

    # Phase 1: fill train to target
    for i, (h, pdbs) in enumerate(merged):
        if len(train_pool) >= target_train:
            break
        train_pool.extend(pdbs)
        merged[i] = None  # mark as assigned
    merged = [m for m in merged if m is not None]

    # Phase 2: fill val to target
    for i, (h, pdbs) in enumerate(merged):
        if len(val_pool) >= target_val:
            break
        val_pool.extend(pdbs)
        merged[i] = None
    merged = [m for m in merged if m is not None]

    # Phase 3: rest → test
    for h, pdbs in merged:
        test_pool.extend(pdbs)

    # Verify disjointness
    train_set = set(train_pool)
    val_set = set(val_pool)
    test_set = set(test_pool)

    assert len(train_set & val_set) == 0, "Train-Val overlap!"
    assert len(train_set & test_set) == 0, "Train-Test overlap!"
    assert len(val_set & test_set) == 0, "Val-Test overlap!"

    actual_total = len(train_set) + len(val_set) + len(test_set)
    assert actual_total == total, f"Total mismatch: {actual_total} vs {total}"

    print(f"\n=== Resulting Split ===")
    print(f"Train: {len(train_set)} ({100*len(train_set)/total:.1f}%)")
    print(f"Val:   {len(val_set)} ({100*len(val_set)/total:.1f}%)")
    print(f"Test:  {len(test_set)} ({100*len(test_set)/total:.1f}%)")
    print(f"Total: {actual_total}")

    # Verify: same CATH Homology not in two splits
    train_homs = set(cath_map[p] for p in train_set if p in cath_map and cath_map[p] not in ("NO_CATH",) and not cath_map[p].startswith("ERROR"))
    val_homs = set(cath_map[p] for p in val_set if p in cath_map and cath_map[p] not in ("NO_CATH",) and not cath_map[p].startswith("ERROR"))
    test_homs = set(cath_map[p] for p in test_set if p in cath_map and cath_map[p] not in ("NO_CATH",) and not cath_map[p].startswith("ERROR"))

    train_val_overlap = train_homs & val_homs
    train_test_overlap = train_homs & test_homs
    val_test_overlap = val_homs & test_homs

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"\n⚠ HOMOLOGY OVERLAP DETECTED!")
        if train_val_overlap:
            print(f"  Train-Val: {train_val_overlap}")
        if train_test_overlap:
            print(f"  Train-Test: {train_test_overlap}")
        if val_test_overlap:
            print(f"  Val-Test: {val_test_overlap}")
    else:
        print(f"\n✓ No CATH Homology overlap between splits — perfect disjointness")

    print(f"Unique homologies: train={len(train_homs)}, val={len(val_homs)}, test={len(test_homs)}")

    # Write split files
    os.makedirs(args.output_dir, exist_ok=True)
    for name, pool in [("train", train_pool), ("val", val_pool), ("test", test_pool)]:
        path = os.path.join(args.output_dir, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(sorted(pool)) + "\n")
        print(f"  Wrote {path} ({len(pool)} proteins)")

    # Save metadata for paper
    metadata = {
        "split_scheme": "cath_homology_disjoint",
        "total_proteins": total,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "train_ratio": len(train_set) / total,
        "val_ratio": len(val_set) / total,
        "test_ratio": len(test_set) / total,
        "train_homologies": sorted(train_homs),
        "val_homologies": sorted(val_homs),
        "test_homologies": sorted(test_homs),
        "min_group_size": args.min_group_size,
        "proteins_without_cath": len(no_cath),
        "total_homology_groups": n_groups,
    }
    meta_path = os.path.join(args.output_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=list)
    print(f"  Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
