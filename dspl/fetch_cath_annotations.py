"""Fetch CATH Homology annotations for all Task 1 proteins via PDBe SIFTS API.

Multi-threaded parallel fetching (~20 threads). Caches results to avoid re-fetching.
Output: data/cath_annotations.json → {pdb_id: cath_homology_str}

Usage:
    python dspl/phase2_crossmodal/fetch_cath_annotations.py
"""
import os, sys, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

ROOT = os.environ.get('DSPL_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, ROOT)

CACHE_FILE = "data/cath_annotations.json"
API_BASE = "https://www.ebi.ac.uk/pdbe/api/mappings"

def load_split(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def fetch_cath(pdb_id):
    """Fetch CATH Homology for one PDB ID. Returns (pdb_id, cath_homology | None)."""
    try:
        url = f"{API_BASE}/{pdb_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        pkey = pdb_id.lower()
        if pkey in data and data[pkey]:
            cath = data[pkey].get("CATH", {})
            if cath:
                # cath is dict: {domain_id: {class, architecture, topology, homology, name}}
                # Take the first domain's homology as representative
                for domain_id, domain_data in cath.items():
                    return (pdb_id, domain_data.get("homology", "UNKNOWN"))
        return (pdb_id, "NO_CATH")
    except Exception as e:
        return (pdb_id, f"ERROR:{e}")

def main():
    # Load all Task 1 protein IDs
    train_ids = load_split("data/splits/adaptability/train.txt")
    val_ids = load_split("data/splits/adaptability/val.txt")
    test_ids = load_split("data/splits/adaptability/test.txt")
    all_ids = list(dict.fromkeys(train_ids + val_ids + test_ids))  # dedup preserve order

    print(f"Task 1 proteins: {len(all_ids)} (train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})")

    # Load existing cache
    cath_map = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cath_map = json.load(f)
        print(f"Loaded {len(cath_map)} cached CATH annotations")

    # Filter to fetch only missing ones
    to_fetch = [pid for pid in all_ids if pid not in cath_map]
    print(f"To fetch: {len(to_fetch)} proteins")

    if not to_fetch:
        print("All proteins already cached!")
        return

    # Multi-threaded fetch
    n_workers = 20
    fetched = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(fetch_cath, pid): pid for pid in to_fetch}
        for future in as_completed(futures):
            pid, result = future.result()
            cath_map[pid] = result
            fetched += 1
            if result.startswith("ERROR"):
                errors += 1

            # Progress every 100
            if fetched % 100 == 0:
                elapsed = time.time() - t0
                rate = fetched / elapsed
                remaining = (len(to_fetch) - fetched) / rate
                print(f"  [{fetched}/{len(to_fetch)}] {rate:.1f}/s, ETA {remaining/60:.0f}min, errors={errors}")

            # Save cache periodically (every 500)
            if fetched % 500 == 0:
                with open(CACHE_FILE, "w") as f:
                    json.dump(cath_map, f, indent=2)
                print(f"  [Cache saved: {len(cath_map)} entries]")

    # Final save
    with open(CACHE_FILE, "w") as f:
        json.dump(cath_map, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone! {len(cath_map)} annotations in {elapsed/60:.1f} min")
    print(f"Errors: {errors}")

    # Stats
    values = list(cath_map.values())
    no_cath = sum(1 for v in values if v == "NO_CATH")
    error_count = sum(1 for v in values if v.startswith("ERROR"))
    valid = len(values) - no_cath - error_count
    unique_homologies = len(set(v for v in values if v not in ("NO_CATH",) and not v.startswith("ERROR")))
    print(f"Valid CATH: {valid}, NO_CATH: {no_cath}, Errors: {error_count}")
    print(f"Unique CATH Homology levels: {unique_homologies}")

    # Top 15 most frequent homologies
    counter = Counter(v for v in values if v not in ("NO_CATH",) and not v.startswith("ERROR"))
    print("\nTop 15 CATH Homology levels:")
    for h, cnt in counter.most_common(15):
        print(f"  {h}: {cnt}")

if __name__ == "__main__":
    main()
