"""Rewrite train_metadata.csv absolute paths to repo-relative paths for HF dataset push.

Maps:
  /data/yjangir/sidegig/wm/data/droid_1k/...   -> droid_1k/...
  /data/yjangir/sidegig/wm/data_wan_1k/...     -> data_wan_1k/...

The `actions` column is already repo-relative (e.g. "actions/scene_00000.npz");
it gets prefixed with "data_wan_1k/" so all paths are repo-rooted.
"""
import csv
import sys
from pathlib import Path

SRC = Path("/data/yjangir/sidegig/wm/data_wan_1k/train_metadata.csv")
DST = Path("/data/yjangir/sidegig/wm/data_wan_1k/train_metadata.csv")
BACKUP = SRC.with_suffix(".csv.bak.preHF")

REPLACEMENTS = [
    ("/data/yjangir/sidegig/wm/data/droid_1k/", "droid_1k/"),
    ("/data/yjangir/sidegig/wm/data_wan_1k/", "data_wan_1k/"),
]

def rewrite_field(col: str, val: str) -> str:
    for src, dst in REPLACEMENTS:
        if val.startswith(src):
            return val.replace(src, dst, 1)
    if col == "actions" and not val.startswith("data_wan_1k/"):
        return f"data_wan_1k/{val}"
    return val

def main() -> int:
    if not SRC.exists():
        print(f"missing: {SRC}", file=sys.stderr)
        return 1
    if not BACKUP.exists():
        BACKUP.write_bytes(SRC.read_bytes())
        print(f"backup written: {BACKUP}")

    with SRC.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    rewritten = 0
    for row in rows:
        for col in fieldnames:
            new_val = rewrite_field(col, row[col])
            if new_val != row[col]:
                rewritten += 1
                row[col] = new_val

    with DST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows: {len(rows)}, fields rewritten: {rewritten}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
