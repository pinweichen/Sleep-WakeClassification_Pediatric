"""
data_ingestion.py — AGV Pipeline
=================================
Converts raw actigraphy CSV files (ID, timestamp, x, y, z, label) into the
directory structure expected by the preprocessing pipeline.

Expected input columns
----------------------
  ID          : subject identifier (integer or string)  [can be absent if
                --use-filename-as-id is set]
  timestamp   : ISO 8601 string, e.g. "2022-08-31T01:24:37.020Z"
  x, y, z     : 3-axis accelerometer values (g)
  label       : "Wake" or "Sleep"  (binary)

Output structure (relative to this file)
-----------------------------------------
  data/data_processed/agv/
    motion/   <subject_id>_motion.csv
    labels/   <subject_id>_labeled.csv
    agv_ids.csv          ← list of processed subject IDs for the pipeline

Motion CSV columns  : agvTime, agvx, agvy, agvz, agvmagnitude, agvenmo, timestamp
Labels CSV columns  : psgtime, psgstg, labels, timestamp, Time

Usage
-----
  # Single CSV with an ID column (one or many subjects):
  python data_ingestion.py --input data/data_processed/agv/raw_data/all_subjects.csv

  # Folder of per-subject CSVs (subject ID taken from the ID column in each file):
  python data_ingestion.py --input data/data_processed/agv/raw_data/

  # Folder of per-subject CSVs where the filename IS the subject ID (no ID column needed):
  python data_ingestion.py --input data/data_processed/agv/raw_data/ --use-filename-as-id

  # Preview without writing files:
  python data_ingestion.py --input data/data_processed/agv/raw_data/ --use-filename-as-id --dry-run
"""

import os
import sys
import argparse
import math
import time as time_module
from datetime import timezone
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent

# AGV_DATA_DIR is exported by run_pipeline.sh to OUTPUT_DIR so no data is
# written inside the repo tree.  Falls back to the original repo-relative path
# for local / manual runs.
_agv_data_dir = os.environ.get('AGV_DATA_DIR')
if _agv_data_dir:
    DATA_ROOT = Path(_agv_data_dir) / 'data_processed' / 'agv'
else:
    DATA_ROOT = REPO_ROOT / 'data' / 'data_processed' / 'agv'

MOTION_DIR  = DATA_ROOT / "motion"
LABELS_DIR  = DATA_ROOT / "labels"

# Label mapping  (extend here if your labels differ)
LABEL_MAP = {
    "wake":  0, "Wake":  0, "WAKE":  0, "W": 0, "0": 0, 0: 0,
    "sleep": 1, "Sleep": 1, "SLEEP": 1, "S": 1, "1": 1, 1: 1,
}


def parse_timestamp(ts_series: pd.Series) -> pd.Series:
    """
    Convert ISO 8601 / datetime strings to Unix epoch float seconds.
    Handles variable millisecond/sub-second precision and both Z and +00:00
    timezone suffixes (e.g. "2022-12-21T02:39:01Z", "2022-12-21T02:39:01.020Z").

    format='ISO8601' is only available in pandas >= 1.5.0 (released 2022-09).
    For compatibility with older cluster installs we instead normalise the Z
    suffix to +00:00 before parsing, which pandas handles correctly in all
    versions without needing a format hint.
    """
    normalised = ts_series.str.replace('Z', '+00:00', regex=False)
    dt = pd.to_datetime(normalised, format='ISO8601', utc=True)
    return dt.astype("int64") / 1e9  # nanoseconds → seconds


def compute_magnitude(x: pd.Series, y: pd.Series, z: pd.Series) -> pd.Series:
    """Vector magnitude of 3-axis acceleration."""
    return np.sqrt(x**2 + y**2 + z**2)


def compute_enmo(magnitude: pd.Series) -> pd.Series:
    """
    Euclidean Norm Minus One (ENMO) — standard actigraphy activity metric.
    Clipped at 0 to remove negative noise.
    """
    return (magnitude - 1.0).clip(lower=0.0)


def process_subject(subject_id, group: pd.DataFrame, dry_run: bool = False):
    """
    Build and write motion + label files for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier string (used as filename prefix).
    group : pd.DataFrame
        Rows from the raw CSV belonging to this subject.
    dry_run : bool
        If True, print a summary but do not write files.
    """
    group = group.copy().reset_index(drop=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    unix_ts = parse_timestamp(group["timestamp"])
    start_time = unix_ts.iloc[0]
    relative_time = unix_ts - start_time

    # ── Motion file ───────────────────────────────────────────────────────────
    mag  = compute_magnitude(group["x"], group["y"], group["z"])
    enmo = compute_enmo(mag)

    motion_df = pd.DataFrame({
        "agvTime":       relative_time,
        "agvx":          group["x"].values,
        "agvy":          group["y"].values,
        "agvz":          group["z"].values,
        "agvmagnitude":  mag.values,
        "agvenmo":       enmo.values,
        "timestamp":     unix_ts.values,
    })

    # ── Label file ────────────────────────────────────────────────────────────
    raw_labels = group["label"].map(LABEL_MAP)
    unmapped = group["label"][raw_labels.isna()].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: unrecognised labels for subject {subject_id}: {unmapped.tolist()}")
        print(f"  These rows will be dropped.")
        valid = raw_labels.notna()
        raw_labels = raw_labels[valid]
        unix_ts_labels = unix_ts[valid]
        relative_labels = relative_time[valid]
        psgtime_series = group["timestamp"][valid]
    else:
        unix_ts_labels = unix_ts
        relative_labels = relative_time
        psgtime_series = group["timestamp"]

    # Map integer label back to stage string expected by psg_label_service
    # 0 → "W" (Wake), 1 → "N2" (treated as consolidated sleep in binary mode)
    stage_map = {0: "W", 1: "N2"}
    labels_df = pd.DataFrame({
        "psgtime":  psgtime_series.values,
        "psgstg":   raw_labels.map(stage_map).values,
        "labels":   raw_labels.values.astype(int),
        "timestamp": unix_ts_labels.values,
        "Time":      relative_labels.values,
    })

    n_wake  = int((raw_labels == 0).sum())
    n_sleep = int((raw_labels == 1).sum())
    print(f"  Subject {subject_id}: {len(motion_df):,} motion rows | "
          f"{len(labels_df):,} label rows  (Wake={n_wake}, Sleep={n_sleep})")

    motion_final = MOTION_DIR / f"{subject_id}_motion.csv"
    labels_final = LABELS_DIR / f"{subject_id}_labeled.csv"

    if dry_run:
        print(f"  [dry-run] Would write: {motion_final}")
        print(f"  [dry-run] Would write: {labels_final}")
        return

    # ── Skip if both output files already exist (complete from a prior run) ──
    # Both files must be present: if only one exists the prior run was
    # interrupted mid-write and we must re-process.
    if motion_final.exists() and labels_final.exists():
        print(f"  Subject {subject_id}: skipping — output files already exist")
        return

    MOTION_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Atomic writes: write to .tmp then rename ──────────────────────────────
    # On Linux (POSIX), os.replace() is atomic: the final filename only appears
    # once the file is fully written.  A crash mid-write leaves a .tmp file and
    # the final name never appears, so the skip check above stays correct.
    motion_tmp = MOTION_DIR / f"{subject_id}_motion.csv.tmp"
    labels_tmp = LABELS_DIR / f"{subject_id}_labeled.csv.tmp"

    # float_format='%.6f' preserves millisecond precision at 50 Hz (20 ms
    # between samples).  Without this, pandas may silently truncate floats.
    motion_df.to_csv(motion_tmp, index=False, float_format='%.6f')
    labels_df.to_csv(labels_tmp, index=False, float_format='%.6f')

    # Both writes succeeded — atomically promote to final filenames.
    motion_tmp.replace(motion_final)
    labels_tmp.replace(labels_final)


def ingest_file(csv_path: Path, id_column: str = "ID", use_filename_as_id: bool = False,
                dry_run: bool = False):
    """
    Load a single CSV and process each subject group.

    Parameters
    ----------
    csv_path           : path to the CSV file
    id_column          : column name that holds the subject identifier
    use_filename_as_id : if True, treat the CSV filename stem as the subject
                         ID and ignore the id_column entirely.  Useful when
                         each file contains exactly one subject.
    dry_run            : preview mode — no files written
    """
    print(f"\nReading: {csv_path}")
    df = pd.read_csv(csv_path)

    if use_filename_as_id:
        # ── Per-subject file mode: filename stem = subject ID ─────────────────
        subject_id = csv_path.stem    # e.g. "subject_001" from "subject_001.csv"
        required = {"timestamp", "x", "y", "z", "label"}
        missing = required - set(df.columns)
        if missing:
            print(f"  ERROR: CSV is missing required columns: {missing}")
            print(f"         Found columns: {list(df.columns)}")
            return []
        print(f"  → Subject ID from filename: {subject_id}  ({len(df):,} rows)")
        process_subject(subject_id, df, dry_run=dry_run)
        return [subject_id]

    else:
        # ── Multi-subject or single-subject CSV with an ID column ─────────────
        required = {id_column, "timestamp", "x", "y", "z", "label"}
        missing = required - set(df.columns)
        if missing:
            print(f"  ERROR: CSV is missing required columns: {missing}")
            print(f"         Found columns: {list(df.columns)}")
            print(f"  TIP: If each CSV is for one subject and has no ID column,")
            print(f"       re-run with --use-filename-as-id")
            return []

        df[id_column] = df[id_column].astype(str)
        subject_ids = sorted(df[id_column].unique())
        print(f"  Found {len(subject_ids)} subject(s): {subject_ids}")

        processed = []
        for sid in subject_ids:
            group = df[df[id_column] == sid]
            process_subject(sid, group, dry_run=dry_run)
            processed.append(sid)

        return processed


def write_ids_csv(subject_ids: list, dry_run: bool = False):
    """Write agv_ids.csv consumed by preprocessing_feature.py.

    When AGV_DATA_DIR is set (via run_pipeline.sh), the file is written to
    AGV_DATA_DIR/agv_ids.csv so that all outputs stay inside OUTPUT_DIR.
    Falls back to REPO_ROOT/data/agv_ids.csv for local/manual runs.
    """
    _agv_data_dir = os.environ.get('AGV_DATA_DIR')
    if _agv_data_dir:
        ids_path = Path(_agv_data_dir) / "agv_ids.csv"
    else:
        ids_path = REPO_ROOT / "data" / "agv_ids.csv"

    ids_df = pd.DataFrame({"subject": subject_ids})
    if dry_run:
        print(f"\n[dry-run] Would write {len(subject_ids)} IDs to {ids_path}")
        return
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_df.to_csv(ids_path, index=False)
    print(f"\nWrote subject ID list: {ids_path}  ({len(subject_ids)} subjects)")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest raw actigraphy CSVs into the AGV pipeline format."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single CSV file, or a directory containing CSV files."
    )
    parser.add_argument(
        "--id-column", default="ID",
        help="Name of the subject-ID column (default: 'ID').  "
             "Ignored when --use-filename-as-id is set."
    )
    parser.add_argument(
        "--use-filename-as-id",
        action="store_true",
        help=(
            "When processing a folder of per-subject CSVs, use each file's "
            "name stem (without extension) as the subject ID instead of reading "
            "an ID column.  Use this when every CSV contains exactly one subject "
            "and has no ID column."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    all_ids = []

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {input_path}")
            sys.exit(1)
        for f in csv_files:
            ids = ingest_file(f,
                              id_column=args.id_column,
                              use_filename_as_id=args.use_filename_as_id,
                              dry_run=args.dry_run)
            all_ids.extend(ids)
    elif input_path.is_file():
        all_ids = ingest_file(input_path,
                              id_column=args.id_column,
                              use_filename_as_id=args.use_filename_as_id,
                              dry_run=args.dry_run)
    else:
        print(f"ERROR: {input_path} is not a valid file or directory.")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    unique_ids = [x for x in all_ids if not (x in seen or seen.add(x))]
    write_ids_csv(unique_ids, dry_run=args.dry_run)

    print(f"\nIngestion complete.  {len(unique_ids)} subject(s) processed.")
    if not args.dry_run:
        print(f"  Motion files : {MOTION_DIR}")
        print(f"  Label files  : {LABELS_DIR}")
        print(f"\nNext steps:")
        print(f"  1. cd preprocessing")
        print(f"  2. python preprocessing_feature.py   (crop + feature extraction)")
        print(f"  3. cd ../source")
        print(f"  4. python analysis_runner_weighted_split_torch.py   (train + evaluate)")


if __name__ == "__main__":
    main()
