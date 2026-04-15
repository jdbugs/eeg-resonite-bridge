#!/usr/bin/env python3
"""
generate_heatmaps.py — Export EEG training windows as heatmap images.

Run after one or more training sessions to convert the labelled CSV data
into per-window images, grid visualisations, and the manifest.json that
the live classifier reads.

Usage:
    python generate_heatmaps.py
    python generate_heatmaps.py --logs ./logs --out ./training_data

Requires:  numpy  Pillow
    pip install numpy Pillow

Output layout:
    training_data/
        manifest.json            ← read by classifier.py
        grid_baseline.png        ← visual inspection grid per label
        grid_eyes_closed.png
        ...
        baseline/
            window_0000.png      ← individual 8×15 heatmap windows
            window_0001.png
            ...
        eyes_closed/
            ...
"""

import argparse
import csv
import json
import pathlib
import sys
from itertools import groupby

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError as e:
    print(f"\nMissing dependency: {e}")
    print("Install with:  pip install numpy Pillow\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BAND_NAMES = [
    'delta', 'theta', 'low_alpha', 'high_alpha',
    'low_beta', 'high_beta', 'low_gamma', 'high_gamma',
]

# One colour per band (dark = low value, saturated = high value)
# Order matches BAND_NAMES: delta top → high_gamma bottom
BAND_COLORS = [
    ( 60,   0, 120),   # delta      — purple
    (  0,  60, 200),   # theta      — blue
    (  0, 150, 180),   # low_alpha  — teal
    (  0, 190, 110),   # high_alpha — green
    (190, 190,   0),   # low_beta   — yellow
    (220, 110,   0),   # high_beta  — orange
    (210,  30,   0),   # low_gamma  — red-orange
    (160,   0,   0),   # high_gamma — dark red
]

WINDOW_SIZE    = 15    # rows (packets) per window
MAX_POOR       = 3     # discard windows with more than this many poor-signal rows
PIXEL_SCALE    = 10    # scale factor: 8×15 raw → 80×150 display pixels
GAP_PX         = 2     # pixel gap between images in the grid
GRID_MAX_COLS  = 10    # max columns in the grid image


# ─────────────────────────────────────────────────────────────────────────────
# CSV reading
# ─────────────────────────────────────────────────────────────────────────────

def read_training_rows(csv_path: pathlib.Path) -> list[dict]:
    """Return all rows that have a non-empty training_label."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'training_label' not in (reader.fieldnames or []):
            return []
        for row in reader:
            if row.get('training_label', '').strip():
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

def chop_windows(rows: list[dict], source: str) -> list[dict]:
    """
    Group consecutive rows by training_label, then chop each group into
    non-overlapping WINDOW_SIZE-row chunks.
    Returns a list of window dicts.
    """
    windows = []
    for label, group in groupby(rows, key=lambda r: r['training_label']):
        segment = list(group)
        for i in range(0, len(segment) - WINDOW_SIZE + 1, WINDOW_SIZE):
            chunk = segment[i : i + WINDOW_SIZE]
            if len(chunk) < WINDOW_SIZE:
                continue
            poor = sum(1 for r in chunk if int(r.get('signal_poor', 0) or 0))
            windows.append({
                'label':  label,
                'rows':   chunk,
                'poor':   poor,
                'source': source,
                'round':  chunk[0].get('training_round', ''),
            })
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Array + image generation
# ─────────────────────────────────────────────────────────────────────────────

def window_to_array(rows: list[dict]) -> np.ndarray:
    """
    Convert WINDOW_SIZE rows to an (8, 15) float32 array.
    arr[band_index, time_step] = normalised band value (0–1).
    """
    arr = np.zeros((len(BAND_NAMES), WINDOW_SIZE), dtype=np.float32)
    for t, row in enumerate(rows):
        for b, band in enumerate(BAND_NAMES):
            try:
                arr[b, t] = float(row.get(band, 0) or 0)
            except (ValueError, TypeError):
                pass
    return arr


def render_window(arr: np.ndarray, scale: int = PIXEL_SCALE) -> Image.Image:
    """
    Render an (8, 15) array as a scaled heatmap image.
    Pixel brightness scales linearly with band value.
    Result size: (15*scale, 8*scale) = (150, 80) at scale=10.
    """
    n_bands, n_steps = arr.shape
    rgb = np.zeros((n_bands, n_steps, 3), dtype=np.uint8)

    for b, (cr, cg, cb) in enumerate(BAND_COLORS):
        v = arr[b]   # shape (n_steps,)
        rgb[b, :, 0] = np.clip(cr * v, 0, 255).astype(np.uint8)
        rgb[b, :, 1] = np.clip(cg * v, 0, 255).astype(np.uint8)
        rgb[b, :, 2] = np.clip(cb * v, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, 'RGB')
    return img.resize((n_steps * scale, n_bands * scale), Image.NEAREST)


def render_grid(images: list[Image.Image], label: str,
                max_cols: int = GRID_MAX_COLS) -> Image.Image | None:
    """
    Arrange window images in a grid with a label header and window IDs.
    """
    if not images:
        return None

    cell_w, cell_h = images[0].size
    n    = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    header_h = 18
    grid_w   = GAP_PX + cols * (cell_w + GAP_PX)
    grid_h   = header_h + GAP_PX + rows * (cell_h + GAP_PX)

    grid = Image.new('RGB', (grid_w, grid_h), (8, 4, 0))
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(images):
        row_i = idx // cols
        col_i = idx % cols
        x = GAP_PX + col_i * (cell_w + GAP_PX)
        y = header_h + GAP_PX + row_i * (cell_h + GAP_PX)
        grid.paste(img, (x, y))
        try:
            draw.text((x + 2, y + 2), str(idx), fill=(220, 140, 0))
        except Exception:
            pass

    try:
        draw.text((GAP_PX, 2), f"{label.upper()}  ({n} windows)", fill=(255, 170, 0))
    except Exception:
        pass

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Similarity report
# ─────────────────────────────────────────────────────────────────────────────

def label_similarity_report(label_arrays: dict[str, np.ndarray]) -> list[tuple]:
    """
    Compute pairwise cosine similarity between per-label mean feature vectors.
    Returns list of (label_a, label_b, similarity) sorted most-similar first.
    """
    labels = sorted(label_arrays)
    means  = {}
    for lbl in labels:
        flat = label_arrays[lbl].reshape(label_arrays[lbl].shape[0], -1).mean(axis=0)
        n    = np.linalg.norm(flat)
        means[lbl] = flat / n if n > 1e-8 else flat

    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = means[labels[i]], means[labels[j]]
            sim  = float(np.dot(a, b))
            pairs.append((labels[i], labels[j], sim))

    return sorted(pairs, key=lambda x: -x[2])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate EEG training heatmaps from session CSV logs.'
    )
    parser.add_argument('--logs', default='./logs',
                        help='Directory containing session_*.csv files (default: ./logs)')
    parser.add_argument('--out',  default='./training_data',
                        help='Output directory for images and manifest (default: ./training_data)')
    args = parser.parse_args()

    logs_dir = pathlib.Path(args.logs)
    out_dir  = pathlib.Path(args.out)

    # ── Find session CSVs ─────────────────────────────────────────────────────
    csv_files = sorted(
        f for f in logs_dir.glob('session_*.csv')
        if '_annotations' not in f.name and '_events' not in f.name
    )

    if not csv_files:
        print(f"No session CSVs found in {logs_dir}")
        print("Run a training session with logging enabled first.")
        sys.exit(0)

    print(f"Found {len(csv_files)} session file(s) in {logs_dir}\n")

    # ── Collect all labelled windows ──────────────────────────────────────────
    all_windows = []
    for path in csv_files:
        rows = read_training_rows(path)
        if not rows:
            print(f"  {path.name}  —  no training labels, skipping")
            continue
        wins = chop_windows(rows, str(path))
        print(f"  {path.name}  —  {len(rows)} labelled rows  →  {len(wins)} windows")
        all_windows.extend(wins)

    if not all_windows:
        print("\nNo training windows found. Run a training session first.")
        sys.exit(0)

    # ── Filter poor-signal windows ────────────────────────────────────────────
    by_label: dict[str, list] = {}
    discarded: dict[str, int] = {}

    for w in all_windows:
        lbl = w['label']
        if w['poor'] > MAX_POOR:
            discarded[lbl] = discarded.get(lbl, 0) + 1
        else:
            by_label.setdefault(lbl, []).append(w)

    print(f"\nWindows per label after filtering (>{MAX_POOR} poor-signal rows discarded):")
    for lbl in sorted(set(list(by_label) + list(discarded))):
        kept = len(by_label.get(lbl, []))
        disc = discarded.get(lbl, 0)
        print(f"  {lbl:<18}  {kept:>4} kept   {disc:>4} discarded")

    # ── Generate images and manifest ──────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest    = []
    label_arrays: dict[str, np.ndarray] = {}

    for lbl in sorted(by_label):
        wins      = by_label[lbl]
        label_dir = out_dir / lbl
        label_dir.mkdir(exist_ok=True)

        images  = []
        arrays  = []

        for idx, w in enumerate(wins):
            arr = window_to_array(w['rows'])
            img = render_window(arr)

            fname = f"window_{idx:04d}.png"
            fpath = label_dir / fname
            img.save(fpath)

            images.append(img)
            arrays.append(arr)

            avg_sq = float(np.mean([
                float(r.get('signal_quality', 200) or 200) for r in w['rows']
            ]))

            manifest.append({
                'label':               lbl,
                'window_id':           idx,
                'source':              w['source'],
                'round':               str(w['round']),
                'avg_signal_quality':  round(avg_sq, 1),
                'signal_poor_rows':    w['poor'],
                'file':                str(fpath),
                # Feature vector — (8, 15) flattened row-major:
                # [delta_t0..t14, theta_t0..t14, ..., high_gamma_t0..t14]
                'vector':              arr.flatten().tolist(),
            })

        label_arrays[lbl] = np.stack(arrays)

        # Grid visualisation
        grid = render_grid(images, lbl)
        if grid:
            gpath = out_dir / f"grid_{lbl}.png"
            grid.save(gpath)
            print(f"\n  grid → {gpath}")

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest → {manifest_path}  ({len(manifest)} windows total)")

    # ── Label similarity report ───────────────────────────────────────────────
    if len(label_arrays) >= 2:
        pairs = label_similarity_report(label_arrays)
        print("\nLabel pair similarity (higher = more similar, harder to distinguish):")
        for l1, l2, sim in pairs:
            bar = '█' * int(sim * 30)
            print(f"  {l1:<18} ↔  {l2:<18}  {sim:.3f}  {bar}")

    print("\nDone. Set CLASSIFIER_ENABLED = True in config.py and restart the bridge.")


if __name__ == '__main__':
    main()
