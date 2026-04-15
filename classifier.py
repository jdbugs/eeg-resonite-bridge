"""
classifier.py — Rolling cosine-similarity nearest-neighbour classifier.

Loaded as an optional module by main.py when cfg.CLASSIFIER_ENABLED = True.

Pipeline:
  1. Each packet's normalised band values are appended to a 15-packet
     rolling buffer.
  2. When the buffer is full, the 8×15 window is flattened into a
     120-dimensional feature vector.
  3. Cosine similarity is computed against every training example in the
     manifest (vectorised via numpy, < 1 ms for typical dataset sizes).
  4. If the best match exceeds cfg.CLASSIFIER_THRESHOLD and the per-label
     cooldown has expired, a command dict is returned to the caller.

The caller (dispatch_loop in main.py) is responsible for sending the OSC
events and broadcasting the result to the UI.

Demo mode: if training_data/manifest.json does not exist, the classifier
loads in demo mode and fires random low-confidence events so the OSC wiring
can be tested in Resonite before real training data exists.
"""

import json
import pathlib
import random
import time
from collections import deque

import numpy as np

BAND_NAMES = [
    'delta', 'theta', 'low_alpha', 'high_alpha',
    'low_beta', 'high_beta', 'low_gamma', 'high_gamma',
]

LABEL_ORDER = ['baseline', 'eyes_closed', 'arithmetic', 'imagery', 'awareness']

WINDOW_LEN = 15   # packets (== seconds at 1 Hz)


class Classifier:
    def __init__(self, cfg):
        self.cfg       = cfg
        self.threshold = getattr(cfg, 'CLASSIFIER_THRESHOLD', 0.75)
        self.cooldown  = getattr(cfg, 'CLASSIFIER_COOLDOWN',  3.0)
        self.data_dir  = pathlib.Path(getattr(cfg, 'TRAINING_DATA_DIR', './training_data'))

        self._buffer       = deque(maxlen=WINDOW_LEN)
        self._last_fire    = {}          # label → timestamp of last fire

        # Precomputed from manifest
        self._train_matrix = None        # np.array (N, 120)
        self._train_norms  = None        # np.array (N,)
        self._labels_list  = []          # list[str] length N
        self.counts_by_label: dict       = {}

        self._demo_mode = False
        self._load()

    # ─────────────────────────────────────────────────────────
    # Public properties
    # ─────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return not self._demo_mode

    @property
    def total_examples(self) -> int:
        return len(self._labels_list)

    # ─────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────

    def _load(self):
        manifest_path = self.data_dir / 'manifest.json'
        if not manifest_path.exists():
            self._demo_mode = True
            return

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            vectors, labels = [], []
            for entry in manifest:
                label  = entry.get('label', '')
                vector = entry.get('vector')
                if not label or not vector:
                    continue
                arr = np.array(vector, dtype=np.float32)
                if arr.shape != (WINDOW_LEN * len(BAND_NAMES),):
                    continue   # unexpected shape — skip
                vectors.append(arr)
                labels.append(label)
                self.counts_by_label[label] = self.counts_by_label.get(label, 0) + 1

            if not vectors:
                self._demo_mode = True
                return

            self._train_matrix = np.stack(vectors)                          # (N, 120)
            self._train_norms  = np.linalg.norm(self._train_matrix, axis=1) + 1e-8
            self._labels_list  = labels

        except Exception as e:
            print(f"  [Classifier] Load error: {e}")
            self._demo_mode = True

    def reload(self):
        """Reload training data from disk — callable from the UI."""
        self._train_matrix  = None
        self._train_norms   = None
        self._labels_list   = []
        self.counts_by_label = {}
        self._demo_mode     = False
        self._last_fire     = {}
        self._buffer.clear()
        self._load()

    # ─────────────────────────────────────────────────────────
    # Update — call once per packet from dispatch_loop
    # ─────────────────────────────────────────────────────────

    def update(self, bands: dict) -> dict | None:
        """
        Add a packet's band values to the rolling buffer and run
        classification when the buffer is full.

        Returns a command dict  {"label", "confidence", "index"}
        if a label fires, otherwise None.
        Adds at most ~1 ms to the dispatch cycle for typical dataset sizes.
        """
        vec = np.array([bands.get(b, 0.0) for b in BAND_NAMES], dtype=np.float32)
        self._buffer.append(vec)

        if len(self._buffer) < WINDOW_LEN:
            return None

        return self._demo_fire() if self._demo_mode else self._classify()

    # ─────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────

    def _classify(self) -> dict | None:
        if self._train_matrix is None:
            return None

        # Build (8, 15) window matching manifest row-major order:
        # arr[band, time] → flatten → [delta_t0..t14, theta_t0..t14, ...]
        arr = np.zeros((len(BAND_NAMES), WINDOW_LEN), dtype=np.float32)
        for t, vec in enumerate(self._buffer):
            arr[:, t] = vec
        feature    = arr.flatten()
        feat_norm  = np.linalg.norm(feature)
        if feat_norm < 1e-8:
            return None

        # Vectorised cosine similarity against all training examples
        sims      = (self._train_matrix @ feature) / (self._train_norms * feat_norm)
        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])
        best_label = self._labels_list[best_idx]

        if best_sim < self.threshold:
            return None

        now = time.monotonic()
        if now - self._last_fire.get(best_label, 0.0) < self.cooldown:
            return None

        self._last_fire[best_label] = now
        return {
            "label":      best_label,
            "confidence": round(best_sim, 4),
            "index":      LABEL_ORDER.index(best_label) if best_label in LABEL_ORDER else -1,
        }

    def _demo_fire(self) -> dict | None:
        """
        Fire random low-confidence events so OSC wiring can be verified in
        Resonite before real training data exists (~5 % of packets, ≈ every 20s).
        """
        if random.random() > 0.05:
            return None
        label = random.choice(LABEL_ORDER)
        return {
            "label":      label,
            "confidence": round(random.uniform(0.30, 0.55), 4),
            "index":      LABEL_ORDER.index(label),
        }
