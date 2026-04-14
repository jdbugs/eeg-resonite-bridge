"""
processor.py — EMA smoothing, session normalization, derived value computation
"""

import time
from collections import deque

BAND_NAMES = [
    'delta', 'theta', 'low_alpha', 'high_alpha',
    'low_beta', 'high_beta', 'low_gamma', 'high_gamma'
]

SCALAR_NAMES = ['attention', 'meditation', 'blink']

EPS = 1e-10


class Processor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ema_state   = {}
        self.session_min = {}
        self.session_max = {}
        self.start_time  = None
        self.warmup_done = False
        self.reset()

    # ─────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────

    def reset(self):
        for key in BAND_NAMES + SCALAR_NAMES:
            self.ema_state[key]   = None
            self.session_min[key] = float('inf')
            self.session_max[key] = float('-inf')
        self.start_time  = time.time()
        self.warmup_done = False
        long_n = getattr(self.cfg, 'LOG_ROLLING_LONG', 30)
        self._roll_att    = deque(maxlen=long_n)
        self._roll_med    = deque(maxlen=long_n)
        self._roll_energy = deque(maxlen=long_n)

    def recalibrate(self):
        """Full reset — clears all EMA state, min/max, and restarts warmup."""
        self.reset()

    # ─────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────

    def _ema(self, key, value, alpha):
        if self.ema_state[key] is None:
            self.ema_state[key] = float(value)
        else:
            self.ema_state[key] = alpha * float(value) + (1.0 - alpha) * self.ema_state[key]
        return self.ema_state[key]

    def _update_minmax(self, key, value):
        if value < self.session_min[key]:
            self.session_min[key] = value
        if value > self.session_max[key]:
            self.session_max[key] = value

        # Adaptive decay — slowly drift min/max toward current value so old
        # extreme anchors don't permanently compress the normalized range.
        decay = getattr(self.cfg, 'MIN_MAX_DECAY', 0.0)
        if decay > 0.0 and self.warmup_done:
            self.session_min[key] += (value - self.session_min[key]) * decay
            self.session_max[key] += (value - self.session_max[key]) * decay

    def _roll_mean(self, buf, n):
        items = list(buf)[-n:]
        return round(sum(items) / len(items), 4) if items else 0.0

    def _normalize(self, key, value):
        mn = self.session_min[key]
        mx = self.session_max[key]
        if mx <= mn:
            return 0.0
        return max(0.0, min(1.0, (value - mn) / (mx - mn + EPS)))

    # ─────────────────────────────────────────────────────
    # Main process method — called once per packet
    # ─────────────────────────────────────────────────────

    def process(self, raw: dict) -> dict:
        """
        raw dict keys:
            signal, attention, meditation, blink,
            delta, theta, low_alpha, high_alpha,
            low_beta, high_beta, low_gamma, high_gamma
        Returns processed dict ready for OSC + UI.
        """

        # Check warmup
        if not self.warmup_done:
            elapsed = time.time() - self.start_time
            if elapsed >= self.cfg.NORMALIZATION_WARMUP:
                self.warmup_done = True

        signal_ok = raw['signal'] < self.cfg.SIGNAL_THRESHOLD

        # ── Bands ──────────────────────────────────────────
        bands_normalized = {}
        bands_raw        = {}

        for name in BAND_NAMES:
            raw_val  = raw.get(name, 0)
            smoothed = self._ema(name, raw_val, self.cfg.EMA_ALPHA_BANDS)
            self._update_minmax(name, smoothed)
            normalized = self._normalize(name, smoothed)
            bands_normalized[name] = round(normalized, 4)
            bands_raw[name]        = raw_val

        # ── Attention / Meditation ─────────────────────────
        att  = self._ema('attention',  raw.get('attention', 0),  self.cfg.EMA_ALPHA_ATTENTION)
        med  = self._ema('meditation', raw.get('meditation', 0), self.cfg.EMA_ALPHA_ATTENTION)
        blnk = self._ema('blink',      raw.get('blink', 0),      self.cfg.EMA_ALPHA_BLINK)

        # ── Derived values ─────────────────────────────────
        alpha = (bands_normalized['low_alpha'] + bands_normalized['high_alpha']) / 2.0
        beta  = (bands_normalized['low_beta']  + bands_normalized['high_beta'])  / 2.0
        theta = bands_normalized['theta']

        alpha_beta_ratio  = min(1.0, alpha  / (beta  + EPS))
        engagement_index  = min(1.0, beta   / (alpha + theta + EPS))
        theta_alpha_ratio = min(1.0, theta  / (alpha + EPS))
        band_energy       = min(1.0, sum(bands_normalized.values()) / 8.0)

        dominant_idx   = max(range(len(BAND_NAMES)),
                             key=lambda i: bands_normalized[BAND_NAMES[i]])
        band_dominance = float(dominant_idx) / 7.0

        derived = {
            'alpha_beta_ratio':  round(alpha_beta_ratio,  4),
            'engagement_index':  round(engagement_index,  4),
            'theta_alpha_ratio': round(theta_alpha_ratio, 4),
            'band_energy':       round(band_energy,       4),
            'band_dominance':    round(band_dominance,    4),
            'dominant_band':     BAND_NAMES[dominant_idx],
        }

        # ── Rolling window stats ────────────────────────────
        self._roll_att.append(round(att / 100.0, 4))
        self._roll_med.append(round(med / 100.0, 4))
        self._roll_energy.append(round(band_energy, 4))

        short_n = getattr(self.cfg, 'LOG_ROLLING_SHORT', 10)
        long_n  = getattr(self.cfg, 'LOG_ROLLING_LONG',  30)
        rolling = {
            'att_10s':    self._roll_mean(self._roll_att,    short_n),
            'med_10s':    self._roll_mean(self._roll_med,    short_n),
            'energy_10s': self._roll_mean(self._roll_energy, short_n),
            'att_30s':    self._roll_mean(self._roll_att,    long_n),
            'med_30s':    self._roll_mean(self._roll_med,    long_n),
            'energy_30s': self._roll_mean(self._roll_energy, long_n),
        }

        # Expose normalization ranges for UI display
        bands_min = {n: round(self.session_min[n], 1) if self.session_min[n] != float('inf')  else 0 for n in BAND_NAMES}
        bands_max = {n: round(self.session_max[n], 1) if self.session_max[n] != float('-inf') else 0 for n in BAND_NAMES}

        return {
            'signal_ok':       signal_ok,
            'signal_quality':  raw['signal'],
            'warmup_done':     self.warmup_done,
            'attention':       round(att  / 100.0, 4),
            'meditation':      round(med  / 100.0, 4),
            'blink':           round(blnk / 255.0, 4),
            'attention_raw':   int(raw.get('attention', 0)),
            'meditation_raw':  int(raw.get('meditation', 0)),
            'bands':           bands_normalized,
            'bands_raw':       bands_raw,
            'bands_min':       bands_min,
            'bands_max':       bands_max,
            'derived':         derived,
            'rolling':         rolling,
        }
