# ─────────────────────────────────────────────────────────────
# EEG → Resonite OSC Bridge — Configuration
# Edit these values to match your setup
# ─────────────────────────────────────────────────────────────

# Serial / Arduino
SERIAL_PORT     = ""      # Leave empty for auto-detect, or set e.g. "COM3" / "/dev/ttyUSB0"
BAUD_RATE       = 9600    # 9600 for standard Brain Library, 57600 for raw EEG mode

# OSC → Resonite
OSC_IP          = "127.0.0.1"   # IP of machine running Resonite (127.0.0.1 = same machine)
OSC_PORT        = 9001           # Must match OSC_Receiver port in Resonite

# Local servers
WS_PORT         = 8765           # WebSocket port for UI communication
HTTP_PORT       = 8766           # HTTP port for serving the UI

# Signal quality — 0 = perfect, 200 = no contact
# Values above this threshold will gate OSC output
SIGNAL_THRESHOLD = 50

# EMA smoothing alphas (0.0–1.0, lower = smoother/slower)
EMA_ALPHA_BANDS      = 0.20
EMA_ALPHA_ATTENTION  = 0.15
EMA_ALPHA_BLINK      = 0.50
EMA_ALPHA_DERIVED    = 0.10

# Normalization warmup period in seconds
# Min/max scaling will be marked unstable during this window
NORMALIZATION_WARMUP = 5

# Min/max normalization adaptive decay
# Each packet, session min creeps up and max creeps down toward current value.
# 0.0 = permanent anchoring (original behaviour), 0.0002 ≈ ~20-minute half-life.
# Keeps normalization current without drifting mid-session.
MIN_MAX_DECAY   = 0.0002

# Session logging
ENABLE_LOGGING  = False
LOG_DIR         = "./logs"

# Rolling window lengths (in packets — at ~1 Hz, 1 packet ≈ 1 second)
# Used to add short-term and medium-term trend columns to the CSV log.
LOG_ROLLING_SHORT = 10   # ~10 seconds
LOG_ROLLING_LONG  = 30   # ~30 seconds
