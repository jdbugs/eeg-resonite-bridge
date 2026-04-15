#!/usr/bin/env python3
"""
EEG → Resonite OSC Bridge
Serial reader + processor + OSC sender + WebSocket server for UI

Run: python main.py
Then open http://localhost:8766/ui.html in your browser
"""

import asyncio
import threading
import queue
import json
import time
import datetime
import csv
import pathlib
import sys
import os
import webbrowser
import http.server
import functools

# ─────────────────────────────────────────────────────────
# Dependency checks with friendly error messages
# ─────────────────────────────────────────────────────────

def check_dep(module, install):
    try:
        return __import__(module)
    except ImportError:
        print(f"\n  MISSING: {module}")
        print(f"  Install:  pip install {install}\n")
        sys.exit(1)

serial_mod  = check_dep("serial",      "pyserial")
ws_mod      = check_dep("websockets",  "websockets")
osc_mod     = check_dep("pythonosc",   "python-osc")

import serial
import serial.tools.list_ports
import websockets
from pythonosc import udp_client
import config as cfg
from processor import Processor, BAND_NAMES

# ── Optional classifier ───────────────────────────────────────────
# Loaded at startup if cfg.CLASSIFIER_ENABLED = True.
# If classifier.py or numpy is missing the bridge still runs normally.
_classifier = None
if getattr(cfg, 'CLASSIFIER_ENABLED', False):
    try:
        from classifier import Classifier
        _classifier = Classifier(cfg)
    except ImportError:
        print("  [WARN] classifier.py or numpy not found — classifier disabled.")
    except Exception as _cls_err:
        print(f"  [WARN] Classifier failed to load: {_cls_err}")

# ─────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────

data_queue  = queue.Queue(maxsize=200)
ws_clients  = set()
event_loop  = None                  # set in main(), used for thread→async bridging

app = {
    "running":        False,
    "logging":        False,
    "stop_event":     threading.Event(),
    "serial_thread":  None,
    "osc_client":     None,
    "processor":      None,
    "log_file":       None,
    "csv_writer":     None,
    "ann_file":       None,   # annotation CSV
    "ann_writer":     None,
    "evt_file":       None,   # events CSV
    "evt_writer":     None,
    "session_stem":   None,   # base path (no extension) for current session files
    "session_start":  None,   # datetime — used for elapsed time + summary
    "stats":          {},     # session accumulators for summary JSON
    "last_dominant":  None,   # for dominant-band change events
    "last_signal_ok": None,   # for signal change events
    "last_warmup":    False,  # for warmup-complete event
    "last_data":      None,
    "connected_port": "",
    "osc_suppressed": False,   # True while loopback test is running
}

# ─────────────────────────────────────────────────────────
# Training protocol
# ─────────────────────────────────────────────────────────

TRAINING_PROTOCOL = [
    {
        "name":        "SETTLING",
        "label":       "",
        "duration":    15,
        "instruction": "Sit comfortably. Let the signal stabilize. Breathe normally.",
    },
    {
        "name":        "BASELINE",
        "label":       "baseline",
        "duration":    20,
        "instruction": "Eyes open. Soft focus. No particular thought. Just rest.",
    },
    {
        "name":        "EYES_CLOSED",
        "label":       "eyes_closed",
        "duration":    20,
        "instruction": "Close your eyes. Let thoughts drift. Don't direct anything.",
    },
    {
        "name":        "ARITHMETIC",
        "label":       "arithmetic",
        "duration":    20,
        "instruction": "Count backwards from 300 by 7s silently. 300... 293... 286...",
    },
    {
        "name":        "IMAGERY",
        "label":       "imagery",
        "duration":    20,
        "instruction": "Eyes closed. Build a detailed familiar space in your mind. Light, textures, scale.",
    },
    {
        "name":        "AWARENESS",
        "label":       "awareness",
        "duration":    20,
        "instruction": "Eyes open. Take in everything in your visual field equally. Wide diffuse attention.",
    },
]

_TRAINING_TOTAL_DURATION = sum(s["duration"] for s in TRAINING_PROTOCOL)

_training = {
    "active":     False,
    "round":      0,
    "label":      "",       # raw label (empty during SETTLING or idle)
    "full_label": "",       # label + round suffix e.g. "baseline_r1"
    "segment":    "",       # segment name e.g. "BASELINE"
    "task":       None,     # asyncio.Task reference
}

# ─────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────

def start_log():
    pathlib.Path(cfg.LOG_DIR).mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{cfg.LOG_DIR}/session_{ts}"
    app['session_stem']  = stem
    app['session_start'] = datetime.datetime.now()

    # Reset session stats accumulators
    app['stats'] = {
        'packet_count':   0,
        'signal_ok_count': 0,
        'att_sum':        0.0,
        'med_sum':        0.0,
        'blink_count':    0,
        'dominant_bands': {},
    }
    app['last_dominant']  = None
    app['last_signal_ok'] = None
    app['last_warmup']    = False

    # ── Main session CSV ──────────────────────────────────
    f      = open(f"{stem}.csv", 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(
        ['timestamp', 'signal_quality', 'attention_raw', 'meditation_raw',
         'attention', 'meditation', 'blink']
        + BAND_NAMES
        + ['alpha_beta_ratio', 'engagement_index', 'theta_alpha_ratio',
           'band_energy', 'band_dominance', 'dominant_band']
        + ['att_10s', 'med_10s', 'energy_10s',
           'att_30s', 'med_30s', 'energy_30s']
        + ['training_label', 'training_round', 'signal_poor']
    )
    app['log_file']   = f
    app['csv_writer'] = writer

    # ── Annotations CSV ───────────────────────────────────
    af = open(f"{stem}_annotations.csv", 'w', newline='')
    aw = csv.writer(af)
    aw.writerow(['timestamp', 'elapsed_s', 'annotation'])
    app['ann_file']   = af
    app['ann_writer'] = aw

    # ── Events CSV ────────────────────────────────────────
    ef = open(f"{stem}_events.csv", 'w', newline='')
    ew = csv.writer(ef)
    ew.writerow(['timestamp', 'elapsed_s', 'event', 'value'])
    app['evt_file']   = ef
    app['evt_writer'] = ew

    return f"{stem}.csv"


def stop_log():
    if app.get('session_stem') and app.get('session_start'):
        _write_summary()

    for key in ('log_file', 'ann_file', 'evt_file'):
        if app[key]:
            try:
                app[key].close()
            except Exception:
                pass
    app['log_file']      = None
    app['csv_writer']    = None
    app['ann_file']      = None
    app['ann_writer']    = None
    app['evt_file']      = None
    app['evt_writer']    = None
    app['session_stem']  = None
    app['session_start'] = None


def _write_summary():
    """Write a JSON summary alongside the session CSV on stop."""
    stats = app.get('stats', {})
    n     = max(stats.get('packet_count', 0), 1)
    start = app['session_start']
    end   = datetime.datetime.now()
    dom   = stats.get('dominant_bands', {})

    summary = {
        'session_start':    start.isoformat(),
        'session_end':      end.isoformat(),
        'duration_seconds': round((end - start).total_seconds(), 1),
        'total_packets':    stats.get('packet_count', 0),
        'signal_ok_pct':    round(stats.get('signal_ok_count', 0) / n * 100, 1),
        'mean_attention':   round(stats.get('att_sum', 0.0) / n, 4),
        'mean_meditation':  round(stats.get('med_sum', 0.0) / n, 4),
        'blink_count':      stats.get('blink_count', 0),
        'dominant_band_distribution': dict(
            sorted(dom.items(), key=lambda x: x[1], reverse=True)
        ),
    }
    try:
        with open(f"{app['session_stem']}_summary.json", 'w') as jf:
            json.dump(summary, jf, indent=2)
    except Exception as e:
        _thread_log(f"Summary write error: {e}", "warn")


def write_event_row(event: str, value: str = ''):
    """Append a timestamped event to the events CSV."""
    if not app.get('evt_writer') or not app.get('session_start'):
        return
    try:
        elapsed = (datetime.datetime.now() - app['session_start']).total_seconds()
        app['evt_writer'].writerow([
            datetime.datetime.now().isoformat(),
            round(elapsed, 2),
            event,
            value,
        ])
        app['evt_file'].flush()
    except Exception as e:
        _thread_log(f"Event write error: {e}", "warn")


def write_annotation_row(text: str):
    """Append a timestamped user annotation to the annotations CSV."""
    if not app.get('ann_writer') or not app.get('session_start'):
        return
    try:
        elapsed = (datetime.datetime.now() - app['session_start']).total_seconds()
        app['ann_writer'].writerow([
            datetime.datetime.now().isoformat(),
            round(elapsed, 2),
            text,
        ])
        app['ann_file'].flush()
    except Exception as e:
        _thread_log(f"Annotation write error: {e}", "warn")


def write_log_row(data: dict):
    if not app['csv_writer']:
        return
    try:
        bands   = data.get('bands',   {})
        derived = data.get('derived', {})
        rolling = data.get('rolling', {})

        app['csv_writer'].writerow(
            [datetime.datetime.now().isoformat(),
             data.get('signal_quality', 200),
             data.get('attention_raw',  0),
             data.get('meditation_raw', 0),
             data.get('attention', 0),
             data.get('meditation', 0),
             data.get('blink',     0)]
            + [bands.get(n, 0) for n in BAND_NAMES]
            + [derived.get('alpha_beta_ratio',  0),
               derived.get('engagement_index',  0),
               derived.get('theta_alpha_ratio', 0),
               derived.get('band_energy',       0),
               derived.get('band_dominance',    0),
               derived.get('dominant_band',     '')]
            + [rolling.get('att_10s',    0),
               rolling.get('med_10s',    0),
               rolling.get('energy_10s', 0),
               rolling.get('att_30s',    0),
               rolling.get('med_30s',    0),
               rolling.get('energy_30s', 0)]
            + [
               # training_label — empty during SETTLING or when idle
               _training.get("full_label", ""),
               # training_round — populated whenever a round is active
               _training.get("round", "") if _training.get("active") else "",
               # signal_poor — 1 only during a labelled segment with poor contact
               1 if (_training.get("full_label") and
                     data.get("signal_quality", 200) > 50) else 0,
              ]
        )

        # ── Update session stats ──────────────────────────
        stats = app.get('stats')
        if stats is not None:
            stats['packet_count'] += 1
            if data.get('signal_ok'):
                stats['signal_ok_count'] += 1
            stats['att_sum'] += data.get('attention', 0)
            stats['med_sum'] += data.get('meditation', 0)
            if data.get('blink', 0) > 0:
                stats['blink_count'] += 1
            dom = derived.get('dominant_band', '')
            if dom:
                stats['dominant_bands'][dom] = stats['dominant_bands'].get(dom, 0) + 1

        # ── Event detection ───────────────────────────────
        if app.get('evt_writer'):
            sig_ok = data.get('signal_ok', False)
            warmup = data.get('warmup_done', False)
            dom    = derived.get('dominant_band', '')
            blink  = data.get('blink', 0)

            if app['last_signal_ok'] is not None and sig_ok != app['last_signal_ok']:
                write_event_row('signal_change', 'ok' if sig_ok else 'lost')
            app['last_signal_ok'] = sig_ok

            if warmup and not app['last_warmup']:
                write_event_row('warmup_complete', '')
            app['last_warmup'] = warmup

            if dom and dom != app['last_dominant']:
                write_event_row('dominant_band_change', dom)
                app['last_dominant'] = dom

            if blink > 0:
                write_event_row('blink_detected', str(round(blink, 4)))

    except Exception as e:
        _thread_log(f"Log write error: {e}", "warn")

# ─────────────────────────────────────────────────────────
# Thread-safe broadcast helpers
# ─────────────────────────────────────────────────────────

def _thread_log(message: str, level: str = "info"):
    """Can be called from any thread."""
    if event_loop and not event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(
            _broadcast(json.dumps({"type": "log", "message": message, "level": level})),
            event_loop
        )
    else:
        print(f"[{level.upper()}] {message}")


def _thread_broadcast(payload: dict):
    """Can be called from any thread."""
    if event_loop and not event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(
            _broadcast(json.dumps(payload)),
            event_loop
        )


async def _broadcast(message: str):
    """Async broadcast to all connected WebSocket clients."""
    if not ws_clients:
        return
    dead = set()
    for ws in ws_clients.copy():
        try:
            await ws.send(message)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


def _send_status():
    _thread_broadcast({
        "type":       "status",
        "running":    app["running"],
        "logging":    app["logging"],
        "signal_ok":  app["last_data"]["signal_ok"] if app.get("last_data") else False,
        "port":       app["connected_port"],
    })

# ─────────────────────────────────────────────────────────
# Serial reader (runs in its own thread)
# ─────────────────────────────────────────────────────────

def _parse_brain_csv(line: str) -> dict | None:
    """
    Brain Library CSV format:
      signal,attention,meditation,delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,lowGamma,highGamma[,blink]
    Returns None on any parse failure.
    """
    parts = line.strip().split(',')
    if len(parts) < 11:
        return None
    try:
        result = {
            'signal':     int(parts[0]),
            'attention':  int(parts[1]),
            'meditation': int(parts[2]),
        }
        for i, name in enumerate(BAND_NAMES):
            result[name] = int(parts[3 + i])
        result['blink'] = int(parts[11]) if len(parts) >= 12 else 0
        return result
    except (ValueError, IndexError):
        return None


def serial_loop(port: str, baud: int, stop_event: threading.Event, processor: Processor):
    """Main serial reading loop. Runs in a daemon thread."""
    ser = None
    _thread_log(f"Connecting to {port} @ {baud} baud …", "info")

    try:
        ser = serial.Serial(port, baud, timeout=2.0)
    except serial.SerialException as e:
        _thread_log(f"Cannot open {port}: {e}", "error")
        app["running"] = False
        _send_status()
        return

    _thread_log(f"Serial open: {port}", "info")
    app["connected_port"] = port
    time.sleep(2.0)      # let Arduino reset
    ser.flushInput()

    consecutive_errors = 0
    last_good_ts       = time.time()

    try:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except serial.SerialException as e:
                _thread_log(f"Serial read error: {e}", "error")
                time.sleep(0.5)
                continue

            if not line:
                # Timeout (no data for 2 seconds)
                if time.time() - last_good_ts > 5.0 and app["running"]:
                    _thread_log("No data for 5 seconds — check headset connection.", "warn")
                    last_good_ts = time.time()
                continue

            raw = _parse_brain_csv(line)
            if raw is None:
                consecutive_errors += 1
                if consecutive_errors == 5:
                    _thread_log(
                        f"Parse errors (5 in a row). Last line: {repr(line[:60])} "
                        f"— Is the Arduino running BrainSerialOut?",
                        "warn"
                    )
                continue

            consecutive_errors = 0
            last_good_ts       = time.time()

            # Process
            try:
                data = processor.process(raw)
            except Exception as e:
                _thread_log(f"Processor error: {e}", "warn")
                continue

            # Queue for asyncio dispatch
            try:
                data_queue.put_nowait(data)
            except queue.Full:
                # Drop oldest, push newest
                try:
                    data_queue.get_nowait()
                    data_queue.put_nowait(data)
                except Exception:
                    pass

    except Exception as e:
        _thread_log(f"Serial thread crashed: {e}", "error")
    finally:
        if ser and ser.is_open:
            ser.close()
        _thread_log("Serial closed.", "info")
        app["running"]        = False
        app["connected_port"] = ""
        _send_status()

# ─────────────────────────────────────────────────────────
# OSC sender
# ─────────────────────────────────────────────────────────

def send_osc(data: dict):
    if app.get("osc_suppressed"):
        return
    c = app.get("osc_client")
    if not c:
        return

    try:
        # Always send system flags
        c.send_message("/eeg/system/signal_ok",
                       1.0 if data["signal_ok"] else 0.0)
        c.send_message("/eeg/system/warmup_done",
                       1.0 if data["warmup_done"] else 0.0)

        if not data["signal_ok"]:
            return   # Gate everything else on signal quality

        c.send_message("/eeg/raw/attention",  float(data["attention"]))
        c.send_message("/eeg/raw/meditation", float(data["meditation"]))
        c.send_message("/eeg/raw/blink",      float(data["blink"]))

        for name, val in data["bands"].items():
            c.send_message(f"/eeg/band/{name}", float(val))

        for name, val in data["derived"].items():
            if isinstance(val, (int, float)):
                c.send_message(f"/eeg/derived/{name}", float(val))

    except Exception as e:
        _thread_log(f"OSC error: {e}", "warn")

# ─────────────────────────────────────────────────────────
# Asyncio data dispatch task
# ─────────────────────────────────────────────────────────

async def dispatch_loop():
    """Reads processed data from queue → OSC + UI + logger + classifier."""
    while True:
        try:
            data = data_queue.get_nowait()
            app["last_data"] = data

            if app["running"]:
                send_osc(data)

            if app["logging"]:
                write_log_row(data)

            # ── Live classifier ───────────────────────────────
            if _classifier and app["running"] and data.get("signal_ok"):
                cmd = _classifier.update(data.get("bands", {}))
                if cmd:
                    _send_classifier_osc(cmd)
                    await _broadcast(json.dumps({"type": "classifier_event", **cmd}))

            await _broadcast(json.dumps({"type": "data", "data": data}))

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Dispatch error: {e}")

        await asyncio.sleep(0.05)   # ~20 Hz UI refresh

# ─────────────────────────────────────────────────────────
# COM port helpers
# ─────────────────────────────────────────────────────────

def _get_ports() -> list[dict]:
    """Return list of dicts with device + description."""
    ports = []
    for p in serial.tools.list_ports.comports():
        desc = p.description or ""
        is_arduino = any(k in desc.lower() for k in ("arduino", "ch340", "ft232", "cp210"))
        ports.append({
            "device":     p.device,
            "description": desc,
            "arduino":    is_arduino,
        })
    return ports

# ─────────────────────────────────────────────────────────
# Training protocol runner
# ─────────────────────────────────────────────────────────

async def run_training_round(round_num: int):
    """
    Runs the full training protocol as an asyncio task.
    Broadcasts a 'training_segment' message at the start of each segment
    so the UI can update the countdown and instructions.
    Writes training_label / training_round / signal_poor into every CSV
    row via the _training state dict (read in write_log_row).
    """
    _training["active"] = True
    _training["round"]  = round_num
    elapsed_total       = 0

    try:
        for seg in TRAINING_PROTOCOL:
            _training["label"]      = seg["label"]
            _training["full_label"] = (
                f"{seg['label']}_r{round_num}" if seg["label"] else ""
            )
            _training["segment"] = seg["name"]

            await _broadcast(json.dumps({
                "type":           "training_segment",
                "segment_name":   seg["name"],
                "label":          seg["label"],
                "instruction":    seg["instruction"],
                "duration":       seg["duration"],
                "elapsed_total":  elapsed_total,
                "total_duration": _TRAINING_TOTAL_DURATION,
                "round":          round_num,
            }))

            _thread_log(
                f"Training r{round_num}: {seg['name']} ({seg['duration']}s) …", "info"
            )
            await asyncio.sleep(seg["duration"])
            elapsed_total += seg["duration"]

        await _broadcast(json.dumps({
            "type":  "training_complete",
            "round": round_num,
        }))
        _thread_log(f"Training round {round_num} complete.", "info")

    except asyncio.CancelledError:
        _thread_log("Training cancelled.", "warn")
        raise

    finally:
        _training["active"]     = False
        _training["label"]      = ""
        _training["full_label"] = ""
        _training["segment"]    = ""
        _training["task"]       = None


# ─────────────────────────────────────────────────────────
# Classifier OSC sender
# ─────────────────────────────────────────────────────────

def _send_classifier_osc(cmd: dict):
    """Send a classifier command event over OSC."""
    c = app.get("osc_client")
    if not c or app.get("osc_suppressed"):
        return
    try:
        c.send_message("/eeg/command/label",      cmd["label"])
        c.send_message("/eeg/command/confidence", float(cmd["confidence"]))
        c.send_message("/eeg/command/index",      int(cmd["index"]))
    except Exception as e:
        _thread_log(f"Classifier OSC error: {e}", "warn")


def _classifier_status_payload() -> dict:
    """Build the classifier_status WebSocket payload."""
    if _classifier is None:
        return {
            "type":      "classifier_status",
            "enabled":   False,
            "loaded":    False,
            "demo":      False,
            "counts":    {},
            "total":     0,
            "threshold": getattr(cfg, 'CLASSIFIER_THRESHOLD', 0.75),
        }
    return {
        "type":      "classifier_status",
        "enabled":   True,
        "loaded":    _classifier.loaded,
        "demo":      _classifier._demo_mode,
        "counts":    _classifier.counts_by_label,
        "total":     _classifier.total_examples,
        "threshold": _classifier.threshold,
    }


# ─────────────────────────────────────────────────────────
# OSC loopback test
# ─────────────────────────────────────────────────────────

TEST_OSC_ADDRESSES = [
    "/eeg/system/signal_ok",
    "/eeg/system/warmup_done",
    "/eeg/raw/attention",
    "/eeg/raw/meditation",
    "/eeg/raw/blink",
    "/eeg/band/delta",
    "/eeg/band/theta",
    "/eeg/band/low_alpha",
    "/eeg/band/high_alpha",
    "/eeg/band/low_beta",
    "/eeg/band/high_beta",
    "/eeg/band/low_gamma",
    "/eeg/band/high_gamma",
    "/eeg/derived/alpha_beta_ratio",
    "/eeg/derived/engagement_index",
    "/eeg/derived/theta_alpha_ratio",
    "/eeg/derived/band_energy",
    "/eeg/derived/band_dominance",
]


async def run_osc_test(osc_port: int) -> dict:
    """
    Loopback OSC test — always uses 127.0.0.1 regardless of configured target IP.
    1. Binds a listener on 127.0.0.1:osc_port
    2. Sends one test message per address to that listener
    3. Waits briefly for UDP packets to arrive
    4. Returns pass/fail results per address
    """
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.dispatcher import Dispatcher

    received: dict = {}

    def _handler(addr, *args):
        received[addr] = float(args[0]) if args else 0.0

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(_handler)

    try:
        server = ThreadingOSCUDPServer(("127.0.0.1", osc_port), dispatcher)
    except OSError as e:
        return {"error": f"Cannot bind 127.0.0.1:{osc_port} — {e}"}

    srv_thread = threading.Thread(target=server.serve_forever, daemon=True, name="osc-test-srv")
    srv_thread.start()
    await asyncio.sleep(0.05)   # let the thread start listening

    # Build test values from last known session data (or sensible dummies)
    last   = app.get("last_data") or {}
    bands  = last.get("bands",   {})
    derived = last.get("derived", {})
    test_vals = {
        "/eeg/system/signal_ok":          1.0,
        "/eeg/system/warmup_done":        1.0,
        "/eeg/raw/attention":             last.get("attention",  0.5),
        "/eeg/raw/meditation":            last.get("meditation", 0.5),
        "/eeg/raw/blink":                 last.get("blink",      0.0),
        **{f"/eeg/band/{n}": bands.get(n, 0.5) for n in BAND_NAMES},
        "/eeg/derived/alpha_beta_ratio":  derived.get("alpha_beta_ratio",  0.5),
        "/eeg/derived/engagement_index":  derived.get("engagement_index",  0.5),
        "/eeg/derived/theta_alpha_ratio": derived.get("theta_alpha_ratio", 0.5),
        "/eeg/derived/band_energy":       derived.get("band_energy",       0.5),
        "/eeg/derived/band_dominance":    derived.get("band_dominance",    0.5),
    }

    # Suppress live OSC so it doesn't bleed into the test listener
    app["osc_suppressed"] = True
    try:
        test_client = udp_client.SimpleUDPClient("127.0.0.1", osc_port)
        for addr, val in test_vals.items():
            test_client.send_message(addr, float(val))
        await asyncio.sleep(0.4)   # give UDP time to deliver on loopback
    finally:
        server.shutdown()
        app["osc_suppressed"] = False

    results = []
    all_passed = True
    for addr in TEST_OSC_ADDRESSES:
        ok = addr in received
        results.append({
            "addr":     addr,
            "received": ok,
            "sent":     round(test_vals.get(addr, 0.5), 4),
            "value":    round(received[addr], 4) if ok else None,
        })
        if not ok:
            all_passed = False

    return {"results": results, "all_passed": all_passed}


# ─────────────────────────────────────────────────────────
# WebSocket command handler
# ─────────────────────────────────────────────────────────

async def handle_command(msg: dict, ws):
    cmd = msg.get("type", "")

    # ── get_ports ───────────────────────────────────────
    if cmd == "get_ports":
        await ws.send(json.dumps({"type": "ports", "ports": _get_ports()}))

    # ── start ────────────────────────────────────────────
    elif cmd == "start":
        if app["running"]:
            await ws.send(json.dumps({
                "type": "log", "message": "Already running.", "level": "warn"
            }))
            return

        port     = msg.get("port", "").strip()
        baud     = int(msg.get("baud",     cfg.BAUD_RATE))
        osc_ip   = msg.get("osc_ip",   cfg.OSC_IP).strip()
        osc_port = int(msg.get("osc_port", cfg.OSC_PORT))

        if not port:
            await ws.send(json.dumps({
                "type": "log", "message": "Select a serial port first.", "level": "error"
            }))
            return

        # Setup OSC
        try:
            app["osc_client"] = udp_client.SimpleUDPClient(osc_ip, osc_port)
            _thread_log(f"OSC → {osc_ip}:{osc_port}", "info")
        except Exception as e:
            await ws.send(json.dumps({
                "type": "log", "message": f"OSC setup failed: {e}", "level": "error"
            }))
            return

        # Start processor + serial thread
        app["processor"] = Processor(cfg)
        app["stop_event"].clear()
        app["running"] = True
        app["serial_thread"] = threading.Thread(
            target=serial_loop,
            args=(port, baud, app["stop_event"], app["processor"]),
            daemon=True,
            name="serial-reader"
        )
        app["serial_thread"].start()

        # Start logging
        if cfg.ENABLE_LOGGING:
            try:
                fname = start_log()
                app["logging"] = True
                _thread_log(f"Logging → {fname}", "info")
            except Exception as e:
                _thread_log(f"Log start failed: {e}", "warn")

        _send_status()
        _thread_log(f"Started — {port} @ {baud}", "info")

    # ── stop ─────────────────────────────────────────────
    elif cmd == "stop":
        if not app["running"]:
            return
        app["stop_event"].set()
        app["running"] = False
        stop_log()
        app["logging"] = False
        _send_status()
        _thread_log("Stopped.", "info")

    # ── toggle_logging ───────────────────────────────────
    elif cmd == "toggle_logging":
        if app["logging"]:
            stop_log()
            app["logging"] = False
            _thread_log("Logging off.", "info")
        else:
            try:
                fname = start_log()
                app["logging"] = True
                _thread_log(f"Logging → {fname}", "info")
            except Exception as e:
                _thread_log(f"Log failed: {e}", "warn")
        _send_status()

    # ── recalibrate ──────────────────────────────────────
    elif cmd == "recalibrate":
        if app["processor"]:
            app["processor"].recalibrate()
            write_event_row('recalibrate', '')
            _thread_log("Recalibrated — normalization reset.", "info")
        else:
            await ws.send(json.dumps({
                "type": "log", "message": "Not running — nothing to recalibrate.", "level": "warn"
            }))

    # ── annotate ─────────────────────────────────────────
    elif cmd == "annotate":
        text = msg.get("text", "").strip()
        if not text:
            return
        if app["logging"]:
            write_annotation_row(text)
            write_event_row('annotation', text[:80])
            _thread_log(f"Annotated: {text[:60]}", "info")
        else:
            await ws.send(json.dumps({
                "type": "log",
                "message": "Logging is off — start logging first to save annotations.",
                "level": "warn"
            }))

    # ── set_osc ──────────────────────────────────────────
    elif cmd == "set_osc":
        osc_ip   = msg.get("ip",   cfg.OSC_IP).strip()
        osc_port = int(msg.get("port", cfg.OSC_PORT))
        try:
            app["osc_client"] = udp_client.SimpleUDPClient(osc_ip, osc_port)
            _thread_log(f"OSC updated → {osc_ip}:{osc_port}", "info")
        except Exception as e:
            await ws.send(json.dumps({
                "type": "log",
                "message": f"OSC update failed: {e}",
                "level": "error"
            }))

    # ── test_osc ─────────────────────────────────────────
    elif cmd == "test_osc":
        osc_port = int(msg.get("port", cfg.OSC_PORT))
        _thread_log(f"OSC loopback test on 127.0.0.1:{osc_port} …", "info")
        try:
            result = await run_osc_test(osc_port)
            if "error" in result:
                _thread_log(f"OSC test failed: {result['error']}", "error")
                await ws.send(json.dumps({"type": "osc_test_result", "error": result["error"]}))
            else:
                n_ok  = sum(1 for r in result["results"] if r["received"])
                n_tot = len(result["results"])
                status = "PASS" if result["all_passed"] else "FAIL"
                level  = "info" if result["all_passed"] else "warn"
                _thread_log(f"OSC test {status} — {n_ok}/{n_tot} addresses received", level)
                await ws.send(json.dumps({"type": "osc_test_result", **result}))
        except Exception as e:
            _thread_log(f"OSC test error: {e}", "error")
            await ws.send(json.dumps({"type": "osc_test_result", "error": str(e)}))

    # ── start_training ───────────────────────────────────
    elif cmd == "start_training":
        if _training["active"]:
            await ws.send(json.dumps({
                "type": "log", "message": "Training already running.", "level": "warn"
            }))
            return
        if not app["logging"]:
            _thread_log(
                "Logging is off — training data will not be saved. "
                "Enable logging to capture labelled EEG data.", "warn"
            )
        round_num = _training["round"] + 1
        _training["task"] = asyncio.create_task(run_training_round(round_num))

    # ── stop_training ────────────────────────────────────
    elif cmd == "stop_training":
        task = _training.get("task")
        if task and not task.done():
            task.cancel()
        _thread_log("Training stopped.", "warn")

    # ── load_classifier ──────────────────────────────────
    elif cmd == "load_classifier":
        if _classifier is None:
            await ws.send(json.dumps({
                "type": "log",
                "message": "Classifier not enabled — set CLASSIFIER_ENABLED = True in config.py.",
                "level": "warn"
            }))
        else:
            _classifier.reload()
            status = (f"{_classifier.total_examples} examples loaded"
                      if _classifier.loaded else "demo mode (no manifest found)")
            _thread_log(f"Classifier reloaded — {status}", "info")
            await ws.send(json.dumps(_classifier_status_payload()))

    # ── set_threshold ────────────────────────────────────
    elif cmd == "set_threshold":
        if _classifier:
            t = max(0.1, min(1.0, float(msg.get("threshold", 0.75))))
            _classifier.threshold = t
            _thread_log(f"Classifier threshold → {t:.2f}", "info")

# ─────────────────────────────────────────────────────────
# WebSocket connection handler
# ─────────────────────────────────────────────────────────

async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"  UI connected ({len(ws_clients)} client(s))")

    # Send initial state
    await websocket.send(json.dumps({
        "type":    "status",
        "running": app["running"],
        "logging": app["logging"],
        "signal_ok": False,
        "port":    app["connected_port"],
    }))
    await websocket.send(json.dumps({
        "type":  "ports",
        "ports": _get_ports()
    }))
    await websocket.send(json.dumps({
        "type":     "config",
        "osc_ip":   cfg.OSC_IP,
        "osc_port": cfg.OSC_PORT,
        "baud":     cfg.BAUD_RATE,
    }))
    await websocket.send(json.dumps(_classifier_status_payload()))

    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
                await handle_command(msg, websocket)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "log",
                    "message": "Malformed command received.",
                    "level": "error"
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "log",
                    "message": f"Command error: {e}",
                    "level": "error"
                }))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)
        print(f"  UI disconnected ({len(ws_clients)} client(s))")

# ─────────────────────────────────────────────────────────
# HTTP server (serves ui.html)
# ─────────────────────────────────────────────────────────

def start_http_server():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)   # serve files relative to this script

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass    # suppress HTTP access log noise

    server = http.server.HTTPServer(("localhost", cfg.HTTP_PORT), QuietHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="http-server")
    t.start()
    return server

# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

async def main():
    global event_loop
    event_loop = asyncio.get_event_loop()

    print(f"\n{'─'*50}")
    print("  EEG → Resonite OSC Bridge")
    print(f"{'─'*50}")
    print(f"  WebSocket : ws://localhost:{cfg.WS_PORT}")
    print(f"  UI        : http://localhost:{cfg.HTTP_PORT}/ui.html")
    print(f"  OSC target: {cfg.OSC_IP}:{cfg.OSC_PORT}")
    if _classifier is None:
        print(f"  Classifier: disabled (CLASSIFIER_ENABLED = False)")
    elif _classifier.loaded:
        print(f"  Classifier: {_classifier.total_examples} examples loaded")
    else:
        print(f"  Classifier: demo mode (run generate_heatmaps.py first)")
    print(f"{'─'*50}\n")
    print("  Press Ctrl+C to stop.\n")

    start_http_server()

    # Small delay then open browser
    await asyncio.sleep(0.6)
    webbrowser.open(f"http://localhost:{cfg.HTTP_PORT}/ui.html")

    asyncio.create_task(dispatch_loop())

    async with websockets.serve(ws_handler, "localhost", cfg.WS_PORT):
        await asyncio.Future()   # run until interrupted


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Shutting down …")
        if app["running"]:
            app["stop_event"].set()
        stop_log()
        print("  Bye.\n")
