// ═══════════════════════════════════════════════════════════════
//  EEG Bridge — Arduino Sketch
//  For: Mindflex Duel TGAM + Arduino Uno → PC via USB Serial
//  Part of the Post-Symbolic EEG → Resonite OSC Bridge project
// ═══════════════════════════════════════════════════════════════
//
//  WIRING:
//  ┌──────────────────────────────────────────────────────────┐
//  │  TGAM board    →   Arduino                               │
//  │  T (TXD)       →   Pin 2  (SoftwareSerial RX)           │
//  │  R (RXD)       →   Pin 3  (SoftwareSerial TX, unused)   │
//  │  GND           →   GND    (shared ground — REQUIRED)     │
//  │  VCC / power   →   AAA batteries on TGAM board ONLY      │
//  │                    DO NOT power TGAM from Arduino 3.3V   │
//  └──────────────────────────────────────────────────────────┘
//
//  SERIAL OUTPUT (hardware serial, 115200 baud, USB to PC):
//  One CSV line per EEG packet (~1 per second):
//
//    signal,attention,meditation,delta,theta,lowAlpha,highAlpha,
//    lowBeta,highBeta,lowGamma,highGamma,blink
//
//  signal:     0–200  (0 = perfect contact, 200 = no signal)
//  attention:  0–100  (NeuroSky eSense)
//  meditation: 0–100  (NeuroSky eSense)
//  delta–highGamma: raw FFT power values (large integers, ~1Hz)
//  blink:      0–255  (event-based, 0 when no blink detected)
//
//  Lines beginning with '#' are status/debug messages.
//  The Python bridge ignores non-CSV lines automatically.
//
//  REQUIRED LIBRARIES (install via Arduino IDE Library Manager):
//  - Brain  by Eric Mika (kitschpatrol/Brain on GitHub)
//
//  IMPORTANT: The #define below MUST appear before any #include
//  to prevent SoftwareSerial buffer overflow at 9600 baud.
// ═══════════════════════════════════════════════════════════════

#define _SS_MAX_RX_BUFF 256   // must be before SoftwareSerial.h

#include <SoftwareSerial.h>
#include <Brain.h>

// ── Pin assignments ─────────────────────────────────────────────
const uint8_t BRAIN_RX_PIN  = 2;   // TGAM T → Arduino pin 2
const uint8_t BRAIN_TX_PIN  = 3;   // unused but required by SoftwareSerial
const uint8_t LED_SIGNAL_PIN = 13;  // built-in LED: ON = good signal

// ── Timing ──────────────────────────────────────────────────────
// Watchdog: warn if no valid packet received for this many ms
const unsigned long PACKET_TIMEOUT_MS = 5000;

// ── Objects ─────────────────────────────────────────────────────
SoftwareSerial brainSerial(BRAIN_RX_PIN, BRAIN_TX_PIN);
Brain brain(brainSerial);

// ── State ───────────────────────────────────────────────────────
unsigned long lastPacketMs    = 0;
bool          timeoutWarned   = false;
uint32_t      packetCount     = 0;

// ───────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);           // USB → PC (main data channel)
  brainSerial.begin(9600);        // SoftwareSerial → TGAM
  pinMode(LED_SIGNAL_PIN, OUTPUT);
  digitalWrite(LED_SIGNAL_PIN, LOW);

  // Status line (Python parser ignores '#' lines)
  Serial.println(F("# EEG Bridge ready — waiting for TGAM data"));
  Serial.println(F("# Format: signal,att,med,d,th,la,ha,lb,hb,lg,hg,blink"));

  lastPacketMs = millis();
}

// ───────────────────────────────────────────────────────────────
void loop() {
  if (brain.update()) {
    // ── Valid packet received ────────────────────────────────────
    lastPacketMs  = millis();
    timeoutWarned = false;
    packetCount++;

    // ── Signal quality LED ───────────────────────────────────────
    // ON = good signal (< 50), dim blink = weak, OFF = no contact
    uint8_t sig = brain.readSignalQuality();
    if (sig == 0) {
      digitalWrite(LED_SIGNAL_PIN, HIGH);          // perfect
    } else if (sig < 50) {
      digitalWrite(LED_SIGNAL_PIN, (millis() / 500) % 2);  // slow blink = acceptable
    } else {
      digitalWrite(LED_SIGNAL_PIN, LOW);           // no signal
    }

    // ── Output CSV line ──────────────────────────────────────────
    // Brain Library CSV (11 fields): signal,att,med,d,th,la,ha,lb,hb,lg,hg
    Serial.print(brain.readCSV());

    // Append blink strength as 12th field
    Serial.print(',');
    Serial.println(brain.readBlink());

  } else {
    // ── Check for parse errors ───────────────────────────────────
    const char* err = brain.readErrors();
    if (err && err[0] != '\0') {
      Serial.print(F("# ERR: "));
      Serial.println(err);
    }

    // ── Packet timeout warning ───────────────────────────────────
    if (!timeoutWarned && (millis() - lastPacketMs > PACKET_TIMEOUT_MS)) {
      Serial.print(F("# WARN: No data for "));
      Serial.print(PACKET_TIMEOUT_MS / 1000);
      Serial.println(F("s — check TGAM wiring and battery"));
      timeoutWarned = true;
    }
  }
}
