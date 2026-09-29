"""Real printed-text vocabulary observed on actual `ekg-757` pages
(`datasets/ekg-757/observed-page-samples.md`, two samples read 2026-09-05).
Centralized here so every synthetic panel reproduces the same fields real
panels carry, instead of each generation step inventing its own strings.
"""
from __future__ import annotations

import random

HEADER_TEXT = "MAC 400     V1.02"

# Gain varies by panel content in both real samples: limb leads (I/II/III,
# aVR/aVL/aVF) print 10mm/mV, precordial (V1-V6) print 5mm/mV. 1mm/mV was
# seen once, on an otherwise-limb panel -- treated as a rare outlier, not
# the default for any lead group.
GAIN_BY_LEAD_GROUP = {
    "limb": "10mm/mV",
    "precordial": "5mm/mV",
}
RARE_GAIN_OUTLIERS = ["1mm/mV"]
RARE_GAIN_PROBABILITY = 0.05

LIMB_LEADS = {"I", "II", "III", "aVR", "aVL", "aVF"}
PRECORDIAL_LEADS = {"V1", "V2", "V3", "V4", "V5", "V6"}

# "AI" was in this list until 2026-09-05, when checking its provenance
# turned up that it was never a confirmed reading -- observed-page-
# samples.md flagged it as an uncertain transcription that may just have
# been a blurry ADS or AC, not a genuine third value. ADS ("Anti Drift
# System", baseline-drift compensation) and AC (mains/notch filter
# status) are both independently attested in GE Marquette ECG
# documentation, though not confirmed against a primary-source printout.
TRAILING_CODES = ["ADS", "AC"]

RED_INK_SERIAL = "MDC72942181716008"
RED_INK_PATIENT_REF_PREFIX = "For "  # followed by a synthetic id + "-001"


def gain_for_leads(lead_names: list[str], rng: random.Random) -> str:
    if rng.random() < RARE_GAIN_PROBABILITY:
        return rng.choice(RARE_GAIN_OUTLIERS)
    is_limb = all(name in LIMB_LEADS for name in lead_names)
    return GAIN_BY_LEAD_GROUP["limb"] if is_limb else GAIN_BY_LEAD_GROUP["precordial"]


class FooterField:
    __slots__ = ("top", "bottom", "bottom_bold")

    def __init__(self, top: str, bottom: str, bottom_bold: bool = False):
        self.top = top
        self.bottom = bottom
        self.bottom_bold = bottom_bold


def footer_fields(lead_names: list[str], patient_ref: str, rng: random.Random, gain: str | None = None) -> list[FooterField]:
    """Real panels print two rows of fields, each column pairing a top
    (black, bold) field with a bottom (red) one -- "Man" over the patient
    ref, the speed over the device serial, the gain over "INNOQ" (bolded
    even though it's on the bottom row, unlike its column-mates). The
    fourth field (trailing code) has no bottom-row counterpart.

    `gain` can be passed in already-decided (e.g. because the caller also
    needs it to size a calibration step, and a second independent
    `gain_for_leads` roll here would risk disagreeing with that one)."""
    if gain is None:
        gain = gain_for_leads(lead_names, rng)
    trailing_code = rng.choice(TRAILING_CODES)
    return [
        FooterField("Man", f"{RED_INK_PATIENT_REF_PREFIX}{patient_ref}"),
        FooterField("25mm/s", RED_INK_SERIAL),
        FooterField(gain, "INNOQ", bottom_bold=True),
        FooterField(trailing_code, ""),
    ]


def synthetic_patient_ref(record_id: str) -> str:
    digits = "".join(ch for ch in record_id if ch.isdigit()) or "000000"
    return f"{digits[-7:].zfill(7)}-001"


# A second, MUTUALLY EXCLUSIVE header/footer style seen on some real
# panels (2026-09-05 reference photo -- re-examined: it showed two
# separate panels side by side, each with its own icon, not one panel
# with both styles at once). "even" REPLACES HEADER_TEXT with a
# "GE {date} {time}" line, and replaces footer_fields' top row (Man /
# speed / gain / code) with a notch-filter / bandpass / heart-rate top
# row -- same header height, same 2-line footer height as "odd", just
# different content in the same slots. Not additive.
MONTH_ABBREVIATIONS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NOTCH_FILTER_HZ = [50, 60]
BANDPASS_FILTERS = [(0.05, 150), (0.08, 150), (0.15, 100), (0.5, 40)]
BPM_RANGE = (60, 180)  # pediatric-skewed range, matches this project's own ekg-757 population


def ge_datetime_header(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    month = rng.choice(MONTH_ABBREVIATIONS)
    year = rng.randint(20, 26)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f"GE {day:02d}.{month}.{year:02d}  {hour:02d}:{minute:02d}"


# GE 12SL measurement/interpretation report block (2026-09-23, reference
# photo scan_0004.jpg): a real GE printout page prints this text-only
# summary alongside the lead panels, no waveform. Vocabulary kept in
# plausible clinical ranges, not exact reference-photo values repeated
# verbatim, so the synthetic corpus doesn't memorize one report.
SERIAL_PREFIX = "SCT"

MEASUREMENT_RANGES_MS = {
    "QRS": (70, 130), "QT": (330, 460), "QTC": (380, 470),
    "PR": (100, 220), "P": (60, 130), "RR": (500, 1100), "PP": (500, 1100),
}
AXIS_RANGE_DEG = (-30, 100)

INTERPRETATION_LINES = [
    "Normal sinus rhythm",
    "Normal sinus rhythm with sinus arrhythmia",
    "Sinus tachycardia",
    "Sinus bradycardia",
    "Rightward axis",
    "Leftward axis",
    "T wave abnormality, consider anterior ischemia",
    "T wave abnormality, consider inferior ischemia",
    "Nonspecific ST abnormality",
    "Abnormal ECG",
    "Normal ECG",
    "Borderline ECG",
]


def report_serial_number(rng: random.Random) -> str:
    return f"{SERIAL_PREFIX}{rng.randint(10**8, 10**9 - 1)}PA"


def report_measurements(rng: random.Random) -> dict[str, int]:
    return {name: rng.randint(*rng_ms) for name, rng_ms in MEASUREMENT_RANGES_MS.items()} | {
        "axis": rng.randint(*AXIS_RANGE_DEG),
    }


def report_interpretation(rng: random.Random) -> list[str]:
    return rng.sample(INTERPRETATION_LINES, k=rng.randint(1, 4))


def even_footer_fields(patient_ref: str, rng: random.Random) -> list[FooterField]:
    """The "even" style's footer top row -- notch filter / bandpass range
    / heart rate -- paired with the SAME bottom row as the "odd" style's
    first three columns (patient ref / device serial / INNOQ). No fourth
    column: this style has no trailing code observed."""
    notch = rng.choice(NOTCH_FILTER_HZ)
    low, high = rng.choice(BANDPASS_FILTERS)
    bpm = rng.randint(*BPM_RANGE)
    return [
        FooterField(f"{notch}Hz", f"{RED_INK_PATIENT_REF_PREFIX}{patient_ref}"),
        FooterField(f"{low:g}-{high:g}Hz", RED_INK_SERIAL),
        FooterField(f"{bpm}BPM", "INNOQ", bottom_bold=True),
    ]
