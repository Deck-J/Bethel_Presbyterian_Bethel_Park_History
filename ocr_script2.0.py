#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bethel OCR / ICR Pipeline
Version: 2.21 (Adaptive Quadrants)

Features:
- Vision OCR via OpenAI (gpt-4o-mini / gpt-4o or configured models)
- Automatic layout + density detection per page
- For ledger / super-dense pages:
    - Adaptive text-region detection (ink clusters)
    - 1 Vision API call per text region (N smaller calls total)
    - Merge region JSON into a single page JSON
- For default pages:
    - Single Vision API call using tiled images (2x3)
- LLM: OCR + light page metadata only (JSON inside ---BEGIN_JSON--- markers)
- Local post-processing:
    - People Sweeper (names, honorifics including Mr/Mrs/Miss, roles, military)
    - Place Sweeper (using dictionary_hints + heuristics)
    - Event detection (birth, death, marriage, baptism, death/military-ish)
- Dictionary hints from dictionary_hints.json:
    { "names": [], "places": [], "church_terms": [], "military_keywords": [] }
- Master JSONL rebuild: ocr_all_pages.jsonl
"""

import os
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
import warnings           # ← system module, not part of PIL
from PIL import Image     # ← only import Image from PIL

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 300000000
from openai import OpenAI
import numpy as np
import cv2

# -------------------------
# Logging
# -------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Paths & constants
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"
OUTPUT_DIR = BASE_DIR / "ocr_output"
OUTPUT_DIR.mkdir(exist_ok=True)
MASTER_JSONL = BASE_DIR / "ocr_all_pages.jsonl"

HINTS_FILE = BASE_DIR / "dictionary_hints.json"
LAYOUT_OVERRIDES_PATH = BASE_DIR / "page_layout_overrides.json"

# Models
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
HEAVY_MODEL = os.getenv("HEAVY_MODEL", "gpt-4o")  # heavy = 4o only

# Retries
MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "2"))
RETRY_SLEEP_SECONDS = float(os.getenv("OCR_RETRY_SLEEP", "4"))

# Env page list: e.g. PAGE_LIST="11,18,39"
PAGE_LIST_ENV = os.getenv("PAGE_LIST", "").strip()

# Version tag used in JSON
TRANSCRIPTION_VERSION = "openai-gpt-4o-vision-v2.21-adaptive-quadrants"

# Layout tile configurations for the "single-call" path
LAYOUT_TILE_CONFIG = {
    "default": (2, 3),   # rows, cols
}

# Optional dynamic downscale: max total pixels; if exceeded, we downscale
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "90000000"))

# -------------------------
# Built-in hints (fallbacks if JSON missing)
# -------------------------

DEFAULT_SURNAMES = [
    "Marshall", "Miller", "McMillan", "Millar", "Clark", "Calhoun",
    "Fleming", "Sample", "Dimmon", "Dimmond", "Fairchild",
    "Elliott", "Elliot", "Hill", "Kelly", "Smith", "Jones",
    "Hutchison", "Hutchinson", "McFarland", "Anderson",
]

DEFAULT_PLACES = [
    "Bethel", "Bethel Church", "Bethel Cemetery", "Pittsburgh",
    "Allegheny", "Minisink", "Deckertown", "Petersburgh",
    "Wilderness", "Antietam", "Gettysburg", "Virginia",
    "Augusta Stone Church", "Westmoreland", "Washington County",
]

HONORIFICS_RELIGIOUS = [
    "Rev.", "Revd.", "Pastor", "Elder", "Ruling Elder",
    "Deacon", "Moderator", "Clerk", "Trustee"
]

HONORIFICS_CIVIC = [
    "Dr.", "Esq.", "Hon.", "Judge"
]

HONORIFICS_MILITARY = [
    "Col.", "Colonel", "Capt.", "Captain", "Lieut.", "Lt.",
    "Sgt.", "Sergeant", "Corp.", "Corporal", "Pvt.", "Private",
    "Gen.", "General"
]

# Generic honorifics explicitly included
HONORIFICS_GENERIC = [
    "Mr.", "Mrs.", "Miss"
]

DEFAULT_MILITARY_KEYWORDS = [
    "Civil War", "war", "regiment", "company", "Co.",
    "killed", "wounded", "died", "battle", "camp",
    "Petersburgh", "Petersburg", "Wilderness", "Gettysburg",
    "Antietam", "Chancellorsville"
]

ALL_HONORIFICS = (
    HONORIFICS_RELIGIOUS
    + HONORIFICS_CIVIC
    + HONORIFICS_MILITARY
    + HONORIFICS_GENERIC
)

# -------------------------
# Load dictionary_hints.json
# -------------------------

def load_hints() -> Dict[str, Any]:
    if not HINTS_FILE.exists():
        logger.warning("[HINTS] No dictionary_hints.json found; using built-in defaults only")
        return {}

    try:
        with HINTS_FILE.open("r", encoding="utf-8") as f:
            hints = json.load(f)

        logger.info(
            "[HINTS] Loaded dictionary_hints.json → "
            f"{len(hints.get('names', []))} names, "
            f"{len(hints.get('places', []))} places, "
            f"{len(hints.get('church_terms', []))} church terms, "
            f"{len(hints.get('military_keywords', []))} military keywords"
        )
        return hints
    except Exception as e:
        logger.error(f"[HINTS] Failed to load dictionary_hints.json: {e}")
        return {}


HINTS = load_hints()


def combined_list(base: List[str], from_hints: Optional[List[str]]) -> List[str]:
    if not from_hints:
        return sorted(set(base))
    return sorted(set(base + from_hints))


COMBINED_SURNAMES = combined_list(DEFAULT_SURNAMES, HINTS.get("names"))
COMBINED_PLACES = combined_list(DEFAULT_PLACES, HINTS.get("places"))
COMBINED_CHURCH_TERMS = combined_list([], HINTS.get("church_terms"))
COMBINED_MILITARY = combined_list(DEFAULT_MILITARY_KEYWORDS, HINTS.get("military_keywords"))

# -------------------------
# Page list / layout
# -------------------------
def parse_page_list() -> List[int]:
    """
    Supports:
      PAGE_LIST="1,2,3"
      PAGE_LIST="1-5"
      PAGE_LIST="1-5,7,9-12"
    """
    if PAGE_LIST_ENV:
        out: List[int] = []
        for token in PAGE_LIST_ENV.split(","):
            token = token.strip()
            if not token:
                continue

            # Range support: e.g. "3-10"
            if "-" in token:
                try:
                    start, end = token.split("-", 1)
                    start_i = int(start)
                    end_i = int(end)
                    if start_i <= end_i:
                        out.extend(range(start_i, end_i + 1))
                    else:
                        out.extend(range(end_i, start_i + 1))
                except ValueError:
                    logger.warning(f"[PAGE_LIST] Skipping invalid range: {token}")
                continue

            # Single page
            try:
                out.append(int(token))
            except ValueError:
                logger.warning(f"[PAGE_LIST] Skipping invalid value: {token}")

        return sorted(set(out))

    # No PAGE_LIST — auto-detect numbered files in `pages/`
    pages: List[int] = []
    for p in PAGES_DIR.glob("*.png"):
        try:
            pages.append(int(p.stem))
        except ValueError:
            continue
    return sorted(pages)

def load_layout_overrides() -> Dict[int, str]:
    if not LAYOUT_OVERRIDES_PATH.exists():
        return {}
    try:
        with LAYOUT_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except ValueError:
                continue
        return out
    except Exception as e:
        logger.warning(f"[LAYOUT] Failed to read overrides: {e}")
        return {}


def infer_layout(page: int, overrides: Dict[int, str]) -> str:
    if page in overrides:
        return overrides[page]

    candidates = [
        PAGES_DIR / f"{page}.png",
        PAGES_DIR / f"page-{page}.png",
    ]
    img_path = None
    for c in candidates:
        if c.exists():
            img_path = c
            break
    if not img_path:
        return "default"

    try:
        with Image.open(img_path) as im:
            w, h = im.size
        area = w * h
        aspect = w / max(1, h)
        if area > 60_000_000:
            return "super_dense"
        if aspect > 1.6:
            return "ledger"
        return "default"
    except Exception:
        return "default"


# -------------------------
# Image helpers
# -------------------------

def tile_image(img: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    w, h = img.size
    tw = w // cols
    th = h // rows
    tiles: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * tw
            upper = r * th
            right = w if c == cols - 1 else (c + 1) * tw
            lower = h if r == rows - 1 else (r + 1) * th
            tiles.append(img.crop((left, upper, right, lower)))
    return tiles


Box = Tuple[int, int, int, int]  # (left, top, right, bottom)


def _merge_overlapping_boxes(
    boxes: List[Box], max_gap_px: int = 20
) -> List[Box]:
    """
    Simple merge of overlapping or very close boxes.
    Used to combine nearby text blobs into larger regions.
    """
    if not boxes:
        return []

    # Sort by top, then left
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[Box] = []

    def overlap_or_close(a: Box, b: Box) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        # Expand a by max_gap_px in all directions
        ax1e = ax1 - max_gap_px
        ay1e = ay1 - max_gap_px
        ax2e = ax2 + max_gap_px
        ay2e = ay2 + max_gap_px

        # Intersection
        inter_x1 = max(ax1e, bx1)
        inter_y1 = max(ay1e, by1)
        inter_x2 = min(ax2e, bx2)
        inter_y2 = min(ay2e, by2)
        return inter_x1 <= inter_x2 and inter_y1 <= inter_y2

    current = boxes[0]
    for b in boxes[1:]:
        if overlap_or_close(current, b):
            x1 = min(current[0], b[0])
            y1 = min(current[1], b[1])
            x2 = max(current[2], b[2])
            y2 = max(current[3], b[3])
            current = (x1, y1, x2, y2)
        else:
            merged.append(current)
            current = b
    merged.append(current)
    return merged


def find_text_regions(
    pil_img: Image.Image,
    max_regions: int = 6,
    min_area_ratio: float = 0.003,
    pad: int = 20,
) -> List[Box]:
    """
    Detect text-heavy regions on the page and return bounding boxes
    in (left, top, right, bottom) format, suitable for PIL.crop.

    - Works on full pages with blank margins / blank pages.
    - Scales kernel sizes based on image dimensions.
    - Merges overlapping / very close boxes.
    """
    rgb = pil_img.convert("RGB")
    np_img = np.array(rgb)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]

    # Binarize (Otsu) then invert so ink is white
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = 255 - th

    # Morphological operations to join strokes into blobs
    # Slightly smaller kernel than before to avoid merging entire page
    kw = max(15, w // 80)   # was w // 50
    kh = max(5, h // 250)   # was h // 200
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

    morphed = cv2.dilate(bin_img, kernel, iterations=1)
    morphed = cv2.erode(morphed, kernel, iterations=1)

    contours, _ = cv2.findContours(
        morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = min_area_ratio * (w * h)
    boxes: List[Box] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        # Do NOT discard very large regions; worst case we OCR the whole page.
        boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return []

    boxes = _merge_overlapping_boxes(boxes, max_gap_px=max(w, h) // 100)

    boxes.sort(key=lambda b: (b[1], b[0]))

    padded: List[Box] = []
    for (x1, y1, x2, y2) in boxes[:max_regions]:
        left = max(0, x1 - pad)
        top = max(0, y1 - pad)
        right = min(w, x2 + pad)
        bottom = min(h, y2 + pad)
        padded.append((left, top, right, bottom))

    logger.info(f"[ADAPTIVE] Detected {len(padded)} text region(s)")
    return padded


def maybe_downscale(im: Image.Image) -> Tuple[Image.Image, bool]:
    w, h = im.size
    area = w * h
    if area <= MAX_IMAGE_PIXELS:
        return im, False
    scale = (MAX_IMAGE_PIXELS / float(area)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.info(
        f"[DOWNSCALE] Original size {w}x{h} ({area} px) → {new_w}x{new_h} (~{new_w*new_h} px)"
    )
    return im.resize((new_w, new_h), Image.LANCZOS), True


def image_to_data_url(im: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# -------------------------
# LLM prompts (OCR + light structure)
# -------------------------

def build_system_prompt() -> str:
    surnames_str = ", ".join(COMBINED_SURNAMES[:80])
    places_str = ", ".join(COMBINED_PLACES[:80])
    church_terms_str = ", ".join(COMBINED_CHURCH_TERMS[:80])
    military_str = ", ".join(COMBINED_MILITARY[:80])
    honorifics_str = ", ".join(ALL_HONORIFICS)

    return f"""
You are an expert historical paleographer and archivist working on 18th–19th century
Presbyterian church records (Bethel and related congregations).

You will receive images (or tiles) from ONE page or a clearly delimited quadrant
of a page. Your tasks:

1) Perform high-fidelity OCR / ICR transcription.
2) Provide light structural metadata about that page or quadrant.
3) Return a SINGLE JSON object between markers:

   ---BEGIN_JSON---
   {{ ... }}
   ---END_JSON---

Any text outside the markers will be ignored by the pipeline.

==================
JSON OBJECT SCHEMA
==================

The JSON object MUST include at least:

{{
  "page_number": <integer>,          // physical page number
  "quadrant_label": <string or null>,// e.g. "Q1", "Q2" for quadrant calls
  "page_title": <string or null>,
  "page_type": "narrative" | "ledger" | "mixed",
  "record_type": <string>,           // e.g. "session minutes", "marriage register",
                                     //      "family visitation ledger",
                                     //      "financial ledger", "cemetery record", etc.
  "page_date_hint": <string or null>,// best-guess year or date like "Feb 10 1855"
  "transcription_method": "{TRANSCRIPTION_VERSION}",
  "transcription_quality": "high" | "medium" | "low",

  "content": <string>,               // cleaned, human-readable transcription
  "raw_text": <string>               // closer to verbatim; may match content
}}

Optional:
- "places": [ <string>, ... ]        // distinct place names you can see
- "notes": <string or null>

The Python pipeline will later perform its own local analysis (people sweeper,
events, honorifics, military, etc.), so you do NOT need to fully structure persons.

=========================
TRANSCRIPTION GUIDELINES
=========================

- Be as faithful as possible to the handwriting / print.
- Do NOT invent people, places, or events that are not present.
- Preserve line breaks where they help readability, but you may unwrap
  multi-column registers into a more linear reading order.
- Do NOT modernize spelling; keep historical forms (e.g. "betwixt", "ye").
- You may expand extremely obvious abbreviations when confident, but keep
  the original form somewhere nearby (e.g. "Presb." → "Presbyterian (Presb.)").

=========================
DICTIONARY / HINTS (SOFT)
=========================

Use these as hints to reduce hallucinations and help with ambiguous letters:

- Common surnames in this corpus (not exhaustive):
  {surnames_str}

- Important places and churches:
  {places_str}

- Additional church terms / organizations:
  {church_terms_str}

- Military / Civil War vocabulary:
  {military_str}

- Honorifics / titles (religious, civic, military, generic):
  {honorifics_str}

Do NOT force text to match the hints; they are only suggestions.

=========================
QUALITY FLAG
=========================

transcription_quality:
- "high"   → clear text, very confident
- "medium" → mostly legible with some uncertain phrases
- "low"    → heavily damaged, many gaps or uncertain words

=========================
CONSTRAINTS
=========================

- The ONLY JSON must be between:

  ---BEGIN_JSON---
  {{ ... }}
  ---END_JSON---

- No comments inside JSON, no trailing commas.
- No images or binary data in JSON.
    """.strip()


def build_user_prompt(page_number: int, layout_type: str, quadrant_label: Optional[str]) -> str:
    q_part = f"quadrant {quadrant_label} of " if quadrant_label else ""
    return f"""
You are transcribing {q_part}page {page_number} of a bound church record volume.

LAYOUT:
- Layout classification for this page: "{layout_type}".
- The images you see are all from THIS same logical unit ({q_part}page {page_number}).
- Combine all image tiles you receive into one coherent transcription for this unit.

TASK:
1) Carefully read ALL tiles.
2) Produce an accurate transcription and light structural metadata.
3) Then output EXACTLY ONE JSON OBJECT between:

   ---BEGIN_JSON---
   {{ ... }}
   ---END_JSON---

The JSON MUST follow the schema from the system prompt, including:
- page_number, page_title, page_type, record_type, page_date_hint
- content, raw_text
- transcription_method, transcription_quality

Also:
- For a quadrant call, set "quadrant_label" = "{quadrant_label}".
- For a whole-page call, set "quadrant_label" = null.

You may optionally include a simple "places" array of place-name strings.

Do NOT put any non-JSON text between the markers.
    """.strip()


# -------------------------
# OpenAI client
# -------------------------

def get_client() -> OpenAI:
    load_dotenv()
    client = OpenAI()
    if not client.api_key:
        raise RuntimeError("You must set OPENAI_API_KEY environment variable")
    return client


# -------------------------
# LLM call + JSON extraction (for a set of tiles)
# -------------------------

def extract_json_from_markers(text: str, page_number: int, quadrant_label: Optional[str]) -> Dict[str, Any]:
    match = re.search(r"---BEGIN_JSON---(.*?)---END_JSON---", text, re.DOTALL)
    raw_json = None

    if match:
        raw_json = match.group(1).strip()
    else:
        candidate = text.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            raw_json = candidate

    if raw_json is None:
        raise ValueError("No JSON block found in model output")

    data = json.loads(raw_json)

    data.setdefault("page_number", page_number)
    data.setdefault("quadrant_label", quadrant_label)
    data.setdefault("transcription_method", TRANSCRIPTION_VERSION)
    data.setdefault("transcription_quality", "medium")

    if "content" not in data and "raw_text" in data:
        data["content"] = data["raw_text"]
    if "raw_text" not in data and "content" in data:
        data["raw_text"] = data["content"]

    data.setdefault("content", "")
    data.setdefault("raw_text", data.get("content", ""))

    for fld in ["page_title", "page_type", "record_type", "page_date_hint"]:
        data.setdefault(fld, None)

    # Normalize any simple places array into list[str]
    places = data.get("places")
    norm_places: List[str] = []
    if isinstance(places, list):
        for p in places:
            if isinstance(p, str):
                norm_places.append(p.strip())
            elif isinstance(p, dict) and "name" in p:
                norm_places.append(str(p["name"]).strip())
    data["places"] = norm_places

    return data


def call_vision_for_tiles(
    client: OpenAI,
    page_number: int,
    tiles: List[Image.Image],
    layout_type: str,
    quadrant_label: Optional[str],
    model: str,
) -> Dict[str, Any]:
    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(page_number, layout_type, quadrant_label)

    message_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for tile in tiles:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(tile),
                "detail": "high",
            },
        })

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        label = quadrant_label or "whole-page"
        logger.info(
            f"Page {page_number} {label}: LLM attempt {attempt}/{MAX_RETRIES}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": message_content},
                ],
            )

            msg = resp.choices[0].message
            if isinstance(msg.content, str):
                assistant_text = msg.content
            else:
                parts_text: List[str] = []
                for part in msg.content:
                    if getattr(part, "type", None) == "text":
                        parts_text.append(part.text)
                assistant_text = "".join(parts_text)

            assistant_text = assistant_text.strip()
            if not assistant_text:
                raise ValueError("Empty assistant_text from model")

            data = extract_json_from_markers(assistant_text, page_number, quadrant_label)
            return data

        except Exception as e:
            last_error = e
            logger.warning(
                f"Page {page_number} {quadrant_label or 'whole'}: attempt {attempt} failed: {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"Failed to process page {page_number} ({quadrant_label or 'whole'}) after {MAX_RETRIES} attempts: {last_error}"
    )


# -------------------------
# Local People / Places / Events Sweeper
# -------------------------

DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*[,/]\s*\d{2,4})?\b"
)

YEAR_RE = re.compile(r"\b(17|18|19)\d{2}\b")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).replace("\r", "")


def clean_do_tokens(tokens: List[str]) -> List[str]:
    """
    Remove stray 'do' / 'Do' tokens that show up as OCR noise in names like
    'Do Do Marshall', 'Do James Miller', etc.

    We ONLY strip tokens that are exactly 'do' (case-insensitive) after
    trimming basic punctuation. Real names like 'Doane' or 'Dorn' are untouched.
    """
    cleaned: List[str] = []
    for tok in tokens:
        t = tok.lower().strip(".,;:")
        if t == "do":
            continue
        cleaned.append(tok)
    return cleaned


def sweep_places(text: str, initial_places: List[str]) -> List[Dict[str, Any]]:
    """Detect places from hints + simple capitalized patterns."""
    found: Dict[str, str] = {}

    lowered = text.lower()

    # 1) Dictionary-based matches
    for place in COMBINED_PLACES:
        if not place:
            continue
        p_lower = place.lower()
        if p_lower in lowered:
            found[place] = ""

    # 2) Use any initial LLM places as seeds
    for p in initial_places:
        if p and p not in found:
            found[p] = ""

    # 3) Very light heuristic: words ending in "Co.", "County", "Township"
    extra_places = re.findall(
        r"\b([A-Z][a-zA-Z]+(?:\s+(?:Co\.|County|Township|Borough|City)))\b", text
    )
    for p in extra_places:
        if p not in found:
            found[p] = ""

    return [{"name": name, "notes": notes or None} for name, notes in sorted(found.items())]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)
    return [p.strip() for p in parts if p and p.strip()]


def detect_honorific_and_role(tokens: List[str]) -> Tuple[Optional[str], Optional[str]]:
    honorific = None
    role = None

    if tokens:
        first = tokens[0]
        if first in ALL_HONORIFICS:
            honorific = first

    joined = " ".join(tokens)
    lowered = joined.lower()

    if "pastor" in lowered:
        role = "Pastor"
    elif "ruling elder" in lowered:
        role = "Ruling Elder"
    elif "elder" in lowered:
        role = "Elder"
    elif "deacon" in lowered:
        role = "Deacon"
    elif "trustee" in lowered:
        role = "Trustee"
    elif "clerk" in lowered:
        role = "Clerk"
    elif "moderator" in lowered:
        role = "Moderator"

    return honorific, role


def detect_military(joined: str) -> Tuple[bool, Optional[str]]:
    lower = joined.lower()
    is_military = any(k.lower() in lower for k in COMBINED_MILITARY)
    rank = None
    for tok in HONORIFICS_MILITARY:
        if tok.lower().strip(".") in lower:
            rank = tok.replace(".", "")
            is_military = True
            break
    return is_military, rank


def sweep_people(text: str) -> List[Dict[str, Any]]:
    """
    Local people sweeper: uses honorifics + dictionary surnames + capitalized patterns.
    Includes generic Mr., Mrs., Miss as honorifics.
    Also strips stray 'do' / 'Do' tokens that are OCR junk.
    """
    text = normalize_whitespace(text)
    people: Dict[str, Dict[str, Any]] = {}

    # Pattern 1: Honorific + Name(s)
    honorific_pattern_union = "|".join(re.escape(h) for h in ALL_HONORIFICS)
    pattern = rf"\b(?:{honorific_pattern_union})\s+[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+)*"

    for m in re.finditer(pattern, text):
        full = m.group(0).strip()
        tokens = full.split()
        tokens = clean_do_tokens(tokens)  # remove stray 'do' tokens
        if not tokens:
            continue

        honorific, role = detect_honorific_and_role(tokens)
        name_tokens = tokens[1:] if honorific else tokens
        name_tokens = clean_do_tokens(name_tokens)  # just in case 'do' was in name portion too

        if not name_tokens:
            continue

        given = name_tokens[0] if name_tokens else None
        surname = name_tokens[-1] if len(name_tokens) > 1 else None

        joined = " ".join(tokens)
        is_military, rank = detect_military(joined)

        cleaned_full = " ".join(tokens)
        key = cleaned_full

        if key not in people:
            people[key] = {
                "full_name": cleaned_full,
                "given_name": given,
                "surname": surname,
                "church_role": role,
                "honorific": honorific,
                "is_military": is_military,
                "military_rank": rank,
                "notes": None,
                "confidence": 0.85,
            }

    # Pattern 2: Plain "Firstname Lastname" where Lastname is in surnames list.
    # We pre-filter 'do' tokens so they never make bogus names.
    surname_set = {s.lower() for s in COMBINED_SURNAMES}

    raw_tokens = text.split()
    filtered_tokens = [t for t in raw_tokens if t.lower().strip(".,;:") != "do"]

    for i in range(len(filtered_tokens) - 1):
        t1 = filtered_tokens[i]
        t2 = filtered_tokens[i + 1]
        if not t1 or not t2:
            continue
        if not t1[0].isupper() or not t2[0].isupper():
            continue

        clean1 = re.sub(r"[^A-Za-z.'-]", "", t1)
        clean2 = re.sub(r"[^A-Za-z.'-]", "", t2)
        if not clean1 or not clean2:
            continue

        if clean2.lower().strip(".") in surname_set:
            full = f"{clean1} {clean2}"
            if full not in people:
                people[full] = {
                    "full_name": full,
                    "given_name": clean1,
                    "surname": clean2,
                    "church_role": None,
                    "honorific": None,
                    "is_military": False,
                    "military_rank": None,
                    "notes": None,
                    "confidence": 0.6,
                }

    return list(people.values())


def find_people_in_text(snippet: str, people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach people to an event by simple substring / surname presence."""
    snippet_lower = snippet.lower()
    participants: List[Dict[str, Any]] = []

    for p in people:
        full = p.get("full_name") or ""
        surname = p.get("surname") or ""
        if full and full.lower() in snippet_lower:
            participants.append(p)
            continue
        if surname and surname.lower() in snippet_lower:
            participants.append(p)
            continue

    unique: Dict[str, Dict[str, Any]] = {}
    for p in participants:
        key = p.get("full_name") or repr(p)
        unique[key] = p
    return list(unique.values())


def sweep_events(
    text: str,
    people: List[Dict[str, Any]],
    places: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect simple birth/death/marriage/baptism/military-ish events from text."""
    events: List[Dict[str, Any]] = []
    sentences = split_sentences(text)
    place_names = [p["name"] for p in places]

    for sent in sentences:
        lowered = sent.lower()
        event_type: Optional[str] = None

        if any(word in lowered for word in ["born", "birth"]):
            event_type = "birth"
        elif any(word in lowered for word in ["baptized", "baptised", "baptism"]):
            event_type = "baptism"
        elif any(word in lowered for word in ["married", "marriage"]):
            event_type = "marriage"
        elif any(word in lowered for word in ["died", "dead", "deceased", "decd", "killed"]):
            event_type = "death"
        elif any(word in lowered for word in ["regiment", "company", "killed at", "battle of"]):
            event_type = "other"

        if not event_type:
            continue

        date_match = DATE_RE.search(sent)
        date = date_match.group(0) if date_match else None
        if not date:
            year_match = YEAR_RE.search(sent)
            if year_match:
                date = year_match.group(0)

        place_for_event = None
        for pl in place_names:
            if pl and pl.lower() in lowered:
                place_for_event = pl
                break

        participants = find_people_in_text(sent, people)

        normalized_summary = f"{event_type.capitalize()} event mentioned: {sent[:180].strip()}"
        if len(sent) > 180:
            normalized_summary += " …"

        events.append({
            "event_type": event_type,
            "date": date,
            "place": place_for_event,
            "raw_entry": sent,
            "normalized_summary": normalized_summary,
            "people": participants,
        })

    return events


def apply_local_sweeper(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach places[], people[], events[] to LLM JSON.
    Uses:
    - sweep_places (with hints)
    - sweep_people (honorifics + names, with 'do' cleanup)
    - sweep_events (birth/death/marriage/baptism/military-ish)
    """
    text = data.get("content") or data.get("raw_text") or ""
    text = text if isinstance(text, str) else str(text)

    # Places: use any LLM places as seeds
    initial_places = data.get("places") or []
    if not isinstance(initial_places, list):
        initial_places = []
    initial_places_str: List[str] = []
    for p in initial_places:
        if isinstance(p, str):
            initial_places_str.append(p)
        elif isinstance(p, dict) and "name" in p:
            initial_places_str.append(str(p["name"]))

    places_struct = sweep_places(text, initial_places_str)
    people_struct = sweep_people(text)
    events_struct = sweep_events(text, people_struct, places_struct)

    data["places"] = places_struct
    data["people"] = people_struct
    data["events"] = events_struct
    return data

# -------------------------
# Output helpers
# -------------------------

def save_page_json(page_number: int, data: Dict[str, Any]) -> Path:
    out_path = OUTPUT_DIR / f"page-{page_number}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def save_raw_broken(page_number: int, error: Exception) -> Path:
    out_path = OUTPUT_DIR / f"page-{page_number}_raw_broken.json"
    payload = {
        "page_number": page_number,
        "error": str(error),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def rebuild_master_jsonl() -> None:
    logger.info(f"[REBUILD] Rebuilding master JSONL at {MASTER_JSONL}")
    page_files: List[Path] = []

    for p in sorted(OUTPUT_DIR.glob("page-*.json")):
        if p.name.endswith("_raw_broken.json"):
            logger.debug(f"[REBUILD] Skipping raw/broken JSON: {p.name}")
            continue
        page_files.append(p)

    count = 0
    with MASTER_JSONL.open("w", encoding="utf-8") as out:
        for p in page_files:
            try:
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning(f"[REBUILD] Failed to include {p.name}: {e}")

    logger.info(f"[REBUILD] Master JSONL rebuilt with {count} pages at {MASTER_JSONL}")


# -------------------------
# Page-level OCR orchestration
# -------------------------

def ocr_page(
    client: OpenAI,
    page: int,
    im: Image.Image,
    layout_type: str,
) -> Dict[str, Any]:
    """
    For default layout: one call with tiled page (2x3).
    For ledger/super_dense: adaptive text-region calls, then merge content.
    """

    w, h = im.size
    area = w * h

    # Adaptive path for dense / ledger pages
    if layout_type in ("super_dense", "ledger"):
        logger.info(
            f"Page {page}: ADAPTIVE-REGION mode (layout={layout_type}), size={w}x{h} ({area} px)"
        )

        regions = find_text_regions(im, max_regions=6, min_area_ratio=0.003, pad=20)

        if regions:
            model = HEAVY_MODEL  # heavy for dense
            region_results: List[Dict[str, Any]] = []

            for idx, box in enumerate(regions, start=1):
                left, top, right, bottom = box
                q_label = f"Q{idx}"
                logger.info(
                    f"Page {page}: processing region {q_label} box={box}"
                )
                region_img = im.crop(box)
                data_q = call_vision_for_tiles(
                    client=client,
                    page_number=page,
                    tiles=[region_img],
                    layout_type=layout_type,
                    quadrant_label=q_label,
                    model=model,
                )
                region_results.append(data_q)

            # Merge regions: use first region as base for metadata
            base = region_results[0].copy()

            merged_content_parts: List[str] = []
            merged_raw_parts: List[str] = []

            for q in region_results:
                c = q.get("content") or ""
                r = q.get("raw_text") or c
                if c:
                    merged_content_parts.append(c.strip())
                if r:
                    merged_raw_parts.append(r.strip())

            full_content = "\n\n".join(merged_content_parts)
            full_raw = "\n\n".join(merged_raw_parts)

            base["content"] = full_content
            base["raw_text"] = full_raw
            base["quadrant_label"] = None  # final page is not a single region

            # Reset places so local sweeper starts clean
            base["places"] = []

            return base
        else:
            logger.info(
                f"Page {page}: no text regions detected; falling back to SINGLE-CALL tiling."
            )

    # Default single-call path with 2x3 tiling
    rows, cols = LAYOUT_TILE_CONFIG["default"]
    tiles = tile_image(im, rows, cols)
    logger.info(
        f"Page {page}: SINGLE-CALL mode layout={layout_type}, tiling={rows}x{cols}, size={w}x{h} ({area} px)"
    )
    model = HEAVY_MODEL if layout_type in ("ledger", "super_dense") else BASELINE_MODEL
    data = call_vision_for_tiles(
        client=client,
        page_number=page,
        tiles=tiles,
        layout_type=layout_type,
        quadrant_label=None,
        model=model,
    )
    return data


# -------------------------
# Main
# -------------------------

def main() -> None:
    client = get_client()
    pages = parse_page_list()

    if not pages:
        logger.error("[MAIN] No pages to process. Set PAGE_LIST or add PNGs to pages/")
        return

    layout_overrides = load_layout_overrides()

    logger.info(f"Processing pages: {pages}")
    logger.info(f"Using baseline model: {BASELINE_MODEL}, heavy model: {HEAVY_MODEL}")
    logger.info(f"dictionary_hints.json present: {HINTS_FILE.exists()}")

    for page in pages:
        try:
            logger.info(f"=== Page {page} ===")

            img_path_candidates = [
                PAGES_DIR / f"{page}.png",
                PAGES_DIR / f"page-{page}.png",
                PAGES_DIR / f"{page}.jpg",
                PAGES_DIR / f"page-{page}.jpg",
                PAGES_DIR / f"{page}.jpeg",
                PAGES_DIR / f"page-{page}.jpeg",
            ]
            img_path = None
            for cand in img_path_candidates:
                if cand.exists():
                    img_path = cand
                    break
            if not img_path:
                raise FileNotFoundError(f"No image found for page {page} in {PAGES_DIR}")

            with Image.open(img_path) as im:
                im.load()
                orig_w, orig_h = im.size
                layout_type = infer_layout(page, layout_overrides)
                logger.info(
                    f"Page {page} → layout={layout_type}, size={orig_w}x{orig_h} ({orig_w*orig_h} px)"
                )

                im2, downscaled = maybe_downscale(im)
                if downscaled:
                    w2, h2 = im2.size
                    logger.info(
                        f"Page {page}: using downscaled image {w2}x{h2} ({w2*h2} px) for OCR"
                    )

                data = ocr_page(client, page, im2, layout_type)

            # Local sweep: people / places / events
            data = apply_local_sweeper(data)

            save_page_json(page, data)
            logger.info(f"Page {page}: saved structured JSON")

        except Exception as e:
            save_raw_broken(page, e)
            logger.error(f"Page {page}: FAILED → {e}")

    rebuild_master_jsonl()


if __name__ == "__main__":
    main()
