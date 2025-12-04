#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bethel OCR / ICR Pipeline
Version: 2.36 (Adaptive + Quadrants + FIFO API Queue + Events w/ People & Places)

Features:
- Vision OCR via OpenAI (gpt-4o-mini / gpt-4o)
- Automatic layout density detection
- Adaptive text-region detection via OpenCV
- For super-dense / ledger pages:
    * Adaptive region mode
    * If only one large region → fixed 2x2 quadrants
    * Quadrants/regions processed in parallel (inner ThreadPoolExecutor)
- For default pages:
    * Single Vision call on 2x3 tiles
- Strong system/user prompts:
    * Explicitly public-domain 18th–19th century records
    * Do NOT refuse / no safety warnings
- Tolerant JSON extraction:
    * Markers, bare JSON, first {...}
    * If none → wrap raw text into minimal JSON stub
- Local post-processing:
    * Places / people / events sweeper (lightweight)
    * Events linked to people (full name / surname) and places, with dates/years
    * Ambiguous names (e.g. Elizabeth) down-ranked as places if corpus shows
      they overwhelmingly occur as first names, not towns
- Page-level parallelism with ThreadPoolExecutor
- Refusal / safety boilerplate scrubber on content/raw_text
- Global FIFO queue for ALL OpenAI calls:
    * Exactly one in-flight API call at a time
    * Strictly ordered across all pages/quadrants
    * Exponential backoff for rate limits / transient failures
"""

import os
import re
import json
import time
import base64
import logging
import warnings
import threading

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread, Event

from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import numpy as np
import cv2

# Suppress decompression bomb warnings but raise pixel limit
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 300_000_000

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

BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
HEAVY_MODEL = os.getenv("HEAVY_MODEL", "gpt-4o")

PAGE_LIST_ENV = os.getenv("PAGE_LIST", "").strip()

MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "2"))
RETRY_SLEEP_SECONDS = float(os.getenv("OCR_RETRY_SLEEP", "4"))
MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "4"))

MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "90000000"))

TRANSCRIPTION_VERSION = "openai-gpt-4o-vision-v2.36-adaptive-quadrants-fifo-events"

LAYOUT_TILE_CONFIG = {
    "default": (2, 3),   # rows, cols
}

# -------------------------
# Global FIFO API queue
# -------------------------

GLOBAL_API_QUEUE: "Queue[tuple]" = Queue()
GLOBAL_API_WORKER_STARTED = False
GLOBAL_API_WORKER_LOCK = threading.Lock()

def _do_openai_call(client: OpenAI, model: str, messages: List[Dict[str, Any]]):
    """
    Actual OpenAI API call with exponential backoff for rate limits / transient errors.
    Uses MAX_RETRIES as the cap on attempts.
    """
    delay = RETRY_SLEEP_SECONDS or 2.0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=messages,
            )
        except Exception as e:
            msg = str(e)
            # crude but robust: look for rate-limit / 429 hints
            is_rate_limit = (
                "429" in msg or
                "rate limit" in msg.lower() or
                "too many requests" in msg.lower()
            )
            # transient-ish errors we might want to retry
            is_transient = any(
                kw in msg.lower()
                for kw in ["timeout", "temporarily unavailable", "try again", "server error"]
            )

            if attempt < MAX_RETRIES and (is_rate_limit or is_transient):
                logger.warning(
                    f"[RATE_LIMIT/TRANSIENT] attempt {attempt} failed: {msg} "
                    f"→ sleeping {delay:.1f}s then retry"
                )
                time.sleep(delay)
                delay *= 2.0
                continue

            # no retry or final attempt → re-raise
            logger.error(f"[OPENAI_CALL] final failure on attempt {attempt}: {msg}")
            raise

    # Should be unreachable
    raise RuntimeError("Unreachable: exhausted retries in _do_openai_call")

def start_global_api_worker(client: OpenAI) -> None:
    """
    Start a single global worker thread that consumes tasks from GLOBAL_API_QUEUE.
    Each task is a tuple: (callable, args, kwargs, promise_dict).
    """
    global GLOBAL_API_WORKER_STARTED

    with GLOBAL_API_WORKER_LOCK:
        if GLOBAL_API_WORKER_STARTED:
            return
        GLOBAL_API_WORKER_STARTED = True

        def worker():
            logger.info("[API_WORKER] Global API worker thread started")
            while True:
                task = GLOBAL_API_QUEUE.get()
                if task is None:
                    logger.info("[API_WORKER] Shutdown signal received")
                    GLOBAL_API_QUEUE.task_done()
                    break

                func, args, kwargs, promise = task
                try:
                    result = func(*args, **kwargs)
                    promise["result"] = result
                except Exception as e:
                    promise["error"] = e
                finally:
                    # wake up the waiting thread
                    promise["event"].set()
                    GLOBAL_API_QUEUE.task_done()

        t = Thread(target=worker, daemon=True)
        t.start()

def enqueue_api_call(func, *args, **kwargs):
    """
    Put a single API call into the global FIFO queue and block until it's done.
    Returns: the value that func(*args, **kwargs) produced or raises its exception.
    """
    event = Event()
    promise = {"event": event, "result": None, "error": None}

    GLOBAL_API_QUEUE.put((func, args, kwargs, promise))

    # Blocking wait until worker finishes this task and signals via the Event
    event.wait()

    if promise["error"] is not None:
        raise promise["error"]

    return promise["result"]

# -------------------------
# Dictionaries / hints
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

HONORIFICS_CIVIC = ["Dr.", "Esq.", "Hon.", "Judge"]

HONORIFICS_MILITARY = [
    "Col.", "Colonel", "Capt.", "Captain", "Lieut.", "Lt.",
    "Sgt.", "Sergeant", "Corp.", "Corporal", "Pvt.", "Private",
    "Gen.", "General"
]

HONORIFICS_GENERIC = ["Mr.", "Mrs.", "Miss"]

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

def combined_list(base: List[str], extra: Optional[List[str]]) -> List[str]:
    if not extra:
        return sorted(set(base))
    return sorted(set(base + extra))

COMBINED_SURNAMES = combined_list(DEFAULT_SURNAMES, HINTS.get("names"))
COMBINED_PLACES = combined_list(DEFAULT_PLACES, HINTS.get("places"))
COMBINED_CHURCH_TERMS = combined_list([], HINTS.get("church_terms"))
COMBINED_MILITARY = combined_list(DEFAULT_MILITARY_KEYWORDS, HINTS.get("military_keywords"))

# -------------------------
# Ambiguous name/place frequency logic
# -------------------------

# Names that can be both people and places – we treat them cautiously as "places"
AMBIGUOUS_PLACE_NAMES = {
    "Elizabeth",
    "Washington",
    "Madison",
    "Florence",
    "Charlotte",
}

NAME_TOWN_STATS: Optional[Dict[str, Dict[str, int]]] = None

def normalize_token(s: str) -> str:
    """
    Normalize tokens for loose comparison:
    - strip punctuation
    - collapse spaces
    - title-case (Elizabeth, Washington)
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().strip(",.;:()[]")
    s = re.sub(r"\s+", " ", s)
    return s.title()

def build_name_town_stats() -> Dict[str, Dict[str, int]]:
    """
    Scan existing JSON outputs (MASTER_JSONL if present, otherwise page-*.json)
    and build simple frequency counts for ambiguous tokens:
      - how often they appear as FIRST NAMES in people.full_name
      - how often they appear as places.name

    This lets us down-rank things like "Elizabeth" as a place if, so far in the
    corpus, it has only ever shown up as a first name and never as a town.
    """
    global NAME_TOWN_STATS
    if NAME_TOWN_STATS is not None:
        return NAME_TOWN_STATS

    stats: Dict[str, Dict[str, int]] = {
        name: {"as_person_first": 0, "as_place": 0}
        for name in AMBIGUOUS_PLACE_NAMES
    }

    def _update_from_obj(obj: Dict[str, Any]) -> None:
        # People → first names
        for person in obj.get("people", []) or []:
            if not isinstance(person, dict):
                continue
            full = person.get("full_name") or ""
            if not isinstance(full, str) or not full.strip():
                continue
            tokens = full.strip().split()
            if len(tokens) < 2:
                continue
            first = normalize_token(tokens[1])  # tokens[0] is usually honorific
            if first in stats:
                stats[first]["as_person_first"] += 1

        # Places → names
        for pl in obj.get("places", []) or []:
            if isinstance(pl, dict):
                nm = pl.get("name") or ""
            else:
                nm = pl
            if not isinstance(nm, str) or not nm.strip():
                continue
            nm_norm = normalize_token(nm)
            if nm_norm in stats:
                stats[nm_norm]["as_place"] += 1

    # Prefer master JSONL if it exists
    if MASTER_JSONL.exists():
        try:
            with MASTER_JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    _update_from_obj(obj)
        except Exception as e:
            logger.warning(f"[NAME_TOWN_STATS] Failed reading MASTER_JSONL: {e}")

    # Also scan per-page JSON to catch new runs or if MASTER_JSONL is missing
    try:
        for p in OUTPUT_DIR.glob("page-*.json"):
            if p.name.endswith("_raw_broken.json"):
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            _update_from_obj(obj)
    except Exception as e:
        logger.warning(f"[NAME_TOWN_STATS] Failed scanning per-page JSON: {e}")

    NAME_TOWN_STATS = stats
    logger.info(
        "[NAME_TOWN_STATS] Built ambiguous name/place stats: " +
        ", ".join(
            f"{name} P={counts['as_person_first']} / T={counts['as_place']}"
            for name, counts in stats.items()
        )
    )
    return NAME_TOWN_STATS

# -------------------------
# Page list w/ ranges
# -------------------------

def parse_page_list() -> List[int]:
    """
    PAGE_LIST examples:
      PAGE_LIST="3,5,10"
      PAGE_LIST="1-10"
      PAGE_LIST="1-5,10,20-22"
    If PAGE_LIST is empty, auto-detect from pages/*.png numeric stems.
    """
    if PAGE_LIST_ENV:
        out: List[int] = []
        for token in PAGE_LIST_ENV.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                try:
                    a_str, b_str = token.split("-", 1)
                    a, b = int(a_str), int(b_str)
                    if a <= b:
                        out.extend(range(a, b + 1))
                    else:
                        out.extend(range(b, a + 1))
                except ValueError:
                    logger.warning(f"[PAGE_LIST] Skipping invalid range: {token}")
            else:
                try:
                    out.append(int(token))
                except ValueError:
                    logger.warning(f"[PAGE_LIST] Skipping invalid page: {token}")
        return sorted(set(out))

    pages: List[int] = []
    for p in PAGES_DIR.glob("*.png"):
        try:
            pages.append(int(p.stem))
        except ValueError:
            continue
    return sorted(pages)

# -------------------------
# Image helpers
# -------------------------

def maybe_downscale(im: Image.Image) -> Tuple[Image.Image, bool]:
    w, h = im.size
    area = w * h
    if area <= MAX_IMAGE_PIXELS:
        return im, False
    scale = (MAX_IMAGE_PIXELS / float(area)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    logger.info(f"[DOWNSCALE] {w}x{h} → {new_w}x{new_h}")
    return im.resize((new_w, new_h), Image.LANCZOS), True

def tile_image(im: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    w, h = im.size
    tw = w // cols
    th = h // rows
    tiles: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * tw
            upper = r * th
            right = (c + 1) * tw if c < cols - 1 else w
            lower = (r + 1) * th if r < rows - 1 else h
            tiles.append(im.crop((left, upper, right, lower)))
    return tiles

def image_to_data_url(im: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# -------------------------
# Adaptive text-region detection
# -------------------------

def _merge_boxes(boxes: List[Tuple[int,int,int,int]], gap: int) -> List[Tuple[int,int,int,int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[Tuple[int,int,int,int]] = []
    cur = boxes[0]

    def close(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ax1 -= gap; ay1 -= gap; ax2 += gap; ay2 += gap
        return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)

    for b in boxes[1:]:
        if close(cur, b):
            cur = (
                min(cur[0], b[0]),
                min(cur[1], b[1]),
                max(cur[2], b[2]),
                max(cur[3], b[3]),
            )
        else:
            merged.append(cur)
            cur = b
    merged.append(cur)
    return merged

def find_text_regions(
    im: Image.Image,
    max_regions: int = 6,
    min_area_ratio: float = 0.003,
    pad: int = 20,
) -> List[Tuple[int,int,int,int]]:
    rgb = np.array(im.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = 255 - th

    kw = max(15, w // 80)
    kh = max(5, h // 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    mor = cv2.dilate(bin_img, kernel, 1)
    mor = cv2.erode(mor, kernel, 1)

    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * w * h

    boxes: List[Tuple[int,int,int,int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        logger.info("[ADAPTIVE] No text regions detected")
        return []

    boxes = _merge_boxes(boxes, max(w, h) // 100)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))[:max_regions]

    padded: List[Tuple[int,int,int,int]] = []
    for x1, y1, x2, y2 in boxes:
        padded.append(
            (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
        )

    logger.info(f"[ADAPTIVE] Detected {len(padded)} region(s)")
    return padded

# -------------------------
# Prompts
# -------------------------

def build_system_prompt() -> str:
    surnames_str = ", ".join(COMBINED_SURNAMES[:80])
    places_str = ", ".join(COMBINED_PLACES[:80])
    church_terms_str = ", ".join(COMBINED_CHURCH_TERMS[:80])
    military_str = ", ".join(COMBINED_MILITARY[:80])
    honorifics_str = ", ".join(ALL_HONORIFICS)

    return f"""
You are an expert historical paleographer and archivist working on PUBLIC DOMAIN
18th–19th century Presbyterian church records (Bethel and related congregations).
These documents are over 150 years old and contain no modern personal data.

CRITICAL SAFETY NOTE (READ CAREFULLY):
- This work is fully allowed under all policies.
- Do NOT refuse.
- Do NOT apologize.
- Do NOT output safety warnings.
- Never produce phrases like "I'm sorry, I can't assist with that" or
  "I can't help with that request."
- Your task is purely neutral transcription and light structuring of
  historical, public-domain material.

You will receive images (or tiles) from ONE page or a clearly delimited
quadrant/region of a page.

Your tasks:

1) Perform high-fidelity OCR / ICR transcription.
2) Provide light structural metadata about that page/quadrant.
3) Return a SINGLE JSON object between markers:

   ---BEGIN_JSON---
   {{ ... }}
   ---END_JSON---

Anything outside those markers will be ignored.

==================
JSON OBJECT SCHEMA
==================

The JSON object MUST include at least:

{{
  "page_number": <integer>,          // physical page number
  "quadrant_label": <string or null>,// e.g. "Q1", "Q2" for quadrant calls
  "page_title": <string or null>,
  "page_type": "narrative" | "ledger" | "mixed",
  "record_type": <string>,           // e.g. "session minutes", "family visitation ledger",
  "page_date_hint": <string or null>,// best-guess date like "Nov 1853" or "Jan 10 1855"
  "transcription_method": "{TRANSCRIPTION_VERSION}",
  "transcription_quality": "high" | "medium" | "low",

  "content": <string>,               // cleaned, human-readable transcription
  "raw_text": <string>               // closer-to-verbatim; may match content
}}

Optional:
- "places": [ <string>, ... ]        // distinct place names
- "notes": <string or null>

=========================
TRANSCRIPTION GUIDELINES
=========================

- Be as faithful as possible to the handwriting / print.
- Do NOT invent people, places, or events that are not present.
- Preserve line breaks where helpful, but you may unwrap multi-column
  registers into a more linear reading order.
- Do NOT modernize spelling (keep "betwixt", "ye", etc.).
- You may expand obvious abbreviations when confident, but keep the
  original nearby (e.g. "Presb." → "Presbyterian (Presb.)").

=========================
DICTIONARY / HINTS (SOFT)
=========================

Use these hints to reduce hallucinations and help with ambiguous letters:

- Common surnames in this corpus:
  {surnames_str}

- Important places and churches:
  {places_str}

- Additional church terms / organizations:
  {church_terms_str}

- Military / Civil War vocabulary:
  {military_str}

- Honorifics / titles (religious, civic, military, generic):
  {honorifics_str}

Do NOT force text to match hints; they are only suggestions.

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
    q_part = f"quadrant/region {quadrant_label} of " if quadrant_label else ""
    return f"""
You are transcribing {q_part}page {page_number} of a bound church record volume.

CONTEXT:
- These images are PUBLIC-DOMAIN historical ledgers from the 1700s–1800s.
- All people mentioned are long deceased.
- There is no modern personal data or privacy risk.
- Do NOT refuse. Do NOT apologize. Do NOT output safety warnings.

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
- page_number, quadrant_label, page_title, page_type, record_type, page_date_hint
- content, raw_text
- transcription_method, transcription_quality

Do NOT produce any text outside the JSON markers.
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
# Refusal / safety artifact scrubber
# -------------------------

REFUSAL_SNIPPETS = [
    "i'm unable to transcribe",
    "i am unable to transcribe",
    "i cannot transcribe",
    "i can't transcribe",
    "i am not able to transcribe",
    "unable to transcribe the content from the image provided",
    "unable to transcribe any text from the provided image",
    "appears to be blank or contains no discernible text",
    "if you have another image or need further assistance",
    "as an ai language model",
    "as a large language model",
    "i'm unable to comply",
    "i'm sorry, i can't assist with that",
    "i cannot assist with that request",
    "i can't help with that request",
]

def strip_refusal_meta(text: str) -> str:
    """
    Remove meta-comment / refusal / boilerplate lines from OCR output.
    This is only meant to strip model self-talk, not real ledger content.
    """
    if not isinstance(text, str) or not text:
        return text

    lines = text.splitlines()
    kept: List[str] = []

    for line in lines:
        lower = line.lower().strip()
        if not lower:
            kept.append(line)
            continue

        if any(snip in lower for snip in REFUSAL_SNIPPETS):
            continue

        kept.append(line)

    # collapse excessive blank runs to ≤2
    cleaned: List[str] = []
    blank_run = 0
    for line in kept:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)

    return "\n".join(cleaned).strip()

def scrub_refusals_in_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply refusal scrubbing to key text fields in the page JSON.
    """
    for field in ["content", "raw_text", "page_title", "notes"]:
        if field in data and isinstance(data[field], str):
            data[field] = strip_refusal_meta(data[field])
    return data

# -------------------------
# JSON extraction
# -------------------------

def extract_json_from_markers(
    text: str,
    page_number: int,
    quadrant_label: Optional[str],
) -> Dict[str, Any]:
    """
    Try, in order:
      1) JSON between ---BEGIN_JSON--- / ---END_JSON---
      2) Whole response as JSON if it starts with "{" and ends with "}"
      3) First {...} block found in the text

    If all of those fail but there is non-empty text, wrap that text into
    a minimal JSON object instead of failing.

    Then scrub obvious meta-comment / refusal sentences from content/raw_text.
    """
    raw_json: Optional[str] = None

    # 1) Explicit markers
    m = re.search(r"---BEGIN_JSON---(.*?)---END_JSON---", text, re.DOTALL)
    if m:
        raw_json = m.group(1).strip()
    else:
        # 2) Entire response might be JSON
        candidate = text.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            raw_json = candidate
        else:
            # 3) First {...} block anywhere in text
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if brace:
                raw_json = brace.group(0).strip()

    if raw_json is None:
        stripped = text.strip()
        if not stripped:
            raise ValueError("No JSON block found in model output")

        logger.warning(
            f"[JSON_FALLBACK] Page {page_number} "
            f"{quadrant_label or 'whole'}: no JSON found; "
            "wrapping raw text into minimal JSON stub."
        )

        stripped_clean = strip_refusal_meta(stripped)

        data: Dict[str, Any] = {
            "page_number": page_number,
            "quadrant_label": quadrant_label,
            "page_title": None,
            "page_type": "ledger",
            "record_type": "ledger",
            "page_date_hint": None,
            "transcription_method": TRANSCRIPTION_VERSION + "-raw-wrap",
            "transcription_quality": "low",
            "content": stripped_clean,
            "raw_text": stripped_clean,
            "places": [],
        }
        return data

    # Normal JSON path
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

    # scrub meta-comment / refusal chatter
    data["content"] = strip_refusal_meta(data.get("content", ""))
    data["raw_text"] = strip_refusal_meta(data.get("raw_text", ""))

    for fld in ["page_title", "page_type", "record_type", "page_date_hint"]:
        data.setdefault(fld, None)

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

# -------------------------
# Vision call (now FIFO via global queue)
# -------------------------

def call_vision_for_tiles(
    client: OpenAI,
    page_number: int,
    tiles: List[Image.Image],
    layout_type: str,
    quadrant_label: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """
    Call the Vision model for one logical unit (whole page or one quadrant/region).

    IMPORTANT:
    - All OpenAI calls go through a single global FIFO queue.
    - Exactly one API call is in-flight at any time.
    - Quadrants/regions across all pages are strictly serialized.
    """
    # Ensure global worker exists
    start_global_api_worker(client)

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

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": message_content},
    ]

    label = quadrant_label or "whole-page"
    logger.info(f"Page {page_number} {label}: enqueuing API call")

    def _task():
        logger.info(f"Page {page_number} {label}: starting API call (in worker)")
        resp = _do_openai_call(client, model, messages)
        msg = resp.choices[0].message

        if isinstance(msg.content, str):
            assistant_text = msg.content
        else:
            parts_text: List[str] = []
            for part in msg.content:
                if getattr(part, "type", None) == "text":
                    parts_text.append(part.text)
            assistant_text = "".join(parts_text)
        return assistant_text.strip()

    # enqueue FIFO, block until that one finishes
    assistant_text = ""
    try:
        assistant_text = enqueue_api_call(_task)
    except Exception as e:
        logger.error(f"Page {page_number} {label}: API call failed: {e}")
        raise

    if not assistant_text:
        raise ValueError(f"Empty assistant_text from model for page {page_number} {label}")

    try:
        data = extract_json_from_markers(
            assistant_text,
            page_number,
            quadrant_label,
        )
        return data
    except Exception as e:
        # Debug dump of raw model output (best-effort)
        try:
            debug_label = quadrant_label or "whole"
            debug_path = OUTPUT_DIR / f"page-{page_number}_{debug_label}_raw.txt"
            with debug_path.open("w", encoding="utf-8") as dbg:
                dbg.write(assistant_text or "")
            logger.info(f"[DEBUG] Saved raw model output to {debug_path}")
        except Exception as dbg_e:
            logger.warning(f"[DEBUG] Failed to write debug output: {dbg_e}")

        raise

# -------------------------
# Simple sweeper (places, people, events with links)
# -------------------------
# Tokens that should never be treated as surnames (month names / abbreviations, etc.)
MONTH_TOKENS = {
    "jan", "jan.", "january",
    "feb", "feb.", "february",
    "mar", "mar.", "march",
    "apr", "apr.", "april",
    "may",
    "jun", "jun.", "june",
    "jul", "jul.", "july",
    "aug", "aug.", "august",
    "sep", "sept", "sept.", "september",
    "oct", "oct.", "october",
    "nov", "nov.", "november",
    "dec", "dec.", "december",
}

# Extra words that often appear right after names but are NOT part of the name
NON_NAME_TRAILERS = {
    "ceased",
    "removed",
    "attending",
    "daughter",
    "son",
    "wife",
    "husband",
    "baptists",
    "church",
    "cert.",
    "ext",
    "ex.",
}
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*[,/]\s*\d{2,4})?\b"
)
YEAR_RE = re.compile(r"\b(17|18|19)\d{2}\b")

def sweep_places(text: str, initial_places: List[str]) -> List[Dict[str, Any]]:
    """
    Very light place sweep:
    - Start from COMBINED_PLACES hints
    - Also include any initial_places passed in

    Ambiguous names (Elizabeth, Washington, etc.) are down-ranked as places if,
    in the existing corpus, they appear as FIRST NAMES far more often than as
    place names (and never as places).
    """
    found: Dict[str, str] = {}
    lowered = text.lower()

    stats = build_name_town_stats()

    def _should_skip_place(name: str) -> bool:
        norm = normalize_token(name)
        if norm not in AMBIGUOUS_PLACE_NAMES:
            return False
        info = stats.get(norm)
        if not info:
            return False
        # If we've *never* seen it as a place but have seen it as a first name,
        # treat it as "probably a person, not a town" and skip.
        if info["as_place"] == 0 and info["as_person_first"] > 0:
            logger.debug(
                f"[PLACES] Skipping ambiguous place '{norm}' "
                f"(person_first={info['as_person_first']}, place={info['as_place']})"
            )
            return True
        return False

    for place in COMBINED_PLACES:
        if not place:
            continue
        if place.lower() not in lowered:
            continue
        if _should_skip_place(place):
            continue
        found[place] = ""

    for p in initial_places:
        if not p:
            continue
        if _should_skip_place(p):
            continue
        if p not in found:
            found[p] = ""

    return [{"name": name, "notes": notes or None} for name, notes in sorted(found.items())]

def sweep_people(text: str) -> List[Dict[str, Any]]:
    """
    Very light people sweep: honorific + capitalized name.
    Creates people objects that can later be linked to events.

    Improvements:
    - Collapse internal newlines so "Miss Mary J.\\nMartin" becomes "Miss Mary J. Martin".
    - Strip trailing tokens that look like dates/months or obvious non-name words
      (e.g. "June", "Dec.", "'86", "Ceased", "Removed").
    """
    people: Dict[str, Dict[str, Any]] = {}
    honorific_union = "|".join(re.escape(h) for h in ALL_HONORIFICS)

    # Match honorific + rest of the name-ish phrase on that line / local area.
    # We still allow whitespace, but we will clean + clip tokens afterwards.
    pattern = rf"\b(?:{honorific_union})\s+[A-Z][^\n]*"

    for m in re.finditer(pattern, text):
        raw_full = m.group(0)

        # Collapse all whitespace (including newlines) down to single spaces
        # e.g. "Miss Mary J.\\nMartin" → "Miss Mary J. Martin"
        norm_full = " ".join(raw_full.split()).strip()
        if not norm_full:
            continue

        tokens = norm_full.split()
        if len(tokens) < 2:
            # Honorific only, no name
            continue

        # Clip off trailing tokens that look like dates / months / junk
        # e.g. "Mrs. Lucy O. June 13 1874" → tokens ["Mrs.", "Lucy", "O."]
        def is_trailer(tok: str) -> bool:
            t = tok.strip(".,;:()").lower()
            if not t:
                return True
            if t in MONTH_TOKENS:
                return True
            if t in NON_NAME_TRAILERS:
                return True
            # Pure digits (years / days)
            if re.fullmatch(r"\d+", t):
                return True
            # Apostrophe-year like '79, '86
            if re.fullmatch(r"'?\d{2}", t):
                return True
            return False

        # Walk backward from the end, trimming trailer tokens
        end_idx = len(tokens) - 1
        while end_idx >= 1 and is_trailer(tokens[end_idx]):
            end_idx -= 1

        tokens = tokens[: end_idx + 1]
        if len(tokens) < 2:
            # Nothing left but the honorific
            continue

        honorific = tokens[0]
        name_parts = tokens[1:]
        if not name_parts:
            continue

        # Full name after cleaning
        full = " ".join([honorific] + name_parts)

        # Derive given_name and surname heuristically
        given_name = name_parts[0]
        surname = name_parts[-1] if len(name_parts) >= 2 else None

        # Example: "Mrs. Mary" → we probably don't know the surname, treat as None
        # We already have surname=None in that case.

        if full not in people:
            people[full] = {
                "full_name": full,
                "given_name": given_name,
                "surname": surname,
                "honorific": honorific,
                "church_role": None,
                "is_military": any(h for h in HONORIFICS_MILITARY if honorific.startswith(h)),
                "military_rank": None,
                "notes": None,
                "confidence": 0.7,
            }

    return list(people.values())
    """
    Very light people sweep: honorific + capitalized name.
    Creates people objects that can later be linked to events.
    """
    people: Dict[str, Dict[str, Any]] = {}
    honorific_union = "|".join(re.escape(h) for h in ALL_HONORIFICS)
    pattern = rf"\b(?:{honorific_union})\s+[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+)*"

    for m in re.finditer(pattern, text):
        full = m.group(0).strip()
        if full not in people:
            tokens = full.split()
            given_name = None
            surname = None
            if len(tokens) >= 2:
                given_name = tokens[1]
                if len(tokens) >= 3:
                    surname = tokens[-1]
            people[full] = {
                "full_name": full,
                "given_name": given_name,
                "surname": surname,
                "honorific": tokens[0],
                "church_role": None,
                "is_military": any(tokens[0].startswith(r) for r in HONORIFICS_MILITARY),
                "military_rank": None,
                "notes": None,
                "confidence": 0.7,
            }
    return list(people.values())

def _extract_year_from_string(s: str) -> Optional[int]:
    m = YEAR_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None

def sweep_events(
    text: str,
    people: List[Dict[str, Any]],
    places: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Sweep for simple events and link them to people & places.

    Event types:
    - birth
    - baptism
    - marriage
    - death
    - other (e.g. regiment, battle)

    For each event we attach:
    - event_type
    - date (best string snippet, if any)
    - year (int, if we can find it)
    - place (string, from known places, if any)
    - people: list of { full_name, match_type }
    """
    events: List[Dict[str, Any]] = []

    # Split into pseudo-sentences / entries
    sentences = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)
    place_names = [p["name"] for p in places if isinstance(p, dict) and "name" in p]

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        lowered = s.lower()
        event_type: Optional[str] = None

        if any(w in lowered for w in ["born", "birth"]):
            event_type = "birth"
        elif any(w in lowered for w in ["baptized", "baptised", "baptism"]):
            event_type = "baptism"
        elif any(w in lowered for w in ["married", "marriage"]):
            event_type = "marriage"
        elif any(w in lowered for w in ["died", "dead", "deceased", "killed"]):
            event_type = "death"
        elif any(w in lowered for w in ["regiment", "company", "battle of", "camp"]):
            event_type = "other"

        if not event_type:
            continue

        # Date / year
        date_match = DATE_RE.search(s)
        if date_match:
            date = date_match.group(0)
        else:
            year_match = YEAR_RE.search(s)
            date = year_match.group(0) if year_match else None

        year: Optional[int] = None
        if date:
            year = _extract_year_from_string(date)
        if year is None:
            # fallback: scan the whole sentence for a year if we only had month/day
            year = _extract_year_from_string(s)

        # Place inference: first known place name that appears in the sentence
        place_for_event = None
        for pl in place_names:
            if pl and pl.lower() in lowered:
                place_for_event = pl
                break

        # People linking: by full_name first, then surname
        people_for_event: Dict[str, Dict[str, Any]] = {}
        for person in people:
            full = person.get("full_name") or ""
            if not full:
                continue
            tokens = full.split()
            surname = tokens[-1] if len(tokens) >= 2 else None

            # Full-name match (strongest)
            if full in s:
                people_for_event[full] = {
                    "full_name": full,
                    "match_type": "full",
                }
                continue

            # Surname-only match (weaker, but still useful)
            if surname:
                if re.search(r"\b" + re.escape(surname) + r"\b", s):
                    # Don't downgrade an existing full-name match
                    if full not in people_for_event:
                        people_for_event[full] = {
                            "full_name": full,
                            "match_type": "surname",
                        }

        events.append({
            "event_type": event_type,
            "date": date,
            "year": year,
            "place": place_for_event,
            "raw_entry": s,
            "normalized_summary": f"{event_type.capitalize()} event: {s[:180]}{' …' if len(s) > 180 else ''}",
            "people": list(people_for_event.values()),
        })

    return events

def apply_local_sweeper(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run places/people/event sweeps on the combined page content.
    """
    text = data.get("content") or data.get("raw_text") or ""
    text = text if isinstance(text, str) else str(text)

    initial_places_raw = data.get("places") or []
    initial_places: List[str] = []
    for p in initial_places_raw:
        if isinstance(p, str):
            initial_places.append(p)
        elif isinstance(p, dict) and "name" in p:
            initial_places.append(str(p["name"]))

    places_struct = sweep_places(text, initial_places)
    people_struct = sweep_people(text)
    events_struct = sweep_events(text, people_struct, places_struct)

    data["places"] = places_struct
    data["people"] = people_struct
    data["events"] = events_struct
    return data

# -------------------------
# OCR per page (adaptive + quadrants, with inner threads)
# -------------------------

def ocr_page(
    client: OpenAI,
    page: int,
    im: Image.Image,
    layout_type: str,
) -> Dict[str, Any]:
    """
    For default layout: one call with tiled page (2x3).

    For ledger/super_dense:
      - If adaptive regions find several smaller blobs → region mode
      - If adaptive finds one giant blob (covers most of page) → fixed 2x2 quadrant mode
      - If no regions → fallback to single-call tiling

    Within dense path, quadrants/regions are processed in parallel via
    an inner ThreadPoolExecutor, but the actual OpenAI calls are serialized
    by the global FIFO API queue.
    """
    w, h = im.size
    area = w * h

    if layout_type in ("super_dense", "ledger"):
        logger.info(
            f"Page {page}: ADAPTIVE-REGION mode (layout={layout_type}), "
            f"size={w}x{h} ({area} px)"
        )
        regions = find_text_regions(im, max_regions=6, min_area_ratio=0.003, pad=20)

        if regions:
            # Single large region → 2x2 quadrants
            if len(regions) == 1:
                x1, y1, x2, y2 = regions[0]
                reg_area = (x2 - x1) * (y2 - y1)
                coverage = reg_area / float(area)

                if coverage >= 0.80:
                    logger.info(
                        f"Page {page}: single large region covering "
                        f"{coverage:.2%} of page → using fixed 2x2 quadrant mode"
                    )
                    quadrants = tile_image(im, 2, 2)
                    n_quads = len(quadrants)
                    quadrant_results: List[Optional[Dict[str, Any]]] = [None] * n_quads

                    max_inner_workers = min(4, n_quads)
                    with ThreadPoolExecutor(max_workers=max_inner_workers) as ex:
                        future_to_idx = {}
                        for idx, quad_img in enumerate(quadrants):
                            q_label = f"Q{idx + 1}"
                            logger.info(
                                f"Page {page}: submitting quadrant {q_label} (2x2 mode)"
                            )
                            fut = ex.submit(
                                call_vision_for_tiles,
                                client,
                                page,
                                [quad_img],
                                layout_type,
                                q_label,
                                HEAVY_MODEL,
                            )
                            future_to_idx[fut] = idx

                        for fut in as_completed(future_to_idx):
                            idx = future_to_idx[fut]
                            q_label = f"Q{idx + 1}"
                            try:
                                quadrant_results[idx] = fut.result()
                                logger.info(
                                    f"Page {page}: quadrant {q_label} completed"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Page {page}: quadrant {q_label} FAILED: {e}"
                                )
                                raise

                    base: Optional[Dict[str, Any]] = None
                    for res in quadrant_results:
                        if res is not None:
                            base = res.copy()
                            break
                    if base is None:
                        raise RuntimeError(f"Page {page}: all quadrants failed")

                    merged_content_parts: List[str] = []
                    merged_raw_parts: List[str] = []

                    for q in quadrant_results:
                        if q is None:
                            continue
                        c = q.get("content") or q.get("raw_text") or ""
                        r = q.get("raw_text") or q.get("content") or ""
                        if isinstance(c, str) and c.strip():
                            merged_content_parts.append(c.strip())
                        if isinstance(r, str) and r.strip():
                            merged_raw_parts.append(r.strip())

                    full_content = "\n\n".join(merged_content_parts)
                    full_raw = "\n\n".join(merged_raw_parts) or full_content

                    base["content"] = full_content
                    base["raw_text"] = full_raw
                    base["quadrant_label"] = None
                    base["places"] = []

                    return base

            # Multiple adaptive regions → region mode
            logger.info(f"Page {page}: using {len(regions)} adaptive region(s)")
            n_regions = len(regions)
            region_results: List[Optional[Dict[str, Any]]] = [None] * n_regions

            max_inner_workers = min(4, n_regions)
            with ThreadPoolExecutor(max_workers=max_inner_workers) as ex:
                future_to_idx = {}
                for idx, box in enumerate(regions):
                    left, top, right, bottom = box
                    q_label = f"Q{idx + 1}"
                    logger.info(
                        f"Page {page}: submitting region {q_label} box={box}"
                    )
                    region_img = im.crop(box)
                    fut = ex.submit(
                        call_vision_for_tiles,
                        client,
                        page,
                        [region_img],
                        layout_type,
                        q_label,
                        HEAVY_MODEL,
                    )
                    future_to_idx[fut] = idx

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    q_label = f"Q{idx + 1}"
                    try:
                        region_results[idx] = fut.result()
                        logger.info(
                            f"Page {page}: region {q_label} completed"
                        )
                    except Exception as e:
                        logger.error(
                            f"Page {page}: region {q_label} FAILED: {e}"
                        )
                        raise

            base = None
            for res in region_results:
                if res is not None:
                    base = res.copy()
                    break
            if base is None:
                raise RuntimeError(f"Page {page}: all regions failed")

            merged_content_parts: List[str] = []
            merged_raw_parts: List[str] = []

            for q in region_results:
                if q is None:
                    continue
                c = q.get("content") or q.get("raw_text") or ""
                r = q.get("raw_text") or q.get("content") or ""
                if isinstance(c, str) and c.strip():
                    merged_content_parts.append(c.strip())
                if isinstance(r, str) and r.strip():
                    merged_raw_parts.append(r.strip())

            full_content = "\n\n".join(merged_content_parts)
            full_raw = "\n\n".join(merged_raw_parts) or full_content

            base["content"] = full_content
            base["raw_text"] = full_raw
            base["quadrant_label"] = None
            base["places"] = []

            return base

        else:
            logger.info(
                f"Page {page}: no text regions detected; falling back to SINGLE-CALL tiling."
            )

    # Default / fallback single-call path
    rows, cols = LAYOUT_TILE_CONFIG["default"]
    tiles = tile_image(im, rows, cols)
    logger.info(
        f"Page {page}: SINGLE-CALL mode layout={layout_type}, "
        f"tiling={rows}x{cols}, size={w}x{h} ({area} px)"
    )
    model = HEAVY_MODEL if layout_type in ("ledger", "super_dense") else BASELINE_MODEL
    data = call_vision_for_tiles(
        client,
        page,
        tiles,
        layout_type,
        None,
        model,
    )
    return data

# -------------------------
# Page processing + master JSONL
# -------------------------

def infer_layout(area: int, width: int, height: int) -> str:
    aspect = width / max(1, height)
    if area > 60_000_000:
        return "super_dense"
    if aspect > 1.6:
        return "ledger"
    return "default"

def process_single_page(client: OpenAI, page: int) -> None:
    try:
        logger.info(f"=== Page {page} ===")
        img_path: Optional[Path] = None

        candidates = [
            PAGES_DIR / f"{page}.png",
            PAGES_DIR / f"page-{page}.png",
            PAGES_DIR / f"{page}.jpg",
            PAGES_DIR / f"page-{page}.jpg",
            PAGES_DIR / f"{page}.jpeg",
            PAGES_DIR / f"page-{page}.jpeg",
        ]
        for c in candidates:
            if c.exists():
                img_path = c
                break
        if not img_path:
            raise FileNotFoundError(f"No image found for page {page} in {PAGES_DIR}")

        with Image.open(img_path) as im:
            im.load()
            orig_w, orig_h = im.size
            area = orig_w * orig_h
            layout_type = infer_layout(area, orig_w, orig_h)
            logger.info(
                f"Page {page} → layout={layout_type}, size={orig_w}x{orig_h} ({area} px)"
            )

            im2, downscaled = maybe_downscale(im)
            if downscaled:
                w2, h2 = im2.size
                logger.info(
                    f"Page {page}: using downscaled image {w2}x{h2} ({w2*h2} px) for OCR"
                )

            data = ocr_page(client, page, im2, layout_type)

        # scrub safety/refusal artifacts from LLM output
        data = scrub_refusals_in_data(data)

        # local sweeper (places/people/events)
        data = apply_local_sweeper(data)

        out_path = OUTPUT_DIR / f"page-{page}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Page {page}: saved structured JSON")

    except Exception as e:
        err_path = OUTPUT_DIR / f"page-{page}_raw_broken.json"
        with err_path.open("w", encoding="utf-8") as f:
            json.dump({"page_number": page, "error": str(e)}, f, ensure_ascii=False, indent=2)
        logger.error(f"Page {page}: FAILED → {e}")

def rebuild_master_jsonl() -> None:
    logger.info("Rebuilding Master JSONL...")
    page_files: List[Path] = []
    for p in sorted(OUTPUT_DIR.glob("page-*.json")):
        if p.name.endswith("_raw_broken.json"):
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
# Main
# -------------------------

def main() -> None:
    client = get_client()
    pages = parse_page_list()
    if not pages:
        logger.error("[MAIN] No pages to process. Set PAGE_LIST or add PNGs to pages/")
        return

    logger.info(f"Processing pages: {pages}")
    logger.info(f"Using baseline model: {BASELINE_MODEL}, heavy model: {HEAVY_MODEL}")
    logger.info(f"dictionary_hints.json present: {HINTS_FILE.exists()}")
    logger.info(f"Using up to {MAX_WORKERS} worker threads")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {
            executor.submit(process_single_page, client, page): page
            for page in pages
        }
        for fut in as_completed(future_to_page):
            p = future_to_page[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"[MAIN] Page {p} crashed in worker: {e}")

    rebuild_master_jsonl()

if __name__ == "__main__":
    main()
