#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bethel OCR / ICR Pipeline
Version: 2.32 (Adaptive + Quadrants + Parallel + Safe Prompts + Refusal & Commentary Scrubbers)

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
- Page-level parallelism with ThreadPoolExecutor
- Refusal / safety boilerplate scrubber on content/raw_text
- Commentary / STRIP header scrubber on content/raw_text
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

from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import numpy as np
import cv2

# Hard cap on concurrent OpenAI calls (pages × quadrants combined)
MAX_PARALLEL_CALLS = int(os.getenv("OCR_MAX_PARALLEL_CALLS", "4"))
API_SEMAPHORE = threading.Semaphore(MAX_PARALLEL_CALLS)

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

TRANSCRIPTION_VERSION = "openai-gpt-4o-vision-v2.32-adaptive-quadrants"

LAYOUT_TILE_CONFIG = {
    "default": (2, 3),   # rows, cols
}

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
# COMMENTARY / STRIP HEADER SCRUBBER
# -------------------------

STRIP_HEADER_RE = re.compile(
    r"---\s*STRIP\s*\d+\s*/\s*\d+\s*---",
    re.IGNORECASE
)

COMMENT_HEADER_RE = re.compile(
    r"(Register\s*:.*?$)|"
    r"(Baptism(?:s|al)?\s+Register.*?$)|"
    r"(Christenings.*?$)|"
    r"(Continuation\s+Page.*?$)|"
    r"(Page\s+\d+\s+of\s+\d+.*?$)",
    re.IGNORECASE | re.MULTILINE
)

MODEL_CHATTER_RE = re.compile(
    r"(This appears to be.*?$)|"
    r"(The following text shows.*?$)|"
    r"(It seems like the page contains.*?$)",
    re.IGNORECASE | re.MULTILINE
)

def strip_commentary(text: str) -> str:
    """
    Remove model-invented headers, 'STRIP X/Y' markers, and obvious explanatory commentary.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    # Remove --- STRIP X/Y --- markers
    text = STRIP_HEADER_RE.sub("", text)

    # Remove section headers like "Register:", "Baptism …", etc.
    text = COMMENT_HEADER_RE.sub("", text)

    # Remove simple explanatory chatter
    text = MODEL_CHATTER_RE.sub("", text)

    # Clean extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def scrub_commentary_in_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply commentary scrubbing to key text fields in the page JSON.
    """
    for fld in ["content", "raw_text", "page_title", "notes"]:
        if fld in data and isinstance(data[fld], str):
            data[fld] = strip_commentary(data[fld])
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
# Vision call
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

    Concurrency is globally limited by API_SEMAPHORE so that the total number
    of in-flight OpenAI calls across all pages/quadrants never exceeds
    MAX_PARALLEL_CALLS.
    """
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
    assistant_text: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        label = quadrant_label or "whole-page"
        logger.info(f"Page {page_number} {label} attempt {attempt}")
        try:
            # Global concurrency limiter for all API calls
            with API_SEMAPHORE:
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

            data = extract_json_from_markers(
                assistant_text,
                page_number,
                quadrant_label,
            )
            return data

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt} failed: {e}")

            # Debug dump of raw model output (best-effort)
            try:
                debug_label = quadrant_label or "whole"
                debug_path = OUTPUT_DIR / f"page-{page_number}_{debug_label}_raw.txt"
                with debug_path.open("w", encoding="utf-8") as dbg:
                    dbg.write(assistant_text or "")
                logger.info(f"[DEBUG] Saved raw model output to {debug_path}")
            except Exception as dbg_e:
                logger.warning(f"[DEBUG] Failed to write debug output: {dbg_e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Page {page_number} failed: {last_error}")

# -------------------------
# Simple sweeper (optional light post-processing)
# -------------------------

DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*[,/]\s*\d{2,4})?\b"
)
YEAR_RE = re.compile(r"\b(17|18|19)\d{2}\b")

def sweep_places(text: str, initial_places: List[str]) -> List[Dict[str, Any]]:
    found: Dict[str, str] = {}
    lowered = text.lower()

    for place in COMBINED_PLACES:
        if place and place.lower() in lowered:
            found[place] = ""

    for p in initial_places:
        if p and p not in found:
            found[p] = ""

    return [{"name": name, "notes": notes or None} for name, notes in sorted(found.items())]

def sweep_people(text: str) -> List[Dict[str, Any]]:
    # Very light people sweep: honorific + capitalized name
    people: Dict[str, Dict[str, Any]] = {}
    honorific_union = "|".join(re.escape(h) for h in ALL_HONORIFICS)
    pattern = rf"\b(?:{honorific_union})\s+[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+)*"

    for m in re.finditer(pattern, text):
        full = m.group(0).strip()
        if full not in people:
            people[full] = {
                "full_name": full,
                "given_name": None,
                "surname": None,
                "honorific": full.split()[0],
                "church_role": None,
                "is_military": any(full.startswith(r) for r in HONORIFICS_MILITARY),
                "military_rank": None,
                "notes": None,
                "confidence": 0.7,
            }
    return list(people.values())

def sweep_events(text: str, people: List[Dict[str, Any]], places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    sentences = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)
    place_names = [p["name"] for p in places]

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
        elif any(w in lowered for w in ["regiment", "company", "battle of"]):
            event_type = "other"

        if not event_type:
            continue

        date_match = DATE_RE.search(s) or YEAR_RE.search(s)
        date = date_match.group(0) if date_match else None

        place_for_event = None
        for pl in place_names:
            if pl and pl.lower() in lowered:
                place_for_event = pl
                break

        events.append({
            "event_type": event_type,
            "date": date,
            "place": place_for_event,
            "raw_entry": s,
            "normalized_summary": f"{event_type.capitalize()} event: {s[:180]}{' …' if len(s) > 180 else ''}",
            "people": [],
        })

    return events

def apply_local_sweeper(data: Dict[str, Any]) -> Dict[str, Any]:
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
    an inner ThreadPoolExecutor.
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

        # scrub STRIP headers + model commentary from output
        data = scrub_commentary_in_data(data)

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
