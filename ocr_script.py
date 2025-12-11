#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bethel OCR / ICR Pipeline
Version: 2.42 (Ledger mode + NameExtractor + Financial Sweeper + Stats-based Name Filter)

Features:
- Vision OCR via OpenAI (gpt-4o-mini / gpt-4o)
- Automatic layout density detection
- Optional per-page layout overrides (page_layout_overrides.json)
- Ledger-specific prompting and schema for attendance/dues sheets
- Adaptive text-region detection via OpenCV for super-dense pages
- For super-dense pages:
    * Adaptive region mode
    * If only one large region → fixed 2x2 quadrants
    * Quadrants/regions processed in parallel (inner ThreadPoolExecutor)
- For default / ledger pages:
    * Single Vision call on tiles (ledger uses wide 1x3 tiling)
- Strong prompts (system + user), now loaded from external files if present:
    * prompts/ocr_system_prompt_v2.37.txt (default narrative)
    * prompts/ocr_user_prompt_v2.37.txt
    * prompts/ocr_system_prompt_ledger.txt (optional ledger-specific)
    * prompts/ocr_user_prompt_ledger.txt
- Tolerant JSON extraction:
    * Markers, bare JSON, first {...}
    * If none → wrap raw text into minimal JSON stub
- Local post-processing (v2.42):
    * NameExtractor with statistical name filter (based on corpus + hints)
    * Place sweeper from dictionary hints
    * Event sweeper with tighter multi-war military detection
    * Financial sweeper for $ / monetary lines
    * Ledger pages bypass sweepers (they already return structured rows)
- Refusal / safety boilerplate scrubber on content/raw_text
- Global FIFO queue for ALL OpenAI calls
- Page-level processing is now sequential:
    * One page at a time
    * That page may still use inner threads for quadrants/regions
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
from typing import Dict, Any, List, Optional, Tuple, Set
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

PROMPTS_DIR = BASE_DIR / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "ocr_system_prompt_v2.37.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "ocr_user_prompt_v2.37.txt"
LEDGER_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "ocr_system_prompt_ledger.txt"
LEDGER_USER_PROMPT_PATH = PROMPTS_DIR / "ocr_user_prompt_ledger.txt"

BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
HEAVY_MODEL = os.getenv("HEAVY_MODEL", "gpt-4o")

PAGE_LIST_ENV = os.getenv("PAGE_LIST", "").strip()

MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "2"))
RETRY_SLEEP_SECONDS = float(os.getenv("OCR_RETRY_SLEEP", "4"))
MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "4"))

MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "90000000"))

TRANSCRIPTION_VERSION = "openai-gpt-4o-vision-v2.42-ledger"

LAYOUT_TILE_CONFIG = {
    "default": (2, 3),   # rows, cols
    "ledger": (1, 3),    # wide tiling for two-page ledgers
}

# -------------------------
# Layout overrides
# -------------------------

def load_layout_overrides() -> Dict[int, str]:
    """
    Optional JSON file allowing manual layout override per page.

    Example page_layout_overrides.json:
    {
      "182": "ledger",
      "183": "ledger",
      "271": "super_dense"
    }
    """
    if not LAYOUT_OVERRIDES_PATH.exists():
        logger.info("[LAYOUT_OVERRIDES] No page_layout_overrides.json – using inferred layouts only")
        return {}
    try:
        with LAYOUT_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        overrides: Dict[int, str] = {}
        for k, v in raw.items():
            try:
                page = int(k)
            except ValueError:
                continue
            v_norm = str(v).strip().lower()
            if v_norm not in {"default", "ledger", "super_dense"}:
                continue
            overrides[page] = v_norm
        logger.info(f"[LAYOUT_OVERRIDES] Loaded overrides for pages: {sorted(overrides.keys())}")
        return overrides
    except Exception as e:
        logger.error(f"[LAYOUT_OVERRIDES] Failed to load {LAYOUT_OVERRIDES_PATH}: {e}")
        return {}

LAYOUT_OVERRIDES = load_layout_overrides()

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
            is_rate_limit = (
                "429" in msg or
                "rate limit" in msg.lower() or
                "too many requests" in msg.lower()
            )
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

            logger.error(f"[OPENAI_CALL] final failure on attempt {attempt}: {msg}")
            raise

    raise RuntimeError("Unreachable: exhausted retries in _do_openai_call")


def start_global_api_worker(client: OpenAI) -> None:
    """
    Start a single global worker thread that consumes tasks from GLOBAL_API_QUEUE.
    Each task: (callable, args, kwargs, promise_dict)
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
                    promise["event"].set()
                    GLOBAL_API_QUEUE.task_done()

        t = Thread(target=worker, daemon=True)
        t.start()


def enqueue_api_call(func, *args, **kwargs):
    """
    Put a single API call into the global FIFO queue and block until it's done.
    Returns: func(*args, **kwargs) or raises its exception.
    """
    event = Event()
    promise = {"event": event, "result": None, "error": None}

    GLOBAL_API_QUEUE.put((func, args, kwargs, promise))
    event.wait()

    if promise["error"] is not None:
        raise promise["error"]

    return promise["result"]

# -------------------------
# Dictionaries / hints
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

# default lists as safety net
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

DEFAULT_MILITARY_KEYWORDS = [
    "Civil War", "World War", "Revolutionary War",
    "regiment", "company", "infantry", "artillery", "cavalry",
    "battery", "brigade", "division",
    "killed in battle", "killed in action", "wounded in battle",
    "battle of", "camp", "army", "navy",
]

DEFAULT_HONORIFICS_RELIGIOUS = [
    "Rev.", "Revd.", "Pastor", "Elder", "Ruling Elder",
    "Deacon", "Moderator", "Clerk", "Trustee",
]

DEFAULT_HONORIFICS_CIVIC = ["Dr.", "Esq.", "Hon.", "Judge"]

DEFAULT_HONORIFICS_MILITARY = [
    "Col.", "Colonel", "Capt.", "Captain", "Lieut.", "Lt.",
    "Sgt.", "Sergeant", "Corp.", "Corporal", "Pvt.", "Private",
    "Gen.", "General",
]

DEFAULT_HONORIFICS_GENERIC = ["Mr.", "Mrs.", "Miss", "Ms."]

def combined_list(base: List[str], extra: Optional[List[str]]) -> List[str]:
    if not extra:
        return sorted(set(base))
    return sorted(set(base + extra))

COMBINED_SURNAMES = combined_list(DEFAULT_SURNAMES, HINTS.get("names"))
COMBINED_PLACES = combined_list(DEFAULT_PLACES, HINTS.get("places"))
COMBINED_CHURCH_TERMS = combined_list([], HINTS.get("church_terms"))
COMBINED_MILITARY = combined_list(DEFAULT_MILITARY_KEYWORDS, HINTS.get("military_keywords"))

ALL_HONORIFICS: List[str] = combined_list(
    DEFAULT_HONORIFICS_RELIGIOUS
    + DEFAULT_HONORIFICS_CIVIC
    + DEFAULT_HONORIFICS_MILITARY
    + DEFAULT_HONORIFICS_GENERIC,
    HINTS.get("honorifics")
)

HONORIFICS_MILITARY = DEFAULT_HONORIFICS_MILITARY[:]  # used for is_military flags

# Suffixes / prefixes / stopwords for NameExtractor
NAME_SUFFIXES = [
    "Jr.", "Sr.", "II", "III", "IV", "Esq.", "M.D.", "D.D.", "Ph.D."
]
SURNAME_PREFIXES = [
    "Mc", "Mac", "O'", "Von", "Van", "De", "Del", "Della", "St.", "St", "La", "Le"
]
NAME_STOPWORDS = {
    "of", "from", "in", "at", "on", "and", "with", "for", "by", "to",
    "church", "congregation", "presbytery", "session", "meeting",
    "minutes", "committee", "report", "budget", "ledger", "adjourned",
    "whereas", "therefore", "hereby", "trustees", "members", "pastorate",
    "treasurer", "secretary"
}

# -------------------------
# Name corpus stats (for statistical filtering)
# -------------------------

_NAME_STATS_CACHE: Optional[Dict[str, Set[str]]] = None


def build_name_stats() -> Dict[str, Set[str]]:
    """
    Build simple name statistics from:
      - dictionary_hints.json (names)
      - existing MASTER_JSONL people entries (if present)
      - per-page JSON files in ocr_output/
    Returns dict with sets:
      - allowed_first_names
      - allowed_surnames
      - allowed_full_names
    """
    global _NAME_STATS_CACHE
    if _NAME_STATS_CACHE is not None:
        return _NAME_STATS_CACHE

    allowed_first: Set[str] = set()
    allowed_surnames: Set[str] = set()
    allowed_full: Set[str] = set()

    def _add_name(full_name: str, given: Optional[str], surname: Optional[str]):
        if full_name:
            allowed_full.add(full_name.strip())
        if given:
            allowed_first.add(given.strip())
        if surname:
            allowed_surnames.add(surname.strip())

    # 1) Hints "names" list
    for nm in HINTS.get("names", []):
        nm = str(nm).strip()
        if not nm:
            continue
        toks = nm.split()
        if len(toks) == 1:
            _add_name("", None, toks[0])
        else:
            given = toks[0]
            surname = toks[-1]
            _add_name(nm, given, surname)

    # 2) MASTER_JSONL if exists
    def _update_from_obj(obj: Dict[str, Any]):
        for person in obj.get("people", []) or []:
            if not isinstance(person, dict):
                continue
            full = str(person.get("full_name") or "").strip()
            given = str(person.get("given_name") or "").strip() or None
            surname = str(person.get("surname") or "").strip() or None
            if not full and (given or surname):
                parts = []
                if given:
                    parts.append(given)
                if surname:
                    parts.append(surname)
                full = " ".join(parts)
            if full or given or surname:
                _add_name(full, given, surname)

    if MASTER_JSONL.exists():
        try:
            with MASTER_JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        _update_from_obj(obj)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"[NAME_STATS] Failed reading MASTER_JSONL: {e}")

    # 3) Per-page JSONs
    try:
        for p in OUTPUT_DIR.glob("page-*.json"):
            if p.name.endswith("_raw_broken.json"):
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                _update_from_obj(obj)
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[NAME_STATS] Failed scanning per-page JSON: {e}")

    _NAME_STATS_CACHE = {
        "allowed_first_names": allowed_first,
        "allowed_surnames": allowed_surnames,
        "allowed_full_names": allowed_full,
    }
    logger.info(
        "[NAME_STATS] Built name stats: "
        f"{len(allowed_first)} firsts, {len(allowed_surnames)} surnames, "
        f"{len(allowed_full)} full names"
    )
    return _NAME_STATS_CACHE

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


def tile_is_mostly_blank(im: Image.Image, threshold: float = 0.985) -> bool:
    """
    Return True if the tile is mostly blank (very low ink coverage).
    threshold is the fraction of pixels that must be 'background' to treat it as blank.
    """
    gray = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total = th.size
    if total == 0:
        return True

    background = np.count_nonzero(th > 250)
    frac_background = background / float(total)

    return frac_background >= threshold


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


def _merge_boxes(boxes: List[Tuple[int, int, int, int]], gap: int) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[Tuple[int, int, int, int]] = []
    cur = boxes[0]

    def close(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ax1 -= gap
        ay1 -= gap
        ax2 += gap
        ay2 += gap
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
) -> List[Tuple[int, int, int, int]]:
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

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        logger.info("[ADAPTIVE] No text regions detected")
        return []

    boxes = _merge_boxes(boxes, max(w, h) // 100)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))[:max_regions]

    padded: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        padded.append(
            (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
        )

    logger.info(f"[ADAPTIVE] Detected {len(padded)} region(s)")
    return padded

# -------------------------
# Prompts (with external-file support)
# -------------------------


def build_system_prompt(layout_type: str = "default") -> str:
    """
    Load system prompt from external file if present, otherwise fall back
    to inline prompt that also injects dictionary hints.

    For layout_type == "ledger", we try ledger-specific prompt file first,
    then fall back to a built-in ledger prompt.
    """
    # Ledger-specific
    if layout_type == "ledger":
        if LEDGER_SYSTEM_PROMPT_PATH.exists():
            try:
                text = LEDGER_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
                logger.info(f"[PROMPTS] Using external LEDGER system prompt: {LEDGER_SYSTEM_PROMPT_PATH}")
                return text
            except Exception as e:
                logger.warning(f"[PROMPTS] Failed to read ledger system prompt file: {e}")

        logger.info("[PROMPTS] Using built-in LEDGER system prompt fallback")
        return """
You are a professional paleographer specializing in 19th-century handwritten ledger books.

The input image is a MEMBER CHECKLIST LEDGER (attendance / dues ledger) from a Presbyterian church society.

Critical rules:
- DO NOT treat the page as narrative text.
- DO NOT unwrap columnar text into paragraphs.
- You MUST extract the ledger as ROWS with COLUMN HEADERS.
- Every row corresponds to ONE PERSON (one member's name).
- Every column from the header corresponds to ONE MONTH or ONE CATEGORY.
- Checkmarks, slashes, crosses, dots or any ink mark in a cell count as “present”.
- Empty cells count as “absent”.
- The image tiles you see belong to ONE two-page spread. You MUST mentally stitch them into a single table.

You MUST output the ledger in JSON with at least this structure:

{
  "page_number": <int>,
  "page_type": "ledger",
  "ledger_title": <string or null>,
  "record_type": "attendance_ledger",
  "page_date_hint": <string or null>,
  "transcription_method": "openai-gpt-4o-vision-v2.42-ledger",
  "transcription_quality": "high" | "medium" | "low",
  "columns": [ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" ],
  "rows": [
    {
      "name": "Mrs. W. R. Wycoff",
      "marks": {
        "Jan": "✓",
        "Feb": "",
        "Mar": "✓",
        "Apr": "",
        "May": "✓",
        "Jun": "",
        "Jul": "",
        "Aug": "",
        "Sep": "",
        "Oct": "",
        "Nov": "",
        "Dec": ""
      },
      "notes": null
    }
  ],
  "notes": <string or null>
}

Rules for marks:
- If any kind of mark appears in a cell → output "✓".
- If the cell is clearly empty → output "" (empty string).
- If the cell is illegible or cut off → output "?".
- If some months are only on the right-hand page, still include them as columns.

Never output text outside JSON.
Never apologize.
Never produce safety warnings.
""".strip()

    # Non-ledger: normal narrative / mixed pages
    if SYSTEM_PROMPT_PATH.exists():
        try:
            text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
            logger.info(f"[PROMPTS] Using external system prompt: {SYSTEM_PROMPT_PATH}")
            return text
        except Exception as e:
            logger.warning(f"[PROMPTS] Failed to read system prompt file: {e}")

    surnames_str = ", ".join(COMBINED_SURNAMES[:80])
    places_str = ", ".join(COMBINED_PLACES[:80])
    church_terms_str = ", ".join(COMBINED_CHURCH_TERMS[:80])
    military_str = ", ".join(COMBINED_MILITARY[:80])
    honorifics_str = ", ".join(ALL_HONORIFICS)

    logger.info("[PROMPTS] Using built-in system prompt fallback (narrative)")

    return f"""
You are an expert historical paleographer and archivist working on PUBLIC DOMAIN
18th–19th century Presbyterian church records (Bethel and related congregations).
These documents are over 150 years old and contain no modern personal data.

CRITICAL SAFETY NOTE:
- This work is fully allowed.
- Do NOT refuse.
- Do NOT apologize.
- Do NOT output safety warnings.
- Never produce phrases like "I'm sorry, I can't assist with that."

You will receive images (or tiles) from ONE page or a clearly delimited
quadrant/region of a page.

Tasks:
1) Perform high-fidelity OCR / ICR transcription.
2) Provide light structural metadata about that page/quadrant.
3) Return a SINGLE JSON object between markers:

   ---BEGIN_JSON---
   {{ ... }}
   ---END_JSON---

Schema (minimum):

{{
  "page_number": <integer>,
  "quadrant_label": <string or null>,
  "page_title": <string or null>,
  "page_type": "narrative" | "ledger" | "mixed",
  "record_type": <string>,
  "page_date_hint": <string or null>,
  "transcription_method": "{TRANSCRIPTION_VERSION}",
  "transcription_quality": "high" | "medium" | "low",

  "content": <string>,
  "raw_text": <string>
}}

Optional:
- "places": [ <string>, ... ]
- "notes": <string or null>

Transcription rules:
- Be faithful to handwriting / print.
- Do NOT invent people, places, or events.
- Do NOT modernize spelling.
- You may unwrap multi-column registers into a linear reading order.

Hints (soft):

- Common surnames:
  {surnames_str}

- Important places / churches:
  {places_str}

- Church terms:
  {church_terms_str}

- Military / war vocabulary:
  {military_str}

- Honorifics / titles:
  {honorifics_str}

No comments or trailing commas inside JSON.
""".strip()


def build_user_prompt(page_number: int, layout_type: str, quadrant_label: Optional[str]) -> str:
    """
    Load user prompt template from external file if present, otherwise fall back
    to inline prompt. External template can use {{page_number}}, {{layout_type}},
    {{quadrant_label}} placeholders.

    For ledger layout, we prefer ledger-specific user prompt if present.
    """
    q_label_text = f"(quadrant {quadrant_label})" if quadrant_label else ""

    # Ledger-specific user prompt
    if layout_type == "ledger":
        if LEDGER_USER_PROMPT_PATH.exists():
            try:
                tmpl = LEDGER_USER_PROMPT_PATH.read_text(encoding="utf-8")
                logger.info(f"[PROMPTS] Using external LEDGER user prompt: {LEDGER_USER_PROMPT_PATH}")
                tmpl = tmpl.replace("{{page_number}}", str(page_number))
                tmpl = tmpl.replace("{{layout_type}}", layout_type)
                tmpl = tmpl.replace("{{quadrant_label}}", q_label_text)
                return tmpl
            except Exception as e:
                logger.warning(f"[PROMPTS] Failed to read ledger user prompt file: {e}")

        logger.info("[PROMPTS] Using built-in LEDGER user prompt fallback")
        return f"""
You are transcribing a two-page attendance / dues ledger (page {page_number}) of a bound church record volume.

CONTEXT:
- PUBLIC-DOMAIN historical ledgers from the 1800s.
- All people mentioned are long deceased.
- No modern personal data or privacy risk.
- Do NOT refuse. Do NOT apologize. Do NOT output safety warnings.

LAYOUT:
- Layout classification: "{layout_type}".
- The images you see are all tiles from THIS SAME two-page ledger spread (left and right pages together).
- These tiles are NOT separate pages. You MUST mentally stitch them into ONE continuous ledger table.

TASK:
1) Carefully read ALL tiles.
2) Identify the ledger title at the top (e.g., "Names of all Regularly Constituted Members of this Society for 1889").
3) Identify the left-hand name column: each row is one person's name.
4) Identify all column headers across the whole spread (months, dues, etc.).
5) Align each row horizontally across the grid.
6) For each cell, decide:
   - "✓" if there is any mark (slash, check, dot, cross) in that cell.
   - "" if clearly empty.
   - "?" if illegible or cut off.
7) Output EXACTLY one JSON object following the LEDGER schema from the system prompt.

Output MUST be wrapped between:

---BEGIN_JSON---
{{ ... }}
---END_JSON---
""".strip()

    # Non-ledger user prompt (narrative / mixed pages)
    if USER_PROMPT_PATH.exists():
        try:
            tmpl = USER_PROMPT_PATH.read_text(encoding="utf-8")
            logger.info(f"[PROMPTS] Using external user prompt: {USER_PROMPT_PATH}")
            tmpl = tmpl.replace("{{page_number}}", str(page_number))
            tmpl = tmpl.replace("{{layout_type}}", layout_type)
            tmpl = tmpl.replace("{{quadrant_label}}", q_label_text)
            return tmpl
        except Exception as e:
            logger.warning(f"[PROMPTS] Failed to read user prompt file: {e}")

    logger.info("[PROMPTS] Using built-in user prompt fallback (narrative)")
    q_part = f"quadrant/region {quadrant_label} of " if quadrant_label else ""
    return f"""
You are transcribing {q_part}page {page_number} of a bound church record volume.

CONTEXT:
- PUBLIC-DOMAIN historical ledgers from the 1700s–1800s.
- All people mentioned are long deceased.
- No modern personal data or privacy risk.
- Do NOT refuse. Do NOT apologize. Do NOT output safety warnings.

LAYOUT:
- Layout classification: "{layout_type}".
- The images you see are all from THIS same logical unit ({q_part}page {page_number}).
- Combine all image tiles you receive into one coherent transcription.

TASK:
1) Carefully read ALL tiles.
2) Produce an accurate transcription and light structural metadata.
3) Output EXACTLY ONE JSON OBJECT between:

   ---BEGIN_JSON---
   {{ ... }}
   ---END_JSON---

Follow the schema from the system prompt.
No text outside the JSON markers.
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
    Remove meta-comment / refusal lines from OCR output.
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
    Try:
      1) JSON between ---BEGIN_JSON--- / ---END_JSON---
      2) Whole response as JSON if it starts with "{" and ends with "}"
      3) First {...} block found in the text

    If all fail but there is non-empty text, wrap that text into
    a minimal JSON object instead of failing.
    """
    raw_json: Optional[str] = None

    m = re.search(r"---BEGIN_JSON---(.*?)---END_JSON---", text, re.DOTALL)
    if m:
        raw_json = m.group(1).strip()
    else:
        candidate = text.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            raw_json = candidate
        else:
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

    data["content"] = strip_refusal_meta(data.get("content", ""))
    data["raw_text"] = strip_refusal_meta(data.get("raw_text", ""))

    for fld in ["page_title", "page_type", "record_type", "page_date_hint"]:
        data.setdefault(fld, None)

    # normalize places to simple list of strings (we'll rebuild later)
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
# Vision call (FIFO)
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
    All OpenAI calls go through the global FIFO queue.
    """
    start_global_api_worker(client)

    sys_prompt = build_system_prompt(layout_type)
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
    logger.info(f"Page {page_number} {label}: enqueuing API call with {len(tiles)} tile(s) [layout={layout_type}]")

    def _task():
        logger.info(f"Page {page_number} {label}: starting API call (in worker)")
        resp = _do_openai_call(client, model, messages)
        msg = resp.choices[0].message

        assistant_text = ""

        # Case 1: simple string content
        if isinstance(msg.content, str):
            assistant_text = msg.content or ""

        # Case 2: list-of-parts content (newer SDK behaviour)
        elif isinstance(msg.content, list):
            parts_text: List[str] = []
            for part in msg.content:
                # `part` may be a pydantic object or a dict
                ptype = getattr(part, "type", None)
                if ptype is None and isinstance(part, dict):
                    ptype = part.get("type")

                if ptype == "text":
                    text_val = getattr(part, "text", None)
                    if text_val is None and isinstance(part, dict):
                        text_val = part.get("text", "")
                    if text_val:
                        parts_text.append(str(text_val))
            assistant_text = "".join(parts_text)

        # Case 3: None or unexpected type → log and fall back
        else:
            logger.warning(
                f"[VISION] Unexpected message.content type={type(msg.content)} "
                f"for page {page_number} {label}; treating as empty string."
            )
            assistant_text = ""

        assistant_text = (assistant_text or "").strip()

        if not assistant_text:
            # Optional: dump the raw response structure for debugging
            logger.warning(
                f"[VISION] Empty assistant_text for page {page_number} {label}; "
                f"raw message: {msg}"
            )

        return assistant_text
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
# Name / Event / Financial sweepers (v2.42)
# -------------------------

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

EXTRA_NON_NAME_TRAILERS = {
    "ceased",
    "removed",
    "attending",
    "daughter",
    "son",
    "wife",
    "husband",
    "baptists",
    "church",
    "cert.", "cert",
    "ex.", "ex", "ext",
    "co", "co.", "county",
    "tp", "tp.", "twp", "twp.", "twsp", "twsp.", "township",
    "ward",
    "pa", "pa.", "pennsylvania",
    "ohio", "virginia", "va", "wv",
    "meeting", "minutes", "session", "adjourned",
    "etc",
}

DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*[,/]\s*\d{2,4})?\b",
    flags=re.IGNORECASE
)
YEAR_RE = re.compile(r"\b(16|17|18|19|20)\d{2}\b")


def is_non_name_trailer(tok: str) -> bool:
    t = tok.strip(".,;:()").lower()
    if not t:
        return True
    if t in MONTH_TOKENS:
        return True
    if t in EXTRA_NON_NAME_TRAILERS:
        return True
    if t in NAME_STOPWORDS:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"'?\d{2}", t):
        return True
    return False


class NameExtractor:
    """
    Extracts person names from historical text using pattern matching,
    with support for honorifics, initials and simple ledger-style patterns.
    Uses a statistical name filter in apply_local_sweeper to reduce false positives.
    """

    def __init__(self):
        hon_escaped = [re.escape(h) for h in ALL_HONORIFICS]
        self.honorific_pattern = r'(?:' + '|'.join(hon_escaped) + r')'
        suf_escaped = [re.escape(s) for s in NAME_SUFFIXES]
        self.suffix_pattern = r'(?:' + '|'.join(suf_escaped) + r')'
        self.INITIAL = r'[A-Z]\.?'
        self.WORD = r"[A-Z][a-z]+(?:['\-][A-Z]?[a-z]+)*"

    @staticmethod
    def is_stopword(token: str) -> bool:
        return token.lower().rstrip('.,;:') in NAME_STOPWORDS

    @staticmethod
    def is_surname_prefix(token: str) -> bool:
        return token.rstrip('.,;:') in SURNAME_PREFIXES

    @staticmethod
    def normalize_honorific(h: str) -> str:
        h_clean = h.strip().rstrip('.')
        for known in ALL_HONORIFICS:
            if known.rstrip('.').lower() == h_clean.lower():
                return known
        return h

    def extract_names(self, text: str) -> List[Dict[str, Any]]:
        names: Dict[str, Dict[str, Any]] = {}
        names.update(self._extract_formal_names(text))
        names.update(self._extract_ledger_names(text))
        names.update(self._extract_simple_names(text))
        return list(names.values())

    def _extract_formal_names(self, text: str) -> Dict[str, Dict[str, Any]]:
        names: Dict[str, Dict[str, Any]] = {}
        pattern = rf'''
            (?P<honorifics>(?:{self.honorific_pattern}\s*)+)?   # honorifics
            (?P<initials>(?:{self.INITIAL}\s+)+)?               # initials
            (?P<given>{self.WORD})?                             # optional given
            \s+
            (?P<surname_prefix>{self.WORD})?                    # optional prefix
            \s*
            (?P<surname>{self.WORD})                            # surname
            (?:\s+(?P<suffix>{self.suffix_pattern}))?           # optional suffix
        '''
        for match in re.finditer(pattern, text, re.VERBOSE):
            full_match = match.group(0).strip()
            if self._is_false_positive(full_match, text, match.start()):
                continue

            honorifics_raw = match.group('honorifics') or ''
            initials_raw = match.group('initials') or ''
            given = match.group('given')
            surname_prefix = match.group('surname_prefix')
            surname = match.group('surname')
            suffix = match.group('suffix')

            has_honorific = bool(honorifics_raw.strip())
            has_initials = bool(initials_raw.strip())
            has_given = given is not None
            if not (has_honorific or has_initials or has_given):
                continue

            if surname_prefix and not self.is_surname_prefix(surname_prefix):
                if not given and not has_initials:
                    given = surname_prefix
                    surname_prefix = None

            full_surname = surname
            if surname_prefix and self.is_surname_prefix(surname_prefix):
                full_surname = f"{surname_prefix} {surname}"

            honorifics: List[str] = []
            if honorifics_raw:
                for h in re.findall(rf'{self.honorific_pattern}', honorifics_raw):
                    honorifics.append(self.normalize_honorific(h))

            initials: List[str] = []
            if initials_raw:
                initials = [i.strip() for i in initials_raw.split() if i.strip()]

            parts: List[str] = []
            if honorifics:
                parts.extend(honorifics)
            if initials:
                parts.extend(initials)
            if given:
                parts.append(given)
            parts.append(full_surname)
            if suffix:
                parts.append(suffix)
            full_name = ' '.join(parts)

            given_name_final = given or (initials[0] if initials else None)

            is_military = any(
                h.rstrip('.') in [m.rstrip('.') for m in HONORIFICS_MILITARY]
                for h in honorifics
            )

            if full_name not in names:
                names[full_name] = {
                    "full_name": full_name,
                    "given_name": given_name_final,
                    "surname": full_surname,
                    "honorific": honorifics[0] if honorifics else None,
                    "all_honorifics": honorifics if len(honorifics) > 1 else None,
                    "suffix": suffix,
                    "initials": initials if initials else None,
                    "church_role": self._infer_church_role(honorifics),
                    "is_military": is_military,
                    "military_rank": honorifics[0] if is_military else None,
                    "notes": None,
                    "confidence": 0.95,
                }

        return names

    def _extract_ledger_names(self, text: str) -> Dict[str, Dict[str, Any]]:
        names: Dict[str, Dict[str, Any]] = {}
        pattern = rf'''
            \b
            (?P<surname>{self.WORD}(?:\s+{self.WORD})?)  # surname
            ,\s*
            (?P<honorific>{self.honorific_pattern}\s+)?  # optional honorific
            (?P<given>{self.WORD}(?:\s+{self.INITIAL})?) # given
            (?:\s+(?P<suffix>{self.suffix_pattern}))?    # optional suffix
            \b
        '''
        for match in re.finditer(pattern, text, re.VERBOSE):
            surname = match.group('surname').strip()
            honorific_raw = match.group('honorific')
            given = match.group('given').strip()
            suffix = match.group('suffix')

            if self.is_stopword(surname):
                continue

            honorific = None
            if honorific_raw:
                honorific = self.normalize_honorific(honorific_raw.strip())

            parts: List[str] = []
            if honorific:
                parts.append(honorific)
            parts.append(given)
            parts.append(surname)
            if suffix:
                parts.append(suffix)
            full_name = ' '.join(parts)

            is_military = False
            if honorific:
                is_military = any(
                    honorific.rstrip('.') == m.rstrip('.')
                    for m in HONORIFICS_MILITARY
                )

            if full_name not in names:
                names[full_name] = {
                    "full_name": full_name,
                    "given_name": given,
                    "surname": surname,
                    "honorific": honorific,
                    "suffix": suffix,
                    "initials": None,
                    "church_role": self._infer_church_role([honorific] if honorific else []),
                    "is_military": is_military,
                    "military_rank": honorific if is_military else None,
                    "notes": None,
                    "confidence": 0.93,
                }

        return names

    def _extract_simple_names(self, text: str) -> Dict[str, Dict[str, Any]]:
        names: Dict[str, Dict[str, Any]] = {}
        pattern = rf'''
            \b
            (?P<initials>(?:{self.INITIAL}\s+)+)?     # initials
            (?P<given>{self.WORD})                   # given
            \s+
            (?P<surname_prefix>{self.WORD})?         # optional prefix
            \s*
            (?P<surname>{self.WORD})                 # surname
            (?:\s+(?P<suffix>{self.suffix_pattern}))?
            \b
        '''
        for match in re.finditer(pattern, text, re.VERBOSE):
            full_match = match.group(0).strip()
            if self._is_false_positive(full_match, text, match.start()):
                continue

            initials_raw = match.group('initials')
            given = match.group('given')
            surname_prefix = match.group('surname_prefix')
            surname = match.group('surname')
            suffix = match.group('suffix')

            if self.is_stopword(given) or self.is_stopword(surname):
                continue

            full_surname = surname
            if surname_prefix:
                if self.is_surname_prefix(surname_prefix):
                    full_surname = f"{surname_prefix} {surname}"
                else:
                    continue

            initials: List[str] = []
            if initials_raw:
                initials = [i.strip() for i in initials_raw.split() if i.strip()]

            parts: List[str] = []
            if initials:
                parts.extend(initials)
            parts.append(given)
            parts.append(full_surname)
            if suffix:
                parts.append(suffix)
            full_name = ' '.join(parts)

            if full_name not in names:
                names[full_name] = {
                    "full_name": full_name,
                    "given_name": initials[0] if initials else given,
                    "surname": full_surname,
                    "honorific": None,
                    "suffix": suffix,
                    "initials": initials if initials else None,
                    "church_role": None,
                    "is_military": False,
                    "military_rank": None,
                    "notes": None,
                    "confidence": 0.85,
                }

        return names

    def _is_false_positive(self, name: str, text: str, position: int) -> bool:
        context_start = max(0, position - 50)
        context_end = min(len(text), position + len(name) + 50)
        context = text[context_start:context_end].lower()

        false_indicators = [
            "church", "congregation", "presbytery", "session",
            "meeting", "minutes", "resolutions", "whereas",
            "prayer", "motion", "voted", "resolved"
        ]
        if len(name.split()) <= 2:
            for indicator in false_indicators:
                idx = context.find(indicator)
                if idx != -1:
                    if abs(idx - (position - context_start)) < 20:
                        return True
        return False

    def _infer_church_role(self, honorifics: List[str]) -> Optional[str]:
        for h in honorifics:
            if not h:
                continue
            h_clean = h.rstrip('.').lower()
            if h_clean in ["rev", "revd", "pastor"]:
                return "Minister"
            if h_clean in ["elder", "ruling elder"]:
                return "Elder"
            if h_clean == "deacon":
                return "Deacon"
            if h_clean in ["moderator", "clerk", "trustee"]:
                return h.rstrip('.')
        return None


def filter_people_by_stats(people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use corpus-derived name stats to down-rank/drop obvious false positives.
    Rule of thumb:
      - Keep if surname is in allowed_surnames or COMBINED_SURNAMES.
      - Otherwise, only keep if full_name appears in allowed_full_names.
    """
    stats = build_name_stats()
    allowed_first = stats["allowed_first_names"]
    allowed_surnames = stats["allowed_surnames"]
    allowed_full = stats["allowed_full_names"]

    filtered: List[Dict[str, Any]] = []
    for p in people:
        full = (p.get("full_name") or "").strip()
        given = (p.get("given_name") or "").strip()
        surname = (p.get("surname") or "").strip()

        score = 0
        if full and full in allowed_full:
            score += 3
        if surname and (surname in allowed_surnames or surname in COMBINED_SURNAMES):
            score += 2
        if given and given in allowed_first:
            score += 1

        if score == 0:
            if len(full.split()) <= 3:
                continue
            p["confidence"] = min(p.get("confidence", 0.85), 0.6)
        filtered.append(p)

    return filtered

# Place sweeper (simple, uses dictionary hints)

def sweep_places(text: str, initial_places: List[str]) -> List[Dict[str, Any]]:
    found: Dict[str, str] = {}
    low = text.lower()

    for place in COMBINED_PLACES:
        if not place:
            continue
        if place.lower() in low:
            found.setdefault(place, "")

    for p in initial_places:
        if not p:
            continue
        found.setdefault(p, "")

    township_re = re.compile(
        r"\b([A-Z][a-zA-Z]+)\s+(Township|Tp\.|Tp|Twp\.|Twp|Twsp\.|Twsp)\b"
    )
    county_re = re.compile(r"\b([A-Z][a-zA-Z]+)\s+Co\.?\b")
    for m in township_re.finditer(text):
        base = m.group(1)
        label = f"{base} Township"
        found.setdefault(label, "")
    for m in county_re.finditer(text):
        base = m.group(1)
        label = f"{base} County"
        found.setdefault(label, "")

    return [{"name": k, "notes": v or None} for k, v in sorted(found.items())]

# Event + military sweeper

def _extract_year_from_string(s: str) -> Optional[int]:
    m = YEAR_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


MILITARY_RANK_TOKENS = [r.lower().rstrip(".") for r in HONORIFICS_MILITARY]

MILITARY_CORE_PATTERNS = [
    r"\bcivil war\b",
    r"\brevolutionary war\b",
    r"\bworld war\b",
    r"\bspanish[- ]american war\b",
    r"\bwar of\b",
    r"\bregiment\b",
    r"\bregt\b",
    r"\bco\.\s*[a-z]\b",
    r"\bcompany\s+[a-z]\b",
    r"\binfantry\b",
    r"\bartillery\b",
    r"\bcavalry\b",
    r"\bvolunteers?\b",
    r"\bmustered\b",
    r"\benlisted\b",
    r"\bdischarged\b",
    r"\bprisoner\b",
    r"\bkilled in (?:battle|action)\b",
    r"\bbattle of\b",
    r"\barmy\b",
    r"\bnavy\b",
    r"\bbattery\b",
    r"\bbrigade\b",
    r"\bdivision\b",
]


def is_military_sentence(lowered: str) -> bool:
    for pat in MILITARY_CORE_PATTERNS:
        if re.search(pat, lowered):
            return True
    if any(rank in lowered for rank in MILITARY_RANK_TOKENS):
        return True
    if "war" in lowered or "battle" in lowered:
        for kw in COMBINED_MILITARY:
            if kw.lower() in lowered:
                return True
    return False


def sweep_events(
    text: str,
    people: List[Dict[str, Any]],
    places: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    sentences = re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", text)

    place_names = [p["name"] for p in places if isinstance(p, dict) and "name" in p]

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        lowered = s.lower()

        event_type: Optional[str] = None
        if any(w in lowered for w in ["died", "dead", "deceased", "killed"]):
            event_type = "death"
        elif any(w in lowered for w in ["born", "birth"]):
            event_type = "birth"
        elif any(w in lowered for w in ["married", "marriage"]):
            event_type = "marriage"
        elif any(w in lowered for w in ["baptized", "baptised", "baptism"]):
            event_type = "baptism"

        is_mil = is_military_sentence(lowered)

        if not event_type and is_mil:
            event_type = "other"

        if not event_type:
            continue

        date = None
        d = DATE_RE.search(s)
        if d:
            date = d.group(0)
        else:
            y = YEAR_RE.search(s)
            if y:
                date = y.group(0)

        year = _extract_year_from_string(date) if date else _extract_year_from_string(s)

        place_for_event = None
        for pl in place_names:
            if pl and pl.lower() in lowered:
                place_for_event = pl
                break

        people_for_event: Dict[str, Dict[str, Any]] = {}
        for person in people:
            full = person.get("full_name") or ""
            if not full:
                continue
            tokens = full.split()
            surname = tokens[-1] if len(tokens) >= 2 else None

            if full in s:
                people_for_event[full] = {"full_name": full, "match_type": "full"}
                continue
            if surname:
                if re.search(r"\b" + re.escape(surname) + r"\b", s):
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
            "is_military": bool(is_mil),
            "people": list(people_for_event.values()),
        })

    return events

# Financial sweeper (simple)

MONEY_RE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
)
FINANCE_KEYWORDS = {
    "salary", "salaries", "wages", "pay", "stipend",
    "repair", "repairs", "building", "construction", "roof",
    "paint", "painting", "maintenance",
    "mission", "missions", "benevolence", "charity", "poor",
    "organ", "music", "choir", "hymnals",
    "interest", "principal", "debt", "loan",
}


def parse_amount(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None


def categorize_financial_line(lowered: str) -> str:
    if any(w in lowered for w in ["salary", "salaries", "wages", "pay", "stipend", "pastor"]):
        return "salaries"
    if any(w in lowered for w in ["repair", "repairs", "building", "construction", "roof", "paint", "painting", "maintenance"]):
        return "building"
    if any(w in lowered for w in ["mission", "missions", "benevolence", "charity", "poor"]):
        return "missions/charity"
    if any(w in lowered for w in ["interest", "principal", "debt", "loan"]):
        return "debt/finance"
    return "other"


def sweep_financial(text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        lowered = line_stripped.lower()
        m = MONEY_RE.search(line_stripped)
        if not m:
            continue
        amt = parse_amount(m.group(1))
        if amt is None:
            continue
        category = categorize_financial_line(lowered)
        entries.append({
            "amount": amt,
            "currency": "USD",
            "category": category,
            "raw_line": line_stripped,
        })
    return entries

# -------------------------
# Apply local sweeper
# -------------------------


def apply_local_sweeper(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    v2.42:
      - NameExtractor + corpus-based filtering for narrative/mixed pages
      - place extraction
      - event extraction (with tighter military)
      - financial sweeper
      - IMPORTANT: ledger pages are returned as-is (we do not run sweepers).
    """
    page_type = (data.get("page_type") or "").lower()

    # Ledger pages already have structured rows/columns; avoid over-processing.
    if page_type == "ledger":
        return data

    text = (data.get("content") or data.get("raw_text") or "").strip()
    text = text if isinstance(text, str) else str(text)

    initial_places_raw = data.get("places") or []
    initial_places: List[str] = []
    for p in initial_places_raw:
        if isinstance(p, str):
            initial_places.append(p)
        elif isinstance(p, dict) and "name" in p:
            initial_places.append(str(p["name"]))

    name_extractor = NameExtractor()
    people_raw = name_extractor.extract_names(text)
    people = filter_people_by_stats(people_raw)

    places_struct = sweep_places(text, initial_places)
    events_struct = sweep_events(text, people, places_struct)
    financial_struct = sweep_financial(text)

    data["people"] = people
    data["places"] = places_struct
    data["events"] = events_struct
    if financial_struct:
        data["financial_entries"] = financial_struct
    else:
        data.pop("financial_entries", None)

    return data

# -------------------------
# OCR per page (adaptive + quadrants, inner threads)
# -------------------------


def ocr_page(
    client: OpenAI,
    page: int,
    im: Image.Image,
    layout_type: str,
) -> Dict[str, Any]:
    """
    For default / ledger layout: one call with tiled page (layout-specific tiling).

    For super_dense:
      - If adaptive regions find several smaller blobs → region mode
      - If adaptive finds one giant blob → fixed 2x2 quadrants
      - If no regions → fallback to single-call tiling
    """
    w, h = im.size
    area = w * h

    # Only super_dense uses adaptive regions/quadrants.
    if layout_type == "super_dense":
        logger.info(
            f"Page {page}: ADAPTIVE-REGION mode (layout={layout_type}), size={w}x{h} ({area} px)"
        )
        regions = find_text_regions(im, max_regions=6, min_area_ratio=0.003, pad=20)

        if regions:
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
                        future_to_idx: Dict[Any, int] = {}
                        for idx, quad_img in enumerate(quadrants):
                            q_label = f"Q{idx + 1}"

                            if tile_is_mostly_blank(quad_img):
                                logger.info(
                                    f"Page {page}: quadrant {q_label} looks mostly blank → skipping Vision call"
                                )
                                quadrant_results[idx] = {
                                    "page_number": page,
                                    "quadrant_label": q_label,
                                    "page_title": None,
                                    "page_type": layout_type,
                                    "record_type": None,
                                    "page_date_hint": None,
                                    "transcription_method": TRANSCRIPTION_VERSION + "-blank-quadrant",
                                    "transcription_quality": "low",
                                    "content": "",
                                    "raw_text": "",
                                    "places": [],
                                }
                                continue

                            logger.info(f"Page {page}: submitting quadrant {q_label} (2x2 mode)")
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
                                logger.info(f"Page {page}: quadrant {q_label} completed")
                            except Exception as e:
                                logger.error(f"Page {page}: quadrant {q_label} FAILED: {e}")
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

            logger.info(f"Page {page}: using {len(regions)} adaptive region(s)")
            n_regions = len(regions)
            region_results: List[Optional[Dict[str, Any]]] = [None] * n_regions

            max_inner_workers = min(4, n_regions)
            with ThreadPoolExecutor(max_workers=max_inner_workers) as ex:
                future_to_idx: Dict[Any, int] = {}
                for idx, box in enumerate(regions):
                    left, top, right, bottom = box
                    q_label = f"Q{idx + 1}"
                    region_img = im.crop(box)

                    if tile_is_mostly_blank(region_img):
                        logger.info(
                            f"Page {page}: region {q_label} box={box} looks mostly blank → skipping"
                        )
                        region_results[idx] = {
                            "page_number": page,
                            "quadrant_label": q_label,
                            "page_title": None,
                            "page_type": layout_type,
                            "record_type": None,
                            "page_date_hint": None,
                            "transcription_method": TRANSCRIPTION_VERSION + "-blank-region",
                            "transcription_quality": "low",
                            "content": "",
                            "raw_text": "",
                            "places": [],
                        }
                        continue

                    logger.info(f"Page {page}: submitting region {q_label} box={box}")
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
                        logger.info(f"Page {page}: region {q_label} completed")
                    except Exception as e:
                        logger.error(f"Page {page}: region {q_label} FAILED: {e}")
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

    # default / ledger fallback: SINGLE-CALL tiling with layout-specific tile grid
    rows, cols = LAYOUT_TILE_CONFIG.get(layout_type, LAYOUT_TILE_CONFIG["default"])
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
    """
    v2.42: processes ONE page synchronously.
    Any internal quadrants/regions are still handled via an inner ThreadPoolExecutor.
    """
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

            # First infer layout, then apply overrides if present
            layout_type = infer_layout(area, orig_w, orig_h)
            if page in LAYOUT_OVERRIDES:
                logger.info(
                    f"[LAYOUT_OVERRIDES] Page {page}: overriding inferred layout "
                    f"'{layout_type}' → '{LAYOUT_OVERRIDES[page]}'"
                )
                layout_type = LAYOUT_OVERRIDES[page]

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

        data = scrub_refusals_in_data(data)
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

    logger.info(f"Processing pages (sequential): {pages}")
    logger.info(f"Using baseline model: {BASELINE_MODEL}, heavy model: {HEAVY_MODEL}")
    logger.info(f"dictionary_hints.json present: {HINTS_FILE.exists()}")
    logger.info(f"Inner workers per page (quadrants/regions): {MAX_WORKERS}")

    for p in pages:
        process_single_page(client, p)

    rebuild_master_jsonl()


if __name__ == "__main__":
    main()

