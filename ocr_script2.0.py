#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bethel OCR / ICR Pipeline
Version: 2.10

Features:
- Smart downscale BEFORE tiling (Option 1: Conservative)
- Vision OCR via OpenAI (gpt-4o-mini / gpt-4.1 auto-selection)
- People Sweeper with military heuristics
- dictionary_hints.json integration (names, places, church_terms, military_keywords)
- Tiled image processing: ledger + super_dense → 2×8 tiles
- JSON extraction via ---BEGIN_JSON--- / ---END_JSON---
- Full record schema: places, people, events, raw_text, content, etc.
- Master JSONL rebuild (ocr_all_pages.jsonl)
"""

import os
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI


# ===========================
# Logging
# ===========================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ===========================
# Paths & Constants
# ===========================

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"
OUTPUT_DIR = BASE_DIR / "ocr_output"
OUTPUT_DIR.mkdir(exist_ok=True)
MASTER_JSONL = BASE_DIR / "ocr_all_pages.jsonl"

HINTS_FILE = BASE_DIR / "dictionary_hints.json"
LAYOUT_OVERRIDES_PATH = BASE_DIR / "page_layout_overrides.json"

BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
HEAVY_MODEL   = os.getenv("HEAVY_MODEL",   "gpt-4.1")

MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "2"))
RETRY_SLEEP_SECONDS = float(os.getenv("OCR_RETRY_SLEEP", "4"))

PAGE_LIST_ENV = os.getenv("PAGE_LIST", "").strip()

TRANSCRIPTION_VERSION = "openai-gpt-4o-vision-v2.10"

# Tiling rules
LAYOUT_TILE_CONFIG = {
    "default": (2, 3),
    "ledger": (2, 8),
    "super_dense": (2, 8),
}

# Conservative downscale (Option 1)
MAX_DIM_THRESHOLD = 8000     # If max dimension > 8000 px -> downscale
TARGET_MAX_DIM    = 6000     # New max dimension after downscale


# ===========================
# Built-in Hints (fallback)
# ===========================

DEFAULT_SURNAMES = [
    "Marshall","Miller","McMillan","Millar","Clark","Calhoun",
    "Fleming","Sample","Dimmon","Dimmond","Fairchild",
    "Elliott","Elliot","Hill","Kelly","Smith","Jones",
    "Hutchison","Hutchinson","McFarland","Anderson"
]

DEFAULT_PLACES = [
    "Bethel","Bethel Church","Bethel Cemetery","Pittsburgh",
    "Allegheny","Minisink","Deckertown","Petersburgh",
    "Wilderness","Antietam","Gettysburg","Virginia",
    "Augusta Stone Church","Westmoreland","Washington County",
    "Squirrel Hill","Upper St. Clair"
]

HONORIFICS_RELIGIOUS = [
    "Rev.","Revd.","Pastor","Elder","Ruling Elder",
    "Deacon","Moderator","Clerk","Trustee"
]

HONORIFICS_CIVIC = ["Dr.","Esq.","Hon.","Judge"]

HONORIFICS_MILITARY = [
    "Col.","Colonel","Capt.","Captain","Lieut.","Lt.",
    "Sgt.","Sergeant","Corp.","Corporal","Pvt.","Private",
    "Gen.","General"
]

DEFAULT_MILITARY_KEYWORDS = [
    "Civil War","war","regiment","company","Co.","killed","wounded",
    "died","battle","camp","Petersburgh","Petersburg",
    "Wilderness","Gettysburg","Antietam","Chancellorsville"
]


# ===========================
# Load dictionary_hints.json
# ===========================

def load_hints() -> Dict[str, Any]:
    if not HINTS_FILE.exists():
        logger.warning("[HINTS] No dictionary_hints.json found; using defaults only")
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
COMBINED_PLACES   = combined_list(DEFAULT_PLACES,   HINTS.get("places"))
COMBINED_CHURCH   = combined_list([],               HINTS.get("church_terms"))
COMBINED_MILITARY = combined_list(DEFAULT_MILITARY_KEYWORDS, HINTS.get("military_keywords"))

ALL_HONORIFICS = HONORIFICS_RELIGIOUS + HONORIFICS_CIVIC + HONORIFICS_MILITARY


# ===========================
# Page List / Layout
# ===========================

def parse_page_list() -> List[int]:
    if PAGE_LIST_ENV:
        out = []
        for t in PAGE_LIST_ENV.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except:
                logger.warning(f"[PAGE_LIST] Invalid page number: {t}")
        return sorted(set(out))

    pages = []
    for p in PAGES_DIR.glob("*.png"):
        try:
            pages.append(int(p.stem))
        except:
            pass
    return sorted(pages)


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
    except:
        return "default"


# ===========================
# Image helpers (tiling + downscale)
# ===========================

def smart_downscale(im: Image.Image) -> Image.Image:
    """Conservative downscale before tiling (Option 1)."""
    w, h = im.size
    max_dim = max(w, h)

    if max_dim <= MAX_DIM_THRESHOLD:
        logger.info(f"Downscale: skipped (max_dim={max_dim} ≤ {MAX_DIM_THRESHOLD})")
        return im

    scale = TARGET_MAX_DIM / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    logger.info(
        f"Downscale: {w}×{h} → {new_w}×{new_h} (scale={scale:.3f}) "
        f"[reason: max_dim {max_dim} > {MAX_DIM_THRESHOLD}]"
    )

    return im.resize((new_w, new_h), Image.LANCZOS)


def tile_image(im: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    w, h = im.size
    tw = w // cols
    th = h // rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (
                c * tw,
                r * th,
                w if c == cols - 1 else (c + 1) * tw,
                h if r == rows - 1 else (r + 1) * th,
            )
            tiles.append(im.crop(box))
    return tiles


def image_to_data_url(im: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ===========================
# Prompts
# ===========================

def build_system_prompt() -> str:
    honorifics_str = ", ".join(ALL_HONORIFICS)
    surnames_str   = ", ".join(COMBINED_SURNAMES[:80])
    places_str     = ", ".join(COMBINED_PLACES[:80])
    church_str     = ", ".join(COMBINED_CHURCH[:80])
    military_str   = ", ".join(COMBINED_MILITARY[:80])

    return f"""
You are an expert historical paleographer working with 19th-century Presbyterian
church records. You receive TILED image segments of ONE page and must produce a
structured JSON record.

YOU MUST OUTPUT:

1) (Optional) free-text transcription
2) EXACTLY ONE JSON OBJECT inside:

---BEGIN_JSON---
{{ ... }}
---END_JSON---

JSON SCHEMA:

{{
  "page_number": <int>,
  "page_title": <string or null>,
  "page_type": "narrative"|"ledger"|"mixed",
  "record_type": <string or null>,
  "page_date_hint": <string or null>,
  "transcription_method": "{TRANSCRIPTION_VERSION}",
  "transcription_quality": "high"|"medium"|"low",
  "content": <string>,
  "raw_text": <string>,
  "places": [ {{ "name":..., "notes":... }} ],
  "people": [
    {{
      "full_name":...,
      "given_name":...,
      "surname":...,
      "church_role":...,
      "honorific":...,
      "is_military":...,
      "military_rank":...,
      "notes":...,
      "confidence":...
    }}
  ],
  "events": [
    {{
      "event_type":...,
      "date":...,
      "place":...,
      "raw_entry":...,
      "normalized_summary":...,
      "people":[...]
    }}
  ]
}}

PEOPLE SWEEPER:
- Collect EVERY plausible personal name.
- Strong signals: honorifics ({honorifics_str}), dictionary surnames ({surnames_str}),
  roles: Pastor, Elder, Ruling Elder, Deacon, Trustee, Clerk, Treasurer, Moderator.

MILITARY HEURISTICS:
- Ranks → is_military=true, assign normalized military_rank.
- Military words: {military_str}

PLACE DETECTION:
- Known places: {places_str}
- Church terms: {church_str}

CONSTRAINTS:
- JSON ONLY between markers.
- No trailing commas.
- No invented text.
""".strip()


def build_user_prompt(page: int, layout: str) -> str:
    return f"""
You are transcribing and structuring page {page}.

LAYOUT: "{layout}"
- You will receive multiple TILES (crops) of the SAME page.
- Combine all tiles in reading order.
- Produce transcription + JSON.

JSON MUST be between ---BEGIN_JSON--- and ---END_JSON---
and follow schema exactly.
""".strip()


# ===========================
# OpenAI Client
# ===========================

def get_client() -> OpenAI:
    load_dotenv()
    client = OpenAI()
    if not client.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return client


# ===========================
# JSON Extraction
# ===========================

def extract_json_from_markers(text: str, page: int) -> Dict[str, Any]:
    m = re.search(r"---BEGIN_JSON---(.*?)---END_JSON---", text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    else:
        raw = text.strip()
        if not (raw.startswith("{") and raw.endswith("}")):
            raise ValueError("No JSON block found")

    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"JSON parse error: {e}")

    data.setdefault("page_number", page)
    data.setdefault("transcription_method", TRANSCRIPTION_VERSION)
    data.setdefault("transcription_quality", "medium")
    data.setdefault("content", data.get("raw_text", ""))
    data.setdefault("raw_text", data.get("content", ""))

    for f in ["page_title","record_type","page_type","page_date_hint"]:
        data.setdefault(f, None)

    for f in ["places","people","events"]:
        data.setdefault(f, [])

    return data


# ===========================
# LLM Call
# ===========================

def call_vision_llm(client: OpenAI, page: int, im: Image.Image, layout: str) -> Dict[str, Any]:
    rows, cols = LAYOUT_TILE_CONFIG.get(layout, LAYOUT_TILE_CONFIG["default"])
    w, h = im.size

    # Downscale BEFORE tiling
    im = smart_downscale(im)
    w2, h2 = im.size

    tiles = tile_image(im, rows, cols)
    logger.info(f"Page {page}: tiling={rows}x{cols}, final_size={w2}x{h2}")

    model = HEAVY_MODEL if layout == "super_dense" else BASELINE_MODEL
    logger.info(f"Page {page}: using model={model}")

    sys_prompt = build_system_prompt()
    user_prompt = build_user_prompt(page, layout)

    msg_content = [{"type": "text", "text": user_prompt}]
    for t in tiles:
        msg_content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(t),
                "detail": "high",
            }
        })

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Page {page}: LLM attempt {attempt}/{MAX_RETRIES}")
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": msg_content},
                ]
            )

            msg = resp.choices[0].message
            if isinstance(msg.content, str):
                text = msg.content
            else:
                text = "".join(
                    part.text for part in msg.content if part.type == "text"
                )

            if not text.strip():
                raise ValueError("Empty assistant_text")

            return extract_json_from_markers(text, page)

        except Exception as e:
            logger.warning(f"Page {page}: attempt {attempt} failed: {e}")
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")


# ===========================
# Output Helpers
# ===========================

def save_page_json(page: int, data: Dict[str, Any]) -> Path:
    out = OUTPUT_DIR / f"page-{page}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def save_raw_broken(page: int, raw: str, err: Exception) -> Path:
    out = OUTPUT_DIR / f"page-{page}_raw_broken.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "page_number": page,
                "error": str(err),
                "raw_response_text": raw,
            }, f, ensure_ascii=False, indent=2
        )
    return out


def rebuild_master_jsonl():
    logger.info(f"[REBUILD] Rebuilding {MASTER_JSONL}")
    files = sorted(
        p for p in OUTPUT_DIR.glob("page-*.json")
        if not p.name.endswith("_raw_broken.json")
    )
    with MASTER_JSONL.open("w", encoding="utf-8") as out:
        for p in files:
            try:
                obj = json.load(p.open("r", encoding="utf-8"))
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"[REBUILD] Failed for {p}: {e}")
    logger.info(f"[REBUILD] Wrote {len(files)} pages.")


# ===========================
# Main
# ===========================

def main():
    client = get_client()
    pages = parse_page_list()
    if not pages:
        logger.error("No pages to process.")
        return

    layout_overrides = {}
    if LAYOUT_OVERRIDES_PATH.exists():
        try:
            layout_overrides = json.load(open(LAYOUT_OVERRIDES_PATH, "r"))
        except:
            pass

    logger.info(f"Processing pages: {pages}")
    logger.info(f"Using baseline={BASELINE_MODEL}, heavy={HEAVY_MODEL}")
    logger.info(f"Hints file present: {HINTS_FILE.exists()}")

    for page in pages:
        try:
            logger.info(f"=== Page {page} ===")

            # Locate image
            candidates = [
                PAGES_DIR / f"{page}.png",
                PAGES_DIR / f"page-{page}.png",
                PAGES_DIR / f"{page}.jpg",
                PAGES_DIR / f"{page}.jpeg",
            ]
            img_path = None
            for c in candidates:
                if c.exists():
                    img_path = c
                    break
            if not img_path:
                raise FileNotFoundError(f"No image for page {page}")

            with Image.open(img_path) as im:
                im.load()
                w, h = im.size

                layout = infer_layout(page, layout_overrides)
                logger.info(f"Page {page} → layout={layout}, size={w}x{h} ({w*h} px)")

                data = call_vision_llm(client, page, im, layout)

            save_page_json(page, data)
            logger.info(f"Page {page}: saved OK")

        except Exception as e:
            save_raw_broken(page, str(e), e)
            logger.error(f"Page {page}: FAILED — {e}")

    rebuild_master_jsonl()


if __name__ == "__main__":
    main()
