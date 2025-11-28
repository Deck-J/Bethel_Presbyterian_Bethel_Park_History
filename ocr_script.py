#!/usr/bin/env python3
"""
Bethel Register OCR Script (batched / GitHub-Actions-friendly)

- Reads configuration from environment variables (and .env if present)
- Supports page ranges via START_PAGE / END_PAGE (including "all")
- Designed to work with:
    pages/page-1.png
    pages/page-2.png
    ...
- Can auto-tile dense pages into vertical strips
- Logs per-page complexity and behavior
- Produces a single JSONL file (ocr_all_pages.jsonl) plus per-page JSONs
- Supports DRY_RUN to skip OpenAI calls and emit stub JSON for testing
"""

import os
import sys
import re
import json
import math
import time
import csv
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageStat

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# -------------------------------------------------------------------
# Environment & configuration
# -------------------------------------------------------------------

# Increase PIL's safety limit; we're dealing with very large PNGs
Image.MAX_IMAGE_PIXELS = None

load_dotenv()  # local only; GitHub Actions uses env directly

# Directories / files
PAGES_DIR = Path(os.getenv("PAGES_DIR", "pages"))
OUTPUT_JSONL = Path(os.getenv("OUTPUT_JSONL", "ocr_all_pages.jsonl"))
PER_PAGE_DIR = Path(os.getenv("PER_PAGE_DIR", "per_page"))
OCR_OUTPUT_DIR = Path(os.getenv("OCR_OUTPUT_DIR", "ocr_output"))

PER_PAGE_DIR.mkdir(parents=True, exist_ok=True)
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")

# Concurrency
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Strips / tiling config
USE_STRIPS = int(os.getenv("USE_STRIPS", "0"))          # 0 = full page, 1 = force strips
AUTO_STRIP_MODE = int(os.getenv("AUTO_STRIP_MODE", "1"))  # 1 = auto-detect dense pages

STRIPS_PER_PAGE = int(os.getenv("STRIPS_PER_PAGE", "6"))
MAX_STRIPS_PER_PAGE = int(os.getenv("MAX_STRIPS_PER_PAGE", "8"))
STRIP_OVERLAP = int(os.getenv("STRIP_OVERLAP", "20"))

# Density thresholds
EASY_INK_THRESHOLD = float(os.getenv("EASY_INK_THRESHOLD", "0.05"))
DENSE_INK_THRESHOLD = float(os.getenv("DENSE_INK_THRESHOLD", "0.12"))
SUPER_DENSE_INK_THRESHOLD = float(os.getenv("SUPER_DENSE_INK_THRESHOLD", "0.18"))

# Downscaling limits
MAX_LONG_DIM = int(os.getenv("MAX_LONG_DIM", "2048"))
MAX_TOTAL_PIXELS = int(os.getenv("MAX_TOTAL_PIXELS", "20000000"))

# Complexity log
COMPLEXITY_LOG = Path(os.getenv("COMPLEXITY_LOG", "page_complexity.csv"))

# Page range
START_PAGE_ENV = os.getenv("START_PAGE")
END_PAGE_ENV = os.getenv("END_PAGE")

# Dry-run mode
DRY_RUN_ENV = os.getenv("DRY_RUN", "0").strip()
DRY_RUN = DRY_RUN_ENV.lower() in ("1", "true", "yes", "y", "on")


def _parse_page_env(value: Optional[str]) -> Optional[int]:
    """
    Parse START_PAGE / END_PAGE environment values.

    Accepted:
      - empty or None         -> None (no limit)
      - "all" (any case)      -> None (no limit)
      - integer string        -> int
    Anything else will be treated as 'no limit' with a warning.
    """
    if value is None:
        return None
    value_str = value.strip()
    if value_str == "":
        return None
    if value_str.lower() == "all":
        return None
    try:
        return int(value_str)
    except ValueError:
        print(f"[WARN] Could not parse page value '{value_str}', treating as 'all'.")
        return None


START_PAGE: Optional[int] = _parse_page_env(START_PAGE_ENV)
END_PAGE: Optional[int] = _parse_page_env(END_PAGE_ENV)

# ---------------------------------------
# LOGGING: Report Final START/END Pages
# ---------------------------------------


def _format_page_val(val: Optional[int]) -> str:
    if val is None:
        return "None (no limit)"
    try:
        return str(int(val))
    except Exception:
        return f"{val} (raw)"


print("==============================================")
print(" OCR PAGE RANGE CONFIGURATION")
print("==============================================")
print(f"  START_PAGE (raw env) : {START_PAGE_ENV!r}")
print(f"  END_PAGE   (raw env) : {END_PAGE_ENV!r}")
print(f"  → START_PAGE parsed  : {_format_page_val(START_PAGE)}")
print(f"  → END_PAGE parsed    : {_format_page_val(END_PAGE)}")
print(f"  DRY_RUN              : {DRY_RUN} (raw={DRY_RUN_ENV!r})")

if START_PAGE is None and END_PAGE is None:
    print("  → Processing ALL pages (local mode or unbounded run).")
elif START_PAGE is not None and END_PAGE is not None:
    print(f"  → Processing pages {START_PAGE} through {END_PAGE}.")
elif START_PAGE is not None:
    print(f"  → Processing pages {START_PAGE} through end-of-list.")
elif END_PAGE is not None:
    print(f"  → Processing pages 1 through {END_PAGE}.")
print("==============================================\n")

# -------------------------------------------------------------------
# OpenAI client (only used if not DRY_RUN)
# -------------------------------------------------------------------

client = None
if not DRY_RUN:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    print("[INFO] DRY_RUN enabled: OpenAI client will NOT be created.\n")

# -------------------------------------------------------------------
# Dictionary hints (names, roles, places, fuzzy variants)
# -------------------------------------------------------------------

DICTIONARY_HINTS_PATH = Path("dictionary_hints.json")


def load_dictionary_hints() -> Dict[str, Any]:
    if not DICTIONARY_HINTS_PATH.exists():
        print("[INFO] No dictionary_hints.json found; proceeding without extra hints.")
        return {}
    try:
        with DICTIONARY_HINTS_PATH.open("r", encoding="utf-8") as f:
            hints = json.load(f)
        print("[INFO] Loaded dictionary_hints.json.")
        return hints
    except Exception as e:
        print(f"[WARN] Failed to load dictionary_hints.json: {e}")
        return {}


DICTIONARY_HINTS = load_dictionary_hints()

# -------------------------------------------------------------------
# Utility: page number extraction
# -------------------------------------------------------------------


def extract_page_number(path: Path) -> int:
    """
    Try to infer page number from filename.

    Expected primary pattern: pages/page-<N>.png
    But we also support legacy patterns like:
      <N>-something.png
    If no match, return -1.
    """
    name = path.name

    # New preferred pattern
    m = re.match(r"^page-(\d+)\.png$", name)
    if m:
        return int(m.group(1))

    # Legacy pattern: leading digits, then dash
    m2 = re.match(r"^(\d+)-", name)
    if m2:
        return int(m2.group(1))

    # Fallback: any digits
    m3 = re.search(r"(\d+)", name)
    if m3:
        return int(m3.group(1))

    return -1


# -------------------------------------------------------------------
# Utility: image complexity measurement
# -------------------------------------------------------------------


def compute_ink_density(im: Image.Image) -> float:
    """
    Rough "ink density" measure: convert to grayscale, normalize,
    and estimate non-background pixels.
    """
    gray = im.convert("L")
    stat = ImageStat.Stat(gray)
    mean = stat.mean[0]  # 0-255
    ink = (255.0 - mean) / 255.0
    return float(max(0.0, min(1.0, ink)))


def resize_for_ocr(im: Image.Image) -> Image.Image:
    """
    Downscale image to stay under MAX_LONG_DIM and MAX_TOTAL_PIXELS.
    Preserves aspect ratio.
    """
    w, h = im.size
    total_pixels = w * h

    # Enforce total pixels limit
    if total_pixels > MAX_TOTAL_PIXELS:
        scale = math.sqrt(MAX_TOTAL_PIXELS / float(total_pixels))
        w = int(w * scale)
        h = int(h * scale)
        im = im.resize((w, h), Image.LANCZOS)
        total_pixels = w * h

    # Enforce max long dimension
    long_dim = max(w, h)
    if long_dim > MAX_LONG_DIM:
        scale = MAX_LONG_DIM / float(long_dim)
        w = int(w * scale)
        h = int(h * scale)
        im = im.resize((w, h), Image.LANCZOS)

    return im


# -------------------------------------------------------------------
# Strip tiling
# -------------------------------------------------------------------


def compute_strip_boxes(width: int, height: int, num_strips: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """
    Compute vertical strip bounding boxes (left, upper, right, lower).
    Overlap is applied horizontally between strips.
    """
    if num_strips <= 1:
        return [(0, 0, width, height)]

    effective_width = width + overlap * (num_strips - 1)
    step = effective_width // num_strips

    boxes = []
    for i in range(num_strips):
        left = i * step - i * overlap
        right = left + step
        left = max(0, left)
        right = min(width, right)
        if right > left:
            boxes.append((left, 0, right, height))
    return boxes


# -------------------------------------------------------------------
# Prompt construction
# -------------------------------------------------------------------


def build_system_prompt(hints: Dict[str, Any]) -> str:
    names = hints.get("names", [])
    church_roles = hints.get("church_roles", [])
    places = hints.get("places", [])
    military_terms = hints.get("military_terms", [])

    parts = [
        "You are an expert historical OCR and transcription assistant specializing in 18th–20th century Presbyterian church registers.",
        "You must:",
        "  - Transcribe the page as faithfully as possible.",
        "  - Normalize a second version with corrected spelling and punctuation where reasonably clear.",
        "  - Identify personal names, roles, places, dates, and military associations (e.g., Civil War, Revolutionary War).",
        "  - Recognize ledger structures and mark empty ledger lines if there are brackets or ruled lines with no text.",
        "  - It is OK to skip empty ledger brackets that have no text, BUT if a ledger line is clearly meant to be present and is empty, include a marker such as '[empty ledger line]'.",
        "  - Do NOT hallucinate text—only complete gaps where context makes the words highly likely based on surrounding handwriting.",
        "",
        "Output must be JSON with keys:",
        "  - page_number (if visible or inferred)",
        "  - page_title (short description)",
        "  - content (cleaned/structured transcription, Markdown allowed)",
        "  - raw_text (rough/near-verbatim transcription)",
        "  - extracted_names (list of names as strings)",
        "  - extracted_dates (list of objects with raw/month/day/year/type)",
        "  - identified_roles (map name -> roles in church, e.g. 'Elder', 'Deacon', 'Pastor')",
        "  - tags (list of strings)",
        "  - military_service (object with fields: has_military_association, conflict, service_side, service_status, evidence_level, evidence_snippets, auto_inferred, confidence, review_status)",
        "  - page_date (an object describing date or range of the page if inferable)",
        "  - page_locations (list of places mentioned or implied).",
        "",
        "If there is clearly a Civil War or Revolutionary War soldier death, explicitly include that in military_service.",
        "If there is no military content, set has_military_association=false.",
        "",
        "Use conservative scholarly guessing on partially illegible words, marking uncertain text in square brackets like '[?]'.",
    ]

    if names:
        parts.append("")
        parts.append("Known frequent personal names to prioritize in matching (do not force them if handwriting clearly differs):")
        parts.append(", ".join(sorted(set(names))))

    if church_roles:
        parts.append("")
        parts.append("Common church roles:")
        parts.append(", ".join(sorted(set(church_roles))))

    if places:
        parts.append("")
        parts.append("Common places/locations that may appear:")
        parts.append(", ".join(sorted(set(places))))

    if military_terms:
        parts.append("")
        parts.append("Common military-related terms to watch for:")
        parts.append(", ".join(sorted(set(military_terms))))

    return "\n".join(parts)


SYSTEM_PROMPT = build_system_prompt(DICTIONARY_HINTS)

# -------------------------------------------------------------------
# OpenAI helper
# -------------------------------------------------------------------


def ocr_with_openai(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Send a single page (or strip) to the vision model and get structured JSON back.
    """
    import base64

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt_user = (
        "Transcribe this page (or page segment) from a historic handwritten Presbyterian church register. "
        "Return ONLY a single JSON object, no extra commentary."
    )

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        },
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    content = resp.choices[0].message.content

    # Try to parse JSON out of the content
    try:
        json_str = content.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(json)?", "", json_str.strip(), flags=re.IGNORECASE)
            json_str = re.sub(r"```$", "", json_str.strip())
        data = json.loads(json_str)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON for {filename}: {e}")
        data = {
            "raw_response": content,
            "parse_error": str(e),
        }

    return data


# -------------------------------------------------------------------
# Defaults / helpers
# -------------------------------------------------------------------


def default_military_service() -> Dict[str, Any]:
    return {
        "has_military_association": False,
        "conflict": None,
        "service_side": None,
        "service_status": None,
        "evidence_level": "none",
        "evidence_snippets": [],
        "auto_inferred": False,
        "confidence": 0.0,
        "review_status": "unknown",
    }


def _image_to_bytes(im: Image.Image) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------------------------------------------
# Per-page processing
# -------------------------------------------------------------------


def analyze_page_complexity(img_path: Path) -> Dict[str, Any]:
    """
    Load the image (possibly huge), downscale for complexity measurement,
    and decide how many strips (if any) to use.
    """
    try:
        im = Image.open(img_path)
    except Exception as e:
        print(f"[ERROR] Unable to open image {img_path}: {e}")
        return {
            "path": str(img_path),
            "failed_to_open": True,
            "ink_density": None,
            "strips_planned": 1,
            "mode": "failed-open",
        }

    w, h = im.size
    im_small = resize_for_ocr(im)
    ink = compute_ink_density(im_small)

    strips = 1
    mode = "full-page"

    if USE_STRIPS == 1:
        strips = STRIPS_PER_PAGE
        mode = "force-strips"
    elif AUTO_STRIP_MODE == 1:
        if ink < EASY_INK_THRESHOLD:
            strips = 1
            mode = "easy-full-page"
        elif ink < DENSE_INK_THRESHOLD:
            strips = 1
            mode = "medium-full-page"
        elif ink < SUPER_DENSE_INK_THRESHOLD:
            strips = STRIPS_PER_PAGE
            mode = "dense-strips"
        else:
            strips = MAX_STRIPS_PER_PAGE
            mode = "super-dense-max-strips"
    else:
        strips = 1
        mode = "auto-disabled-full-page"

    return {
        "path": str(img_path),
        "width": w,
        "height": h,
        "ink_density": ink,
        "strips_planned": strips,
        "mode": mode,
    }


def write_complexity_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """
    Append a row to the complexity CSV. Creates file with header if needed.
    """
    file_exists = csv_path.exists()
    fieldnames = [
        "path",
        "page_number",
        "width",
        "height",
        "ink_density",
        "strips_planned",
        "mode",
        "error",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def merge_military_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple military_service blocks from strips into a single page-level block.
    """
    if not blocks:
        return default_military_service()

    any_military = any(b.get("has_military_association") for b in blocks)
    if not any_military:
        return default_military_service()

    def conf(b: Dict[str, Any]) -> float:
        try:
            return float(b.get("confidence", 0.0))
        except Exception:
            return 0.0

    best = max(blocks, key=conf)

    all_snips: List[str] = []
    for b in blocks:
        sn = b.get("evidence_snippets") or []
        for s in sn:
            if s not in all_snips:
                all_snips.append(s)

    merged = dict(best)
    merged["evidence_snippets"] = all_snips
    merged["has_military_association"] = True
    if "review_status" not in merged:
        merged["review_status"] = "auto-merged"
    return merged


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    Append one JSON object as a single line to the master JSONL.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def process_page(img_path: Path) -> Optional[Dict[str, Any]]:
    """
    Process a single page:
      - analyze complexity
      - if DRY_RUN: create stub JSON only
      - else: OCR full-page or strips
      - write per-page JSON
      - append to master JSONL
    """
    page_number = extract_page_number(img_path)
    print(f"[OCR] Processing page {page_number} — {img_path}")

    complexity = analyze_page_complexity(img_path)
    complexity_row = {
        "path": complexity.get("path"),
        "page_number": page_number,
        "width": complexity.get("width"),
        "height": complexity.get("height"),
        "ink_density": complexity.get("ink_density"),
        "strips_planned": complexity.get("strips_planned"),
        "mode": complexity.get("mode"),
        "error": None,
    }

    strips_planned = complexity.get("strips_planned", 1)
    mode = complexity.get("mode", "unknown")

    try:
        if DRY_RUN:
            # In dry-run mode, DO NOT call OpenAI.
            result = {
                "page_number": page_number,
                "page_title": f"Page {page_number}",
                "content": "",
                "raw_text": "",
                "extracted_names": [],
                "extracted_dates": [],
                "identified_roles": {},
                "tags": [],
                "military_service": default_military_service(),
                "page_date": None,
                "page_locations": [],
                "strip_mode": mode,
                "strips_used": strips_planned,
                "dry_run": True,
            }
        else:
            im = Image.open(img_path)
            im = resize_for_ocr(im)  # size-limited version for OCR
            w, h = im.size

            if strips_planned <= 1:
                buf = _image_to_bytes(im)
                result = ocr_with_openai(buf, img_path.name)
            else:
                boxes = compute_strip_boxes(w, h, strips_planned, STRIP_OVERLAP)
                combined_content = []
                combined_raw = []
                extracted_names: List[str] = []
                extracted_dates: List[Dict[str, Any]] = []
                identified_roles: Dict[str, Any] = {}
                tags: List[str] = []
                military_blocks: List[Dict[str, Any]] = []
                page_date = None
                page_locations = []

                for idx, box in enumerate(boxes):
                    left, top, right, bottom = box
                    strip = im.crop(box)
                    strip_filename = f"{img_path.stem}_strip_{idx+1}.png"
                    strip_path = PER_PAGE_DIR / strip_filename
                    strip.save(strip_path)

                    buf = _image_to_bytes(strip)
                    data = ocr_with_openai(buf, strip_filename)

                    if isinstance(data, dict):
                        content = data.get("content") or ""
                        raw_text = data.get("raw_text") or data.get("transcription") or ""
                        if content:
                            combined_content.append(content)
                        if raw_text:
                            combined_raw.append(raw_text)

                        extracted_names.extend(data.get("extracted_names") or [])
                        extracted_dates.extend(data.get("extracted_dates") or [])
                        tags.extend(data.get("tags") or [])

                        ir = data.get("identified_roles") or {}
                        for k, v in ir.items():
                            if k not in identified_roles:
                                identified_roles[k] = v
                            else:
                                existing = identified_roles[k]
                                if isinstance(existing, list):
                                    new_roles = v if isinstance(v, list) else [v]
                                    for r in new_roles:
                                        if r not in existing:
                                            existing.append(r)
                                else:
                                    new_list = [existing]
                                    new_roles = v if isinstance(v, list) else [v]
                                    for r in new_roles:
                                        if r not in new_list:
                                            new_list.append(r)
                                    identified_roles[k] = new_list

                        if data.get("military_service"):
                            military_blocks.append(data["military_service"])

                        if data.get("page_date"):
                            page_date = data["page_date"]
                        if data.get("page_locations"):
                            page_locations = data["page_locations"]

                page_military = merge_military_blocks(military_blocks)

                result = {
                    "page_number": page_number,
                    "page_title": f"Page {page_number}",
                    "content": "\n\n".join(combined_content).strip(),
                    "raw_text": "\n\n".join(combined_raw).strip(),
                    "extracted_names": sorted(set(extracted_names)),
                    "extracted_dates": extracted_dates,
                    "identified_roles": identified_roles,
                    "tags": sorted(set(tags)),
                    "military_service": page_military,
                    "page_date": page_date,
                    "page_locations": page_locations,
                    "strip_mode": mode,
                    "strips_used": strips_planned,
                    "dry_run": False,
                }

        if isinstance(result, dict):
            if "page_number" not in result or result["page_number"] in (None, -1):
                result["page_number"] = page_number
            if "page_title" not in result:
                result["page_title"] = f"Page {page_number}"

        per_page_json_path = OCR_OUTPUT_DIR / f"{img_path.stem}.json"
        with per_page_json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        append_jsonl(OUTPUT_JSONL, result)

        write_complexity_row(COMPLEXITY_LOG, complexity_row)
        return result

    except Exception as e:
        print(f"[ERROR] Failed to process page {img_path}: {e}")
        traceback.print_exc()
        complexity_row["error"] = str(e)
        write_complexity_row(COMPLEXITY_LOG, complexity_row)
        return None


# -------------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------------


def collect_page_files() -> List[Path]:
    if not PAGES_DIR.exists():
        print(f"[ERROR] PAGES_DIR does not exist: {PAGES_DIR}")
        return []
    files = sorted(PAGES_DIR.glob("*.png"))
    return files


def main() -> None:
    print(f"[INFO] PAGES_DIR      = {PAGES_DIR}")
    print(f"[INFO] OUTPUT_JSONL   = {OUTPUT_JSONL}")
    print(f"[INFO] PER_PAGE_DIR   = {PER_PAGE_DIR}")
    print(f"[INFO] OCR_OUTPUT_DIR = {OCR_OUTPUT_DIR}")
    print(f"[INFO] MAX_WORKERS    = {MAX_WORKERS}")
    print(f"[INFO] USE_STRIPS     = {USE_STRIPS}")
    print(f"[INFO] AUTO_STRIP_MODE= {AUTO_STRIP_MODE}")
    print(f"[INFO] DRY_RUN        = {DRY_RUN}")
    print()

    files = collect_page_files()
    if not files:
        print("[ERROR] No PNG files found in PAGES_DIR.")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} page candidates.")

    filtered: List[Path] = []
    for p in files:
        page_number = extract_page_number(p)
        if START_PAGE is not None and page_number != -1 and page_number < START_PAGE:
            continue
        if END_PAGE is not None and page_number != -1 and page_number > END_PAGE:
            continue
        filtered.append(p)

    print(f"[INFO] After range filter: {len(filtered)} pages will be processed.")

    if not filtered:
        print("[WARN] No pages matched the specified range. Exiting.")
        return

    print(f"[INFO] Writing/append JSONL to: {OUTPUT_JSONL}")
    print(f"[INFO] Complexity log at: {COMPLEXITY_LOG}")
    print()

    start_time = time.time()
    results: List[Optional[Dict[str, Any]]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(process_page, p): p for p in filtered}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Exception while processing {p}: {e}")
                traceback.print_exc()

    elapsed = time.time() - start_time
    processed_count = sum(1 for r in results if r is not None)
    print()
    print("==============================================")
    print(" OCR RUN COMPLETE")
    print("==============================================")
    print(f"  Pages processed successfully: {processed_count}")
    print(f"  Total pages attempted:        {len(filtered)}")
    print(f"  Elapsed time (seconds):       {elapsed:.1f}")
    print(f"  DRY_RUN                       : {DRY_RUN}")
    print("==============================================")
    print()


if __name__ == "__main__":
    main()