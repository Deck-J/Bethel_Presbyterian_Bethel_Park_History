import os
import re
import json
import time
import base64
import io
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------
# PIL SAFETY / IMAGE LIMITS
# ---------------------------------------------------------------
# Trusted local scans, so we disable the default decompression bomb
# limit and manage size ourselves via downscaling.
# ---------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None  # we downscale aggressively before OCR

# ===============================================================
# ENVIRONMENT & CONFIGURATION
# ===============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory where your page images live (e.g., PNGs 1.png, 2.png, …)
PAGES_DIR = Path(os.getenv("PAGES_DIR", "pages"))

# JSONL file containing one structured JSON object per page
OUTPUT_JSONL = Path(os.getenv("OUTPUT_JSONL", "ocr_all_pages.jsonl"))

# Directory for intermediate per-strip images (for USE_STRIPS=1)
PER_PAGE_DIR = Path(os.getenv("PER_PAGE_DIR", "per_page"))

# Directory for per-page Pass 1 OCR output (raw text)
OCR_OUTPUT_DIR = Path(os.getenv("OCR_OUTPUT_DIR", "ocr_output"))

# Concurrency settings
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# Models
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")        # Vision-capable model
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")       # Text-only model for Pass 2

# Strip-based OCR configuration (global defaults)
USE_STRIPS = int(os.getenv("USE_STRIPS", "0"))            # 0 = off, 1 = always use strips
STRIPS_PER_PAGE = int(os.getenv("STRIPS_PER_PAGE", "6"))  # typical sweet spot: 6
STRIP_OVERLAP = int(os.getenv("STRIP_OVERLAP", "20"))     # pixel overlap to avoid cutting lines

# Automatic per-page strip decision based on ink density
AUTO_STRIP_MODE = int(os.getenv("AUTO_STRIP_MODE", "1"))  # 1 = auto choose strips per page
EASY_INK_THRESHOLD = float(os.getenv("EASY_INK_THRESHOLD", "0.05"))
DENSE_INK_THRESHOLD = float(os.getenv("DENSE_INK_THRESHOLD", "0.12"))
SUPER_DENSE_INK_THRESHOLD = float(os.getenv("SUPER_DENSE_INK_THRESHOLD", "0.18"))
MAX_STRIPS_PER_PAGE = int(os.getenv("MAX_STRIPS_PER_PAGE", "8"))

# CSV log file for page complexity output
COMPLEXITY_LOG = Path(os.getenv("COMPLEXITY_LOG", "page_complexity.csv"))

# Image downscaling configuration tuned for GPT-4o vision behavior.
# We downscale eagerly to:
#   - longest side <= MAX_LONG_DIM (default 2048)
#   - total pixels <= MAX_TOTAL_PIXELS (default 20M)
MAX_LONG_DIM = int(os.getenv("MAX_LONG_DIM", "2048"))        # max width or height in pixels
MAX_TOTAL_PIXELS = int(os.getenv("MAX_TOTAL_PIXELS", "20000000"))  # 20M pixels upper bound

# Optional page-range constraints (used in GitHub Actions & local)
START_PAGE_ENV = os.getenv("START_PAGE")
END_PAGE_ENV = os.getenv("END_PAGE")
START_PAGE = int(START_PAGE_ENV) if START_PAGE_ENV and START_PAGE_ENV.strip() != "" else None
END_PAGE = int(END_PAGE_ENV) if END_PAGE_ENV and END_PAGE_ENV.strip() != "" else None

client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================================================
# DICTIONARY HINTS
# ===============================================================

"""
dictionary_hints.json should live next to this script and look like:

{
  "names": [...],
  "church_roles": [...],
  "church_terms": [...],
  "military_terms": [...],
  "places": [...],
  "fuzzy_name_variants": {
      "Kiddoo": ["Kidoo","Kiddo",...],
      ...
  }
}
"""

DICTIONARY_HINTS_FILE = Path("dictionary_hints.json")
if DICTIONARY_HINTS_FILE.exists():
    try:
        DICTIONARY_HINTS = json.loads(DICTIONARY_HINTS_FILE.read_text())
    except Exception:
        DICTIONARY_HINTS = {
            "names": [],
            "church_roles": [],
            "church_terms": [],
            "military_terms": [],
            "places": [],
            "fuzzy_name_variants": {}
        }
else:
    DICTIONARY_HINTS = {
        "names": [],
        "church_roles": [],
        "church_terms": [],
        "military_terms": [],
        "places": [],
        "fuzzy_name_variants": {}
    }

DICTIONARY_HINTS_JSON = json.dumps(DICTIONARY_HINTS, ensure_ascii=False)

# ===============================================================
# SYSTEM PROMPTS
# ===============================================================

SYSTEM_PROMPT_PASS1 = """
You are an expert paleographer and historical archivist specializing in 
18th–19th century American Presbyterian church registers.

Your job is to perform STRICT diplomatic transcription from images of 
handwritten manuscript pages.

Rules you MUST follow:

1. TRANSCRIBE EXACTLY AS WRITTEN
   - Preserve all original spelling (even when incorrect or archaic).
   - Preserve capitalization, punctuation, and line breaks.
   - Do NOT modernize or expand abbreviations.
   - Do NOT guess missing text beyond what is clearly visible.
   - If a word or portion is illegible, mark it exactly as:
       [illegible]    (single unreadable word)
       [illegible x]  (e.g., [illegible 5] for a short unreadable phrase).

2. LAYOUT
   - Maintain line breaks exactly as seen.
   - Do NOT join lines that are visually separate.
   - Do NOT create new paragraphs unless the manuscript clearly groups
     lines into a block.

3. LEDGER BRACKETS, BOXES, AND FRAMES
   - If a ledger bracket, ruled box, or margin indicator is present AND 
     it contains no handwriting, output exactly:
         [empty ledger line]
     on its own line.
   - If the bracket/box contains ANY handwriting (even faint), transcribe
     that handwriting normally.
   - Do NOT mark empty boxes as “[illegible]” unless actual ink is present
     but unreadable.
   - Do NOT hallucinate text inside empty structures.

4. WHAT NOT TO DO
   - Do NOT summarize or paraphrase.
   - Do NOT “fix” spelling or grammar.
   - Do NOT expand names (e.g., Wm → William).
   - Do NOT introduce names that are not in the image.
   - Do NOT remove or alter page noise unless it is clearly non-text 
     (smudges, ink spills, etc.).

5. DICTIONARY HINTS
   You will receive dictionary hints in the user message, containing known 
   names, places, and terms from Bethel Church records.
   - Prefer these spellings ONLY when the manuscript supports them visually.
   - Do NOT force-match hints when the handwriting does not align.
   - Do NOT introduce a hinted term unless there is real visual evidence.

Return ONLY the literal transcription. No commentary, no JSON, no markup.
"""

SYSTEM_PROMPT_PASS2 = """
You are a historical data extraction system for 18th–19th century 
Presbyterian church records. You convert diplomatic transcriptions into 
structured JSON with no hallucinations.

Follow these rules exactly.

---------------------------------------------------------------------
1. CONTENT FIDELITY
---------------------------------------------------------------------
- Treat the transcription as authoritative.
- DO NOT invent dates, names, causes, or historical information.
- Preserve original spelling and vocabulary.
- Use conservative inference: no guessing.

---------------------------------------------------------------------
2. NORMALIZATION (transcription_normalized)
---------------------------------------------------------------------
- Fix spacing and OCR artifacts.
- Merge obviously broken words that were accidentally split across lines.
- Preserve original wording and spelling.
- Do NOT modernize or rephrase.

---------------------------------------------------------------------
3. JSON SCHEMA (REQUIRED, EXACT)
---------------------------------------------------------------------

Output a JSON object with EXACTLY these fields:

{
  "page_number": int,
  "page_id": str,
  "transcription": str,
  "transcription_normalized": str,
  "extracted_names": [str],
  "extracted_dates": [
     {
       "raw": str,
       "year": str or null,
       "month": str or null,
       "day": str or null,
       "iso_date": str or null,
       "type": "full_date" | "partial_date" | "year_only"
     }
  ],
  "tags": [str],
  "cause_of_death": {
       "primary_cause": str,
       "category": "disease" | "battle" | "accident" | "unspecified",
       "raw_indicator": str
  } | null,
  "military_service": {
       "has_military_association": bool,
       "conflict": str or null,
       "service_side": str or null,
       "military_rank": str or null,
       "service_status": str or null,
       "evidence_snippets": [str],
       "certainty": float
  } | null,
  "church_role": {
       "has_role": bool,
       "roles": [str],
       "evidence_snippets": [str]
  } | null,
  "place_events": [
      {
        "place": str,
        "raw_place": str,
        "date_raw": str or null,
        "year": str or null,
        "month": str or null,
        "day": str or null,
        "iso_date": str or null,
        "date_type": "full_date" | "partial_date" | "year_only" | "unknown"
      }
  ],
  "page_date": {
    "best_raw": str or null,
    "year": str or null,
    "month": str or null,
    "day": str or null,
    "iso_date": str or null,
    "type": "full_date" | "partial_date" | "year_only" | "unknown"
  }
}

No extra fields. No commentary. ONLY this JSON object.

---------------------------------------------------------------------
4. NAME EXTRACTION
---------------------------------------------------------------------
Extract ALL personal names, including:
- Deceased individuals.
- Parents (“son of James Miller Sr.”).
- Spouses.
- Children.
- Clergy (Rev., Dr., Pastor).
- Elders, ruling elders, deacons.
- Military officers (Capt., Lt., Ensign, Sergeant, Private).
- Individuals referenced indirectly.

Rules:
- Keep names exactly as written (including abbreviations like Wm., Jas.).
- Do NOT expand or modernize names.
- Do NOT invent names.

---------------------------------------------------------------------
5. DATE EXTRACTION
---------------------------------------------------------------------
Detect:
- Full dates (e.g., “April 14th 1864”).
- Partial dates (e.g., “May 11th”, “Sept 4th”).
- Year-only references (e.g., “1865”).

For each date mention:
- Keep raw string as it appears (“raw”).
- Extract year / month / day where explicitly given.
- type:
    - "full_date"  if day + month + year are all present.
    - "partial_date" if only day + month OR month + year OR day + year.
    - "year_only" if only the year is given.
- iso_date:
    - "YYYY-MM-DD" if all components exist.
    - null otherwise.

Do NOT guess missing year if it is not clearly in the text.

---------------------------------------------------------------------
6. PAGE-LEVEL DATE
---------------------------------------------------------------------
Derive a single page-level date summary in "page_date":
- best_raw: the most representative date phrase for the page.
- Prefer, in order:
  1) A clear header or section date that appears to govern the page.
  2) Otherwise, the earliest full date in the main body.
  3) If only a year is present, you may use it with type="year_only".
- year / month / day / iso_date / type:
  - follow the same rules as extracted_dates.
  - type = "full_date" | "partial_date" | "year_only" | "unknown".
- If there is no clear date on the page, set:
  - best_raw = null,
  - year/month/day/iso_date = null,
  - type = "unknown".

Do NOT invent a date that is not clearly present on the page.

---------------------------------------------------------------------
7. CAUSE OF DEATH DETECTION
---------------------------------------------------------------------
Identify cause-of-death expressions such as:
- “died suddenly”
- “died very suddenly”
- “apoplexy”
- “dysentery”
- “consumption”
- “general inflammation”
- “inflammation of the lungs”
- “camp fever”
- “typhoid fever”
- “scarlet fever”
- “scarlatina”
- “fell slain in battle”
- “killed by the enemy’s firing”
- “killed in battle”
- “died of fever”
- “drowned”
- “accidental death”

Return:
- primary_cause: short human-readable description (e.g., “Apoplexy”, 
  “Inflammation of the lungs”, “Killed in battle”).
- category:
    - "disease"
    - "battle"
    - "accident"
    - "unspecified"
- raw_indicator: the exact phrase from the transcription that signaled 
  this cause.

If no cause is present, set cause_of_death = null.

---------------------------------------------------------------------
8. MILITARY SERVICE DETECTION (Civil War, Revolutionary War, etc.)
---------------------------------------------------------------------
Detect military associations ONLY when explicitly stated, or when a 
combination of rank + unit + battlefield strongly indicates service.

Examples of VALID evidence:

Civil War:
- “fell slain on the battlefield”
- “fell slain on the battlefield on the Wilderness of Virginia”
- “killed dead in battle”
- “killed near Petersburg”
- “in the defence of the best government on earth”
- “Union soldier”, “Confederate soldier”
- “wounded and brought home”
- “died of wounds received in battle”

Revolutionary War:
- “Ranger on the Frontier”
- “Washington County Militia”
- “Virginia Line”
- “Revolutionary soldier”
- “DAR Patriot”
- “SAR marker”
- “8th Battalion, Bedford County Militia”

Other wars:
- “War of 1812”
- “French & Indian War”
- “Mexican War”
- “campaign of 1814”
- “frontier service under Col. ____”

Rules:
- Do NOT infer service if the text is generic or ambiguous.
- Use conflict labels exactly like:
    - "US Civil War"
    - "Revolutionary War"
    - "War of 1812"
    - "French & Indian War"
    - "Mexican War"
    - or null if unknown.

---------------------------------------------------------------------
9. MILITARY RANK EXTRACTION
---------------------------------------------------------------------
Extract rank exactly as written when present, such as:
- “Lieut”
- “Lieutenant”
- “Lt.”
- “Capt.”
- “Captain”
- “Ensign”
- “Sergeant”
- “Sgt.”
- “Corporal”
- “Major”
- “Colonel”
- “Ranger”
- “Private”

Rules:
- Do NOT modernize rank names.
- Store them exactly in "military_rank".
- If no rank is mentioned, military_rank = null.

For military_service, when evidence is present:
- has_military_association = true
- conflict = appropriate conflict name or null
- service_side = "Union" | "Confederate" | "Patriot" | "Loyalist" | "Unknown"
- military_rank = extracted rank or null
- service_status (examples):
    - "DiedInBattle"
    - "DiedOfWounds"
    - "DiedOfDiseaseOrWounds"
    - "KilledInAction"
    - "Veteran"
    - "ActiveService"
    - "Uncertain"
- evidence_snippets = list of short quotes from the transcription
- certainty = a float in [0.0, 1.0] representing how confident you are.

If no evidence:
- military_service = null.

---------------------------------------------------------------------
10. CHURCH ROLE EXTRACTION
---------------------------------------------------------------------
Identify church-related roles and statuses such as:
- Elder
- Ruling Elder
- Deacon
- Pastor
- Minister
- Rev. (treated as minister/pastor)
- Session Clerk / Clerk of Session
- Trustee
- Superintendent
- Teacher
- Organist / Chorister
- Member / Communicant / Non-communicant
- Candidate
- Missionary
- Visiting preacher
- “consistent member”
- “member of Bethel Church”
- similar ecclesiastical titles.

Rules:
- Extract the role(s) exactly as written.
- A single person may have multiple roles (e.g., “Elder and Deacon”).
- If membership status is clearly described (e.g., “a consistent member of 
  Bethel Church for many years”), treat that as a role.

Return:
"church_role": {
    "has_role": true,
    "roles": [list of role strings exactly as written],
    "evidence_snippets": [short quotes from the transcription]
}

If no church role appears, set:
"church_role": null.

---------------------------------------------------------------------
11. TAG LOGIC
---------------------------------------------------------------------
Use tags to quickly mark major features. Examples:

- "civil_war"              (any Civil War military service)
- "revolutionary_war"      (any Revolutionary War service)
- "war_of_1812"
- "french_and_indian_war"
- "mexican_war"
- "veteran"                (if clearly a veteran)
- "military_rank"          (if a rank is extracted)
- "battle_related"         (battle death or wounds)
- "disease_related"
- "accidental_death"
- "sudden_death"
- "infant"
- "mother_and_child"
- "elder"
- "ruling_elder"
- "deacon"
- "pastor"
- "minister"
- "session_clerk"
- "member"
- "communicant"
- "non_communicant"
- "missionary"
- "teacher"

Choose only those that clearly apply.
If none apply, return an empty list.

---------------------------------------------------------------------
12. PLACE + DATE EVENTS
---------------------------------------------------------------------
In addition to extracted_dates, build a list of PLACE + DATE 
associations for this page in "place_events".

Places include (but are not limited to):
- Countries and regions: "America", "North America", "South America",
  "Chile", "Peru", "Brazil", "China", "Japan", "Korea", "India",
  "Syria", "Palestine", "Turkey", "Europe", "Africa", etc.
- Broader geographic labels: "foreign fields", "heathen lands",
  "the East", "the West", when clearly geographic.
- You may also include domestic place names (states, cities, counties)
  when they appear in a clearly dated missionary or historical context.

For each place event:
- place: normalized place name (e.g., "South America", "Korea").
- raw_place: the exact text as written in the transcription.
- date_raw: the nearest date expression clearly associated with this place
  in the same phrase, line, or sentence (if any).
- year / month / day / iso_date / date_type: 
  - If you have a full date, set all fields accordingly and date_type="full_date".
  - If you only have a month and year or day and month, use date_type="partial_date".
  - If you only have a year (e.g., "in 1867"), set year and date_type="year_only".
  - If you cannot confidently connect a specific date, set these fields to 
    null and use date_type="unknown".

Rules:
- Do NOT invent dates that are not clearly connected to the place.
- If the book context implies a year but the page itself does not state it,
  do NOT guess the year.
- It is acceptable to have place_events with a place and date_type="unknown"
  when no explicit date is tied to that place.

---------------------------------------------------------------------
13. OUTPUT RULES
---------------------------------------------------------------------
- Return ONLY the JSON object (no Markdown, no code fences).
- Do NOT include explanations or commentary.
- Do NOT include any text before or after the JSON.
"""

SYSTEM_PROMPT_PASS2 = SYSTEM_PROMPT_PASS2.strip()

# ===============================================================
# RETRY DECORATOR
# ===============================================================

def retry(exceptions, tries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    Simple retry decorator for transient errors (network issues, rate limits, etc.).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"[retry] {func.__name__} failed with {e}, retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ===============================================================
# CSV LOGGING FOR COMPLEXITY
# ===============================================================

def init_complexity_log():
    """
    Create CSV log header if file does not exist.
    """
    if not COMPLEXITY_LOG.exists():
        COMPLEXITY_LOG.write_text("page_id,page_number,ink_ratio,classification\n", encoding="utf-8")


def append_complexity_log(page_id: str, page_number: int, ink_ratio: float, classification: str):
    """
    Append a single complexity entry to the CSV log.
    """
    with open(COMPLEXITY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{page_id},{page_number},{ink_ratio:.4f},{classification}\n")

# ===============================================================
# IMAGE UTILITIES (DOWNSCALING + ENCODING)
# ===============================================================

def load_and_maybe_downscale(image_path: Path) -> Image.Image:
    """
    Open an image and downscale it if it's excessively large, according to:
      - longest side <= MAX_LONG_DIM
      - total pixels <= MAX_TOTAL_PIXELS
    """
    img = Image.open(image_path)
    w, h = img.size
    total_pixels = w * h

    if max(w, h) > MAX_LONG_DIM or total_pixels > MAX_TOTAL_PIXELS:
        scale_long = MAX_LONG_DIM / float(max(w, h))
        scale_area = (MAX_TOTAL_PIXELS / float(total_pixels)) ** 0.5
        scale = min(scale_long, scale_area, 1.0)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        print(f"   [Image] Downscaling {image_path.name} from {w}x{h} to {new_w}x{new_h}")
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def encode_image_obj(img: Image.Image) -> str:
    """
    Encode a PIL Image object as base64 PNG.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image(path: Path) -> str:
    """
    Encode an image file directly from disk (used for strips).
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def slice_page_to_strips(image_path: Path, out_dir: Path, strips: int, overlap: int) -> List[Path]:
    """
    Slice a page image into horizontal strips, with vertical overlap between
    adjacent strips to avoid cutting lines in half.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_and_maybe_downscale(image_path)
    w, h = img.size

    slice_height = h // strips
    out_paths: List[Path] = []

    for i in range(strips):
        top = max(0, i * slice_height - overlap)
        bottom = min(h, (i + 1) * slice_height + overlap)
        crop = img.crop((0, top, w, bottom))
        fname = f"{image_path.stem}_strip_{i+1}.png"
        outpath = out_dir / fname
        crop.save(outpath)
        out_paths.append(outpath)

    return out_paths

# ===============================================================
# PAGE COMPLEXITY ESTIMATION
# ===============================================================

def estimate_page_complexity(image_path: Path) -> Dict[str, Any]:
    """
    Estimate how 'dense' the handwriting on a page is by calculating
    ink coverage on a downscaled grayscale version (pure PIL).
    """
    img = load_and_maybe_downscale(image_path)
    preview_size = (512, 512)

    img_small = img.convert("L")
    img_small.thumbnail(preview_size, Image.LANCZOS)

    pixels = list(img_small.getdata())
    total = len(pixels)
    DARK_THRESHOLD = 190  # 0=black, 255=white

    dark_count = sum(1 for p in pixels if p < DARK_THRESHOLD)
    ink_ratio = dark_count / total

    if ink_ratio >= DENSE_INK_THRESHOLD:
        classification = "dense"
    elif ink_ratio <= EASY_INK_THRESHOLD:
        classification = "easy"
    else:
        classification = "medium"

    return {
        "ink_ratio": ink_ratio,
        "classification": classification,
    }

# ===============================================================
# FUZZY NAME NORMALIZATION (POST-OCR)
# ===============================================================

def apply_fuzzy_name_normalization(text: str, dict_hints: dict) -> str:
    """
    Normalize obvious OCR variants for certain names based on
    dictionary_hints['fuzzy_name_variants'].
    Fife/Fyfe are *not* merged on purpose.
    """
    fuzzy = dict_hints.get("fuzzy_name_variants", {})
    if not fuzzy:
        return text

    variant_to_canonical = {}
    for canonical, variants in fuzzy.items():
        # keep Fife/Fyfe distinct on purpose
        if canonical in {"Fife", "Fyfe"}:
            continue
        for v in variants:
            variant_to_canonical[v] = canonical

    if not variant_to_canonical:
        return text

    sorted_variants = sorted(variant_to_canonical.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(v) for v in sorted_variants) + r")\b"

    def replace_match(match: re.Match) -> str:
        word = match.group(0)
        return variant_to_canonical.get(word, word)

    return re.sub(pattern, replace_match, text)

# ===============================================================
# PASS 1 — OCR (FULL PAGE & STRIPS)
# ===============================================================

@retry(Exception)
def ocr_full_image(image_path: Path) -> str:
    """
    OCR on a full page image using the Vision model and PASS1 prompt.
    """
    img = load_and_maybe_downscale(image_path)
    img_b64 = encode_image_obj(img)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PASS1},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe the following page image exactly as written. Dictionary hints are provided below."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": f"Dictionary hints: {DICTIONARY_HINTS_JSON}"
                },
            ],
        },
    ]

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content


@retry(Exception)
def ocr_strip_image(image_path: Path) -> str:
    """
    OCR on a single strip (partial page).
    """
    img_b64 = encode_image(image_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PASS1},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Transcribe this cropped portion of a page exactly as written. It is part of a larger page. Dictionary hints are provided below."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": f"Dictionary hints: {DICTIONARY_HINTS_JSON}"
                },
            ],
        },
    ]

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content


def ocr_page(image_path: Path, page_id: str, page_number: int) -> str:
    """
    High-level OCR for a single page.
    - Scores complexity
    - Logs to CSV
    - Decides full-page vs strip-based OCR
    - If super-dense, increases strip count up to MAX_STRIPS_PER_PAGE
    """
    complexity = estimate_page_complexity(image_path)
    ink_ratio = complexity["ink_ratio"]
    classification = complexity["classification"]

    print(f"   [Complexity] ink_ratio={ink_ratio:.3f} => {classification}")
    append_complexity_log(page_id, page_number, ink_ratio, classification)

    use_strips_now = (USE_STRIPS == 1)

    if AUTO_STRIP_MODE == 1:
        if classification == "dense":
            use_strips_now = True
        else:  # easy or medium
            use_strips_now = False

    if not use_strips_now:
        print("   [Pass1] Using full-page OCR")
        return ocr_full_image(image_path)

    strips_for_page = STRIPS_PER_PAGE
    if classification == "dense" and ink_ratio >= SUPER_DENSE_INK_THRESHOLD:
        strips_for_page = max(STRIPS_PER_PAGE, MAX_STRIPS_PER_PAGE)

    print(f"   [Pass1] Using strip OCR: {strips_for_page} strips, overlap={STRIP_OVERLAP}px")
    strips = slice_page_to_strips(image_path, PER_PAGE_DIR, strips_for_page, STRIP_OVERLAP)

    results: Dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(ocr_strip_image, s): s for s in strips}
        for future in as_completed(future_map):
            strip_path = future_map[future]
            try:
                text = future.result()
                results[strip_path] = text
            except Exception as e:
                print(f"   [Pass1] ERROR OCR strip {strip_path.name}: {e}")
                results[strip_path] = f"[OCR ERROR: {e}]"

    ordered_texts = [results[p] for p in sorted(results.keys(), key=lambda x: x.name)]
    return "\n\n".join(ordered_texts)

# ===============================================================
# PASS 2 — STRUCTURED EXTRACTION
# ===============================================================

@retry(Exception)
def extract_structured_data(page_text: str, page_number: int, page_id: str) -> Dict[str, Any]:
    """
    Use TEXT_MODEL to convert transcription into structured JSON.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PASS2},
        {
            "role": "user",
            "content": (
                f"PAGE NUMBER: {page_number}\n"
                f"PAGE ID: {page_id}\n\n"
                f"TRANSCRIPTION:\n{page_text}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0,
    )

    raw_content = resp.choices[0].message.content

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        data = {
            "page_number": page_number,
            "page_id": page_id,
            "transcription": page_text,
            "transcription_normalized": page_text,
            "extracted_names": [],
            "extracted_dates": [],
            "tags": [],
            "cause_of_death": None,
            "military_service": None,
            "church_role": None,
            "place_events": [],
            "page_date": {
                "best_raw": None,
                "year": None,
                "month": None,
                "day": None,
                "iso_date": None,
                "type": "unknown"
            },
            "error": "JSON decode error in Pass 2",
            "raw_model_output": raw_content,
        }

    data.setdefault("page_number", page_number)
    data.setdefault("page_id", page_id)
    if "place_events" not in data:
        data["place_events"] = []
    if "page_date" not in data:
        data["page_date"] = {
            "best_raw": None,
            "year": None,
            "month": None,
            "day": None,
            "iso_date": None,
            "type": "unknown"
        }
    return data

# ===============================================================
# PAGE PROCESSING
# ===============================================================

def process_page(image_path: Path):
    """
    Pass 1 + Pass 2 for a single page;
    writes OCR output per page and appends structured JSON to JSONL.
    """
    page_stem = image_path.stem

    m = re.match(r"(\d+)", page_stem)
    if m:
        page_number = int(m.group(1))
    else:
        page_number = -1

    page_id = page_stem

    print(f"\n➡ Processing page: {page_id} (page_number={page_number})")

    # Pass 1: OCR (full-page or strips)
    raw_text = ocr_page(image_path, page_id, page_number)

    # Fuzzy name normalization (Kiddoo, Woods, etc.)
    raw_text = apply_fuzzy_name_normalization(raw_text, DICTIONARY_HINTS)

    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pass1_out = {
        "page_number": page_number,
        "page_id": page_id,
        "content": raw_text,
    }
    pass1_path = OCR_OUTPUT_DIR / f"{page_id}.json"
    pass1_path.write_text(json.dumps(pass1_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   [Pass1] Saved raw OCR to {pass1_path}")

    # Pass 2: structured extraction
    structured = extract_structured_data(raw_text, page_number, page_id)
    print(f"   [Pass2] Extracted structured data for {page_id}")

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(structured, ensure_ascii=False) + "\n")

    print(f"✅ Completed page {page_id}.")

# ===============================================================
# MAIN
# ===============================================================

def main():
    print("===========================================")
    print(" Bethel Register OCR & Extraction Pipeline ")
    print("===========================================")
    print(f"Using Vision model: {VISION_MODEL}")
    print(f"Using Text model:   {TEXT_MODEL}")
    print(f"PAGES_DIR:          {PAGES_DIR}")
    print(f"OUTPUT_JSONL:       {OUTPUT_JSONL}")
    print(f"USE_STRIPS:         {USE_STRIPS}")
    print(f"AUTO_STRIP_MODE:    {AUTO_STRIP_MODE}")
    if AUTO_STRIP_MODE:
        print(f"  EASY_INK_THRESHOLD:        {EASY_INK_THRESHOLD}")
        print(f"  DENSE_INK_THRESHOLD:       {DENSE_INK_THRESHOLD}")
        print(f"  SUPER_DENSE_INK_THRESHOLD: {SUPER_DENSE_INK_THRESHOLD}")
    print(f"STRIPS_PER_PAGE:    {STRIPS_PER_PAGE}")
    print(f"MAX_STRIPS_PER_PAGE:{MAX_STRIPS_PER_PAGE}")
    print(f"STRIP_OVERLAP:      {STRIP_OVERLAP}px")
    print(f"MAX_WORKERS:        {MAX_WORKERS}")
    print(f"MAX_LONG_DIM:       {MAX_LONG_DIM}px")
    print(f"MAX_TOTAL_PIXELS:   {MAX_TOTAL_PIXELS}")
    print(f"START_PAGE:         {START_PAGE}")
    print(f"END_PAGE:           {END_PAGE}")
    print("-------------------------------------------")

    init_complexity_log()

    if not PAGES_DIR.exists():
        print(f"ERROR: PAGES_DIR does not exist: {PAGES_DIR}")
        return

    image_files = sorted(PAGES_DIR.glob("*.png"))
    if not image_files:
        print(f"No PNG files found in {PAGES_DIR}")
        return

    processed_pages = set()
    if OUTPUT_JSONL.exists():
        try:
            with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        processed_pages.add(str(rec.get("page_id")))
                    except Exception:
                        continue
        except Exception:
            pass

    for img in image_files:
        stem = img.stem
        m = re.match(r"(\d+)", stem)
        if m:
            page_number = int(m.group(1))
        else:
            page_number = -1

        # Apply page range filtering if specified
        if START_PAGE is not None and page_number != -1 and page_number < START_PAGE:
            continue
        if END_PAGE is not None and page_number != -1 and page_number > END_PAGE:
            continue

        if stem in processed_pages:
            print(f"⏩ Skipping already-processed page: {stem}")
            continue

        process_page(img)

    print("\n🎉 ALL DONE — Full OCR + Extraction Complete.")

if __name__ == "__main__":
    main()