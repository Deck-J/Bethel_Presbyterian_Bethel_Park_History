#!/usr/bin/env bash
set -euo pipefail

echo "==========================================="
echo " Bethel OCR Repo Bootstrap Script"
echo "==========================================="

# 1) Basic checks
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed or not on PATH."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "WARNING: python3 not found on PATH. You'll need it locally for testing."
fi

# 2) Init git repo if needed
if [ ! -d ".git" ]; then
  echo "Initializing new git repository..."
  git init
else
  echo "Git repository already initialized."
fi

# 3) Create directories
echo "Creating required directories (if missing)..."
mkdir -p pages
mkdir -p per_page
mkdir -p ocr_output
mkdir -p .github/workflows

echo "  - pages/"
echo "  - per_page/"
echo "  - ocr_output/"
echo "  - .github/workflows/"

# 4) Create requirements.txt if missing
if [ -f "requirements.txt" ]; then
  echo "requirements.txt already exists; leaving it as-is."
else
  echo "Creating requirements.txt..."
  cat > requirements.txt <<'EOF'
openai>=1.0.0
python-dotenv>=1.0.0
Pillow>=10.0.0
EOF
fi

# 5) Create .env.example WITH UPDATED INSTRUCTIONS
if [ -f ".env.example" ]; then
  echo ".env.example already exists; leaving it as-is."
else
  echo "Creating updated .env.example..."
  cat > .env.example <<'EOF'
# ============================================================
#  Bethel Register OCR Pipeline – Environment Configuration
# ============================================================
# Copy this file to ".env" and fill in your actual values.
# Every setting here can also be overridden in GitHub Actions.
# ============================================================

# ------------------------------
# OpenAI API key
# ------------------------------
OPENAI_API_KEY=sk-...

# ------------------------------
# Directories
# ------------------------------
# Where your input page images live
PAGES_DIR=pages

# Where the master JSONL output should go
OUTPUT_JSONL=ocr_all_pages.jsonl

# Intermediate per-strip images
PER_PAGE_DIR=per_page

# Raw OCR (Pass 1) outputs
OCR_OUTPUT_DIR=ocr_output

# ------------------------------
# Models
# ------------------------------
VISION_MODEL=gpt-4o
TEXT_MODEL=gpt-4o-mini

# ------------------------------
# Concurrency
# ------------------------------
MAX_WORKERS=4

# ============================================================
# Strip / Tiling Configuration
# ============================================================
# USE_STRIPS:
#   0 = never tile (use full-page OCR only)
#   1 = always tile, ignoring auto-detection
#
# AUTO_STRIP_MODE:
#   1 = auto-detect dense pages and tile only when needed
#   0 = turn off auto detection
USE_STRIPS=0
AUTO_STRIP_MODE=1

STRIPS_PER_PAGE=6
MAX_STRIPS_PER_PAGE=8
STRIP_OVERLAP=20

# ------------------------------
# Page-density thresholds
# ------------------------------
# EASY_INK_THRESHOLD  – below this: never tile
# DENSE_INK_THRESHOLD – above this: tile
# SUPER_DENSE_INK_THRESHOLD – above this: use MAX_STRIPS_PER_PAGE
EASY_INK_THRESHOLD=0.05
DENSE_INK_THRESHOLD=0.12
SUPER_DENSE_INK_THRESHOLD=0.18

# ------------------------------
# Downscaling limits
# ------------------------------
# Controls maximum image size sent to the API.
MAX_LONG_DIM=2048
MAX_TOTAL_PIXELS=20000000

# ------------------------------
# Page complexity log
# ------------------------------
COMPLEXITY_LOG=page_complexity.csv

# ============================================================
# PAGE RANGE CONTROL (IMPORTANT)
# ============================================================
# To control which page images are processed:
#
# Options for START_PAGE / END_PAGE:
#   - Leave blank    → process ALL pages
#   - Use a number   → e.g., START_PAGE=1
#   - Use "all"      → same as blank (process ALL)
#
# Examples:
#
#  Process all pages:
#     START_PAGE=all
#     END_PAGE=all
#
#  Process pages 1–20 only:
#     START_PAGE=1
#     END_PAGE=20
#
#  Process a single page (e.g., page 37):
#     START_PAGE=37
#     END_PAGE=37
#
#  Process everything except the first 50 pages:
#     START_PAGE=51
#     END_PAGE=all
#
# The script auto-detects invalid or missing values and defaults to ALL.
# ------------------------------------------------------------
START_PAGE=
END_PAGE=
EOF
fi

# 6) Create starter dictionary_hints.json if missing
if [ -f "dictionary_hints.json" ]; then
  echo "dictionary_hints.json already exists; leaving it as-is."
else
  echo "Creating starter dictionary_hints.json..."
  cat > dictionary_hints.json <<'EOF'
{
  "names": [
    "Oliver Miller",
    "James Miller",
    "John McMillan",
    "Samuel Kiddoo",
    "William Kiddoo",
    "John Woods",
    "Joseph Woods"
  ],
  "church_roles": [
    "Elder",
    "Ruling Elder",
    "Deacon",
    "Pastor",
    "Minister",
    "Communicant",
    "Non-communicant",
    "Member"
  ],
  "church_terms": [
    "communicant",
    "communicants",
    "non-communicants",
    "session",
    "ruling elder",
    "Bethel Church"
  ],
  "places": [
    "Bethel",
    "Pittsburgh",
    "Washington County",
    "Allegheny County"
  ],
  "military_terms": [
    "Private",
    "Lieut",
    "Lieutenant",
    "Lt.",
    "Capt.",
    "Captain",
    "Ensign",
    "Sergeant",
    "Ranger",
    "Militia",
    "Virginia Line",
    "Washington County Militia",
    "Revolutionary soldier",
    "Union",
    "Confederate"
  ],
  "fuzzy_name_variants": {
    "Kiddoo": [
      "Kidoo",
      "Kiddo",
      "Kiddoe",
      "Kiddow",
      "Kidlow",
      "Kiddco",
      "Kiddao",
      "Kiddon",
      "Kidlas",
      "Kiddale"
    ],
    "Woods": [
      "Wood",
      "Wodds",
      "Wods",
      "Woode",
      "Woodes",
      "Woods.",
      "Wods.",
      "Woobs"
    ],
    "McCully": [
      "McCulley",
      "McCuly",
      "Mc Cully",
      "McCully.",
      "M’Cully",
      "Mc-Cully"
    ],
    "McKee": [
      "McKey",
      "McKie",
      "Mc Kee",
      "McKee.",
      "M’Kee"
    ],
    "Johnston": [
      "Johnstone",
      "Johnstun",
      "Johnfton",
      "Johnston.",
      "Johnstn"
    ],
    "Hultz": [
      "Hultz.",
      "Hults",
      "Hulz",
      "Hulze",
      "Hultze"
    ],
    "Marshall": [
      "Marshal",
      "Marshel",
      "Marschell",
      "Marshl"
    ]
  }
}
EOF
fi

# 7) Create GitHub Actions workflow if missing
WORKFLOW_PATH=".github/workflows/ocr.yml"
if [ -f "$WORKFLOW_PATH" ]; then
  echo "$WORKFLOW_PATH already exists; leaving it as-is."
else
  echo "Creating $WORKFLOW_PATH..."
  cat > "$WORKFLOW_PATH" <<'EOF'
name: Bethel OCR & Extraction

on:
  workflow_dispatch:
    inputs:
      start_page:
        description: "First page number (or 'all')"
        required: false
        default: ""
      end_page:
        description: "Last page number (or 'all')"
        required: false
        default: ""
      use_strips:
        description: "Force strips (1) or let auto decide (0)"
        required: false
        default: "0"
      strips_per_page:
        description: "Base strips per page"
        required: false
        default: "6"
      max_strips_per_page:
        description: "Max strips for super-dense pages"
        required: false
        default: "8"
      max_long_dim:
        description: "Max image dimension (pixels)"
        required: false
        default: "2048"
      max_total_pixels:
        description: "Max total pixels after downscaling"
        required: false
        default: "20000000"

jobs:
  run-ocr:
    runs-on: ubuntu-latest
    timeout-minutes: 360

    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      PAGES_DIR: pages
      OUTPUT_JSONL: ocr_all_pages.jsonl
      PER_PAGE_DIR: per_page
      OCR_OUTPUT_DIR: ocr_output

      START_PAGE: ${{ github.event.inputs.start_page }}
      END_PAGE: ${{ github.event.inputs.end_page }}

      USE_STRIPS: ${{ github.event.inputs.use_strips }}
      AUTO_STRIP_MODE: "1"
      STRIPS_PER_PAGE: ${{ github.event.inputs.strips_per_page }}
      MAX_STRIPS_PER_PAGE: ${{ github.event.inputs.max_strips_per_page }}
      STRIP_OVERLAP: "20"

      EASY_INK_THRESHOLD: "0.05"
      DENSE_INK_THRESHOLD: "0.12"
      SUPER_DENSE_INK_THRESHOLD: "0.18"

      MAX_LONG_DIM: ${{ github.event.inputs.max_long_dim }}
      MAX_TOTAL_PIXELS: ${{ github.event.inputs.max_total_pixels }}

      VISION_MODEL: "gpt-4o"
      TEXT_MODEL: "gpt-4o-mini"

      MAX_WORKERS: "4"

      COMPLEXITY_LOG: "page_complexity.csv"

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          lfs: true
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Compute run branch name
        id: branchname
        run: |
          BRANCH="ocr-run-${GITHUB_RUN_ID}"
          echo "branch_name=${BRANCH}" >> $GITHUB_OUTPUT
          echo "BRANCH_NAME=${BRANCH}" >> $GITHUB_ENV

      - name: Create and switch to run branch
        run: git checkout -b "${BRANCH_NAME}"

      - name: Run OCR Pipeline
        run: python ocr_script.py

      - name: Commit OCR results
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add \
            ocr_all_pages.jsonl \
            page_complexity.csv \
            ocr_output/ \
            per_page/ || true
          if git diff --cached --quiet; then
            echo "Nothing to commit."
          else
            git commit -m "OCR results for ${BRANCH_NAME}"
          fi

      - name: Push run branch
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: git push origin "${BRANCH_NAME}"

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: bethel-ocr-${{ steps.branchname.outputs.branch_name }}
          path: |
            ocr_all_pages.jsonl
            ocr_output/
            per_page/
            page_complexity.csv
EOF
fi

# 8) .gitignore
if [ -f ".gitignore" ]; then
  echo ".gitignore already exists; leaving it as-is."
else
  echo "Creating .gitignore..."
  cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.env

# Virtual envs
.venv/
venv/

# OCR outputs
per_page/
ocr_output/
page_complexity.csv

# OS junk
.DS_Store
Thumbs.db
EOF
fi

echo
echo "Bootstrap complete!"
echo "Next steps:"
echo "  1) Review/update dictionary_hints.json if needed"
echo "  2) Create .env from .env.example and set your OPENAI_API_KEY"
echo "  3) Place your PNGs into /pages/"
echo "  4) Commit & push to GitHub"
echo "  5) Run the OCR workflow manually from the Actions tab"
echo
echo "Everything is ready to go."