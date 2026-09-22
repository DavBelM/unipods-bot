"""Vercel entrypoint. Exposes the same FastAPI app as main.py (local dev),
just without the static file mount — Vercel serves static/ directly via
vercel.json rewrites instead of routing it through this function.
"""

import sys
from pathlib import Path

# Make the project root importable so `from app_core import app` etc. work
# regardless of Vercel's exact working directory for this function.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_core import app  # noqa: E402
