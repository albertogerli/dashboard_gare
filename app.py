#!/usr/bin/env python3
"""Dashboard Gare Pubbliche - Streamlit (Extended)"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz, process
import os
import requests
import hashlib
import re
import time
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv


def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for Streamlit/Arrow rendering.

    Streamlit renders dataframes via Arrow; some pandas dtypes (mixed object, tz-aware
    datetimes, Period, etc.) can crash conversion. This helper normalizes those cases.
    """
    if df is None:
        return pd.DataFrame()

    df_copy = df.copy()

    # Arrow does not like MultiIndex in many cases
    if isinstance(df_copy.index, pd.MultiIndex):
        df_copy = df_copy.reset_index()

    # Ensure column names are strings
    df_copy.columns = [str(c) for c in df_copy.columns]

    for col in df_copy.columns:
        s = df_copy[col]

        # Normalize tz-aware datetimes to tz-naive
        try:
            if isinstance(s.dtype, pd.DatetimeTZDtype):
                df_copy[col] = pd.to_datetime(s, errors="coerce").dt.tz_convert(None)
                continue
        except Exception:
            pass

        # Period dtype -> string
        try:
            if isinstance(s.dtype, pd.PeriodDtype):
                df_copy[col] = s.astype(str)
                continue
        except Exception:
            pass

        # Categories can contain mixed types -> string
        try:
            if isinstance(s.dtype, pd.CategoricalDtype):
                df_copy[col] = s.astype(str)
                continue
        except Exception:
            pass

        # Objects (often mixed / dict / list) -> string
        if s.dtype == "object":
            def _norm_obj(v):
                if isinstance(v, (dict, list, tuple, set)):
                    try:
                        return json.dumps(v, ensure_ascii=False, default=str)
                    except Exception:
                        return str(v)
                return v

            df_copy[col] = s.map(_norm_obj).astype(str).replace(
                {"nan": "", "None": "", "NaT": ""}
            )

    return df_copy


_ILLEGAL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

def _sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove control characters that openpyxl rejects."""
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].apply(lambda v: _ILLEGAL_CHARS_RE.sub('', str(v)) if pd.notna(v) else v)
    return df


def show_dataframe(df: pd.DataFrame, label: str | None = None, preview_rows: int = 50, **kwargs):
    """Render a DataFrame without crashing the app (Arrow/pyarrow fail-safe).

    Streamlit serializes DataFrames through Arrow; in production this can sporadically fail
    (mixed dtypes, nested objects, tz-aware datetimes, etc.) and crash the script.
    This helper tries progressively safer fallbacks and logs enough context to debug.
    """
    try:
        return st.dataframe(safe_dataframe(df), **kwargs)
    except Exception as e1:
        tag = f"`{label}`" if label else "(senza label)"
        st.warning(f"⚠️ Tabella non renderizzabile via Arrow {tag}. Uso fallback (preview).")

        with st.expander("Dettagli errore tabella (debug)", expanded=False):
            st.write("Errore Arrow/Streamlit durante rendering DataFrame.")
            st.exception(e1)
            try:
                if df is not None:
                    st.write({"shape": getattr(df, "shape", None)})
                    st.write("dtypes:")
                    st.code(getattr(df, "dtypes", None).to_string() if hasattr(df, "dtypes") else "N/A")
                    st.write("preview (prime righe):")
                    st.code(df.head(3).to_string())
            except Exception:
                pass

        try:
            df_preview = df.head(int(preview_rows)).copy() if df is not None else pd.DataFrame()
            for c in df_preview.columns:
                df_preview[c] = df_preview[c].astype(str)
            return st.dataframe(df_preview, **kwargs)
        except Exception as e2:
            st.error("❌ Anche il fallback DataFrame è fallito. Mostro output testuale.")
            with st.expander("Dettagli fallback (debug)", expanded=False):
                st.exception(e2)

            try:
                df_preview = df.head(int(preview_rows)).copy() if df is not None else pd.DataFrame()
                st.code(df_preview.to_string())
            except Exception:
                st.code("(impossibile generare preview)")

            try:
                if df is not None and len(df) <= 200_000 and df.shape[1] <= 200:
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Scarica CSV (debug)",
                        data=csv_bytes,
                        file_name=f"{(label or 'dataframe')}.csv",
                        mime="text/csv",
                    )
            except Exception:
                pass
            return None


# Carica variabili d'ambiente dal file .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Config
st.set_page_config(
    page_title="Dashboard Gare Pubbliche",
    page_icon="📊",
    layout="wide"
)

# Brand colors (manual)
BRAND_BLUE = "#34657F"       # Blu primario
BRAND_GREEN = "#26D07C"      # Verde primario (CTA)
BRAND_DEEP = "#06415B"       # Deep Lagoon
BRAND_CLEAR = "#B9D7D8"      # Clear Water
BRAND_SURFACE = "#EDF6F6"    # Tint derivata (leggibile su bianco)

# UI semantic colors
CGL_GREEN = BRAND_GREEN
CGL_BLUE = BRAND_BLUE
CGL_CYAN = BRAND_CLEAR
CGL_DARK = BRAND_DEEP
CGL_BLACK = "#000000"
CGL_WHITE = "#ffffff"
CGL_ORANGE = "#ff9500"  # warning/attenzione (non brand)
CGL_RED = "#ff3b30"     # error (non brand)

# Plotly defaults (brand)
BRAND_COLORWAY = [
    BRAND_GREEN,
    BRAND_BLUE,
    BRAND_DEEP,
    BRAND_CLEAR,
    "#1FAE8A",  # teal (derivata)
    "#0B6B8C",  # deep blue (derivata)
    "#7EC9C9",  # light teal (derivata)
    "#A8E8CA",  # light green (derivata)
]
BRAND_GRID_RGBA = "rgba(185,215,216,0.35)"  # BRAND_CLEAR con alpha
BRAND_CONTINUOUS_SCALE = [BRAND_SURFACE, BRAND_CLEAR, BRAND_BLUE, BRAND_DEEP]

pio.templates["brand_manual"] = go.layout.Template(
    layout=go.Layout(
        colorway=BRAND_COLORWAY,
        font=dict(color=BRAND_DEEP, family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(color=BRAND_DEEP)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=BRAND_GRID_RGBA, zerolinecolor=BRAND_GRID_RGBA, linecolor=BRAND_GRID_RGBA),
        yaxis=dict(gridcolor=BRAND_GRID_RGBA, zerolinecolor=BRAND_GRID_RGBA, linecolor=BRAND_GRID_RGBA),
    )
)
pio.templates.default = "brand_manual"
px.defaults.template = "brand_manual"
px.defaults.color_discrete_sequence = BRAND_COLORWAY
px.defaults.color_continuous_scale = BRAND_CONTINUOUS_SCALE

# Custom CSS - Brand manual + Accessibilità
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Sans+Pro:wght@400;600;700&display=swap');

    :root {{
        --brand-blue: {BRAND_BLUE};
        --brand-green: {BRAND_GREEN};
        --brand-deep: {BRAND_DEEP};
        --brand-clear: {BRAND_CLEAR};
        --brand-surface: {BRAND_SURFACE};
        --brand-text: {BRAND_DEEP};
        --brand-text-invert: #FFFFFF;
        --brand-border: rgba(185, 215, 216, 0.7);
        --brand-shadow: rgba(6, 65, 91, 0.10);
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: var(--brand-text);
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--brand-text);
        line-height: 1.25;
    }}

    /* Metric cards */
    .stMetric > div {{
        background: linear-gradient(135deg, var(--brand-surface) 0%, #ffffff 100%);
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid var(--brand-green);
        box-shadow: 0 2px 10px var(--brand-shadow);
        border: 1px solid rgba(185, 215, 216, 0.35);
    }}

    .stMetric label {{
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: rgba(6, 65, 91, 0.75) !important;
    }}

    .stMetric [data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: var(--brand-text) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: var(--brand-surface);
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(185, 215, 216, 0.55);
    }}

    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px 18px;
        border-radius: 8px;
        color: rgba(6, 65, 91, 0.85);
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--brand-green) !important;
        color: var(--brand-text-invert) !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--brand-deep) 0%, #052F43 100%);
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: rgba(255, 255, 255, 0.92);
    }}

    [data-testid="stSidebar"] label {{
        color: rgba(255, 255, 255, 0.92) !important;
        font-weight: 600;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--brand-green) 0%, var(--brand-blue) 100%);
        color: var(--brand-text-invert);
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(6, 65, 91, 0.25);
        filter: brightness(0.98);
    }}

    .stButton > button:active {{
        transform: translateY(0px);
        filter: brightness(0.96);
    }}

    /* Inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div {{
        border-radius: 10px;
        border-color: var(--brand-border);
    }}

    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {{
        border-color: var(--brand-green);
        box-shadow: 0 0 0 2px rgba(38, 208, 124, 0.20);
    }}

    /* Info/Warning/Error boxes */
    .stAlert {{
        border-radius: 10px;
        font-size: 0.95rem;
        border-color: rgba(185, 215, 216, 0.65);
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        font-weight: 700;
        color: var(--brand-text);
        font-size: 1rem;
    }}

    /* Links */
    a {{
        color: var(--brand-blue);
        text-decoration: underline;
        text-decoration-color: rgba(52, 101, 127, 0.45);
    }}

    a:hover {{
        color: var(--brand-deep);
        text-decoration-color: rgba(6, 65, 91, 0.55);
    }}

    /* Focus for accessibility */
    *:focus-visible {{
        outline: 3px solid var(--brand-green);
        outline-offset: 2px;
    }}

    .stCaption, small {{
        font-size: 0.85rem;
        color: rgba(6, 65, 91, 0.70);
    }}

    /* Download button */
    .stDownloadButton > button {{
        background: var(--brand-deep);
        color: var(--brand-text-invert);
        border: 2px solid var(--brand-green);
        border-radius: 10px;
        font-weight: 700;
    }}

    .stDownloadButton > button:hover {{
        background: var(--brand-green);
        color: var(--brand-deep);
    }}

    /* Chart container */
    [data-testid="stPlotlyChart"] {{
        border-radius: 12px;
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)

# ==================== AI VISUALIZATION HELPERS ====================

# Favorites storage path
FAVORITES_PATH = Path(__file__).parent.parent / "data" / "output" / "dashboard" / "favorites.json"

# CIG enrichment cache path
CIG_ENRICHMENT_CACHE_PATH = Path(__file__).parent.parent / "data" / "output" / "dashboard" / "cig_enrichment_cache.json"
CIG_ENRICHMENT_CACHE_VERSION = 1
CIG_ENRICHMENT_TTL_DAYS_DEFAULT = 30

_CIG_REGEX = re.compile(r'^[A-Z0-9]{10}$', flags=re.I)
_DURATION_KEYWORDS_RE = re.compile(
    r"(durata|mesi|anni|giorni|rinnovo|proroga|quinquenn|trienn|bienn|annuale|decorrenza|stipula|consegna)",
    flags=re.I
)
_URL_RE = re.compile(r'((?:https?://|www\.)[^\s\"\'<>]+)', flags=re.I)

def load_cig_enrichment_cache() -> dict:
    """Load persisted CIG enrichment cache."""
    if not CIG_ENRICHMENT_CACHE_PATH.exists():
        return {"version": CIG_ENRICHMENT_CACHE_VERSION, "items": {}}
    try:
        data = json.loads(CIG_ENRICHMENT_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": CIG_ENRICHMENT_CACHE_VERSION, "items": {}}
        if data.get("version") != CIG_ENRICHMENT_CACHE_VERSION:
            # Basic forward-compat: keep items if present
            items = data.get("items", {})
            return {"version": CIG_ENRICHMENT_CACHE_VERSION, "items": items if isinstance(items, dict) else {}}
        items = data.get("items", {})
        return {"version": CIG_ENRICHMENT_CACHE_VERSION, "items": items if isinstance(items, dict) else {}}
    except Exception:
        return {"version": CIG_ENRICHMENT_CACHE_VERSION, "items": {}}

def save_cig_enrichment_cache(cache: dict) -> None:
    """Save cache with atomic write."""
    CIG_ENRICHMENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CIG_ENRICHMENT_CACHE_PATH.with_suffix(".tmp")
    payload = cache if isinstance(cache, dict) else {"version": CIG_ENRICHMENT_CACHE_VERSION, "items": {}}
    payload.setdefault("version", CIG_ENRICHMENT_CACHE_VERSION)
    payload.setdefault("items", {})
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp_path.replace(CIG_ENRICHMENT_CACHE_PATH)

def _normalize_cig(cig: str) -> str:
    if cig is None:
        return ""
    s = str(cig).strip().upper()
    if s in {"NAN", "NONE"}:
        return ""
    return s

def _is_valid_cig(cig: str) -> bool:
    s = _normalize_cig(cig)
    return bool(s) and bool(_CIG_REGEX.match(s))

def _chunked_read_paths_for_gare_unificate():
    """Return candidate paths for gare_unificate (gz or csv).
    Prefers full CSV (has testo_completo) over dashboard gz (may lack it)."""
    unified_path = Path(__file__).parent.parent / "data" / "output" / "categorie" / "gare_unificate.csv"
    old_path = Path(__file__).parent.parent / "data" / "output" / "categorie" / "gare_filtrate_tutte.csv"
    gz_path = Path(__file__).parent / "data" / "gare_unificate.csv.gz"
    for p in [unified_path, old_path, gz_path]:
        if p.exists():
            yield p

def load_testo_completo_for_cigs(cigs: set[str], max_chars_per_cig: int = 20000) -> dict:
    """Load testo_completo/oggetto on-demand for selected CIGs without loading full dataset."""
    target = {_normalize_cig(c) for c in (cigs or set()) if _normalize_cig(c)}
    target = {c for c in target if _is_valid_cig(c)}
    if not target:
        return {}

    path = next(_chunked_read_paths_for_gare_unificate(), None)
    if path is None:
        return {}

    # NOTE: Do NOT use pandas chunked read here.
    # `testo_completo` is very large: pandas chunks can still spike memory and crash Streamlit Cloud.
    # We stream the CSV to keep memory flat.
    import csv
    import sys
    import gzip

    out: dict[str, str] = {}
    try:
        try:
            csv.field_size_limit(sys.maxsize)
        except Exception:
            # Some platforms reject very large limits; best-effort.
            try:
                csv.field_size_limit(10_000_000)
            except Exception:
                pass

        is_gz = str(path).endswith(".gz")
        opener = gzip.open if is_gz else open  # type: ignore
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            header = next(reader, None)
            if not header:
                return {}

            idx = {str(name).strip(): i for i, name in enumerate(header)}
            cig_i = idx.get("cig")
            if cig_i is None:
                return {}

            testo_i = idx.get("testo_completo")
            oggetto_i = idx.get("oggetto")
            award_i = idx.get("data_aggiudicazione")

            for row in reader:
                if not row or len(row) <= cig_i:
                    continue
                cig = _normalize_cig(row[cig_i])
                if cig not in target or cig in out:
                    continue

                parts = []
                if oggetto_i is not None and len(row) > oggetto_i:
                    oggetto = (row[oggetto_i] or "").strip()
                    if oggetto:
                        parts.append(f"OGGETTO: {oggetto}")
                if award_i is not None and len(row) > award_i:
                    award = (row[award_i] or "").strip()
                    if award:
                        parts.append(f"DATA_AGGIUDICAZIONE_RAW: {award}")

                body = ""
                if testo_i is not None and len(row) > testo_i:
                    body = row[testo_i] or ""

                text = ("\n".join(parts) + ("\n" if parts else "") + body).strip()
                if text:
                    out[cig] = text[:max_chars_per_cig]
                else:
                    out[cig] = ""

                if len(out) >= len(target):
                    break
    except Exception:
        # Return whatever we managed to read
        return out
    return out

def extract_duration_snippets(text: str, max_snippets: int = 10, window: int = 280) -> tuple[list[str], list[str]]:
    """Extract relevant snippets and candidate URLs from text."""
    if not text:
        return [], []
    t = str(text)

    urls = []
    for m in _URL_RE.finditer(t):
        u = m.group(1).strip().rstrip(').,;\'"')
        if u.lower().startswith('www.'):
            u = 'https://' + u
        urls.append(u)
    # de-dup preserving order
    seen = set()
    urls_unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            urls_unique.append(u)

    snippets = []
    for m in _DURATION_KEYWORDS_RE.finditer(t):
        start = max(0, m.start() - window)
        end = min(len(t), m.end() + window)
        snip = t[start:end].replace('\n', ' ').strip()
        if snip and snip not in snippets:
            snippets.append(snip)
        if len(snippets) >= max_snippets:
            break

    if not snippets:
        head = t[:2500].replace('\n', ' ').strip()
        if head:
            snippets = [head]
    return snippets[:max_snippets], urls_unique[:10]

def _safe_extract_text_from_html(html: str) -> str:
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        # Basic fallback: remove tags
        return re.sub(r'<[^>]+>', ' ', html)

def _safe_extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""
    try:
        from pypdf import PdfReader  # type: ignore
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages[:10]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""

def fetch_url_text_best_effort(url: str, timeout_s: int = 12, max_bytes: int = 5_000_000) -> tuple[str, str]:
    """
    Fetch a URL and extract text (HTML/PDF). Returns (text, error).
    Best-effort: short timeouts, size cap, no exceptions.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "", "unsupported_scheme"
    except Exception:
        return "", "invalid_url"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
    }

    try:
        r = requests.get(url, headers=headers, timeout=timeout_s, stream=True)
        ct = (r.headers.get("content-type") or "").lower()
        content = b""
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk:
                continue
            content += chunk
            if len(content) > max_bytes:
                return "", "size_cap_exceeded"

        if "application/pdf" in ct or url.lower().endswith(".pdf"):
            text = _safe_extract_text_from_pdf_bytes(content)
            return text, ""
        html = content.decode(r.encoding or "utf-8", errors="ignore")
        text = _safe_extract_text_from_html(html)
        return text, ""
    except Exception as e:
        return "", f"fetch_error:{type(e).__name__}"

def call_responses_api_structured(model: str, prompt: str, instructions: str, json_schema: dict) -> dict:
    """Call OpenAI Responses API requesting strict JSON schema output."""
    api_key = get_openai_api_key()
    if not api_key:
        return {"error": "Missing OPENAI_API_KEY"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "input": prompt,
        "instructions": instructions,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cig_enrichment",
                "schema": json_schema,
                "strict": True,
            }
        },
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=90,
        )
        try:
            data = response.json()
        except Exception:
            data = None

        if response.status_code >= 400:
            msg = None
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    msg = err.get("message") or err.get("type") or str(err)
                elif isinstance(err, str):
                    msg = err
            if not msg:
                try:
                    msg = (response.text or "").strip()
                except Exception:
                    msg = ""
            msg = (msg or "")[:800]
            return {"error": f"HTTP {response.status_code}: {msg}"}

        if not isinstance(data, dict):
            return {"error": "Invalid API response (non-JSON)"}

        # Prefer output_json if present
        if isinstance(data, dict) and "output" in data:
            for item in data["output"]:
                if item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if content.get("type") == "output_json" and isinstance(content.get("json"), dict):
                        return content["json"]
                    if content.get("type") == "output_text":
                        text = content.get("text", "")
                        try:
                            return json.loads(text)
                        except Exception:
                            return {"error": "Invalid JSON output", "raw": text}
        return {"error": "No structured output"}
    except requests.exceptions.RequestException as e:
        return {"error": f"request_error:{type(e).__name__}"}
    except Exception as e:
        return {"error": f"api_error:{type(e).__name__}"}

def _duration_to_days(value, unit):
    if value is None or unit is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    u = str(unit).lower().strip()
    if u in {"day", "days", "giorno", "giorni"}:
        return int(round(v))
    if u in {"month", "months", "mese", "mesi"}:
        return int(round(v * 30))
    if u in {"year", "years", "anno", "anni"}:
        return int(round(v * 365))
    return None

def _add_duration(start_dt: pd.Timestamp, value, unit: str) -> pd.Timestamp:
    u = str(unit).lower().strip()
    if u in {"day", "days", "giorno", "giorni"}:
        return start_dt + pd.to_timedelta(float(value), unit="D")
    if u in {"month", "months", "mese", "mesi"}:
        return start_dt + pd.DateOffset(months=int(round(float(value))))
    if u in {"year", "years", "anno", "anni"}:
        return start_dt + pd.DateOffset(years=int(round(float(value))))
    return start_dt + pd.to_timedelta(float(value), unit="D")

def enrich_cigs_via_llm(
    cigs: list[str],
    use_web: bool,
    force: bool,
    ttl_days: int = CIG_ENRICHMENT_TTL_DAYS_DEFAULT,
    progress_cb=None,
    save_every: int = 5,
) -> tuple[dict, list[dict]]:
    """
    Enrich a list of CIGs using local text + gpt-5-nano (structured).
    Returns (updated_cache, results_rows_for_ui).
    """
    cache = load_cig_enrichment_cache()
    items = cache.setdefault("items", {})
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    results = []

    def _short_err(err) -> str:
        s = str(err or "").replace("\n", " ").strip()
        return s[:90]

    def _is_fatal_err(err: str) -> bool:
        e = (err or "").lower()
        return (
            ("http 401" in e)
            or ("http 403" in e)
            or ("invalid_api_key" in e)
            or ("incorrect api key" in e)
            or ("you do not have access" in e and "model" in e)
            or ("model" in e and "not found" in e)
            or ("http 404" in e and "model" in e)
        )

    def _is_fresh_by_updated_at(item: dict) -> bool:
        if force:
            return False
        if not isinstance(item, dict) or not item.get("result"):
            return False
        updated_at = item.get("updated_at")
        if not updated_at:
            return False
        try:
            ts = pd.to_datetime(updated_at, utc=True, errors="coerce")
            if ts is pd.NaT:
                return False
            age_days = (pd.Timestamp.utcnow() - ts).days
            return age_days <= int(ttl_days)
        except Exception:
            return False

    # Avoid loading `testo_completo` for CIGs already fresh in cache (saves time/memory on Streamlit Cloud)
    valid_cigs = []
    to_process = []
    for cig in cigs:
        cig_n = _normalize_cig(cig)
        if not _is_valid_cig(cig_n):
            continue
        valid_cigs.append(cig_n)
        existing = items.get(cig_n, {})
        if _is_fresh_by_updated_at(existing):
            res = existing.get("result") or {}
            results.append({
                "cig": cig_n,
                "status": "cached",
                "confidence": res.get("confidence") if isinstance(res, dict) else None,
                "duration_base_days": res.get("duration_base_days") if isinstance(res, dict) else None,
                "duration_max_days": res.get("duration_max_days") if isinstance(res, dict) else None,
                "explicit_start_date": res.get("explicit_start_date") if isinstance(res, dict) else None,
                "explicit_end_date": res.get("explicit_end_date") if isinstance(res, dict) else None,
                "notes": res.get("notes", "") if isinstance(res, dict) else "",
            })
            if progress_cb:
                progress_cb(len(results), len(valid_cigs), cig_n, "cached")
        else:
            to_process.append(cig_n)

    texts = load_testo_completo_for_cigs(set(to_process))

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "duration_base": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string", "enum": ["days", "months", "years"]},
                },
                "required": ["value", "unit"],
            },
            "duration_max": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string", "enum": ["days", "months", "years"]},
                },
                "required": ["value", "unit"],
            },
            "explicit_start_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "explicit_end_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            "start_event": {"type": "string", "enum": ["aggiudicazione", "stipula", "consegna", "decorrenza", "unknown"]},
            "renewal_mentioned": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_snippets": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "notes": {"type": "string"},
        },
        "required": [
            "duration_base",
            "duration_max",
            "explicit_start_date",
            "explicit_end_date",
            "start_event",
            "renewal_mentioned",
            "confidence",
            "evidence_snippets",
            "notes",
        ],
    }

    instructions = (
        "Sei un analista di contratti pubblici. Devi estrarre durata e date SOLO se presenti nel testo fornito.\n"
        "Regole:\n"
        "- Non inventare mai date o durate.\n"
        "- Se trovi termini come 'quinquennale', 'triennale', 'biennale' traduci in anni (5/3/2).\n"
        "- Se trovi 'per 36 mesi' traduci in months=36.\n"
        "- duration_base = durata del periodo iniziale.\n"
        "- duration_max = durata massima includendo opzioni di rinnovo/proroga SOLO se il testo le quantifica.\n"
        "- Se c'è 'proroga tecnica' senza durata, metti renewal_mentioned=true ma duration_max=null.\n"
        "- explicit_start_date e explicit_end_date solo se esplicite in formato chiaro; altrimenti null.\n"
        "- start_event scegli tra aggiudicazione/stipula/consegna/decorrenza/unknown in base al contesto.\n"
        "- evidence_snippets: max 3 frammenti brevi dal testo (non più di ~200 caratteri ciascuno).\n"
        "- confidence 0..1 in base a chiarezza e coerenza.\n"
        "Rispondi SOLO con JSON conforme allo schema."
    )

    total = len(valid_cigs)
    processed = len(results)
    for cig_n in to_process:
        existing = items.get(cig_n, {})
        updated_at = existing.get("updated_at")
        input_hash_old = existing.get("input_hash")

        text = texts.get(cig_n, "")
        snippets, urls = extract_duration_snippets(text)

        web_sources = []
        if use_web and urls:
            for u in urls[:2]:
                fetched, err = fetch_url_text_best_effort(u)
                if fetched:
                    sn2, _ = extract_duration_snippets(fetched)
                    snippets.extend([s for s in sn2 if s not in snippets])
                    web_sources.append({"type": "url", "url": u})
                else:
                    web_sources.append({"type": "url", "url": u, "error": err})

        # Build input hash
        joined = "\n".join(snippets[:12])
        input_hash = hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()

        # TTL check (stricter: also requires same input_hash)
        fresh_enough = False
        if updated_at and not force and input_hash_old == input_hash and existing.get("result"):
            try:
                ts = pd.to_datetime(updated_at, utc=True, errors="coerce")
                if ts is not pd.NaT:
                    age_days = (pd.Timestamp.utcnow() - ts).days
                    fresh_enough = age_days <= int(ttl_days)
            except Exception:
                fresh_enough = False

        if fresh_enough:
            res = existing.get("result") or {}
            results.append({
                "cig": cig_n,
                "status": "cached",
                "confidence": res.get("confidence") if isinstance(res, dict) else None,
                "duration_base_days": res.get("duration_base_days") if isinstance(res, dict) else None,
                "duration_max_days": res.get("duration_max_days") if isinstance(res, dict) else None,
                "explicit_start_date": res.get("explicit_start_date") if isinstance(res, dict) else None,
                "explicit_end_date": res.get("explicit_end_date") if isinstance(res, dict) else None,
                "notes": res.get("notes", "") if isinstance(res, dict) else "",
            })
            processed += 1
            if progress_cb:
                progress_cb(processed, total, cig_n, "cached")
            continue

        if not snippets:
            items[cig_n] = {
                "updated_at": now_iso,
                "model": "gpt-5-nano",
                "input_hash": input_hash,
                "result": {
                    "duration_base_days": None,
                    "duration_max_days": None,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "start_event": "unknown",
                    "renewal_mentioned": False,
                    "confidence": 0.0,
                    "evidence": [],
                    "sources": [{"type": "local_text"}],
                    "notes": "Testo insufficiente per estrarre durata/scadenza.",
                },
                "errors": ["no_snippets"],
            }
            results.append({"cig": cig_n, "status": "no_text", "confidence": 0.0, "notes": "no snippets"})
            processed += 1
            if progress_cb:
                progress_cb(processed, total, cig_n, "no_text")
            if save_every and processed % int(save_every) == 0:
                cache["items"] = items
                save_cig_enrichment_cache(cache)
            continue

        prompt = (
            f"CIG: {cig_n}\n"
            "Ecco frammenti di testo (snippets) da cui estrarre durata/date:\n"
            + "\n\n".join([f"- {s}" for s in snippets[:12]])
        )

        # Retry with exponential backoff for transient errors
        last = None
        for attempt in range(3):
            try:
                last = call_responses_api_structured("gpt-5-nano", prompt, instructions, schema)
            except Exception as e:
                last = {"error": f"call_error:{type(e).__name__}"}
            if isinstance(last, dict) and not last.get("error"):
                break
            time.sleep(1.5 * (2 ** attempt))

        if not isinstance(last, dict) or last.get("error"):
            err_str = last.get("error") if isinstance(last, dict) else "unknown_error"
            items[cig_n] = {
                "updated_at": now_iso,
                "model": "gpt-5-nano",
                "input_hash": input_hash,
                "result": None,
                "errors": [err_str],
            }
            results.append({"cig": cig_n, "status": "error", "confidence": None, "notes": _short_err(err_str)})
            processed += 1
            if progress_cb:
                progress_cb(processed, total, cig_n, f"error: {_short_err(err_str)}")
            if save_every and processed % int(save_every) == 0:
                cache["items"] = items
                save_cig_enrichment_cache(cache)
            if _is_fatal_err(str(err_str)):
                # Stop early: likely all remaining will fail too
                results.append({"cig": cig_n, "status": "fatal", "confidence": None, "notes": _short_err(err_str)})
                break
            continue

        # Normalize into cache result
        duration_base = last.get("duration_base")
        duration_max = last.get("duration_max")
        base_days = _duration_to_days(duration_base.get("value"), duration_base.get("unit")) if isinstance(duration_base, dict) else None
        max_days = _duration_to_days(duration_max.get("value"), duration_max.get("unit")) if isinstance(duration_max, dict) else None

        evidence = last.get("evidence_snippets") if isinstance(last.get("evidence_snippets"), list) else []
        evidence = [str(x)[:250] for x in evidence][:3]

        result_obj = {
            "duration_base_days": base_days,
            "duration_max_days": max_days,
            "explicit_start_date": last.get("explicit_start_date"),
            "explicit_end_date": last.get("explicit_end_date"),
            "start_event": last.get("start_event", "unknown"),
            "renewal_mentioned": bool(last.get("renewal_mentioned", False)),
            "confidence": float(last.get("confidence", 0.0)) if last.get("confidence") is not None else 0.0,
            "evidence": evidence,
            "sources": [{"type": "local_text"}] + web_sources,
            "notes": str(last.get("notes", ""))[:500],
        }

        items[cig_n] = {
            "updated_at": now_iso,
            "model": "gpt-5-nano",
            "input_hash": input_hash,
            "result": result_obj,
            "errors": [],
        }

        results.append({
            "cig": cig_n,
            "status": "ok",
            "confidence": result_obj["confidence"],
            "duration_base_days": base_days,
            "duration_max_days": max_days,
            "explicit_start_date": result_obj["explicit_start_date"],
            "explicit_end_date": result_obj["explicit_end_date"],
            "notes": result_obj["notes"],
        })
        processed += 1
        if progress_cb:
            progress_cb(processed, total, cig_n, "ok")
        if save_every and processed % int(save_every) == 0:
            cache["items"] = items
            save_cig_enrichment_cache(cache)

    cache["items"] = items
    save_cig_enrichment_cache(cache)
    return cache, results

def load_favorites():
    """Load saved favorite charts"""
    if FAVORITES_PATH.exists():
        try:
            with open(FAVORITES_PATH) as f:
                return json.load(f)
        except:
            return []
    return []

def save_favorites(favorites):
    """Save favorite charts"""
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAVORITES_PATH, 'w') as f:
        json.dump(favorites, f, indent=2, default=str)

def add_favorite(chart_config):
    """Add a chart to favorites"""
    favorites = load_favorites()
    chart_id = hashlib.md5(json.dumps(chart_config, sort_keys=True, default=str).encode()).hexdigest()[:8]
    chart_config['id'] = chart_id
    chart_config['created_at'] = datetime.now().isoformat()
    # Check if already exists
    if not any(f.get('id') == chart_id for f in favorites):
        favorites.append(chart_config)
        save_favorites(favorites)
    return chart_id

def remove_favorite(chart_id):
    """Remove a chart from favorites"""
    favorites = load_favorites()
    favorites = [f for f in favorites if f.get('id') != chart_id]
    save_favorites(favorites)

def get_openai_api_key():
    """Get OpenAI API key from session state or environment"""
    # Prima controlla session state (inserita dall'utente)
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    # Fallback a variabile d'ambiente
    return os.getenv('OPENAI_API_KEY')

def call_responses_api(prompt: str, instructions: str, model: str = "gpt-5-nano", timeout_s: int = 60) -> str | None:
    """Call OpenAI Responses API and return output_text (or None)."""
    api_key = get_openai_api_key()
    if not api_key:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {"model": model, "input": prompt, "instructions": instructions}

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
        try:
            data = response.json()
        except Exception:
            data = None

        if response.status_code >= 400:
            msg = None
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    msg = err.get("message") or err.get("type") or str(err)
                elif isinstance(err, str):
                    msg = err
            if not msg:
                try:
                    msg = (response.text or "").strip()
                except Exception:
                    msg = ""
            msg = (msg or "")[:800]
            return f"# Errore API: HTTP {response.status_code}: {msg}"

        if not isinstance(data, dict):
            return "# Errore API: risposta non JSON"

        # Extract text from response
        if 'output' in data:
            for item in data['output']:
                if item.get('type') == 'message':
                    for content in item.get('content', []):
                        if content.get('type') == 'output_text':
                            return content.get('text', '')
        return None
    except Exception as e:
        return f"# Errore API: {str(e)}"

def analyze_prompt(prompt: str, df_info: str) -> dict:
    """Step 1: Analyze prompt and suggest fields/values/chart type"""

    instructions = """Sei un esperto di data analysis. Analizza la richiesta dell'utente e identifica:
1. Le colonne del dataset da usare
2. I valori specifici menzionati (es. nomi aziende, anni, regioni) - usa pattern parziali con *
3. Il tipo di grafico più adatto
4. Eventuali filtri da applicare

IMPORTANTE per la ricerca testuale:
- Se l'utente cerca "AEC" cerca in supplier_name o aggiudicatario con str.contains('AEC', case=False, na=False)
- Usa sempre na=False per evitare errori con valori mancanti
- Usa pattern parziali per trovare nomi simili

Rispondi SOLO in formato JSON con questa struttura:
{
    "columns": ["lista", "colonne", "da usare"],
    "values": {"colonna": ["valori", "specifici"]},
    "search_patterns": {"colonna": "pattern*"},
    "chart_type": "bar/line/scatter/pie/treemap/heatmap",
    "chart_description": "Descrizione del grafico proposto",
    "filters": {"colonna": "valore"},
    "aggregation": "sum/mean/count"
}

NON aggiungere commenti, SOLO JSON valido.

Colonne disponibili:
""" + df_info

    try:
        result = call_responses_api(prompt, instructions, model="gpt-5.1-codex-mini")

        if not result or result.startswith("# Errore"):
            return {"error": result or "Nessuna risposta"}

        # Clean JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]

        return json.loads(result.strip())
    except Exception as e:
        return {"error": str(e)}

def generate_chart_code(prompt: str, df_info: str, analysis: dict = None) -> str:
    """Step 2: Generate chart code based on analysis using Responses API"""

    instructions = """Sei un esperto di data visualization con Plotly. Genera codice Python per creare grafici.

REGOLE CRITICHE:
1. Genera SOLO codice Python valido che usa plotly.express o plotly.graph_objects
2. Il DataFrame si chiama `df` ed è già disponibile
3. Il codice deve finire con la variabile `fig` (la figura Plotly)
4. NON usare st.plotly_chart, restituisci solo `fig`
5. Usa colori professionali e layout pulito
6. Aggiungi sempre titolo e labels
7. NON includere import, sono già fatti (px, go, pd, np disponibili)
8. Il codice deve essere eseguibile direttamente

GESTIONE VALORI MANCANTI (IMPORTANTE!):
- Per ricerche testuali usa SEMPRE: df[df['colonna'].str.contains('pattern', case=False, na=False)]
- Prima di filtrare, rimuovi NaN: df_clean = df.dropna(subset=['colonna_filtro'])
- Per filtri booleani usa: df[df['colonna'].fillna(False) == valore]
- Mai usare mask con NaN direttamente

RICERCA NOMI:
- Per cercare aziende/fornitori usa str.contains() con na=False
- Esempio: df[df['supplier_name'].str.contains('AEC', case=False, na=False)]

Colonne disponibili nel DataFrame:
""" + df_info

    # Add analysis context if available
    if analysis and not analysis.get('error'):
        instructions += f"\n\nAnalisi preliminare:\n{json.dumps(analysis, indent=2)}"

    try:
        code = call_responses_api(f"Crea questo grafico: {prompt}", instructions, model="gpt-5.1-codex-mini", timeout_s=90)

        if not code or code.startswith("# Errore"):
            return code or "# Errore: Nessuna risposta"

        # Clean up code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()
    except Exception as e:
        return f"# Errore: {str(e)}"

def execute_chart_code(code: str, df: pd.DataFrame):
    """Safely execute chart code and return figure"""
    try:
        local_vars = {'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np, 'make_subplots': make_subplots}
        exec(code, local_vars)
        return local_vars.get('fig'), None
    except Exception as e:
        return None, str(e)

def _compact_record_for_ai(record: dict, max_chars: int = 3500) -> str:
    """Compact a record dict into a readable, size-bounded text block for LLM prompts."""
    if not isinstance(record, dict):
        return ""
    order = [
        "chiave", "cig", "ocid", "fonte",
        "oggetto", "tender_title", "tender_description",
        "ente_appaltante", "buyer_name",
        "aggiudicatario", "supplier_name",
        "importo_aggiudicazione", "award_amount",
        "sconto", "procedura", "categoria", "_categoria", "quick_category",
        "regione", "comune", "buyer_locality",
        "data_aggiudicazione", "award_date",
        "data_scadenza", "durata_appalto",
        # scadenze calcolate (se presenti)
        "scadenza_contratto", "scadenza_contratto_max", "scadenza_fonte",
        "llm_confidence", "llm_notes",
        "anac_url",
    ]
    seen = set()
    lines = []
    for k in order + [k for k in record.keys() if k not in order]:
        if k in seen:
            continue
        seen.add(k)
        v = record.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "nat"}:
            continue
        if len(s) > 500:
            s = s[:500] + "…"
        lines.append(f"- {k}: {s}")
        if sum(len(x) + 1 for x in lines) >= max_chars:
            break
    out = "\n".join(lines)
    return out[:max_chars]

def ai_analyze_gara(record: dict, question: str | None = None, model: str = "gpt-5-nano") -> str | None:
    """Run an AI analysis on a single gara/contratto record."""
    if not get_openai_api_key():
        return None

    record_txt = _compact_record_for_ai(record)
    user_q = (question or "").strip()
    prompt = (
        "Analizza questa singola gara/contratto (record dal dataset) e produci un report.\n\n"
        "RECORD:\n"
        f"{record_txt}\n\n"
        + (f"DOMANDA UTENTE:\n{user_q}\n\n" if user_q else "")
        + "Vincoli:\n"
        "- Non inventare dati non presenti nel record.\n"
        "- Se un campo è mancante, scrivi 'non disponibile'.\n"
        "- Se noti incoerenze (date future/passate, importi strani), segnala.\n"
    )

    instructions = (
        "Sei un analista di gare pubbliche. Rispondi in italiano in MARKDOWN, con sezioni:\n"
        "1) Sintesi (2-3 righe)\n"
        "2) Dati chiave estratti (bullet)\n"
        "3) Scadenza/tempo (se applicabile): interpreta data_scadenza/CONSIP/durata_appalto/scadenza_contratto e indica la fonte\n"
        "4) Ambiguità/Rischi/Ipotesi (solo se presenti, bullet)\n"
        "5) Azioni consigliate per verifica (max 6 bullet)\n"
        "Tono professionale, conciso."
    )

    return call_responses_api(prompt, instructions, model=model, timeout_s=60)

def _ai_select_label_from_row(row: dict, id_value: str) -> str:
    """Build a compact label for a gara selector."""
    if not isinstance(row, dict):
        return str(id_value)
    obj = row.get("oggetto") or row.get("tender_title") or row.get("tender_description") or ""
    obj = str(obj).replace("\n", " ").strip()
    if len(obj) > 90:
        obj = obj[:90] + "…"
    buyer = row.get("buyer_name") or row.get("ente_appaltante") or ""
    buyer = str(buyer).strip()
    supplier = row.get("supplier_name") or row.get("aggiudicatario") or ""
    supplier = str(supplier).strip()
    bits = [str(id_value)]
    if obj:
        bits.append(obj)
    if buyer:
        bits.append(buyer[:40] + ("…" if len(buyer) > 40 else ""))
    if supplier:
        bits.append(supplier[:40] + ("…" if len(supplier) > 40 else ""))
    return " | ".join(bits[:4])

def get_current_filters():
    """Get current sidebar filter values from session state"""
    filters = {}
    # Mappa dei filtri con i loro nomi visualizzabili
    filter_map = {
        'fonte_sel': ('Fonte', None),
        'anno_sel': ('Anno', None),
        'regione_sel': ('Regione', None),
        'categoria_sel': ('Categoria', None),
        'procedura_sel': ('Procedura', None),
        'tipo_appalto_sel': ('Tipologia Contratto', None),
        'sottocategoria_sel': ('Sottocategoria', None),
    }
    for key, (label, default) in filter_map.items():
        if key in st.session_state:
            val = st.session_state[key]
            if val is not None and val != default:
                filters[label] = val
    return filters

def render_chart_with_save(fig, chart_title: str, chart_description: str, chart_key: str):
    """Render a Plotly chart with save to favorites button"""
    col_chart, col_btn = st.columns([20, 1])
    with col_chart:
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_key}")
    with col_btn:
        # Check if already in favorites
        favorites = load_favorites()
        is_favorite = any(f.get('id') == chart_key for f in favorites)

        if is_favorite:
            btn_label = "★"
            btn_help = "Già nei preferiti"
        else:
            btn_label = "☆"
            btn_help = "Salva nei preferiti"

        if st.button(btn_label, key=f"fav_btn_{chart_key}", help=btn_help):
            if not is_favorite:
                # Get current filters
                current_filters = get_current_filters()
                # Save chart config (serialized figure) with filters
                chart_config = {
                    'type': 'standard',
                    'title': chart_title,
                    'description': chart_description,
                    'fig_json': fig.to_json(),
                    'filters': current_filters
                }
                add_favorite(chart_config)
                st.toast(f"✅ '{chart_title}' aggiunto ai preferiti!")
                st.rerun()
            else:
                remove_favorite(chart_key)
                st.toast(f"❌ '{chart_title}' rimosso dai preferiti")
                st.rerun()

# Load data
@st.cache_data
def load_data():
    # Path per Streamlit Cloud (file nella cartella data/)
    cloud_path = Path(__file__).parent / "data" / "data.json"
    local_path = Path(__file__).parent.parent / "data" / "output" / "dashboard" / "data.json"

    if cloud_path.exists():
        with open(cloud_path) as f:
            return json.load(f)
    elif local_path.exists():
        with open(local_path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=3600)  # Cache per 1 ora, poi ricarica - v2 con 17k sconti
def load_raw_data():
    # Path per Streamlit Cloud deployment (file compresso)
    gz_path = Path(__file__).parent / "data" / "gare_unificate.csv.gz"
    unified_path = Path(__file__).parent.parent / "data" / "output" / "categorie" / "gare_unificate.csv"
    old_path = Path(__file__).parent.parent / "data" / "output" / "categorie" / "gare_filtrate_tutte.csv"

    # Colonne da escludere per risparmiare memoria (testo_completo è molto grande)
    # La carichiamo solo se necessario
    cols_to_exclude = ['testo_completo']

    # Tipi di dato ottimizzati per risparmiare memoria
    # Nota: evitare 'category' per colonne con molti valori unici (causa errori Arrow/PyArrow)
    # Usare 'category' solo per colonne con pochi valori ripetuti (es. regione, fonte, procedura)
    dtype_opt = {
        'chiave': 'str',  # Molti valori unici - NON usare category
        'cig': 'str',
        'ocid': 'str',
        'oggetto': 'str',
        'importo_aggiudicazione': 'float64',
        'sconto': 'float64',
        'data_aggiudicazione': 'str',
        'data_scadenza': 'str',
        'durata_appalto': 'float64',
        'fonte': 'category',  # Pochi valori (Gazzetta, OCDS, etc)
        'categoria': 'category',  # ~20 categorie
        'categoria_originale': 'str',
        'sottocategoria': 'str',  # Troppi valori per category
        'quick_category': 'str',
        'procedura': 'str',  # Molti valori unici - normalizzare dopo
        'procedura_originale': 'str',
        'criterio_aggiudicazione': 'str',
        'procurement_method': 'str',
        'procurement_method_details': 'str',
        'tipo_appalto': 'category',  # ~5 tipi
        'tipo_appalto_originale': 'str',
        'regione': 'str',  # Normalizzare dopo
        'comune': 'str',  # Troppi comuni per category
        'buyer_locality': 'str',
        'supplier_name': 'str',
        'anno': 'float64',  # Float per gestire NaN, convertire dopo
        'offerte_ricevute': 'float64',  # Float per gestire NaN
        'num_lotti': 'float64',  # Float per gestire NaN
        'lotto': 'str',
        'codice_gruppo': 'str',
        'filter_confidence': 'float64',
        'tipo_accordo': 'str',
        'edizione': 'str',
        'tipo_intervento': 'str',
        'tipo_impianto': 'str',
        'tipo_illuminazione': 'str',
        'tipo_energia': 'str',
        'tipo_efficientamento': 'str',
        'cup': 'str',
        'cpv_code': 'str',
        'cpv_description': 'str',
        'edizione_consip': 'str',
        'tipo_accordo_consip': 'str',
    }

    try:
        # Prima prova il file compresso (Streamlit Cloud)
        if gz_path.exists():
            # Leggi prima le colonne disponibili
            df_cols = pd.read_csv(gz_path, compression='gzip', nrows=0)
            usecols = [c for c in df_cols.columns if c not in cols_to_exclude]
            dtype_use = {c: dtype_opt.get(c, 'str') for c in usecols}
            df = pd.read_csv(gz_path, compression='gzip', usecols=usecols, dtype=dtype_use, low_memory=False)
        elif unified_path.exists():
            df_cols = pd.read_csv(unified_path, nrows=0)
            usecols = [c for c in df_cols.columns if c not in cols_to_exclude]
            dtype_use = {c: dtype_opt.get(c, 'str') for c in usecols}
            df = pd.read_csv(unified_path, usecols=usecols, dtype=dtype_use, low_memory=False)
        elif old_path.exists():
            df_cols = pd.read_csv(old_path, nrows=0)
            usecols = [c for c in df_cols.columns if c not in cols_to_exclude]
            dtype_use = {c: dtype_opt.get(c, 'str') for c in usecols}
            df = pd.read_csv(old_path, usecols=usecols, dtype=dtype_use, low_memory=False)
        else:
            st.error("Nessun file dati trovato!")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Errore caricamento dati: {e}")
        return pd.DataFrame()

    if df is not None and len(df) > 0:
        # Standardizza nomi colonne per retrocompatibilità
        if 'importo_aggiudicazione' in df.columns:
            df['award_amount'] = df['importo_aggiudicazione']
        if 'oggetto' in df.columns:
            df['tender_title'] = df['oggetto']
        # tender_description non più disponibile (esclusa per memoria)
        df['tender_description'] = ''
        if 'ente_appaltante' in df.columns:
            df['buyer_name'] = df['ente_appaltante']
        if 'data_aggiudicazione' in df.columns:
            df['award_date'] = pd.to_datetime(df['data_aggiudicazione'], errors='coerce')
        if 'categoria' in df.columns:
            # Normalizza categoria: uppercase e strip per evitare duplicati (es. "ILLUMINAZIONE" vs "illuminazione")
            df['_categoria'] = df['categoria'].str.upper().str.strip()
        if 'aggiudicatario' in df.columns:
            df['supplier_name'] = df['aggiudicatario']
    return df

@st.cache_data
def load_consip_data():
    # Path per Streamlit Cloud (file nella cartella data/)
    cloud_path = Path(__file__).parent / "data" / "ServizioLuce.xlsx"
    # Path locale alternativo
    local_path = Path(__file__).parent.parent / "data" / "output" / "ServizioLuce.xlsx"

    try:
        if cloud_path.exists():
            return pd.read_excel(cloud_path)
        elif local_path.exists():
            return pd.read_excel(local_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Errore caricamento dati CONSIP: {e}")
        return pd.DataFrame()

@st.cache_data
def load_comuni_istat():
    """Carica coordinate e info di tutti i comuni italiani da file ISTAT."""
    istat_path = Path(__file__).parent / "data" / "comuni_istat.csv"
    if not istat_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(istat_path, dtype=str)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    return df

def _normalize_comune_name(s):
    """Normalizza nome comune per matching: lowercase, strip, rimuovi accenti."""
    if pd.isna(s) or str(s).strip() == '' or str(s).lower() in ('nan', 'none'):
        return ''
    import unicodedata
    s = str(s).strip()
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()

@st.cache_data
def _build_istat_lookup(_comuni_istat_df):
    """Costruisce dizionari di lookup per geocoding e backfill regione."""
    if _comuni_istat_df is None or len(_comuni_istat_df) == 0:
        return {}, {}
    # Lookup: comune_normalized -> (lat, lon, regione, comune_ufficiale)
    geo_lookup = {}
    regione_lookup = {}
    for _, row in _comuni_istat_df.iterrows():
        key = str(row.get('comune_normalized', '')).strip()
        if not key:
            continue
        geo_lookup[key] = (row['lat'], row['lon'], row.get('regione', ''), row.get('comune', ''))
        regione_lookup[key] = row.get('regione', '')
    # Alias comuni con nomi usati comunemente diversi dal nome ISTAT ufficiale
    _aliases = {
        'reggio emilia': "reggio nell'emilia",
        'reggio calabria': 'reggio di calabria',
        'forli': "forli'", 'cesena': 'cesena',
        'massa': 'massa', 'carrara': 'carrara',
    }
    for alias, official in _aliases.items():
        if official in geo_lookup and alias not in geo_lookup:
            geo_lookup[alias] = geo_lookup[official]
            regione_lookup[alias] = regione_lookup[official]
    return geo_lookup, regione_lookup

def _geocode_comune(nome, geo_lookup):
    """Restituisce (lat, lon) per un comune usando il lookup ISTAT."""
    key = _normalize_comune_name(nome)
    if not key:
        return None, None
    hit = geo_lookup.get(key)
    if hit:
        return hit[0], hit[1]
    # Fuzzy match per varianti di nomi (es. "San Donato Milanese" vs "San Donato")
    from rapidfuzz import fuzz, process
    candidates = list(geo_lookup.keys())
    if not candidates:
        return None, None
    match = process.extractOne(key, candidates, scorer=fuzz.ratio, score_cutoff=85)
    if match:
        hit = geo_lookup[match[0]]
        return hit[0], hit[1]
    return None, None

data = load_data()
raw_df = load_raw_data()
consip_raw_df = load_consip_data()
comuni_istat_df = load_comuni_istat()
_geo_lookup, _regione_lookup = _build_istat_lookup(comuni_istat_df)

# Verifica che i dati siano stati caricati correttamente
if raw_df is None or len(raw_df) == 0:
    st.error("⚠️ Impossibile caricare i dati. Verificare che i file siano presenti nella cartella data/")
    st.stop()

# Preprocess raw data
if 'award_date' in raw_df.columns:
    raw_df['award_date'] = pd.to_datetime(raw_df['award_date'], errors='coerce')
if 'anno' not in raw_df.columns and 'award_date' in raw_df.columns:
    raw_df['anno'] = raw_df['award_date'].dt.year
if 'award_date' in raw_df.columns:
    raw_df['mese'] = raw_df['award_date'].dt.month
else:
    raw_df['mese'] = np.nan

# Converti colonne numeriche - forza conversione
if 'award_amount' in raw_df.columns:
    raw_df['award_amount'] = pd.to_numeric(raw_df['award_amount'], errors='coerce')
if 'sconto' in raw_df.columns:
    raw_df['sconto'] = pd.to_numeric(raw_df['sconto'], errors='coerce')

# Calcola sconto se non esiste o tutto NaN
if 'sconto' not in raw_df.columns or raw_df['sconto'].isna().all():
    if 'tender_amount' in raw_df.columns and 'award_amount' in raw_df.columns:
        raw_df['tender_amount'] = pd.to_numeric(raw_df['tender_amount'], errors='coerce')
        raw_df['sconto'] = ((raw_df['tender_amount'] - raw_df['award_amount']) / raw_df['tender_amount'] * 100).clip(0, 100)

# Pulisci sconti invalidi (negativi, 0, o > 100)
# Sconti = 0 o null non vanno considerati nelle analisi
if 'sconto' in raw_df.columns:
    raw_df.loc[raw_df['sconto'] <= 0, 'sconto'] = np.nan  # 0 e negativi -> NaN
    raw_df.loc[raw_df['sconto'] > 100, 'sconto'] = np.nan

# Normalizza nomi regioni (unifica varianti)
if 'regione' in raw_df.columns:
    regioni_map = {
        'Emilia romagna': 'Emilia-Romagna',
        'Friuli venezia giulia': 'Friuli-Venezia Giulia',
        'Valle d\'aosta': 'Valle d\'Aosta',
        'Provincia autonoma di trento': 'Trentino-Alto Adige',
        'Centrale': 'Lazio',  # Assume "Centrale" = Roma/Lazio
        'Non classificato': np.nan,  # Rimuovi non classificati
        'nan': np.nan,
    }
    # Converti a string prima di replace per evitare warning con CategoricalDtype
    raw_df['regione'] = raw_df['regione'].astype(str).replace(regioni_map).astype('category')

# Backfill regione da ISTAT (per record con comune ma senza regione)
if _regione_lookup and 'comune' in raw_df.columns:
    # Converti a string per evitare errori con CategoricalDtype durante il backfill
    raw_df['regione'] = raw_df['regione'].astype(str)
    missing_regione = raw_df['regione'].isin(['nan', '', 'None', '<NA>']) | raw_df['regione'].isna()
    if missing_regione.any():
        comuni_norm = raw_df.loc[missing_regione, 'comune'].apply(_normalize_comune_name)
        regioni_fill = comuni_norm.map(_regione_lookup)
        raw_df.loc[missing_regione, 'regione'] = regioni_fill
        # Se anche buyer_locality presente, usa come fallback
        if 'buyer_locality' in raw_df.columns:
            still_missing = raw_df['regione'].isin(['nan', '', 'None', '<NA>']) | raw_df['regione'].isna()
            locality_norm = raw_df.loc[still_missing, 'buyer_locality'].apply(_normalize_comune_name)
            regioni_fill2 = locality_norm.map(_regione_lookup)
            raw_df.loc[still_missing, 'regione'] = regioni_fill2
    # Riconverti a category dopo backfill
    raw_df['regione'] = raw_df['regione'].replace({'nan': np.nan, 'None': np.nan, '': np.nan, '<NA>': np.nan}).astype('category')

# Normalizza procedure (estrai nome pulito da formato "COD:XX ; TITLE:Nome")
if 'procedura' in raw_df.columns:
    def normalize_procedura(x):
        if pd.isna(x) or str(x) == 'nan':
            return np.nan
        x = str(x)
        # Estrai solo il TITLE se presente
        if 'TITLE:' in x:
            x = x.split('TITLE:')[-1].strip()
        # Normalizza nomi comuni
        proc_map = {
            'Procedura aperta': 'Procedura Aperta',
            'Aperta': 'Procedura Aperta',
            'PROCEDURA APERTA': 'Procedura Aperta',
            'AFFIDAMENTO DIRETTO': 'Affidamento Diretto',
            'PROCEDURA NEGOZIATA PER AFFIDAMENTI SOTTO SOGLIA': 'Procedura Negoziata Sotto Soglia',
            'AFFIDAMENTO DIRETTO IN ADESIONE AD ACCORDO QUADRO/CONVENZIONE': 'Adesione Accordo Quadro',
            'PROCEDURA NEGOZIATA SENZA PREVIA PUBBLICAZIONE': 'Procedura Negoziata',
            'PROCEDURA RISTRETTA': 'Procedura Ristretta',
        }
        return proc_map.get(x, x[:40] if len(x) > 40 else x)

    raw_df['procedura'] = raw_df['procedura'].apply(normalize_procedura)

# Normalizza tipo appalto in classi standard e scarta stringhe che sembrano procedure
if 'tipo_appalto' in raw_df.columns:
    def normalize_tipo_appalto(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == '' or s.lower() == 'nan':
            return np.nan
        s_low = s.lower()
        # se contiene termini di procedura, non è una tipologia contrattuale valida
        if any(k in s_low for k in ['procedura', 'affidamento', 'negoziat', 'ristrett', 'apert', 'adesione']):
            return np.nan
        if 'lavor' in s_low:
            return 'Lavori'
        if 'fornitur' in s_low or 'fornit' in s_low:
            return 'Forniture'
        if 'concession' in s_low:
            return 'Concessioni'
        if any(k in s_low for k in ['serviz', 'manutenz', 'gestione', 'nolo', 'supporto']):
            return 'Servizi'
        return 'Altro'

    raw_df['tipo_appalto_norm'] = raw_df['tipo_appalto'].apply(normalize_tipo_appalto).astype('category')

# Sidebar filters
st.sidebar.title("🔍 Filtri")

# ==================== SIDEBAR: OPENAI API KEY (SESSION) ====================
with st.sidebar.expander("🤖 AI", expanded=False):
    if get_openai_api_key():
        st.success("API Key presente (sessione corrente).")
        if st.button("🧹 Rimuovi API Key", key="sidebar_clear_openai_key"):
            st.session_state.openai_api_key = ""
            st.rerun()
    else:
        st.caption("La chiave viene salvata solo per questa sessione Streamlit.")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="sidebar_openai_key_input",
            help="Inseriscila una sola volta: verrà usata per Enrichment (Scadenze) + Analisi AI + AI Charts/Chat.",
        )
        if st.button("✅ Salva API Key", type="primary", key="sidebar_save_openai_key"):
            if api_key_input and api_key_input.startswith("sk-"):
                st.session_state.openai_api_key = api_key_input
                st.success("✅ Salvata. La pagina si aggiorna…")
                st.rerun()
            else:
                st.error("❌ API Key non valida (deve iniziare con 'sk-').")

# CSS per migliorare l'aspetto dei selectbox (NO f-string: evita crash da parentesi graffe CSS)
st.sidebar.markdown(
    """
<style>
/* Stile dropdown più chiaro */
div[data-baseweb="select"] > div {
    border: 1px solid #ccc;
    border-radius: 4px;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--brand-green);
}
/* Radio buttons orizzontali compatti */
div.row-widget.stRadio > div {
    flex-direction: row;
    flex-wrap: wrap;
    gap: 0.5rem;
}
div.row-widget.stRadio > div > label {
    padding: 0.25rem 0.5rem;
    background: var(--brand-surface);
    border-radius: 4px;
    margin: 0;
    font-size: 0.85rem;
}
div.row-widget.stRadio > div > label:has(input:checked) {
    background: var(--brand-green);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

# Fonte filter (radio buttons - poche opzioni)
if 'fonte' in raw_df.columns:
    fonti_disponibili = sorted(raw_df['fonte'].dropna().unique().tolist())
    fonte_options = ["Tutte"] + fonti_disponibili
    fonte_sel_label = st.sidebar.radio("Fonte dati", fonte_options, horizontal=True)
    fonte_sel = None if fonte_sel_label == "Tutte" else fonte_sel_label
else:
    fonte_sel = None

# Anno filter (selectbox - molte opzioni ordinate)
anni = [None] + sorted([int(y) for y in raw_df['anno'].dropna().unique() if 2015 <= y <= 2025], reverse=True)
anno_sel = st.sidebar.selectbox("📅 Anno", anni, format_func=lambda x: "Tutti gli anni" if x is None else str(x))

# Regione filter (selectbox - molte opzioni)
if 'regione' in raw_df.columns and raw_df['regione'].notna().any():
    regioni_df = sorted([str(r) for r in raw_df['regione'].dropna().unique() if str(r) not in ('nan', 'None', '', '<NA>')])
    regioni = [None] + regioni_df
else:
    regioni = [None] + [r['Regione'] for r in data['geo']]
regione_sel = st.sidebar.selectbox("🗺️ Regione", regioni, format_func=lambda x: "Tutte le regioni" if x is None else x)

# Tipo Appalto filter (usa colonna normalizzata se presente)
if 'tipo_appalto_norm' in raw_df.columns and raw_df['tipo_appalto_norm'].notna().any():
    tipo_list = sorted(raw_df['tipo_appalto_norm'].dropna().unique().tolist())
    tipo_options = ["Tutti"] + tipo_list
    tipo_sel_label = st.sidebar.radio("Tipologia contratto", tipo_options, horizontal=True)
    tipo_appalto_sel = None if tipo_sel_label == "Tutti" else tipo_sel_label
elif 'tipo_appalto' in raw_df.columns and raw_df['tipo_appalto'].notna().any():
    tipo_list = sorted(raw_df['tipo_appalto'].dropna().unique().tolist())
    tipo_options = ["Tutti"] + tipo_list
    tipo_sel_label = st.sidebar.radio("Tipologia contratto", tipo_options, horizontal=True)
    tipo_appalto_sel = None if tipo_sel_label == "Tutti" else tipo_sel_label
else:
    tipo_appalto_sel = None

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtri avanzati**")

# Categoria filter (selectbox - molte opzioni)
if 'categoria' in raw_df.columns and raw_df['categoria'].notna().any():
    cat_list = sorted(raw_df['categoria'].dropna().unique().tolist())
    categorie = [None] + cat_list
elif '_categoria' in raw_df.columns and raw_df['_categoria'].notna().any():
    cat_list = sorted(raw_df['_categoria'].dropna().unique().tolist())
    categorie = [None] + cat_list
else:
    categorie = [None] + data['filter_options']['categorie_macro']
categoria_sel = st.sidebar.selectbox("🏷️ Categoria", categorie, format_func=lambda x: "Tutte le categorie" if x is None else x)

# Sottocategoria filter (dinamico basato su categoria)
if 'quick_category' in raw_df.columns:
    if categoria_sel and 'categoria' in raw_df.columns:
        sottocategorie_list = sorted(raw_df[raw_df['categoria'] == categoria_sel]['quick_category'].dropna().unique().tolist())
    else:
        sottocategorie_list = sorted(raw_df['quick_category'].dropna().unique().tolist())
    sottocategorie = [None] + sottocategorie_list
else:
    sottocategorie = [None]
sottocategoria_sel = st.sidebar.selectbox("📂 Sottocategoria", sottocategorie, format_func=lambda x: "Tutte" if x is None else x)

# Procedura filter (selectbox - molte opzioni)
if 'procedura' in raw_df.columns and raw_df['procedura'].notna().any():
    proc_list = sorted(raw_df['procedura'].dropna().unique().tolist())
    procedure = [None] + proc_list
    procedura_sel = st.sidebar.selectbox("⚖️ Procedura", procedure, format_func=lambda x: "Tutte le procedure" if x is None else x)
else:
    procedura_sel = None

# Apply filters to raw data
filtered_df = raw_df.copy()
if fonte_sel:
    filtered_df = filtered_df[filtered_df['fonte'] == fonte_sel]
if anno_sel:
    filtered_df = filtered_df[filtered_df['anno'] == anno_sel]
if regione_sel and 'regione' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['regione'].astype(str) == str(regione_sel)]
if categoria_sel:
    # Usa la nuova colonna 'categoria' normalizzata, altrimenti fallback a '_categoria'
    if 'categoria' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['categoria'] == categoria_sel]
    elif '_categoria' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['_categoria'] == categoria_sel]
if procedura_sel and 'procedura' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['procedura'] == procedura_sel]
if tipo_appalto_sel:
    if 'tipo_appalto_norm' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['tipo_appalto_norm'] == tipo_appalto_sel]
    elif 'tipo_appalto' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['tipo_appalto'] == tipo_appalto_sel]
if sottocategoria_sel and 'quick_category' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['quick_category'] == sottocategoria_sel]

# Crea una chiave unica per i filtri attivi - serve per resettare il multiselect
filter_key = f"{fonte_sel}_{anno_sel}_{regione_sel}_{categoria_sel}_{procedura_sel}_{tipo_appalto_sel}_{sottocategoria_sel}"

# Title
st.title("📊 Dashboard Gare Pubbliche Italiane")

# Mostra info sulle fonti dati se disponibili
if 'fonte' in raw_df.columns:
    fonti_counts = raw_df['fonte'].value_counts()
    fonti_str = " | ".join([f"{fonte}: {count:,}".replace(",", ".") for fonte, count in fonti_counts.items()])
    st.markdown(f"**Analisi di {len(raw_df):,} contratti pubblici** ({fonti_str}) | Dati filtrati: {len(filtered_df):,} gare".replace(",", "."))
else:
    st.markdown(f"**Analisi di {len(raw_df):,} contratti pubblici** | Dati filtrati: {len(filtered_df):,} gare".replace(",", "."))

# ==================== KPI ROW ====================
st.markdown("---")
st.subheader("📈 Indicatori Chiave")

# Helper per trovare colonne dinamicamente
def get_col(df, candidates):
    """Trova la prima colonna esistente e con dati validi"""
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None

# Identifica colonne chiave
amount_col = get_col(filtered_df, ['award_amount', 'importo_aggiudicazione', 'tender_amount'])
buyer_col = get_col(filtered_df, ['buyer_name', 'ente_appaltante', 'stazione_appaltante'])
supplier_col = get_col(filtered_df, ['supplier_name', 'aggiudicatario', 'award_supplier_name'])
sconto_col = get_col(filtered_df, ['sconto', 'ribasso', 'discount'])
participants_col = get_col(filtered_df, ['offerte_ricevute', 'parties_count', 'num_partecipanti'])

col1, col2, col3, col4, col5, col6 = st.columns(6)

# KPI calcolati su dati FILTRATI
total_gare = len(filtered_df)

# Valore totale
if amount_col and total_gare > 0:
    amounts = pd.to_numeric(filtered_df[amount_col], errors='coerce')
    total_value = amounts.sum()
else:
    total_value = 0

# Sconto medio
if sconto_col and total_gare > 0:
    sconti = pd.to_numeric(filtered_df[sconto_col], errors='coerce')
    avg_sconto = sconti.mean()
else:
    avg_sconto = np.nan

# Partecipanti medi
if participants_col and total_gare > 0:
    parts = pd.to_numeric(filtered_df[participants_col], errors='coerce')
    avg_participants = parts.mean()
else:
    avg_participants = np.nan

# Unique buyers
unique_buyers = filtered_df[buyer_col].nunique() if buyer_col else 0

# Unique suppliers
unique_suppliers = filtered_df[supplier_col].nunique() if supplier_col else 0

col1.metric("🏛️ Totale Gare", f"{total_gare:,}".replace(",", "."),
            help="Numero totale di gare/lotti nel dataset filtrato")
col2.metric("💰 Valore Totale", f"€{total_value/1e9:.2f}B" if total_value > 0 else "€0",
            help="Somma degli importi di aggiudicazione di tutte le gare")
col3.metric("📉 Sconto Medio", f"{avg_sconto:.1f}%" if pd.notna(avg_sconto) and not np.isnan(avg_sconto) else "N/D",
            help="Media dei ribassi percentuali offerti rispetto alla base d'asta")
col4.metric("👥 Partecipanti Medi", f"{avg_participants:.1f}" if pd.notna(avg_participants) and not np.isnan(avg_participants) else "N/D",
            help="Numero medio di offerte ricevute per gara")
col5.metric("🏢 Stazioni Appaltanti", f"{unique_buyers:,}".replace(",", "."),
            help="Numero di enti pubblici distinti che hanno bandito gare")
col6.metric("🏭 Fornitori Unici", f"{unique_suppliers:,}".replace(",", "."),
            help="Numero di imprese diverse che hanno vinto almeno una gara")

# Row 2: More KPIs (fonti dati) - tutti basati su filtered_df
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Valori statistici
if amount_col and total_gare > 0:
    valid_amounts = pd.to_numeric(filtered_df[amount_col], errors='coerce').dropna()
    median_value = valid_amounts.median() if len(valid_amounts) > 0 else 0
    max_value = valid_amounts.max() if len(valid_amounts) > 0 else 0
else:
    median_value = 0
    max_value = 0

# Conta per fonte (SEMPRE su filtered_df!)
if 'fonte' in filtered_df.columns and filtered_df['fonte'].notna().any():
    gare_gazzetta = len(filtered_df[filtered_df['fonte'] == 'Gazzetta'])
    gare_ocds = len(filtered_df[filtered_df['fonte'] == 'OCDS'])
    gare_consip = len(filtered_df[filtered_df['fonte'] == 'CONSIP'])
else:
    gare_gazzetta = 0
    gare_ocds = total_gare
    gare_consip = 0

# Chiavi uniche
chiave_col = get_col(filtered_df, ['chiave', 'cig', 'CIG', 'ocid'])
chiavi_uniche = filtered_df[chiave_col].nunique() if chiave_col else total_gare

col1.metric("📊 Valore Mediano", f"€{median_value/1e3:.0f}K" if median_value > 0 else "N/D",
            help="Valore centrale: 50% delle gare ha importo inferiore, 50% superiore")
col2.metric("🔝 Gara Max", f"€{max_value/1e6:.1f}M" if max_value > 0 else "N/D",
            help="Gara con l'importo di aggiudicazione più alto nel dataset")
col3.metric("📰 Gazzetta", f"{gare_gazzetta:,}".replace(",", "."),
            help="Gare estratte dalla Gazzetta Ufficiale Europea (TED)")
col4.metric("📊 OCDS", f"{gare_ocds:,}".replace(",", "."),
            help="Gare dal portale ANAC in formato Open Contracting Data Standard")
col5.metric("🏛️ CONSIP", f"{gare_consip:,}".replace(",", "."),
            help="Gare da convenzioni CONSIP (Servizio Luce, SIE)")
col6.metric("🔑 Chiavi Uniche", f"{chiavi_uniche:,}".replace(",", "."),
            help="Numero di identificativi univoci (CIG o OCID) distinti")

# ==================== ALERT SCADENZE IMMINENTI (BANNER) ====================
try:
    _consip_for_alert = load_consip_data()
    _consip_map_alert = None
    if _consip_for_alert is not None and len(_consip_for_alert) > 0 and 'CIG' in _consip_for_alert.columns:
        # Calcolo leggero della mappa CONSIP per alert
        _dfc_a = _consip_for_alert.copy()
        for _col_a in ['DataAggiudicazione', 'DATA_ULTIMO_PERFEZIONAMENTO', 'DATA_COMUNICAZIONE_ESITO', 'DataPubblicazione']:
            if _col_a in _dfc_a.columns:
                _dfc_a[_col_a] = pd.to_datetime(_dfc_a[_col_a], format='%d/%m/%Y', errors='coerce')
        _dfc_a['durata_giorni_consip'] = pd.to_numeric(_dfc_a.get('DURATA_PREVISTA', pd.Series(dtype='float64')), errors='coerce')
        _start_a = _dfc_a.get('DataAggiudicazione', pd.Series([pd.NaT] * len(_dfc_a)))
        for _fb in ['DATA_ULTIMO_PERFEZIONAMENTO', 'DATA_COMUNICAZIONE_ESITO', 'DataPubblicazione']:
            if _fb in _dfc_a.columns:
                _start_a = _start_a.fillna(_dfc_a[_fb])
        _dfc_a['scadenza_consip'] = _start_a + pd.to_timedelta(_dfc_a['durata_giorni_consip'], unit='D')
        _dfc_a['cig'] = _dfc_a['CIG'].astype(str).str.strip()
        _dfc_a = _dfc_a[_dfc_a['cig'].ne('') & _dfc_a['scadenza_consip'].notna()]
        _consip_map_alert = _dfc_a.groupby('cig', as_index=False).agg({'scadenza_consip': 'max', 'durata_giorni_consip': 'max'})

    # Calcola scadenze su tutto il dataset (senza filtri) per alert globale
    _alert_cols = [c for c in ['cig', 'comune', 'buyer_locality', 'regione', 'award_amount', 'importo_aggiudicazione',
                               'data_aggiudicazione', 'data_scadenza', 'durata_appalto', '_categoria', 'categoria',
                               'oggetto'] if c in raw_df.columns]
    _alert_df = raw_df[_alert_cols].copy()
    if 'cig' in _alert_df.columns:
        _alert_df['cig'] = _alert_df['cig'].fillna('').astype(str).str.strip().replace({'nan': '', 'None': ''})
    else:
        _alert_df['cig'] = ''
    if 'data_aggiudicazione' in _alert_df.columns:
        _alert_df['award_date'] = pd.to_datetime(_alert_df['data_aggiudicazione'], errors='coerce')
    else:
        _alert_df['award_date'] = pd.NaT
    if 'importo_aggiudicazione' in _alert_df.columns:
        _alert_df['award_amount'] = pd.to_numeric(_alert_df.get('award_amount', _alert_df.get('importo_aggiudicazione')), errors='coerce')
    # Scadenza: solo fonti affidabili (no stime) per alert
    _alert_df['scadenza'] = pd.NaT
    if 'data_scadenza' in _alert_df.columns:
        _alert_df['scadenza'] = pd.to_datetime(_alert_df['data_scadenza'], errors='coerce')
    if _consip_map_alert is not None and len(_consip_map_alert) > 0:
        _alert_df = _alert_df.merge(_consip_map_alert[['cig', 'scadenza_consip']], on='cig', how='left')
        _alert_df['scadenza'] = _alert_df['scadenza'].fillna(_alert_df['scadenza_consip'])
    if 'durata_appalto' in _alert_df.columns:
        _dur = pd.to_numeric(_alert_df['durata_appalto'], errors='coerce')
        _scad_dur = _alert_df['award_date'] + pd.to_timedelta(_dur, unit='D')
        _alert_df['scadenza'] = _alert_df['scadenza'].fillna(_scad_dur)
    # Filtra solo scadenze valide e future
    _oggi = pd.Timestamp.now().normalize()
    _alert_df['giorni'] = (_alert_df['scadenza'] - _oggi).dt.days
    _alert_valid = _alert_df[_alert_df['scadenza'].notna() & (_alert_df['giorni'] >= -30)]  # include scaduti da max 30gg

    _n30 = (_alert_valid['giorni'].between(-30, 30)).sum()
    _n90 = (_alert_valid['giorni'].between(-30, 90)).sum()
    _n365 = (_alert_valid['giorni'].between(-30, 365)).sum()
    _v30 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 30), 'award_amount'].sum()
    _v365 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 365), 'award_amount'].sum()

    _comune_col_alert = next((c for c in ['comune', 'buyer_locality'] if c in _alert_valid.columns), None)
    _cat_col_alert = next((c for c in ['_categoria', 'categoria'] if c in _alert_valid.columns), None)

    if _n30 > 0:
        with st.expander(f"🔴 {_n30} contratti in scadenza/scaduti entro 30 giorni (€{_v30/1e6:.1f}M)", expanded=False):
            _cols_30 = []
            if _comune_col_alert:
                _top_comuni_30 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 30)].groupby(_comune_col_alert).size().nlargest(5)
                _cols_30.append("**Top comuni:** " + ", ".join(f"{c} ({n})" for c, n in _top_comuni_30.items()))
            if _cat_col_alert:
                _top_cat_30 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 30)].groupby(_cat_col_alert).size().nlargest(3)
                _cols_30.append("**Top categorie:** " + ", ".join(f"{c} ({n})" for c, n in _top_cat_30.items()))
            st.markdown(" | ".join(_cols_30) if _cols_30 else "Dettagli nel tab Scadenze")
    if _n90 > _n30:
        with st.expander(f"🟠 {_n90} contratti in scadenza entro 90 giorni", expanded=False):
            _cols_90 = []
            if _comune_col_alert:
                _top_comuni_90 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 90)].groupby(_comune_col_alert).size().nlargest(5)
                _cols_90.append("**Top comuni:** " + ", ".join(f"{c} ({n})" for c, n in _top_comuni_90.items()))
            if _cat_col_alert:
                _top_cat_90 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 90)].groupby(_cat_col_alert).size().nlargest(3)
                _cols_90.append("**Top categorie:** " + ", ".join(f"{c} ({n})" for c, n in _top_cat_90.items()))
            st.markdown(" | ".join(_cols_90) if _cols_90 else "Dettagli nel tab Scadenze")
    if _n365 > _n90:
        with st.expander(f"🟡 {_n365} contratti in scadenza entro 12 mesi (€{_v365/1e6:.1f}M)", expanded=False):
            _cols_365 = []
            if _comune_col_alert:
                _top_comuni_365 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 365)].groupby(_comune_col_alert).size().nlargest(5)
                _cols_365.append("**Top comuni:** " + ", ".join(f"{c} ({n})" for c, n in _top_comuni_365.items()))
            if _cat_col_alert:
                _top_cat_365 = _alert_valid.loc[_alert_valid['giorni'].between(-30, 365)].groupby(_cat_col_alert).size().nlargest(3)
                _cols_365.append("**Top categorie:** " + ", ".join(f"{c} ({n})" for c, n in _top_cat_365.items()))
            st.markdown(" | ".join(_cols_365) if _cols_365 else "Dettagli nel tab Scadenze")
except Exception:
    pass  # Alert non bloccante: se fallisce, la dashboard continua

# ==================== TAB NAVIGATION (CLUSTER UI) ====================
st.markdown("---")

# Cluster selection con radio buttons
st.markdown(
    """
<style>
.cluster-container {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 15px;
}
.cluster-btn {
    padding: 8px 16px;
    border-radius: 20px;
    border: 2px solid #e0e0e0;
    background: white;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s;
}
.cluster-btn:hover {
    border-color: var(--brand-green);
    background: var(--brand-surface);
}
.cluster-btn.active {
    background: var(--brand-green);
    border-color: var(--brand-green);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

# Cluster pills
cluster_names = ["📊 Panoramica", "🏆 Operatori", "🗺️ Territoriale", "📈 Analisi Avanzata", "🤖 AI & Preferiti"]
selected_cluster = st.radio(
    "🎯 Area di Analisi",
    cluster_names,
    horizontal=True,
    label_visibility="collapsed"
)

# Cluster descriptions
cluster_info = {
    "📊 Panoramica": "Geografia • Categorie • Trend • Statistiche",
    "🏆 Operatori": "Aggiudicatari • Ricerca • Confronto • Network",
    "🗺️ Territoriale": "Città • Mappa CONSIP • Convenzioni",
    "📈 Analisi Avanzata": "Mercato • Scadenze • Stagionalità",
    "🤖 AI & Preferiti": "Grafici AI • Chat AI • Predizioni ML • Mappa Avanzata • Preferiti"
}
st.caption(f"*{cluster_info.get(selected_cluster, '')}*")

# Tab definitions per cluster
if selected_cluster == "📊 Panoramica":
    tab1, tab2, tab3, tab6, tab20 = st.tabs(["🗺️ Geografia", "📦 Categorie", "📈 Trend", "📊 Statistiche", "🔎 Ricerca"])
    tab4 = tab5 = tab7 = tab8 = tab9 = tab10 = tab11 = tab12 = tab13 = tab14 = tab15 = tab16 = tab17 = tab18 = tab19 = None
elif selected_cluster == "🏆 Operatori":
    tab4, tab9, tab12, tab14 = st.tabs(["🏆 Aggiudicatari", "🔎 Aggiudicatario", "⚔️ Confronto", "🌐 Network"])
    tab1 = tab2 = tab3 = tab5 = tab6 = tab7 = tab8 = tab10 = tab11 = tab13 = tab15 = tab16 = tab17 = tab18 = tab19 = None
elif selected_cluster == "🗺️ Territoriale":
    tab7, tab8, tab5 = st.tabs(["🔍 Città", "🗺️ Mappa CONSIP", "🏛️ CONSIP"])
    tab1 = tab2 = tab3 = tab4 = tab6 = tab9 = tab10 = tab11 = tab12 = tab13 = tab14 = tab15 = tab16 = tab17 = tab18 = tab19 = None
elif selected_cluster == "📈 Analisi Avanzata":
    tab10, tab11, tab13 = st.tabs(["📉 Analisi Mercato", "📅 Scadenze", "📆 Stagionalità"])
    tab1 = tab2 = tab3 = tab4 = tab5 = tab6 = tab7 = tab8 = tab9 = tab12 = tab14 = tab15 = tab16 = tab17 = tab18 = tab19 = None
else:  # AI & Preferiti
    tab15, tab17, tab18, tab19, tab16 = st.tabs(["🤖 AI Charts", "💬 Chat AI", "🔮 Predizioni ML", "🗺️ Mappa Pro", "⭐ Preferiti"])
    tab1 = tab2 = tab3 = tab4 = tab5 = tab6 = tab7 = tab8 = tab9 = tab10 = tab11 = tab12 = tab13 = tab14 = None

# ==================== TAB 1: GEOGRAFIA ====================
if tab1:
  with tab1:
    # Identifica colonne dinamicamente
    regione_col_geo = next((c for c in filtered_df.columns if c.lower() == 'regione'), None)
    comune_col_geo = next((c for c in filtered_df.columns if c.lower() in ['comune', 'citta', 'buyer_locality']), None)
    amount_col_geo = next((c for c in filtered_df.columns if c.lower() in ['importo_aggiudicazione', 'award_amount']), None)
    id_col_geo = next((c for c in filtered_df.columns if c.lower() in ['chiave', 'cig', 'ocid']), None)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗺️ Mappa Città per Valore")
        # Calcola dati città da filtered_df
        if comune_col_geo and amount_col_geo and id_col_geo:
            # Coordinate città italiane principali
            city_coords = {
                'Roma': (41.9028, 12.4964), 'Milano': (45.4642, 9.1900), 'Napoli': (40.8518, 14.2681),
                'Torino': (45.0703, 7.6869), 'Palermo': (38.1157, 13.3615), 'Genova': (44.4056, 8.9463),
                'Bologna': (44.4949, 11.3426), 'Firenze': (43.7696, 11.2558), 'Bari': (41.1171, 16.8719),
                'Catania': (37.5079, 15.0830), 'Venezia': (45.4408, 12.3155), 'Verona': (45.4384, 10.9916),
                'Messina': (38.1938, 15.5540), 'Padova': (45.4064, 11.8768), 'Trieste': (45.6495, 13.7768),
                'Brescia': (45.5416, 10.2118), 'Parma': (44.8015, 10.3279), 'Taranto': (40.4644, 17.2470),
                'Prato': (43.8777, 11.1020), 'Modena': (44.6471, 10.9252), 'Reggio Calabria': (38.1113, 15.6473),
                'Reggio Emilia': (44.6989, 10.6297), 'Perugia': (43.1107, 12.3908), 'Livorno': (43.5485, 10.3106),
                'Ravenna': (44.4184, 12.2035), 'Cagliari': (39.2238, 9.1217), 'Foggia': (41.4621, 15.5444),
                'Rimini': (44.0678, 12.5695), 'Salerno': (40.6824, 14.7681), 'Ferrara': (44.8381, 11.6198)
            }
            cities_agg = filtered_df.groupby(comune_col_geo, observed=True).agg({
                amount_col_geo: 'sum',
                id_col_geo: 'count',
                'sconto': 'mean'
            }).reset_index()
            cities_agg.columns = ['citta', 'valore', 'num_gare', 'sconto_medio']
            cities_agg = cities_agg.dropna(subset=['citta'])
            cities_agg = cities_agg[cities_agg['citta'] != '']
            # Aggiungi coordinate
            cities_agg['lat'] = cities_agg['citta'].map(lambda x: city_coords.get(x, (None, None))[0])
            cities_agg['lng'] = cities_agg['citta'].map(lambda x: city_coords.get(x, (None, None))[1])
            cities_df = cities_agg.dropna(subset=['lat', 'lng']).sort_values('valore', ascending=False).head(30)

            if len(cities_df) > 0:
                fig = px.scatter_map(
                    cities_df,
                    lat='lat',
                    lon='lng',
                    size='valore',
                    color='sconto_medio',
                    hover_name='citta',
                    hover_data={'num_gare': True, 'valore': ':.2s', 'sconto_medio': ':.1f'},
                    color_continuous_scale='RdYlGn',
                    size_max=50,
                    zoom=5,
                    center={'lat': 42.0, 'lon': 12.5},
                )
                fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nessuna città con coordinate disponibili per i filtri selezionati")
        else:
            st.info("Colonne geografiche non disponibili")

    with col2:
        st.subheader("🇮🇹 Classifica Regioni")
        # Calcola dati regioni da filtered_df
        if regione_col_geo and amount_col_geo and id_col_geo:
            geo_df = filtered_df.groupby(regione_col_geo, observed=True).agg({
                amount_col_geo: 'sum',
                id_col_geo: 'count',
                'sconto': 'mean'
            }).reset_index()
            geo_df.columns = ['Regione', 'valore', 'num_gare', 'sconto_medio']
            geo_df = geo_df.dropna(subset=['Regione'])
            geo_df = geo_df[geo_df['Regione'] != '']
            geo_df = geo_df.sort_values('valore', ascending=False)

            if len(geo_df) > 0:
                fig_regioni = px.bar(
                    geo_df,
                    x='valore',
                    y='Regione',
                    orientation='h',
                    color='sconto_medio',
                    color_continuous_scale='RdYlGn',
                    text=geo_df['valore'].apply(lambda x: f'€{x/1e9:.1f}B' if x >= 1e9 else f'€{x/1e6:.0f}M')
                )
                fig_regioni.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                fig_regioni.update_traces(textposition='outside')
                render_chart_with_save(fig_regioni, "Classifica Regioni per Valore", "Bar chart regioni italiane ordinate per valore aggiudicazioni", "geo_regioni")
            else:
                st.info("Nessuna regione disponibile per i filtri selezionati")
        else:
            st.info("Colonna regione non disponibile")

    # === MAPPA COROPLETICA ITALIA ===
    st.markdown("---")
    st.subheader("🗺️ Mappa Coropletica Italia")
    st.markdown("*Intensità colorata per valore gare o numero gare per regione*")

    if regione_col_geo and amount_col_geo and id_col_geo:
        # Prepara dati per choropleth
        choropleth_df = filtered_df.groupby(regione_col_geo, observed=True).agg({
            amount_col_geo: 'sum',
            id_col_geo: 'count',
            'sconto': 'mean'
        }).reset_index()
        choropleth_df.columns = ['regione', 'valore', 'num_gare', 'sconto_medio']
        choropleth_df = choropleth_df.dropna(subset=['regione'])
        choropleth_df = choropleth_df[choropleth_df['regione'] != '']

        if len(choropleth_df) > 0:
            # Opzioni visualizzazione
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                choropleth_metric = st.radio(
                    "Colora per:",
                    ["💰 Valore Totale", "📊 Numero Gare", "📉 Sconto Medio"],
                    horizontal=True,
                    key="choropleth_metric"
                )

            # Determina metrica e scala colori
            if "Valore" in choropleth_metric:
                color_col = 'valore'
                color_label = 'Valore (€)'
                choropleth_df['display_value'] = choropleth_df['valore'].apply(lambda x: f"€{x/1e9:.2f}B" if x >= 1e9 else f"€{x/1e6:.0f}M")
                color_scale = 'Blues'
            elif "Numero" in choropleth_metric:
                color_col = 'num_gare'
                color_label = 'N. Gare'
                choropleth_df['display_value'] = choropleth_df['num_gare'].apply(lambda x: f"{x:,}".replace(",", "."))
                color_scale = 'Greens'
            else:
                color_col = 'sconto_medio'
                color_label = 'Sconto %'
                choropleth_df['display_value'] = choropleth_df['sconto_medio'].apply(lambda x: f"{x:.1f}%")
                color_scale = 'RdYlGn'

            # Coordinate centri regioni italiane
            regioni_coords = {
                'Piemonte': (45.0522, 7.5153), 'Valle d\'Aosta': (45.7370, 7.3204), "Valle d'Aosta": (45.7370, 7.3204),
                'Lombardia': (45.4791, 9.8452), 'Trentino-Alto Adige': (46.4993, 11.3567),
                'Veneto': (45.4347, 11.8754), 'Friuli-Venezia Giulia': (45.6495, 13.7768),
                'Liguria': (44.4471, 8.7464), 'Emilia-Romagna': (44.5075, 11.3514),
                'Toscana': (43.4587, 11.1196), 'Umbria': (42.9317, 12.5711), 'Marche': (43.3017, 13.4533),
                'Lazio': (41.9813, 12.6893), 'Abruzzo': (42.2012, 13.5201), 'Molise': (41.6748, 14.4185),
                'Campania': (40.8405, 14.3340), 'Puglia': (41.0125, 16.5042), 'Basilicata': (40.4927, 15.9714),
                'Calabria': (38.9052, 16.5942), 'Sicilia': (37.5994, 14.0154), 'Sardegna': (40.1209, 9.0129)
            }

            # Aggiungi coordinate
            choropleth_df['lat'] = choropleth_df['regione'].map(lambda x: regioni_coords.get(x, (42.0, 12.5))[0])
            choropleth_df['lon'] = choropleth_df['regione'].map(lambda x: regioni_coords.get(x, (42.0, 12.5))[1])

            # Crea mappa con bolle proporzionali
            fig_choro = px.scatter_map(
                choropleth_df,
                lat='lat',
                lon='lon',
                size=color_col,
                color=color_col,
                hover_name='regione',
                hover_data={
                    'valore': ':,.0f',
                    'num_gare': ':,',
                    'sconto_medio': ':.1f',
                    'lat': False,
                    'lon': False
                },
                color_continuous_scale=color_scale,
                size_max=60,
                zoom=4.8,
                center={'lat': 42.0, 'lon': 12.0},
                opacity=0.7
            )

            fig_choro.update_layout(
                height=550,
                margin={"r":0, "t":30, "l":0, "b":0},
                coloraxis_colorbar=dict(title=color_label),
                title=f"Distribuzione per {color_label}"
            )

            st.plotly_chart(fig_choro, use_container_width=True)

            # Legenda/statistiche
            col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
            with col_leg1:
                top_region = choropleth_df.loc[choropleth_df[color_col].idxmax(), 'regione']
                st.metric("🥇 Top Regione", top_region)
            with col_leg2:
                st.metric("📈 Max", choropleth_df['display_value'].iloc[choropleth_df[color_col].argmax()])
            with col_leg3:
                st.metric("📉 Min", choropleth_df['display_value'].iloc[choropleth_df[color_col].argmin()])
            with col_leg4:
                coverage = len(choropleth_df)
                st.metric("🗺️ Regioni", f"{coverage}/20")

        else:
            st.info("Dati regionali non sufficienti per la mappa")
    else:
        st.info("Colonne geografiche non disponibili")

    # Dettaglio regioni con selezione interattiva
    st.markdown("---")
    st.subheader("📋 Dettaglio per Regione")
    if regione_col_geo and amount_col_geo and id_col_geo:
        geo_detail = filtered_df.groupby(regione_col_geo, observed=True).agg({
            amount_col_geo: 'sum',
            id_col_geo: 'count',
            'sconto': 'mean'
        }).reset_index()
        geo_detail.columns = ['Regione', 'valore', 'N. Gare', 'Sconto Medio %']
        geo_detail['Valore (€B)'] = geo_detail['valore'] / 1e9
        geo_detail = geo_detail.dropna(subset=['Regione'])
        geo_detail = geo_detail[geo_detail['Regione'] != '']
        geo_detail = geo_detail.sort_values('valore', ascending=False)

        # Mostra tabella
        show_dataframe(geo_detail[['Regione', 'N. Gare', 'Valore (€B)', 'Sconto Medio %']], use_container_width=True)

        # Selezione regione per vedere dettaglio gare
        st.markdown("---")
        st.subheader("🔍 Esplora Gare per Regione")

        regioni_disponibili = ["Tutte le regioni"] + geo_detail['Regione'].tolist()
        regione_esplora = st.selectbox("Seleziona regione da esplorare", regioni_disponibili, key="esplora_regione")

        # Filtra dati per regione selezionata
        if regione_esplora == "Tutte le regioni":
            gare_regione = filtered_df.copy()
            titolo_export = "tutte_regioni"
        else:
            gare_regione = filtered_df[filtered_df[regione_col_geo] == regione_esplora]
            titolo_export = regione_esplora.lower().replace(" ", "_").replace("'", "")

        # Colonne da mostrare nel dettaglio
        cols_display = ['data_aggiudicazione', 'oggetto', 'ente_appaltante', 'aggiudicatario',
                        'importo_aggiudicazione', 'sconto', 'categoria', 'procedura', 'comune']
        cols_available = [c for c in cols_display if c in gare_regione.columns]

        # Ordina per data più recente
        if 'data_aggiudicazione' in gare_regione.columns:
            gare_regione = gare_regione.sort_values('data_aggiudicazione', ascending=False)

        # Mostra statistiche
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Gare totali", f"{len(gare_regione):,}".replace(",", "."))
        col_stat2.metric("Valore totale", f"€{gare_regione[amount_col_geo].sum()/1e6:.1f}M")
        col_stat3.metric("Sconto medio", f"{gare_regione['sconto'].mean():.1f}%" if gare_regione['sconto'].notna().any() else "N/D")
        col_stat4.metric("Enti coinvolti", f"{gare_regione['ente_appaltante'].nunique():,}".replace(",", ".") if 'ente_appaltante' in gare_regione.columns else "N/D")

        # Mostra ultime gare
        st.markdown(f"**Ultime {min(100, len(gare_regione))} gare:**")
        gare_display = gare_regione[cols_available].head(100).copy()

        # Formatta importo per visualizzazione
        if 'importo_aggiudicazione' in gare_display.columns:
            gare_display['importo_aggiudicazione'] = gare_display['importo_aggiudicazione'].apply(
                lambda x: f"€{x/1e6:.2f}M" if pd.notna(x) and x >= 1e6 else (f"€{x/1e3:.0f}K" if pd.notna(x) else "N/D")
            )
        if 'sconto' in gare_display.columns:
            gare_display['sconto'] = gare_display['sconto'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/D")
        if 'data_aggiudicazione' in gare_display.columns:
            gare_display['data_aggiudicazione'] = pd.to_datetime(gare_display['data_aggiudicazione']).dt.strftime('%Y-%m-%d')

        # Rinomina colonne per display
        col_rename = {
            'data_aggiudicazione': 'Data',
            'oggetto': 'Oggetto',
            'ente_appaltante': 'Ente',
            'aggiudicatario': 'Aggiudicatario',
            'importo_aggiudicazione': 'Importo',
            'sconto': 'Sconto',
            'categoria': 'Categoria',
            'procedura': 'Procedura',
            'comune': 'Comune'
        }
        gare_display = gare_display.rename(columns={k: v for k, v in col_rename.items() if k in gare_display.columns})

        show_dataframe(gare_display, use_container_width=True, height=400)

        # Download CSV
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)

        # Prepara CSV per download (dati completi, non formattati)
        csv_export = gare_regione[cols_available].copy()
        csv_data = csv_export.to_csv(index=False).encode('utf-8')

        with col_dl1:
            st.download_button(
                label=f"📥 Scarica CSV ({len(gare_regione):,} gare)".replace(",", "."),
                data=csv_data,
                file_name=f"gare_{titolo_export}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Scarica tutte le gare della regione selezionata in formato CSV"
            )

        with col_dl2:
            # Export Excel se openpyxl disponibile
            try:
                from io import BytesIO
                excel_buffer = BytesIO()
                _sanitize_for_excel(csv_export).to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label=f"📥 Scarica Excel ({len(gare_regione):,} gare)".replace(",", "."),
                    data=excel_data,
                    file_name=f"gare_{titolo_export}_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Scarica tutte le gare della regione selezionata in formato Excel"
                )
            except ImportError:
                pass
    else:
        st.info("Dati geografici non disponibili")

# ==================== TAB 2: CATEGORIE ====================
if tab2:
  with tab2:
    # Identifica colonne dinamicamente
    cat_col_tab2 = next((c for c in filtered_df.columns if c.lower() in ['categoria', '_categoria', 'category']), None)
    amount_col_tab2 = next((c for c in filtered_df.columns if c.lower() in ['importo_aggiudicazione', 'award_amount']), None)
    id_col_tab2 = next((c for c in filtered_df.columns if c.lower() in ['chiave', 'cig', 'ocid']), None)
    offerte_col = next((c for c in filtered_df.columns if c.lower() in ['offerte_ricevute', 'num_offerte']), None)

    # Calcola dati categorie da filtered_df
    if cat_col_tab2 and amount_col_tab2 and id_col_tab2:
        agg_dict = {amount_col_tab2: 'sum', id_col_tab2: 'count', 'sconto': 'mean'}
        if offerte_col:
            agg_dict[offerte_col] = 'mean'
        cat_df = filtered_df.groupby(cat_col_tab2, observed=True).agg(agg_dict).reset_index()
        col_names = ['Categoria_Main', 'valore', 'num_gare', 'sconto_medio']
        if offerte_col:
            col_names.append('partecipanti_medi')
        cat_df.columns = col_names[:len(cat_df.columns)]
        cat_df = cat_df.dropna(subset=['Categoria_Main'])
        cat_df = cat_df[cat_df['Categoria_Main'] != '']
        cat_df = cat_df.sort_values('valore', ascending=False)
        if 'partecipanti_medi' not in cat_df.columns:
            cat_df['partecipanti_medi'] = 1
        else:
            # Assicura valori positivi per size in scatter plot
            cat_df['partecipanti_medi'] = cat_df['partecipanti_medi'].fillna(1).clip(lower=0.1)
    else:
        # Fallback ai dati pre-calcolati se colonne non trovate
        cat_df = pd.DataFrame(data.get('categories', []))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Distribuzione per Categoria")
        if len(cat_df) > 0:
            fig_tree = px.treemap(
                cat_df,
                path=['Categoria_Main'],
                values='valore',
                color='sconto_medio',
                color_continuous_scale='RdYlGn',
                hover_data={'num_gare': True, 'sconto_medio': ':.1f'}
            )
            fig_tree.update_layout(height=450)
            render_chart_with_save(fig_tree, "Treemap Categorie", "Distribuzione categorie merceologiche per valore", "treemap_categorie")
        else:
            st.info("Nessuna categoria disponibile per i filtri selezionati")

    with col2:
        st.subheader("📊 Categorie per Numero Gare vs Valore")
        if len(cat_df) > 0:
            # Prepara dati per scatter: rimuovi NaN e valori non validi
            scatter_df = cat_df.copy()
            scatter_df['sconto_medio'] = scatter_df['sconto_medio'].fillna(0)
            scatter_df['partecipanti_medi'] = scatter_df['partecipanti_medi'].fillna(1).clip(lower=0.1)
            scatter_df = scatter_df[scatter_df['num_gare'] > 0]
            fig = px.scatter(
                scatter_df,
                x='num_gare',
                y='valore',
                size='partecipanti_medi',
                color='sconto_medio',
                hover_name='Categoria_Main',
                color_continuous_scale='RdYlGn',
                labels={'num_gare': 'Numero Gare', 'valore': 'Valore (€)', 'sconto_medio': 'Sconto %'},
                hover_data={'partecipanti_medi': ':.1f', 'sconto_medio': ':.1f'}
            )
            fig.update_layout(height=450)
            render_chart_with_save(fig, "Scatter Categorie", "Categorie per numero gare vs valore", "scatter_categorie")
        else:
            st.info("Nessuna categoria disponibile")

    # Radar chart categorie
    st.subheader("🎯 Confronto Categorie (Radar)")
    if len(cat_df) > 0:
        cat_normalized = cat_df.copy()
        for col in ['num_gare', 'valore', 'sconto_medio', 'partecipanti_medi']:
            if col in cat_normalized.columns:
                col_range = cat_normalized[col].max() - cat_normalized[col].min()
                if col_range > 0:
                    cat_normalized[col] = (cat_normalized[col] - cat_normalized[col].min()) / col_range
                else:
                    cat_normalized[col] = 0.5

        fig_radar = go.Figure()
        for _, row in cat_normalized.head(5).iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['num_gare'], row['valore'], row['sconto_medio'], row.get('partecipanti_medi', 0)],
                theta=['N. Gare', 'Valore', 'Sconto', 'Partecipanti'],
                fill='toself',
                name=str(row['Categoria_Main'])[:20]
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=400)
        render_chart_with_save(fig_radar, "Radar Categorie", "Confronto categorie su 4 dimensioni", "radar_categorie")
    else:
        st.info("Dati insufficienti per il radar chart")

# ==================== TAB 3: TREND ====================
if tab3:
  with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Trend Sconti e Partecipanti (Doppio Asse)")
        # Calcola trend da filtered_df
        offerte_col_trend = next((c for c in filtered_df.columns if c.lower() in ['offerte_ricevute', 'num_offerte']), None)

        if 'anno' in filtered_df.columns and 'sconto' in filtered_df.columns:
            agg_dict_trend = {'sconto': ['mean', 'median']}
            if offerte_col_trend:
                agg_dict_trend[offerte_col_trend] = 'mean'

            trends_df = filtered_df[filtered_df['anno'].between(2015, 2025)].groupby('anno', observed=True).agg(agg_dict_trend).reset_index()
            trends_df.columns = ['anno', 'media', 'mediana'] + (['partecipanti_medi'] if offerte_col_trend else [])

            if 'partecipanti_medi' not in trends_df.columns:
                trends_df['partecipanti_medi'] = 0

            if len(trends_df) > 0:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(x=trends_df['anno'], y=trends_df['media'], name='Sconto Medio %',
                               line=dict(color=CGL_GREEN, width=3), fill='tozeroy', fillcolor='rgba(0,208,132,0.15)'),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=trends_df['anno'], y=trends_df['partecipanti_medi'], name='Partecipanti Medi',
                               line=dict(color=CGL_BLUE, width=3, dash='dash')),
                    secondary_y=True
                )
                fig.add_trace(
                    go.Scatter(x=trends_df['anno'], y=trends_df['mediana'], name='Mediana Sconto',
                               line=dict(color=CGL_CYAN, width=2, dash='dot')),
                    secondary_y=False
                )
                fig.update_yaxes(title_text="Sconto %", secondary_y=False)
                fig.update_yaxes(title_text="N. Partecipanti", secondary_y=True)
                fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                render_chart_with_save(fig, "Trend Sconti e Partecipanti", "Andamento storico sconto medio e partecipanti", "trend_sconti")
            else:
                st.info("Dati insufficienti per il trend")
        else:
            st.info("Colonne anno/sconto non disponibili")

    with col2:
        st.subheader("📊 Volume Gare per Anno (OCDS + Gazzetta)")
        # Calcola direttamente da filtered_df per includere sia OCDS che Gazzetta
        if 'anno' in filtered_df.columns:
            # Identifica colonne dinamicamente
            id_col_vol = get_col(filtered_df, ['chiave', 'CIG', 'ocid', 'id'])
            amount_col_vol = get_col(filtered_df, ['importo_aggiudicazione', 'award_amount', 'tender_amount'])

            # Conta record con e senza anno
            with_anno = filtered_df['anno'].notna().sum()
            without_anno = filtered_df['anno'].isna().sum()

            if id_col_vol:
                # Crea colonna anno_display che include "N/D" per record senza data
                df_vol = filtered_df.copy()
                df_vol['anno_display'] = df_vol['anno'].apply(
                    lambda x: str(int(x)) if pd.notna(x) and 2015 <= x <= 2025 else ('< 2015' if pd.notna(x) and x < 2015 else 'N/D')
                )

                agg_dict_vol = {id_col_vol: 'count'}
                if amount_col_vol:
                    agg_dict_vol[amount_col_vol] = 'sum'
                if 'sconto' in filtered_df.columns:
                    agg_dict_vol['sconto'] = 'mean'

                volume_df = df_vol.groupby('anno_display', observed=True).agg(agg_dict_vol).reset_index()

                # Rinomina colonne
                rename_dict = {'anno_display': 'anno', id_col_vol: 'count'}
                if amount_col_vol:
                    rename_dict[amount_col_vol] = 'valore'
                if 'sconto' in agg_dict_vol:
                    rename_dict['sconto'] = 'sconto_medio'
                volume_df = volume_df.rename(columns=rename_dict)

                # Ordina: anni numerici prima, poi "< 2015", poi "N/D"
                def sort_key(x):
                    if x == 'N/D':
                        return 9999
                    elif x == '< 2015':
                        return 2014
                    else:
                        try:
                            return int(x)
                        except:
                            return 9998
                volume_df['sort_order'] = volume_df['anno'].apply(sort_key)
                volume_df = volume_df.sort_values('sort_order').drop(columns=['sort_order'])

                # Colore diverso per N/D
                volume_df['tipo'] = volume_df['anno'].apply(lambda x: 'Data N/D' if x == 'N/D' else 'Con Data')

                fig = px.bar(
                    volume_df,
                    x='anno',
                    y='count',
                    color='tipo',
                    color_discrete_map={'Con Data': CGL_GREEN, 'Data N/D': CGL_ORANGE},
                    labels={'count': 'Numero Gare', 'anno': 'Anno', 'tipo': 'Stato Data'},
                    text='count'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=400, xaxis={'categoryorder': 'array', 'categoryarray': volume_df['anno'].tolist()})
                st.plotly_chart(fig, use_container_width=True)

                # Mostra breakdown per fonte
                if 'fonte' in filtered_df.columns:
                    # Per record con data
                    with_data = filtered_df[filtered_df['anno'].notna() & filtered_df['anno'].between(2015, 2025)]
                    fonte_with = with_data.groupby('fonte', observed=True)[id_col_vol].count().to_dict() if len(with_data) > 0 else {}
                    # Per record senza data
                    no_data = filtered_df[filtered_df['anno'].isna()]
                    fonte_no = no_data.groupby('fonte', observed=True)[id_col_vol].count().to_dict() if len(no_data) > 0 else {}

                    st.caption(f"📊 Con data: {fonte_with}")
                    if fonte_no:
                        st.caption(f"⚠️ Senza data: {fonte_no}")
            else:
                st.warning("Colonna ID non trovata per il conteggio")
        else:
            # Fallback ai dati pre-calcolati
            fig = px.bar(
                trends_df,
                x='anno',
                y='count',
                color='media',
                color_continuous_scale='Blues',
                labels={'count': 'Numero Gare', 'anno': 'Anno', 'media': 'Sconto %'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Trend per categoria - calcola da filtered_df
    st.subheader("📊 Trend Sconti per Categoria")
    cat_col_trend = next((c for c in filtered_df.columns if c.lower() in ['categoria', '_categoria', 'category']), None)

    if cat_col_trend and 'anno' in filtered_df.columns and 'sconto' in filtered_df.columns:
        trends_cat_df = filtered_df[filtered_df['anno'].between(2015, 2025)].groupby(
            ['anno', cat_col_trend], observed=True
        ).agg({'sconto': 'mean'}).reset_index()
        trends_cat_df.columns = ['anno', 'categoria', 'media']
        trends_cat_df = trends_cat_df.dropna()

        # Prendi solo top 10 categorie per valore per leggibilità
        top_cats = filtered_df.groupby(cat_col_trend, observed=True).size().nlargest(10).index.tolist()
        trends_cat_df = trends_cat_df[trends_cat_df['categoria'].isin(top_cats)]

        if len(trends_cat_df) > 0:
            fig_trend_cat = px.line(
                trends_cat_df,
                x='anno',
                y='media',
                color='categoria',
                markers=True,
                labels={'anno': 'Anno', 'media': 'Sconto Medio %', 'categoria': 'Categoria'}
            )
            fig_trend_cat.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=-0.4))
            render_chart_with_save(fig_trend_cat, "Trend per Categoria", "Evoluzione sconti per categoria merceologica", "trend_categorie")

            # Heatmap Anno x Categoria
            st.subheader("🔥 Heatmap Sconto: Anno × Categoria")
            pivot = trends_cat_df.pivot(index='categoria', columns='anno', values='media')
            fig_heatmap = px.imshow(
                pivot,
                color_continuous_scale='RdYlGn',
                labels={'color': 'Sconto %'},
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            render_chart_with_save(fig_heatmap, "Heatmap Anno x Categoria", "Mappa termica sconti per anno e categoria", "heatmap_categorie")
        else:
            st.info("Dati insufficienti per trend per categoria")
    else:
        st.info("Colonne necessarie non disponibili")

# ==================== TAB 20: RICERCA GARE ====================
if 'tab20' in locals() and tab20:
  with tab20:
    st.subheader("🔎 Ricerca Gare per parola chiave")
    st.caption("La ricerca applica anche i filtri della sidebar (fonte/anno/regione/…)")

    # Colonne utili per filtri numerici
    sconto_col_search = 'sconto' if 'sconto' in filtered_df.columns else None
    partecipanti_col_search = next((c for c in ['offerte_ricevute', 'parties_count', 'num_partecipanti', 'num_offerte'] if c in filtered_df.columns), None)

    # Sorgenti e colonne ricercabili proposte (solo testuali)
    default_search_cols = [
        'oggetto', 'tender_title', 'tender_description', 'categoria', 'categoria_originale', 'quick_category',
        'aggiudicatario', 'supplier_name', 'ente_appaltante', 'buyer_name', 'comune', 'regione',
        'cpv_description', 'cpv_code'
    ]
    available_cols = [c for c in default_search_cols if c in filtered_df.columns]
    cols_sel = st.multiselect("Cerca nei campi", options=available_cols, default=available_cols)

    col_q1, col_q2 = st.columns([2,1])
    with col_q1:
        query = st.text_input("Parola chiave o più termini (separa con ;)", key="kw_query", placeholder="es. illuminazione; videosorveglianza")
    with col_q2:
        combine_terms = st.radio("Combinazione termini", ["OR", "AND"], horizontal=True, key="kw_op")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        use_regex = st.checkbox("Usa regex", value=False, help="Se disattivo, i caratteri speciali verranno escapati")
    with col_opt2:
        case_sensitive = st.checkbox("Maiuscole/minuscole", value=False)
    with col_opt3:
        limit_preview = st.number_input("Righe anteprima", min_value=50, max_value=2000, value=200, step=50)

    # Filtri numerici opzionali
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sconto_min = st.number_input("Sconto min %", min_value=0.0, max_value=100.0, value=0.0, step=1.0) if sconto_col_search else None
    with col_f2:
        sconto_max = st.number_input("Sconto max %", min_value=0.0, max_value=100.0, value=100.0, step=1.0) if sconto_col_search else None
    with col_f3:
        part_min = st.number_input("Partecipanti min", min_value=0, max_value=100, value=0, step=1) if partecipanti_col_search else None
    with col_f4:
        part_max = st.number_input("Partecipanti max", min_value=0, max_value=100, value=100, step=1) if partecipanti_col_search else None

    do_search = st.button("Cerca", type="primary")

    if do_search:
        df_src = filtered_df
        if not cols_sel:
            st.warning("Seleziona almeno un campo da cercare")
        else:
            # Prepara termini
            terms = []
            if query:
                parts = [p.strip() for p in query.split(';') if p.strip()]
                terms.extend(parts)
            if not terms:
                st.info("Inserisci almeno un termine di ricerca")
            else:
                flags = 0 if case_sensitive else re.IGNORECASE
                # Costruisci maschera per colonna e combina tra colonne con OR
                col_masks = []
                for col in cols_sel:
                    s = df_src[col].astype(str)
                    if combine_terms == "OR":
                        mask_col = s.str.contains(re.escape(terms[0]) if not use_regex else terms[0], regex=True, na=False, flags=flags)
                        for t in terms[1:]:
                            patt = re.escape(t) if not use_regex else t
                            mask_col = mask_col | s.str.contains(patt, regex=True, na=False, flags=flags)
                    else:
                        mask_col = s.str.contains(re.escape(terms[0]) if not use_regex else terms[0], regex=True, na=False, flags=flags)
                        for t in terms[1:]:
                            patt = re.escape(t) if not use_regex else t
                            mask_col = mask_col & s.str.contains(patt, regex=True, na=False, flags=flags)
                    col_masks.append(mask_col)

                if col_masks:
                    mask = col_masks[0]
                    for m in col_masks[1:]:
                        mask = mask | m  # OR tra colonne
                    results = df_src[mask].copy()
                else:
                    results = df_src.head(0).copy()

                # Filtri numerici: sconto (%) e partecipanti
                if sconto_col_search and len(results) > 0:
                    s = pd.to_numeric(results[sconto_col_search], errors='coerce')
                    results = results[(s >= sconto_min) & (s <= sconto_max)]
                if partecipanti_col_search and len(results) > 0:
                    p = pd.to_numeric(results[partecipanti_col_search], errors='coerce')
                    results = results[(p >= (part_min if part_min is not None else -1)) & (p <= (part_max if part_max is not None else 1e9))]

                st.success(f"Trovate {len(results):,} gare".replace(",", "."))

                # Colonne per anteprima
                preview_cols = [
                    'chiave', 'cig', 'ocid',
                    'data_aggiudicazione', 'award_date', 'oggetto', 'tender_title', 'ente_appaltante', 'buyer_name',
                    'aggiudicatario', 'supplier_name', 'importo_aggiudicazione', 'award_amount', 'sconto', 'categoria',
                    'procedura', 'regione', 'comune', 'fonte'
                ]
                # aggiungi partecipanti se esiste
                if partecipanti_col_search:
                    preview_cols.append(partecipanti_col_search)
                preview_cols = [c for c in preview_cols if c in results.columns]
                preview = results[preview_cols].copy()

                # Formattazioni base
                if 'data_aggiudicazione' in preview.columns:
                    preview['data_aggiudicazione'] = pd.to_datetime(preview['data_aggiudicazione'], errors='coerce').dt.strftime('%Y-%m-%d')
                if 'award_date' in preview.columns:
                    preview['award_date'] = pd.to_datetime(preview['award_date'], errors='coerce').dt.strftime('%Y-%m-%d')
                if 'importo_aggiudicazione' in preview.columns:
                    preview['importo_aggiudicazione'] = pd.to_numeric(preview['importo_aggiudicazione'], errors='coerce').apply(lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x>=1e6 else (f"€{x/1e3:.0f}K" if pd.notna(x) else ''))
                if 'award_amount' in preview.columns:
                    preview['award_amount'] = pd.to_numeric(preview['award_amount'], errors='coerce').apply(lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) and x>=1e6 else (f"€{x/1e3:.0f}K" if pd.notna(x) else ''))
                if 'sconto' in preview.columns:
                    preview['sconto'] = pd.to_numeric(preview['sconto'], errors='coerce').apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '')
                if partecipanti_col_search and partecipanti_col_search in preview.columns:
                    preview[partecipanti_col_search] = pd.to_numeric(preview[partecipanti_col_search], errors='coerce').astype('Int64')

                show_dataframe(preview.head(int(limit_preview)), use_container_width=True, height=500)

                # ==================== AI: ANALISI SINGOLA GARA ====================
                st.markdown("### 🤖 Analisi AI (seleziona una gara)")
                if not get_openai_api_key():
                    st.info("Inserisci la tua OpenAI API Key nella sidebar (sezione 🤖 AI) per usare l’analisi.")
                else:
                    id_col = next((c for c in ['chiave', 'cig', 'ocid'] if c in results.columns), None)
                    if not id_col:
                        st.info("Nessun identificativo (chiave/cig/ocid) disponibile nei risultati per fare l’analisi.")
                    else:
                        max_opts = min(500, len(results))
                        cand = results.head(max_opts).copy()
                        cand[id_col] = cand[id_col].astype(str).str.strip()
                        cand = cand[cand[id_col].ne("") & ~cand[id_col].str.lower().isin({"nan", "none"})].copy()
                        # de-dup per id
                        cand = cand.drop_duplicates(subset=[id_col], keep="first")

                        label_map = {}
                        for r in cand.to_dict(orient="records"):
                            k = str(r.get(id_col, "")).strip()
                            if not k:
                                continue
                            label_map[k] = _ai_select_label_from_row(r, k)

                        options = list(label_map.keys())
                        if not options:
                            st.info("Nessuna gara selezionabile per analisi AI nei primi risultati.")
                        else:
                            sel_id = st.selectbox(
                                "Gara",
                                options=options,
                                format_func=lambda x: label_map.get(x, x),
                                key="ai_select_search_result",
                            )
                            question = st.text_area(
                                "Domanda (opzionale)",
                                placeholder="Es. che cosa sappiamo sulla scadenza e cosa manca da verificare?",
                                key="ai_question_search_result",
                                height=80,
                            )
                            if st.button("🤖 Analisi AI", type="primary", key="ai_run_search_result"):
                                cache_key = f"search:{sel_id}:{hashlib.md5(question.encode('utf-8')).hexdigest()[:8]}"
                                if "ai_gara_cache" not in st.session_state:
                                    st.session_state.ai_gara_cache = {}
                                if cache_key in st.session_state.ai_gara_cache:
                                    st.markdown(st.session_state.ai_gara_cache[cache_key])
                                else:
                                    rec_df = results[results[id_col].astype(str).str.strip() == sel_id]
                                    if len(rec_df) == 0:
                                        st.error("Record non trovato nei risultati (id non più presente).")
                                    else:
                                        rec = rec_df.iloc[0].to_dict()
                                        out = ai_analyze_gara(rec, question=question, model="gpt-5-nano")
                                        if not out:
                                            st.error("Errore: nessuna risposta (controlla API key / rete / permessi modello).")
                                        else:
                                            st.session_state.ai_gara_cache[cache_key] = out
                                            st.markdown(out)

                # Download completi
                st.markdown("---")
                col_dl1, col_dl2 = st.columns(2)
                csv_bytes = results.to_csv(index=False).encode('utf-8')
                with col_dl1:
                    st.download_button(
                        label=f"📥 Scarica CSV (tutte le {len(results):,} righe)".replace(",", "."),
                        data=csv_bytes,
                        file_name=f"gare_search_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with col_dl2:
                    try:
                        from io import BytesIO
                        buf = BytesIO()
                        _sanitize_for_excel(results).to_excel(buf, index=False, engine='openpyxl')
                        st.download_button(
                            label=f"📥 Scarica Excel (tutte le {len(results):,} righe)".replace(",", "."),
                            data=buf.getvalue(),
                            file_name=f"gare_search_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        pass

# ==================== TAB 4: AGGIUDICATARI ====================
if tab4:
  with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 20 Aggiudicatari per Valore")

        # Calcola top aggiudicatari direttamente da filtered_df (rispetta i filtri)
        supplier_col = 'supplier_name' if 'supplier_name' in filtered_df.columns else 'aggiudicatario'
        value_col = 'award_amount' if 'award_amount' in filtered_df.columns else 'importo_aggiudicazione'

        if supplier_col in filtered_df.columns and value_col in filtered_df.columns:
            # Calcola da dati filtrati
            top_df = filtered_df.groupby(supplier_col, observed=True).agg({
                value_col: 'sum',
                'ocid': 'count'
            }).reset_index()
            top_df.columns = ['Aggiudicatario', 'valore', 'num_gare']
            top_df = top_df.dropna(subset=['Aggiudicatario'])
            top_df = top_df[top_df['Aggiudicatario'] != '']
            top_df = top_df.sort_values('valore', ascending=False).head(20)
        else:
            # Fallback ai dati pre-calcolati
            top_df = pd.DataFrame(data['top_aggiudicatari'])

        # Tronca nomi troppo lunghi per visualizzazione
        top_df['Aggiudicatario_display'] = top_df['Aggiudicatario'].apply(
            lambda x: x[:35] + '...' if isinstance(x, str) and len(x) > 38 else x
        )

        fig_top = px.bar(
            top_df,
            x='valore',
            y='Aggiudicatario_display',
            orientation='h',
            color='num_gare',
            color_continuous_scale='Viridis',
            text=top_df['valore'].apply(lambda x: f'€{x/1e6:.0f}M'),
            labels={'valore': 'Valore (€)', 'num_gare': 'N. Gare', 'Aggiudicatario_display': 'Aggiudicatario'}
        )
        fig_top.update_layout(height=600, yaxis={'categoryorder': 'total ascending', 'title': 'Aggiudicatario'})
        fig_top.update_traces(textposition='outside')
        render_chart_with_save(fig_top, "Top 20 Aggiudicatari", "Ranking aggiudicatari per valore totale aggiudicazioni", "top_aggiudicatari")

    with col2:
        st.subheader("📊 Concentrazione Mercato")

        # Market concentration
        top_df_sorted = top_df.sort_values('valore', ascending=False)
        top_df_sorted['cumsum'] = top_df_sorted['valore'].cumsum()
        top_df_sorted['cumsum_pct'] = top_df_sorted['cumsum'] / top_df_sorted['valore'].sum() * 100
        top_df_sorted['rank'] = range(1, len(top_df_sorted) + 1)

        fig_conc = px.area(
            top_df_sorted,
            x='rank',
            y='cumsum_pct',
            labels={'rank': 'Top N Aggiudicatari', 'cumsum_pct': '% Valore Cumulato'},
            title='Curva di Concentrazione'
        )
        fig_conc.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
        fig_conc.update_layout(height=300)
        render_chart_with_save(fig_conc, "Concentrazione Mercato", "Curva di concentrazione aggiudicatari", "concentrazione_mercato")

        # Stats
        top5_share = top_df.head(5)['valore'].sum() / top_df['valore'].sum() * 100
        top10_share = top_df.head(10)['valore'].sum() / top_df['valore'].sum() * 100

        st.metric("🔝 Quota Top 5", f"{top5_share:.1f}%")
        st.metric("🔝 Quota Top 10", f"{top10_share:.1f}%")

        # HHI Index
        total_val = top_df['valore'].sum()
        hhi = ((top_df['valore'] / total_val * 100) ** 2).sum()
        st.metric("📊 Indice HHI", f"{hhi:.0f}", help="<1500=competitivo, 1500-2500=moderato, >2500=concentrato")

# ==================== TAB 5: CONSIP ====================
if tab5:
  with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏛️ CONSIP per Tipo Accordo")
        consip_df = pd.DataFrame(data['consip']['by_tipo'])

        fig_consip = px.pie(
            consip_df,
            values='valore',
            names='TipoAccordo',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_consip.update_layout(height=350)
        render_chart_with_save(fig_consip, "CONSIP per Tipo", "Distribuzione gare CONSIP per tipo accordo", "consip_tipo")

    with col2:
        st.subheader("📊 Confronto Tipi Accordo")
        fig = px.bar(
            consip_df,
            x='TipoAccordo',
            y=['num_gare', 'valore'],
            barmode='group',
            labels={'value': 'Valore', 'variable': 'Metrica'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # SIE Edizioni
    if data['consip'].get('sie_edizioni'):
        st.subheader("📈 Edizioni SIE")
        sie_df = pd.DataFrame(data['consip']['sie_edizioni'])
        fig = px.bar(
            sie_df,
            x='Edizione',
            y='valore',
            color='num_gare',
            text=sie_df['valore'].apply(lambda x: f'€{x/1e6:.0f}M'),
            labels={'valore': 'Valore (€)', 'num_gare': 'N. Gare'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # CONSIP per regione
    if data['consip'].get('per_regione'):
        st.subheader("🗺️ CONSIP per Regione")
        consip_reg = pd.DataFrame(data['consip']['per_regione'])
        fig = px.bar(
            consip_reg.head(15),
            x='valore',
            y='Regione',
            orientation='h',
            color='num_gare',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 6: STATISTICHE AVANZATE ====================
if tab6:
  with tab6:
    st.subheader("📊 Analisi Statistica Avanzata")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Distribuzione Sconti")
        # Filtra sconti validi: escludi 0, null e valori > 100
        valid_sconti = filtered_df[(filtered_df['sconto'] > 0) & (filtered_df['sconto'] <= 100)]
        if len(valid_sconti) > 0:
            fig = px.histogram(
                valid_sconti,
                x='sconto',
                nbins=50,
                color_discrete_sequence=[CGL_GREEN],
                labels={'sconto': 'Sconto %'}
            )
            sconto_mean = valid_sconti['sconto'].mean()
            sconto_median = valid_sconti['sconto'].median()
            fig.add_vline(x=sconto_mean, line_dash="dash", line_color="red",
                          annotation_text=f"Media: {sconto_mean:.1f}%", annotation_position="top right")
            fig.add_vline(x=sconto_median, line_dash="dash", line_color="green",
                          annotation_text=f"Mediana: {sconto_median:.1f}%", annotation_position="bottom right")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"ℹ️ Analisi basata su {len(valid_sconti):,} gare con sconto > 0%")
        else:
            st.info("Nessun dato di sconto valido disponibile")

    with col2:
        st.markdown("### 💰 Distribuzione Valori (Log)")
        valid_amounts = filtered_df[filtered_df['award_amount'] > 0]['award_amount']
        fig = px.histogram(
            x=np.log10(valid_amounts),
            nbins=50,
            color_discrete_sequence=[CGL_BLUE],
            labels={'x': 'Log10(Valore €)'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### 👥 Distribuzione Offerte Ricevute")
        partecipanti_col = 'offerte_ricevute' if 'offerte_ricevute' in filtered_df.columns else 'parties_count'
        if partecipanti_col in filtered_df.columns:
            valid_part = filtered_df[filtered_df[partecipanti_col].notna() & (filtered_df[partecipanti_col] >= 1) & (filtered_df[partecipanti_col] <= 30)]
            if len(valid_part) > 10:
                fig = px.histogram(
                    valid_part,
                    x=partecipanti_col,
                    nbins=20,
                    color_discrete_sequence=[CGL_CYAN],
                    labels={partecipanti_col: 'N. Offerte'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key="dist_offerte")
            else:
                st.info("Dati offerte insufficienti")
        else:
            st.info("Campo offerte non disponibile")

    # Box plot per categoria
    st.subheader("📦 Box Plot Sconti per Categoria")
    cat_col = '_categoria' if '_categoria' in filtered_df.columns else 'categoria'
    # Filtra sconti validi: escludi 0, null e valori > 100
    valid_box = filtered_df[(filtered_df['sconto'] > 0) & (filtered_df['sconto'] <= 100)]
    if cat_col in valid_box.columns and len(valid_box) > 50:
        fig = px.box(
            valid_box,
            x=cat_col,
            y='sconto',
            color=cat_col,
            labels={cat_col: 'Categoria', 'sconto': 'Sconto %'}
        )
        fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key="box_sconti_cat")
    else:
        st.info("Dati insufficienti per box plot")

    # Correlazione
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔗 Sconto vs Valore")
        valid_data = filtered_df[filtered_df['award_amount'] > 0]
        sample = valid_data.sample(min(5000, len(valid_data))) if len(valid_data) > 0 else valid_data
        fig = px.scatter(
            sample,
            x='award_amount',
            y='sconto',
            color='_categoria',
            opacity=0.5,
            log_x=True,
            labels={'award_amount': 'Valore (€)', 'sconto': 'Sconto %'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Distribuzione Mensile")
        if 'mese' in filtered_df.columns and filtered_df['mese'].notna().any():
            valid_monthly = filtered_df[filtered_df['mese'].notna() & filtered_df['anno'].notna()]
            if len(valid_monthly) > 10:
                monthly = valid_monthly.groupby(['anno', 'mese'], observed=True).agg({
                    'award_amount': 'sum',
                    'sconto': 'mean'
                }).reset_index()
                monthly['periodo'] = monthly['anno'].astype(int).astype(str) + '-' + monthly['mese'].astype(int).astype(str).str.zfill(2)
                monthly = monthly[monthly['anno'].between(2015, 2025)]

                if len(monthly) > 0:
                    fig = px.line(
                        monthly.sort_values('periodo'),
                        x='periodo',
                        y='award_amount',
                        labels={'periodo': 'Periodo', 'award_amount': 'Valore (€)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key="dist_mensile_stat")
                else:
                    st.info("Nessun dato mensile disponibile")
            else:
                st.info("Dati mensili insufficienti")
        else:
            st.info("Campo mese non disponibile")

    # Statistiche descrittive
    st.subheader("📋 Statistiche Descrittive")

    offerte_col = 'offerte_ricevute' if 'offerte_ricevute' in filtered_df.columns else None
    stats = {
        'Metrica': ['Valore Gara', 'Sconto %', 'Offerte Ricevute'],
        'Media': [
            filtered_df['award_amount'].mean(),
            filtered_df['sconto'].mean(),
            filtered_df[offerte_col].mean() if offerte_col else 0
        ],
        'Mediana': [
            filtered_df['award_amount'].median(),
            filtered_df['sconto'].median(),
            filtered_df[offerte_col].median() if offerte_col else 0
        ],
        'Std': [
            filtered_df['award_amount'].std(),
            filtered_df['sconto'].std(),
            filtered_df[offerte_col].std() if offerte_col else 0
        ],
        'Min': [
            filtered_df['award_amount'].min(),
            filtered_df['sconto'].min(),
            filtered_df[offerte_col].min() if offerte_col else 0
        ],
        'Max': [
            filtered_df['award_amount'].max(),
            filtered_df['sconto'].max(),
            filtered_df[offerte_col].max() if offerte_col else 0
        ],
    }

    stats_df = pd.DataFrame(stats)
    show_dataframe(stats_df, use_container_width=True, hide_index=True)

# ==================== TAB 7: RICERCA CITTÀ / STAZIONE APPALTANTE ====================
if tab7:
  with tab7:
    st.subheader("🔍 Ricerca Servizi per Città o Stazione Appaltante")

    # Helper per trovare colonne dinamicamente (definito anche qui per sicurezza)
    def get_col_city(df, candidates):
        for col in candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    # Identifica colonne dinamicamente
    locality_col = get_col_city(filtered_df, ['comune', 'buyer_locality', 'citta', 'city'])
    amount_col_city = get_col_city(filtered_df, ['award_amount', 'importo_aggiudicazione', 'tender_amount'])
    buyer_col_city = get_col_city(filtered_df, ['buyer_name', 'ente_appaltante', 'stazione_appaltante'])
    supplier_col_city = get_col_city(filtered_df, ['supplier_name', 'aggiudicatario', 'award_supplier_name'])
    sconto_col_city = get_col_city(filtered_df, ['sconto', 'ribasso', 'discount'])
    cat_col_city = get_col_city(filtered_df, ['_categoria', 'categoria', 'category'])
    id_col_city = get_col_city(filtered_df, ['chiave', 'cig', 'ocid', 'CIG'])

    # Get unique cities and stazioni appaltanti from filtered data
    if locality_col and locality_col in filtered_df.columns:
        cities_list = sorted(filtered_df[locality_col].dropna().unique().tolist())
    else:
        cities_list = []

    if buyer_col_city and buyer_col_city in filtered_df.columns:
        stazioni_list = sorted(filtered_df[buyer_col_city].dropna().unique().tolist())
    else:
        stazioni_list = []

    st.info(f"💡 I risultati rispettano i filtri selezionati nella sidebar ({len(filtered_df):,} gare filtrate)".replace(",", "."))

    # Tipo di ricerca
    tipo_ricerca = st.radio(
        "Cerca per:",
        ["🏙️ Città", "🏛️ Stazione Appaltante"],
        horizontal=True,
        key="tipo_ricerca_territoriale"
    )

    # Search box
    col1, col2 = st.columns([2, 1])
    with col1:
        if tipo_ricerca == "🏙️ Città":
            if cities_list:
                citta_search = st.selectbox(
                    "Seleziona o cerca una città",
                    options=[""] + cities_list,
                    index=0,
                    help="Digita per cercare tra le città",
                    key="search_citta_territoriale"
                )
                stazione_search = None
            else:
                st.warning("Colonna città non trovata nel dataset")
                citta_search = None
                stazione_search = None
        else:
            if stazioni_list:
                stazione_search = st.multiselect(
                    "Seleziona una o più stazioni appaltanti",
                    options=stazioni_list,
                    default=[],
                    help="Puoi selezionare più stazioni appaltanti per aggregare i dati",
                    key="search_stazione_territoriale"
                )
                citta_search = None
            else:
                st.warning("Colonna stazione appaltante non trovata nel dataset")
                citta_search = None
                stazione_search = []
    with col2:
        solo_attivi = st.checkbox("Solo contratti attivi (2023-2025)", value=True)

    # Determina quale ricerca è attiva
    search_active = False
    search_label = ""
    city_df = pd.DataFrame()

    if tipo_ricerca == "🏙️ Città" and citta_search and locality_col:
        # Filter data for selected city from already filtered data
        city_df = filtered_df[filtered_df[locality_col].str.upper() == citta_search.upper()].copy()
        search_label = citta_search.upper()
        search_icon = "📍"
        search_active = True

    elif tipo_ricerca == "🏛️ Stazione Appaltante" and stazione_search and len(stazione_search) > 0 and buyer_col_city:
        # Filter data for selected stazioni appaltanti (multiple selection)
        city_df = filtered_df[filtered_df[buyer_col_city].isin(stazione_search)].copy()
        if len(stazione_search) == 1:
            search_label = stazione_search[0]
        else:
            search_label = f"{len(stazione_search)} Stazioni Appaltanti"
        search_icon = "🏛️"
        search_active = True

    if search_active and len(city_df) > 0:
        if solo_attivi and 'anno' in city_df.columns:
            city_df = city_df[city_df['anno'] >= 2023]

        if len(city_df) > 0:
            st.markdown(f"### {search_icon} {search_label}")

            # KPIs - usa colonne dinamiche
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📋 Totale Gare", f"{len(city_df):,}".replace(",", "."))

            if amount_col_city:
                valore_tot = pd.to_numeric(city_df[amount_col_city], errors='coerce').sum()
                col2.metric("💰 Valore Totale", f"€{valore_tot/1e6:.1f}M")
            else:
                col2.metric("💰 Valore Totale", "N/D")

            if sconto_col_city:
                sconto_medio = pd.to_numeric(city_df[sconto_col_city], errors='coerce').mean()
                col3.metric("📉 Sconto Medio", f"{sconto_medio:.1f}%" if pd.notna(sconto_medio) else "N/D")
            else:
                col3.metric("📉 Sconto Medio", "N/D")

            # Per stazione appaltante mostra città, per città mostra enti
            if tipo_ricerca == "🏛️ Stazione Appaltante" and locality_col:
                col4.metric("🏙️ Città", f"{city_df[locality_col].nunique()}")
            elif buyer_col_city:
                col4.metric("🏢 Enti Appaltanti", f"{city_df[buyer_col_city].nunique()}")
            else:
                col4.metric("🏢 Enti", "N/D")

            # Services by category
            st.markdown("---")
            st.markdown("#### 📦 Servizi per Categoria")

            if cat_col_city and amount_col_city:
                # Costruisci aggregazione dinamicamente
                agg_dict = {}
                if id_col_city:
                    agg_dict[id_col_city] = 'count'
                if amount_col_city:
                    agg_dict[amount_col_city] = 'sum'
                if sconto_col_city:
                    agg_dict[sconto_col_city] = 'mean'

                if agg_dict:
                    cat_city = city_df.groupby(cat_col_city, observed=True).agg(agg_dict).reset_index()
                    # Rinomina colonne
                    new_cols = ['Categoria']
                    if id_col_city:
                        new_cols.append('N. Gare')
                    if amount_col_city:
                        new_cols.append('Valore (€)')
                    if sconto_col_city:
                        new_cols.append('Sconto Medio %')
                    cat_city.columns = new_cols
                    cat_city = cat_city.sort_values('Valore (€)' if 'Valore (€)' in cat_city.columns else 'N. Gare', ascending=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(
                            cat_city,
                            values='N. Gare' if 'N. Gare' in cat_city.columns else cat_city.columns[1],
                            names='Categoria',
                            title='Distribuzione per Categoria',
                            hole=0.3
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        if 'Valore (€)' in cat_city.columns:
                            fig = px.bar(
                                cat_city,
                                x='Valore (€)',
                                y='Categoria',
                                orientation='h',
                                color='Sconto Medio %' if 'Sconto Medio %' in cat_city.columns else None,
                                color_continuous_scale='RdYlGn',
                                title='Valore per Categoria'
                            )
                            fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dati categoria non disponibili per questo filtro")

            # Top suppliers
            st.markdown("---")
            fornitori_title = "Top Fornitori" if tipo_ricerca == "🏛️ Stazione Appaltante" else "Top Fornitori nella Città"
            st.markdown(f"#### 🏆 {fornitori_title}")

            if supplier_col_city:
                agg_dict_sup = {}
                if id_col_city:
                    agg_dict_sup[id_col_city] = 'count'
                if amount_col_city:
                    agg_dict_sup[amount_col_city] = 'sum'
                if sconto_col_city:
                    agg_dict_sup[sconto_col_city] = 'mean'

                if agg_dict_sup:
                    top_suppliers = city_df.groupby(supplier_col_city, observed=True).agg(agg_dict_sup).reset_index()
                    new_cols_sup = ['Fornitore']
                    if id_col_city:
                        new_cols_sup.append('N. Gare')
                    if amount_col_city:
                        new_cols_sup.append('Valore (€)')
                    if sconto_col_city:
                        new_cols_sup.append('Sconto Medio %')
                    top_suppliers.columns = new_cols_sup
                    top_suppliers = top_suppliers.sort_values('Valore (€)' if 'Valore (€)' in top_suppliers.columns else 'N. Gare', ascending=False).head(15)

                    if 'Valore (€)' in top_suppliers.columns:
                        fig = px.bar(
                            top_suppliers,
                            x='Valore (€)',
                            y='Fornitore',
                            orientation='h',
                            color='N. Gare' if 'N. Gare' in top_suppliers.columns else None,
                            color_continuous_scale='Viridis',
                            text=top_suppliers['Valore (€)'].apply(lambda x: f'€{x/1e6:.1f}M' if x > 1e6 else f'€{x/1e3:.0f}K')
                        )
                        fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        show_dataframe(top_suppliers, use_container_width=True)
            else:
                st.info("Dati fornitori non disponibili")

            # Top buyers (stazioni appaltanti) - solo se ricerca per città
            if tipo_ricerca == "🏙️ Città":
                st.markdown("---")
                st.markdown("#### 🏛️ Stazioni Appaltanti nella Città")

                if buyer_col_city:
                    agg_dict_buy = {}
                    if id_col_city:
                        agg_dict_buy[id_col_city] = 'count'
                    if amount_col_city:
                        agg_dict_buy[amount_col_city] = 'sum'
                    if sconto_col_city:
                        agg_dict_buy[sconto_col_city] = 'mean'

                    if agg_dict_buy:
                        top_buyers = city_df.groupby(buyer_col_city, observed=True).agg(agg_dict_buy).reset_index()
                        new_cols_buy = ['Stazione Appaltante']
                        if id_col_city:
                            new_cols_buy.append('N. Gare')
                        if amount_col_city:
                            new_cols_buy.append('Valore (€)')
                        if sconto_col_city:
                            new_cols_buy.append('Sconto Medio %')
                        top_buyers.columns = new_cols_buy
                        top_buyers = top_buyers.sort_values('Valore (€)' if 'Valore (€)' in top_buyers.columns else 'N. Gare', ascending=False).head(15)

                        if 'Valore (€)' in top_buyers.columns:
                            fig = px.bar(
                                top_buyers,
                                x='Valore (€)',
                                y='Stazione Appaltante',
                                orientation='h',
                                color='Sconto Medio %' if 'Sconto Medio %' in top_buyers.columns else None,
                                color_continuous_scale='RdYlGn',
                                text=top_buyers['Valore (€)'].apply(lambda x: f'€{x/1e6:.1f}M' if x > 1e6 else f'€{x/1e3:.0f}K')
                            )
                            fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            show_dataframe(top_buyers, use_container_width=True)
                else:
                    st.info("Dati stazioni appaltanti non disponibili")

            # Detailed services table
            st.markdown("---")
            st.markdown("#### 📋 Dettaglio Servizi Attivi")

            # Prepare display columns dinamicamente
            date_col = get_col_city(city_df, ['award_date', 'data_aggiudicazione', 'data'])
            title_col = get_col_city(city_df, ['tender_title', 'oggetto', 'descrizione'])

            display_cols = []
            col_names = []
            if date_col:
                display_cols.append(date_col)
                col_names.append('Data')
            if buyer_col_city:
                display_cols.append(buyer_col_city)
                col_names.append('Stazione Appaltante')
            if cat_col_city:
                display_cols.append(cat_col_city)
                col_names.append('Categoria')
            if supplier_col_city:
                display_cols.append(supplier_col_city)
                col_names.append('Fornitore')
            if title_col:
                display_cols.append(title_col)
                col_names.append('Oggetto')
            if amount_col_city:
                display_cols.append(amount_col_city)
                col_names.append('Valore')
            if sconto_col_city:
                display_cols.append(sconto_col_city)
                col_names.append('Sconto')

            # Filtra solo colonne esistenti
            display_cols = [c for c in display_cols if c in city_df.columns]

            if display_cols:
                display_df = city_df[display_cols].copy()

                # Formatta date
                if date_col and date_col in display_df.columns:
                    display_df[date_col] = pd.to_datetime(display_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')

                # Formatta valori
                if amount_col_city and amount_col_city in display_df.columns:
                    display_df[amount_col_city] = pd.to_numeric(display_df[amount_col_city], errors='coerce').apply(lambda x: f'€{x:,.0f}'.replace(',', '.') if pd.notna(x) else '-')

                # Formatta sconto
                if sconto_col_city and sconto_col_city in display_df.columns:
                    display_df[sconto_col_city] = pd.to_numeric(display_df[sconto_col_city], errors='coerce').apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-')

                display_df.columns = col_names[:len(display_df.columns)]

                # Sort by date descending
                if 'Data' in display_df.columns:
                    display_df = display_df.sort_values('Data', ascending=False)

                # Pagination
                page_size = 50
                total_pages = (len(display_df) - 1) // page_size + 1
                page = st.number_input('Pagina', min_value=1, max_value=max(1, total_pages), value=1, key='city_page')

                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size

                show_dataframe(display_df.iloc[start_idx:end_idx], use_container_width=True, height=500)
                st.caption(f"Mostrando {start_idx+1}-{min(end_idx, len(display_df))} di {len(display_df)} gare")

            # Export button
            st.download_button(
                label="📥 Scarica CSV completo",
                data=city_df.to_csv(index=False).encode('utf-8'),
                file_name=f'gare_{citta_search.lower().replace(" ", "_")}.csv',
                mime='text/csv'
            )

            # Trend over years
            if 'anno' in city_df.columns:
                st.markdown("---")
                st.markdown("#### 📈 Trend Storico")

                agg_dict_year = {}
                if id_col_city:
                    agg_dict_year[id_col_city] = 'count'
                if amount_col_city:
                    agg_dict_year[amount_col_city] = 'sum'
                if sconto_col_city:
                    agg_dict_year[sconto_col_city] = 'mean'

                if agg_dict_year:
                    yearly = city_df.groupby('anno', observed=True).agg(agg_dict_year).reset_index()
                    year_cols = ['Anno']
                    if id_col_city:
                        year_cols.append('N. Gare')
                    if amount_col_city:
                        year_cols.append('Valore')
                    if sconto_col_city:
                        year_cols.append('Sconto Medio')
                    yearly.columns = year_cols
                    yearly = yearly[yearly['Anno'].between(2015, 2025)]

                    if len(yearly) > 0 and 'Valore' in yearly.columns:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(
                            go.Bar(x=yearly['Anno'], y=yearly['Valore'], name='Valore (€)', marker_color=CGL_GREEN),
                            secondary_y=False
                        )
                        if 'N. Gare' in yearly.columns:
                            fig.add_trace(
                                go.Scatter(x=yearly['Anno'], y=yearly['N. Gare'], name='N. Gare', line=dict(color=CGL_BLUE, width=3)),
                                secondary_y=True
                            )
                        fig.update_yaxes(title_text="Valore (€)", secondary_y=False)
                        fig.update_yaxes(title_text="Numero Gare", secondary_y=True)
                        fig.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                        st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Nessuna gara trovata per {citta_search}")
    else:
        # Show top cities summary - USA filtered_df!
        st.markdown("#### 🏙️ Top 20 Città per Valore (dati filtrati)")

        if locality_col and locality_col in filtered_df.columns:
            # Costruisci aggregazione dinamica
            agg_dict_summary = {}
            if id_col_city:
                agg_dict_summary[id_col_city] = 'count'
            if amount_col_city:
                agg_dict_summary[amount_col_city] = 'sum'
            if sconto_col_city:
                agg_dict_summary[sconto_col_city] = 'mean'
            if buyer_col_city:
                agg_dict_summary[buyer_col_city] = 'nunique'

            if agg_dict_summary:
                city_summary = filtered_df.groupby(locality_col, observed=True).agg(agg_dict_summary).reset_index()
                sum_cols = ['Città']
                if id_col_city:
                    sum_cols.append('N. Gare')
                if amount_col_city:
                    sum_cols.append('Valore (€)')
                if sconto_col_city:
                    sum_cols.append('Sconto Medio %')
                if buyer_col_city:
                    sum_cols.append('N. Enti')
                city_summary.columns = sum_cols
                city_summary = city_summary.sort_values('Valore (€)' if 'Valore (€)' in city_summary.columns else 'N. Gare', ascending=False).head(20)

                if 'Valore (€)' in city_summary.columns:
                    fig = px.bar(
                        city_summary,
                        x='Valore (€)',
                        y='Città',
                        orientation='h',
                        color='N. Gare' if 'N. Gare' in city_summary.columns else None,
                        color_continuous_scale='Viridis',
                        text=city_summary['Valore (€)'].apply(lambda x: f'€{x/1e9:.1f}B' if x > 1e9 else f'€{x/1e6:.0f}M')
                    )
                    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                show_dataframe(city_summary, use_container_width=True)
        else:
            st.info("Dati città non disponibili per i filtri selezionati")

# ==================== TAB 8: MAPPA CONSIP ====================
if tab8:
  with tab8:
    st.subheader("🗺️ Mappa Contratti CONSIP")

    # Use ServizioLuce data for CONSIP
    if len(consip_raw_df) > 0:
        # Preprocess CONSIP data
        consip_df_raw = consip_raw_df.copy()
        consip_df_raw['anno'] = pd.to_datetime(consip_df_raw['DataAggiudicazione'], errors='coerce').dt.year
        if consip_df_raw['anno'].isna().all():
            consip_df_raw['anno'] = pd.to_datetime(consip_df_raw['DataPubblicazione'], errors='coerce').dt.year
        consip_df_raw['award_amount'] = consip_df_raw['ImportoAggiudicazione'].fillna(consip_df_raw['ImportoGara'])
        consip_df_raw['sconto'] = consip_df_raw['Sconto'].fillna(consip_df_raw['Sconto %'])
        consip_df_raw['buyer_locality'] = consip_df_raw['Comune']
        consip_df_raw['buyer_name'] = consip_df_raw['denominazione_centro_costo'].fillna(consip_df_raw['DENOMINAZIONE_SA_DELEGANTE'])
        consip_df_raw['tender_title'] = consip_df_raw['OggettoGara'].fillna(consip_df_raw['Oggetto'])

    if len(consip_raw_df) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            tipo_accordo_list = ['Tutti'] + sorted(consip_df_raw['TipoAccordo'].dropna().unique().tolist())
            tipo_sel = st.selectbox("Tipo Accordo", tipo_accordo_list)

        with col2:
            anni_consip = ['Tutti'] + sorted([int(y) for y in consip_df_raw['anno'].dropna().unique() if 2015 <= y <= 2025])
            anno_consip_sel = st.selectbox("Anno Contratto", anni_consip, key='anno_consip')

        with col3:
            # Edizione filter if SIE
            if 'Edizione' in consip_df_raw.columns:
                edizioni = ['Tutte'] + sorted([str(e) for e in consip_df_raw['Edizione'].dropna().unique()])
                edizione_sel = st.selectbox("Edizione", edizioni)
            else:
                edizione_sel = 'Tutte'

        # Apply filters
        filtered_consip = consip_df_raw.copy()
        if tipo_sel != 'Tutti':
            filtered_consip = filtered_consip[filtered_consip['TipoAccordo'] == tipo_sel]
        if anno_consip_sel != 'Tutti':
            filtered_consip = filtered_consip[filtered_consip['anno'] == anno_consip_sel]
        if edizione_sel != 'Tutte' and 'Edizione' in filtered_consip.columns:
            filtered_consip = filtered_consip[filtered_consip['Edizione'].astype(str) == edizione_sel]

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏛️ Contratti CONSIP", f"{len(filtered_consip):,}".replace(",", "."))
        col2.metric("💰 Valore Totale", f"€{filtered_consip['award_amount'].sum()/1e6:.0f}M")
        col3.metric("📉 Sconto Medio", f"{filtered_consip['sconto'].mean():.1f}%" if len(filtered_consip) > 0 else "-")
        col4.metric("🏢 Enti Coinvolti", f"{filtered_consip['buyer_name'].nunique()}")

        st.markdown("---")

        # Aggregate by city for map
        city_coords = {
            'ROMA': [41.9028, 12.4964], 'MILANO': [45.4642, 9.1900], 'NAPOLI': [40.8518, 14.2681],
            'TORINO': [45.0703, 7.6869], 'PALERMO': [38.1157, 13.3615], 'GENOVA': [44.4056, 8.9463],
            'BOLOGNA': [44.4949, 11.3426], 'FIRENZE': [43.7696, 11.2558], 'BARI': [41.1171, 16.8719],
            'CATANIA': [37.5079, 15.0830], 'VENEZIA': [45.4408, 12.3155], 'VERONA': [45.4384, 10.9916],
            'MESSINA': [38.1938, 15.5540], 'PADOVA': [45.4064, 11.8768], 'TRIESTE': [45.6495, 13.7768],
            'BRESCIA': [45.5416, 10.2118], 'PARMA': [44.8015, 10.3279], 'MODENA': [44.6471, 10.9252],
            'REGGIO CALABRIA': [38.1113, 15.6471], 'REGGIO EMILIA': [44.6989, 10.6297],
            'PERUGIA': [43.1107, 12.3908], 'LIVORNO': [43.5485, 10.3106], 'RAVENNA': [44.4184, 12.2035],
            'CAGLIARI': [39.2238, 9.1217], 'FOGGIA': [41.4621, 15.5444], 'RIMINI': [44.0678, 12.5695],
            'SALERNO': [40.6824, 14.7681], 'FERRARA': [44.8381, 11.6198], 'SASSARI': [40.7259, 8.5556],
            'LATINA': [41.4676, 12.9037], 'MONZA': [45.5845, 9.2744], 'BERGAMO': [45.6983, 9.6773],
            'TRENTO': [46.0748, 11.1217], 'VICENZA': [45.5455, 11.5354], 'TERNI': [42.5636, 12.6427],
            'NOVARA': [45.4465, 8.6220], 'PIACENZA': [45.0526, 9.6930], 'ANCONA': [43.6158, 13.5189],
            'UDINE': [46.0711, 13.2346], 'BOLZANO': [46.4983, 11.3548], 'LECCE': [40.3516, 18.1718],
            'PISA': [43.7228, 10.4017], 'AREZZO': [43.4633, 11.8797], 'PESCARA': [42.4618, 14.2161],
            'ALESSANDRIA': [44.9131, 8.6151], 'PESARO': [43.9098, 12.9131], 'LA SPEZIA': [44.1025, 9.8240],
            'CATANZARO': [38.9098, 16.5877], 'POTENZA': [40.6404, 15.8056], 'CAMPOBASSO': [41.5610, 14.6687],
            "L'AQUILA": [42.3498, 13.3995], 'AOSTA': [45.7372, 7.3209], 'COMO': [45.8081, 9.0852],
            'VARESE': [45.8206, 8.8257], 'PAVIA': [45.1847, 9.1582], 'CREMONA': [45.1336, 10.0227],
            'MANTOVA': [45.1564, 10.7914], 'LECCO': [45.8566, 9.3977], 'LODI': [45.3097, 9.5010],
            'SONDRIO': [46.1699, 9.8715], 'VERBANIA': [45.9227, 8.5519], 'VERCELLI': [45.3220, 8.4186],
            'ASTI': [44.9007, 8.2069], 'BIELLA': [45.5628, 8.0583], 'CUNEO': [44.3844, 7.5427],
            'IMPERIA': [43.8896, 8.0386], 'SAVONA': [44.3091, 8.4772], 'BELLUNO': [46.1403, 12.2167],
            'ROVIGO': [45.0702, 11.7897], 'TREVISO': [45.6669, 12.2430], 'GORIZIA': [45.9415, 13.6220],
            'PORDENONE': [45.9576, 12.6603], 'FORLI': [44.2227, 12.0408], 'CESENA': [44.1391, 12.2464],
            'REGGIO NELL EMILIA': [44.6989, 10.6297], 'PRATO': [43.8777, 11.1020], 'LUCCA': [43.8430, 10.5057],
            'PISTOIA': [43.9303, 10.9078], 'MASSA': [44.0353, 10.1395], 'CARRARA': [44.0793, 10.0982],
            'SIENA': [43.3188, 11.3308], 'GROSSETO': [42.7635, 11.1124], 'VITERBO': [42.4168, 12.1080],
            'RIETI': [42.4037, 12.8579], 'FROSINONE': [41.6399, 13.3428], 'ISERNIA': [41.5935, 14.2330],
            'BENEVENTO': [41.1297, 14.7826], 'AVELLINO': [40.9146, 14.7906], 'CASERTA': [41.0742, 14.3322],
            'TARANTO': [40.4644, 17.2470], 'BRINDISI': [40.6327, 17.9419], 'COSENZA': [39.3088, 16.2505],
            'CROTONE': [39.0851, 17.1175], 'VIBO VALENTIA': [38.6759, 16.1001], 'TRAPANI': [38.0174, 12.5140],
            'AGRIGENTO': [37.3111, 13.5766], 'CALTANISSETTA': [37.4901, 14.0629], 'ENNA': [37.5676, 14.2795],
            'RAGUSA': [36.9282, 14.7322], 'SIRACUSA': [37.0755, 15.2866], 'NUORO': [40.3210, 9.3313],
            'ORISTANO': [39.9062, 8.5896]
        }

        consip_by_city = filtered_consip.groupby('buyer_locality', observed=True).agg({
            'CIG': 'count',
            'award_amount': 'sum',
            'sconto': 'mean',
            'TipoAccordo': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/D'
        }).reset_index()
        consip_by_city.columns = ['citta', 'num_gare', 'valore', 'sconto_medio', 'tipo_principale']

        # Add coordinates
        consip_by_city['lat'] = consip_by_city['citta'].str.upper().map(lambda x: city_coords.get(x, [None, None])[0])
        consip_by_city['lng'] = consip_by_city['citta'].str.upper().map(lambda x: city_coords.get(x, [None, None])[1])
        consip_by_city = consip_by_city.dropna(subset=['lat', 'lng'])

        col1, col2 = st.columns([2, 1])

        with col1:
            if len(consip_by_city) > 0:
                # Color by tipo accordo
                fig = px.scatter_map(
                    consip_by_city,
                    lat='lat',
                    lon='lng',
                    size='valore',
                    color='tipo_principale',
                    hover_name='citta',
                    hover_data={'num_gare': True, 'valore': ':.2s', 'sconto_medio': ':.1f'},
                    size_max=40,
                    zoom=5,
                    center={'lat': 42.0, 'lon': 12.5},
                    title=f'Distribuzione CONSIP - {tipo_sel if tipo_sel != "Tutti" else "Tutti i tipi"}'
                )
                fig.update_layout(height=550, margin={"r":0,"t":30,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nessun dato CONSIP con coordinate disponibili per i filtri selezionati")

        with col2:
            st.markdown("#### 📊 Riepilogo per Tipo")
            tipo_summary = filtered_consip.groupby('TipoAccordo', observed=True).agg({
                'CIG': 'count',
                'award_amount': 'sum'
            }).reset_index()
            tipo_summary.columns = ['Tipo', 'N. Gare', 'Valore (€)']
            tipo_summary['Valore (€)'] = tipo_summary['Valore (€)'].apply(lambda x: f'€{x/1e6:.0f}M')
            show_dataframe(tipo_summary, use_container_width=True, hide_index=True)

            st.markdown("#### 🏙️ Top 10 Città")
            top_cities = consip_by_city.nlargest(10, 'valore')[['citta', 'num_gare', 'valore']]
            top_cities['valore'] = top_cities['valore'].apply(lambda x: f'€{x/1e6:.0f}M')
            top_cities.columns = ['Città', 'Gare', 'Valore']
            show_dataframe(top_cities, use_container_width=True, hide_index=True)

        # Timeline
        st.markdown("---")
        st.markdown("#### 📅 Timeline Contratti CONSIP")

        timeline = filtered_consip.groupby(['anno', 'TipoAccordo'], observed=True).agg({
            'CIG': 'count',
            'award_amount': 'sum'
        }).reset_index()
        timeline.columns = ['Anno', 'Tipo', 'N. Gare', 'Valore']
        timeline = timeline[timeline['Anno'].between(2015, 2025)]

        fig = px.bar(
            timeline,
            x='Anno',
            y='Valore',
            color='Tipo',
            barmode='stack',
            labels={'Valore': 'Valore (€)', 'Anno': 'Anno'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("---")
        st.markdown("#### 📋 Dettaglio Contratti CONSIP")

        display_consip = filtered_consip[['DataAggiudicazione', 'Comune', 'Regione', 'TipoAccordo', 'Edizione', 'OggettoGara', 'award_amount', 'sconto', 'Aggiudicatario']].copy()
        display_consip['DataAggiudicazione'] = pd.to_datetime(display_consip['DataAggiudicazione'], errors='coerce').dt.strftime('%Y-%m-%d')
        display_consip['award_amount'] = display_consip['award_amount'].apply(lambda x: f'€{x:,.0f}'.replace(',', '.') if pd.notna(x) else '-')
        display_consip['sconto'] = display_consip['sconto'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-')
        display_consip.columns = ['Data', 'Città', 'Regione', 'Tipo', 'Edizione', 'Oggetto', 'Valore', 'Sconto', 'Aggiudicatario']
        display_consip = display_consip.sort_values('Data', ascending=False)

        show_dataframe(display_consip.head(100), use_container_width=True, height=400)

        # Download
        st.download_button(
            label="📥 Scarica tutti i contratti CONSIP",
            data=filtered_consip.to_csv(index=False).encode('utf-8'),
            file_name=f'consip_{tipo_sel.lower()}_{anno_consip_sel}.csv',
            mime='text/csv'
        )
    else:
        st.warning("Nessun dato CONSIP disponibile nel dataset")

# ==================== TAB 9: RICERCA AGGIUDICATARIO ====================
if tab9:
  with tab9:
    st.subheader("🔎 Ricerca Aggiudicatario")

    # Mostra info sui filtri attivi
    filtri_attivi = []
    if fonte_sel: filtri_attivi.append(f"Fonte: {fonte_sel}")
    if anno_sel: filtri_attivi.append(f"Anno: {anno_sel}")
    if regione_sel: filtri_attivi.append(f"Regione: {regione_sel}")
    if categoria_sel: filtri_attivi.append(f"Categoria: {categoria_sel}")
    if procedura_sel: filtri_attivi.append(f"Procedura: {procedura_sel}")
    if tipo_appalto_sel: filtri_attivi.append(f"Tipo: {tipo_appalto_sel}")
    if sottocategoria_sel: filtri_attivi.append(f"Sottocategoria: {sottocategoria_sel}")

    if filtri_attivi:
        st.info(f"🔍 **Filtri attivi**: {', '.join(filtri_attivi)} | **{len(filtered_df):,}** gare filtrate".replace(",", "."))
    else:
        st.caption(f"📊 Mostrando tutti i {len(filtered_df):,} record".replace(",", "."))

    # Helper per colonne dinamiche
    def get_col_forn(df, candidates):
        for col in candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    supplier_col = get_col_forn(filtered_df, ['aggiudicatario', 'supplier_name', 'award_supplier_name'])
    amount_col_forn = get_col_forn(filtered_df, ['importo_aggiudicazione', 'award_amount', 'tender_amount'])
    buyer_col_forn = get_col_forn(filtered_df, ['ente_appaltante', 'buyer_name'])
    id_col_forn = get_col_forn(filtered_df, ['chiave', 'CIG', 'ocid', 'id'])

    if not supplier_col or not amount_col_forn:
        st.warning("Dati insufficienti per l'analisi fornitore")
    else:
        # Get unique suppliers sorted by total value - USA FILTERED_DF per rispettare filtri
        supplier_totals = filtered_df.groupby(supplier_col, observed=True)[amount_col_forn].sum().sort_values(ascending=False)
        suppliers_list = [s for s in supplier_totals.index.tolist() if pd.notna(s)]

        # Search box with text input
        col1, col2 = st.columns([3, 1])
        with col1:
            search_text = st.text_input("🔍 Cerca aggiudicatario (digita almeno 3 caratteri)", "", key="search_agg")

        # Filter suppliers based on search
        if len(search_text) >= 3:
            matching_suppliers = [s for s in suppliers_list if search_text.upper() in str(s).upper()][:50]
        else:
            matching_suppliers = suppliers_list[:100]  # Top 100 by value

        with col2:
            st.caption(f"{len(matching_suppliers)} risultati")

        # MULTISELECT per selezionare più aggiudicatari manualmente
        # La key cambia quando cambiano i filtri, così si resetta la selezione
        aggiudicatari_sel = st.multiselect(
            "Seleziona uno o più aggiudicatari da aggregare",
            options=matching_suppliers,
            format_func=lambda x: f"{x} (€{supplier_totals.get(x, 0)/1e6:.1f}M)" if pd.notna(supplier_totals.get(x, 0)) else str(x),
            help="Puoi selezionare più nomi per aggregare i dati (es. varianti dello stesso soggetto). La selezione si resetta quando cambi i filtri.",
            key=f"multisel_agg_{filter_key}"
        )

        if aggiudicatari_sel:
            # Mostra titolo con numero di aggiudicatari selezionati
            if len(aggiudicatari_sel) == 1:
                st.markdown(f"### 🏢 {aggiudicatari_sel[0]}")
            else:
                st.markdown(f"### 🏢 {len(aggiudicatari_sel)} aggiudicatari selezionati")
                with st.expander("📋 Lista aggiudicatari selezionati", expanded=False):
                    for agg in aggiudicatari_sel:
                        val = supplier_totals.get(agg, 0)
                        st.write(f"• {agg} (€{val/1e6:.1f}M)" if pd.notna(val) else f"• {agg}")

            # Filtra dati per tutti gli aggiudicatari selezionati - USA FILTERED_DF
            supplier_df = filtered_df[filtered_df[supplier_col].isin(aggiudicatari_sel)].copy()

            # Calcola totale aggregato
            total_aggregato = supplier_df[amount_col_forn].sum() if amount_col_forn else 0
            st.info(f"📊 **Totale aggregato**: {len(supplier_df):,} gare, €{total_aggregato/1e6:.1f}M".replace(",", "."))

            # KPIs - gestisci NaN
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🏆 Gare Vinte", f"{len(supplier_df):,}".replace(",", "."))
            col2.metric("💰 Valore Totale", f"€{supplier_df[amount_col_forn].sum()/1e6:.1f}M" if amount_col_forn else "N/D")
            sconto_medio = supplier_df['sconto'].dropna().mean() if 'sconto' in supplier_df.columns else np.nan
            col3.metric("📉 Sconto Medio", f"{sconto_medio:.1f}%" if pd.notna(sconto_medio) else "N/D")
            col4.metric("🏛️ Enti Serviti", f"{supplier_df[buyer_col_forn].nunique()}" if buyer_col_forn and buyer_col_forn in supplier_df.columns else "N/D")
            city_col = get_col_forn(supplier_df, ['citta', 'buyer_locality', 'comune'])
            col5.metric("📍 Città", f"{supplier_df[city_col].nunique()}" if city_col else "N/D")

            # Additional KPIs
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("📅 Prima Gara", f"{int(supplier_df['anno'].min())}" if supplier_df['anno'].notna().any() else "-")
            col2.metric("📅 Ultima Gara", f"{int(supplier_df['anno'].max())}" if supplier_df['anno'].notna().any() else "-")
            col3.metric("💵 Valore Medio", f"€{supplier_df[amount_col_forn].mean()/1e3:.0f}K" if amount_col_forn else "N/D")
            col4.metric("💵 Gara Max", f"€{supplier_df[amount_col_forn].max()/1e6:.1f}M" if amount_col_forn else "N/D")
            consip_count = len(supplier_df[supplier_df['TipoAccordo'].notna()]) if 'TipoAccordo' in supplier_df.columns else 0
            col5.metric("🏛️ Gare CONSIP", f"{consip_count}")

            st.markdown("---")

            # Charts row
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📈 Trend Annuale")
                agg_dict = {'sconto': 'mean'}
                if id_col_forn and id_col_forn in supplier_df.columns:
                    agg_dict[id_col_forn] = 'count'
                if amount_col_forn and amount_col_forn in supplier_df.columns:
                    agg_dict[amount_col_forn] = 'sum'
                yearly = supplier_df.groupby('anno', observed=True).agg(agg_dict).reset_index()
                yearly.columns = ['Anno'] + ['N. Gare' if c == id_col_forn else ('Valore' if c == amount_col_forn else 'Sconto Medio') for c in agg_dict.keys()]
                yearly = yearly[yearly['Anno'].between(2015, 2025)]
                yearly['Anno'] = yearly['Anno'].astype(int)  # Anni interi

                if 'Valore' in yearly.columns and 'N. Gare' in yearly.columns:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Bar(x=yearly['Anno'], y=yearly['Valore'], name='Valore (€)', marker_color=CGL_GREEN),
                        secondary_y=False
                    )
                    fig.add_trace(
                        go.Scatter(x=yearly['Anno'], y=yearly['N. Gare'], name='N. Gare', line=dict(color=CGL_BLUE, width=3)),
                        secondary_y=True
                    )
                    fig.update_yaxes(title_text="Valore (€)", secondary_y=False)
                    fig.update_yaxes(title_text="Numero Gare", secondary_y=True)
                    fig.update_xaxes(dtick=1, tickformat='d')  # Tick ogni anno, formato intero
                    fig.update_layout(height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📦 Per Categoria")
                cat_col = get_col_forn(supplier_df, ['_categoria', 'categoria', 'category'])
                if cat_col and id_col_forn and amount_col_forn:
                    cat_supplier = supplier_df.groupby(cat_col, observed=True).agg({
                        id_col_forn: 'count',
                        amount_col_forn: 'sum'
                    }).reset_index()
                    cat_supplier.columns = ['Categoria', 'N. Gare', 'Valore']
                    cat_supplier = cat_supplier.sort_values('Valore', ascending=True)

                    fig = px.bar(
                        cat_supplier,
                        x='Valore',
                        y='Categoria',
                        orientation='h',
                        color='N. Gare',
                        color_continuous_scale='Blues',
                        text=cat_supplier['Valore'].apply(lambda x: f'€{x/1e6:.1f}M' if x > 1e6 else f'€{x/1e3:.0f}K')
                    )
                    fig.update_layout(height=350)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dati categoria non disponibili")

            # Geographic distribution
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🗺️ Distribuzione Geografica")
                city_col_geo = get_col_forn(supplier_df, ['citta', 'buyer_locality', 'comune'])
                if city_col_geo and id_col_forn and amount_col_forn:
                    geo_supplier = supplier_df.groupby(city_col_geo, observed=True).agg({
                        id_col_forn: 'count',
                        amount_col_forn: 'sum'
                    }).reset_index()
                    geo_supplier.columns = ['Città', 'N. Gare', 'Valore']
                    geo_supplier = geo_supplier.sort_values('Valore', ascending=False).head(15)

                    fig = px.bar(
                        geo_supplier,
                        x='Valore',
                        y='Città',
                        orientation='h',
                        color='N. Gare',
                        color_continuous_scale='Viridis',
                        text=geo_supplier['Valore'].apply(lambda x: f'€{x/1e6:.1f}M' if x > 1e6 else f'€{x/1e3:.0f}K')
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dati geografici non disponibili")

            with col2:
                st.markdown("#### 🏛️ Top Enti Appaltanti")
                if buyer_col_forn and id_col_forn and amount_col_forn:
                    buyer_supplier = supplier_df.groupby(buyer_col_forn, observed=True).agg({
                        id_col_forn: 'count',
                        amount_col_forn: 'sum'
                    }).reset_index()
                    buyer_supplier.columns = ['Ente', 'N. Gare', 'Valore']
                    buyer_supplier = buyer_supplier.sort_values('Valore', ascending=False).head(15)

                    fig = px.bar(
                        buyer_supplier,
                        x='Valore',
                        y='Ente',
                        orientation='h',
                        color='N. Gare',
                        color_continuous_scale='Oranges',
                        text=buyer_supplier['Valore'].apply(lambda x: f'€{x/1e6:.1f}M' if x > 1e6 else f'€{x/1e3:.0f}K')
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dati enti non disponibili")

            # Sconto distribution
            st.markdown("---")
            st.markdown("#### 📊 Distribuzione Sconti Applicati")

            if 'sconto' in supplier_df.columns and supplier_df['sconto'].notna().any():
                col1, col2 = st.columns(2)
                with col1:
                    # Filtra sconti validi: escludi 0, null e valori > 100
                    valid_sconto = supplier_df[(supplier_df['sconto'] > 0) & (supplier_df['sconto'] <= 100)]
                    if len(valid_sconto) > 0:
                        fig = px.histogram(
                            valid_sconto,
                            x='sconto',
                            nbins=30,
                            color_discrete_sequence=[CGL_BLUE],
                            labels={'sconto': 'Sconto %'}
                        )
                        sconto_mean = valid_sconto['sconto'].mean()
                        if pd.notna(sconto_mean):
                            fig.add_vline(x=sconto_mean, line_dash="dash", line_color="red",
                                          annotation_text=f"Media: {sconto_mean:.1f}%")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"ℹ️ Basato su {len(valid_sconto)} gare con sconto > 0%")
                    else:
                        st.info("Dati sconto non sufficienti (sconto > 0%)")

                with col2:
                    # Sconto by category
                    cat_col_sc = get_col_forn(supplier_df, ['_categoria', 'categoria', 'category'])
                    if cat_col_sc:
                        sconto_cat = supplier_df.groupby(cat_col_sc, observed=True)['sconto'].mean().sort_values(ascending=True).reset_index()
                        sconto_cat.columns = ['Categoria', 'Sconto Medio']

                        fig = px.bar(
                            sconto_cat,
                            x='Sconto Medio',
                            y='Categoria',
                            orientation='h',
                            color='Sconto Medio',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dati sconto non disponibili")

            # Detailed table
            st.markdown("---")
            st.markdown("#### 📋 Storico Completo Gare")

            # Costruisci lista colonne disponibili
            display_cols = []
            col_mapping = {}
            date_col = get_col_forn(supplier_df, ['data_aggiudicazione', 'award_date'])
            if date_col:
                display_cols.append(date_col)
                col_mapping[date_col] = 'Data'
            city_col_d = get_col_forn(supplier_df, ['citta', 'buyer_locality', 'comune'])
            if city_col_d:
                display_cols.append(city_col_d)
                col_mapping[city_col_d] = 'Città'
            if buyer_col_forn:
                display_cols.append(buyer_col_forn)
                col_mapping[buyer_col_forn] = 'Ente Appaltante'
            cat_col_d = get_col_forn(supplier_df, ['_categoria', 'categoria', 'category'])
            if cat_col_d:
                display_cols.append(cat_col_d)
                col_mapping[cat_col_d] = 'Categoria'
            title_col = get_col_forn(supplier_df, ['oggetto', 'tender_title', 'description'])
            if title_col:
                display_cols.append(title_col)
                col_mapping[title_col] = 'Oggetto'
            if amount_col_forn:
                display_cols.append(amount_col_forn)
                col_mapping[amount_col_forn] = 'Valore'
            if 'sconto' in supplier_df.columns:
                display_cols.append('sconto')
                col_mapping['sconto'] = 'Sconto'

            if display_cols:
                display_supplier = supplier_df[display_cols].copy()
                if date_col and date_col in display_supplier.columns:
                    display_supplier[date_col] = pd.to_datetime(display_supplier[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
                if amount_col_forn and amount_col_forn in display_supplier.columns:
                    display_supplier[amount_col_forn] = display_supplier[amount_col_forn].apply(lambda x: f'€{x:,.0f}'.replace(',', '.') if pd.notna(x) else '-')
                if 'sconto' in display_supplier.columns:
                    display_supplier['sconto'] = display_supplier['sconto'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-')
                display_supplier = display_supplier.rename(columns=col_mapping)
                if 'Data' in display_supplier.columns:
                    display_supplier = display_supplier.sort_values('Data', ascending=False)

                # Pagination
                page_size = 50
                total_pages = (len(display_supplier) - 1) // page_size + 1
                page = st.number_input('Pagina', min_value=1, max_value=max(1, total_pages), value=1, key='supplier_page')

                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size

                show_dataframe(display_supplier.iloc[start_idx:end_idx], use_container_width=True, height=400)
                st.caption(f"Mostrando {start_idx+1}-{min(end_idx, len(display_supplier))} di {len(display_supplier)} gare")

            # Export
            export_name = "_".join([a[:15] for a in aggiudicatari_sel[:3]]).lower().replace(" ", "_")
            st.download_button(
                label="📥 Scarica storico completo CSV",
                data=supplier_df.to_csv(index=False).encode('utf-8'),
                file_name=f'gare_{export_name}.csv',
                mime='text/csv'
            )

        else:
            # Show top suppliers - USA FILTERED_DF per rispettare i filtri!
            st.markdown("#### 🏆 Top 50 Aggiudicatari per Valore Totale")

            # Costruisci agg_dict dinamico
            agg_dict_top = {}
            if id_col_forn:
                agg_dict_top[id_col_forn] = 'count'
            if amount_col_forn:
                agg_dict_top[amount_col_forn] = 'sum'
            if 'sconto' in filtered_df.columns:
                agg_dict_top['sconto'] = 'mean'
            if buyer_col_forn:
                agg_dict_top[buyer_col_forn] = 'nunique'
            if 'anno' in filtered_df.columns:
                agg_dict_top['anno'] = ['min', 'max']

            if agg_dict_top:
                top_suppliers_summary = filtered_df.groupby(supplier_col, observed=True).agg(agg_dict_top).reset_index()
                # Rinomina colonne
                new_cols = ['Aggiudicatario']
                for col in agg_dict_top.keys():
                    if col == id_col_forn:
                        new_cols.append('N. Gare')
                    elif col == amount_col_forn:
                        new_cols.append('Valore (€)')
                    elif col == 'sconto':
                        new_cols.append('Sconto Medio %')
                    elif col == buyer_col_forn:
                        new_cols.append('N. Enti')
                    elif col == 'anno':
                        new_cols.extend(['Prima Gara', 'Ultima Gara'])
                top_suppliers_summary.columns = new_cols[:len(top_suppliers_summary.columns)]
                top_suppliers_summary = top_suppliers_summary.sort_values('Valore (€)' if 'Valore (€)' in top_suppliers_summary.columns else 'N. Gare', ascending=False).head(50)

                # Tronca nomi troppo lunghi per visualizzazione
                top_suppliers_summary['Aggiudicatario_display'] = top_suppliers_summary['Aggiudicatario'].apply(
                    lambda x: x[:35] + '...' if isinstance(x, str) and len(x) > 38 else x
                )

                fig = px.bar(
                    top_suppliers_summary.head(20),
                    x='Valore (€)' if 'Valore (€)' in top_suppliers_summary.columns else 'N. Gare',
                    y='Aggiudicatario_display',
                    orientation='h',
                    color='N. Gare' if 'N. Gare' in top_suppliers_summary.columns else None,
                    color_continuous_scale='Viridis',
                    text=top_suppliers_summary.head(20)['Valore (€)'].apply(lambda x: f'€{x/1e9:.1f}B' if x > 1e9 else f'€{x/1e6:.0f}M') if 'Valore (€)' in top_suppliers_summary.columns else None,
                    labels={'Aggiudicatario_display': 'Aggiudicatario'}
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending', 'title': 'Aggiudicatario'})
                fig.update_traces(textposition='outside')
                render_chart_with_save(fig, "Top 20 Aggiudicatari (Da ricerca)", "Classifica top 20 aggiudicatari per valore", "top50_aggiudicatari")

                # Table
                display_top = top_suppliers_summary.copy()
                if 'Valore (€)' in display_top.columns:
                    display_top['Valore (€)'] = display_top['Valore (€)'].apply(lambda x: f'€{x/1e6:.0f}M')
                if 'Sconto Medio %' in display_top.columns:
                    display_top['Sconto Medio %'] = display_top['Sconto Medio %'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-')
                show_dataframe(display_top, use_container_width=True, height=400)

# ==================== TAB 10: ANALISI MERCATO ====================
if tab10:
  with tab10:
    st.subheader("📉 Analisi di Mercato Avanzata")

    # Concentrazione mercato
    st.markdown("### 🎯 Concentrazione del Mercato")

    # Identifica colonne dinamicamente
    supplier_col_mkt = get_col(filtered_df, ['supplier_name', 'aggiudicatario', 'award_supplier_name'])
    amount_col_mkt = get_col(filtered_df, ['award_amount', 'importo_aggiudicazione'])
    cat_col_mkt = get_col(filtered_df, ['_categoria', 'categoria', 'category'])

    if supplier_col_mkt and amount_col_mkt and cat_col_mkt:
        # Calcola dati una volta sola
        hhi_by_cat = []
        cr4_by_cat = []
        for cat in filtered_df[cat_col_mkt].dropna().unique():
            cat_data = filtered_df[filtered_df[cat_col_mkt] == cat]
            supplier_shares = cat_data.groupby(supplier_col_mkt, observed=True)[amount_col_mkt].sum()
            total = supplier_shares.sum()
            if total > 0:
                hhi = ((supplier_shares / total * 100) ** 2).sum()
                hhi_by_cat.append({'Categoria': cat, 'HHI': hhi, 'Valore': total})
                cr4 = supplier_shares.sort_values(ascending=False).head(4).sum() / total * 100
                cr4_by_cat.append({'Categoria': cat, 'CR4': cr4})

        # HHI per categoria (full width)
        st.markdown("#### Indice HHI per Categoria")
        if hhi_by_cat:
            hhi_df = pd.DataFrame(hhi_by_cat).sort_values('HHI', ascending=False)
            fig = px.bar(
                hhi_df,
                x='HHI',
                y='Categoria',
                orientation='h',
                color='HHI',
                color_continuous_scale='RdYlGn_r',
                title='HHI: <1500 competitivo, >2500 concentrato'
            )
            fig.add_vline(x=1500, line_dash="dash", line_color="green")
            fig.add_vline(x=2500, line_dash="dash", line_color="red")
            fig.update_layout(
                height=max(400, len(hhi_df) * 30),
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=250)
            )
            render_chart_with_save(fig, "Indice HHI per Categoria", "Concentrazione mercato per categoria (HHI)", "hhi_categoria")

        st.markdown("---")

        # CR4 per categoria (full width)
        st.markdown("#### CR4 - Quota Top 4 Fornitori")
        if cr4_by_cat:
            cr4_df = pd.DataFrame(cr4_by_cat).sort_values('CR4', ascending=False)
            fig = px.bar(
                cr4_df,
                x='CR4',
                y='Categoria',
                orientation='h',
                color='CR4',
                color_continuous_scale='RdYlGn_r',
                title='% mercato controllato dai top 4'
            )
            fig.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="60%")
            fig.update_layout(
                height=max(400, len(cr4_df) * 30),
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=250)
            )
            render_chart_with_save(fig, "CR4 per Categoria", "Quota mercato dei top 4 fornitori", "cr4_categoria")

        st.markdown("---")

        # N. Operatori per categoria (full width)
        st.markdown("#### N. Operatori per Categoria")
        operators_by_cat = filtered_df.groupby(cat_col_mkt, observed=True)[supplier_col_mkt].nunique().reset_index()
        operators_by_cat.columns = ['Categoria', 'N. Operatori']
        operators_by_cat = operators_by_cat.sort_values('N. Operatori', ascending=True)

        fig = px.bar(
            operators_by_cat,
            x='N. Operatori',
            y='Categoria',
            orientation='h',
            color='N. Operatori',
            color_continuous_scale='Blues',
            text='N. Operatori'
        )
        fig.update_layout(
            height=max(400, len(operators_by_cat) * 30),
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=250)
        )
        fig.update_traces(textposition='outside')
        render_chart_with_save(fig, "Operatori per Categoria", "Numero operatori unici per categoria", "operatori_categoria")
    else:
        st.warning("Dati insufficienti per l'analisi di mercato con i filtri selezionati")

    # Analisi competitività
    st.markdown("---")
    st.markdown("### 🏃 Analisi Competitività")

    # Helper function for dynamic column detection in Tab 10
    def get_col_t10(df, candidates):
        for col in candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    # Define dynamic columns for Tab 10 sections
    id_col_t10 = get_col_t10(filtered_df, ['chiave', 'CIG', 'ocid', 'id'])
    amount_col_t10 = get_col_t10(filtered_df, ['importo_aggiudicazione', 'award_amount', 'tender_amount'])
    supplier_col_t10 = get_col_t10(filtered_df, ['aggiudicatario', 'supplier_name', 'award_supplier_name'])
    buyer_col_t10 = get_col_t10(filtered_df, ['ente_appaltante', 'buyer_name'])
    cat_col_t10 = get_col_t10(filtered_df, ['_categoria', 'categoria', 'category'])

    col1, col2 = st.columns(2)

    with col1:
        # Usa offerte_ricevute se disponibile
        partecipanti_col = 'offerte_ricevute' if 'offerte_ricevute' in filtered_df.columns else 'parties_count'

        if partecipanti_col in filtered_df.columns and filtered_df[partecipanti_col].notna().sum() > 50:
            # Converti a numerico
            filtered_df[partecipanti_col] = pd.to_numeric(filtered_df[partecipanti_col], errors='coerce')

            # Verifica se abbiamo abbastanza dati sconto
            has_sconto = 'sconto' in filtered_df.columns and filtered_df['sconto'].notna().sum() > 100
            valid_with_sconto = filtered_df[filtered_df[partecipanti_col].between(1, 20) & filtered_df['sconto'].notna()] if has_sconto else pd.DataFrame()

            if len(valid_with_sconto) > 50 and id_col_t10:
                # ANALISI CON SCONTO
                st.markdown("#### Sconto vs N. Partecipanti")
                comp_analysis = valid_with_sconto.groupby(partecipanti_col, observed=True).agg({
                    'sconto': 'mean',
                    id_col_t10: 'count'
                }).reset_index()
                comp_analysis.columns = ['N. Partecipanti', 'Sconto Medio', 'N. Gare']
                comp_analysis = comp_analysis[comp_analysis['N. Gare'] >= 5]

                if len(comp_analysis) > 2:
                    fig = px.scatter(
                        comp_analysis,
                        x='N. Partecipanti',
                        y='Sconto Medio',
                        size='N. Gare',
                        color='Sconto Medio',
                        color_continuous_scale='RdYlGn',
                        title='Più partecipanti = più sconto?'
                    )
                    z = np.polyfit(comp_analysis['N. Partecipanti'], comp_analysis['Sconto Medio'], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=comp_analysis['N. Partecipanti'],
                        y=p(comp_analysis['N. Partecipanti']),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    corr = valid_with_sconto[[partecipanti_col, 'sconto']].corr().iloc[0, 1]
                    st.metric("📊 Correlazione Partecipanti-Sconto", f"{corr:.3f}",
                              help="Positivo = più partecipanti, più sconto")
                else:
                    st.info("Dati sconto insufficienti per l'analisi")
            elif id_col_t10:
                # ANALISI ALTERNATIVA: Distribuzione gare per N. Partecipanti
                st.markdown("#### Distribuzione Gare per N. Partecipanti")
                valid_data = filtered_df[filtered_df[partecipanti_col].between(1, 20)]

                if len(valid_data) > 50:
                    agg_dict_comp = {id_col_t10: 'count'}
                    if amount_col_t10:
                        agg_dict_comp[amount_col_t10] = 'mean'

                    comp_analysis = valid_data.groupby(partecipanti_col, observed=True).agg(agg_dict_comp).reset_index()
                    col_names_comp = ['N. Partecipanti', 'N. Gare']
                    if amount_col_t10:
                        col_names_comp.append('Importo Medio')
                    comp_analysis.columns = col_names_comp
                    comp_analysis = comp_analysis[comp_analysis['N. Gare'] >= 10].sort_values('N. Partecipanti')

                    if len(comp_analysis) > 2:
                        fig = px.bar(
                            comp_analysis,
                            x='N. Partecipanti',
                            y='N. Gare',
                            color='Importo Medio' if 'Importo Medio' in comp_analysis.columns else None,
                            color_continuous_scale='Viridis',
                            title='Numero gare per livello di competizione',
                            text='N. Gare'
                        )
                        fig.update_traces(textposition='outside')
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)

                        # KPIs
                        col_a, col_b = st.columns(2)
                        with col_a:
                            most_common = comp_analysis.loc[comp_analysis['N. Gare'].idxmax(), 'N. Partecipanti']
                            st.metric("🏆 N. Partecipanti più comune", f"{int(most_common)}")
                        with col_b:
                            avg_part = valid_data[partecipanti_col].mean()
                            st.metric("📊 Media partecipanti", f"{avg_part:.1f}")

                        st.caption(f"ℹ️ Campo 'sconto' disponibile solo per {filtered_df['sconto'].notna().sum()} gare")
                    else:
                        st.info("Dati insufficienti per l'analisi")
                else:
                    st.info("Dati insufficienti per l'analisi")
            else:
                st.info("Colonna ID non trovata per l'analisi")
        else:
            st.info("Campo 'offerte_ricevute' non disponibile o insufficiente")

    with col2:
        st.markdown("#### Sconto vs Valore Gara")
        # Bin by value ranges - usa colonna dinamica
        if amount_col_t10 and amount_col_t10 in filtered_df.columns:
            filtered_df['value_bin'] = pd.cut(
                filtered_df[amount_col_t10],
                bins=[0, 50000, 150000, 500000, 2000000, 10000000, float('inf')],
                labels=['<50K', '50-150K', '150-500K', '500K-2M', '2-10M', '>10M']
            )

            agg_dict_val = {'sconto': 'mean'}
            if id_col_t10:
                agg_dict_val[id_col_t10] = 'count'

            value_analysis = filtered_df.groupby('value_bin', observed=True).agg(agg_dict_val).reset_index()
            col_names = ['Fascia Valore', 'Sconto Medio']
            if id_col_t10:
                col_names.append('N. Gare')
            value_analysis.columns = col_names

            if len(value_analysis) > 0 and value_analysis['Sconto Medio'].notna().any():
                fig = px.bar(
                    value_analysis,
                    x='Fascia Valore',
                    y='Sconto Medio',
                    color='Sconto Medio',
                    color_continuous_scale='RdYlGn',
                    text=value_analysis['Sconto Medio'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-'),
                    title='Sconto medio per fascia di valore'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dati insufficienti per l'analisi per valore")
        else:
            st.info("Campo importo non disponibile")

    # Stagionalità
    st.markdown("---")
    st.markdown("### 📅 Analisi Stagionalità")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribuzione Mensile")
        # Filtra solo i record con mese valido
        valid_monthly = filtered_df[filtered_df['mese'].notna() & filtered_df['mese'].between(1, 12)]

        if len(valid_monthly) > 50 and id_col_t10:
            # Build agg dict dynamically
            agg_dict_month = {id_col_t10: 'count', 'sconto': 'mean'}
            if amount_col_t10:
                agg_dict_month[amount_col_t10] = 'sum'

            monthly_dist = valid_monthly.groupby('mese', observed=True).agg(agg_dict_month).reset_index()
            # Rename columns
            new_cols_month = ['Mese', 'N. Gare', 'Sconto Medio']
            if amount_col_t10:
                new_cols_month.insert(2, 'Valore')
            monthly_dist.columns = new_cols_month[:len(monthly_dist.columns)]

            month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
            monthly_dist['Mese Nome'] = monthly_dist['Mese'].apply(lambda x: month_names[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else 'N/D')
            monthly_dist = monthly_dist.sort_values('Mese')

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=monthly_dist['Mese Nome'], y=monthly_dist['N. Gare'], name='N. Gare', marker_color=CGL_GREEN),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=monthly_dist['Mese Nome'], y=monthly_dist['Sconto Medio'], name='Sconto %',
                           line=dict(color=CGL_BLUE, width=3)),
                secondary_y=True
            )
            fig.update_layout(height=350, title='Gare e Sconti per Mese')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dati mensili insufficienti")

    with col2:
        st.markdown("#### Heatmap Mese x Anno")
        # Usa colonna dinamica per il conteggio
        count_col_heat = id_col_t10 if id_col_t10 else (filtered_df.columns[0] if len(filtered_df.columns) > 0 else None)

        if count_col_heat and 'anno' in filtered_df.columns and 'mese' in filtered_df.columns:
            valid_heatmap = filtered_df[filtered_df['anno'].notna() & filtered_df['mese'].notna() & filtered_df['mese'].between(1, 12)]
            if len(valid_heatmap) > 0:
                pivot_monthly = valid_heatmap.groupby(['anno', 'mese'], observed=True)[count_col_heat].count().reset_index()
                pivot_monthly.columns = ['Anno', 'Mese', 'N. Gare']
                pivot_monthly = pivot_monthly[(pivot_monthly['Anno'].between(2018, 2025)) & (pivot_monthly['Mese'].between(1, 12))]

                if len(pivot_monthly) > 0:
                    pivot_table = pivot_monthly.pivot(index='Anno', columns='Mese', values='N. Gare').fillna(0)

                    fig = px.imshow(
                        pivot_table,
                        labels={'x': 'Mese', 'y': 'Anno', 'color': 'N. Gare'},
                        color_continuous_scale='Blues',
                        title='Volume gare per periodo'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dati heatmap insufficienti")
            else:
                st.info("Dati heatmap insufficienti")
        else:
            st.info("Colonne anno/mese non disponibili")

    # Anomalie e outlier
    st.markdown("---")
    st.markdown("### 🔍 Rilevamento Anomalie")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Gare con Sconto Anomalo")
        # Sconti molto alti (>80%) o molto bassi (<5%) - escludi 0 e null
        if 'sconto' in filtered_df.columns and filtered_df['sconto'].notna().any():
            valid_sconti_anom = filtered_df[(filtered_df['sconto'] > 0) & (filtered_df['sconto'] <= 100)]
            high_discount = valid_sconti_anom[valid_sconti_anom['sconto'] > 80]
            low_discount = valid_sconti_anom[valid_sconti_anom['sconto'] < 5]

            st.metric("⬆️ Sconto > 80%", f"{len(high_discount):,}".replace(",", "."),
                      help="Gare con sconto superiore all'80%")
            st.metric("⬇️ Sconto < 5%", f"{len(low_discount):,}".replace(",", "."),
                      help="Gare con sconto tra 0% e 5% (escluso 0)")

            # Distribution - escludi sconti = 0
            fig = px.histogram(
                valid_sconti_anom,
                x='sconto',
                nbins=100,
                title='Distribuzione Sconti (esclusi valori = 0)',
                color_discrete_sequence=[CGL_GREEN]
            )
            fig.add_vline(x=5, line_dash="dash", line_color="red")
            fig.add_vline(x=80, line_dash="dash", line_color="red")
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Campo sconto non disponibile")

    with col2:
        st.markdown("#### Gare di Valore Elevato")
        if amount_col_t10 and amount_col_t10 in filtered_df.columns:
            large_contracts = filtered_df[filtered_df[amount_col_t10] > 10000000]
            very_large = filtered_df[filtered_df[amount_col_t10] > 50000000]

            st.metric("💰 Gare > €10M", f"{len(large_contracts):,}".replace(",", "."))
            st.metric("💎 Gare > €50M", f"{len(very_large):,}".replace(",", "."))

            if len(large_contracts) > 0:
                # Build columns list dynamically
                cols_to_show = []
                col_labels = []
                if buyer_col_t10 and buyer_col_t10 in large_contracts.columns:
                    cols_to_show.append(buyer_col_t10)
                    col_labels.append('Ente')
                cols_to_show.append(amount_col_t10)
                col_labels.append('Valore')
                if cat_col_t10 and cat_col_t10 in large_contracts.columns:
                    cols_to_show.append(cat_col_t10)
                    col_labels.append('Categoria')

                if cols_to_show:
                    top_large = large_contracts.nlargest(5, amount_col_t10)[cols_to_show].copy()
                    top_large[amount_col_t10] = top_large[amount_col_t10].apply(lambda x: f'€{x/1e6:.0f}M')
                    top_large.columns = col_labels
                    show_dataframe(top_large, use_container_width=True, hide_index=True)
        else:
            st.info("Campo importo non disponibile")

    with col3:
        st.markdown("#### Fornitori Dominanti")
        # Fornitori con >30% del mercato in almeno una categoria
        if supplier_col_t10 and amount_col_t10 and cat_col_t10:
            dominant = []
            for cat in filtered_df[cat_col_t10].dropna().unique():
                cat_data = filtered_df[filtered_df[cat_col_t10] == cat]
                total = cat_data[amount_col_t10].sum()
                if total > 0:
                    top_supplier = cat_data.groupby(supplier_col_t10, observed=True)[amount_col_t10].sum().nlargest(1)
                    if len(top_supplier) > 0:
                        share = top_supplier.iloc[0] / total * 100
                        if share > 30:
                            dominant.append({
                                'Categoria': str(cat)[:25],
                                'Fornitore': str(top_supplier.index[0])[:30],
                                'Quota': f'{share:.0f}%'
                            })

            if dominant:
                show_dataframe(pd.DataFrame(dominant), use_container_width=True, hide_index=True)
            else:
                st.info("Nessun fornitore con quota >30% in una categoria")
        else:
            st.info("Dati insufficienti per l'analisi dominanti")

    # Efficienza procedure
    st.markdown("---")
    st.markdown("### ⚡ Efficienza delle Procedure")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sconto Medio per Tipo Procedura")
        # Cerca il campo procedura disponibile - includi 'procedura' che esiste nel dataset
        proc_col = None
        for col in ['procedura', 'procurement_method_details', 'procurement_method', 'TipoSceltaContraente']:
            if col in filtered_df.columns and filtered_df[col].notna().sum() > 10:
                proc_col = col
                break

        if proc_col:
            proc_df = filtered_df[filtered_df[proc_col].notna() & filtered_df['sconto'].notna()].copy()

            if len(proc_df) > 20:
                # Pulisci i nomi delle procedure
                def clean_proc_name(x):
                    if pd.isna(x):
                        return x
                    x = str(x)
                    if 'TITLE:' in x:
                        x = x.split('TITLE:')[-1].strip()
                    return x[:35] + '...' if len(x) > 35 else x

                proc_df['proc_clean'] = proc_df[proc_col].apply(clean_proc_name)

                # Build aggregation dict dynamically
                agg_dict_proc = {'sconto': 'mean'}
                if id_col_t10:
                    agg_dict_proc[id_col_t10] = 'count'
                if amount_col_t10:
                    agg_dict_proc[amount_col_t10] = 'sum'

                proc_analysis = proc_df.groupby('proc_clean', observed=True).agg(agg_dict_proc).reset_index()
                # Rename columns
                proc_cols = ['Procedura', 'Sconto Medio']
                if id_col_t10:
                    proc_cols.append('N. Gare')
                if amount_col_t10:
                    proc_cols.append('Valore')
                proc_analysis.columns = proc_cols[:len(proc_analysis.columns)]

                # Abbassa la soglia minima
                if 'N. Gare' in proc_analysis.columns:
                    proc_analysis = proc_analysis[proc_analysis['N. Gare'] > 5].sort_values('Sconto Medio', ascending=True)
                else:
                    proc_analysis = proc_analysis.sort_values('Sconto Medio', ascending=True)

                if len(proc_analysis) > 0:
                    fig = px.bar(
                        proc_analysis.tail(10),
                        x='Sconto Medio',
                        y='Procedura',
                        orientation='h',
                        color='Sconto Medio',
                        color_continuous_scale='RdYlGn',
                        text=proc_analysis.tail(10)['Sconto Medio'].apply(lambda x: f'{x:.1f}%')
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True, key="proc_sconto")
                else:
                    st.info("Dati insufficienti per l'analisi")
            else:
                st.info("Dati procedure insufficienti")
        else:
            st.info("Campo tipo procedura non disponibile")

    with col2:
        st.markdown("#### Performance per Regione")
        # Usa 'regione' invece di 'buyer_locality'
        region_col = 'regione' if 'regione' in filtered_df.columns else 'buyer_region'

        if region_col in filtered_df.columns and filtered_df[region_col].notna().sum() > 10:
            # Build aggregation dict dynamically
            agg_dict_reg = {'sconto': 'mean'}
            if id_col_t10:
                agg_dict_reg[id_col_t10] = 'count'
            if amount_col_t10:
                agg_dict_reg[amount_col_t10] = 'sum'

            regional_perf = filtered_df[filtered_df[region_col].notna() & filtered_df['sconto'].notna()].groupby(region_col, observed=True).agg(agg_dict_reg).reset_index()
            # Rename columns
            reg_cols = ['Regione', 'Sconto Medio']
            if id_col_t10:
                reg_cols.append('N. Gare')
            if amount_col_t10:
                reg_cols.append('Valore')
            regional_perf.columns = reg_cols[:len(regional_perf.columns)]

            # Abbassa soglia minima
            if 'N. Gare' in regional_perf.columns:
                regional_perf = regional_perf[regional_perf['N. Gare'] > 5].sort_values('Sconto Medio', ascending=False)
            else:
                regional_perf = regional_perf.sort_values('Sconto Medio', ascending=False)

            if len(regional_perf) > 0:
                hover_dict = {}
                if 'N. Gare' in regional_perf.columns:
                    hover_dict['N. Gare'] = True
                if 'Valore' in regional_perf.columns:
                    hover_dict['Valore'] = ':,.0f'

                fig = px.bar(
                    regional_perf.head(15),
                    x='Sconto Medio',
                    y='Regione',
                    orientation='h',
                    color='Sconto Medio',
                    color_continuous_scale='RdYlGn',
                    hover_data=hover_dict if hover_dict else None,
                    text=regional_perf.head(15)['Sconto Medio'].apply(lambda x: f'{x:.1f}%')
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="region_sconto")
            else:
                st.info("Dati insufficienti per l'analisi regionale")
        else:
            st.info("Dati regione non disponibili")

    # Summary stats
    st.markdown("---")
    st.markdown("### 📊 Riepilogo Statistico")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Valori**")
        if amount_col_t10 and amount_col_t10 in filtered_df.columns and filtered_df[amount_col_t10].notna().any():
            st.write(f"Media: €{filtered_df[amount_col_t10].mean()/1e3:.0f}K")
            st.write(f"Mediana: €{filtered_df[amount_col_t10].median()/1e3:.0f}K")
            st.write(f"Std: €{filtered_df[amount_col_t10].std()/1e6:.1f}M")
            st.write(f"Totale: €{filtered_df[amount_col_t10].sum()/1e9:.1f}B")
        else:
            st.write("Dati importo non disponibili")

    with col2:
        st.markdown("**Sconti**")
        if 'sconto' in filtered_df.columns and filtered_df['sconto'].notna().any():
            st.write(f"Media: {filtered_df['sconto'].mean():.1f}%")
            st.write(f"Mediana: {filtered_df['sconto'].median():.1f}%")
            st.write(f"Std: {filtered_df['sconto'].std():.1f}%")
            st.write(f"Range: {filtered_df['sconto'].min():.0f}%-{filtered_df['sconto'].max():.0f}%")
        else:
            st.write("Dati sconto non disponibili")

    with col3:
        st.markdown("**Volumi**")
        st.write(f"Gare totali: {len(filtered_df):,}".replace(",", "."))
        if supplier_col_t10 and supplier_col_t10 in filtered_df.columns:
            st.write(f"Fornitori: {filtered_df[supplier_col_t10].nunique():,}".replace(",", "."))
        else:
            st.write("Fornitori: N/D")
        buyer_col_stat = buyer_col_t10 if buyer_col_t10 else ('buyer_name' if 'buyer_name' in filtered_df.columns else 'ente_appaltante')
        locality_stat_col = 'comune' if 'comune' in filtered_df.columns else 'buyer_locality'
        st.write(f"Enti: {filtered_df[buyer_col_stat].nunique():,}".replace(",", ".") if buyer_col_stat and buyer_col_stat in filtered_df.columns else "Enti: N/D")
        st.write(f"Città: {filtered_df[locality_stat_col].nunique():,}".replace(",", ".") if locality_stat_col in filtered_df.columns else "Città: N/D")

    with col4:
        st.markdown("**Periodo**")
        if 'anno' in filtered_df.columns and filtered_df['anno'].notna().any():
            st.write(f"Dal: {int(filtered_df['anno'].min())}")
            st.write(f"Al: {int(filtered_df['anno'].max())}")
            st.write(f"Anni: {filtered_df['anno'].nunique()}")
        else:
            st.write("Dati anno non disponibili")
        if cat_col_t10 and cat_col_t10 in filtered_df.columns:
            st.write(f"Categorie: {filtered_df[cat_col_t10].nunique()}")
        else:
            st.write("Categorie: N/D")

# ==================== TAB 11: SCADENZE CONTRATTI ====================
if tab11:
  with tab11:
    st.header("📅 Scadenze Contratti")
    st.markdown("Analisi dei contratti in scadenza nei prossimi anni")

    # Carica dati CONSIP per scadenze
    consip_exp = load_consip_data()

    def _to_dt(series, fmt=None):
        if series is None:
            return pd.Series(dtype="datetime64[ns]")
        s = pd.to_datetime(series, format=fmt, errors='coerce')
        try:
            if getattr(s.dt, 'tz', None) is not None and s.dt.tz is not None:
                s = s.dt.tz_convert(None)
        except Exception:
            pass
        return s

    def _macro_area_from_regione(regione: str):
        if pd.isna(regione):
            return None
        r = str(regione).strip()
        macro_map = {
            "Piemonte": "Nord Ovest",
            "Valle d'Aosta": "Nord Ovest",
            "Valle d’Aosta": "Nord Ovest",
            "Liguria": "Nord Ovest",
            "Lombardia": "Nord Ovest",
            "Veneto": "Nord Est",
            "Trentino-Alto Adige": "Nord Est",
            "Trentino Alto Adige": "Nord Est",
            "Friuli-Venezia Giulia": "Nord Est",
            "Friuli Venezia Giulia": "Nord Est",
            "Emilia-Romagna": "Nord Est",
            "Emilia Romagna": "Nord Est",
            "Toscana": "Centro",
            "Umbria": "Centro",
            "Marche": "Centro",
            "Lazio": "Centro",
            "Abruzzo": "Sud",
            "Molise": "Sud",
            "Campania": "Sud",
            "Puglia": "Sud",
            "Basilicata": "Sud",
            "Calabria": "Sud",
            "Sicilia": "Isole",
            "Sardegna": "Isole",
        }
        return macro_map.get(r, None)

    def _build_consip_scadenza_map(df_consip: pd.DataFrame) -> pd.DataFrame:
        if df_consip is None or len(df_consip) == 0 or 'CIG' not in df_consip.columns:
            return pd.DataFrame(columns=['cig', 'scadenza_consip', 'durata_giorni_consip'])

        dfc = df_consip.copy()
        for col in ['DataAggiudicazione', 'DATA_ULTIMO_PERFEZIONAMENTO', 'DATA_COMUNICAZIONE_ESITO', 'DataPubblicazione']:
            if col in dfc.columns:
                dfc[col] = _to_dt(dfc[col], fmt='%d/%m/%Y')

        dfc['durata_giorni_consip'] = pd.to_numeric(dfc.get('DURATA_PREVISTA', pd.Series(dtype='float64')), errors='coerce')

        if 'DataAggiudicazione' in dfc.columns:
            start = dfc['DataAggiudicazione']
        else:
            start = pd.Series([pd.NaT] * len(dfc))
        for fallback_col in ['DATA_ULTIMO_PERFEZIONAMENTO', 'DATA_COMUNICAZIONE_ESITO', 'DataPubblicazione']:
            if fallback_col in dfc.columns:
                start = start.fillna(dfc[fallback_col])
        dfc['data_inizio_scadenza_consip'] = start
        dfc['scadenza_consip'] = dfc['data_inizio_scadenza_consip'] + pd.to_timedelta(dfc['durata_giorni_consip'], unit='D')

        dfc['cig'] = dfc['CIG'].astype(str).str.strip()
        dfc = dfc[dfc['cig'].ne('') & dfc['scadenza_consip'].notna()].copy()

        # Pulizia date fuori scala (evita 19xx e scadenze troppo lontane nel futuro)
        max_year = pd.Timestamp.now().year + 30
        year = dfc['scadenza_consip'].dt.year
        dfc = dfc[(year >= 2000) & (year <= max_year)]

        # Dedup: per lo stesso CIG prendo la scadenza massima (caso lotti/righe multiple)
        out = dfc.groupby('cig', as_index=False).agg({
            'scadenza_consip': 'max',
            'durata_giorni_consip': 'max'
        })
        return out

    def _compute_scadenze_contratti(df_base: pd.DataFrame, consip_map: pd.DataFrame, include_stime: bool, cig_enrichment_items=None) -> pd.DataFrame:
        if df_base is None or len(df_base) == 0:
            return pd.DataFrame()

        # Colonne minime per ridurre memoria
        keep_candidates = [
            'chiave', 'cig', 'ocid',
            'buyer_name', 'ente_appaltante',
            'supplier_name', 'aggiudicatario',
            'comune', 'buyer_locality', 'regione',
            'oggetto',
            'award_amount', 'importo_aggiudicazione',
            'award_date', 'data_aggiudicazione',
            'data_scadenza', 'durata_appalto',
            '_categoria', 'categoria', 'quick_category', 'tipo_appalto'
        ]
        keep_cols = [c for c in keep_candidates if c in df_base.columns]
        d = df_base[keep_cols].copy()

        # Normalizza campi principali
        if 'cig' in d.columns:
            d['cig'] = d['cig'].fillna('').astype(str).str.strip()
            d['cig'] = d['cig'].replace({'nan': '', 'None': ''})
        else:
            d['cig'] = ''

        if 'award_date' in d.columns:
            d['award_date'] = _to_dt(d['award_date'])
        elif 'data_aggiudicazione' in d.columns:
            d['award_date'] = _to_dt(d['data_aggiudicazione'])
        else:
            d['award_date'] = pd.NaT

        if 'award_amount' in d.columns:
            d['award_amount'] = pd.to_numeric(d['award_amount'], errors='coerce')
        elif 'importo_aggiudicazione' in d.columns:
            d['award_amount'] = pd.to_numeric(d['importo_aggiudicazione'], errors='coerce')
        else:
            d['award_amount'] = np.nan

        # (1) data_scadenza esplicita (se presente)
        if 'data_scadenza' in d.columns:
            d['scadenza_da_data_scadenza'] = _to_dt(d['data_scadenza'])
        else:
            d['scadenza_da_data_scadenza'] = pd.NaT

        # (2) scadenza da CONSIP (arricchimento per CIG)
        if consip_map is not None and len(consip_map) > 0:
            d = d.merge(consip_map, on='cig', how='left')
        else:
            d['scadenza_consip'] = pd.NaT
            d['durata_giorni_consip'] = np.nan

        # (3) scadenza da durata_appalto (dataset principale)
        if 'durata_appalto' in d.columns:
            d['durata_giorni_dataset'] = pd.to_numeric(d['durata_appalto'], errors='coerce')
        else:
            d['durata_giorni_dataset'] = np.nan
        d['scadenza_da_durata_appalto'] = d['award_date'] + pd.to_timedelta(d['durata_giorni_dataset'], unit='D')

        # (3.5) Regex extraction da oggetto (titolo gara)
        d['scadenza_da_regex'] = pd.NaT
        if 'oggetto' in d.columns:
            obj = d['oggetto'].fillna('').astype(str)
            # Pattern: "durata X mesi/anni/giorni" (solo con keyword "durata" per evitare falsi positivi)
            dur_match = obj.str.extract(r'durata\s*[:\s]?\s*(\d{1,4})\s*(mes[ei]|ann[oi]|giorn[oi])', flags=re.IGNORECASE, expand=True)
            dur_num = pd.to_numeric(dur_match[0], errors='coerce')
            dur_unit = dur_match[1].str.lower().str[:3]
            dur_days = np.where(dur_unit == 'mes', dur_num * 30, np.where(dur_unit == 'ann', dur_num * 365, dur_num))
            dur_days_series = pd.Series(dur_days, index=d.index, dtype='float64')
            # Clamp: max 30 anni (10950 giorni), ignora valori assurdi
            dur_days_series = dur_days_series.where(dur_days_series.between(1, 10950))
            valid_dur = dur_days_series.notna() & d['award_date'].notna()
            d.loc[valid_dur, 'scadenza_da_regex'] = d.loc[valid_dur, 'award_date'] + pd.to_timedelta(dur_days_series[valid_dur], unit='D')
            # Pattern implicito: triennale, biennale, quinquennale, ecc.
            still_nat = d['scadenza_da_regex'].isna()
            implicit = obj.str.extract(r'(triennal|biennal|quinquennal|quadriennal|settennal|novennal)', flags=re.IGNORECASE, expand=False)
            implicit_map = {'triennal': 3, 'biennal': 2, 'quinquennal': 5, 'quadriennal': 4, 'settennal': 7, 'novennal': 9}
            implicit_years = implicit.str.lower().map(implicit_map)
            implicit_days = implicit_years * 365
            d.loc[still_nat & implicit_days.notna(), 'scadenza_da_regex'] = (
                d.loc[still_nat & implicit_days.notna(), 'award_date'] + pd.to_timedelta(implicit_days[still_nat & implicit_days.notna()], unit='D')
            )

        # (4) enrichment LLM (da cache) - scadenza base/max se disponibili
        d['scadenza_base_llm'] = pd.NaT
        d['scadenza_max_llm'] = pd.NaT
        d['llm_confidence'] = np.nan
        d['llm_notes'] = ''

        if cig_enrichment_items and isinstance(cig_enrichment_items, dict) and d['cig'].notna().any():
            present = set(d['cig'].fillna('').astype(str).str.strip().tolist())
            rows = []
            for cig_key in present:
                item = cig_enrichment_items.get(cig_key)
                if not item or not isinstance(item, dict):
                    continue
                res = item.get('result')
                if not isinstance(res, dict):
                    continue
                rows.append({
                    'cig': cig_key,
                    'llm_duration_base_days': res.get('duration_base_days'),
                    'llm_duration_max_days': res.get('duration_max_days'),
                    'llm_explicit_start_date': res.get('explicit_start_date'),
                    'llm_explicit_end_date': res.get('explicit_end_date'),
                    'llm_confidence_cache': res.get('confidence'),
                    'llm_notes_cache': res.get('notes', ''),
                })
            if rows:
                llm_df = pd.DataFrame(rows)
                llm_df['llm_duration_base_days'] = pd.to_numeric(llm_df['llm_duration_base_days'], errors='coerce')
                llm_df['llm_duration_max_days'] = pd.to_numeric(llm_df['llm_duration_max_days'], errors='coerce')
                llm_df['llm_explicit_start_dt'] = pd.to_datetime(llm_df['llm_explicit_start_date'], errors='coerce')
                llm_df['llm_explicit_end_dt'] = pd.to_datetime(llm_df['llm_explicit_end_date'], errors='coerce')
                d = d.merge(
                    llm_df[['cig', 'llm_duration_base_days', 'llm_duration_max_days', 'llm_explicit_start_dt', 'llm_explicit_end_dt', 'llm_confidence_cache', 'llm_notes_cache']],
                    on='cig',
                    how='left'
                )

                start_llm = d['llm_explicit_start_dt'].fillna(d['award_date'])
                d['scadenza_base_llm'] = d['llm_explicit_end_dt']
                d.loc[d['scadenza_base_llm'].isna() & start_llm.notna() & d['llm_duration_base_days'].notna(),
                      'scadenza_base_llm'] = start_llm + pd.to_timedelta(d['llm_duration_base_days'], unit='D')
                d.loc[start_llm.notna() & d['llm_duration_max_days'].notna(),
                      'scadenza_max_llm'] = start_llm + pd.to_timedelta(d['llm_duration_max_days'], unit='D')

                # Propaga campi UI-friendly
                if 'llm_confidence_cache' in d.columns:
                    d['llm_confidence'] = pd.to_numeric(d['llm_confidence_cache'], errors='coerce')
                if 'llm_notes_cache' in d.columns:
                    d['llm_notes'] = d['llm_notes_cache'].fillna('').astype(str)

        # (5) stima da categoria (fallback)
        if include_stime:
            durate_stimate = {
                'Servizio Luce': 9, 'Illuminazione': 9,
                'Manutenzione': 4, 'Infrastrutture': 5,
                'Strade': 5, 'Edifici': 5, 'Scuole': 5,
                'Pulizie': 3, 'Riscaldamento': 7, 'Energia': 7, 'Termici': 7,
                'Vigilanza': 3, 'Videosorveglianza': 4,
                'Facchinaggio': 3, 'Verde': 3, 'Ambiente': 4,
                'Traslochi': 2, 'Portierato': 3, 'Disinfestazione': 2,
                'Rifiuti': 5, 'Acqua': 5, 'Acquedotti': 5,
                'Trasporti': 4, 'Mobilita': 4, 'Parcheggi': 5,
                'ICT': 3, 'Digitale': 3, 'Smart': 3,
                'Sanitario': 4, 'Sociale': 3, 'Formazione': 2,
                'Strutture Sportive': 5, 'Strutture_Sportive': 5, 'Gallerie': 5, 'Tunnel': 5,
                'Impianti': 5, 'Ricarica': 5, 'Colonnine': 5,
            }

            def get_durata_anni(cat):
                if pd.isna(cat):
                    return 3
                s = str(cat).lower()
                for key, val in durate_stimate.items():
                    if key.lower() in s:
                        return val
                return 3

            cat_col = '_categoria' if '_categoria' in d.columns else ('categoria' if 'categoria' in d.columns else None)
            if cat_col:
                d['durata_anni_stima'] = d[cat_col].apply(get_durata_anni)
                d['scadenza_stimata'] = d['award_date'] + pd.to_timedelta(d['durata_anni_stima'] * 365, unit='D')
            else:
                d['durata_anni_stima'] = np.nan
                d['scadenza_stimata'] = pd.NaT
        else:
            d['durata_anni_stima'] = np.nan
            d['scadenza_stimata'] = pd.NaT

        # Scadenza finale (priorità: esplicita > CONSIP > durata > regex > LLM > stima)
        d['scadenza_contratto'] = (
            d['scadenza_da_data_scadenza']
            .fillna(d['scadenza_consip'])
            .fillna(d['scadenza_da_durata_appalto'])
            .fillna(d['scadenza_da_regex'])
            .fillna(d['scadenza_base_llm'])
            .fillna(d['scadenza_stimata'])
        )

        # Fonte scadenza
        d['scadenza_fonte'] = np.select(
            [
                d['scadenza_da_data_scadenza'].notna(),
                d['scadenza_consip'].notna(),
                d['scadenza_da_durata_appalto'].notna(),
                d['scadenza_da_regex'].notna(),
                d['scadenza_base_llm'].notna(),
                d['scadenza_stimata'].notna()
            ],
            [
                'data_scadenza',
                'consip',
                'durata_appalto',
                'regex_oggetto',
                'llm',
                'stima_categoria'
            ],
            default='mancante'
        )

        # Pulizia date fuori scala (evita 19xx e scadenze troppo lontane nel futuro)
        max_year = pd.Timestamp.now().year + 30
        year = d['scadenza_contratto'].dt.year
        invalid = d['scadenza_contratto'].notna() & ((year < 2000) | (year > max_year))
        d.loc[invalid, 'scadenza_contratto'] = pd.NaT
        d.loc[invalid, 'scadenza_fonte'] = 'invalid'

        # Scadenza max (solo se stimata da LLM con rinnovi/proroghe quantificate)
        d['scadenza_contratto_max'] = d['scadenza_max_llm']
        year_max = d['scadenza_contratto_max'].dt.year
        invalid_max = d['scadenza_contratto_max'].notna() & ((year_max < 2000) | (year_max > max_year))
        d.loc[invalid_max, 'scadenza_contratto_max'] = pd.NaT

        oggi_ts = pd.Timestamp.now().normalize()
        d['giorni_alla_scadenza'] = (d['scadenza_contratto'] - oggi_ts).dt.days
        d['giorni_alla_scadenza_max'] = (d['scadenza_contratto_max'] - oggi_ts).dt.days
        d['stato_scadenza'] = np.select(
            [d['scadenza_contratto'].isna(), d['giorni_alla_scadenza'] < 0],
            ['Sconosciuta', 'Scaduto'],
            default='Attivo'
        )

        # Link di dettaglio (ANAC) - utile se serve verificare un CIG manualmente
        d['anac_url'] = d['cig'].apply(
            lambda x: f"https://dati.anticorruzione.it/superset/dashboard/dettaglio_cig/?cig={x}&standalone=2" if x else ""
        )
        return d

    consip_map_scadenze = _build_consip_scadenza_map(consip_exp)

    # ==================== ENRICHMENT AI SCADENZE ====================
    cig_cache = load_cig_enrichment_cache()
    cig_cache_items = cig_cache.get("items", {}) if isinstance(cig_cache, dict) else {}

    st.subheader("🔄 Arricchimento scadenze via AI")
    try:
        # Calcola candidati (senza stime) per mostrare conteggio
        df_for_candidates = _compute_scadenze_contratti(
            filtered_df, consip_map_scadenze, include_stime=False,
            cig_enrichment_items=cig_cache_items
        )
        candidates = []
        n_cached = 0
        if df_for_candidates is not None and len(df_for_candidates) > 0:
            base_mask = df_for_candidates['cig'].fillna('').astype(str).str.strip().apply(_is_valid_cig)
            miss = df_for_candidates['scadenza_contratto'].isna() | df_for_candidates['scadenza_fonte'].isin(['mancante', 'invalid'])
            candidates = (
                df_for_candidates.loc[base_mask & miss, 'cig']
                .dropna().astype(str).str.strip().str.upper().unique().tolist()
            )
            candidates.sort()
            n_cached = sum(1 for c in candidates if c in cig_cache_items and isinstance(cig_cache_items.get(c), dict) and cig_cache_items[c].get("result"))

        has_key = bool(get_openai_api_key())
        if not has_key:
            st.info("🔑 Per arricchire le scadenze, inserisci la OpenAI API Key nella sidebar.")
        else:
            st.caption(f"**{len(candidates):,}** contratti senza scadenza · **{n_cached:,}** già in cache".replace(",", "."))

        # Opzioni avanzate (nascoste)
        batch_size = 50
        force_refresh = False
        manual_cigs = ""
        include_all = False

        with st.expander("⚙️ Opzioni avanzate", expanded=False):
            batch_size = st.selectbox("Batch", [50, 200, 500], index=0, key="cig_enrich_batch")
            include_all = st.checkbox("Includi anche contratti con scadenza", value=False, key="cig_enrich_all")
            force_refresh = st.checkbox("Forza refresh (ignora cache)", value=False, key="cig_enrich_force")
            manual_cigs = st.text_input("CIG specifici (separati da virgola)", value="", key="cig_enrich_manual")

        # Bottone principale
        if has_key:
            run_btn = st.button("▶ Arricchisci scadenze mancanti", type="primary", key="cig_enrich_run")
            if run_btn:
                # Se include_all, ricalcola candidati senza filtro missing
                if include_all and df_for_candidates is not None and len(df_for_candidates) > 0:
                    candidates = (
                        df_for_candidates.loc[base_mask, 'cig']
                        .dropna().astype(str).str.strip().str.upper().unique().tolist()
                    )
                    candidates.sort()

                # Manual CIGs override
                manual_list = []
                if manual_cigs.strip():
                    manual_list = re.split(r"[\s,;]+", manual_cigs.strip())
                    manual_list = [_normalize_cig(c) for c in manual_list if _is_valid_cig(c)]

                cigs_to_run = manual_list if manual_list else candidates[:int(batch_size)]
                if not cigs_to_run:
                    st.info("Nessun CIG da processare con i filtri attuali.")
                else:
                    prog = st.progress(0)
                    status_box = st.empty()

                    def _cb(done, total, cig_now, status):
                        try:
                            prog.progress(min(1.0, done / max(1, total)))
                        except Exception:
                            pass
                        status_box.write(f"{done}/{total} - {cig_now} - {status}")

                    updated_cache, results_rows = enrich_cigs_via_llm(
                        cigs_to_run,
                        use_web=False,
                        force=force_refresh,
                        ttl_days=CIG_ENRICHMENT_TTL_DAYS_DEFAULT,
                        progress_cb=_cb,
                        save_every=5,
                    )
                    cig_cache_items = updated_cache.get("items", {}) if isinstance(updated_cache, dict) else cig_cache_items
                    prog.progress(1.0)
                    status_box.write("Completato.")

                    if results_rows:
                        res_df = pd.DataFrame(results_rows)
                        show_dataframe(res_df, label="cig_enrichment_results", use_container_width=True, hide_index=True)
                        try:
                            if "status" in res_df.columns and len(res_df) > 0:
                                counts = res_df["status"].value_counts(dropna=False).to_dict()
                                st.caption(f"Esiti: {counts}")
                                if "fatal" in counts:
                                    st.error("Errore FATALE: controlla API key / permessi modello / rete.")
                                elif counts.get("error", 0) == len(res_df):
                                    st.warning("Tutti i CIG in errore. Controlla la colonna 'notes'.")
                        except Exception:
                            pass
                    st.success("Cache aggiornata.")
    except Exception as e:
        st.error("❗️ Errore nella sezione Enrichment. La dashboard resta attiva.")
        with st.expander("Dettagli errore", expanded=False):
            st.exception(e)

    # ==================== VISTA TERRITORIALE: ATTIVI + SCADENZE ====================
    try:
        st.subheader("🧭 Contratti attivi per città/area e scadenza")
        st.caption("Scadenza calcolata con priorità: `data_scadenza` → CONSIP → `durata_appalto` → LLM (gpt-5-nano via cache) → stima per categoria (se abilitata). Scadenza max mostrata solo se il testo cita rinnovi/proroghe quantificate.")

        col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 2])
        with col_cfg1:
            raggruppa = st.radio("Raggruppa per", ["Comune", "Regione", "Macro-area"], horizontal=True, key="scad_raggruppa")
        with col_cfg2:
            include_stime = st.checkbox("Includi stime (fallback)", value=True, key="scad_include_stime")
        with col_cfg3:
            solo_attivi = st.checkbox("Solo contratti attivi", value=True, key="scad_solo_attivi")

        df_scad = _compute_scadenze_contratti(filtered_df, consip_map_scadenze, include_stime=include_stime, cig_enrichment_items=cig_cache_items)

        if df_scad is None or len(df_scad) == 0:
            st.info("Dati insufficienti per calcolare le scadenze con i filtri correnti.")
        else:
            # Colonne geografiche
            regione_col_scad = next((c for c in ['regione'] if c in df_scad.columns), None)
            comune_col_scad = next((c for c in ['comune', 'buyer_locality'] if c in df_scad.columns), None)

            if raggruppa == "Regione":
                group_col = regione_col_scad
            elif raggruppa == "Macro-area":
                if regione_col_scad:
                    df_scad['macro_area'] = df_scad[regione_col_scad].apply(_macro_area_from_regione)
                    group_col = 'macro_area'
                else:
                    group_col = None
            else:
                group_col = comune_col_scad

            if not group_col:
                st.warning("⚠️ Colonne geografiche non disponibili per il raggruppamento selezionato.")
            else:
                base = df_scad.copy()
                base[group_col] = base[group_col].astype('string').str.strip()
                base = base[base[group_col].notna() & base[group_col].ne('')].copy()

                if solo_attivi:
                    base = base[base['stato_scadenza'] == 'Attivo']

                # Orizzonte: utile per evidenziare scadenze “vicine” senza nascondere le aree
                max_anni = st.slider("Orizzonte scadenze (anni)", min_value=1, max_value=15, value=5, key="scad_orizzonte")
                solo_entro_orizzonte = st.checkbox("Mostra solo aree con scadenze entro orizzonte", value=False, key="scad_solo_entro")
                horizon_days = max_anni * 365

                if len(base) == 0:
                    st.info("Nessun contratto disponibile con i filtri correnti.")
                else:
                    id_col = next((c for c in ['chiave', 'ocid', 'cig'] if c in base.columns), 'cig')
                    base_future = base[base['giorni_alla_scadenza'].notna() & (base['giorni_alla_scadenza'] >= 0)].copy()
                    base_entro = base_future[base_future['giorni_alla_scadenza'] <= horizon_days].copy()

                    prossima = base_future.groupby(group_col, observed=True)['scadenza_contratto'].min().reset_index()
                    prossima = prossima.rename(columns={'scadenza_contratto': 'prossima_scadenza'})
                    prossima_max = base_future.groupby(group_col, observed=True)['scadenza_contratto_max'].min().reset_index()
                    prossima_max = prossima_max.rename(columns={'scadenza_contratto_max': 'prossima_scadenza_max'})

                    entro_counts = base_entro.groupby(group_col, observed=True).agg(
                        scadenze_entro_orizzonte=(id_col, 'nunique'),
                    ).reset_index()

                    summary = base.groupby(group_col, observed=True).agg(
                        contratti=(id_col, 'nunique'),
                        valore=('award_amount', 'sum'),
                        scadenze_12m=('giorni_alla_scadenza', lambda s: ((s >= 0) & (s <= 365)).sum()),
                    ).reset_index()
                    summary = summary.merge(prossima, on=group_col, how='left')
                    summary = summary.merge(prossima_max, on=group_col, how='left')
                    summary = summary.merge(entro_counts, on=group_col, how='left')
                    summary['scadenze_entro_orizzonte'] = summary['scadenze_entro_orizzonte'].fillna(0).astype(int)
                    summary['giorni_alla_prossima_scadenza'] = (summary['prossima_scadenza'] - pd.Timestamp.now().normalize()).dt.days
                    summary = summary.sort_values(['giorni_alla_prossima_scadenza', 'contratti'], ascending=[True, False])

                    if solo_entro_orizzonte:
                        summary = summary[summary['scadenze_entro_orizzonte'] > 0]

                    if len(summary) == 0:
                        st.info("Nessuna area con scadenze nell'orizzonte selezionato.")
                    else:
                        # KPI
                        k1, k2, k3, k4 = st.columns(4)
                        with k1:
                            st.metric("📍 Aree (con scadenze)", f"{summary[group_col].nunique():,}".replace(",", "."))
                        with k2:
                            st.metric("📋 Contratti", f"{int(summary['contratti'].sum()):,}".replace(",", "."))
                        with k3:
                            st.metric("⚠️ Scadenze 12 mesi", f"{int(summary['scadenze_12m'].sum()):,}".replace(",", "."))
                        with k4:
                            st.metric("💰 Valore (somma)", f"€{summary['valore'].sum()/1e9:.2f}B")

                        # Tabella
                        display = summary.copy()
                        display['prossima_scadenza'] = display['prossima_scadenza'].dt.strftime('%d/%m/%Y')
                        display['prossima_scadenza_max'] = display['prossima_scadenza_max'].dt.strftime('%d/%m/%Y')
                        display['valore'] = display['valore'].apply(lambda x: f"€{x/1e6:.1f}M" if pd.notna(x) else "-")
                        display.columns = [
                            raggruppa,
                            'Contratti',
                            'Valore',
                            'Prossima Scadenza (base)',
                            'Prossima Scadenza (max)',
                            'Scadenze 12M',
                            f"Scadenze entro {max_anni}a",
                            'Giorni a prossima scadenza'
                        ]
                        show_dataframe(display, label="scadenze_summary_by_area", use_container_width=True, hide_index=True)

                    # ===== MAPPA A BOLLE SCADENZE =====
                    if len(summary) > 0 and raggruppa == "Comune" and _geo_lookup:
                        st.markdown("#### 🗺️ Mappa Scadenze per Comune")
                        map_df = summary.copy()
                        # Geocode comuni tramite ISTAT lookup
                        coords = map_df[group_col].apply(lambda x: _geocode_comune(x, _geo_lookup))
                        map_df['lat'] = coords.apply(lambda c: c[0] if c else None)
                        map_df['lon'] = coords.apply(lambda c: c[1] if c else None)
                        map_df = map_df.dropna(subset=['lat', 'lon'])

                        if len(map_df) > 0:
                            # Colore: urgenza basata su giorni alla prossima scadenza
                            map_df['urgenza'] = map_df['giorni_alla_prossima_scadenza'].clip(0, horizon_days)
                            # Dimensione: numero scadenze entro orizzonte (min 3 per visibilità)
                            map_df['size'] = map_df['scadenze_entro_orizzonte'].clip(lower=1)
                            # Hover: info dettagliate
                            map_df['hover_text'] = (
                                map_df[group_col] + '<br>'
                                + 'Contratti: ' + map_df['contratti'].astype(str) + '<br>'
                                + 'Scadenze entro ' + str(max_anni) + 'a: ' + map_df['scadenze_entro_orizzonte'].astype(str) + '<br>'
                                + 'Prossima: ' + map_df['prossima_scadenza'].dt.strftime('%d/%m/%Y').fillna('-')
                            )

                            fig_map = px.scatter_map(
                                map_df,
                                lat='lat',
                                lon='lon',
                                size='size',
                                color='urgenza',
                                hover_name=group_col,
                                hover_data={
                                    'contratti': True,
                                    'scadenze_entro_orizzonte': True,
                                    'scadenze_12m': True,
                                    'urgenza': False,
                                    'size': False,
                                    'lat': False,
                                    'lon': False,
                                },
                                color_continuous_scale=['#d32f2f', '#ff9800', '#fdd835', '#66bb6a'],
                                size_max=40,
                                zoom=5,
                                center={'lat': 42.0, 'lon': 12.5},
                            )
                            fig_map.update_layout(
                                height=550,
                                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                coloraxis_colorbar_title="Giorni a scadenza",
                            )
                            st.plotly_chart(fig_map, use_container_width=True)
                            st.caption(f"Comuni mappati: {len(map_df)} su {len(summary)} ({len(map_df)/max(1,len(summary))*100:.0f}%)")
                        else:
                            st.info("📍 Nessun comune con coordinate disponibili per la mappa.")

                    elif len(summary) > 0 and raggruppa == "Regione" and _geo_lookup:
                        st.markdown("#### 🗺️ Mappa Scadenze per Regione")
                        regioni_coords = {
                            'Lombardia': (45.47, 9.85), 'Lazio': (41.89, 12.48), 'Campania': (40.83, 14.25),
                            'Sicilia': (37.60, 14.02), 'Veneto': (45.43, 11.87), 'Emilia-Romagna': (44.49, 11.34),
                            'Piemonte': (45.05, 7.52), 'Puglia': (41.12, 16.87), 'Toscana': (43.41, 11.22),
                            'Calabria': (38.91, 16.59), 'Sardegna': (40.12, 9.01), 'Liguria': (44.32, 8.40),
                            'Marche': (43.62, 13.52), 'Abruzzo': (42.35, 13.39), 'Friuli-Venezia Giulia': (45.64, 13.80),
                            'Trentino-Alto Adige': (46.50, 11.35), 'Umbria': (42.94, 12.62), 'Basilicata': (40.64, 15.81),
                            'Molise': (41.56, 14.66), "Valle d'Aosta": (45.74, 7.32),
                        }
                        map_df = summary.copy()
                        map_df['lat'] = map_df[group_col].map(lambda x: regioni_coords.get(x, (None, None))[0])
                        map_df['lon'] = map_df[group_col].map(lambda x: regioni_coords.get(x, (None, None))[1])
                        map_df = map_df.dropna(subset=['lat', 'lon'])
                        if len(map_df) > 0:
                            map_df['urgenza'] = map_df['giorni_alla_prossima_scadenza'].clip(0, horizon_days)
                            map_df['size'] = map_df['scadenze_entro_orizzonte'].clip(lower=1)
                            fig_map = px.scatter_map(
                                map_df, lat='lat', lon='lon', size='size', color='urgenza',
                                hover_name=group_col,
                                hover_data={'contratti': True, 'scadenze_entro_orizzonte': True, 'urgenza': False, 'size': False, 'lat': False, 'lon': False},
                                color_continuous_scale=['#d32f2f', '#ff9800', '#fdd835', '#66bb6a'],
                                size_max=60, zoom=5, center={'lat': 42.0, 'lon': 12.5},
                            )
                            fig_map.update_layout(height=550, margin={"r": 0, "t": 0, "l": 0, "b": 0}, coloraxis_colorbar_title="Giorni a scadenza")
                            st.plotly_chart(fig_map, use_container_width=True)

                    if len(summary) > 0:
                        # Drilldown per area
                        st.markdown("#### 🔎 Dettaglio per area")
                        area_sel = st.selectbox(
                            f"Seleziona {raggruppa.lower()}",
                            summary[group_col].tolist(),
                            key="scad_area_sel"
                        )
                        dettaglio = base[base[group_col] == area_sel].copy()
                        dettaglio = dettaglio[dettaglio['giorni_alla_scadenza'].notna() & (dettaglio['giorni_alla_scadenza'] >= 0)]
                        dettaglio_entro = st.checkbox(f"Dettaglio: solo scadenze entro {max_anni} anni", value=False, key="scad_det_entro")
                        if dettaglio_entro:
                            dettaglio = dettaglio[dettaglio['giorni_alla_scadenza'] <= horizon_days]
                        dettaglio = dettaglio.sort_values('scadenza_contratto', ascending=True)

                        cols_det = []
                        for c in ['scadenza_contratto', 'scadenza_contratto_max', 'giorni_alla_scadenza', 'scadenza_fonte', 'llm_confidence', 'llm_notes', 'cig', 'buyer_name', 'supplier_name', '_categoria', 'award_amount', 'award_date', 'anac_url']:
                            if c in dettaglio.columns:
                                cols_det.append(c)

                        if len(dettaglio) > 0 and cols_det:
                            det = dettaglio[cols_det].copy()
                            if 'award_amount' in det.columns:
                                det['award_amount'] = det['award_amount'].apply(lambda x: f"€{x/1e3:.0f}K" if pd.notna(x) else "-")
                            if 'award_date' in det.columns:
                                det['award_date'] = det['award_date'].dt.strftime('%d/%m/%Y')
                            if 'scadenza_contratto' in det.columns:
                                det['scadenza_contratto'] = det['scadenza_contratto'].dt.strftime('%d/%m/%Y')
                            if 'scadenza_contratto_max' in det.columns:
                                det['scadenza_contratto_max'] = det['scadenza_contratto_max'].dt.strftime('%d/%m/%Y')
                            if 'llm_confidence' in det.columns:
                                det['llm_confidence'] = pd.to_numeric(det['llm_confidence'], errors='coerce').round(2)
                            if 'llm_notes' in det.columns:
                                det['llm_notes'] = det['llm_notes'].apply(lambda x: str(x)[:120] if pd.notna(x) else '')
                            det = det.rename(columns={
                                'scadenza_contratto': 'Scadenza (base)',
                                'scadenza_contratto_max': 'Scadenza (max)',
                                'giorni_alla_scadenza': 'Giorni alla scadenza',
                                'scadenza_fonte': 'Fonte scadenza',
                                'llm_confidence': 'Confidence LLM',
                                'llm_notes': 'Note LLM',
                                'cig': 'CIG',
                                'buyer_name': 'Ente',
                                'supplier_name': 'Aggiudicatario',
                                '_categoria': 'Categoria',
                                'award_amount': 'Importo',
                                'award_date': 'Aggiudicazione',
                                'anac_url': 'Dettaglio ANAC'
                            })
                            show_dataframe(det, label="scadenze_drilldown_area", use_container_width=True, hide_index=True)

                            # ===== AI: Analisi su singola gara/contratto (dal dettaglio area) =====
                            with st.expander("🤖 Analisi AI su una gara (dal dettaglio)", expanded=False):
                                if not get_openai_api_key():
                                    st.info("Inserisci la tua OpenAI API Key nella sidebar (sezione 🤖 AI) per usare l’analisi.")
                                else:
                                    id_col_ai = next((c for c in ['cig', 'chiave', 'ocid'] if c in dettaglio.columns), None)
                                    if not id_col_ai:
                                        st.info("Nessun identificativo (cig/chiave/ocid) disponibile per selezionare una gara.")
                                    else:
                                        max_opts_ai = min(300, len(dettaglio))
                                        cand_ai = dettaglio.head(max_opts_ai).copy()
                                        cand_ai[id_col_ai] = cand_ai[id_col_ai].astype(str).str.strip()
                                        cand_ai = cand_ai[cand_ai[id_col_ai].notna() & cand_ai[id_col_ai].ne("")].copy()
                                        cand_ai = cand_ai.drop_duplicates(subset=[id_col_ai], keep="first")

                                        label_map_ai = {}
                                        for r in cand_ai.to_dict(orient="records"):
                                            k = str(r.get(id_col_ai, "")).strip()
                                            if not k or k.lower() in {"nan", "none"}:
                                                continue
                                            label_map_ai[k] = _ai_select_label_from_row(r, k)

                                        options_ai = list(label_map_ai.keys())
                                        if not options_ai:
                                            st.info("Nessuna gara selezionabile per analisi AI nel dettaglio.")
                                        else:
                                            sel_id_ai = st.selectbox(
                                                "Gara",
                                                options=options_ai,
                                                format_func=lambda x: label_map_ai.get(x, x),
                                                key="ai_select_scadenze_dettaglio",
                                            )
                                            question_ai = st.text_area(
                                                "Domanda (opzionale)",
                                                placeholder="Es. cosa sappiamo su scadenza/rinnovi e quali verifiche fare?",
                                                key="ai_question_scadenze_dettaglio",
                                                height=80,
                                            )
                                            if st.button("🤖 Analisi AI", type="primary", key="ai_run_scadenze_dettaglio"):
                                                if "ai_gara_cache" not in st.session_state:
                                                    st.session_state.ai_gara_cache = {}
                                                cache_key = f"scadenze:{raggruppa}:{area_sel}:{sel_id_ai}:{hashlib.md5(question_ai.encode('utf-8')).hexdigest()[:8]}"
                                                if cache_key in st.session_state.ai_gara_cache:
                                                    st.markdown(st.session_state.ai_gara_cache[cache_key])
                                                else:
                                                    rec_df = dettaglio[dettaglio[id_col_ai].astype(str).str.strip() == sel_id_ai]
                                                    if len(rec_df) == 0:
                                                        st.error("Record non trovato nel dettaglio (id non più presente).")
                                                    else:
                                                        rec = rec_df.iloc[0].to_dict()
                                                        out = ai_analyze_gara(rec, question=question_ai, model="gpt-5-nano")
                                                        if not out:
                                                            st.error("Errore: nessuna risposta (controlla API key / rete / permessi modello).")
                                                        else:
                                                            st.session_state.ai_gara_cache[cache_key] = out
                                                            st.markdown(out)
                        else:
                            st.info("Nessun dettaglio disponibile per l'area selezionata.")

                        # Download dettaglio
                        csv_det = dettaglio.to_csv(index=False)
                        st.download_button(
                            f"📥 Scarica dettaglio ({raggruppa}: {area_sel})",
                            csv_det,
                            f"contratti_scadenza_{raggruppa.lower()}_{str(area_sel).replace(' ', '_')}.csv",
                            "text/csv"
                        )
    except Exception as e:
        st.error("❗️ Errore nella sezione Scadenze (vista territoriale). La dashboard resta attiva.")
        with st.expander("Dettagli errore (Scadenze - vista territoriale)", expanded=False):
            st.exception(e)

    st.markdown("---")

    if len(consip_exp) > 0 and 'DURATA_PREVISTA' in consip_exp.columns:
        # Calcola scadenze (CONSIP) con fallback su date alternative quando DataAggiudicazione è mancante
        for col in ['DataAggiudicazione', 'DATA_ULTIMO_PERFEZIONAMENTO', 'DATA_COMUNICAZIONE_ESITO', 'DataPubblicazione']:
            if col in consip_exp.columns:
                consip_exp[col] = _to_dt(consip_exp[col], fmt='%d/%m/%Y')
        consip_exp['durata_giorni'] = pd.to_numeric(consip_exp['DURATA_PREVISTA'], errors='coerce')
        start = consip_exp['DataAggiudicazione'] if 'DataAggiudicazione' in consip_exp.columns else pd.Series([pd.NaT] * len(consip_exp))
        for fallback_col in ['DATA_ULTIMO_PERFEZIONAMENTO', 'DATA_COMUNICAZIONE_ESITO', 'DataPubblicazione']:
            if fallback_col in consip_exp.columns:
                start = start.fillna(consip_exp[fallback_col])
        consip_exp['data_inizio_scadenza'] = start
        consip_exp['ScadenzaContratto'] = consip_exp['data_inizio_scadenza'] + pd.to_timedelta(consip_exp['durata_giorni'], unit='D')

        # Filtra contratti validi con scadenza
        contratti_validi = consip_exp[consip_exp['ScadenzaContratto'].notna()].copy()
        from datetime import datetime
        oggi = datetime.now()

        # Contratti futuri
        contratti_futuri = contratti_validi[contratti_validi['ScadenzaContratto'] > oggi].copy()
        contratti_futuri['anno_scadenza'] = contratti_futuri['ScadenzaContratto'].dt.year

        # Filtra anni ragionevoli (no errori tipo 2129)
        contratti_futuri = contratti_futuri[contratti_futuri['anno_scadenza'] <= 2040]

        # KPI scadenze
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Contratti Attivi", f"{len(contratti_futuri):,}".replace(",", "."))
        with col2:
            valore_attivo = contratti_futuri['ImportoAggiudicazione'].sum() if 'ImportoAggiudicazione' in contratti_futuri.columns else 0
            st.metric("💰 Valore Attivo", f"€{valore_attivo/1e6:.1f}M")
        with col3:
            scad_2025 = len(contratti_futuri[contratti_futuri['anno_scadenza'] == 2025])
            st.metric("⚠️ Scadenza 2025", f"{scad_2025}")
        with col4:
            scad_2026 = len(contratti_futuri[contratti_futuri['anno_scadenza'] == 2026])
            st.metric("📌 Scadenza 2026", f"{scad_2026}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Contratti CONSIP in Scadenza per Anno")
            scadenze_anno = contratti_futuri.groupby('anno_scadenza', observed=True).agg({
                'CIG': 'count',
                'ImportoAggiudicazione': 'sum'
            }).reset_index()
            scadenze_anno.columns = ['Anno', 'N. Contratti', 'Valore']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=scadenze_anno['Anno'], y=scadenze_anno['N. Contratti'], name='N. Contratti', marker_color=CGL_GREEN),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=scadenze_anno['Anno'], y=scadenze_anno['Valore']/1e6, name='Valore (M€)', line=dict(color=CGL_BLUE, width=3)),
                secondary_y=True
            )
            fig.update_layout(height=400, title='Scadenze per Anno')
            fig.update_yaxes(title_text="N. Contratti", secondary_y=False)
            fig.update_yaxes(title_text="Valore (M€)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏢 Scadenze per Tipo Accordo")
            if 'TipoAccordo' in contratti_futuri.columns:
                scad_tipo = contratti_futuri.groupby('TipoAccordo', observed=True).agg({
                    'CIG': 'count',
                    'ImportoAggiudicazione': 'sum'
                }).reset_index()
                scad_tipo.columns = ['Tipo', 'N. Contratti', 'Valore']

                fig = px.pie(scad_tipo, values='N. Contratti', names='Tipo',
                             title='Distribuzione per Tipo Accordo',
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Timeline scadenze prossimi 3 anni
        st.markdown("---")
        st.subheader("📅 Timeline Scadenze Prossimi 3 Anni")

        prossimi_3_anni = contratti_futuri[contratti_futuri['anno_scadenza'] <= oggi.year + 3].copy()
        prossimi_3_anni['mese_scadenza'] = prossimi_3_anni['ScadenzaContratto'].dt.to_period('M').astype(str)

        if len(prossimi_3_anni) > 0:
            timeline = prossimi_3_anni.groupby('mese_scadenza', observed=True).agg({
                'CIG': 'count',
                'ImportoAggiudicazione': 'sum'
            }).reset_index()
            timeline.columns = ['Mese', 'N. Contratti', 'Valore']

            fig = px.bar(timeline, x='Mese', y='N. Contratti',
                         color='Valore', color_continuous_scale='Reds',
                         title='Contratti in Scadenza per Mese')
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nessun contratto in scadenza nei prossimi 3 anni")

        # Dettaglio contratti in scadenza
        st.markdown("---")
        st.subheader("📋 Dettaglio Contratti in Scadenza")

        # Filtro anno
        anni_disponibili = sorted(contratti_futuri['anno_scadenza'].unique())
        anno_filtro = st.selectbox("Filtra per Anno Scadenza", ["Tutti"] + [str(a) for a in anni_disponibili])

        if anno_filtro != "Tutti":
            contratti_mostra = contratti_futuri[contratti_futuri['anno_scadenza'] == int(anno_filtro)]
        else:
            contratti_mostra = contratti_futuri

        # Tabella dettaglio
        cols_display = ['ScadenzaContratto', 'TipoAccordo', 'Comune', 'Regione', 'Aggiudicatario', 'ImportoAggiudicazione', 'durata_giorni']
        cols_available = [c for c in cols_display if c in contratti_mostra.columns]

        if len(contratti_mostra) > 0:
            display_df = contratti_mostra[cols_available].copy()
            display_df['ScadenzaContratto'] = display_df['ScadenzaContratto'].dt.strftime('%d/%m/%Y')
            if 'ImportoAggiudicazione' in display_df.columns:
                display_df['ImportoAggiudicazione'] = display_df['ImportoAggiudicazione'].apply(lambda x: f'€{x/1e3:.0f}K' if pd.notna(x) else '-')
            if 'Aggiudicatario' in display_df.columns:
                display_df['Aggiudicatario'] = display_df['Aggiudicatario'].apply(lambda x: str(x)[:40] if pd.notna(x) else '-')

            display_df.columns = ['Scadenza', 'Tipo', 'Comune', 'Regione', 'Aggiudicatario', 'Valore', 'Durata (gg)']
            show_dataframe(display_df.sort_values('Scadenza'), label="consip_scadenze_dettaglio", use_container_width=True, hide_index=True)

            # Download
            csv = contratti_mostra.to_csv(index=False)
            st.download_button(
                "📥 Scarica Contratti in Scadenza (CSV)",
                csv,
                "contratti_scadenza.csv",
                "text/csv"
            )

        # Stima scadenze altri contratti (non CONSIP)
        st.markdown("---")
        st.subheader("📊 Stima Scadenze Altri Contratti (Non CONSIP)")
        st.markdown("""
        Per i contratti non CONSIP, stimiamo le scadenze basandoci su durate tipiche per categoria:
        - **Servizio Luce**: 9 anni (3285 giorni)
        - **Manutenzione**: 3-5 anni
        - **Pulizie**: 3 anni
        - **Riscaldamento**: 5-9 anni
        - **Vigilanza**: 3 anni
        - **Altri**: 3 anni (default)
        """)

        # Calcola stime per altri contratti
        durate_stimate = {
            'Servizio Luce': 9,
            'Manutenzione': 4,
            'Pulizie': 3,
            'Riscaldamento': 7,
            'Vigilanza': 3,
            'Facchinaggio': 3,
            'Verde': 3,
            'Traslochi': 2,
            'Portierato': 3,
            'Disinfestazione': 2
        }

        raw_estimate = filtered_df.copy()
        raw_estimate['award_date'] = pd.to_datetime(raw_estimate['award_date'], errors='coerce')
        # Converti a timezone-naive
        try:
            if raw_estimate['award_date'].dt.tz is not None:
                raw_estimate['award_date'] = raw_estimate['award_date'].dt.tz_convert(None)
        except:
            pass

        def get_durata_anni(cat):
            if pd.isna(cat):
                return 3
            for key, val in durate_stimate.items():
                if key.lower() in str(cat).lower():
                    return val
            return 3

        raw_estimate['durata_anni'] = raw_estimate['_categoria'].apply(get_durata_anni)
        raw_estimate['scadenza_stimata'] = raw_estimate['award_date'] + pd.to_timedelta(raw_estimate['durata_anni'] * 365, unit='D')

        # Converti scadenza a timezone-naive se necessario
        try:
            if raw_estimate['scadenza_stimata'].dt.tz is not None:
                raw_estimate['scadenza_stimata'] = raw_estimate['scadenza_stimata'].dt.tz_convert(None)
        except:
            pass

        raw_estimate['anno_scadenza_stima'] = raw_estimate['scadenza_stimata'].dt.year

        # Filtra scadenze future e ragionevoli (usa anno per evitare problemi timezone)
        anno_corrente = pd.Timestamp.now().year
        stima_future = raw_estimate[raw_estimate['scadenza_stimata'].notna() & (raw_estimate['anno_scadenza_stima'] >= anno_corrente) & (raw_estimate['anno_scadenza_stima'] <= 2040)]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Stima Scadenze per Anno")
            stima_anno = stima_future.groupby('anno_scadenza_stima', observed=True).agg({
                'ocid': 'count',
                'award_amount': 'sum'
            }).reset_index()
            stima_anno.columns = ['Anno', 'N. Contratti (stima)', 'Valore (stima)']

            fig = px.bar(stima_anno, x='Anno', y='N. Contratti (stima)',
                         color='Valore (stima)', color_continuous_scale='Blues',
                         title='Stima Contratti in Scadenza')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Stima Scadenze per Categoria")
            stima_cat = stima_future.groupby('_categoria', observed=True).agg({
                'ocid': 'count',
                'award_amount': 'sum'
            }).reset_index()
            stima_cat.columns = ['Categoria', 'N. Contratti', 'Valore']
            stima_cat = stima_cat.sort_values('N. Contratti', ascending=True).tail(10)

            fig = px.bar(stima_cat, x='N. Contratti', y='Categoria',
                         orientation='h', color='Valore',
                         color_continuous_scale='Greens',
                         title='Top 10 Categorie per Scadenze Future')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Alert scadenze imminenti
        st.markdown("---")
        st.subheader("⚠️ Alert: Contratti in Scadenza Prossimi 12 Mesi")

        # Usa anno per evitare problemi timezone
        imminenti = stima_future[stima_future['anno_scadenza_stima'] <= anno_corrente + 1]

        # Filtri geografici per Alert
        if len(imminenti) > 0:
            # Identifica colonne geografiche
            regione_col_alert = next((c for c in imminenti.columns if c.lower() == 'regione'), None)
            comune_col_alert = next((c for c in imminenti.columns if c.lower() in ['comune', 'citta', 'buyer_locality']), None)

            col_filter1, col_filter2 = st.columns(2)

            with col_filter1:
                if regione_col_alert and imminenti[regione_col_alert].notna().any():
                    regioni_alert = ['Tutte'] + sorted(imminenti[regione_col_alert].dropna().unique().tolist())
                    regione_alert_sel = st.selectbox("🗺️ Filtra per Regione", regioni_alert, key="alert_regione")
                else:
                    regione_alert_sel = 'Tutte'

            with col_filter2:
                # Filtra città in base alla regione selezionata
                imminenti_filtered_reg = imminenti.copy()
                if regione_alert_sel != 'Tutte' and regione_col_alert:
                    imminenti_filtered_reg = imminenti[imminenti[regione_col_alert] == regione_alert_sel]

                if comune_col_alert and imminenti_filtered_reg[comune_col_alert].notna().any():
                    comuni_alert = ['Tutte'] + sorted(imminenti_filtered_reg[comune_col_alert].dropna().unique().tolist())
                    comune_alert_sel = st.selectbox("🏙️ Filtra per Città", comuni_alert, key="alert_comune")
                else:
                    comune_alert_sel = 'Tutte'

            # Applica filtri
            imminenti_filtrati = imminenti.copy()
            if regione_alert_sel != 'Tutte' and regione_col_alert:
                imminenti_filtrati = imminenti_filtrati[imminenti_filtrati[regione_col_alert] == regione_alert_sel]
            if comune_alert_sel != 'Tutte' and comune_col_alert:
                imminenti_filtrati = imminenti_filtrati[imminenti_filtrati[comune_col_alert] == comune_alert_sel]

            # KPI
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Contratti Imminenti", f"{len(imminenti_filtrati):,}".replace(",", "."))
            with col2:
                st.metric("💰 Valore a Rischio", f"€{imminenti_filtrati['award_amount'].sum()/1e9:.2f}B")
            with col3:
                st.metric("🏢 Enti Coinvolti", f"{imminenti_filtrati['buyer_name'].nunique():,}".replace(",", "."))

            # Top categorie imminenti
            if len(imminenti_filtrati) > 0:
                imm_cat = imminenti_filtrati.groupby('_categoria', observed=True)['ocid'].count().sort_values(ascending=False).head(5)
                st.markdown("**Top 5 Categorie con Scadenze Imminenti:**")
                for cat, count in imm_cat.items():
                    st.write(f"- {cat}: {count} contratti")

                # Mappa scadenze imminenti
                st.markdown("---")
                st.markdown("#### 🗺️ Mappa Scadenze Imminenti")

                # Coordinate città italiane principali
                city_coords_alert = {
                    'Roma': (41.9028, 12.4964), 'Milano': (45.4642, 9.1900), 'Napoli': (40.8518, 14.2681),
                    'Torino': (45.0703, 7.6869), 'Palermo': (38.1157, 13.3615), 'Genova': (44.4056, 8.9463),
                    'Bologna': (44.4949, 11.3426), 'Firenze': (43.7696, 11.2558), 'Bari': (41.1171, 16.8719),
                    'Catania': (37.5079, 15.0830), 'Venezia': (45.4408, 12.3155), 'Verona': (45.4384, 10.9916),
                    'Messina': (38.1938, 15.5540), 'Padova': (45.4064, 11.8768), 'Trieste': (45.6495, 13.7768),
                    'Brescia': (45.5416, 10.2118), 'Parma': (44.8015, 10.3279), 'Taranto': (40.4644, 17.2470),
                    'Prato': (43.8777, 11.1020), 'Modena': (44.6471, 10.9252), 'Reggio Calabria': (38.1113, 15.6473),
                    'Reggio Emilia': (44.6989, 10.6297), 'Perugia': (43.1107, 12.3908), 'Livorno': (43.5485, 10.3106),
                    'Ravenna': (44.4184, 12.2035), 'Cagliari': (39.2238, 9.1217), 'Foggia': (41.4621, 15.5444),
                    'Rimini': (44.0678, 12.5695), 'Salerno': (40.6824, 14.7681), 'Ferrara': (44.8381, 11.6198),
                    'Ancona': (43.6158, 13.5189), 'Trento': (46.0679, 11.1211), 'Bolzano': (46.4983, 11.3548),
                    'Pescara': (42.4618, 14.2161), "L'Aquila": (42.3498, 13.3995), 'Campobasso': (41.5603, 14.6626),
                    'Potenza': (40.6404, 15.8056), 'Catanzaro': (38.9098, 16.5877), 'Aosta': (45.7372, 7.3209)
                }

                # Coordinate regioni (centroidi)
                regioni_coords_alert = {
                    'Lombardia': (45.4791, 9.8452), 'Lazio': (41.8931, 12.4828), 'Campania': (40.8333, 14.2500),
                    'Sicilia': (37.5994, 14.0154), 'Veneto': (45.4347, 11.8711), 'Emilia-Romagna': (44.4938, 11.3387),
                    'Piemonte': (45.0522, 7.5155), 'Puglia': (41.1171, 16.8719), 'Toscana': (43.4148, 11.2213),
                    'Calabria': (38.9098, 16.5877), 'Sardegna': (40.1209, 9.0129), 'Liguria': (44.3168, 8.3965),
                    'Marche': (43.6158, 13.5189), 'Abruzzo': (42.3541, 13.3919), 'Friuli-Venezia Giulia': (45.6361, 13.8040),
                    'Trentino-Alto Adige': (46.4993, 11.3548), 'Umbria': (42.9384, 12.6218), 'Basilicata': (40.6404, 15.8056),
                    'Molise': (41.5603, 14.6626), "Valle d'Aosta": (45.7372, 7.3209)
                }

                # Prova prima con città, poi con regioni
                map_created = False

                if comune_col_alert and imminenti_filtrati[comune_col_alert].notna().any():
                    # Mappa per città
                    cities_alert = imminenti_filtrati.groupby(comune_col_alert, observed=True).agg({
                        'award_amount': 'sum',
                        'ocid': 'count'
                    }).reset_index()
                    cities_alert.columns = ['citta', 'valore', 'num_contratti']
                    cities_alert = cities_alert.dropna(subset=['citta'])
                    cities_alert = cities_alert[cities_alert['citta'] != '']

                    cities_alert['lat'] = cities_alert['citta'].map(lambda x: city_coords_alert.get(x, (None, None))[0])
                    cities_alert['lng'] = cities_alert['citta'].map(lambda x: city_coords_alert.get(x, (None, None))[1])
                    cities_alert_valid = cities_alert.dropna(subset=['lat', 'lng']).sort_values('valore', ascending=False).head(30)

                    if len(cities_alert_valid) > 0:
                        fig_map = px.scatter_map(
                            cities_alert_valid,
                            lat='lat',
                            lon='lng',
                            size='valore',
                            color='num_contratti',
                            hover_name='citta',
                            hover_data={'num_contratti': True, 'valore': ':.2s'},
                            color_continuous_scale='Reds',
                            size_max=50,
                            zoom=5,
                            center={'lat': 42.0, 'lon': 12.5},
                        )
                        fig_map.update_layout(height=450, margin={"r":0,"t":0,"l":0,"b":0})
                        st.plotly_chart(fig_map, use_container_width=True)
                        map_created = True

                if not map_created and regione_col_alert and imminenti_filtrati[regione_col_alert].notna().any():
                    # Mappa per regioni se città non disponibili
                    regioni_alert_df = imminenti_filtrati.groupby(regione_col_alert, observed=True).agg({
                        'award_amount': 'sum',
                        'ocid': 'count'
                    }).reset_index()
                    regioni_alert_df.columns = ['regione', 'valore', 'num_contratti']
                    regioni_alert_df = regioni_alert_df.dropna(subset=['regione'])

                    regioni_alert_df['lat'] = regioni_alert_df['regione'].map(lambda x: regioni_coords_alert.get(x, (42.0, 12.5))[0])
                    regioni_alert_df['lon'] = regioni_alert_df['regione'].map(lambda x: regioni_coords_alert.get(x, (42.0, 12.5))[1])

                    if len(regioni_alert_df) > 0:
                        fig_map = px.scatter_map(
                            regioni_alert_df,
                            lat='lat',
                            lon='lon',
                            size='valore',
                            color='num_contratti',
                            hover_name='regione',
                            hover_data={'num_contratti': True, 'valore': ':.2s'},
                            color_continuous_scale='Reds',
                            size_max=60,
                            zoom=5,
                            center={'lat': 42.0, 'lon': 12.5},
                        )
                        fig_map.update_layout(height=450, margin={"r":0,"t":0,"l":0,"b":0})
                        st.plotly_chart(fig_map, use_container_width=True)
                        map_created = True

                if not map_created:
                    st.info("📍 Dati geografici non disponibili per visualizzare la mappa")
            else:
                st.info("Nessun contratto con i filtri selezionati")
        else:
            st.success("✅ Nessun contratto in scadenza nei prossimi 12 mesi")

    else:
        st.warning("⚠️ Dati CONSIP non disponibili per l'analisi delle scadenze")

        # Mostra comunque stime per altri contratti
        st.subheader("📊 Stima Scadenze Basata su Durate Tipiche")
        st.info("Calcolo delle scadenze stimate basate sulla data di aggiudicazione e durate tipiche per categoria")

# ==================== TAB 12: CONFRONTO AGGIUDICATARI ====================
if tab12:
  with tab12:
    st.subheader("⚔️ Confronto tra Aggiudicatari")

    supplier_col = 'supplier_name' if 'supplier_name' in filtered_df.columns else 'award_supplier_name'

    # Get top suppliers for selection
    top_suppliers_for_compare = filtered_df.groupby(supplier_col, observed=True)['award_amount'].sum().sort_values(ascending=False).head(100).index.tolist()

    col1, col2 = st.columns(2)
    with col1:
        supplier_a = st.selectbox("🔵 Aggiudicatario A", top_suppliers_for_compare, key="compare_a")
    with col2:
        supplier_b = st.selectbox("🔴 Aggiudicatario B", [s for s in top_suppliers_for_compare if s != supplier_a][:99], key="compare_b")

    if supplier_a and supplier_b:
        df_a = filtered_df[filtered_df[supplier_col] == supplier_a]
        df_b = filtered_df[filtered_df[supplier_col] == supplier_b]

        # KPI Comparison
        st.markdown("### 📊 Confronto KPI")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("🔵 Gare A", f"{len(df_a):,}".replace(",", "."))
            st.metric("🔴 Gare B", f"{len(df_b):,}".replace(",", "."))
        with col2:
            st.metric("🔵 Valore A", f"€{df_a['award_amount'].sum()/1e6:.1f}M")
            st.metric("🔴 Valore B", f"€{df_b['award_amount'].sum()/1e6:.1f}M")
        with col3:
            sconto_a = df_a['sconto'].mean() if 'sconto' in df_a.columns else 0
            sconto_b = df_b['sconto'].mean() if 'sconto' in df_b.columns else 0
            st.metric("🔵 Sconto Medio A", f"{sconto_a:.1f}%")
            st.metric("🔴 Sconto Medio B", f"{sconto_b:.1f}%")
        with col4:
            region_col_kpi = 'regione' if 'regione' in df_a.columns else 'Regione' if 'Regione' in df_a.columns else 'buyer_region'
            regioni_a = df_a[region_col_kpi].nunique() if region_col_kpi in df_a.columns else 0
            regioni_b = df_b[region_col_kpi].nunique() if region_col_kpi in df_b.columns else 0
            st.metric("🔵 Regioni A", f"{regioni_a}")
            st.metric("🔴 Regioni B", f"{regioni_b}")
        with col5:
            enti_a = df_a['buyer_name'].nunique() if 'buyer_name' in df_a.columns else 0
            enti_b = df_b['buyer_name'].nunique() if 'buyer_name' in df_b.columns else 0
            st.metric("🔵 Enti A", f"{enti_a}")
            st.metric("🔴 Enti B", f"{enti_b}")

        st.markdown("---")

        # Trend comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Trend Annuale Comparato")
            trend_a = df_a.groupby('anno', observed=True).agg({'award_amount': 'sum', 'ocid': 'count'}).reset_index()
            trend_a['Aggiudicatario'] = supplier_a[:30]
            trend_b = df_b.groupby('anno', observed=True).agg({'award_amount': 'sum', 'ocid': 'count'}).reset_index()
            trend_b['Aggiudicatario'] = supplier_b[:30]
            trend_compare = pd.concat([trend_a, trend_b])

            fig = px.line(trend_compare, x='anno', y='award_amount', color='Aggiudicatario',
                         markers=True, labels={'anno': 'Anno', 'award_amount': 'Valore (€)'})
            fig.update_layout(height=350)
            render_chart_with_save(fig, "Trend Confronto Aggiudicatari", "Trend annuale comparato tra due aggiudicatari", "compare_trend")

        with col2:
            st.markdown("### 📦 Categorie a Confronto")
            cat_col = '_categoria' if '_categoria' in filtered_df.columns else 'categoria'
            if cat_col in df_a.columns:
                cat_a = df_a.groupby(cat_col, observed=True)['award_amount'].sum().reset_index()
                cat_a['Aggiudicatario'] = supplier_a[:20]
                cat_b = df_b.groupby(cat_col, observed=True)['award_amount'].sum().reset_index()
                cat_b['Aggiudicatario'] = supplier_b[:20]
                cat_compare = pd.concat([cat_a, cat_b])

                fig = px.bar(cat_compare, x=cat_col, y='award_amount', color='Aggiudicatario',
                            barmode='group', labels={cat_col: 'Categoria', 'award_amount': 'Valore (€)'})
                fig.update_layout(height=350, xaxis_tickangle=-45)
                render_chart_with_save(fig, "Categorie Confronto", "Confronto categorie tra due aggiudicatari", "compare_categories")

        # Aree di influenza
        st.markdown("### 🗺️ Aree di Influenza Territoriale")

        region_col = 'regione' if 'regione' in filtered_df.columns else 'Regione' if 'Regione' in filtered_df.columns else 'buyer_region'
        if region_col in df_a.columns:
            col1, col2 = st.columns(2)

            with col1:
                reg_a = df_a.groupby(region_col, observed=True)['award_amount'].sum().sort_values(ascending=False).head(10).reset_index()
                reg_a.columns = ['Regione', 'Valore']
                st.markdown(f"**🔵 Top Regioni {supplier_a[:25]}**")
                fig = px.bar(reg_a, x='Valore', y='Regione', orientation='h', color_discrete_sequence=['#636EFA'])
                fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="influence_a")

            with col2:
                reg_b = df_b.groupby(region_col, observed=True)['award_amount'].sum().sort_values(ascending=False).head(10).reset_index()
                reg_b.columns = ['Regione', 'Valore']
                st.markdown(f"**🔴 Top Regioni {supplier_b[:25]}**")
                fig = px.bar(reg_b, x='Valore', y='Regione', orientation='h', color_discrete_sequence=['#EF553B'])
                fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="influence_b")

            # Overlap analysis
            st.markdown("### 🔄 Sovrapposizione Territoriale")
            regioni_a_set = set(df_a[region_col].dropna().unique())
            regioni_b_set = set(df_b[region_col].dropna().unique())
            overlap = regioni_a_set & regioni_b_set
            only_a = regioni_a_set - regioni_b_set
            only_b = regioni_b_set - regioni_a_set

            col1, col2, col3 = st.columns(3)
            col1.metric("🔵 Solo A", f"{len(only_a)} regioni")
            col2.metric("🟣 Entrambi", f"{len(overlap)} regioni")
            col3.metric("🔴 Solo B", f"{len(only_b)} regioni")

            if overlap:
                st.info(f"**Regioni in comune**: {', '.join(sorted(overlap))}")

# ==================== TAB 13: STAGIONALITÀ ====================
if tab13:
  with tab13:
    st.subheader("📆 Analisi Stagionalità")

    # Helper per trovare colonne dinamicamente
    def get_col_stag(df, candidates):
        for col in candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    # Identifica colonne per stagionalità
    amount_col_stag = get_col_stag(filtered_df, ['importo_aggiudicazione', 'award_amount', 'tender_amount'])
    id_col_stag = get_col_stag(filtered_df, ['chiave', 'CIG', 'ocid', 'id'])
    supplier_col_stag = get_col_stag(filtered_df, ['aggiudicatario', 'supplier_name', 'award_supplier_name'])

    # Deriva mese da data_aggiudicazione se non esiste
    if 'mese' not in filtered_df.columns or filtered_df['mese'].isna().all():
        date_col = 'data_aggiudicazione' if 'data_aggiudicazione' in filtered_df.columns else 'award_date'
        if date_col in filtered_df.columns:
            temp_dates = pd.to_datetime(filtered_df[date_col], errors='coerce')
            filtered_df = filtered_df.copy()
            filtered_df['mese'] = temp_dates.dt.month

    # Monthly distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📅 Distribuzione Mensile Gare")
        if 'mese' in filtered_df.columns and filtered_df['mese'].notna().any():
            df_monthly = filtered_df[filtered_df['mese'].notna()].copy()
            # Aggrega - conta righe e somma importi
            monthly = df_monthly.groupby('mese', observed=True).size().reset_index(name='n_gare')
            if amount_col_stag:
                monthly['valore'] = df_monthly.groupby('mese', observed=True)[amount_col_stag].sum().values

            monthly['mese_nome'] = monthly['mese'].map({
                1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mag', 6: 'Giu',
                7: 'Lug', 8: 'Ago', 9: 'Set', 10: 'Ott', 11: 'Nov', 12: 'Dic'
            })
            monthly = monthly.sort_values('mese')

            if len(monthly) > 0:
                color_col = 'valore' if 'valore' in monthly.columns else None
                fig = px.bar(monthly, x='mese_nome', y='n_gare',
                            color=color_col, color_continuous_scale='Viridis',
                            labels={'mese_nome': 'Mese', 'n_gare': 'N. Gare', 'valore': 'Valore'})
                fig.update_layout(height=350)
                render_chart_with_save(fig, "Distribuzione Mensile Gare", "Numero gare per mese", "monthly_dist")

                # Best/worst months
                best_month = monthly.loc[monthly['n_gare'].idxmax(), 'mese_nome']
                worst_month = monthly.loc[monthly['n_gare'].idxmin(), 'mese_nome']
                st.success(f"📈 **Mese più attivo**: {best_month}")
                st.warning(f"📉 **Mese meno attivo**: {worst_month}")
            else:
                st.info("Nessun dato mensile disponibile per i filtri selezionati")
        else:
            st.info("Dati mensili non disponibili - verifica che il campo data_aggiudicazione sia presente")

    with col2:
        st.markdown("### 📊 Heatmap Anno × Mese")
        if 'mese' in filtered_df.columns and 'anno' in filtered_df.columns:
            df_with_dates = filtered_df[filtered_df['mese'].notna() & filtered_df['anno'].notna()]
            if len(df_with_dates) > 0:
                # Conta per anno e mese
                pivot_monthly = df_with_dates.groupby(['anno', 'mese'], observed=True).size().reset_index(name='n_gare')
                pivot_table = pivot_monthly.pivot(index='mese', columns='anno', values='n_gare').fillna(0)

                mese_names = {1: 'Gen', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mag', 6: 'Giu',
                             7: 'Lug', 8: 'Ago', 9: 'Set', 10: 'Ott', 11: 'Nov', 12: 'Dic'}
                pivot_table.index = pivot_table.index.map(mese_names)

                fig = px.imshow(pivot_table, color_continuous_scale=BRAND_CONTINUOUS_SCALE,
                               labels={'color': 'N. Gare'}, aspect='auto')
                fig.update_layout(height=350)
                render_chart_with_save(fig, "Heatmap Anno/Mese", "Distribuzione gare per anno e mese", "heatmap_year_month")
            else:
                st.info("Nessun dato per heatmap")

    st.markdown("---")

    # Quarterly analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Analisi Trimestrale")
        if 'mese' in filtered_df.columns and filtered_df['mese'].notna().any():
            df_quarter = filtered_df[filtered_df['mese'].notna()].copy()
            df_quarter['trimestre'] = ((df_quarter['mese'] - 1) // 3) + 1

            # Aggrega usando colonne dinamiche
            quarterly = df_quarter.groupby('trimestre', observed=True).size().reset_index(name='n_gare')
            if amount_col_stag:
                quarterly['valore'] = df_quarter.groupby('trimestre', observed=True)[amount_col_stag].sum().values
            quarterly['trimestre_nome'] = quarterly['trimestre'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

            if len(quarterly) > 0 and 'valore' in quarterly.columns:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=quarterly['trimestre_nome'], y=quarterly['valore'],
                                    name='Valore (€)', marker_color=CGL_GREEN), secondary_y=False)
                fig.add_trace(go.Scatter(x=quarterly['trimestre_nome'], y=quarterly['n_gare'],
                                        name='N. Gare', line=dict(color=CGL_BLUE, width=3)), secondary_y=True)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True, key="quarterly_analysis")
            else:
                st.info("Nessun dato trimestrale")

    with col2:
        st.markdown("### 📊 Valore Medio per Trimestre")
        if 'mese' in filtered_df.columns and filtered_df['mese'].notna().any() and amount_col_stag:
            df_quarter = filtered_df[filtered_df['mese'].notna()].copy()
            df_quarter['trimestre'] = ((df_quarter['mese'] - 1) // 3) + 1

            # Calcola valore medio per gara per trimestre
            quarterly_avg = df_quarter.groupby('trimestre', observed=True).agg({
                amount_col_stag: ['mean', 'median', 'count']
            }).reset_index()
            quarterly_avg.columns = ['trimestre', 'valore_medio', 'valore_mediano', 'n_gare']
            quarterly_avg['trimestre_nome'] = quarterly_avg['trimestre'].map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

            if len(quarterly_avg) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=quarterly_avg['trimestre_nome'],
                    y=quarterly_avg['valore_medio'] / 1000,  # In migliaia
                    marker_color=CGL_CYAN,
                    text=quarterly_avg['valore_medio'].apply(lambda x: f'€{x/1000:.0f}K'),
                    textposition='outside',
                    name='Media'
                ))
                fig.add_trace(go.Scatter(
                    x=quarterly_avg['trimestre_nome'],
                    y=quarterly_avg['valore_mediano'] / 1000,
                    mode='lines+markers',
                    line=dict(color=CGL_ORANGE, width=2),
                    name='Mediana'
                ))
                fig.update_layout(height=300, yaxis_title='Valore (€K)', xaxis_title='Trimestre')
                st.plotly_chart(fig, use_container_width=True, key="quarterly_valore_medio")
            else:
                st.info("Nessun dato valore medio per trimestre")
        else:
            st.info("Dati valore/mese non disponibili")

    # Year-over-year growth with year selection
    st.markdown("---")
    st.markdown("### 📈 Evoluzione Temporale Aggiudicatari")

    # Year range selection - default to last 5 years with data
    available_years = sorted(filtered_df['anno'].dropna().unique())
    available_years = [int(y) for y in available_years if 2010 <= y <= 2030]

    if len(available_years) > 0:
        # Default: ultimi 5 anni o dall'inizio se meno
        default_start_idx = max(0, len(available_years) - 5)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            anno_inizio = st.selectbox("Anno inizio", available_years, index=default_start_idx, key="growth_start_year")
        with col2:
            anni_fine_options = [y for y in available_years if y >= anno_inizio]
            anno_fine = st.selectbox("Anno fine", anni_fine_options, index=len(anni_fine_options)-1 if anni_fine_options else 0, key="growth_end_year")
        with col3:
            n_suppliers = st.slider("Numero aggiudicatari", 5, 20, 10, key="n_suppliers_growth")

        # Filter by year range
        df_years = filtered_df[(filtered_df['anno'] >= anno_inizio) & (filtered_df['anno'] <= anno_fine)]

        # Usa colonna dinamica per supplier
        if supplier_col_stag and amount_col_stag and len(df_years) > 0:
            top_for_growth = df_years.groupby(supplier_col_stag, observed=True)[amount_col_stag].sum().sort_values(ascending=False).head(n_suppliers).index.tolist()
        else:
            top_for_growth = []
    else:
        st.info("Nessun anno disponibile nei dati filtrati")
        top_for_growth = []
        df_years = pd.DataFrame()
        anno_inizio = 2020
        anno_fine = 2024

    # Line chart - evolution over time
    st.markdown("#### 📊 Trend Valore per Anno")
    growth_lines = []
    if supplier_col_stag and amount_col_stag:
        for supplier in top_for_growth:
            supplier_yearly = df_years[df_years[supplier_col_stag] == supplier].groupby('anno', observed=True)[amount_col_stag].sum().reset_index()
            supplier_yearly['Aggiudicatario'] = supplier[:35]
            supplier_yearly.columns = ['anno', 'valore', 'Aggiudicatario']
            growth_lines.append(supplier_yearly)

    if growth_lines:
        growth_lines_df = pd.concat(growth_lines, ignore_index=True)
        fig = px.line(growth_lines_df, x='anno', y='valore', color='Aggiudicatario',
                     markers=True, labels={'anno': 'Anno', 'valore': 'Valore (€)'})
        fig.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=-0.4, font=dict(size=10)))
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True, key="growth_lines")

    # Growth rate bar charts - split by value and count
    st.markdown(f"### 📈 Crescita % ({anno_inizio} → {anno_fine})")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Crescita Valore Aggiudicato")
        growth_value_data = []
        if supplier_col_stag and amount_col_stag:
            for supplier in top_for_growth:
                supplier_df = df_years[df_years[supplier_col_stag] == supplier]
                yearly = supplier_df.groupby('anno', observed=True)[amount_col_stag].sum()
                val_inizio = yearly.get(anno_inizio, 0)
                val_fine = yearly.get(anno_fine, 0)
                if val_inizio > 0:
                    growth = ((val_fine - val_inizio) / val_inizio * 100)
                else:
                    growth = 100 if val_fine > 0 else 0
                growth_value_data.append({
                    'Aggiudicatario': supplier[:30],
                    'Crescita %': round(growth, 1),
                    f'Valore {anno_inizio}': val_inizio,
                    f'Valore {anno_fine}': val_fine
                })

        if growth_value_data:
            growth_val_df = pd.DataFrame(growth_value_data).sort_values('Crescita %', ascending=True)
            fig = px.bar(growth_val_df, x='Crescita %', y='Aggiudicatario', orientation='h',
                        color='Crescita %', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                        hover_data={f'Valore {anno_inizio}': ':,.0f', f'Valore {anno_fine}': ':,.0f'})
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True, key="growth_value")

            # Summary stats
            avg_growth = growth_val_df['Crescita %'].mean()
            positive = len(growth_val_df[growth_val_df['Crescita %'] > 0])
            st.caption(f"Media: {avg_growth:.1f}% | Positivi: {positive}/{len(growth_val_df)}")

    with col2:
        st.markdown("#### 🏆 Crescita Gare Vinte")
        growth_count_data = []
        if supplier_col_stag and id_col_stag:
            for supplier in top_for_growth:
                supplier_df = df_years[df_years[supplier_col_stag] == supplier]
                yearly_count = supplier_df.groupby('anno', observed=True)[id_col_stag].count()
                count_inizio = yearly_count.get(anno_inizio, 0)
                count_fine = yearly_count.get(anno_fine, 0)
                if count_inizio > 0:
                    growth = ((count_fine - count_inizio) / count_inizio * 100)
                else:
                    growth = 100 if count_fine > 0 else 0
                growth_count_data.append({
                    'Aggiudicatario': supplier[:30],
                    'Crescita %': round(growth, 1),
                    f'Gare {anno_inizio}': int(count_inizio),
                    f'Gare {anno_fine}': int(count_fine)
                })

        if growth_count_data:
            growth_cnt_df = pd.DataFrame(growth_count_data).sort_values('Crescita %', ascending=True)
            fig = px.bar(growth_cnt_df, x='Crescita %', y='Aggiudicatario', orientation='h',
                        color='Crescita %', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                        hover_data={f'Gare {anno_inizio}': True, f'Gare {anno_fine}': True})
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True, key="growth_count")

            # Summary stats
            avg_growth = growth_cnt_df['Crescita %'].mean()
            positive = len(growth_cnt_df[growth_cnt_df['Crescita %'] > 0])
            st.caption(f"Media: {avg_growth:.1f}% | Positivi: {positive}/{len(growth_cnt_df)}")

    # Detailed table
    st.markdown("---")
    st.markdown("#### 📋 Dettaglio Completo")
    detail_data = []
    if supplier_col_stag and amount_col_stag and id_col_stag:
        for supplier in top_for_growth:
            supplier_df = df_years[df_years[supplier_col_stag] == supplier]
            yearly_val = supplier_df.groupby('anno', observed=True)[amount_col_stag].sum()
            yearly_cnt = supplier_df.groupby('anno', observed=True)[id_col_stag].count()

            val_inizio = yearly_val.get(anno_inizio, 0)
            val_fine = yearly_val.get(anno_fine, 0)
            cnt_inizio = yearly_cnt.get(anno_inizio, 0)
            cnt_fine = yearly_cnt.get(anno_fine, 0)

            growth_val = ((val_fine - val_inizio) / val_inizio * 100) if val_inizio > 0 else (100 if val_fine > 0 else 0)
            growth_cnt = ((cnt_fine - cnt_inizio) / cnt_inizio * 100) if cnt_inizio > 0 else (100 if cnt_fine > 0 else 0)

            detail_data.append({
                'Aggiudicatario': supplier[:40],
                f'Valore {anno_inizio}': f"€{val_inizio:,.0f}",
                f'Valore {anno_fine}': f"€{val_fine:,.0f}",
                'Δ Valore %': f"{growth_val:+.1f}%",
                f'Gare {anno_inizio}': int(cnt_inizio),
                f'Gare {anno_fine}': int(cnt_fine),
                'Δ Gare %': f"{growth_cnt:+.1f}%"
            })

    if detail_data:
        show_dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

# ==================== TAB 14: NETWORK ANALYSIS ====================
if tab14:
  with tab14:
    st.subheader("🌐 Network Enti-Fornitori")

    # Helper per colonne dinamiche
    def get_col_net(df, candidates):
        for col in candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    supplier_col_net = get_col_net(filtered_df, ['aggiudicatario', 'supplier_name', 'award_supplier_name'])
    buyer_col_net = get_col_net(filtered_df, ['ente_appaltante', 'buyer_name', 'buyer_locality'])
    amount_col_net = get_col_net(filtered_df, ['importo_aggiudicazione', 'award_amount', 'tender_amount'])
    id_col_net = get_col_net(filtered_df, ['chiave', 'CIG', 'ocid', 'id'])

    st.markdown("### 🔗 Analisi Relazioni")

    if not all([supplier_col_net, buyer_col_net, amount_col_net, id_col_net]):
        st.warning("Dati insufficienti per l'analisi di rete. Colonne mancanti.")
    else:
        # Top relationships
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🤝 Top Coppie Ente-Fornitore")
            relationships = filtered_df.groupby([buyer_col_net, supplier_col_net], observed=True).agg({
                id_col_net: 'count',
                amount_col_net: 'sum'
            }).reset_index()
            relationships.columns = ['Ente', 'Fornitore', 'N. Gare', 'Valore']
            relationships = relationships.sort_values('Valore', ascending=False).head(20)

            relationships['Ente_short'] = relationships['Ente'].str[:40]
            relationships['Fornitore_short'] = relationships['Fornitore'].str[:30]
            relationships['Coppia'] = relationships['Ente_short'] + ' ↔ ' + relationships['Fornitore_short']

            fig = px.bar(relationships.head(15), x='Valore', y='Coppia', orientation='h',
                        color='N. Gare', color_continuous_scale='Viridis',
                        hover_data={'Ente': True, 'Fornitore': True})
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            render_chart_with_save(fig, "Top Coppie Ente-Fornitore", "Relazioni più frequenti ente-fornitore", "top_relationships")

        with col2:
            st.markdown("#### 🏆 Fornitori più Fedeli (ripetuti)")
            loyalty = filtered_df.groupby([supplier_col_net, buyer_col_net], observed=True).size().reset_index(name='gare_insieme')
            loyalty_agg = loyalty.groupby(supplier_col_net, observed=True).agg({
                buyer_col_net: 'count',  # quanti enti diversi
                'gare_insieme': 'sum'   # totale gare
            }).reset_index()
            loyalty_agg.columns = ['Fornitore', 'N. Enti', 'Totale Gare']
            loyalty_agg['Gare/Ente'] = loyalty_agg['Totale Gare'] / loyalty_agg['N. Enti']
            loyalty_agg = loyalty_agg.sort_values('Gare/Ente', ascending=False).head(20)

            fig = px.scatter(loyalty_agg, x='N. Enti', y='Gare/Ente', size='Totale Gare',
                            hover_name='Fornitore', color='Totale Gare',
                            color_continuous_scale='Plasma',
                            labels={'N. Enti': 'Numero Enti Diversi', 'Gare/Ente': 'Media Gare per Ente'})
            fig.update_layout(height=500)
            render_chart_with_save(fig, "Fornitori Fedeli", "Analisi fedeltà fornitori agli enti", "loyalty_scatter")

        st.markdown("---")

        # Concentration analysis
        st.markdown("### 📊 Concentrazione per Ente")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Enti con più Fornitori Diversi")
            enti_diversity = filtered_df.groupby(buyer_col_net, observed=True)[supplier_col_net].nunique().sort_values(ascending=False).head(15).reset_index()
            enti_diversity.columns = ['Ente', 'N. Fornitori']

            fig = px.bar(enti_diversity, x='N. Fornitori', y='Ente', orientation='h',
                        color='N. Fornitori', color_continuous_scale='Greens')
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            render_chart_with_save(fig, "Enti con più Fornitori", "Diversificazione fornitori per ente", "enti_diversity")

        with col2:
            st.markdown("#### ⚠️ Enti con Alta Concentrazione (pochi fornitori)")
            # Enti con almeno 10 gare ma pochi fornitori
            enti_stats = filtered_df.groupby(buyer_col_net, observed=True).agg({
                id_col_net: 'count',
                supplier_col_net: 'nunique'
            }).reset_index()
            enti_stats.columns = ['Ente', 'N. Gare', 'N. Fornitori']
            enti_stats = enti_stats[enti_stats['N. Gare'] >= 10]  # almeno 10 gare
            enti_stats['Concentrazione'] = enti_stats['N. Gare'] / enti_stats['N. Fornitori']
            enti_stats = enti_stats.sort_values('Concentrazione', ascending=False).head(15)

            fig = px.bar(enti_stats, x='Concentrazione', y='Ente', orientation='h',
                        color='N. Gare', color_continuous_scale='Reds',
                        hover_data={'N. Fornitori': True})
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            render_chart_with_save(fig, "Enti Alta Concentrazione", "Enti con pochi fornitori ma molte gare", "enti_concentration")

        # Anomaly detection - price outliers
        st.markdown("---")
        st.markdown("### 🔍 Rilevamento Anomalie Prezzi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Outlier per Importo (Z-Score > 3)")
            if amount_col_net:
                # Calculate z-score
                mean_val = filtered_df[amount_col_net].mean()
                std_val = filtered_df[amount_col_net].std()
                if std_val > 0:
                    filtered_df['z_score'] = (filtered_df[amount_col_net] - mean_val) / std_val

                    outliers = filtered_df[filtered_df['z_score'].abs() > 3].copy()
                    if len(outliers) > 0:
                        outliers_display = outliers[[supplier_col_net, buyer_col_net, amount_col_net, 'z_score']].head(20)
                        outliers_display[amount_col_net] = outliers_display[amount_col_net].apply(lambda x: f'€{x/1e6:.2f}M' if pd.notna(x) else 'N/A')
                        outliers_display['z_score'] = outliers_display['z_score'].apply(lambda x: f'{x:.1f}')
                        outliers_display.columns = ['Fornitore', 'Ente', 'Importo', 'Z-Score']
                        show_dataframe(outliers_display, use_container_width=True, height=300)
                        st.warning(f"⚠️ Trovati {len(outliers)} outlier su {len(filtered_df)} gare ({len(outliers)/len(filtered_df)*100:.2f}%)")
                    else:
                        st.success("✅ Nessun outlier significativo rilevato")
                else:
                    st.info("Dati insufficienti per calcolare outlier")
            else:
                st.info("Colonna importo non disponibile")

        with col2:
            st.markdown("#### 📉 Distribuzione Sconti Anomali")
            if 'sconto' in filtered_df.columns and filtered_df['sconto'].notna().any():
                sconto_valid = filtered_df[filtered_df['sconto'].between(0, 100)]['sconto']
                if len(sconto_valid) > 10:
                    sconto_stats = sconto_valid.describe()
                    q1 = sconto_stats['25%']
                    q3 = sconto_stats['75%']
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    anomalous_sconti = filtered_df[(filtered_df['sconto'] < lower_bound) | (filtered_df['sconto'] > upper_bound)]

                    fig = px.histogram(filtered_df[filtered_df['sconto'].between(0, 100)], x='sconto', nbins=50,
                                      color_discrete_sequence=[CGL_BLUE])
                    fig.add_vline(x=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower bound")
                    fig.add_vline(x=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper bound")
                    fig.update_layout(height=300, xaxis_title='Sconto %', yaxis_title='Frequenza')
                    st.plotly_chart(fig, use_container_width=True, key="sconto_anomalies")

                    if len(anomalous_sconti) > 0:
                        st.info(f"📊 Sconti anomali: {len(anomalous_sconti)} gare fuori range [{lower_bound:.1f}%, {upper_bound:.1f}%]")
                else:
                    st.info("Dati sconto insufficienti per l'analisi")
            else:
                st.info("Colonna sconto non disponibile o vuota")

        # === NETWORK GRAPH INTERATTIVO ===
        st.markdown("---")
        st.markdown("### 🕸️ Network Graph Interattivo")
        st.markdown("*Visualizzazione grafo relazioni Enti-Fornitori*")

        # Opzioni del grafo
        col_net_opt1, col_net_opt2, col_net_opt3 = st.columns(3)
        with col_net_opt1:
            n_top_nodes = st.slider("Top nodi da visualizzare", 10, 50, 20, key="network_nodes")
        with col_net_opt2:
            min_gare_edge = st.slider("Min gare per connessione", 1, 10, 2, key="network_min_gare")
        with col_net_opt3:
            layout_type = st.selectbox("Layout grafo", ["Circolare", "Forza", "Random"], key="network_layout")

        # Prepara dati per il network
        top_enti = filtered_df.groupby(buyer_col_net, observed=True)[amount_col_net].sum().sort_values(ascending=False).head(n_top_nodes).index.tolist()
        top_fornitori = filtered_df.groupby(supplier_col_net, observed=True)[amount_col_net].sum().sort_values(ascending=False).head(n_top_nodes).index.tolist()

        # Filtra solo relazioni tra top nodi
        network_df = filtered_df[
            (filtered_df[buyer_col_net].isin(top_enti)) &
            (filtered_df[supplier_col_net].isin(top_fornitori))
        ].copy()

        # Aggrega per coppia
        edges_df = network_df.groupby([buyer_col_net, supplier_col_net], observed=True).agg({
            id_col_net: 'count',
            amount_col_net: 'sum'
        }).reset_index()
        edges_df.columns = ['ente', 'fornitore', 'n_gare', 'valore']
        edges_df = edges_df[edges_df['n_gare'] >= min_gare_edge]

        if len(edges_df) > 0:
            import math

            # Raccogli tutti i nodi unici
            all_nodes = list(set(edges_df['ente'].tolist() + edges_df['fornitore'].tolist()))
            n_nodes = len(all_nodes)

            # Assegna posizioni ai nodi
            node_positions = {}
            if layout_type == "Circolare":
                for i, node in enumerate(all_nodes):
                    angle = 2 * math.pi * i / n_nodes
                    node_positions[node] = (math.cos(angle), math.sin(angle))
            elif layout_type == "Random":
                import random
                random.seed(42)
                for node in all_nodes:
                    node_positions[node] = (random.uniform(-1, 1), random.uniform(-1, 1))
            else:  # Forza - bipartito
                enti_in_graph = [n for n in all_nodes if n in top_enti]
                fornitori_in_graph = [n for n in all_nodes if n in top_fornitori and n not in enti_in_graph]

                for i, node in enumerate(enti_in_graph):
                    y_pos = (i / max(1, len(enti_in_graph) - 1)) * 2 - 1 if len(enti_in_graph) > 1 else 0
                    node_positions[node] = (-0.8, y_pos)

                for i, node in enumerate(fornitori_in_graph):
                    y_pos = (i / max(1, len(fornitori_in_graph) - 1)) * 2 - 1 if len(fornitori_in_graph) > 1 else 0
                    node_positions[node] = (0.8, y_pos)

            # Crea traces per edges
            edge_x = []
            edge_y = []

            for _, row in edges_df.iterrows():
                if row['ente'] in node_positions and row['fornitore'] in node_positions:
                    x0, y0 = node_positions[row['ente']]
                    x1, y1 = node_positions[row['fornitore']]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(150,150,150,0.5)'),
                hoverinfo='none',
                mode='lines'
            )

            # Crea traces per nodi
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []

            # Calcola metriche per dimensione nodi
            node_values = {}
            for node in all_nodes:
                if node in top_enti:
                    node_values[node] = network_df[network_df[buyer_col_net] == node][amount_col_net].sum()
                else:
                    node_values[node] = network_df[network_df[supplier_col_net] == node][amount_col_net].sum()

            max_val = max(node_values.values()) if node_values else 1

            for node in all_nodes:
                if node in node_positions:
                    x, y = node_positions[node]
                    node_x.append(x)
                    node_y.append(y)

                    short_name = node[:25] + "..." if len(node) > 25 else node
                    val = node_values.get(node, 0)
                    val_display = f"€{val/1e6:.1f}M" if val >= 1e6 else f"€{val/1e3:.0f}K"
                    node_text.append(f"{short_name}<br>{val_display}")

                    if node in top_enti:
                        node_color.append(CGL_BLUE)
                    else:
                        node_color.append(CGL_GREEN)

                    size = 15 + (node_values.get(node, 0) / max_val) * 35
                    node_size.append(size)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="bottom center",
                textfont=dict(size=8),
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line=dict(width=2, color='white')
                )
            )

            fig_network = go.Figure(data=[edge_trace, node_trace])

            fig_network.update_layout(
                title=f"Network Enti-Fornitori ({len(all_nodes)} nodi, {len(edges_df)} connessioni)",
                title_font_size=14,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                plot_bgcolor='rgba(248,249,250,1)'
            )

            fig_network.add_annotation(
                x=0.02, y=0.98, xref="paper", yref="paper",
                text="🔵 Enti Appaltanti",
                showarrow=False, font=dict(size=10),
                bgcolor="white", borderpad=4
            )
            fig_network.add_annotation(
                x=0.02, y=0.93, xref="paper", yref="paper",
                text="🟢 Fornitori",
                showarrow=False, font=dict(size=10),
                bgcolor="white", borderpad=4
            )

            st.plotly_chart(fig_network, use_container_width=True)

            # Statistiche network
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("🔵 Enti nel grafo", len([n for n in all_nodes if n in top_enti]))
            with col_stat2:
                st.metric("🟢 Fornitori nel grafo", len([n for n in all_nodes if n not in top_enti]))
            with col_stat3:
                st.metric("🔗 Connessioni", len(edges_df))
            with col_stat4:
                avg_connections = edges_df['n_gare'].mean()
                st.metric("📊 Media gare/conn.", f"{avg_connections:.1f}")

            # Top connessioni
            st.markdown("#### 🔝 Top 10 Connessioni più Forti")
            top_edges = edges_df.nlargest(10, 'n_gare').copy()
            top_edges['ente'] = top_edges['ente'].str[:35]
            top_edges['fornitore'] = top_edges['fornitore'].str[:30]
            top_edges['valore'] = top_edges['valore'].apply(lambda x: f"€{x/1e6:.1f}M")
            top_edges.columns = ['Ente', 'Fornitore', 'N. Gare', 'Valore Totale']
            show_dataframe(top_edges, use_container_width=True, hide_index=True)

        else:
            st.info("Dati insufficienti per il network graph. Prova a ridurre il minimo gare per connessione.")

# ==================== TAB 15: AI CHARTS ====================
if tab15:
  with tab15:
    st.subheader("🤖 Visualizzazioni AI")
    st.markdown("**Workflow 2-step**: Prima analizziamo la tua richiesta, poi generiamo il grafico")

    # Check API key - prima da session, poi da env
    api_key_available = get_openai_api_key() is not None

    # Se non c'è API key, mostra input per inserirla
    if not api_key_available:
        st.warning("⚠️ Per usare questa funzione, inserisci la tua OpenAI API Key")

        with st.expander("🔑 Inserisci API Key", expanded=True):
            st.markdown("""
            La chiave verrà salvata **solo per questa sessione** e non sarà memorizzata sul server.
            Puoi ottenere una chiave su [platform.openai.com](https://platform.openai.com/api-keys)
            """)

            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="La chiave inizia con 'sk-' ed è lunga circa 50 caratteri"
            )

            if st.button("✅ Salva per questa sessione", type="primary"):
                if api_key_input and api_key_input.startswith("sk-"):
                    st.session_state.openai_api_key = api_key_input
                    st.success("✅ API Key salvata! La pagina si aggiornerà...")
                    st.rerun()
                else:
                    st.error("❌ API Key non valida. Deve iniziare con 'sk-'")

    if get_openai_api_key():
        # Show available columns in expander
        with st.expander("📋 Colonne disponibili nel dataset", expanded=False):
            cols_info = filtered_df.dtypes.to_frame('tipo').reset_index()
            cols_info.columns = ['Colonna', 'Tipo']
            show_dataframe(cols_info, use_container_width=True, hide_index=True)

        # Examples - UI migliorata con cards
        st.markdown("### 💡 Esempi di richieste")
        examples = [
            ("🥧", "Torta categorie", "Grafico a torta delle categorie per valore totale"),
            ("📈", "Trend mensile", "Andamento mensile aggiudicazioni per anno"),
            ("🎯", "Scatter sconto", "Scatter plot importo vs sconto colorato per regione"),
            ("🏆", "Top fornitori", "Top 10 aggiudicatari per numero gare vinte"),
            ("🔥", "Heatmap tempo", "Heatmap anno/mese con valore medio aggiudicazioni"),
            ("🌳", "Mappa regioni", "Treemap regioni con valore totale e numero gare")
        ]

        # Layout 2 righe x 3 colonne per visibilità migliore
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)

        for i, (icon, label, full_prompt) in enumerate(examples):
            col = row1_cols[i] if i < 3 else row2_cols[i - 3]
            with col:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {BRAND_SURFACE}, #ffffff); padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 3px solid {BRAND_GREEN};">
                    <span style="font-size: 1.2em;">{icon}</span> <strong>{label}</strong>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Usa questo", key=f"example_{i}", use_container_width=True):
                    st.session_state['ai_prompt'] = full_prompt
                    st.session_state.pop('ai_analysis', None)  # Reset analysis
                    st.rerun()

        st.markdown("---")

        # Input prompt
        prompt = st.text_area(
            "✏️ Descrivi il grafico che vuoi creare:",
            value=st.session_state.get('ai_prompt', ''),
            height=80,
            placeholder="Es: Quante gare ha vinto AEC ogni anno e in quali regioni?"
        )

        # Step 1: Analyze
        col_btn1, col_btn2, col_space = st.columns([1, 1, 3])
        with col_btn1:
            analyze_btn = st.button("🔍 1. Analizza", type="secondary", use_container_width=True)
        with col_btn2:
            generate_btn = st.button("🚀 2. Genera", type="primary", use_container_width=True, disabled=('ai_analysis' not in st.session_state))

        # Get dataframe info for context
        df_info = f"""
Colonne: {list(filtered_df.columns)}
Righe: {len(filtered_df)}
Tipi: {filtered_df.dtypes.to_dict()}
Colonne numeriche: {list(filtered_df.select_dtypes(include=[np.number]).columns)}
Esempio valori:
{filtered_df.head(3).to_string()}
"""

        # Step 1: Analysis
        if analyze_btn and prompt:
            with st.spinner("🔍 Analizzo la richiesta..."):
                analysis = analyze_prompt(prompt, df_info)
                st.session_state['ai_analysis'] = analysis
                st.session_state['ai_prompt_for_gen'] = prompt
                st.rerun()

        # Show analysis results
        if 'ai_analysis' in st.session_state and st.session_state.get('ai_analysis'):
            analysis = st.session_state['ai_analysis']

            if analysis.get('error'):
                st.error(f"❌ Errore nell'analisi: {analysis['error']}")
            else:
                st.success("✅ Analisi completata! Modifica i parametri se necessario, poi clicca **Genera**")

                # Editable analysis parameters
                with st.container():
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Tipo grafico editabile
                        chart_types_options = ['bar', 'line', 'scatter', 'pie', 'treemap', 'heatmap']
                        chart_types_labels = ['📊 Barre', '📈 Linee', '🎯 Scatter', '🥧 Torta', '🌳 Treemap', '🔥 Heatmap']
                        current_chart = analysis.get('chart_type', 'bar')
                        if current_chart not in chart_types_options:
                            current_chart = 'bar'
                        selected_chart = st.selectbox(
                            "📊 Tipo grafico",
                            options=chart_types_options,
                            format_func=lambda x: chart_types_labels[chart_types_options.index(x)],
                            index=chart_types_options.index(current_chart),
                            key="edit_chart_type"
                        )
                        # Update analysis
                        st.session_state['ai_analysis']['chart_type'] = selected_chart

                    with col2:
                        # Aggregazione editabile
                        agg_options = ['count', 'sum', 'mean']
                        agg_labels = ['Conteggio', 'Somma', 'Media']
                        current_agg = analysis.get('aggregation', 'count')
                        if current_agg not in agg_options:
                            current_agg = 'count'
                        selected_agg = st.selectbox(
                            "⚙️ Aggregazione",
                            options=agg_options,
                            format_func=lambda x: agg_labels[agg_options.index(x)],
                            index=agg_options.index(current_agg),
                            key="edit_aggregation"
                        )
                        st.session_state['ai_analysis']['aggregation'] = selected_agg

                    with col3:
                        # Colonne selezionabili
                        available_cols = list(filtered_df.columns)
                        current_cols = analysis.get('columns', [])
                        # Filtra colonne valide
                        valid_cols = [c for c in current_cols if c in available_cols]
                        selected_cols = st.multiselect(
                            "📋 Colonne da usare",
                            options=available_cols,
                            default=valid_cols[:5] if valid_cols else [],
                            key="edit_columns"
                        )
                        st.session_state['ai_analysis']['columns'] = selected_cols

                # Valori/filtri trovati
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔎 Valori/Pattern trovati:**")
                    values = analysis.get('values', {})
                    search_patterns = analysis.get('search_patterns', {})
                    all_patterns = {**values, **search_patterns}
                    if all_patterns:
                        for k, v in list(all_patterns.items())[:4]:
                            st.markdown(f"- `{k}`: **{v}**")
                    else:
                        st.caption("Nessun valore specifico")

                with col2:
                    st.markdown(f"**📝 Descrizione:** {analysis.get('chart_description', 'N/A')}")

                # Sezione modifica con LLM
                st.markdown("---")
                st.markdown("**🔧 Vuoi modificare l'analisi?**")

                col_comment, col_btn = st.columns([4, 1])
                with col_comment:
                    modification = st.text_input(
                        "Scrivi cosa vuoi cambiare:",
                        placeholder="Es: usa solo dati 2023, raggruppa per regione invece che anno, cambia in grafico a torta...",
                        key="ai_modification"
                    )
                with col_btn:
                    modify_btn = st.button("🔄 Modifica", key="modify_analysis_btn", use_container_width=True)

                if modify_btn and modification:
                    with st.spinner("🔄 Modifico l'analisi..."):
                        # Chiedi all'LLM di modificare l'analisi
                        modify_prompt = f"""Analisi attuale:
{json.dumps(analysis, indent=2)}

Richiesta di modifica dell'utente: {modification}

Rispondi con il JSON aggiornato (stesso formato) applicando le modifiche richieste."""

                        modified = analyze_prompt(modify_prompt, df_info)
                        if modified and not modified.get('error'):
                            st.session_state['ai_analysis'] = modified
                            st.rerun()
                        else:
                            st.error("Errore nella modifica, riprova")

                # Bottone reset
                if st.button("🗑️ Reset analisi", key="reset_analysis"):
                    st.session_state.pop('ai_analysis', None)
                    st.rerun()

        # Step 2: Generate
        if generate_btn:
            analysis = st.session_state.get('ai_analysis')
            gen_prompt = st.session_state.get('ai_prompt_for_gen', prompt)

            with st.spinner("🤖 Genero il grafico con AI..."):
                code = generate_chart_code(gen_prompt, df_info, analysis)

                if code and not code.startswith("# Errore"):
                    st.session_state['last_ai_code'] = code
                    st.session_state['last_ai_prompt'] = gen_prompt

                    # Show generated code
                    with st.expander("📝 Codice generato", expanded=False):
                        st.code(code, language="python")

                    # Execute code
                    fig, error = execute_chart_code(code, filtered_df)

                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="ai_generated_chart")

                        # Save to favorites button
                        col1, col2, col3 = st.columns([1, 1, 3])
                        with col1:
                            if st.button("⭐ Salva nei Preferiti", key="save_ai_fav"):
                                chart_config = {
                                    'type': 'ai_generated',
                                    'prompt': gen_prompt,
                                    'code': code,
                                    'title': gen_prompt[:50] + "..." if len(gen_prompt) > 50 else gen_prompt
                                }
                                chart_id = add_favorite(chart_config)
                                st.success(f"✅ Salvato! ID: {chart_id}")
                        with col2:
                            if st.button("🔄 Rigenera", key="regenerate_ai"):
                                st.session_state.pop('ai_analysis', None)
                                st.rerun()
                    else:
                        st.error(f"❌ Errore nell'esecuzione: {error}")
                        with st.expander("🔧 Codice con errore"):
                            st.code(code, language="python")
                        st.info("💡 Prova a riformulare la richiesta con più dettagli")
                else:
                    st.error("❌ Errore nella generazione del codice")
                    if code:
                        st.code(code)

# ==================== TAB 16: PREFERITI ====================
if tab16:
  with tab16:
    st.subheader("⭐ I Miei Grafici Preferiti")

    favorites = load_favorites()

    if not favorites:
        st.info("🔍 Non hai ancora salvato nessun grafico nei preferiti")
        st.markdown("""
        ### Come salvare grafici:
        1. **Grafici AI**: Vai al tab **🤖 AI Charts**, genera un grafico e clicca su **⭐ Salva nei Preferiti**
        2. **Grafici Standard**: Su alcuni grafici trovi il bottone **☆** per salvarli
        3. Torna qui per vedere tutti i tuoi grafici salvati!
        """)
    else:
        # Filter by type
        ai_favs = [f for f in favorites if f.get('type') == 'ai_generated']
        std_favs = [f for f in favorites if f.get('type') == 'standard']

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Totale", len(favorites))
        col_info2.metric("AI Generated", len(ai_favs))
        col_info3.metric("Standard", len(std_favs))

        # Layout selection
        layout = st.radio("Layout", ["🔲 Griglia", "📜 Lista"], horizontal=True, key="fav_layout")

        if layout == "🔲 Griglia":
            # Grid layout - 2 columns
            cols = st.columns(2)
            for i, fav in enumerate(favorites):
                with cols[i % 2]:
                    with st.container():
                        fav_type = "🤖" if fav.get('type') == 'ai_generated' else "📊"
                        st.markdown(f"#### {fav_type} {fav.get('title', 'Grafico')[:35]}")
                        st.caption(f"Creato: {fav.get('created_at', 'N/A')[:10]}")

                        # Show filters if present
                        if fav.get('filters'):
                            filters_str = " | ".join([f"**{k}**: {v}" for k, v in fav['filters'].items()])
                            st.markdown(f"🔍 Filtri: {filters_str}")

                        # Render based on type
                        if fav.get('type') == 'ai_generated' and fav.get('code'):
                            fig, error = execute_chart_code(fav['code'], filtered_df)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"fav_chart_{fav.get('id', i)}")
                            else:
                                st.warning(f"Errore: {error}")
                        elif fav.get('type') == 'standard' and fav.get('fig_json'):
                            try:
                                import plotly.io as pio
                                fig = pio.from_json(fav['fig_json'])
                                st.plotly_chart(fig, use_container_width=True, key=f"fav_chart_{fav.get('id', i)}")
                            except Exception as e:
                                st.warning(f"Errore nel caricare il grafico: {e}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Rimuovi", key=f"del_{fav.get('id', i)}"):
                                remove_favorite(fav.get('id'))
                                st.rerun()
                        with col2:
                            if fav.get('code'):
                                with st.expander("📝 Codice"):
                                    st.code(fav.get('code', ''), language="python")
                            elif fav.get('description'):
                                st.caption(fav.get('description', ''))

                        st.markdown("---")
        else:
            # List layout
            for i, fav in enumerate(favorites):
                fav_type = "🤖 AI" if fav.get('type') == 'ai_generated' else "📊 Standard"
                with st.expander(f"{fav_type}: {fav.get('title', 'Grafico')[:50]}", expanded=i==0):
                    if fav.get('prompt'):
                        st.caption(f"Prompt: {fav.get('prompt', 'N/A')}")
                    if fav.get('description'):
                        st.caption(f"Descrizione: {fav.get('description', 'N/A')}")
                    # Show filters if present
                    if fav.get('filters'):
                        filters_str = " | ".join([f"**{k}**: {v}" for k, v in fav['filters'].items()])
                        st.markdown(f"🔍 Filtri applicati: {filters_str}")
                    st.caption(f"Creato: {fav.get('created_at', 'N/A')}")

                    # Render based on type
                    if fav.get('type') == 'ai_generated' and fav.get('code'):
                        fig, error = execute_chart_code(fav['code'], filtered_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"fav_list_{fav.get('id', i)}")
                        else:
                            st.warning(f"Errore: {error}")
                    elif fav.get('type') == 'standard' and fav.get('fig_json'):
                        try:
                            import plotly.io as pio
                            fig = pio.from_json(fav['fig_json'])
                            st.plotly_chart(fig, use_container_width=True, key=f"fav_list_{fav.get('id', i)}")
                        except Exception as e:
                            st.warning(f"Errore nel caricare il grafico: {e}")

                    if st.button("🗑️ Rimuovi", key=f"del_list_{fav.get('id', i)}"):
                        remove_favorite(fav.get('id'))
                        st.rerun()

        # Export all favorites
        st.markdown("---")
        st.download_button(
            "📥 Esporta Preferiti (JSON)",
            data=json.dumps(favorites, indent=2, default=str),
            file_name="grafici_preferiti.json",
            mime="application/json"
        )

# ==================== TAB 17: CHAT AI ====================
if tab17:
  with tab17:
    st.subheader("💬 Chat AI - Interroga i Dati")
    st.markdown("Chiedi qualsiasi cosa sui dati delle gare. Ti chiederò conferma prima di analizzare!")

    # Initialize states
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'pending_search' not in st.session_state:
        st.session_state['pending_search'] = None
    if 'selected_suppliers' not in st.session_state:
        st.session_state['selected_suppliers'] = []

    # Trova colonne dinamiche
    supplier_col = next((c for c in filtered_df.columns if c.lower() in ['supplier_name', 'aggiudicatario']), None)
    category_col = next((c for c in filtered_df.columns if c.lower() in ['category', 'categoria']), None)
    amount_col = next((c for c in filtered_df.columns if c.lower() in ['award_amount', 'importo_aggiudicazione']), None)

    # Chat input
    chat_input = st.chat_input("Fai una domanda sui dati delle gare...")

    # Display chat history
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg.get('chart'):
                st.plotly_chart(msg['chart'], use_container_width=True)

    # STEP 1: Se c'è una ricerca pendente, mostra opzioni di selezione
    if st.session_state.get('pending_search'):
        search_info = st.session_state['pending_search']
        st.info(f"🔍 **Ricerca per:** {', '.join(search_info['keywords'])}")

        st.markdown("### Seleziona i fornitori che vuoi analizzare:")

        # Mostra fornitori trovati con checkbox
        found_suppliers = search_info.get('found_suppliers', {})

        if found_suppliers:
            for keyword, suppliers in found_suppliers.items():
                if suppliers:
                    st.markdown(f"**Risultati per '{keyword}':** ({len(suppliers)} trovati)")
                    cols = st.columns(2)
                    for i, (sup_name, sup_info) in enumerate(suppliers.items()):
                        with cols[i % 2]:
                            checked = st.checkbox(
                                f"{sup_name[:50]}...",
                                key=f"sup_{hash(sup_name)}",
                                help=f"Gare: {sup_info['n_gare']}, Valore: €{sup_info['valore']/1e6:.2f}M"
                            )
                            if checked and sup_name not in st.session_state['selected_suppliers']:
                                st.session_state['selected_suppliers'].append(sup_name)
                            elif not checked and sup_name in st.session_state['selected_suppliers']:
                                st.session_state['selected_suppliers'].remove(sup_name)

                            st.caption(f"📊 {sup_info['n_gare']} gare | €{sup_info['valore']/1e6:.2f}M | {sup_info['periodo']}")

            st.markdown("---")
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("✅ Analizza Selezionati", type="primary", disabled=len(st.session_state['selected_suppliers'])==0):
                    # Procedi con l'analisi
                    selected = st.session_state['selected_suppliers']
                    original_query = search_info['original_query']

                    # Genera report dettagliato
                    report = f"## 📊 Analisi per: {', '.join([s[:30] for s in selected])}\n\n"

                    for sup_name in selected:
                        sup_data = filtered_df[filtered_df[supplier_col] == sup_name]
                        n_gare = len(sup_data)
                        valore = sup_data[amount_col].sum() if amount_col else 0

                        report += f"### 🏢 {sup_name}\n"
                        report += f"- **Gare totali:** {n_gare}\n"
                        report += f"- **Valore totale:** €{valore/1e6:.2f}M\n"

                        # Categorie
                        if category_col and category_col in sup_data.columns:
                            cats = sup_data.groupby(category_col, observed=True).size().nlargest(5)
                            report += f"- **Categorie principali:**\n"
                            for cat, count in cats.items():
                                report += f"  - {cat}: {count} gare\n"

                        # Andamento per anno
                        if 'anno' in sup_data.columns:
                            yearly = sup_data.groupby('anno', observed=True).agg({
                                supplier_col: 'count',
                                amount_col: 'sum' if amount_col else 'count'
                            }).reset_index()
                            yearly.columns = ['Anno', 'N_Gare', 'Valore']
                            yearly = yearly[yearly['Anno'] >= 2018].sort_values('Anno')

                            if len(yearly) > 0:
                                report += f"\n**📈 Andamento per anno:**\n"
                                report += "| Anno | Gare | Valore |\n|------|------|--------|\n"
                                for _, row in yearly.iterrows():
                                    report += f"| {int(row['Anno'])} | {int(row['N_Gare'])} | €{row['Valore']/1e6:.2f}M |\n"

                        report += "\n---\n"

                    # Salva nella history
                    st.session_state['chat_history'].append({'role': 'assistant', 'content': report})

                    # Pulisci stato
                    st.session_state['pending_search'] = None
                    st.session_state['selected_suppliers'] = []
                    st.rerun()

            with col2:
                if st.button("❌ Annulla"):
                    st.session_state['pending_search'] = None
                    st.session_state['selected_suppliers'] = []
                    st.rerun()
        else:
            st.warning("Nessun fornitore trovato per questa ricerca.")
            if st.button("🔙 Torna indietro"):
                st.session_state['pending_search'] = None
                st.rerun()

    # STEP 0: Nuova domanda
    elif chat_input:
        st.session_state['chat_history'].append({'role': 'user', 'content': chat_input})
        query_lower = chat_input.lower()

        # Estrai potenziali nomi di aziende
        keywords_to_search = ['city', 'green', 'light', 'aec', 'enel', 'a2a', 'iren', 'hera', 'edison', 'eni', 'sorgenia', 'axpo', 'engie', 'citelum', 'siemens', 'philips', 'gewiss']
        found_keywords = [kw for kw in keywords_to_search if kw in query_lower]

        # Aggiungi parole lunghe dalla query
        extra_words = [w for w in query_lower.replace("'", " ").split()
                      if len(w) > 4 and w not in ['come', 'negli', 'ultimi', 'anni', 'quanto', 'quali', 'della', 'delle',
                                                   'nella', 'nelle', 'gare', 'aggiudicatario', 'fornitore', 'andamento',
                                                   'hanno', 'vinto', 'categoria', 'categorie', 'quale']]
        found_keywords.extend(extra_words)
        found_keywords = list(set(found_keywords))

        if found_keywords and supplier_col:
            # Cerca fornitori nel database
            all_suppliers = filtered_df[supplier_col].dropna().unique()
            found_suppliers = {}

            for keyword in found_keywords:
                matches = {}
                for sup in all_suppliers:
                    if keyword in str(sup).lower():
                        sup_data = filtered_df[filtered_df[supplier_col] == sup]
                        n_gare = len(sup_data)
                        valore = sup_data[amount_col].sum() if amount_col else 0
                        anni = sup_data['anno'].dropna().unique() if 'anno' in sup_data.columns else []
                        periodo = f"{int(min(anni))}-{int(max(anni))}" if len(anni) > 0 else "N/A"

                        matches[sup] = {
                            'n_gare': n_gare,
                            'valore': valore,
                            'periodo': periodo
                        }

                # Ordina per numero gare e prendi top 10
                sorted_matches = dict(sorted(matches.items(), key=lambda x: x[1]['n_gare'], reverse=True)[:10])
                if sorted_matches:
                    found_suppliers[keyword] = sorted_matches

            if found_suppliers:
                # Salva ricerca pendente
                st.session_state['pending_search'] = {
                    'keywords': found_keywords,
                    'found_suppliers': found_suppliers,
                    'original_query': chat_input
                }
                st.rerun()
            else:
                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'content': f"❌ Non ho trovato fornitori nel database che corrispondono a: {', '.join(found_keywords)}\n\nProva con un altro nome o controlla l'ortografia."
                })
                st.rerun()
        else:
            # Domanda generica senza ricerca fornitore
            with st.chat_message('assistant'):
                with st.spinner("🤔 Analizzo..."):
                    df_summary = f"""
DATI: {len(filtered_df):,} gare, €{filtered_df[amount_col].sum()/1e9:.2f}B totali
TOP 5 FORNITORI: {filtered_df.groupby(supplier_col, observed=True)[amount_col].sum().nlargest(5).to_dict() if supplier_col and amount_col else 'N/A'}
TOP 5 CATEGORIE: {filtered_df.groupby(category_col, observed=True)[amount_col].sum().nlargest(5).to_dict() if category_col and amount_col else 'N/A'}
"""
                    chat_prompt = f"Domanda: {chat_input}\n\nDati disponibili:\n{df_summary}\n\nRispondi in italiano, brevemente."
                    response = call_responses_api(chat_prompt, "Esperto gare pubbliche. Risposte brevi e precise.")

                    if response:
                        st.markdown(response)
                        st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
                    else:
                        st.error("Errore nella risposta")

    # Quick questions
    st.markdown("---")
    st.markdown("**💡 Domande rapide:**")
    quick_cols = st.columns(4)
    quick_questions = [
        "Gare Edison ultimi anni",
        "Andamento Enel",
        "Gare A2A per categoria",
        "Fornitori illuminazione"
    ]
    for i, q in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state['chat_history'].append({'role': 'user', 'content': q})
                st.rerun()

    # Clear chat
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Pulisci Chat", key="clear_chat"):
            st.session_state['chat_history'] = []
            st.session_state['pending_search'] = None
            st.session_state['selected_suppliers'] = []
            st.rerun()

# ==================== TAB 18: PREDIZIONI ML ====================
if tab18:
  with tab18:
    st.subheader("🔮 Predizione Vincitori con ML")
    st.markdown("Analisi predittiva basata su dati storici per stimare probabilità di vittoria")

    # Prepare ML data
    @st.cache_data
    def prepare_ml_data(df):
        """Prepara dati per ML analysis"""
        # Trova le colonne giuste (case-insensitive)
        supplier_col = next((c for c in df.columns if c.lower() in ['supplier_name', 'aggiudicatario']), None)
        category_col = next((c for c in df.columns if c.lower() in ['category', 'categoria']), None)
        cig_col = next((c for c in df.columns if c.lower() == 'cig'), None)
        amount_col = next((c for c in df.columns if c.lower() in ['award_amount', 'importo_aggiudicazione']), None)

        if not supplier_col or not category_col:
            return None, None, None, None, None

        # Supplier stats
        agg_dict = {cig_col: 'count'} if cig_col else {}
        if amount_col:
            agg_dict[amount_col] = ['sum', 'mean']
        if 'sconto' in df.columns:
            agg_dict['sconto'] = 'mean'
        if 'anno' in df.columns:
            agg_dict['anno'] = ['min', 'max']

        supplier_stats = df.groupby(supplier_col, observed=True).agg(agg_dict).reset_index()
        supplier_stats.columns = ['supplier', 'n_gare', 'valore_tot', 'valore_medio', 'sconto_medio', 'anno_min', 'anno_max'][:len(supplier_stats.columns)]

        if 'anno_min' in supplier_stats.columns and 'anno_max' in supplier_stats.columns:
            supplier_stats['anni_attivita'] = supplier_stats['anno_max'] - supplier_stats['anno_min'] + 1
            supplier_stats['gare_per_anno'] = supplier_stats['n_gare'] / supplier_stats['anni_attivita'].replace(0, 1)

        # Category performance
        cat_perf = df.groupby([supplier_col, category_col], observed=True).size().reset_index(name='wins_in_cat')

        return supplier_stats, cat_perf, supplier_col, category_col, amount_col

    result = prepare_ml_data(filtered_df)
    supplier_stats, cat_perf, supplier_col, category_col, amount_col = result if result[0] is not None else (None, None, None, None, None)

    if supplier_stats is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🎯 Simula Gara")

            # Category selection - trova la colonna giusta
            cat_col_name = category_col or 'categoria'
            categories = sorted(filtered_df[cat_col_name].dropna().unique().tolist()) if cat_col_name in filtered_df.columns else []
            selected_cat = st.selectbox("📦 Categoria gara", options=categories[:50] if categories else ['N/A'])

            # Region - trova la colonna giusta
            region_col = next((c for c in filtered_df.columns if c.lower() == 'regione'), None)
            regions = sorted(filtered_df[region_col].dropna().unique().tolist()) if region_col else []
            selected_region = st.selectbox("📍 Regione", options=['Tutte'] + regions)

            # Value range
            value_range = st.slider(
                "💰 Valore gara (€K)",
                min_value=10,
                max_value=10000,
                value=(100, 1000),
                step=50
            )

            predict_btn = st.button("🔮 Calcola Predizioni", type="primary", use_container_width=True)

        with col2:
            if predict_btn and selected_cat != 'N/A':
                st.markdown("### 📊 Probabilità Vincita")

                with st.spinner("🧠 Calcolo predizioni..."):
                    # Filter relevant data usando le colonne dinamiche
                    cat_data = filtered_df[filtered_df[cat_col_name] == selected_cat].copy()
                    if selected_region != 'Tutte' and region_col and region_col in cat_data.columns:
                        cat_data = cat_data[cat_data[region_col] == selected_region]

                    # Calculate win probability based on historical performance
                    if len(cat_data) > 0:
                        # Trova colonne dinamicamente
                        cig_col = next((c for c in cat_data.columns if c.lower() == 'cig'), cat_data.columns[0])
                        amt_col = amount_col or 'award_amount'

                        agg_dict = {cig_col: 'count'}
                        if amt_col in cat_data.columns:
                            agg_dict[amt_col] = 'sum'
                        if 'sconto' in cat_data.columns:
                            agg_dict['sconto'] = 'mean'

                        cat_winners = cat_data.groupby(supplier_col, observed=True).agg(agg_dict).reset_index()
                        col_names = ['Fornitore', 'Gare Vinte']
                        if amt_col in cat_data.columns:
                            col_names.append('Valore Totale')
                        if 'sconto' in cat_data.columns:
                            col_names.append('Sconto Medio')
                            cat_winners.columns = col_names[:len(cat_winners.columns)]
    
                            # Calculate probability score
                            total_wins = cat_winners['Gare Vinte'].sum()
                            if not total_wins or total_wins <= 0:
                                cat_winners['Prob. Base (%)'] = 0.0
                            else:
                                cat_winners['Prob. Base (%)'] = (cat_winners['Gare Vinte'] / total_wins * 100).round(1)
    
                            # Adjust for value range compatibility (se c'è la colonna valore)
                        if 'Valore Totale' in cat_winners.columns:
                            value_mid = (value_range[0] + value_range[1]) / 2 * 1000
                            cat_winners['Valore Medio'] = cat_winners['Valore Totale'] / cat_winners['Gare Vinte']
                            max_val = cat_winners['Valore Medio'].max()
                            if max_val > 0:
                                cat_winners['Score Valore'] = 1 - abs(cat_winners['Valore Medio'] - value_mid) / max_val
                                cat_winners['Score Valore'] = cat_winners['Score Valore'].clip(0.3, 1)
                            else:
                                cat_winners['Score Valore'] = 1.0
                        else:
                            cat_winners['Score Valore'] = 1.0
                            cat_winners['Valore Medio'] = 0

                            # Final probability
                            cat_winners['🎯 Probabilità (%)'] = (cat_winners['Prob. Base (%)'] * cat_winners['Score Valore']).round(1)
                            cat_winners = cat_winners.nlargest(10, '🎯 Probabilità (%)')
    
                            # Display results
                            for _, row in cat_winners.head(5).iterrows():
                                prob_raw = row['🎯 Probabilità (%)']
                                try:
                                    prob = float(prob_raw)
                                    if not np.isfinite(prob):
                                        prob = 0.0
                                except Exception:
                                    prob = 0.0
    
                                color = "🟢" if prob > 15 else "🟡" if prob > 5 else "🔴"
                                try:
                                    valore_medio = float(row.get('Valore Medio', 0) or 0)
                                    if not np.isfinite(valore_medio):
                                        valore_medio = 0.0
                                except Exception:
                                    valore_medio = 0.0
                                valore_str = f"€{valore_medio/1e6:.2f}M" if valore_medio > 0 else "N/A"
                                st.markdown(f"""
                                **{color} {row['Fornitore'][:40]}**
                                - Probabilità: **{prob}%**
                                - Gare vinte in categoria: {row['Gare Vinte']}
                                - Valore medio: {valore_str}
                                """)
                                progress_val = max(0.0, min(prob / 30.0, 1.0))
                                st.progress(progress_val)
    
                            # Chart
                            fig = px.bar(
                                cat_winners.head(10),
                                x='🎯 Probabilità (%)',
                                y='Fornitore',
                                orientation='h',
                                title=f'Top 10 Probabili Vincitori - {selected_cat[:30]}',
                                color='🎯 Probabilità (%)',
                                color_continuous_scale=BRAND_CONTINUOUS_SCALE
                            )
                            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Dati insufficienti per questa categoria/regione")

            # Historical accuracy note
            st.info("""
            💡 **Come funziona:**
            - Analisi storica delle vittorie per categoria
            - Ponderazione per range di valore simili
            - Score basato su frequenza vittorie e compatibilità budget

            ⚠️ Le predizioni sono indicative e basate su dati storici
            """)

        # Supplier Deep Dive
        st.markdown("---")
        st.markdown("### 🔍 Analisi Fornitore Specifico")

        col1, col2 = st.columns([1, 2])
        with col1:
            top_suppliers = supplier_stats.nlargest(100, 'n_gare')['supplier'].tolist()
            selected_supplier = st.selectbox("Seleziona fornitore", options=top_suppliers)

        if selected_supplier:
            supplier_data = filtered_df[filtered_df[supplier_col] == selected_supplier]

            with col2:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Gare Vinte", f"{len(supplier_data):,}")
                amt_col_val = amount_col if amount_col and amount_col in supplier_data.columns else 'award_amount'
                if amt_col_val in supplier_data.columns:
                    m2.metric("Valore Totale", f"€{supplier_data[amt_col_val].sum()/1e6:.1f}M")
                else:
                    m2.metric("Valore Totale", "N/A")
                m3.metric("Sconto Medio", f"{supplier_data['sconto'].mean():.1f}%" if 'sconto' in supplier_data.columns and supplier_data['sconto'].notna().any() else "N/A")
                if 'anno' in supplier_data.columns and supplier_data['anno'].notna().any():
                    m4.metric("Anni Attività", f"{int(supplier_data['anno'].max() - supplier_data['anno'].min() + 1)}")
                else:
                    m4.metric("Anni Attività", "N/A")

            # Category breakdown
            cig_col_bd = next((c for c in supplier_data.columns if c.lower() == 'cig'), supplier_data.columns[0])
            agg_bd = {cig_col_bd: 'count'}
            if amt_col_val in supplier_data.columns:
                agg_bd[amt_col_val] = 'sum'

            cat_breakdown = supplier_data.groupby(cat_col_name, observed=True).agg(agg_bd).reset_index()
            if amt_col_val in supplier_data.columns:
                cat_breakdown.columns = ['Categoria', 'N. Gare', 'Valore']
            else:
                cat_breakdown.columns = ['Categoria', 'N. Gare']
            cat_breakdown = cat_breakdown.nlargest(5, 'N. Gare')

            fig = px.pie(
                cat_breakdown,
                values='N. Gare',
                names='Categoria',
                title=f'Categorie principali - {selected_supplier[:30]}'
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Dati insufficienti per l'analisi ML. Verifica che il dataset contenga le colonne necessarie.")

# ==================== TAB 19: MAPPA PRO ====================
if tab19:
  with tab19:
    st.subheader("🗺️ Mappa Interattiva Avanzata")
    st.markdown("Esplora i dati geografici con visualizzazioni avanzate")

    # Helper per trovare colonne dinamicamente (come negli altri tab)
    def get_col_map(df, candidates):
        for col in candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    # Identifica colonne chiave
    regione_col = get_col_map(filtered_df, ['regione', 'Regione', 'buyer_region'])
    amount_col = get_col_map(filtered_df, ['importo_aggiudicazione', 'award_amount', 'tender_amount'])
    id_col = get_col_map(filtered_df, ['chiave', 'CIG', 'ocid', 'id'])
    categoria_col = get_col_map(filtered_df, ['categoria', '_categoria', 'category'])
    supplier_col = get_col_map(filtered_df, ['aggiudicatario', 'supplier_name', 'award_supplier_name'])
    comune_col = get_col_map(filtered_df, ['comune', 'citta', 'buyer_locality', 'city'])

    # Map type selection
    map_type = st.radio(
        "Tipo visualizzazione",
        ["🌡️ Heatmap Valore", "📍 Cluster Città", "🎯 Drill-down Regioni", "⏱️ Animazione Temporale"],
        horizontal=True
    )

    # Prepare geo data
    @st.cache_data
    def get_region_coords():
        """Italian regions coordinates"""
        return {
            'Lombardia': (45.47, 9.19), 'Lazio': (41.89, 12.48), 'Campania': (40.85, 14.25),
            'Sicilia': (37.60, 14.02), 'Veneto': (45.44, 11.88), 'Emilia-Romagna': (44.49, 11.34),
            'Piemonte': (45.07, 7.69), 'Puglia': (41.13, 16.87), 'Toscana': (43.77, 11.25),
            'Calabria': (38.91, 16.59), 'Sardegna': (39.22, 9.12), 'Liguria': (44.41, 8.93),
            'Marche': (43.62, 13.52), 'Abruzzo': (42.35, 13.40), 'Friuli-Venezia Giulia': (45.64, 13.80),
            'Trentino-Alto Adige': (46.07, 11.12), 'Umbria': (42.86, 12.64), 'Basilicata': (40.64, 15.80),
            'Molise': (41.56, 14.67), "Valle d'Aosta": (45.74, 7.32)
        }

    region_coords = get_region_coords()

    if map_type == "🌡️ Heatmap Valore":
        if regione_col and amount_col:
            region_data = filtered_df.groupby(regione_col, observed=True).agg({
                amount_col: 'sum'
            }).reset_index()
            region_data['N_Gare'] = filtered_df.groupby(regione_col, observed=True).size().values
            region_data.columns = ['Regione', 'Valore', 'N_Gare']

            # Add coordinates
            region_data['lat'] = region_data['Regione'].map(lambda x: region_coords.get(x, (42, 12))[0])
            region_data['lon'] = region_data['Regione'].map(lambda x: region_coords.get(x, (42, 12))[1])
            region_data['Valore_B'] = region_data['Valore'] / 1e9

            fig = px.density_map(
                region_data,
                lat='lat',
                lon='lon',
                z='Valore_B',
                radius=50,
                center={'lat': 42.0, 'lon': 12.5},
                zoom=4.5,
                title='Heatmap Valore Gare per Regione (€B)',
                color_continuous_scale=BRAND_CONTINUOUS_SCALE
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Stats table
            show_dataframe(
                region_data[['Regione', 'N_Gare', 'Valore_B']]
                .rename(columns={'Valore_B': 'Valore (€B)'})
                .sort_values('Valore (€B)', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(f"Dati regione non disponibili. Colonne trovate: regione={regione_col}, importo={amount_col}")

    elif map_type == "📍 Cluster Città":
        if comune_col and amount_col:
            # Aggregazione per comune
            city_data = filtered_df.groupby(comune_col, observed=True).agg({
                amount_col: 'sum'
            }).reset_index()
            city_data['N_Gare'] = filtered_df.groupby(comune_col, observed=True).size().values
            city_data.columns = ['Città', 'Valore', 'N_Gare']
            city_data['Valore_M'] = city_data['Valore'] / 1e6

            # Size slider
            min_gare = st.slider("Minimo gare per visualizzare", 1, 100, 10)
            city_filtered = city_data[city_data['N_Gare'] >= min_gare].nlargest(50, 'Valore')

            if len(city_filtered) > 0:
                # Top cities bar chart (senza mappa dato che non abbiamo lat/lon)
                fig = px.bar(
                    city_filtered.head(20),
                    x='Valore_M',
                    y='Città',
                    orientation='h',
                    title=f'Top 20 Città per Valore (>= {min_gare} gare)',
                    color='N_Gare',
                    color_continuous_scale='Viridis',
                    labels={'Valore_M': 'Valore (€M)', 'N_Gare': 'N. Gare'}
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                show_dataframe(city_filtered.head(30), use_container_width=True, hide_index=True)
            else:
                st.info(f"Nessuna città con >= {min_gare} gare")
        else:
            st.warning("Dati città non disponibili")

    elif map_type == "🎯 Drill-down Regioni":
        if regione_col:
            # Region selector
            regioni_list = sorted(filtered_df[regione_col].dropna().unique().tolist())
            if len(regioni_list) > 0:
                selected_region = st.selectbox(
                    "Seleziona Regione per drill-down",
                    options=regioni_list
                )

                if selected_region:
                    region_df = filtered_df[filtered_df[regione_col] == selected_region]
                    st.info(f"📊 {len(region_df):,} gare in {selected_region}")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Top categories in region
                        if categoria_col and amount_col:
                            cat_region = region_df.groupby(categoria_col, observed=True).agg({
                                amount_col: 'sum'
                            }).reset_index()
                            cat_region['N_Gare'] = region_df.groupby(categoria_col, observed=True).size().values
                            cat_region.columns = ['Categoria', 'Valore', 'N_Gare']
                            cat_region = cat_region.nlargest(10, 'Valore')

                            fig = px.bar(
                                cat_region,
                                x='Valore',
                                y='Categoria',
                                orientation='h',
                                title=f'Top 10 Categorie - {selected_region}',
                                color='Valore',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Top suppliers in region
                        if supplier_col and amount_col:
                            sup_region = region_df.groupby(supplier_col, observed=True).agg({
                                amount_col: 'sum'
                            }).reset_index()
                            sup_region['N_Gare'] = region_df.groupby(supplier_col, observed=True).size().values
                            sup_region.columns = ['Fornitore', 'Valore', 'N_Gare']
                            sup_region = sup_region.nlargest(10, 'Valore')

                            fig = px.bar(
                                sup_region,
                                x='Valore',
                                y='Fornitore',
                                orientation='h',
                                title=f'Top 10 Fornitori - {selected_region}',
                                color='N_Gare',
                                color_continuous_scale='Greens'
                            )
                            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)

                    # Trend temporale regione
                    if 'anno' in region_df.columns and amount_col:
                        trend_data = region_df[region_df['anno'].between(2018, 2025)]
                        if len(trend_data) > 0:
                            trend_region = trend_data.groupby('anno', observed=True).agg({
                                amount_col: 'sum'
                            }).reset_index()
                            trend_region['N_Gare'] = trend_data.groupby('anno', observed=True).size().values
                            trend_region.columns = ['Anno', 'Valore', 'N_Gare']

                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            fig.add_trace(
                                go.Bar(x=trend_region['Anno'], y=trend_region['Valore']/1e6, name='Valore (€M)'),
                                secondary_y=False
                            )
                            fig.add_trace(
                                go.Scatter(x=trend_region['Anno'], y=trend_region['N_Gare'], name='N. Gare', mode='lines+markers'),
                                secondary_y=True
                            )
                            fig.update_layout(title=f'Trend Temporale - {selected_region}', height=350)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nessuna regione disponibile nei dati filtrati")
        else:
            st.warning("Colonna regione non disponibile")

    else:  # Animazione Temporale
        if 'anno' in filtered_df.columns and regione_col and amount_col:
            st.markdown("### ⏱️ Animazione Evoluzione Gare nel Tempo")

            # Filtra anni sensati
            anim_df = filtered_df[filtered_df['anno'].between(2018, 2025)]

            if len(anim_df) > 0:
                # Prepare data by year and region
                anim_data = anim_df.groupby(['anno', regione_col], observed=True).agg({
                    amount_col: 'sum'
                }).reset_index()
                anim_data['N_Gare'] = anim_df.groupby(['anno', regione_col], observed=True).size().values
                anim_data.columns = ['Anno', 'Regione', 'Valore', 'N_Gare']

                # Add coordinates
                anim_data['lat'] = anim_data['Regione'].map(lambda x: region_coords.get(x, (42, 12))[0])
                anim_data['lon'] = anim_data['Regione'].map(lambda x: region_coords.get(x, (42, 12))[1])
                anim_data['Valore_M'] = anim_data['Valore'] / 1e6
                anim_data['Anno'] = anim_data['Anno'].astype(int)

                fig = px.scatter_map(
                    anim_data,
                    lat='lat',
                    lon='lon',
                    size='Valore_M',
                    color='Valore_M',
                    animation_frame='Anno',
                    hover_name='Regione',
                    center={'lat': 42.0, 'lon': 12.5},
                    zoom=4.5,
                    title='Evoluzione Valore Gare per Regione (2018-2025)',
                    color_continuous_scale='Plasma',
                    size_max=50
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Summary stats
                year_totals = anim_data.groupby('Anno', observed=True)['Valore'].sum() / 1e9
                fig2 = px.area(
                    x=year_totals.index,
                    y=year_totals.values,
                    title='Valore Totale Gare per Anno (€B)',
                    labels={'x': 'Anno', 'y': 'Valore (€B)'}
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Nessun dato nel range 2018-2025")
        else:
            st.warning(f"Dati insufficienti per animazione. anno={('anno' in filtered_df.columns)}, regione={regione_col}, importo={amount_col}")

# ==================== FOOTER ====================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📅 **Periodo dati**: 2015-2025")
with col2:
    st.markdown(f"📊 **Record totali**: {len(raw_df):,}".replace(",", "."))
with col3:
    st.markdown("🔄 **Fonte**: OCDS Italia")

st.markdown("*Dashboard generata automaticamente* | © 2024")
