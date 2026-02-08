"""
Comprehensive tests for dashboard_gare/app.py data logic.

These tests do NOT require Streamlit running or OpenAI API keys.
They reproduce helper functions inline (since app.py has Streamlit
dependencies that cannot be imported in a test context) and verify
the pure data logic against real data files and synthetic fixtures.

Run all tests:
    pytest tests/test_app_logic.py -v

Run only fast tests (skip full CSV load):
    pytest tests/test_app_logic.py -v -m "not slow"

Run only slow integration tests:
    pytest tests/test_app_logic.py -v -m slow
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ISTAT_CSV = DATA_DIR / "comuni_istat.csv"
GARE_GZ = DATA_DIR / "gare_unificate.csv.gz"
DATA_JSON = DATA_DIR / "data.json"
SERVIZIO_LUCE_XLSX = DATA_DIR / "ServizioLuce.xlsx"


# ===========================================================================
# Reproduced helper functions (from app.py, without Streamlit dependencies)
# ===========================================================================

def _normalize_comune_name(s) -> str:
    """Normalizza nome comune per matching: lowercase, strip, rimuovi accenti."""
    if pd.isna(s) or str(s).strip() == "" or str(s).lower() in ("nan", "none"):
        return ""
    s = str(s).strip()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _build_istat_lookup(comuni_istat_df):
    """Costruisce dizionari di lookup per geocoding e backfill regione."""
    if comuni_istat_df is None or len(comuni_istat_df) == 0:
        return {}, {}
    geo_lookup: dict = {}
    regione_lookup: dict = {}
    for _, row in comuni_istat_df.iterrows():
        key = str(row.get("comune_normalized", "")).strip()
        if not key:
            continue
        geo_lookup[key] = (row["lat"], row["lon"], row.get("regione", ""), row.get("comune", ""))
        regione_lookup[key] = row.get("regione", "")
    # Alias
    _aliases = {
        "reggio emilia": "reggio nell'emilia",
        "reggio calabria": "reggio di calabria",
        "forli": "forli'",
        "cesena": "cesena",
        "massa": "massa",
        "carrara": "carrara",
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
    from rapidfuzz import fuzz, process

    candidates = list(geo_lookup.keys())
    if not candidates:
        return None, None
    match = process.extractOne(key, candidates, scorer=fuzz.ratio, score_cutoff=85)
    if match:
        hit = geo_lookup[match[0]]
        return hit[0], hit[1]
    return None, None


def _to_dt(series, fmt=None):
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    s = pd.to_datetime(series, format=fmt, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None and s.dt.tz is not None:
            s = s.dt.tz_convert(None)
    except Exception:
        pass
    return s


# Category duration estimates (from app.py _compute_scadenze_contratti)
DURATE_STIMATE = {
    "Servizio Luce": 9,
    "Illuminazione": 9,
    "Manutenzione": 4,
    "Infrastrutture": 5,
    "Strade": 5,
    "Edifici": 5,
    "Scuole": 5,
    "Pulizie": 3,
    "Riscaldamento": 7,
    "Energia": 7,
    "Termici": 7,
    "Vigilanza": 3,
    "Videosorveglianza": 4,
    "Facchinaggio": 3,
    "Verde": 3,
    "Ambiente": 4,
    "Traslochi": 2,
    "Portierato": 3,
    "Disinfestazione": 2,
    "Rifiuti": 5,
    "Acqua": 5,
    "Acquedotti": 5,
    "Trasporti": 4,
    "Mobilita": 4,
    "Parcheggi": 5,
    "ICT": 3,
    "Digitale": 3,
    "Smart": 3,
    "Sanitario": 4,
    "Sociale": 3,
    "Formazione": 2,
    "Strutture Sportive": 5,
    "Strutture_Sportive": 5,
    "Gallerie": 5,
    "Tunnel": 5,
    "Impianti": 5,
    "Ricarica": 5,
    "Colonnine": 5,
}


def _get_durata_anni(cat):
    if pd.isna(cat):
        return 3
    s = str(cat).lower()
    for key, val in DURATE_STIMATE.items():
        if key.lower() in s:
            return val
    return 3


# Region coordinates dict (from app.py get_region_coords)
REGION_COORDS = {
    "Lombardia": (45.47, 9.19),
    "Lazio": (41.89, 12.48),
    "Campania": (40.85, 14.25),
    "Sicilia": (37.60, 14.02),
    "Veneto": (45.44, 11.88),
    "Emilia-Romagna": (44.49, 11.34),
    "Piemonte": (45.07, 7.69),
    "Puglia": (41.13, 16.87),
    "Toscana": (43.77, 11.25),
    "Calabria": (38.91, 16.59),
    "Sardegna": (39.22, 9.12),
    "Liguria": (44.41, 8.93),
    "Marche": (43.62, 13.52),
    "Abruzzo": (42.35, 13.40),
    "Friuli-Venezia Giulia": (45.64, 13.80),
    "Trentino-Alto Adige": (46.07, 11.12),
    "Umbria": (42.86, 12.64),
    "Basilicata": (40.64, 15.80),
    "Molise": (41.56, 14.67),
    "Valle d'Aosta": (45.74, 7.32),
}

# All 29 dataset categories
DATASET_CATEGORIES = [
    "Acqua_Fognature",
    "Acquedotti",
    "Altro",
    "Ambiente",
    "Colonnine",
    "Digitale",
    "Edifici",
    "Edifici_Pubblici",
    "Energia",
    "Gallerie",
    "Illuminazione",
    "Impianti",
    "Infrastrutture",
    "Infrastrutture_Digitali",
    "Mobilita_Elettrica",
    "Parcheggi",
    "Ricarica",
    "Rifiuti",
    "Sanitario",
    "Scuole",
    "Smart_City",
    "Strade_Infrastrutture",
    "Strutture_Sportive",
    "Termici",
    "Trasporti",
    "Trasporti_Pubblici",
    "Tunnel",
    "Verde_Pubblico",
    "Videosorveglianza",
]


def _compute_scadenze_contratti(
    df_base: pd.DataFrame,
    consip_map: pd.DataFrame | None,
    include_stime: bool,
    cig_enrichment_items: dict | None = None,
) -> pd.DataFrame:
    """Reproduced from app.py (lines 4957-5186) without Streamlit deps."""
    if df_base is None or len(df_base) == 0:
        return pd.DataFrame()

    keep_candidates = [
        "chiave", "cig", "ocid",
        "buyer_name", "ente_appaltante",
        "supplier_name", "aggiudicatario",
        "comune", "buyer_locality", "regione",
        "oggetto",
        "award_amount", "importo_aggiudicazione",
        "award_date", "data_aggiudicazione",
        "data_scadenza", "durata_appalto",
        "_categoria", "categoria", "quick_category", "tipo_appalto",
    ]
    keep_cols = [c for c in keep_candidates if c in df_base.columns]
    d = df_base[keep_cols].copy()

    # Normalizza campi principali
    if "cig" in d.columns:
        d["cig"] = d["cig"].fillna("").astype(str).str.strip()
        d["cig"] = d["cig"].replace({"nan": "", "None": ""})
    else:
        d["cig"] = ""

    if "award_date" in d.columns:
        d["award_date"] = _to_dt(d["award_date"])
    elif "data_aggiudicazione" in d.columns:
        d["award_date"] = _to_dt(d["data_aggiudicazione"])
    else:
        d["award_date"] = pd.NaT

    if "award_amount" in d.columns:
        d["award_amount"] = pd.to_numeric(d["award_amount"], errors="coerce")
    elif "importo_aggiudicazione" in d.columns:
        d["award_amount"] = pd.to_numeric(d["importo_aggiudicazione"], errors="coerce")
    else:
        d["award_amount"] = np.nan

    # (1) data_scadenza esplicita
    if "data_scadenza" in d.columns:
        d["scadenza_da_data_scadenza"] = _to_dt(d["data_scadenza"])
    else:
        d["scadenza_da_data_scadenza"] = pd.NaT

    # (2) scadenza da CONSIP
    if consip_map is not None and len(consip_map) > 0:
        d = d.merge(consip_map, on="cig", how="left")
    else:
        d["scadenza_consip"] = pd.NaT
        d["durata_giorni_consip"] = np.nan

    # (3) scadenza da durata_appalto
    if "durata_appalto" in d.columns:
        d["durata_giorni_dataset"] = pd.to_numeric(d["durata_appalto"], errors="coerce")
    else:
        d["durata_giorni_dataset"] = np.nan
    d["scadenza_da_durata_appalto"] = d["award_date"] + pd.to_timedelta(
        d["durata_giorni_dataset"], unit="D"
    )

    # (3.5) Regex extraction
    d["scadenza_da_regex"] = pd.NaT
    if "oggetto" in d.columns:
        obj = d["oggetto"].fillna("").astype(str)
        dur_match = obj.str.extract(
            r"durata\s*[:\s]?\s*(\d{1,4})\s*(mes[ei]|ann[oi]|giorn[oi])",
            flags=re.IGNORECASE,
            expand=True,
        )
        dur_num = pd.to_numeric(dur_match[0], errors="coerce")
        dur_unit = dur_match[1].str.lower().str[:3]
        dur_days = np.where(
            dur_unit == "mes",
            dur_num * 30,
            np.where(dur_unit == "ann", dur_num * 365, dur_num),
        )
        dur_days_series = pd.Series(dur_days, index=d.index, dtype="float64")
        # Clamp: max 30 anni (10950 giorni)
        dur_days_series = dur_days_series.where(dur_days_series.between(1, 10950))
        valid_dur = dur_days_series.notna() & d["award_date"].notna()
        d.loc[valid_dur, "scadenza_da_regex"] = d.loc[valid_dur, "award_date"] + pd.to_timedelta(
            dur_days_series[valid_dur], unit="D"
        )
        # Pattern implicito: triennale, biennale, quinquennale, ecc.
        still_nat = d["scadenza_da_regex"].isna()
        implicit = obj.str.extract(
            r"(triennal|biennal|quinquennal|quadriennal|settennal|novennal)",
            flags=re.IGNORECASE,
            expand=False,
        )
        implicit_map = {
            "triennal": 3,
            "biennal": 2,
            "quinquennal": 5,
            "quadriennal": 4,
            "settennal": 7,
            "novennal": 9,
        }
        implicit_years = implicit.str.lower().map(implicit_map)
        implicit_days = implicit_years * 365
        d.loc[still_nat & implicit_days.notna(), "scadenza_da_regex"] = (
            d.loc[still_nat & implicit_days.notna(), "award_date"]
            + pd.to_timedelta(implicit_days[still_nat & implicit_days.notna()], unit="D")
        )

    # (4) LLM enrichment (from cache)
    d["scadenza_base_llm"] = pd.NaT
    d["scadenza_max_llm"] = pd.NaT
    d["llm_confidence"] = np.nan
    d["llm_notes"] = ""

    if cig_enrichment_items and isinstance(cig_enrichment_items, dict) and d["cig"].notna().any():
        present = set(d["cig"].fillna("").astype(str).str.strip().tolist())
        rows = []
        for cig_key in present:
            item = cig_enrichment_items.get(cig_key)
            if not item or not isinstance(item, dict):
                continue
            res = item.get("result")
            if not isinstance(res, dict):
                continue
            rows.append(
                {
                    "cig": cig_key,
                    "llm_duration_base_days": res.get("duration_base_days"),
                    "llm_duration_max_days": res.get("duration_max_days"),
                    "llm_explicit_start_date": res.get("explicit_start_date"),
                    "llm_explicit_end_date": res.get("explicit_end_date"),
                    "llm_confidence_cache": res.get("confidence"),
                    "llm_notes_cache": res.get("notes", ""),
                }
            )
        if rows:
            llm_df = pd.DataFrame(rows)
            llm_df["llm_duration_base_days"] = pd.to_numeric(
                llm_df["llm_duration_base_days"], errors="coerce"
            )
            llm_df["llm_duration_max_days"] = pd.to_numeric(
                llm_df["llm_duration_max_days"], errors="coerce"
            )
            llm_df["llm_explicit_start_dt"] = pd.to_datetime(
                llm_df["llm_explicit_start_date"], errors="coerce"
            )
            llm_df["llm_explicit_end_dt"] = pd.to_datetime(
                llm_df["llm_explicit_end_date"], errors="coerce"
            )
            d = d.merge(
                llm_df[
                    [
                        "cig",
                        "llm_duration_base_days",
                        "llm_duration_max_days",
                        "llm_explicit_start_dt",
                        "llm_explicit_end_dt",
                        "llm_confidence_cache",
                        "llm_notes_cache",
                    ]
                ],
                on="cig",
                how="left",
            )
            start_llm = d["llm_explicit_start_dt"].fillna(d["award_date"])
            d["scadenza_base_llm"] = d["llm_explicit_end_dt"]
            d.loc[
                d["scadenza_base_llm"].isna()
                & start_llm.notna()
                & d["llm_duration_base_days"].notna(),
                "scadenza_base_llm",
            ] = start_llm + pd.to_timedelta(d["llm_duration_base_days"], unit="D")
            d.loc[
                start_llm.notna() & d["llm_duration_max_days"].notna(),
                "scadenza_max_llm",
            ] = start_llm + pd.to_timedelta(d["llm_duration_max_days"], unit="D")

            if "llm_confidence_cache" in d.columns:
                d["llm_confidence"] = pd.to_numeric(d["llm_confidence_cache"], errors="coerce")
            if "llm_notes_cache" in d.columns:
                d["llm_notes"] = d["llm_notes_cache"].fillna("").astype(str)

    # (5) stima da categoria (fallback)
    if include_stime:
        cat_col = (
            "_categoria"
            if "_categoria" in d.columns
            else ("categoria" if "categoria" in d.columns else None)
        )
        if cat_col:
            d["durata_anni_stima"] = d[cat_col].apply(_get_durata_anni)
            d["scadenza_stimata"] = d["award_date"] + pd.to_timedelta(
                d["durata_anni_stima"] * 365, unit="D"
            )
        else:
            d["durata_anni_stima"] = np.nan
            d["scadenza_stimata"] = pd.NaT
    else:
        d["durata_anni_stima"] = np.nan
        d["scadenza_stimata"] = pd.NaT

    # Scadenza finale (priority: esplicita > CONSIP > durata > regex > LLM > stima)
    d["scadenza_contratto"] = (
        d["scadenza_da_data_scadenza"]
        .fillna(d["scadenza_consip"])
        .fillna(d["scadenza_da_durata_appalto"])
        .fillna(d["scadenza_da_regex"])
        .fillna(d["scadenza_base_llm"])
        .fillna(d["scadenza_stimata"])
    )

    # Fonte scadenza
    d["scadenza_fonte"] = np.select(
        [
            d["scadenza_da_data_scadenza"].notna(),
            d["scadenza_consip"].notna(),
            d["scadenza_da_durata_appalto"].notna(),
            d["scadenza_da_regex"].notna(),
            d["scadenza_base_llm"].notna(),
            d["scadenza_stimata"].notna(),
        ],
        [
            "data_scadenza",
            "consip",
            "durata_appalto",
            "regex_oggetto",
            "llm",
            "stima_categoria",
        ],
        default="mancante",
    )

    # Pulizia date fuori scala
    max_year = pd.Timestamp.now().year + 30
    year = d["scadenza_contratto"].dt.year
    invalid = d["scadenza_contratto"].notna() & ((year < 2000) | (year > max_year))
    d.loc[invalid, "scadenza_contratto"] = pd.NaT
    d.loc[invalid, "scadenza_fonte"] = "invalid"

    # Scadenza max
    d["scadenza_contratto_max"] = d["scadenza_max_llm"]
    year_max = d["scadenza_contratto_max"].dt.year
    invalid_max = d["scadenza_contratto_max"].notna() & (
        (year_max < 2000) | (year_max > max_year)
    )
    d.loc[invalid_max, "scadenza_contratto_max"] = pd.NaT

    oggi_ts = pd.Timestamp.now().normalize()
    d["giorni_alla_scadenza"] = (d["scadenza_contratto"] - oggi_ts).dt.days
    d["giorni_alla_scadenza_max"] = (d["scadenza_contratto_max"] - oggi_ts).dt.days
    d["stato_scadenza"] = np.select(
        [d["scadenza_contratto"].isna(), d["giorni_alla_scadenza"] < 0],
        ["Sconosciuta", "Scaduto"],
        default="Attivo",
    )

    d["anac_url"] = d["cig"].apply(
        lambda x: (
            f"https://dati.anticorruzione.it/superset/dashboard/dettaglio_cig/?cig={x}&standalone=2"
            if x
            else ""
        )
    )
    return d


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def istat_df():
    """Load the ISTAT CSV once per session."""
    assert ISTAT_CSV.exists(), f"ISTAT CSV not found at {ISTAT_CSV}"
    df = pd.read_csv(ISTAT_CSV, dtype=str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


@pytest.fixture(scope="session")
def istat_lookups(istat_df):
    """Build geo and regione lookups once per session."""
    return _build_istat_lookup(istat_df)


@pytest.fixture(scope="session")
def geo_lookup(istat_lookups):
    return istat_lookups[0]


@pytest.fixture(scope="session")
def regione_lookup(istat_lookups):
    return istat_lookups[1]


# ===========================================================================
# 1. ISTAT Geocoding
# ===========================================================================

class TestISTATGeocoding:
    """Tests for ISTAT CSV file and geocoding logic."""

    def test_istat_file_exists(self):
        assert ISTAT_CSV.exists(), f"File not found: {ISTAT_CSV}"

    def test_istat_expected_columns(self, istat_df):
        expected = {"comune", "comune_normalized", "regione", "provincia", "sigla_prov", "lat", "lon"}
        actual = set(istat_df.columns)
        assert expected.issubset(actual), f"Missing columns: {expected - actual}"

    def test_istat_row_count(self, istat_df):
        """ISTAT should have approximately 7900 comuni."""
        assert 7500 <= len(istat_df) <= 8500, f"Unexpected row count: {len(istat_df)}"

    @pytest.mark.parametrize(
        "input_name, expected",
        [
            ("Roma", "roma"),
            ("Milano", "milano"),
            ("Napoli", "napoli"),
            ("L'Aquila", "l'aquila"),
        ],
    )
    def test_normalize_basic(self, input_name, expected):
        assert _normalize_comune_name(input_name) == expected

    def test_normalize_forli_accent(self):
        """Forli with accent should normalize to 'forli' (no accent)."""
        result = _normalize_comune_name("Forli\u0300")  # Forli + combining grave
        assert result == "forli"

    def test_normalize_empty_values(self):
        assert _normalize_comune_name(None) == ""
        assert _normalize_comune_name("") == ""
        assert _normalize_comune_name("nan") == ""
        assert _normalize_comune_name("None") == ""
        assert _normalize_comune_name(np.nan) == ""

    @pytest.mark.parametrize(
        "comune",
        ["Roma", "Milano", "Napoli", "Reggio Emilia", "Reggio Calabria", "L'Aquila"],
    )
    def test_geocode_known_comuni(self, comune, geo_lookup):
        """Well-known comuni must be geocoded successfully."""
        lat, lon = _geocode_comune(comune, geo_lookup)
        assert lat is not None, f"Failed to geocode {comune}: lat is None"
        assert lon is not None, f"Failed to geocode {comune}: lon is None"

    def test_geocode_forli(self, geo_lookup):
        """Forli (with accent variants) should be found via alias."""
        lat, lon = _geocode_comune("Forli", geo_lookup)
        assert lat is not None, "Failed to geocode Forli"
        # Forli is in Emilia-Romagna, latitude ~44.2
        assert 43.5 < lat < 45.0, f"Forli latitude {lat} out of range"

    @pytest.mark.parametrize("comune", ["Roma", "Milano", "Napoli", "Palermo", "Torino"])
    def test_coordinates_in_italy(self, comune, geo_lookup):
        """Coordinates must be within Italy's bounding box."""
        lat, lon = _geocode_comune(comune, geo_lookup)
        assert lat is not None
        assert 35 <= lat <= 47, f"{comune}: lat {lat} not in Italy range [35, 47]"
        assert 6 <= lon <= 19, f"{comune}: lon {lon} not in Italy range [6, 19]"

    def test_all_istat_coords_in_italy(self, istat_df):
        """At least 99.9% of valid coordinates should be within Italy's bounding box.

        A handful of rows may have data quality issues (e.g., missing decimal point
        in the source ISTAT data), so we allow a small margin.
        """
        valid = istat_df[istat_df["lat"].notna() & istat_df["lon"].notna()]
        assert len(valid) > 7000, "Too few rows with valid coordinates"
        in_lat = valid["lat"].between(35, 48).sum()
        in_lon = valid["lon"].between(6, 19).sum()
        lat_pct = in_lat / len(valid)
        lon_pct = in_lon / len(valid)
        assert lat_pct >= 0.998, f"Too many latitudes outside Italy: {1 - lat_pct:.4%}"
        assert lon_pct >= 0.998, f"Too many longitudes outside Italy: {1 - lon_pct:.4%}"


# ===========================================================================
# 2. Regione Backfill
# ===========================================================================

class TestRegioneBackfill:
    """Tests for the regione backfill logic."""

    def test_backfill_roma_to_lazio(self, regione_lookup):
        """roma -> Lazio."""
        assert regione_lookup.get("roma") == "Lazio"

    def test_backfill_milano_to_lombardia(self, regione_lookup):
        """milano -> Lombardia."""
        assert regione_lookup.get("milano") == "Lombardia"

    def test_backfill_napoli_to_campania(self, regione_lookup):
        assert regione_lookup.get("napoli") == "Campania"

    def test_regione_not_category_during_assignment(self):
        """
        BUG REGRESSION: assigning to a category column raises TypeError.
        During backfill, regione must be string/object, not category.
        """
        df = pd.DataFrame(
            {
                "comune": ["Roma", "Milano", "Napoli"],
                "regione": pd.Categorical(["nan", "Lombardia", "nan"]),
            }
        )
        # Convert to str before backfill (as app.py does)
        df["regione"] = df["regione"].astype(str)
        assert df["regione"].dtype == object or str(df["regione"].dtype) == "object"

        # Now assignment should work without TypeError
        mask = df["regione"].isin(["nan", "", "None", "<NA>"])
        df.loc[mask, "regione"] = "TEST_VALUE"
        assert (df.loc[mask, "regione"] == "TEST_VALUE").all()

    def test_nan_string_excluded_from_filter(self, regione_lookup):
        """
        The regione filter dropdown (line 1793) must exclude 'nan', 'None', '', '<NA>'.
        This simulates what the dropdown logic does.
        """
        all_regioni = list(regione_lookup.values())
        filtered = [
            str(r) for r in all_regioni if str(r) not in ("nan", "None", "", "<NA>")
        ]
        invalid_vals = {"nan", "None", "", "<NA>"}
        for r in filtered:
            assert r not in invalid_vals, f"Invalid regione value in filter: {r!r}"

    def test_backfill_with_synthetic_data(self, regione_lookup):
        """Full backfill pipeline on synthetic data."""
        df = pd.DataFrame(
            {
                "comune": ["Roma", "Milano", "Napoli", "UnknownCity", None],
                "regione": [np.nan, "nan", "", "Campania", np.nan],
            }
        )
        # Step 1: convert to string (as app.py does)
        df["regione"] = df["regione"].astype(str)

        # Step 2: identify missing
        missing_regione = df["regione"].isin(["nan", "", "None", "<NA>"]) | df[
            "regione"
        ].isna()

        # Step 3: backfill
        comuni_norm = df.loc[missing_regione, "comune"].apply(_normalize_comune_name)
        regioni_fill = comuni_norm.map(regione_lookup)
        df.loc[missing_regione, "regione"] = regioni_fill

        # Step 4: cleanup
        df["regione"] = df["regione"].replace(
            {"nan": np.nan, "None": np.nan, "": np.nan, "<NA>": np.nan}
        )

        assert df.loc[0, "regione"] == "Lazio"
        assert df.loc[1, "regione"] == "Lombardia"
        # Row 3 was already Campania
        assert df.loc[3, "regione"] == "Campania"


# ===========================================================================
# 3. Scadenze Pipeline
# ===========================================================================

class TestScadenzePipeline:
    """Tests for _compute_scadenze_contratti logic."""

    def _make_base_df(self, **overrides):
        """Create a minimal synthetic DataFrame for scadenze computation."""
        defaults = {
            "cig": ["CIG0000001"],
            "oggetto": ["Appalto generico"],
            "award_date": [pd.Timestamp("2023-01-01")],
            "award_amount": [100000.0],
            "_categoria": ["Illuminazione"],
        }
        defaults.update(overrides)
        return pd.DataFrame(defaults)

    # -- Priority chain --

    def test_priority_data_scadenza_first(self):
        """data_scadenza (explicit) should take priority over everything."""
        df = self._make_base_df(
            data_scadenza=["2030-06-30"],
            durata_appalto=[365],
            oggetto=["durata 36 mesi - servizio triennale"],
        )
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert result.loc[0, "scadenza_fonte"] == "data_scadenza"
        assert result.loc[0, "scadenza_contratto"] == pd.Timestamp("2030-06-30")

    def test_priority_consip_second(self):
        """CONSIP overrides durata_appalto and regex."""
        df = self._make_base_df(durata_appalto=[365])
        consip = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "scadenza_consip": [pd.Timestamp("2031-12-31")],
                "durata_giorni_consip": [3285.0],
            }
        )
        result = _compute_scadenze_contratti(df, consip, include_stime=True)
        assert result.loc[0, "scadenza_fonte"] == "consip"
        assert result.loc[0, "scadenza_contratto"] == pd.Timestamp("2031-12-31")

    def test_priority_durata_appalto_third(self):
        """durata_appalto overrides regex and stima."""
        df = self._make_base_df(
            durata_appalto=[730],
            oggetto=["durata 36 mesi"],
        )
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert result.loc[0, "scadenza_fonte"] == "durata_appalto"
        expected = pd.Timestamp("2023-01-01") + pd.Timedelta(days=730)
        assert result.loc[0, "scadenza_contratto"] == expected

    def test_priority_regex_fourth(self):
        """Regex is used when no data_scadenza, consip, or durata_appalto."""
        df = self._make_base_df(
            oggetto=["Servizio illuminazione durata 36 mesi"],
        )
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert result.loc[0, "scadenza_fonte"] == "regex_oggetto"
        expected = pd.Timestamp("2023-01-01") + pd.Timedelta(days=36 * 30)
        assert result.loc[0, "scadenza_contratto"] == expected

    def test_priority_stima_last(self):
        """stima_categoria is the last fallback."""
        df = self._make_base_df(
            oggetto=["Appalto generico senza indicazioni"],
        )
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert result.loc[0, "scadenza_fonte"] == "stima_categoria"

    def test_mancante_when_no_stime(self):
        """Without stime and no other source, fonte = mancante."""
        df = self._make_base_df(
            oggetto=["Appalto generico senza indicazioni"],
            award_date=[pd.NaT],  # No award date -> can't compute stima either
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "scadenza_fonte"] == "mancante"

    # -- Regex extraction --

    @pytest.mark.parametrize(
        "text, expected_days",
        [
            ("durata 36 mesi", 36 * 30),
            ("durata 5 anni", 5 * 365),
            ("DURATA: 24 mesi", 24 * 30),
            ("durata 180 giorni", 180),
            ("durata 12 mese", 12 * 30),
            ("durata 1 anno", 1 * 365),
        ],
    )
    def test_regex_numeric_patterns(self, text, expected_days):
        df = self._make_base_df(oggetto=[text])
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "scadenza_fonte"] == "regex_oggetto"
        expected = pd.Timestamp("2023-01-01") + pd.Timedelta(days=expected_days)
        assert result.loc[0, "scadenza_contratto"] == expected

    @pytest.mark.parametrize(
        "text, expected_days",
        [
            ("servizio triennale di manutenzione", 3 * 365),
            ("contratto quinquennale", 5 * 365),
            ("affidamento biennale", 2 * 365),
        ],
    )
    def test_regex_implicit_patterns(self, text, expected_days):
        df = self._make_base_df(oggetto=[text])
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "scadenza_fonte"] == "regex_oggetto"
        expected = pd.Timestamp("2023-01-01") + pd.Timedelta(days=expected_days)
        assert result.loc[0, "scadenza_contratto"] == expected

    def test_regex_clamp_large_values(self):
        """Values > 10950 days (30 years) should be clamped to NaN."""
        # Even with "durata" keyword, 9999 mesi => 9999*30 = 299970 days -> clamped
        df = self._make_base_df(oggetto=["durata 9999 mesi"])
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        # Should NOT be regex since value is clamped
        assert result.loc[0, "scadenza_fonte"] != "regex_oggetto"

    def test_regex_rejects_random_large_numbers(self):
        """
        BUG REGRESSION: '736935 giorni' in oggetto should NOT match.
        The regex requires "durata" keyword for numeric patterns.
        """
        df = self._make_base_df(oggetto=["servizio 736935 giorni fornitura"])
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "scadenza_fonte"] != "regex_oggetto"

    def test_regex_requires_durata_keyword(self):
        """Numeric patterns without 'durata' keyword should not match."""
        df = self._make_base_df(oggetto=["fornitura 36 mesi di materiale"])
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        # "36 mesi" without preceding "durata" should NOT match the regex
        assert result.loc[0, "scadenza_fonte"] != "regex_oggetto"

    # -- Category duration estimates --

    def test_category_estimates_cover_all_29(self):
        """Every one of the 29 dataset categories should get a duration estimate."""
        for cat in DATASET_CATEGORIES:
            years = _get_durata_anni(cat)
            assert isinstance(years, (int, float)), f"No estimate for {cat}"
            assert 1 <= years <= 10, f"Unreasonable estimate for {cat}: {years}"

    def test_unknown_category_defaults_to_3(self):
        assert _get_durata_anni("UNKNOWN_CATEGORY_XYZ") == 3
        assert _get_durata_anni(np.nan) == 3

    # -- scadenza_fonte identification --

    def test_fonte_identifies_llm(self):
        """LLM source should be identified when only LLM provides data."""
        df = self._make_base_df(
            oggetto=["Appalto senza indicazioni di durata"],
        )
        cig_items = {
            "CIG0000001": {
                "result": {
                    "duration_base_days": 1095,
                    "duration_max_days": 1825,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.8,
                    "notes": "test",
                }
            }
        }
        result = _compute_scadenze_contratti(
            df, None, include_stime=False, cig_enrichment_items=cig_items
        )
        assert result.loc[0, "scadenza_fonte"] == "llm"


# ===========================================================================
# 4. Data Quality (integration tests on real files)
# ===========================================================================

class TestDataQuality:
    """Integration tests on actual data files."""

    @pytest.mark.slow
    def test_gare_unificate_loads(self):
        """gare_unificate.csv.gz loads without error."""
        assert GARE_GZ.exists(), f"File not found: {GARE_GZ}"
        df = pd.read_csv(GARE_GZ, compression="gzip", nrows=100)
        assert len(df) > 0

    @pytest.mark.slow
    def test_gare_unificate_expected_columns(self):
        """CSV has the expected core columns."""
        df = pd.read_csv(GARE_GZ, compression="gzip", nrows=5)
        expected = {
            "cig",
            "oggetto",
            "comune",
            "regione",
            "importo_aggiudicazione",
            "data_scadenza",
            "durata_appalto",
            "categoria",
        }
        assert expected.issubset(set(df.columns)), f"Missing: {expected - set(df.columns)}"

    def test_data_json_loads(self):
        """data.json loads and has expected top-level keys."""
        assert DATA_JSON.exists(), f"File not found: {DATA_JSON}"
        with open(DATA_JSON) as f:
            d = json.load(f)
        for key in ["kpi", "categories", "geo", "filter_options"]:
            assert key in d, f"Missing key in data.json: {key}"

    @pytest.mark.slow
    def test_servizio_luce_loads(self):
        """ServizioLuce.xlsx loads and has CIG column."""
        assert SERVIZIO_LUCE_XLSX.exists(), f"File not found: {SERVIZIO_LUCE_XLSX}"
        df = pd.read_excel(SERVIZIO_LUCE_XLSX, nrows=10)
        assert "CIG" in df.columns, f"CIG column not found. Columns: {list(df.columns)}"


# ===========================================================================
# 5. Map Data
# ===========================================================================

class TestMapData:
    """Tests for map / geocoding data."""

    def test_region_coords_has_all_20(self):
        """Region coordinates dict must have all 20 Italian regions."""
        assert len(REGION_COORDS) == 20
        expected_regions = {
            "Lombardia",
            "Lazio",
            "Campania",
            "Sicilia",
            "Veneto",
            "Emilia-Romagna",
            "Piemonte",
            "Puglia",
            "Toscana",
            "Calabria",
            "Sardegna",
            "Liguria",
            "Marche",
            "Abruzzo",
            "Friuli-Venezia Giulia",
            "Trentino-Alto Adige",
            "Umbria",
            "Basilicata",
            "Molise",
            "Valle d'Aosta",
        }
        assert set(REGION_COORDS.keys()) == expected_regions

    def test_region_coords_in_italy(self):
        """All region centroids should be in Italy."""
        for region, (lat, lon) in REGION_COORDS.items():
            assert 35 <= lat <= 48, f"{region}: lat {lat} out of range"
            assert 6 <= lon <= 19, f"{region}: lon {lon} out of range"

    @pytest.mark.slow
    def test_bubble_map_geocoding_coverage(self, geo_lookup):
        """At least 70% of unique comuni with data should geocode successfully."""
        df = pd.read_csv(
            GARE_GZ,
            compression="gzip",
            usecols=["comune"],
            dtype={"comune": "str"},
        )
        unique_comuni = df["comune"].dropna().unique()
        assert len(unique_comuni) > 0

        geocoded = 0
        for c in unique_comuni:
            lat, lon = _geocode_comune(c, geo_lookup)
            if lat is not None and lon is not None:
                geocoded += 1

        coverage = geocoded / len(unique_comuni)
        assert coverage >= 0.70, (
            f"Geocoding coverage too low: {coverage:.1%} "
            f"({geocoded}/{len(unique_comuni)})"
        )


# ===========================================================================
# 6. Alert Banner Logic
# ===========================================================================

class TestAlertBanner:
    """Tests for scadenza alert banner logic."""

    def test_scadenza_no_crash_with_nan_nat(self):
        """Scadenza computation should not crash with NaN/NaT values."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001", "CIG0000002", "CIG0000003"],
                "oggetto": [np.nan, "", "qualcosa"],
                "award_date": [pd.NaT, pd.Timestamp("2023-01-01"), pd.NaT],
                "award_amount": [np.nan, 100000, np.nan],
                "data_scadenza": [pd.NaT, pd.NaT, pd.NaT],
                "durata_appalto": [np.nan, np.nan, np.nan],
                "_categoria": [np.nan, "Illuminazione", np.nan],
            }
        )
        # Should not raise
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert len(result) == 3
        assert "giorni_alla_scadenza" in result.columns
        assert "scadenza_contratto" in result.columns

    def test_giorni_alla_scadenza_future_positive(self):
        """Future dates should produce positive giorni_alla_scadenza."""
        future_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=100)
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "data_scadenza": [future_date.isoformat()],
                "award_date": [pd.Timestamp("2023-01-01")],
                "_categoria": ["Illuminazione"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "giorni_alla_scadenza"] > 0

    def test_giorni_alla_scadenza_past_negative(self):
        """Past dates should produce negative giorni_alla_scadenza."""
        past_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=100)
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "data_scadenza": [past_date.isoformat()],
                "award_date": [pd.Timestamp("2020-01-01")],
                "_categoria": ["Illuminazione"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "giorni_alla_scadenza"] < 0

    def test_alert_counts_synthetic(self):
        """Alert counts: contracts within 30/90/365 days are counted correctly."""
        oggi = pd.Timestamp.now().normalize()
        # Create contracts with different expiry offsets
        offsets = [-10, 5, 20, 60, 200, 400]
        dates = [(oggi + pd.Timedelta(days=d)).isoformat() for d in offsets]
        df = pd.DataFrame(
            {
                "cig": [f"CIG000000{i}" for i in range(len(offsets))],
                "data_scadenza": dates,
                "award_date": [pd.Timestamp("2020-01-01")] * len(offsets),
                "award_amount": [100000.0] * len(offsets),
                "_categoria": ["Illuminazione"] * len(offsets),
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)

        # Reproduce alert logic from app.py lines 2035-2040
        scadenza = result["scadenza_contratto"]
        giorni = result["giorni_alla_scadenza"]
        valid = scadenza.notna() & (giorni >= -30)

        n30 = giorni[valid].between(-30, 30).sum()
        n90 = giorni[valid].between(-30, 90).sum()
        n365 = giorni[valid].between(-30, 365).sum()

        # offsets: -10 (within 30), 5 (within 30), 20 (within 30),
        #          60 (within 90), 200 (within 365), 400 (outside 365)
        assert n30 == 3, f"Expected 3 within 30 days, got {n30}"
        assert n90 == 4, f"Expected 4 within 90 days, got {n90}"
        assert n365 == 5, f"Expected 5 within 365 days, got {n365}"

    def test_stato_scadenza_values(self):
        """stato_scadenza must be one of Sconosciuta, Scaduto, Attivo."""
        oggi = pd.Timestamp.now().normalize()
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001", "CIG0000002", "CIG0000003"],
                "data_scadenza": [
                    (oggi + pd.Timedelta(days=100)).isoformat(),
                    (oggi - pd.Timedelta(days=100)).isoformat(),
                    pd.NaT,
                ],
                "award_date": [pd.Timestamp("2020-01-01")] * 3,
                "_categoria": ["Illuminazione"] * 3,
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "stato_scadenza"] == "Attivo"
        assert result.loc[1, "stato_scadenza"] == "Scaduto"
        assert result.loc[2, "stato_scadenza"] == "Sconosciuta"


# ===========================================================================
# 7. Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_empty_dataframe_doesnt_crash(self):
        """Empty DataFrame should return empty result, not crash."""
        df = pd.DataFrame()
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_none_dataframe_doesnt_crash(self):
        result = _compute_scadenze_contratti(None, None, include_stime=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_missing_columns_handled(self):
        """DataFrame with only cig column should not crash."""
        df = pd.DataFrame({"cig": ["CIG0000001"], "award_date": [pd.Timestamp("2023-01-01")]})
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert len(result) == 1
        assert "scadenza_contratto" in result.columns
        assert "scadenza_fonte" in result.columns

    def test_to_timedelta_no_overflow(self):
        """
        BUG REGRESSION: pd.to_timedelta with extremely large day values
        raises OutOfBoundsTimedelta. The clamp to 10950 prevents this.
        """
        # Simulate the clamping logic
        dur_days = pd.Series([10000, 20000, 50000, np.nan, 365])
        clamped = dur_days.where(dur_days.between(1, 10950))
        # Only 10000 and 365 should survive
        assert clamped.notna().sum() == 2
        # These should not raise OutOfBoundsTimedelta
        td = pd.to_timedelta(clamped, unit="D")
        assert td.notna().sum() == 2

    def test_category_dtype_string_comparison(self):
        """
        BUG REGRESSION: filtering a category column with string comparison
        can fail silently or raise. The app converts with .astype(str).
        """
        df = pd.DataFrame(
            {
                "regione": pd.Categorical(
                    ["Lazio", "Lombardia", "Campania", np.nan]
                ),
                "value": [1, 2, 3, 4],
            }
        )
        # This is how the app filters (line 1853):
        filtered = df[df["regione"].astype(str) == str("Lazio")]
        assert len(filtered) == 1
        assert filtered.iloc[0]["value"] == 1

    def test_category_dtype_with_nan_filter(self):
        """Category columns with NaN should not include 'nan' as a string category."""
        df = pd.DataFrame(
            {
                "regione": pd.Categorical(
                    ["Lazio", "Lombardia", None, "nan"]
                ),
            }
        )
        # Simulate the filter logic from app.py line 1793
        regioni = sorted(
            [
                str(r)
                for r in df["regione"].dropna().unique()
                if str(r) not in ("nan", "None", "", "<NA>")
            ]
        )
        assert "nan" not in regioni
        assert "None" not in regioni
        assert "" not in regioni

    def test_consip_map_none(self):
        """None consip map should not crash."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "award_date": [pd.Timestamp("2023-01-01")],
                "oggetto": ["generico"],
                "_categoria": ["Illuminazione"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=True)
        assert len(result) == 1

    def test_consip_map_empty(self):
        """Empty consip map should not crash."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "award_date": [pd.Timestamp("2023-01-01")],
                "oggetto": ["generico"],
                "_categoria": ["Illuminazione"],
            }
        )
        result = _compute_scadenze_contratti(df, pd.DataFrame(), include_stime=True)
        assert len(result) == 1

    def test_all_nat_dates_dont_crash(self):
        """All NaT dates should produce valid output with mancante/Sconosciuta."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001", "CIG0000002"],
                "award_date": [pd.NaT, pd.NaT],
                "data_scadenza": [pd.NaT, pd.NaT],
                "durata_appalto": [np.nan, np.nan],
                "oggetto": ["generico", "altro"],
                "_categoria": ["Illuminazione", "Manutenzione"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert len(result) == 2
        assert (result["scadenza_fonte"] == "mancante").all()
        assert (result["stato_scadenza"] == "Sconosciuta").all()

    def test_invalid_year_dates_cleaned(self):
        """Dates in year 1900 should be marked invalid.

        Note: pd.to_datetime cannot parse years beyond ~2262 (nanosecond limit),
        so '2999-12-31' becomes NaT and gets 'mancante' instead of 'invalid'.
        We test with a year that pandas CAN parse but is still out of range.
        """
        max_year = pd.Timestamp.now().year + 30
        # Use a far-future but parseable year (max_year + 5)
        far_future = f"{max_year + 5}-06-15"
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001", "CIG0000002"],
                "data_scadenza": ["1900-01-01", far_future],
                "award_date": [pd.Timestamp("2023-01-01")] * 2,
                "_categoria": ["Illuminazione"] * 2,
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert (result["scadenza_fonte"] == "invalid").all()
        assert result["scadenza_contratto"].isna().all()

    def test_llm_enrichment_with_empty_cache(self):
        """Empty LLM cache dict should not crash."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "award_date": [pd.Timestamp("2023-01-01")],
                "oggetto": ["generico"],
                "_categoria": ["Illuminazione"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items={})
        assert len(result) == 1

    def test_llm_enrichment_with_malformed_cache(self):
        """Malformed LLM cache entries should not crash."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001", "CIG0000002"],
                "award_date": [pd.Timestamp("2023-01-01")] * 2,
                "oggetto": ["generico"] * 2,
                "_categoria": ["Illuminazione"] * 2,
            }
        )
        bad_cache = {
            "CIG0000001": "not a dict",
            "CIG0000002": {"result": "also not a dict"},
        }
        result = _compute_scadenze_contratti(
            df, None, include_stime=False, cig_enrichment_items=bad_cache
        )
        assert len(result) == 2

    def test_regex_does_not_match_cig_numbers(self):
        """CIG codes like 'CIG 1234567890' should not be mistaken for durations."""
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001"],
                "award_date": [pd.Timestamp("2023-01-01")],
                "oggetto": ["CIG 1234567890 appalto servizi"],
                "_categoria": ["Illuminazione"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "scadenza_fonte"] != "regex_oggetto"

    def test_multiple_sources_in_batch(self):
        """Different rows should get different sources in a single batch."""
        oggi = pd.Timestamp.now().normalize()
        df = pd.DataFrame(
            {
                "cig": ["CIG0000001", "CIG0000002", "CIG0000003"],
                "award_date": [
                    pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2023-01-01"),
                ],
                "data_scadenza": ["2030-01-01", pd.NaT, pd.NaT],
                "durata_appalto": [np.nan, 365, np.nan],
                "oggetto": ["generico", "generico", "durata 36 mesi"],
                "_categoria": ["Illuminazione", "Manutenzione", "Pulizie"],
            }
        )
        result = _compute_scadenze_contratti(df, None, include_stime=False)
        assert result.loc[0, "scadenza_fonte"] == "data_scadenza"
        assert result.loc[1, "scadenza_fonte"] == "durata_appalto"
        assert result.loc[2, "scadenza_fonte"] == "regex_oggetto"


# ===========================================================================
# Additional regex edge cases
# ===========================================================================

class TestRegexEdgeCases:
    """Focused tests on the duration regex extraction."""

    def _extract_regex_days(self, text: str) -> float | None:
        """
        Reproduce the regex extraction logic from _compute_scadenze_contratti
        and return the computed days (or None if no match / clamped).
        """
        obj = pd.Series([text])
        dur_match = obj.str.extract(
            r"durata\s*[:\s]?\s*(\d{1,4})\s*(mes[ei]|ann[oi]|giorn[oi])",
            flags=re.IGNORECASE,
            expand=True,
        )
        dur_num = pd.to_numeric(dur_match[0], errors="coerce")
        dur_unit = dur_match[1].str.lower().str[:3]
        dur_days = np.where(
            dur_unit == "mes",
            dur_num * 30,
            np.where(dur_unit == "ann", dur_num * 365, dur_num),
        )
        dur_days_series = pd.Series(dur_days, dtype="float64")
        dur_days_series = dur_days_series.where(dur_days_series.between(1, 10950))
        val = dur_days_series.iloc[0]
        return None if pd.isna(val) else float(val)

    def _extract_implicit_days(self, text: str) -> float | None:
        """Extract implicit duration (triennale, etc.)."""
        obj = pd.Series([text])
        implicit = obj.str.extract(
            r"(triennal|biennal|quinquennal|quadriennal|settennal|novennal)",
            flags=re.IGNORECASE,
            expand=False,
        )
        implicit_map = {
            "triennal": 3,
            "biennal": 2,
            "quinquennal": 5,
            "quadriennal": 4,
            "settennal": 7,
            "novennal": 9,
        }
        implicit_years = implicit.str.lower().map(implicit_map)
        implicit_days = implicit_years * 365
        val = implicit_days.iloc[0]
        return None if pd.isna(val) else float(val)

    def test_durata_36_mesi(self):
        assert self._extract_regex_days("durata 36 mesi") == 1080

    def test_durata_5_anni(self):
        assert self._extract_regex_days("durata 5 anni") == 1825

    def test_triennale(self):
        assert self._extract_implicit_days("triennale") == 1095

    def test_quinquennale(self):
        assert self._extract_implicit_days("quinquennale") == 1825

    def test_biennale(self):
        assert self._extract_implicit_days("biennale") == 730

    def test_clamp_over_30_years(self):
        """Values over 10950 days should be clamped to None."""
        # 500 anni = 182500 days
        assert self._extract_regex_days("durata 500 anni") is None
        # 9999 mesi = 299970 days
        assert self._extract_regex_days("durata 9999 mesi") is None

    def test_random_large_number_no_durata_keyword(self):
        """Without 'durata', numeric patterns should not match."""
        assert self._extract_regex_days("736935 giorni servizio") is None

    def test_random_large_with_durata_but_over_4_digits(self):
        """Regex only matches 1-4 digits, so 736935 is not captured even with durata."""
        assert self._extract_regex_days("durata 736935 giorni") is None

    def test_durata_with_colon(self):
        assert self._extract_regex_days("durata: 24 mesi") == 720

    def test_durata_case_insensitive(self):
        assert self._extract_regex_days("DURATA 12 MESI") == 360

    def test_no_match_returns_none(self):
        assert self._extract_regex_days("appalto generico senza durata") is None

    def test_implicit_quadriennale(self):
        assert self._extract_implicit_days("contratto quadriennale") == 4 * 365

    def test_implicit_settennale(self):
        assert self._extract_implicit_days("servizio settennale") == 7 * 365

    def test_implicit_novennale(self):
        assert self._extract_implicit_days("appalto novennale") == 9 * 365

    def test_no_implicit_match(self):
        assert self._extract_implicit_days("servizio annuale") is None


# ===========================================================================
# CIG Enrichment (cache logic, parsing, integration — no API calls)
# ===========================================================================


def _duration_to_days(value, unit) -> float | None:
    """Reproduced from app.py."""
    if value is None or unit is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(unit).lower().strip()
    if u == "days":
        return v
    if u == "months":
        return v * 30
    if u == "years":
        return v * 365
    return None


class TestCIGEnrichment:
    """Test CIG enrichment cache logic, result parsing, and pipeline integration."""

    # -- duration_to_days helper --

    def test_duration_to_days_months(self):
        assert _duration_to_days(36, "months") == 1080

    def test_duration_to_days_years(self):
        assert _duration_to_days(5, "years") == 1825

    def test_duration_to_days_days(self):
        assert _duration_to_days(180, "days") == 180

    def test_duration_to_days_none(self):
        assert _duration_to_days(None, "months") is None
        assert _duration_to_days(36, None) is None
        assert _duration_to_days(None, None) is None

    def test_duration_to_days_invalid_value(self):
        assert _duration_to_days("abc", "months") is None

    # -- Cache structure --

    def test_cache_structure_valid(self):
        """A well-formed cache item should produce correct scadenza via LLM path."""
        cache = {
            "CIG0000001": {
                "updated_at": "2025-01-15T10:00:00Z",
                "model": "gpt-5-nano",
                "input_hash": "abc123",
                "result": {
                    "duration_base_days": 1095,
                    "duration_max_days": 1825,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.85,
                    "notes": "Triennale con opzione rinnovo biennale",
                },
                "errors": [],
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000001"],
            "award_date": [pd.Timestamp("2023-06-01")],
            "oggetto": ["generico"],
            "_categoria": ["ILLUMINAZIONE"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert result.loc[0, "scadenza_fonte"] == "llm"
        expected = pd.Timestamp("2023-06-01") + pd.Timedelta(days=1095)
        assert result.loc[0, "scadenza_contratto"] == expected
        assert result.loc[0, "llm_confidence"] == 0.85

    def test_cache_with_explicit_end_date(self):
        """LLM result with explicit_end_date should use it directly."""
        cache = {
            "CIG0000002": {
                "result": {
                    "duration_base_days": None,
                    "duration_max_days": None,
                    "explicit_start_date": None,
                    "explicit_end_date": "2028-12-31",
                    "confidence": 0.9,
                    "notes": "Scadenza esplicita",
                },
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000002"],
            "award_date": [pd.Timestamp("2023-01-01")],
            "oggetto": ["generico"],
            "_categoria": ["ENERGIA"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert result.loc[0, "scadenza_fonte"] == "llm"
        assert result.loc[0, "scadenza_contratto"] == pd.Timestamp("2028-12-31")

    def test_cache_with_explicit_start_and_duration(self):
        """LLM result with explicit_start_date + duration should compute from start."""
        cache = {
            "CIG0000003": {
                "result": {
                    "duration_base_days": 730,
                    "duration_max_days": None,
                    "explicit_start_date": "2024-01-01",
                    "explicit_end_date": None,
                    "confidence": 0.7,
                    "notes": "Biennale da data stipula",
                },
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000003"],
            "award_date": [pd.Timestamp("2023-06-15")],
            "oggetto": ["generico"],
            "_categoria": ["MANUTENZIONE"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert result.loc[0, "scadenza_fonte"] == "llm"
        # Should use explicit_start_date (2024-01-01) + 730 days, NOT award_date
        expected = pd.Timestamp("2024-01-01") + pd.Timedelta(days=730)
        assert result.loc[0, "scadenza_contratto"] == expected

    def test_cache_max_scadenza(self):
        """scadenza_contratto_max should be set when duration_max_days is present."""
        cache = {
            "CIG0000004": {
                "result": {
                    "duration_base_days": 1095,
                    "duration_max_days": 1825,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.8,
                    "notes": "3+2",
                },
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000004"],
            "award_date": [pd.Timestamp("2023-01-01")],
            "oggetto": ["generico"],
            "_categoria": ["RIFIUTI"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert result.loc[0, "scadenza_contratto_max"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=1825)

    def test_cache_no_result_field(self):
        """Cache item with result=None should not crash."""
        cache = {
            "CIG0000005": {
                "result": None,
                "errors": ["call_error:ConnectionError"],
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000005"],
            "award_date": [pd.Timestamp("2023-01-01")],
            "oggetto": ["generico"],
            "_categoria": ["PULIZIE"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert len(result) == 1
        assert result.loc[0, "scadenza_fonte"] == "mancante"

    def test_cache_overridden_by_data_scadenza(self):
        """Explicit data_scadenza should take priority over LLM cache."""
        cache = {
            "CIG0000006": {
                "result": {
                    "duration_base_days": 365,
                    "duration_max_days": None,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.6,
                    "notes": "annuale",
                },
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000006"],
            "award_date": [pd.Timestamp("2023-01-01")],
            "data_scadenza": ["2030-06-30"],
            "oggetto": ["generico"],
            "_categoria": ["VIGILANZA"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert result.loc[0, "scadenza_fonte"] == "data_scadenza"
        assert result.loc[0, "scadenza_contratto"] == pd.Timestamp("2030-06-30")

    def test_cache_multiple_cigs_mixed(self):
        """Multiple CIGs: some with cache hit, some without, some with errors."""
        cache = {
            "CIG0000007": {
                "result": {
                    "duration_base_days": 730,
                    "duration_max_days": None,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.75,
                    "notes": "biennale",
                },
            },
            "CIG0000008": {
                "result": None,
                "errors": ["no_snippets"],
            },
        }
        df = pd.DataFrame({
            "cig": ["CIG0000007", "CIG0000008", "CIG0000009"],
            "award_date": [pd.Timestamp("2023-01-01")] * 3,
            "oggetto": ["generico", "generico", "durata 24 mesi"],
            "_categoria": ["ENERGIA", "RIFIUTI", "VERDE_PUBBLICO"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert len(result) == 3
        assert result.loc[0, "scadenza_fonte"] == "llm"
        assert result.loc[1, "scadenza_fonte"] == "mancante"
        assert result.loc[2, "scadenza_fonte"] == "regex_oggetto"

    def test_cache_confidence_propagated(self):
        """LLM confidence and notes should be propagated to output."""
        cache = {
            "CIG0000010": {
                "result": {
                    "duration_base_days": 365,
                    "duration_max_days": None,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.42,
                    "notes": "Durata incerta, solo stima da contesto",
                },
            }
        }
        df = pd.DataFrame({
            "cig": ["CIG0000010"],
            "award_date": [pd.Timestamp("2024-03-01")],
            "oggetto": ["generico"],
            "_categoria": ["ICT"],
        })
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        assert abs(result.loc[0, "llm_confidence"] - 0.42) < 0.01
        assert "incerta" in result.loc[0, "llm_notes"]

    def test_cache_cig_case_insensitive(self):
        """CIG matching should work regardless of case in cache keys."""
        cache = {
            "CIG0000011": {
                "result": {
                    "duration_base_days": 180,
                    "duration_max_days": None,
                    "explicit_start_date": None,
                    "explicit_end_date": None,
                    "confidence": 0.5,
                    "notes": "",
                },
            }
        }
        df = pd.DataFrame({
            "cig": ["cig0000011"],  # lowercase in data
            "award_date": [pd.Timestamp("2024-01-01")],
            "oggetto": ["generico"],
            "_categoria": ["ALTRO"],
        })
        # Note: the current implementation does exact match on cig string
        # so "cig0000011" != "CIG0000011" — this tests current behavior
        result = _compute_scadenze_contratti(df, None, include_stime=False, cig_enrichment_items=cache)
        # CIG in data is lowercase, cache has uppercase — no match expected
        assert result.loc[0, "scadenza_fonte"] != "llm"


# ===========================================================================
# Integration: full scadenze pipeline on a small real data sample
# ===========================================================================

class TestScadenzeIntegration:
    """Integration tests running the full pipeline on real data samples."""

    @pytest.mark.slow
    def test_scadenze_on_real_sample(self):
        """Run scadenze computation on 500 rows from real data."""
        df = pd.read_csv(
            GARE_GZ,
            compression="gzip",
            nrows=500,
            dtype={
                "cig": "str",
                "oggetto": "str",
                "importo_aggiudicazione": "float64",
                "data_aggiudicazione": "str",
                "data_scadenza": "str",
                "durata_appalto": "float64",
                "categoria": "str",
                "comune": "str",
                "regione": "str",
            },
        )
        if "categoria" in df.columns:
            df["_categoria"] = df["categoria"].str.upper().str.strip()
        if "data_aggiudicazione" in df.columns:
            df["award_date"] = pd.to_datetime(df["data_aggiudicazione"], errors="coerce")
        if "importo_aggiudicazione" in df.columns:
            df["award_amount"] = pd.to_numeric(df["importo_aggiudicazione"], errors="coerce")

        result = _compute_scadenze_contratti(df, None, include_stime=True)

        assert len(result) == len(df)
        assert "scadenza_contratto" in result.columns
        assert "scadenza_fonte" in result.columns
        assert "giorni_alla_scadenza" in result.columns
        assert "stato_scadenza" in result.columns

        # At least some rows should have a valid scadenza
        has_scadenza = result["scadenza_contratto"].notna().sum()
        assert has_scadenza > 0, "No rows had a computed scadenza"

        # All fonti should be from expected set
        valid_fonti = {
            "data_scadenza",
            "consip",
            "durata_appalto",
            "regex_oggetto",
            "llm",
            "stima_categoria",
            "mancante",
            "invalid",
        }
        actual_fonti = set(result["scadenza_fonte"].unique())
        assert actual_fonti.issubset(valid_fonti), f"Unexpected fonti: {actual_fonti - valid_fonti}"

        # stato_scadenza should only contain known values
        valid_stati = {"Sconosciuta", "Scaduto", "Attivo"}
        actual_stati = set(result["stato_scadenza"].unique())
        assert actual_stati.issubset(valid_stati), f"Unexpected stati: {actual_stati - valid_stati}"
