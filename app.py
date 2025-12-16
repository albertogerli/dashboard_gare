#!/usr/bin/env python3
"""Dashboard Gare Pubbliche - Streamlit (Extended)"""


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz, process
import os
import requests
import hashlib
from datetime import datetime
from dotenv import load_dotenv


def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object dtype columns to string to avoid Arrow serialization warnings."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).replace('nan', '').replace('None', '')
    return df_copy


# Carica variabili d'ambiente dal file .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Config
st.set_page_config(
    page_title="Dashboard Gare Pubbliche",
    page_icon="📊",
    layout="wide"
)

# City Green Light Corporate Colors
CGL_GREEN = "#00d084"  # Verde principale
CGL_BLUE = "#2ea3f2"   # Blu accent
CGL_CYAN = "#0693e3"   # Cyan secondario
CGL_ORANGE = "#ff9500" # Arancione per trend
CGL_RED = "#ff3b30"    # Rosso per warning
CGL_DARK = "#1a1a2e"   # Scuro per sfondo
CGL_BLACK = "#000000"
CGL_WHITE = "#ffffff"

# Custom CSS - City Green Light Theme + Accessibilità
st.markdown("""
<style>
    /* Import Google Fonts per accessibilità */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Sans+Pro:wght@400;600;700&display=swap');

    /* Reset e base accessibile */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Titoli accessibili con buon contrasto */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1a1a2e;
        line-height: 1.3;
    }

    h1 { font-size: 2rem; }
    h2 { font-size: 1.75rem; }
    h3 { font-size: 1.5rem; }
    h4 { font-size: 1.25rem; }

    /* Metric cards con colori CGL */
    .stMetric > div {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #00d084;
        box-shadow: 0 2px 8px rgba(0, 208, 132, 0.1);
    }

    .stMetric label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #4a5568 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    /* Tabs styling CGL */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.95rem;
        padding: 10px 20px;
        border-radius: 8px;
        color: #4a5568;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00d084 !important;
        color: white !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* Buttons CGL */
    .stButton > button {
        background: linear-gradient(135deg, #00d084 0%, #0693e3 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 208, 132, 0.3);
    }

    /* DataFrames e tabelle */
    .stDataFrame {
        font-size: 0.9rem;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Selectbox e inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div {
        border-radius: 8px;
        border-color: #e2e8f0;
    }

    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #00d084;
        box-shadow: 0 0 0 2px rgba(0, 208, 132, 0.2);
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
        font-size: 0.95rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1a1a2e;
        font-size: 1rem;
    }

    /* Link accessibility */
    a {
        color: #2ea3f2;
        text-decoration: underline;
    }

    a:hover {
        color: #0693e3;
    }

    /* Focus states per accessibilità */
    *:focus-visible {
        outline: 3px solid #00d084;
        outline-offset: 2px;
    }

    /* Caption e small text */
    .stCaption, small {
        font-size: 0.85rem;
        color: #718096;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #1a1a2e;
        color: white;
        border: 2px solid #00d084;
    }

    .stDownloadButton > button:hover {
        background: #00d084;
    }

    /* Chart container */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==================== AI VISUALIZATION HELPERS ====================

# Favorites storage path
FAVORITES_PATH = Path(__file__).parent.parent / "data" / "output" / "dashboard" / "favorites.json"

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

def call_responses_api(prompt: str, instructions: str) -> str:
    """Call OpenAI Responses API with gpt-5.1-codex-mini"""
    api_key = get_openai_api_key()
    if not api_key:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-5.1-codex-mini",
        "input": prompt,
        "instructions": instructions
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
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
        result = call_responses_api(prompt, instructions)

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
        code = call_responses_api(f"Crea questo grafico: {prompt}", instructions)

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
        'tipo_appalto_sel': ('Tipo Appalto', None),
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
        st.plotly_chart(fig, width="stretch", key=f"chart_{chart_key}")
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
        'fonte': 'category',  # Pochi valori (Gazzetta, OCDS, etc)
        'categoria': 'category',  # ~20 categorie
        'categoria_originale': 'str',
        'sottocategoria': 'str',  # Troppi valori per category
        'procedura': 'str',  # Molti valori unici - normalizzare dopo
        'procedura_originale': 'str',
        'tipo_appalto': 'category',  # ~5 tipi
        'tipo_appalto_originale': 'str',
        'regione': 'str',  # Normalizzare dopo
        'comune': 'str',  # Troppi comuni per category
        'anno': 'float64',  # Float per gestire NaN, convertire dopo
        'offerte_ricevute': 'float64',  # Float per gestire NaN
        'num_lotti': 'float64',  # Float per gestire NaN
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
            # Filtra dtype solo per colonne esistenti
            dtype_use = {k: v for k, v in dtype_opt.items() if k in usecols}
            df = pd.read_csv(gz_path, compression='gzip', usecols=usecols, dtype=dtype_use, low_memory=True)
        elif unified_path.exists():
            df_cols = pd.read_csv(unified_path, nrows=0)
            usecols = [c for c in df_cols.columns if c not in cols_to_exclude]
            dtype_use = {k: v for k, v in dtype_opt.items() if k in usecols}
            df = pd.read_csv(unified_path, usecols=usecols, dtype=dtype_use, low_memory=True)
        elif old_path.exists():
            df_cols = pd.read_csv(old_path, nrows=0)
            usecols = [c for c in df_cols.columns if c not in cols_to_exclude]
            dtype_use = {k: v for k, v in dtype_opt.items() if k in usecols}
            df = pd.read_csv(old_path, usecols=usecols, dtype=dtype_use, low_memory=True)
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

data = load_data()
raw_df = load_raw_data()
consip_raw_df = load_consip_data()

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

# Sidebar filters
st.sidebar.title("🔍 Filtri")

# Fonte filter (Gazzetta/OCDS/ServizioLuce)
if 'fonte' in raw_df.columns:
    fonti_disponibili = raw_df['fonte'].dropna().unique().tolist()
    fonti = [None] + sorted(fonti_disponibili)
    fonte_sel = st.sidebar.selectbox("Fonte dati", fonti, format_func=lambda x: "Tutte" if x is None else x)
else:
    fonte_sel = None

# Anno filter
anni = [None] + sorted([int(y) for y in raw_df['anno'].dropna().unique() if 2015 <= y <= 2025])
anno_sel = st.sidebar.selectbox("Anno", anni, format_func=lambda x: "Tutti" if x is None else str(x))

# Regione filter
if 'regione' in raw_df.columns and raw_df['regione'].notna().any():
    regioni_df = sorted(raw_df['regione'].dropna().unique().tolist())
    regioni = [None] + regioni_df
else:
    regioni = [None] + [r['Regione'] for r in data['geo']]
regione_sel = st.sidebar.selectbox("Regione", regioni, format_func=lambda x: "Tutte" if x is None else x)

# Categoria filter - usa colonna normalizzata 'categoria' se disponibile
if 'categoria' in raw_df.columns and raw_df['categoria'].notna().any():
    cat_list = sorted(raw_df['categoria'].dropna().unique().tolist())
    categorie = [None] + cat_list
elif '_categoria' in raw_df.columns and raw_df['_categoria'].notna().any():
    cat_list = sorted(raw_df['_categoria'].dropna().unique().tolist())
    categorie = [None] + cat_list
else:
    categorie = [None] + data['filter_options']['categorie_macro']
categoria_sel = st.sidebar.selectbox("Categoria", categorie, format_func=lambda x: "Tutte" if x is None else x)

# Procedura filter - usa colonna normalizzata
if 'procedura' in raw_df.columns and raw_df['procedura'].notna().any():
    proc_list = sorted(raw_df['procedura'].dropna().unique().tolist())
    procedure = [None] + proc_list
    procedura_sel = st.sidebar.selectbox("Procedura", procedure, format_func=lambda x: "Tutte" if x is None else x)
else:
    procedura_sel = None

# Tipo Appalto filter - usa colonna normalizzata
if 'tipo_appalto' in raw_df.columns and raw_df['tipo_appalto'].notna().any():
    tipo_list = sorted(raw_df['tipo_appalto'].dropna().unique().tolist())
    tipi_appalto = [None] + tipo_list
    tipo_appalto_sel = st.sidebar.selectbox("Tipo Appalto", tipi_appalto, format_func=lambda x: "Tutti" if x is None else x)
else:
    tipo_appalto_sel = None

# Sottocategoria filter (dinamico basato su categoria) - usa categoria_originale
if categoria_sel and 'categoria' in raw_df.columns:
    if 'categoria_originale' in raw_df.columns:
        sottocategorie_list = sorted(raw_df[raw_df['categoria'] == categoria_sel]['categoria_originale'].dropna().unique().tolist())
    elif 'categorie_regex' in raw_df.columns:
        sottocategorie_list = sorted(raw_df[raw_df['categoria'] == categoria_sel]['categorie_regex'].dropna().unique().tolist())
    else:
        sottocategorie_list = []
    sottocategorie = [None] + sottocategorie_list
elif 'categoria_originale' in raw_df.columns:
    sottocategorie = [None] + sorted(raw_df['categoria_originale'].dropna().unique().tolist())[:50]  # Limit per performance
elif 'categorie_regex' in raw_df.columns:
    sottocategorie = [None] + sorted(raw_df['categorie_regex'].dropna().unique().tolist())
else:
    sottocategorie = [None]
sottocategoria_sel = st.sidebar.selectbox("Sottocategoria", sottocategorie, format_func=lambda x: "Tutte" if x is None else x)

# Apply filters to raw data
filtered_df = raw_df.copy()
if fonte_sel:
    filtered_df = filtered_df[filtered_df['fonte'] == fonte_sel]
if anno_sel:
    filtered_df = filtered_df[filtered_df['anno'] == anno_sel]
if regione_sel and 'regione' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['regione'] == regione_sel]
if categoria_sel:
    # Usa la nuova colonna 'categoria' normalizzata, altrimenti fallback a '_categoria'
    if 'categoria' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['categoria'] == categoria_sel]
    elif '_categoria' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['_categoria'] == categoria_sel]
if procedura_sel and 'procedura' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['procedura'] == procedura_sel]
if tipo_appalto_sel and 'tipo_appalto' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['tipo_appalto'] == tipo_appalto_sel]
if sottocategoria_sel:
    if 'categoria_originale' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['categoria_originale'] == sottocategoria_sel]
    elif 'categorie_regex' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['categorie_regex'] == sottocategoria_sel]

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

col1.metric("🏛️ Totale Gare", f"{total_gare:,}".replace(",", "."))
col2.metric("💰 Valore Totale", f"€{total_value/1e9:.2f}B" if total_value > 0 else "€0")
col3.metric("📉 Sconto Medio", f"{avg_sconto:.1f}%" if pd.notna(avg_sconto) and not np.isnan(avg_sconto) else "N/D")
col4.metric("👥 Partecipanti Medi", f"{avg_participants:.1f}" if pd.notna(avg_participants) and not np.isnan(avg_participants) else "N/D")
col5.metric("🏢 Stazioni Appaltanti", f"{unique_buyers:,}".replace(",", "."))
col6.metric("🏭 Fornitori Unici", f"{unique_suppliers:,}".replace(",", "."))

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

col1.metric("📊 Valore Mediano", f"€{median_value/1e3:.0f}K" if median_value > 0 else "N/D")
col2.metric("🔝 Gara Max", f"€{max_value/1e6:.1f}M" if max_value > 0 else "N/D")
col3.metric("📰 Gazzetta", f"{gare_gazzetta:,}".replace(",", "."))
col4.metric("📊 OCDS", f"{gare_ocds:,}".replace(",", "."))
col5.metric("🏛️ CONSIP", f"{gare_consip:,}".replace(",", "."))
col6.metric("🔑 Chiavi Uniche", f"{chiavi_uniche:,}".replace(",", "."))

# ==================== TAB NAVIGATION (CLUSTER UI) ====================
st.markdown("---")

# Cluster selection con radio buttons
st.markdown("""
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
    border-color: #00d084;
    background: #f0fff8;
}
.cluster-btn.active {
    background: #00d084;
    border-color: #00d084;
    color: white;
}
</style>
""", unsafe_allow_html=True)

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
    tab1, tab2, tab3, tab6 = st.tabs(["🗺️ Geografia", "📦 Categorie", "📈 Trend", "📊 Statistiche"])
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

    # Dettaglio regioni
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
        st.dataframe(geo_detail[['Regione', 'N. Gare', 'Valore (€B)', 'Sconto Medio %']], use_container_width=True)
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
            fig = px.scatter(
                cat_df,
                x='num_gare',
                y='valore',
                size='partecipanti_medi',
                color='sconto_medio',
                text='Categoria_Main',
                color_continuous_scale='RdYlGn',
                labels={'num_gare': 'Numero Gare', 'valore': 'Valore (€)', 'sconto_medio': 'Sconto %'}
            )
            fig.update_traces(textposition='top center', textfont_size=9)
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
                st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch", key="dist_offerte")
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
        st.plotly_chart(fig, width="stretch", key="box_sconti_cat")
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
        st.plotly_chart(fig, width="stretch")

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
                    st.plotly_chart(fig, width="stretch", key="dist_mensile_stat")
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
    st.dataframe(stats_df, width="stretch", hide_index=True)

# ==================== TAB 7: RICERCA CITTÀ ====================
if tab7:
  with tab7:
    st.subheader("🔍 Ricerca Servizi Attivi per Città")

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

    # Get unique cities from filtered data
    if locality_col and locality_col in filtered_df.columns:
        cities_list = sorted(filtered_df[locality_col].dropna().unique().tolist())
    else:
        cities_list = []
        st.warning("Colonna città non trovata nel dataset filtrato")

    st.info(f"💡 I risultati rispettano i filtri selezionati nella sidebar ({len(filtered_df):,} gare filtrate)".replace(",", "."))

    # Search box
    col1, col2 = st.columns([2, 1])
    with col1:
        citta_search = st.selectbox(
            "Seleziona o cerca una città",
            options=[""] + cities_list,
            index=0,
            help="Digita per cercare"
        )
    with col2:
        solo_attivi = st.checkbox("Solo contratti attivi (2023-2025)", value=True)

    if citta_search and locality_col:
        # Filter data for selected city from already filtered data
        city_df = filtered_df[filtered_df[locality_col].str.upper() == citta_search.upper()].copy()

        if solo_attivi and 'anno' in city_df.columns:
            city_df = city_df[city_df['anno'] >= 2023]

        if len(city_df) > 0:
            st.markdown(f"### 📍 {citta_search.upper()}")

            # City KPIs - usa colonne dinamiche
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🏛️ Totale Gare", f"{len(city_df):,}".replace(",", "."))

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

            if buyer_col_city:
                col4.metric("🏢 Enti Appaltanti", f"{city_df[buyer_col_city].nunique()}")
            else:
                col4.metric("🏢 Enti Appaltanti", "N/D")

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
                        st.plotly_chart(fig, width="stretch")

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
                            st.plotly_chart(fig, width="stretch")
            else:
                st.info("Dati categoria non disponibili per questo filtro")

            # Top suppliers in city
            st.markdown("---")
            st.markdown("#### 🏆 Top Fornitori nella Città")

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
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.dataframe(top_suppliers, width="stretch")
            else:
                st.info("Dati fornitori non disponibili")

            # Top buyers (stazioni appaltanti)
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
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.dataframe(top_buyers, width="stretch")
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

                st.dataframe(display_df.iloc[start_idx:end_idx], width="stretch", height=500)
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
                        st.plotly_chart(fig, width="stretch")

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
                    st.plotly_chart(fig, width="stretch")

                st.dataframe(city_summary, width="stretch")
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
                st.plotly_chart(fig, width="stretch")
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
            st.dataframe(safe_dataframe(tipo_summary), width="stretch", hide_index=True)

            st.markdown("#### 🏙️ Top 10 Città")
            top_cities = consip_by_city.nlargest(10, 'valore')[['citta', 'num_gare', 'valore']]
            top_cities['valore'] = top_cities['valore'].apply(lambda x: f'€{x/1e6:.0f}M')
            top_cities.columns = ['Città', 'Gare', 'Valore']
            st.dataframe(safe_dataframe(top_cities), width="stretch", hide_index=True)

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
        st.plotly_chart(fig, width="stretch")

        # Detailed table
        st.markdown("---")
        st.markdown("#### 📋 Dettaglio Contratti CONSIP")

        display_consip = filtered_consip[['DataAggiudicazione', 'Comune', 'Regione', 'TipoAccordo', 'Edizione', 'OggettoGara', 'award_amount', 'sconto', 'Aggiudicatario']].copy()
        display_consip['DataAggiudicazione'] = pd.to_datetime(display_consip['DataAggiudicazione'], errors='coerce').dt.strftime('%Y-%m-%d')
        display_consip['award_amount'] = display_consip['award_amount'].apply(lambda x: f'€{x:,.0f}'.replace(',', '.') if pd.notna(x) else '-')
        display_consip['sconto'] = display_consip['sconto'].apply(lambda x: f'{x:.1f}%' if pd.notna(x) else '-')
        display_consip.columns = ['Data', 'Città', 'Regione', 'Tipo', 'Edizione', 'Oggetto', 'Valore', 'Sconto', 'Aggiudicatario']
        display_consip = display_consip.sort_values('Data', ascending=False)

        st.dataframe(display_consip.head(100), width="stretch", height=400)

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
                    st.plotly_chart(fig, width="stretch")

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
                    st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")
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
                        st.plotly_chart(fig, width="stretch")
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
                        st.plotly_chart(fig, width="stretch")
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

                st.dataframe(display_supplier.iloc[start_idx:end_idx], width="stretch", height=400)
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
                st.dataframe(display_top, width="stretch", height=400)

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
                    st.plotly_chart(fig, width="stretch")
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
                        st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")
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
            st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")
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
            st.plotly_chart(fig, width="stretch")
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
                    st.dataframe(top_large, width="stretch", hide_index=True)
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
                st.dataframe(pd.DataFrame(dominant), width="stretch", hide_index=True)
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
                    st.plotly_chart(fig, width="stretch", key="proc_sconto")
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
                st.plotly_chart(fig, width="stretch", key="region_sconto")
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

    if len(consip_exp) > 0 and 'DataAggiudicazione' in consip_exp.columns and 'DURATA_PREVISTA' in consip_exp.columns:
        # Calcola scadenze
        consip_exp['DataAggiudicazione'] = pd.to_datetime(consip_exp['DataAggiudicazione'], format='%d/%m/%Y', errors='coerce')
        consip_exp['durata_giorni'] = pd.to_numeric(consip_exp['DURATA_PREVISTA'], errors='coerce')
        consip_exp['ScadenzaContratto'] = consip_exp['DataAggiudicazione'] + pd.to_timedelta(consip_exp['durata_giorni'], unit='D')

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
            st.plotly_chart(fig, width="stretch")

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
                st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")
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
            st.dataframe(display_df.sort_values('Scadenza'), width="stretch", hide_index=True)

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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

        # Alert scadenze imminenti
        st.markdown("---")
        st.subheader("⚠️ Alert: Contratti in Scadenza Prossimi 12 Mesi")

        # Usa anno per evitare problemi timezone
        imminenti = stima_future[stima_future['anno_scadenza_stima'] <= anno_corrente + 1]

        if len(imminenti) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Contratti Imminenti", f"{len(imminenti):,}".replace(",", "."))
            with col2:
                st.metric("💰 Valore a Rischio", f"€{imminenti['award_amount'].sum()/1e9:.2f}B")
            with col3:
                st.metric("🏢 Enti Coinvolti", f"{imminenti['buyer_name'].nunique():,}".replace(",", "."))

            # Top categorie imminenti
            imm_cat = imminenti.groupby('_categoria', observed=True)['ocid'].count().sort_values(ascending=False).head(5)
            st.markdown("**Top 5 Categorie con Scadenze Imminenti:**")
            for cat, count in imm_cat.items():
                st.write(f"- {cat}: {count} contratti")
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
                st.plotly_chart(fig, width="stretch", key="influence_a")

            with col2:
                reg_b = df_b.groupby(region_col, observed=True)['award_amount'].sum().sort_values(ascending=False).head(10).reset_index()
                reg_b.columns = ['Regione', 'Valore']
                st.markdown(f"**🔴 Top Regioni {supplier_b[:25]}**")
                fig = px.bar(reg_b, x='Valore', y='Regione', orientation='h', color_discrete_sequence=['#EF553B'])
                fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, width="stretch", key="influence_b")

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

                fig = px.imshow(pivot_table, color_continuous_scale='YlOrRd',
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
                st.plotly_chart(fig, width="stretch", key="quarterly_analysis")
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
                st.plotly_chart(fig, width="stretch", key="quarterly_valore_medio")
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
        st.plotly_chart(fig, width="stretch", key="growth_lines")

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
            st.plotly_chart(fig, width="stretch", key="growth_value")

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
            st.plotly_chart(fig, width="stretch", key="growth_count")

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
        st.dataframe(pd.DataFrame(detail_data), width="stretch", hide_index=True)

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
                        st.dataframe(outliers_display, width="stretch", height=300)
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
                    st.plotly_chart(fig, width="stretch", key="sconto_anomalies")

                    if len(anomalous_sconti) > 0:
                        st.info(f"📊 Sconti anomali: {len(anomalous_sconti)} gare fuori range [{lower_bound:.1f}%, {upper_bound:.1f}%]")
                else:
                    st.info("Dati sconto insufficienti per l'analisi")
            else:
                st.info("Colonna sconto non disponibile o vuota")

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
            st.dataframe(cols_info, width="stretch", hide_index=True)

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
                <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 3px solid #00d084;">
                    <span style="font-size: 1.2em;">{icon}</span> <strong>{label}</strong>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Usa questo", key=f"example_{i}", width="stretch"):
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
            analyze_btn = st.button("🔍 1. Analizza", type="secondary", width="stretch")
        with col_btn2:
            generate_btn = st.button("🚀 2. Genera", type="primary", width="stretch", disabled=('ai_analysis' not in st.session_state))

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
                    modify_btn = st.button("🔄 Modifica", key="modify_analysis_btn", width="stretch")

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
                        st.plotly_chart(fig, width="stretch", key="ai_generated_chart")

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
                                st.plotly_chart(fig, width="stretch", key=f"fav_chart_{fav.get('id', i)}")
                            else:
                                st.warning(f"Errore: {error}")
                        elif fav.get('type') == 'standard' and fav.get('fig_json'):
                            try:
                                import plotly.io as pio
                                fig = pio.from_json(fav['fig_json'])
                                st.plotly_chart(fig, width="stretch", key=f"fav_chart_{fav.get('id', i)}")
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
                            st.plotly_chart(fig, width="stretch", key=f"fav_list_{fav.get('id', i)}")
                        else:
                            st.warning(f"Errore: {error}")
                    elif fav.get('type') == 'standard' and fav.get('fig_json'):
                        try:
                            import plotly.io as pio
                            fig = pio.from_json(fav['fig_json'])
                            st.plotly_chart(fig, width="stretch", key=f"fav_list_{fav.get('id', i)}")
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
                st.plotly_chart(msg['chart'], width="stretch")

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
            if st.button(q, key=f"quick_{i}", width="stretch"):
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

            predict_btn = st.button("🔮 Calcola Predizioni", type="primary", width="stretch")

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
                        for idx, row in cat_winners.head(5).iterrows():
                            prob = row['🎯 Probabilità (%)']
                            color = "🟢" if prob > 15 else "🟡" if prob > 5 else "🔴"
                            valore_str = f"€{row['Valore Medio']/1e6:.2f}M" if row['Valore Medio'] > 0 else "N/A"
                            st.markdown(f"""
                            **{color} {row['Fornitore'][:40]}**
                            - Probabilità: **{prob}%**
                            - Gare vinte in categoria: {row['Gare Vinte']}
                            - Valore medio: {valore_str}
                            """)
                            st.progress(min(prob/30, 1.0))

                        # Chart
                        fig = px.bar(
                            cat_winners.head(10),
                            x='🎯 Probabilità (%)',
                            y='Fornitore',
                            orientation='h',
                            title=f'Top 10 Probabili Vincitori - {selected_cat[:30]}',
                            color='🎯 Probabilità (%)',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, width="stretch")
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
            st.plotly_chart(fig, width="stretch")

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
                color_continuous_scale='YlOrRd'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")

            # Stats table
            st.dataframe(
                safe_dataframe(region_data[['Regione', 'N_Gare', 'Valore_B']].rename(columns={'Valore_B': 'Valore (€B)'}).sort_values('Valore (€B)', ascending=False)),
                width="stretch",
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
                st.plotly_chart(fig, width="stretch")

                st.dataframe(city_filtered.head(30), width="stretch", hide_index=True)
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
                            st.plotly_chart(fig, width="stretch")

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
                            st.plotly_chart(fig, width="stretch")

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
                            st.plotly_chart(fig, width="stretch")
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
                st.plotly_chart(fig, width="stretch")

                # Summary stats
                year_totals = anim_data.groupby('Anno', observed=True)['Valore'].sum() / 1e9
                fig2 = px.area(
                    x=year_totals.index,
                    y=year_totals.values,
                    title='Valore Totale Gare per Anno (€B)',
                    labels={'x': 'Anno', 'y': 'Valore (€B)'}
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, width="stretch")
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
