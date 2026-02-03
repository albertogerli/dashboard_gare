# Dashboard Gare Pubbliche Italiane — Guida Completa (funzioni, filtri, tab)

Ultimo aggiornamento: 2026-02-03  
Repository: `dashboard_gare` — app Streamlit in `app.py`

## 1) Obiettivo della dashboard

La dashboard serve a:
- esplorare un dataset di gare/contratti pubblici (OCDS/ANAC, Gazzetta, CONSIP);
- analizzare geografia, categorie, trend, operatori (enti/fornitori), concentrazione di mercato;
- individuare **contratti attivi** e **scadenze** per città/area (anche con enrichment automatico via CIG);
- generare visualizzazioni e analisi assistite con AI, salvabili nei preferiti.

## 2) Sorgenti dati (input principali)

La dashboard lavora (in modo combinato) con questi input locali:

1) `data/gare_unificate.csv.gz`  
   Dataset “master” di gare/contratti (colonne variabili; la dashboard cerca dinamicamente i campi principali).

2) `data/data.json`  
   Aggregati/pre-calcoli utilizzati come fallback per alcune viste (es. top aggiudicatari/categorie/consip).

3) `data/ServizioLuce.xlsx`  
   Dati CONSIP/Servizio Luce (e, se presenti, edizioni/varianti) utilizzati per viste CONSIP e scadenze CONSIP.

Output/cache locali:
- `data/output/dashboard/favorites.json` — grafici salvati nei preferiti.
- `data/output/dashboard/cig_enrichment_cache.json` — cache enrichment scadenze via CIG (LLM).

## 3) Struttura generale della UI

### 3.1 Sidebar: Filtri globali (sempre attivi)
I filtri in sidebar si applicano a **tutte le viste** (tabelle, grafici, KPI), salvo dove indicato diversamente.

**Sezione “Fonte dati” (radio)**
- Campo: `fonte` (se presente nel dataset).
- Opzioni: “Tutte” + valori unici (es. `OCDS`, `Gazzetta`, `CONSIP`).
- Effetto: filtra il dataframe di lavoro `filtered_df`.

**Sezione “Anno” (selectbox)**
- Campo: `anno`.
- Opzioni: “Tutti gli anni” + anni (in pratica 2015–2025, ordinati decrescenti).

**Sezione “Regione” (selectbox)**
- Campo: `regione` (se presente), altrimenti fallback alla lista regioni in `data.json`.

**Sezione “Tipologia contratto” (radio)**
- Campo preferito: `tipo_appalto_norm` (normalizzato), altrimenti `tipo_appalto`.
- Opzioni: “Tutti” + lista di tipologie presenti.

**Sezione “Filtri avanzati”**
- **Categoria** (selectbox):
  - Campo preferito: `categoria`, altrimenti `_categoria`, altrimenti fallback a `data.json`.
- **Sottocategoria** (selectbox):
  - Campo: `quick_category` (se presente).
  - Logica: se è selezionata una categoria, la lista delle sottocategorie viene filtrata su quella categoria.
- **Procedura** (selectbox):
  - Campo: `procedura` (se presente).

Nota: la dashboard costruisce una `filter_key` (combinazione dei filtri attivi) usata per “resettare” selezioni multiselect quando i filtri cambiano.

### 3.2 Header e KPI globali
Subito dopo i filtri, la dashboard mostra:
- titolo e conteggio record totali + record filtrati (e breakdown per fonte se disponibile);
- KPI su `filtered_df`:
  - Totale gare
  - Valore totale aggiudicato (somma importi)
  - Sconto medio
  - Partecipanti medi (se campo presente)
  - Numero stazioni appaltanti uniche
  - Numero fornitori unici
  - KPI addizionali: valore mediano, gara max, conteggi per fonte (Gazzetta/OCDS/CONSIP), chiavi uniche

### 3.3 Navigazione per “Cluster”
La parte centrale della UI è organizzata in 5 “aree di analisi” (radio orizzontale):
- `📊 Panoramica`
- `🏆 Operatori`
- `🗺️ Territoriale`
- `📈 Analisi Avanzata`
- `🤖 AI & Preferiti`

Ogni cluster abilita un set di tab Streamlit.

## 4) Cluster “📊 Panoramica”

### Tab 1 — `🗺️ Geografia`
Obiettivo: analisi territoriale “macro” (città/regione).

Sezioni principali:
- **Mappa Città per Valore**: scatter su mappa (coordinate hardcoded per principali città italiane), con:
  - size = valore (somma importi)
  - color = sconto medio
  - hover = città, num gare, valore, sconto medio
  - prende top 30 per valore con coordinate note
- **Classifica Regioni**:
  - bar chart orizzontale per valore totale per regione
  - colore su sconto medio
- **Tabella regioni**: `Regione`, `N. Gare`, `Valore (€B)`, `Sconto Medio %`
- **Esplora gare per regione**:
  - selectbox regione (o “Tutte”)
  - tabella “ultime gare” (max 100) con formattazioni base
  - download CSV + Excel delle righe filtrate

Note/limiti:
- la mappa città mostra solo città presenti in dizionario coordinate (nessun geocoding automatico).

### Tab 2 — `📦 Categorie`
Obiettivo: capire distribuzione per categoria e confronto multi-metrica.

Sezioni:
- **Treemap categorie** (valore, colore sconto medio, hover num gare)
- **Scatter categorie**:
  - x = num gare
  - y = valore
  - size = partecipanti medi (se disponibile, altrimenti 1)
  - color = sconto medio
- **Radar “Top 5”**:
  - normalizzazione su 4 dimensioni: num gare, valore, sconto medio, partecipanti medi

### Tab 3 — `📈 Trend`
Obiettivo: trend storici su anni e pattern per categoria.

Sezioni:
- **Trend Sconti e Partecipanti (doppio asse)**:
  - sconto medio (area/linea) + partecipanti medi (linea tratteggiata) + mediana sconto (linea puntinata)
- **Volume gare per anno**:
  - include record con `anno` mancante (categoria “N/D”)
  - bar chart con colore “Con Data / Data N/D”
  - caption con breakdown per `fonte` (con data vs senza data)
- **Trend sconti per categoria** (top 10 categorie per conteggio):
  - line chart anno vs sconto medio per categoria
  - heatmap categoria × anno (sconto medio)

### Tab 6 — `📊 Statistiche`
Obiettivo: statistiche descrittive e relazioni tra variabili (valore, sconto, offerte).

Sezioni:
- Distribuzione sconti (hist + linee media/mediana) su sconti validi (0–100, >0)
- Distribuzione valori (hist su log10(importo))
- Distribuzione offerte ricevute (se campo presente, range “ragionevole”)
- Box plot sconti per categoria (su sconti validi)
- Correlazioni:
  - scatter sconto vs valore (x log, campione fino a 5000)
  - distribuzione mensile valori (se `mese` e `anno` presenti)
- Tabella statistiche descrittive (media/mediana/std/min/max) per valore, sconto, offerte

### Tab 20 — `🔎 Ricerca`
Obiettivo: ricerca full-text + filtri numerici, con export.

Controlli:
- multiselect “Cerca nei campi” (lista predefinita di colonne testuali presenti)
- query testuale con separatore `;` per più termini
- radio combinazione termini: `OR` / `AND`
- checkbox:
  - “Usa regex”
  - “Maiuscole/minuscole”
- number input “Righe anteprima”
- filtri numerici (se colonne disponibili):
  - sconto min/max
  - partecipanti min/max

Output:
- conteggio risultati
- tabella anteprima (formattata)
- download CSV + Excel di tutte le righe risultanti

## 5) Cluster “🏆 Operatori”

### Tab 4 — `🏆 Aggiudicatari`
Obiettivo: ranking e concentrazione lato fornitori.

Sezioni:
- **Top 20 aggiudicatari per valore** (bar orizzontale, colore per num gare)
- **Concentrazione mercato**:
  - curva cumulata (% valore cumulato)
  - quota Top 5 e Top 10
  - indice HHI (interpretazione: <1500 competitivo, 1500–2500 moderato, >2500 concentrato)

### Tab 9 — `🔎 Aggiudicatario`
Obiettivo: “scheda fornitore” con ricerca e aggregazione di più varianti nome.

Controlli:
- barra ricerca testo (min 3 caratteri)
- multiselect per selezionare 1+ aggiudicatari (reset quando cambiano i filtri globali)

Output (se selezionati):
- KPI: gare vinte, valore totale, sconto medio, enti serviti, città/regione coperte (dipende colonne)
- grafici:
  - trend annuale (valore + n gare)
  - distribuzione per categoria
  - distribuzione geografica (top città) + top enti appaltanti
  - distribuzione sconti e sconto per categoria (se disponibile)
- tabella storico completo (paginata)
- export CSV

Output (se non selezionati):
- Top 50 aggiudicatari per valore nel perimetro filtrato, grafico + tabella

### Tab 12 — `⚔️ Confronto`
Obiettivo: confronto tra 2 aggiudicatari.

Controlli:
- selectbox A e B (da top 100 per valore)

Output:
- confronto KPI (gare/valore/sconto/regione/enti)
- trend annuale comparato
- categorie a confronto
- aree di influenza (top regioni per A e per B)
- sovrapposizione territoriale (regioni solo A / entrambi / solo B)

### Tab 14 — `🌐 Network`
Obiettivo: analizzare relazioni enti–fornitori (ripetizioni, concentrazioni).

Sezioni:
- **Top coppie Ente–Fornitore**
- **Fornitori “fedeli”** (scatter: N enti vs gare/ente, size totale gare)
- **Concentrazione per ente**:
  - enti con più fornitori diversi
  - enti con alta concentrazione (molte gare, pochi fornitori)
- **Network graph interattivo** (Plotly):
  - slider “Top nodi da visualizzare” (10–50)
  - slider “Min gare per connessione” (1–10)
  - select “Layout grafo”: Circolare / Forza (bipartito) / Random
  - nodi: blu = enti, verde = fornitori; size ~ valore
  - tabella top 10 connessioni più forti

## 6) Cluster “🗺️ Territoriale”

### Tab 7 — `🔍 Città`
Obiettivo: esplorare “contratti/servizi” per una città o per una stazione appaltante (territoriale “micro”).

Controlli:
- radio: cerca per `🏙️ Città` oppure `🏛️ Stazione Appaltante`
- selectbox città (se colonna disponibile) oppure multiselect stazioni appaltanti
- checkbox “Solo contratti attivi (2023-2025)”

Output (quando una città/SA è selezionata):
- KPI: totale gare, valore totale, sconto medio, e (a seconda del tipo ricerca) numero enti o numero città
- **Servizi per categoria**: pie + bar (valore per categoria)
- **Top fornitori** (bar)
- (solo ricerca per città) **Top stazioni appaltanti nella città** (bar)
- **Dettaglio servizi attivi**: tabella paginata (date/ente/categoria/fornitore/oggetto/valore/sconto)
- download CSV completo
- trend storico (valore + n gare) per anno

Output (quando non selezionato nulla):
- Top 20 città per valore (grafico + tabella)

### Tab 8 — `🗺️ Mappa CONSIP`
Obiettivo: analisi dedicata ai soli contratti CONSIP (da `ServizioLuce.xlsx`) con mappa e timeline.

Controlli:
- selectbox “Tipo accordo”
- selectbox “Anno contratto”
- selectbox “Edizione” (se colonna presente)

Output:
- KPI (contratti, valore totale, sconto medio, enti coinvolti)
- mappa (scatter_map) aggregata per città (coordinate hardcoded estese)
- tabella riepilogo per tipo
- top 10 città
- timeline contratti per anno e tipo (stacked)
- tabella dettaglio (max 100) + download CSV completo

### Tab 5 — `🏛️ CONSIP`
Obiettivo: riepiloghi CONSIP da `data.json` (aggregati).

Output:
- pie per “Tipo Accordo”
- bar confronto (num gare vs valore)
- edizioni SIE (se presenti in `data.json`)
- CONSIP per regione (top 15)

## 7) Cluster “📈 Analisi Avanzata”

### Tab 10 — `📉 Analisi Mercato`
Obiettivo: metriche avanzate su concorrenza, stagionalità, anomalie e performance procedure.

Macro-sezioni:
- **Concentrazione del mercato**
  - HHI per categoria
  - CR4 per categoria
  - N. operatori per categoria
- **Analisi competitività**
  - sconto vs partecipanti (se colonna disponibile)
  - distribuzione n partecipanti
  - sconto vs valore gara
- **Analisi stagionalità**
  - distribuzione mensile
  - heatmap mese × anno
- **Rilevamento anomalie**
  - gare con sconto anomalo
  - gare di valore elevato
  - fornitori “dominanti”
- **Efficienza procedure**
  - sconto medio per procedura
  - performance per regione
- **Riepilogo statistico** (valori, sconti, volumi, periodo)

### Tab 11 — `📅 Scadenze`
Obiettivo: stimare e mostrare **contratti attivi** e **scadenze** per città/area, includendo:
- scadenze CONSIP “reali” (da `ServizioLuce.xlsx` con `DURATA_PREVISTA`)
- scadenze da campi nel dataset (`data_scadenza`, `durata_appalto`)
- scadenze stimate per categoria (fallback)
- **enrichment automatico via CIG** con LLM `gpt-5-nano` (quando mancano le scadenze)

#### 11.1 Enrichment automatico (CIG) — gpt-5-nano
Sezione: “✨ Enrichment automatico (CIG)” (expander).

Controlli:
- checkbox “Abilita enrichment LLM”
- checkbox “Abilita fallback web (solo URL nel testo)”
- selectbox batch: 50 / 200 / 1000
- checkbox “Solo scadenze mancanti/invalid”
- checkbox “Forza refresh cache”
- number input TTL cache (giorni)
- input “CIG manuali” (lista separata da virgole/spazi)
- bottone “Esegui enrichment”

Comportamento:
- seleziona i CIG candidati nel perimetro `filtered_df` (rispettando i filtri globali), validi (regex), e senza scadenza calcolabile (se “solo mancanti”).
- carica `testo_completo`/`oggetto` *on-demand* dal CSV “master” a chunk (per evitare RAM alta).
- estrae snippet attorno a keyword (durata, mesi, anni, rinnovo, proroga, stipula, decorrenza…).
- chiama OpenAI Responses API con `model="gpt-5-nano"` e output strutturato JSON “strict”.
- calcola:
  - **Scadenza base**: fine periodo iniziale (o data fine esplicita se presente)
  - **Scadenza max**: solo se nel testo ci sono rinnovi/proroghe quantificate
- salva risultati in cache `data/output/dashboard/cig_enrichment_cache.json` (scrittura atomica).

API Key:
- la chiave è letta da `st.session_state.openai_api_key` (se inserita in app) oppure da env `OPENAI_API_KEY`.
- se manca, la UI mostra warning e blocca l’esecuzione enrichment (nessun crash).

#### 11.2 Vista territoriale: contratti attivi per città/area e scadenza
Sezione: “🧭 Contratti attivi per città/area e scadenza”.

Controlli:
- radio “Raggruppa per”: Comune / Regione / Macro-area
- checkbox “Includi stime (fallback)”
- checkbox “Solo contratti attivi”
- slider “Orizzonte scadenze (anni)” (1–15)
- checkbox “Mostra solo aree con scadenze entro orizzonte”
- drilldown:
  - selectbox area
  - checkbox “Dettaglio: solo scadenze entro N anni”
  - download CSV dettaglio area

Output (summary per area):
- Contratti (n unici)
- Valore (somma)
- Prossima scadenza (base)
- Prossima scadenza (max, se disponibile)
- Scadenze entro 12 mesi
- Scadenze entro orizzonte
- Giorni alla prossima scadenza

Output (drilldown):
- scadenza base/max, giorni, fonte scadenza, confidence e note LLM, CIG, ente, aggiudicatario, categoria, importo, aggiudicazione, link dettaglio ANAC.

Priorità calcolo scadenza (base):
1. `data_scadenza`
2. CONSIP (mappa scadenze da `ServizioLuce.xlsx`)
3. `durata_appalto`
4. LLM (cache enrichment)
5. stima per categoria (se abilitata)

#### 11.3 Sezioni CONSIP e stime (legacy)
La tab include anche:
- KPI contratti CONSIP attivi e scadenze per anno/tipo
- timeline prossimi 3 anni (CONSIP)
- tabella dettaglio contratti in scadenza (CONSIP) + download
- stime scadenze “altri contratti” (non CONSIP) per categoria con alert prossimi 12 mesi + mappa

### Tab 13 — `📆 Stagionalità`
Obiettivo: pattern temporali e “crescita” fornitori.

Sezioni:
- distribuzione mensile gare (e valore se disponibile)
- heatmap anno × mese
- analisi trimestrale (volumi e valori)
- evoluzione temporale aggiudicatari (trend per anno)
- crescita % valore e crescita % numero gare (tra 2 anni selezionati) + tabella dettaglio

## 8) Cluster “🤖 AI & Preferiti”

### Tab 15 — `🤖 AI Charts`
Obiettivo: generare grafici Plotly guidati da linguaggio naturale con workflow 2-step.

API Key:
- se `OPENAI_API_KEY` non è disponibile, mostra expander “Inserisci API Key”:
  - input password
  - bottone “Salva per questa sessione” → salva in `st.session_state.openai_api_key` + `st.rerun()`

Workflow:
1) **Analizza**: l’LLM propone un JSON di “analysis” (tipo grafico, aggregazione, colonne, filtri/pattern).
2) **Genera**: l’LLM genera codice Python/Plotly che viene eseguito su `filtered_df`.

Controlli dopo analisi:
- select tipo grafico (bar/line/scatter/pie/treemap/heatmap)
- select aggregazione (count/sum/mean)
- multiselect colonne da usare
- modifica assistita (“Scrivi cosa vuoi cambiare”)
- reset analisi

Output:
- grafico Plotly
- expander con codice generato
- bottone “⭐ Salva nei Preferiti”
- bottone “Rigenera”

### Tab 17 — `💬 Chat AI`
Obiettivo: Q&A sui dati con una UX “assistita” (se riconosce potenziali fornitori, chiede selezione prima di analizzare).

Comportamento:
- l’input chat può attivare una fase di “pending search”:
  - trova keyword (lista + parole lunghe della query)
  - propone checkbox per fornitori matching
  - poi “Analizza Selezionati” genera un mini-report strutturato
- domande generiche: l’LLM risponde usando un riassunto (totali + top 5) del perimetro filtrato.

Funzioni extra:
- “Domande rapide” (4 pulsanti)
- “Pulisci chat”

### Tab 18 — `🔮 Predizioni ML`
Obiettivo: predire “probabilità di vittoria” (indicativa) su base storica.

Sezioni:
- **Simula gara**:
  - select categoria
  - select regione (tutte o una)
  - slider range valore gara
  - bottone “Calcola Predizioni”
  - output: top 10 fornitori con punteggio/probabilità (heuristic), progress bar e grafico
- **Analisi fornitore specifico**:
  - selectbox fornitore (top 100 per n gare)
  - KPI (gare, valore, sconto medio, anni attività)
  - breakdown categorie (pie)

### Tab 19 — `🗺️ Mappa Pro`
Obiettivo: esplorazioni geografiche “avanzate” (senza geocoding esterno).

Controllo principale:
- radio “Tipo visualizzazione”:
  1) `🌡️ Heatmap Valore` (regioni con coordinate)
  2) `📍 Cluster Città` (top città per valore; no lat/lon → bar + tabella)
  3) `🎯 Drill-down Regioni` (seleziona una regione, top categorie/fornitori + trend)
  4) `⏱️ Animazione Temporale` (scatter_map animato per anno su regioni)

### Tab 16 — `⭐ Preferiti`
Obiettivo: gestire grafici salvati (AI e standard).

Funzionalità:
- KPI conteggi (totale/AI/standard)
- scelta layout: griglia (2 colonne) o lista (expander)
- render dei grafici salvati (riesegue codice AI o ricarica fig_json per standard)
- rimozione preferiti
- export JSON di tutti i preferiti

## 9) Impostazioni e variabili d’ambiente

- `.env` viene caricato (se presente) dalla directory parent del file `app.py`.
- `OPENAI_API_KEY`:
  - può essere impostata via env oppure inserita in UI (sidebar → expander `🤖 AI`) e salvata in sessione.
  - viene usata per: AI Charts, Chat AI, **Analisi AI singola gara**, Enrichment scadenze (CIG).

## 9.1) Analisi AI singola gara (selezione record)

Oltre alla Chat AI generica, la dashboard permette di selezionare **una singola gara/contratto** e lanciare un report AI “record-based” (senza inventare dati):

1) **Tab 20 — `🔎 Ricerca`** (cluster `📊 Panoramica`)
   - dopo aver fatto una ricerca (Cerca), compare la sezione **“🤖 Analisi AI (seleziona una gara)”**
   - selezioni una gara dai risultati (max 500 opzioni)
   - puoi inserire una domanda opzionale
   - clic “🤖 Analisi AI” → output in Markdown

2) **Tab 11 — `📅 Scadenze`** (cluster `📈 Analisi Avanzata`)
   - nel **drilldown per area** (dettaglio area), trovi un expander **“🤖 Analisi AI su una gara (dal dettaglio)”**
   - selezioni il CIG (o altro id presente) e lanci l’analisi

## 10) Note operative / limiti noti

- Molte colonne sono “dinamiche”: la dashboard prova più nomi (`award_amount` vs `importo_aggiudicazione`, ecc.).
- Coordinate città/regioni sono dizionari hardcoded (nessun geocoding).
- La stima scadenze per contratti non CONSIP è heuristica (durate tipiche per categoria).
- Enrichment LLM:
  - non “inventa” date; produce scadenze solo se trova durata/data in testo locale o in URL espliciti (fallback web).
  - la “scadenza max” appare solo se i rinnovi/proroghe sono **quantificati** nel testo.
- La dashboard sanitizza le tabelle per evitare crash di serializzazione Arrow (conversioni a stringa, tz-naive, ecc.).
