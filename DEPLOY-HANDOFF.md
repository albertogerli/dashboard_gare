# Handoff Deploy - Dashboard Gare Pubbliche

## 📦 File da consegnare

```
dashboard_gare/
├── app.py                 # Applicazione principale (Streamlit)
├── requirements.txt       # Dipendenze Python
├── Dockerfile            # Immagine Docker pronta
├── .dockerignore         # Esclusioni build
├── data/                 # Dati dell'applicazione
│   └── *.csv.gz / *.json
└── .env.example          # Template variabili ambiente
```

---

## ⚙️ Specifiche Tecniche

| Parametro | Valore |
|-----------|--------|
| **Runtime** | Python 3.11 |
| **Framework** | Streamlit 1.28+ |
| **Porta** | 8000 |
| **Protocollo** | HTTP (HTTPS gestito da Azure) |
| **Health check** | `GET /_stcore/health` |
| **Risorse minime** | 1 vCPU, 2 GB RAM |
| **Risorse consigliate** | 2 vCPU, 4 GB RAM |

---

## 🔧 Variabili d'Ambiente

```env
# Opzionali - se usate API esterne
OPENAI_API_KEY=sk-xxx           # Per funzionalità AI (se presenti)
AZURE_STORAGE_CONNECTION_STRING=xxx  # Se dati su Blob Storage
```

> ⚠️ L'app funziona anche senza variabili d'ambiente se i dati sono inclusi nel container.

---

## 🐳 Build & Run (Docker)

```bash
# Build locale (test)
docker build -t dashboard-gare .
docker run -p 8000:8000 dashboard-gare

# Verifica: http://localhost:8000
```

---

## ☁️ Deploy Consigliato: Azure Container Apps

### Perché Container Apps:
- Scale-to-zero (risparmio costi)
- HTTPS automatico
- Nessun cluster da gestire
- CI/CD integrato

### Comandi Azure CLI:

```bash
# 1. Risorse
az group create --name rg-dashboard-gare --location westeurope
az acr create --resource-group rg-dashboard-gare --name <nome-univoco> --sku Basic --admin-enabled true

# 2. Build su Azure (no Docker locale)
az acr build --registry <nome-acr> --image dashboard-gare:v1 .

# 3. Deploy
az containerapp env create --name env-dashboard --resource-group rg-dashboard-gare --location westeurope

az containerapp create \
    --name dashboard-gare \
    --resource-group rg-dashboard-gare \
    --environment env-dashboard \
    --image <nome-acr>.azurecr.io/dashboard-gare:v1 \
    --target-port 8000 \
    --ingress external \
    --cpu 1 --memory 2Gi \
    --min-replicas 0 \
    --max-replicas 3
```

---

## 🔄 Aggiornamento Dati

### Opzione A: Rebuild (semplice)
Quando i dati cambiano:
1. Sostituire i file in `data/`
2. Rebuild immagine
3. Redeploy

### Opzione B: Azure Blob Storage (consigliata)
- Creare Storage Account + container `dati-gare`
- Caricare `gare_unificate.csv.gz` su Blob
- Configurare `AZURE_STORAGE_CONNECTION_STRING` nel container
- Modifiche al codice richieste (posso fornirle se necessario)

---

## 📋 Checklist Pre-Deploy

- [ ] File `app.py` presente
- [ ] File `requirements.txt` presente
- [ ] File `Dockerfile` presente
- [ ] Cartella `data/` con dati aggiornati
- [ ] Test locale Docker funzionante
- [ ] Nome ACR univoco scelto
- [ ] Resource Group Azure creato
- [ ] Budget/subscription Azure confermato

---

## 💰 Costi Stimati (Azure Container Apps)

| Scenario | Costo mensile |
|----------|---------------|
| Uso sporadico (scale-to-zero) | €5-10 |
| Uso moderato (sempre attivo) | €15-25 |
| Uso intenso (2+ repliche) | €30-50 |

+ Container Registry Basic: ~€5/mese
+ Storage (opzionale): ~€1-5/mese

---

## 📞 Contatti

**Referente progetto:** Alberto Gerli
**Email:** alberto@albertogerli.it

---

## Note Tecniche Aggiuntive

1. **L'app è stateless** - può scalare orizzontalmente
2. **I preferiti utente** sono salvati in `data/output/dashboard/favorites.json` (se serve persistenza, usare storage esterno)
3. **Cache enrichment CIG** in `data/output/dashboard/cig_enrichment_cache.json`
4. **Nessun database SQL** - tutti i dati sono in CSV/JSON
