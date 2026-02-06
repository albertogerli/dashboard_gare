# Deploy Dashboard Gare su Azure

## File creati

| File | Descrizione |
|------|-------------|
| `Dockerfile` | Immagine Docker ottimizzata per Streamlit |
| `.dockerignore` | Esclude file non necessari dal build |
| `deploy-azure.sh` | Script automatico per deploy su Azure Container Apps |

---

## Opzione 1: Deploy Automatico (Consigliato)

```bash
# 1. Installa Azure CLI se non presente
# https://docs.microsoft.com/cli/azure/install-azure-cli

# 2. Rendi eseguibile e lancia lo script
chmod +x deploy-azure.sh
./deploy-azure.sh
```

Lo script crea tutto automaticamente: Resource Group, Container Registry, Container App.

---

## Opzione 2: Deploy Manuale

### Prerequisiti
- Azure CLI installato (`az --version`)
- Docker installato (opzionale, Azure può buildare)

### Passi

```bash
# Login
az login

# Crea risorse
az group create --name rg-dashboard-gare --location westeurope
az acr create --resource-group rg-dashboard-gare --name acrdashboardgare --sku Basic --admin-enabled true

# Build immagine (Azure builda per te, no Docker locale necessario)
az acr build --registry acrdashboardgare --image dashboard-gare:latest .

# Crea environment
az containerapp env create --name env-dashboard --resource-group rg-dashboard-gare --location westeurope

# Deploy
az containerapp create \
    --name dashboard-gare \
    --resource-group rg-dashboard-gare \
    --environment env-dashboard \
    --image acrdashboardgare.azurecr.io/dashboard-gare:latest \
    --target-port 8000 \
    --ingress external \
    --cpu 1 --memory 2Gi
```

---

## 🔄 Aggiornamento Database/Dati

### Problema
I dati sono nel container → quando aggiorni i dati, devi rifare il deploy.

### Soluzioni

#### A) Rebuild + Redeploy (Semplice)
```bash
# Aggiorna i file CSV/JSON localmente, poi:
az acr build --registry acrdashboardgare --image dashboard-gare:latest .
az containerapp update --name dashboard-gare --resource-group rg-dashboard-gare \
    --image acrdashboardgare.azurecr.io/dashboard-gare:latest
```

#### B) Azure Blob Storage (Consigliato per dati frequenti)
1. Crea Storage Account + Container
2. Carica `gare_unificate.csv.gz` su Blob
3. Modifica `app.py` per leggere da Blob:

```python
# Aggiungi a requirements.txt:
# azure-storage-blob>=12.0.0

from azure.storage.blob import BlobServiceClient
import os

def load_data_from_blob():
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client("dati-gare", "gare_unificate.csv.gz")

    # Scarica in memoria o file temporaneo
    with open("/tmp/gare_unificate.csv.gz", "wb") as f:
        f.write(blob_client.download_blob().readall())

    return pd.read_csv("/tmp/gare_unificate.csv.gz", compression="gzip")
```

4. Imposta variabile ambiente nel container:
```bash
az containerapp update --name dashboard-gare --resource-group rg-dashboard-gare \
    --set-env-vars "AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;..."
```

5. Aggiorna dati caricando nuovo file su Blob (no redeploy!)

#### C) Azure File Share (Mount come volume)
```bash
# Crea storage e file share
az storage account create --name stgdashboardgare --resource-group rg-dashboard-gare
az storage share create --name dati --account-name stgdashboardgare

# Monta nel container
az containerapp update --name dashboard-gare \
    --resource-group rg-dashboard-gare \
    --set-env-vars "DATA_PATH=/mnt/data" \
    # + configurazione volume mount
```

---

## 💰 Costi Stimati

| Risorsa | Costo/mese |
|---------|------------|
| Container Apps (1 vCPU, 2GB, scale to 0) | ~€5-15 |
| Container Registry Basic | ~€5 |
| Storage (opzionale, 10GB) | ~€1 |
| **Totale** | **~€10-20/mese** |

Con scale-to-zero, paghi solo quando qualcuno usa l'app!

---

## 🔧 Troubleshooting

```bash
# Vedere i log
az containerapp logs show --name dashboard-gare --resource-group rg-dashboard-gare --follow

# Stato app
az containerapp show --name dashboard-gare --resource-group rg-dashboard-gare

# Riavviare
az containerapp revision restart --name dashboard-gare --resource-group rg-dashboard-gare
```
