#!/bin/bash
# Deploy Dashboard Gare su Azure Container Apps
# Esegui: chmod +x deploy-azure.sh && ./deploy-azure.sh

set -e

# ============================================
# CONFIGURAZIONE - MODIFICA QUESTI VALORI
# ============================================
RESOURCE_GROUP="rg-dashboard-gare"
LOCATION="westeurope"
ACR_NAME="acrdashboardgare"  # deve essere unico globalmente, solo lettere minuscole
APP_NAME="dashboard-gare"
CONTAINER_ENV="env-dashboard-gare"

# ============================================
# 1. Login Azure (se non già fatto)
# ============================================
echo "🔐 Verifica login Azure..."
az account show > /dev/null 2>&1 || az login

# ============================================
# 2. Crea Resource Group
# ============================================
echo "📦 Creazione Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# ============================================
# 3. Crea Azure Container Registry
# ============================================
echo "🏗️ Creazione Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# ============================================
# 4. Build e Push immagine Docker
# ============================================
echo "🐳 Build e push immagine Docker..."
az acr build \
    --registry $ACR_NAME \
    --image $APP_NAME:latest \
    .

# ============================================
# 5. Crea Container Apps Environment
# ============================================
echo "🌐 Creazione Container Apps Environment..."
az containerapp env create \
    --name $CONTAINER_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# ============================================
# 6. Deploy Container App
# ============================================
echo "🚀 Deploy applicazione..."
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_ENV \
    --image "$ACR_NAME.azurecr.io/$APP_NAME:latest" \
    --registry-server "$ACR_NAME.azurecr.io" \
    --registry-username $ACR_NAME \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8000 \
    --ingress external \
    --cpu 1 \
    --memory 2Gi \
    --min-replicas 0 \
    --max-replicas 3

# ============================================
# 7. Mostra URL dell'app
# ============================================
echo ""
echo "✅ Deploy completato!"
echo ""
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
echo "🌍 La tua app è disponibile su: https://$APP_URL"
echo ""
echo "📊 Per vedere i log:"
echo "   az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo "🔄 Per aggiornare dopo modifiche:"
echo "   az acr build --registry $ACR_NAME --image $APP_NAME:latest ."
echo "   az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --image $ACR_NAME.azurecr.io/$APP_NAME:latest"
