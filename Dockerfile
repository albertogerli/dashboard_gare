# Dockerfile per Dashboard Gare Pubbliche - Streamlit su Azure
# Ottimizzato per Azure Container Apps / Web App for Containers

FROM python:3.11-slim

# Metadata
LABEL maintainer="alberto@albertogerli.it"
LABEL description="Dashboard Gare Pubbliche - Streamlit"

# Variabili ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Directory di lavoro
WORKDIR /app

# Installa dipendenze di sistema (minime per slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY app.py .
COPY data/ ./data/

# Crea directory per dati runtime (se necessario)
RUN mkdir -p /app/data/output/dashboard

# Crea utente non-root per sicurezza
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Porta esposta (Railway assegna $PORT)
EXPOSE ${PORT}

# Comando di avvio — usa $PORT da Railway
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
