#!/bin/sh
# Railway sets $PORT; override Streamlit's env var to use it
export STREAMLIT_SERVER_PORT="${PORT:-8501}"
exec streamlit run app.py --server.address=0.0.0.0
