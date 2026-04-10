#!/bin/sh
export STREAMLIT_SERVER_PORT=8501
exec streamlit run app.py --server.address=0.0.0.0
