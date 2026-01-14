@echo off
start "Streamlit" cmd /k "streamlit run app.py"
timeout /t 5
start "Cloudflare Tunnel" cmd /k "cloudflared tunnel --config=cloudflared.yml run"