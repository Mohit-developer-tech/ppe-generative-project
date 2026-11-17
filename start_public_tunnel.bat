@echo off
title PPE Generative Dashboard - Public Cloudflare Tunnel
echo ================================================================
echo Starting Public Cloudflare Tunnel for PPE Dashboard (Port 8000)...
echo ================================================================
echo.
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000
pause
